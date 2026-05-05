"""
core/auth.py
Simple authentication with SQLite
"""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
import secrets
import hmac
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ollama_pro_enterprise.db"

# Global DB connection
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
conn.row_factory = sqlite3.Row

# Initialize schema if needed
def _init_db():
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        api_key TEXT,
        settings TEXT DEFAULT '{}',
        is_active INTEGER DEFAULT 1,
        last_login TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    -- UPGRADE: persistent login tokens so a browser reload does not log
    -- the user out. SessionManager.persist_login() inserts a row here and
    -- writes the token to the URL (?t=...). SessionManager.restore_from_token()
    -- reads the token back on every page load and rehydrates st.session_state.
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token       TEXT PRIMARY KEY,
        user_id     INTEGER NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at  TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id);
    """)
    conn.commit()
    # Create default admin if none exists.
    # UPGRADE: allow the bootstrap admin password to come from the
    # ADMIN_PASSWORD env var so production deploys aren't stuck on "admin123".
    cur = conn.execute("SELECT COUNT(*) as c FROM users")
    if cur.fetchone()["c"] == 0:
        admin_pwd = os.environ.get("ADMIN_PASSWORD", "admin123")
        salt, pwd = SecurityManager.hash_password(admin_pwd)
        api = SecurityManager.generate_api_key()
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, api_key) VALUES (?,?,?,?,?)",
            ("admin", pwd, salt, "admin", api)
        )
        conn.commit()

class SecurityManager:
    @staticmethod
    def hash_password(password: str) -> tuple[str, str]:
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
        return salt, pwd_hash

    @staticmethod
    def verify_password(password: str, salt: str, pwd_hash: str) -> bool:
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
        return hmac.compare_digest(test, pwd_hash)

    @staticmethod
    def generate_api_key() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def check_permission(role: str, feature: str) -> bool:
        # Simple RBAC
        perms = {
            "admin": {"*"},
            "pro": {"agents", "rag", "analytics", "sql_builder"},
            "user": {"chat", "export", "project"},
            "guest": {"chat"},
        }
        allowed = perms.get(role, set())
        return "*" in allowed or feature in allowed

class AuthManager:
    """Session-aware authentication"""

    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
        cur = conn.execute(
            "SELECT id, username, password_hash, salt, role, api_key, settings FROM users WHERE username = ? AND is_active = 1",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if SecurityManager.verify_password(password, row["salt"], row["password_hash"]):
            conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
            conn.commit()
            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "api_key": row["api_key"],
                "settings": json.loads(row["settings"] or "{}"),
            }
        return None

    @staticmethod
    def create_user(username: str, password: str, role: str = "user") -> bool:
        # UPGRADE: validate inputs. Previously empty/1-char usernames and
        # passwords were accepted, leading to weak accounts that the login
        # form happily authenticated.
        username = (username or "").strip()
        if len(username) < 3:
            return False
        if len(password or "") < 6:
            return False
        try:
            salt, pwd_hash = SecurityManager.hash_password(password)
            api_key = SecurityManager.generate_api_key()
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, api_key) VALUES (?,?,?,?,?)",
                (username, pwd_hash, salt, role, api_key),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ── Persistent login tokens ────────────────────────────────────────────
    # These three methods are the heart of the "stay logged in across page
    # reloads" fix. Wire them up via SessionManager.persist_login() and
    # SessionManager.restore_from_token().

    TOKEN_TTL_SECONDS: int = 7 * 24 * 3600   # 7 days

    @staticmethod
    def create_login_token(user_id: int, ttl_seconds: Optional[int] = None) -> str:
        """Issue a fresh login token for the given user and return it."""
        from datetime import timedelta  # local import keeps top imports clean
        token = secrets.token_urlsafe(32)
        ttl = ttl_seconds if ttl_seconds is not None else AuthManager.TOKEN_TTL_SECONDS
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat(timespec="seconds")),
        )
        conn.commit()
        return token

    @staticmethod
    def user_from_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a token to a user dict, or return None if the token is
        unknown / expired / belongs to a deactivated account. Expired
        tokens are cleaned up opportunistically on access.
        """
        if not token:
            return None
        cur = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.api_key, u.settings, t.expires_at
            FROM auth_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ? AND u.is_active = 1
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        # Expiry check (string compare on ISO-8601 is correct for this format)
        try:
            expires = datetime.fromisoformat(row["expires_at"])
        except (TypeError, ValueError):
            expires = datetime.utcnow()  # malformed → treat as expired
        if expires < datetime.utcnow():
            AuthManager.revoke_token(token)
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "api_key": row["api_key"],
            "settings": json.loads(row["settings"] or "{}"),
        }

    @staticmethod
    def revoke_token(token: str) -> None:
        """Delete a login token (called on logout or expiry)."""
        if not token:
            return
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
        conn.commit()


# Initialize on import
_init_db()

__all__ = ["AuthManager", "SecurityManager", "conn"]
