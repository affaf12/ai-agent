"""
core/session.py
Session management with SQLite persistence
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
import streamlit as st

try:
    from core.config import CONFIG
except ImportError:
    from config import CONFIG

# Same fallback pattern as CONFIG, in case session.py is imported before
# the `core` package is recognised on sys.path.
try:
    from core.auth import AuthManager
except ImportError:
    from auth import AuthManager  # type: ignore

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ollama_pro_enterprise.db"

def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT DEFAULT 'New Chat',
        messages TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

_init_db()

_QUERY_PARAM_TOKEN_KEY = "t"   # ?t=<token> in the URL


def _read_token_from_url() -> str:
    """Read the login token from the URL query string.

    Streamlit's API changed in 1.30 — st.query_params replaced
    st.experimental_get_query_params(). Support both for safety.
    """
    try:
        return str(st.query_params.get(_QUERY_PARAM_TOKEN_KEY, "") or "")
    except AttributeError:
        legacy = st.experimental_get_query_params().get(_QUERY_PARAM_TOKEN_KEY, [""])
        return legacy[0] if legacy else ""


def _write_token_to_url(token: str) -> None:
    try:
        st.query_params[_QUERY_PARAM_TOKEN_KEY] = token
    except AttributeError:
        st.experimental_set_query_params(**{_QUERY_PARAM_TOKEN_KEY: token})


def _clear_token_from_url() -> None:
    try:
        if _QUERY_PARAM_TOKEN_KEY in st.query_params:
            del st.query_params[_QUERY_PARAM_TOKEN_KEY]
    except AttributeError:
        st.experimental_set_query_params()  # wipes all params


class SessionManager:
    @staticmethod
    def init_session():
        defaults = {
            "user": None,
            "auth_token": None,           # NEW: persistent login token
            "messages": [],
            "model": "llama3.1:8b",
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 4096,
            "rag_enabled": False,
            "vision_enabled": False,
            "tts_enabled": False,
            "show_token_count": False,
            "current_session": None,
            "last_activity": datetime.now(),
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # ── Inactivity timeout ─────────────────────────────────────────────
        last = st.session_state.get("last_activity")
        if last and isinstance(last, datetime):
            if (datetime.now() - last).total_seconds() > CONFIG.SESSION_TIMEOUT:
                # Full wipe + revoke any persistent token (so an idle user
                # really IS logged out, not just locally cleared).
                stale_token = st.session_state.get("auth_token")
                if stale_token:
                    AuthManager.revoke_token(stale_token)
                _clear_token_from_url()
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                SessionManager.init_session()
        st.session_state["last_activity"] = datetime.now()

        # ── BUG FIX: restore login across browser reloads ──────────────────
        # If we have no in-memory user, see if the URL carries a valid token
        # from a previous session and rehydrate from the database.
        if st.session_state.get("user") is None:
            SessionManager.restore_from_token()

    # ── Persistent login helpers ───────────────────────────────────────────
    @staticmethod
    def restore_from_token() -> bool:
        """Try to log the current browser session back in via the URL token.

        Returns True if a user was restored, False otherwise.
        """
        token = _read_token_from_url()
        if not token:
            return False
        user = AuthManager.user_from_token(token)
        if not user:
            # Token is unknown or expired — clean it out of the URL so we
            # don't keep retrying every reload.
            _clear_token_from_url()
            return False
        st.session_state["user"] = user
        st.session_state["auth_token"] = token
        return True

    @staticmethod
    def persist_login(user: dict) -> None:
        """Call right after AuthManager.authenticate() succeeds.

        Issues a new persistent token, stores it in session_state, and writes
        it to the URL so the next page reload survives.
        """
        token = AuthManager.create_login_token(user["id"])
        st.session_state["user"] = user
        st.session_state["auth_token"] = token
        _write_token_to_url(token)

    @staticmethod
    def logout() -> None:
        """Sign the current user out everywhere — DB, memory, and URL."""
        token = st.session_state.get("auth_token")
        if token:
            AuthManager.revoke_token(token)
        _clear_token_from_url()
        SessionManager.clear()

    @staticmethod
    def create_session(user_id: int, title: str = "New Chat") -> int:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO sessions (user_id, title, messages) VALUES (?,?,?)",
            (user_id, title, "[]")
        )
        conn.commit()
        sid = cur.lastrowid
        conn.close()
        return sid

    @staticmethod
    def list_sessions(user_id: int, limit: int = 20):
        conn = _get_conn()
        cur = conn.execute(
            "SELECT id, title, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    @staticmethod
    def load_session(session_id: int, user_id: int | None = None):
        """Load a chat session.

        BUG FIX: previously any user could load ANY session_id. Now callers
        should pass user_id; if provided, the row must belong to that user
        or None is returned. user_id is optional for backward compatibility.
        """
        conn = _get_conn()
        if user_id is None:
            cur = conn.execute(
                "SELECT messages FROM sessions WHERE id = ?",
                (session_id,),
            )
        else:
            cur = conn.execute(
                "SELECT messages FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
        row = cur.fetchone()
        conn.close()
        if row:
            return {"messages": json.loads(row["messages"])}
        return None

    @staticmethod
    def save_session(session_id: int, messages: list,
                     metadata: dict | None = None, user_id: int | None = None):
        """Save a session.

        BUG FIX: same ownership concern as load_session. If user_id is
        supplied the UPDATE is scoped, so one user cannot overwrite
        another user's chat history by guessing IDs.

        metadata is accepted for future extension (kept for API stability).
        """
        conn = _get_conn()
        if user_id is None:
            conn.execute(
                "UPDATE sessions SET messages = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (json.dumps(messages), session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET messages = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND user_id = ?",
                (json.dumps(messages), session_id, user_id),
            )
        conn.commit()
        conn.close()

    @staticmethod
    def clear():
        """Local-only clear. For full sign-out (revoke token + clear URL)
        prefer SessionManager.logout()."""
        for k in list(st.session_state.keys()):
            del st.session_state[k]
