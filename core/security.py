"""
core/security.py

Thin compatibility / utility layer.

BUG FIX (history):
    This module previously defined its OWN SecurityManager class — different
    from the one in core/auth.py. The two classes had incompatible
    signatures (this one returned raw bytes for salt/key, auth.py returns
    hex strings) and different role names ("power_user" here vs "pro" in
    auth.py). Whichever module you imported from, you got a different
    behaviour. That class has been removed; SecurityManager is now
    re-exported from core.auth so there is exactly one source of truth.

CLEANUP:
    The old file imported ~30 modules (numpy, pandas, plotly, PIL,
    asyncio, threading, ThreadPoolExecutor, sqlite3, …) and used almost
    none of them. All dead imports have been deleted.
"""

from __future__ import annotations

import re
from typing import Final

# Re-export the canonical SecurityManager so existing
# `from core.security import SecurityManager` imports keep working.
from core.auth import SecurityManager  # noqa: F401  (re-export)

__all__ = ["SecurityManager", "sanitize_input", "ROLES"]


# ---------------------------------------------------------------------------
# Role catalogue
# ---------------------------------------------------------------------------
# Exposed as a module-level constant so the rest of the app has one
# documented place to look up which features each role can use. The actual
# permission check still lives on SecurityManager.check_permission() in
# auth.py, which uses its own internal table — keep these two in sync if
# you change roles.
ROLES: Final[dict[str, list[str]]] = {
    "admin":   ["*"],
    "pro":     ["chat", "agents", "rag", "analytics", "export", "sql_builder"],
    "user":    ["chat", "rag", "export", "project"],
    "guest":   ["chat"],
}


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------
_SCRIPT_RE = re.compile(r"<script.*?>.*?</script>", re.DOTALL | re.IGNORECASE)
_JS_URL_RE = re.compile(r"javascript:", re.IGNORECASE)

# UPGRADE: hard cap on user input. Previously hard-coded inside the
# function; pulled out here so it is easy to change in one place.
MAX_INPUT_CHARS: Final[int] = 10_000


def sanitize_input(text: str) -> str:
    """Best-effort sanitiser for free-text user input.

    - Strips obvious <script>…</script> blocks.
    - Strips ``javascript:`` URI schemes.
    - Truncates to MAX_INPUT_CHARS to bound memory and prompt size.

    NOTE: this is NOT a security boundary. If you render user text as HTML
    you must still escape it at render time (see formatter._render_html,
    which uses html.escape()). This function is for cleaning prompts /
    log strings, not for preventing XSS in arbitrary HTML output.
    """
    if not text:
        return ""
    text = _SCRIPT_RE.sub("", text)
    text = _JS_URL_RE.sub("", text)
    return text[:MAX_INPUT_CHARS]
