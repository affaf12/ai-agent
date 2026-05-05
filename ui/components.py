import streamlit as st
import hashlib
import html
import base64
from datetime import datetime
from typing import Optional, List
 
 
class UIComponents:
    """Reusable UI components"""
 
    @staticmethod
    def apply_custom_theme():
        """Apply enterprise styling. Must be called on every script rerun."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            color: #666;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 1rem;
            border-radius: 0.5rem;
            background: #f8f9fa;
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 1rem 1rem 0.2rem 1rem;
            padding: 1rem;
            margin: 0.5rem 0 0.5rem auto;
            max-width: 80%;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }
        .assistant-message {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 1rem 1rem 1rem 0.2rem;
            padding: 1rem;
            margin: 0.5rem auto 0.5rem 0;
            max-width: 80%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .message-time {
            font-size: 0.7rem;
            color: #999;
            margin-top: 0.3rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-primary {
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }
        .badge-success {
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
        }
        .badge-warning {
            background: rgba(245, 158, 11, 0.1);
            color: #f59e0b;
        }
        .badge-error {
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
        }
        .sidebar-header {
            text-align: center;
            padding: 1rem 0;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        .stat-card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.875rem;
        }
        .tool-button {
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            padding: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tool-button:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        div[data-testid="stExpander"] div[role="button"] {
            font-weight: 600;
            color: #333;
        }
        .agent-box {
            border-left: 3px solid #667eea;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 0 0.5rem 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
 
    @staticmethod
    def render_message(
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        images: Optional[List] = None,
        idx: int = 0,
    ):
        """Render a chat message with ChatGPT-style copy/edit buttons.
 
        Parameters
        ----------
        role      : "user" or "assistant"
        content   : raw text content of the message
        timestamp : datetime (or ISO string) of when the message was created
        images    : optional list of image objects for user messages
        idx       : position index in the message list — used for key namespacing
        """
 
        # ── ROLE VALIDATION ───────────────────────────────────���────────────────
        if role not in ("user", "assistant"):
            st.warning(f"render_message: unexpected role '{role}' — expected 'user' or 'assistant'.")
            return
 
        # ── TIMESTAMP NORMALISATION ─────────────────────────────────────────────
        if timestamp is None:
            timestamp = datetime.now()
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    st.warning(f"⚠️ Could not parse timestamp: {timestamp} — using current time")
                    timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
 
        time_str = timestamp.strftime("%H:%M")
 
        # ── STREAMING GUARD ─────────────────────────────────────────────────────
        is_streaming = content.endswith("▌")
        display_content = content.rstrip("▌") if is_streaming else content
 
        # ── XSS SAFETY ──────────────────────────────────────────────────────────
        safe_content = html.escape(display_content)
 
        # ── UNIQUE BUTTON KEYS ───────────────────────────────────────────────────
        content_hash = hashlib.md5(
            display_content.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        key_suffix = f"{idx}_{content_hash}"
 
        # ════════════════════════════════════════════════════════════════════════
        # USER MESSAGE
        # ════════════════════════════════════════════════════════════════════════
        if role == "user":
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; margin:1rem 0;">
                    <div>
                        <div style="background:#f4f4f4; border-radius:1.2rem;
                                    padding:0.8rem 1rem; max-width:80%; color:#000;">
                            {safe_content}
                        </div>
                        <div class="message-time" style="text-align:right; padding-right:0.4rem;">
                            {time_str}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            # ✅ FIX: Check images exists AND is not empty
            if images and len(images) > 0:
                _, img_col = st.columns([0.2, 0.8])
                with img_col:
                    st.image(images, width=300)
 
            if not is_streaming:
                _, edit_col, _ = st.columns([0.82, 0.06, 0.12])
                with edit_col:
                    if st.button("✏️", key=f"edit_{key_suffix}", help="Edit message"):
                        st.session_state["edit_prompt"] = display_content
                        st.session_state["edit_idx"] = idx
                        st.rerun()
 
        # ════════════════════════════════════════════════════════════════════════
        # ASSISTANT MESSAGE
        # ════════════════════════════════════════════════════════════════════════
        else:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start;
                            margin:1rem 0; align-items:flex-start;">
                    <div style="margin-right:0.6rem; font-size:1.2rem;
                                margin-top:0.2rem; flex-shrink:0;">🦙</div>
                    <div style="max-width:80%;">
                        <div style="background:#ffffff; border:1px solid #e5e7eb;
                                    border-radius:1.2rem; padding:0.8rem 1rem;
                                    color:#000;
                                    box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                            {safe_content}
                        </div>
                        <div style="font-size:0.7rem; color:#999;
                                    margin-top:0.2rem; padding-left:0.4rem;">
                            {time_str}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            if not is_streaming:
                copy_col, regen_col = st.columns([0.08, 0.92])

                with copy_col:
                    b64 = base64.b64encode(display_content.encode("utf-8")).decode("ascii")

                    # onclick: try navigator.clipboard first, fall back to
                    # execCommand for browsers/contexts that block the Async API.
                    onclick = (
                        "var b=this.getAttribute('data-b64');"
                        "var a=Uint8Array.from(atob(b),function(c){return c.charCodeAt(0)});"
                        "var t=new TextDecoder().decode(a);"
                        "var btn=this;"
                        "function fallback(txt){"
                        "  var ta=document.createElement('textarea');"
                        "  ta.value=txt;"
                        "  ta.style.cssText='position:fixed;opacity:0;top:0;left:0;';"
                        "  document.body.appendChild(ta);"
                        "  ta.focus();ta.select();"
                        "  document.execCommand('copy');"
                        "  document.body.removeChild(ta);"
                        "  btn.textContent=String.fromCharCode(10003);"
                        "  setTimeout(function(){btn.textContent=String.fromCharCode(128203);},1500)"
                        "}"
                        "if(navigator.clipboard&&navigator.clipboard.writeText){"
                        "  navigator.clipboard.writeText(t)"
                        "  .then(function(){"
                        "    btn.textContent=String.fromCharCode(10003);"
                        "    setTimeout(function(){btn.textContent=String.fromCharCode(128203);},1500)"
                        "  })"
                        "  .catch(function(){fallback(t)})"
                        "}else{fallback(t)}"
                    )

                    # ✅ st.html replaces the deprecated components.html
                    copy_html = (
                        "<html><body style='margin:0;padding:0;background:transparent;'>"
                        f"<button data-b64='{b64}' onclick=\"{onclick}\" "
                        "style='background:none;border:none;cursor:pointer;"
                        "font-size:15px;padding:2px 4px;opacity:0.6;"
                        "font-family:sans-serif;' "
                        "title='Copy to clipboard'>"
                        f"{chr(128203)}"
                        "</button>"
                        "</body></html>"
                    )
                    st.html(copy_html)

                with regen_col:
                    if st.button("🔄", key=f"regen_{key_suffix}", help="Regenerate response"):
                        st.session_state["regenerate_idx"] = idx
                        st.rerun()
 
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def metric_card(label: str, value: str, delta: Optional[str] = None):
        """Render a metric card with color-coded delta.

        All inputs are HTML-escaped for safety.
        Delta color reflects sign: red for negative, green for positive, grey for neutral.
        """
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))

        delta_html = ""
        if delta is not None and str(delta) != "":
            delta_str = str(delta).strip()
            stripped = delta_str.lstrip("+−-").rstrip("%").replace(",", "")
            
            # ✅ FIX: Handle zero/empty case
            try:
                numeric = float(stripped) if stripped else 0.0
            except ValueError:
                numeric = 0.0
            
            # Determine color based on numeric value
            if numeric < 0:
                colour = "#ef4444"  # red
            elif numeric > 0:
                colour = "#10b981"  # green
            else:
                colour = "#6b7280"  # neutral grey
                
            delta_html = (
                f'<div style="color:{colour}; font-size:0.875rem;">'
                f"{html.escape(delta_str)}</div>"
            )
            
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{safe_value}</div>
                <div class="stat-label">{safe_label}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )