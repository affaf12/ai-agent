import streamlit as st
import asyncio
import hashlib
import hmac
import json
import os
import re
import io
import queue as _queue_module
import zipfile
import uuid
import tempfile
import time
import base64
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Callable
from contextlib import contextmanager
from functools import lru_cache, wraps
from importlib.metadata import version as _get_version

import numpy as np
import pandas as pd
try:
    import plotly.express as px
except ImportError:
    px = None
from PIL import Image, ImageOps
import requests

# Check Streamlit version for fragment support
try:
    _st_version = tuple(int(x) for x in _get_version("streamlit").split(".")[:2])
    _FRAGMENT_SUPPORTED = _st_version >= (1, 35)
except Exception:
    _FRAGMENT_SUPPORTED = hasattr(st, "fragment")

# Ensure project root is on sys.path for `config` import
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import core types for type hints
try:
    from core.ollama_client import OllamaClient as _BaseOllamaClient
except ImportError:
    from ollama_client import OllamaClient as _BaseOllamaClient
from core.session import SessionManager
from core.auth import AuthManager, SecurityManager
from core.analytics import Analytics
# NEW: Use rag_system_pro instead of old features.rag
try:
    from core.rag_system_pro import RAGSystem
    from core.rag_system_pro.agents import (
        ExcelAgent, CodeAgent, DocWriterAgent,
        WebResearchAgent, SQLAgent, LLMClient,
        available_agents
    )
    PRO_AGENTS_AVAILABLE = True
except ImportError:
    # Fallback to old system
    try:
        from features.rag import RAGSystem
        from features.agents import Agent, AgentOrchestrator
    except ImportError:
        RAGSystem = None
        AgentOrchestrator = None
    PRO_AGENTS_AVAILABLE = False
from features.multimodal import MultimodalProcessor
from features.export import ExportManager
try:
    from core.rag_system_pro.agents.analytics_agent import AnalyticsAgent
    from dataclasses import asdict
    ANALYTICS_PRO_AVAILABLE = True
except ImportError:
    ANALYTICS_PRO_AVAILABLE = False
from ui.components import UIComponents
from core.config import CONFIG, get_optimal_model, get_model_info, LLM_MODEL

# --- PATCH: Ollama Cloud Support ---
try:
    from ollama import Client as _OllamaOfficial
    _OFFICIAL_AVAILABLE = True
except ImportError:
    _OFFICIAL_AVAILABLE = False

class OllamaClientCloud:
    """Drop-in replacement that adds Bearer auth for ollama.com"""
    def __init__(self, host: str):
        api_key = st.secrets.get("OLLAMA_API_KEY", "") if hasattr(st, 'secrets') else ""
        headers = {}
        if api_key and "ollama.com" in host:
            headers = {"Authorization": f"Bearer {api_key}"}

        if _OFFICIAL_AVAILABLE:
            self.client = _OllamaOfficial(host=host, headers=headers)
            self._use_official = True
        else:
            self.client = _BaseOllamaClient(host)
            self._use_official = False
            if headers and hasattr(self.client, 'session'):
                self.client.session.headers.update(headers)

    def health(self):
        try:
            if self._use_official:
                models = self.client.list()
                models_info = models.get('models', []) if isinstance(models, dict) else models
                return True, None, models_info
            else:
                return self.client.health()
        except Exception as e:
            return False, str(e), []

    def chat_stream(self, model, messages, options):
        if self._use_official:
            stream = self.client.chat(model=model, messages=messages, options=options or {}, stream=True)
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    yield content
        else:
            yield from self.client.chat_stream(model, messages, options)

    def chat_once(self, model, messages, options):
        if self._use_official:
            resp = self.client.chat(model=model, messages=messages, options=options or {}, stream=False)
            return resp['message']['content'], None
        else:
            return self.client.chat_once(model, messages, options)

# Override the original client
OllamaClient = OllamaClientCloud

# --- PATCH: remove timestamp HTML that was showing as raw code ---
def _clean_render_message(role: str, content: str, timestamp=None, images=None, idx=None, **kwargs):
    """Override UIComponents.render_message to hide the broken timestamp div."""
    try:
        # Use native chat_message for clean bubbles
        with st.chat_message(role if role in ("user", "assistant") else "assistant"):
            if images and role == "user":
                # show images if provided
                cols = st.columns(min(len(images), 4))
                for i, img in enumerate(images[:4]):
                    with cols[i % 4]:
                        try:
                            st.image(img, use_container_width=True)
                        except Exception:
                            pass
            st.markdown(content)
            # intentionally NOT rendering timestamp to avoid the <div style="font-size:0.7rem..."> bug
    except Exception:
        # fallback
        st.write(f"**{role}:** {content}")

# Apply patch
UIComponents.render_message = staticmethod(_clean_render_message)

# Smart model auto-selection enabled

# --- INTENT DETECTION (prevents RAG on greetings) ---
GREETINGS = {"hello", "hi", "hey", "how are you", "good morning", "good evening", "sup", "yo", "what's up", "greetings"}
CODING_SIGNALS = {"error", "fix", "bug", "install", "python", "javascript", "code", "traceback", "exception", "import", "module", "syntax", "typeerror", "keyerror", "how to"}

def is_greeting(q: str) -> bool:
    if not q:
        return False
    ql = q.lower().strip()
    return any(g in ql for g in GREETINGS) and len(ql.split()) <= 6

def should_use_rag(q: str) -> bool:
    if is_greeting(q):
        return False
    if len(q.split()) <= 2:
        return False
    ql = q.lower()
    return any(s in ql for s in CODING_SIGNALS) or "how to" in ql or "?" in ql

# --- GLOBAL RAG WITH KNOWLEDGE BASE ---
try:
    from executor import DebugLoop
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False
    DebugLoop = None

GLOBAL_RAG = None
def _init_global_rag():
    global GLOBAL_RAG
    if GLOBAL_RAG is not None:
        return GLOBAL_RAG

    from pathlib import Path

    # Use new RAG system if available
    if PRO_AGENTS_AVAILABLE:
        rag = RAGSystem(
            llm_model=get_optimal_model(), # Auto-selected based on RAM
            embed_model="nomic-embed-text",
            index_path="data/faiss.index"
        )
        print(f"[RAG INIT] Using rag_system_pro with index at data/faiss.index")
    else:
        # Fallback to old system
        if RAGSystem is None:
            print("[RAG INIT] RAGSystem not available")
            return None
        rag = RAGSystem()
        # Use relative path instead of hard-coded
        kb_path = Path("data/knowledge")
        print(f"[RAG INIT] Loading knowledge from {kb_path}")
        try:
            if kb_path.exists():
                files = list(kb_path.rglob("*.md")) + list(kb_path.rglob("*.txt"))
                loaded = 0
                for f in files:
                    if any(s in f.name.lower() for s in ["prompt", "rules", "system", "instruction", "guideline"]):
                        continue
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")
                        rag.add_document("global", f.name, text)
                        loaded += 1
                    except Exception:
                        pass
                print(f"[RAG INIT] DONE - {loaded} files")
        except Exception as e:
            print(f"[RAG INIT] ERROR: {e}")

    GLOBAL_RAG = rag
    return rag

# Initialize on import
_init_global_rag()

# Multi-task import (optional)
try:
    from features.multi_task.orchestrator import run_multi_task_project
    MULTI_TASK_AVAILABLE = True
except ImportError:
    MULTI_TASK_AVAILABLE = False

# --- PATCH: SecurityManager missing method ---
if not hasattr(SecurityManager, 'sanitize_input'):
    def _sanitize_input(text: str) -> str:
        import re
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        return text[:10000]
    SecurityManager.sanitize_input = staticmethod(_sanitize_input)

# =============================================================================
# BACKGROUND GENERATION INFRASTRUCTURE
# =============================================================================

@dataclass
class GenerationJob:
    """A single in-flight LLM generation, owned by one (user, session) pair."""
    stop_event: threading.Event
    chunk_queue: "_queue_module.Queue[Optional[str]]"
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    full_response: str = ""
    done: bool = False
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    thread: Optional[threading.Thread] = None

_GENERATION_REGISTRY: Dict[str, GenerationJob] = {}
_REGISTRY_LOCK = threading.Lock()

def _job_key(user_id: Any, session_id: Any) -> str:
    return f"{user_id}::{session_id or 'unsaved'}"

def _get_job(user_id: Any, session_id: Any) -> Optional[GenerationJob]:
    with _REGISTRY_LOCK:
        return _GENERATION_REGISTRY.get(_job_key(user_id, session_id))

def _set_job(user_id: Any, session_id: Any, job: GenerationJob) -> None:
    with _REGISTRY_LOCK:
        _GENERATION_REGISTRY[_job_key(user_id, session_id)] = job

def _pop_job(user_id: Any, session_id: Any) -> Optional[GenerationJob]:
    with _REGISTRY_LOCK:
        return _GENERATION_REGISTRY.pop(_job_key(user_id, session_id), None)

def _start_generation(
    client: "OllamaClient",
    model: str,
    ollama_msgs: List[Dict[str, Any]],
    options: Dict[str, Any],
    user_id: Any,
    session_id: Any,
) -> GenerationJob:
    """Spawn a daemon thread that streams chunks into a queue. Returns the job."""
    stop_event = threading.Event()
    chunk_queue: "_queue_module.Queue[Optional[str]]" = _queue_module.Queue()
    job = GenerationJob(stop_event=stop_event, chunk_queue=chunk_queue)

    def _worker() -> None:
        try:
            for chunk in client.chat_stream(model, ollama_msgs, options):
                if stop_event.is_set():
                    break
                with job.state_lock:
                    job.full_response += chunk
                chunk_queue.put(chunk)
        except Exception as e: # noqa: BLE001
            with job.state_lock:
                job.error = str(e)
        finally:
            with job.state_lock:
                job.done = True
            chunk_queue.put(None)

    thread = threading.Thread(target=_worker, name=f"gen-{user_id}-{session_id}", daemon=True)
    job.thread = thread
    _set_job(user_id, session_id, job)
    thread.start()
    return job

# =============================================================================
# ENHANCED FILE MANAGEMENT SYSTEM
# =============================================================================

@dataclass
class FileQueueItem:
    """Represents a file in the upload queue"""
    file_id: str
    name: str
    size: int
    file_type: str
    bytes_data: bytes
    uploaded_at: datetime
    processed: bool = False

    def get_size_mb(self) -> float:
        """Get file size in MB"""
        return self.size / (1024 * 1024)

    def get_size_display(self) -> str:
        """Get human-readable file size"""
        mb = self.get_size_mb()
        if mb > 1:
            return f"{mb:.2f} MB"
        kb = self.size / 1024
        if kb > 1:
            return f"{kb:.2f} KB"
        return f"{self.size} bytes"

def _init_file_queue():
    """Initialize file queue in session state"""
    if 'file_queue' not in st.session_state:
        st.session_state.file_queue = {} # {file_id: FileQueueItem}
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = [] # List of processed file names

def _add_file_to_queue(file_obj) -> str:
    """Add a file to the processing queue and return file_id"""
    _init_file_queue()

    file_id = str(uuid.uuid4())[:8]
    file_item = FileQueueItem(
        file_id=file_id,
        name=file_obj.name,
        size=len(file_obj.getvalue()),
        file_type=file_obj.type or "unknown",
        bytes_data=file_obj.getvalue(),
        uploaded_at=datetime.now()
    )

    st.session_state.file_queue[file_id] = file_item
    return file_id

def _get_queued_files() -> List[FileQueueItem]:
    """Get all files in queue (processed and unprocessed)"""
    _init_file_queue()
    return list(st.session_state.file_queue.values())

def _get_unprocessed_files() -> List[FileQueueItem]:
    """Get only unprocessed files"""
    return [f for f in _get_queued_files() if not f.processed]

def _get_file_from_queue(file_id: str) -> Optional[FileQueueItem]:
    """Get a specific file from queue"""
    _init_file_queue()
    return st.session_state.file_queue.get(file_id)

def _mark_file_processed(file_id: str) -> None:
    """Mark a file as processed"""
    _init_file_queue()
    if file_id in st.session_state.file_queue:
        st.session_state.file_queue[file_id].processed = True
        st.session_state.processed_files.append(st.session_state.file_queue[file_id].name)

def _clear_processed_files() -> None:
    """Clear all processed files from queue"""
    _init_file_queue()
    st.session_state.file_queue = {
        fid: f for fid, f in st.session_state.file_queue.items()
        if not f.processed
    }

def _get_queue_info() -> Dict[str, Any]:
    """Get queue statistics"""
    _init_file_queue()
    all_files = _get_queued_files()
    unprocessed = _get_unprocessed_files()

    total_size = sum(f.size for f in all_files)

    return {
        'total_files': len(all_files),
        'unprocessed_count': len(unprocessed),
        'total_size': total_size,
        'total_size_display': FileQueueItem(
            file_id="", name="", size=total_size, file_type="",
            bytes_data=b"", uploaded_at=datetime.now()
        ).get_size_display()
    }

def login_screen():
    """Render authentication screen"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<h1 class="main-header" style="text-align: center;">🦋 Ollama Pro v7.5</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="text-align: center;">Enterprise AI Platform</p>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submit:
                    user = AuthManager.authenticate(username, password)
                    if user:
                        SessionManager.persist_login(user)
                        st.success("Welcome back!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Choose Username")
                new_pass = st.text_input("Choose Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                reg_submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if reg_submit:
                    if new_pass!= confirm_pass:
                        st.error("Passwords don't match")
                    elif len((new_user or "").strip()) < 3:
                        st.error("Username must be at least 3 characters")
                    elif len(new_pass) < 8:
                        st.error("Password must be at least 8 characters")
                    elif AuthManager.create_user(new_user, new_pass):
                        st.success("Account created! Please sign in.")
                    else:
                        st.error("Username already exists")

def render_sidebar(client: OllamaClient):
    """Render sidebar with all controls"""
    user = st.session_state.user

    st.markdown(f"""
    <div class="sidebar-header">
        <h3 style="margin: 0;">🦋 Pro v7.5</h3>
        <p style="margin: 0.5rem 0; color: #666;">
            <span class="badge badge-primary">{user['role'].upper()}</span>
        </p>
        <p style="margin: 0; font-size: 0.875rem; color: #999;">@{user['username']}</p>
    </div>
    """, unsafe_allow_html=True)

    # User actions
    if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
        SessionManager.logout()
        st.rerun()

    st.divider()

    # Model selector with info
    ok, err, models_info = client.health()
    models = [m.name if hasattr(m, "name") else m.get("name") for m in models_info] if models_info else []

    if models:
        selected_model = st.selectbox(
            "🤖 Model",
            models,
            index=0 if st.session_state.model not in models else models.index(st.session_state.model)
        )
        st.session_state.model = selected_model

        # Show model info
        model_info = next((m for m in models_info if (m.name if hasattr(m, "name") else m.get("name")) == selected_model), None)
        with st.expander("Model Details"):
            if model_info:
                info_dict = asdict(model_info) if hasattr(model_info, "__dataclass_fields__") else dict(model_info)
                st.json({k: v for k, v in info_dict.items() if k!= "name"})

    st.divider()

    # Feature toggles
    st.markdown("**⚙️ Features**")

    st.toggle("🔍 RAG System", key="rag_enabled",
              help="Retrieval-Augmented Generation for knowledge base")

    st.toggle("🖼️ Vision", key="vision_enabled",
              help="Multi-modal image understanding")

    st.toggle("🔊 Text-to-Speech", key="tts_enabled",
              help="AI-generated voice responses")

    st.toggle("📊 Token Counter", key="show_token_count",
              help="Show real-time token usage")

    # Parameters
    with st.expander("🎛️ Parameters"):
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1,
                  help="Creativity vs determinism")
        st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
        st.slider("Context Window", 512, 4096, key="num_ctx", step=256)

        st.text_area("System Prompt", key="system_prompt", height=100)

    st.divider()

    # Session management
    st.markdown("**💬 Sessions**")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        new_id = SessionManager.create_session(user['id'])
        st.session_state.current_session = new_id
        st.rerun()

    # List previous sessions
    sessions = SessionManager.list_sessions(user['id'], limit=20)
    for sess in sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            title = sess['title'][:25] + "..." if len(sess['title']) > 25 else sess['title']
            if st.button(f"📝 {title}", key=f"sess_{sess['id']}", use_container_width=True):
                loaded = SessionManager.load_session(sess['id'])
                if loaded:
                    st.session_state.messages = loaded['messages']
                    st.session_state.current_session = sess['id']
                    st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sess['id']}"):
                try:
                    db_path = Path(CONFIG.DATABASE_URL.replace("sqlite:///", ""))
                    with sqlite3.connect(db_path) as conn:
                        conn.execute(
                            "DELETE FROM sessions WHERE id =? AND user_id =?",
                            (sess['id'], user['id']),
                        )
                        conn.commit()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
                if st.session_state.current_session == sess['id']:
                    st.session_state.messages = []
                    st.session_state.current_session = None
                    cancelled = _pop_job(user['id'], sess['id'])
                    if cancelled is not None:
                        cancelled.stop_event.set()
                st.rerun()

    # Knowledge base
    st.divider()
    with st.expander("📚 Knowledge Base"):
        uploaded = st.file_uploader("Upload documents",
                                   accept_multiple_files=True,
                                   type=CONFIG.ALLOWED_UPLOAD_TYPES)
        if uploaded and st.button("Index Documents"):
            processor = MultimodalProcessor()
            rag = _init_global_rag()

            progress = st.progress(0)
            for i, file in enumerate(uploaded):
                content = processor.extract_document_text(file.getvalue(), file.name)
                rag.add_document(user['id'], file.name, content, processor)
                progress.progress((i + 1) / len(uploaded))

            st.success(f"Indexed {len(uploaded)} documents")

def _build_ollama_messages(system_prompt: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert recent app messages to the Ollama wire format."""
    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for m in history[-6:]:
        msg_data = {"role": m["role"], "content": m["content"]}
        if m.get("images"):
            msg_data["images"] = m["images"]
        msgs.append(msg_data)
    return msgs

def _finalize_job(job: GenerationJob, user: Dict[str, Any], rag_used: bool) -> None:
    """Persist the finished job's output and clear it from the registry."""
    with job.state_lock:
        text = job.full_response
        error = job.error
    stopped = job.stop_event.is_set()
    suffix = " [...stopped]" if stopped and text else ""

    if error and not text:
        content = f"⚠️ Error: {error}"
    elif not text:
        content = "⚠️ No response received. Please make sure Ollama is running and the model is loaded.\n\nRun: `ollama run " + st.session_state.get('model', 'llama3') + "`"
    else:
        content = text + suffix

    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })

    # Persist
    if st.session_state.current_session:
        try:
            SessionManager.save_session(
                st.session_state.current_session,
                st.session_state.messages,
            )
        except Exception as e:
            st.warning(f"Could not save session: {e}")

    # Analytics
    try:
        Analytics.log_interaction(
            user['id'],
            st.session_state.current_session or "unsaved",
            st.session_state.model,
            0,
            len(text),
            (time.time() - job.started_at) * 1000,
            {"rag_used": rag_used, "stopped": stopped, "error": bool(error)},
        )
    except Exception:
        pass

    st.session_state.pending_images = []
    st.session_state.partial_response = ""
    st.session_state.stop_requested = False
    _pop_job(user['id'], st.session_state.current_session)

def render_file_queue_display():
    """Display file queue with metadata"""
    _init_file_queue()
    unprocessed_files = _get_unprocessed_files()
    queue_info = _get_queue_info()

    if not unprocessed_files:
        return False

    st.markdown("### 📎 Files Ready to Process")

    # Queue statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 Files Queued", queue_info['unprocessed_count'])
    with col2:
        st.metric("💾 Total Size", queue_info['total_size_display'])
    with col3:
        st.metric("✅ Processed", len(st.session_state.processed_files))

    # File list with metadata
    st.markdown("#### Queued Files:")

    for file_item in unprocessed_files:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.write(f"📄 **{file_item.name}**")
        with col2:
            st.caption(f"Size: {file_item.get_size_display()}")
        with col3:
            st.caption(f"Type: {file_item.file_type}")
        with col4:
            st.caption(f"Added: {file_item.uploaded_at.strftime('%H:%M')}")

    st.markdown("---")
    return True

def _extract_file_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract plain text from any supported file type.
    Returns empty string on failure.
    Supports: PDF, TXT, MD, JSON, CSV, XLSX, PY
    """
    ext = Path(filename).suffix.lower()

    try:
        # ── PDF ─────────────────────────────────────────────────────────────
        if ext == ".pdf":
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages[:20]: # max 20 pages
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                return "\n\n".join(text_parts)
            except ImportError:
                pass
            # fallback: PyPDF2
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                pages = []
                for page in reader.pages[:20]:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                return "\n\n".join(pages)
            except ImportError:
                return "[PDF extraction requires: pip install pdfplumber]"

        # ── Plain text, Markdown, Python ──────────────────────────────────
        elif ext in (".txt", ".md", ".py"):
            return file_bytes.decode("utf-8", errors="ignore")

        # ── JSON ─────────────────────────────────────────────────────────
        elif ext == ".json":
            data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            return json.dumps(data, indent=2, ensure_ascii=False)

        # ── CSV ──────────────────────────────────────────────────────────
        elif ext == ".csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
            summary = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            summary += df.head(50).to_string(index=False)
            return summary

        # ── Excel ─────────────────────────────────────────────────────────
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
            summary = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            summary += df.head(50).to_string(index=False)
            return summary

    except Exception as e:
        return f"[Could not extract text: {e}]"

    return ""

def render_business_analysis_tab(client: OllamaClient):
    """
    🏢 Business Analysis Tab
    Upload any file → all 7 AI agents (CEO/CFO/COO/CTO/HR/Sales/PM)
    analyze it → generate a full professional report.
    """
    st.markdown("## 🏢 Multi-Agent Business Analysis")
    st.markdown(
        "Upload a document and let **7 AI executives** analyze it from their perspective. "
        "Get a full professional report in one click."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "📂 Upload document for business analysis",
            type=["pdf", "xlsx", "xls", "csv", "txt", "md", "json"],
            key="biz_upload",
            help="PDF, Excel, CSV, or any text document"
        )
    with col2:
        st.markdown("**🤖 Agents that will run:**")
        agents_list = [
            ("👔", "CEO", "Strategy & risks"),
            ("💰", "CFO", "Financial analysis"),
            ("⚙️", "COO", "Operations"),
            ("💻", "CTO", "Technology"),
            ("👥", "HR", "Team & hiring"),
            ("📈", "Sales", "Revenue & market"),
            ("📊", "PM", "Project plan"),
        ]
        for emoji, role, desc in agents_list:
            st.markdown(f"{emoji} **{role}** — {desc}")

    if not uploaded:
        st.info("👆 Upload a file above to start the analysis.")
        return

    # Show file info
    file_bytes = uploaded.getvalue()
    file_size_kb = len(file_bytes) / 1024
    st.success(f"✅ **{uploaded.name}** loaded ({file_size_kb:.1f} KB)")

    # Agent selection
    with st.expander("⚙️ Customize agents (optional)", expanded=False):
        selected = st.multiselect(
            "Select agents to run",
            ["ceo", "cfo", "coo", "cto", "hr", "sales", "pm"],
            default=["ceo", "cfo", "coo", "cto", "hr", "sales", "pm"],
        )
        run_parallel = st.toggle("⚡ Run agents in parallel (faster)", value=True)

    if st.button("🚀 Run Business Analysis", type="primary", use_container_width=True):
        # 1. Extract text
        with st.spinner(f"📄 Extracting content from {uploaded.name}..."):
            context = _extract_file_text(uploaded.name, file_bytes)

        if not context or context.startswith("[Could not"):
            st.error(f"❌ Could not extract text: {context}")
            return

        st.info(f"📝 Extracted {len(context):,} characters from document.")

        # 2. Run agents
        try:
            from features.multi_task.business_orchestrator import BusinessOrchestrator
            orchestrator = BusinessOrchestrator(
                ollama_client=client,
                model=st.session_state.model,
                parallel=run_parallel,
                agents_to_run=selected if selected else ["ceo", "cfo", "coo"],
            )
        except ImportError:
            st.error("❌ BusinessOrchestrator not found. Make sure business_orchestrator.py is in features/multi_task/")
            return

        progress_bar = st.progress(0, text="Starting agents...")
        status_placeholder = st.empty()

        total = len(selected)
        result = {"success": False}

        with st.spinner("🤖 All agents analyzing document..."):
            try:
                result = orchestrator.run(context=context, file_name=uploaded.name)
                progress_bar.progress(100, text="✅ All agents complete!")
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        if not result.get("success"):
            st.error("Analysis did not complete successfully.")
            return

        # 3. Display results
        st.markdown("---")
        st.markdown("## 📋 Analysis Results")

        outputs = result.get("agent_outputs", [])
        for output in outputs:
            with st.expander(f"{output['emoji']} {output['role']}", expanded=True):
                if output["success"]:
                    st.markdown(output["analysis"])
                else:
                    st.error(output["analysis"])

        st.markdown("---")
        st.markdown("## 🔥 Final Recommendations")
        st.info(
            "Based on combined agent analysis:\n\n"
            "1. **Now (0-30 days):** Address critical risks from CEO + CFO analysis.\n"
            "2. **Short-term (1-3 months):** Execute Phase 1 from PM's project plan. Fill key HR gaps.\n"
            "3. **Medium-term (3-6 months):** Deploy CTO's tech roadmap + Sales go-to-market strategy.\n"
            "4. **Review quarterly** and re-run this analysis as the business evolves."
        )

        # 4. Download buttons
        st.markdown("---")
        st.markdown("### 📥 Download Report")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                "📄 Download Markdown Report",
                data=result["report_markdown"].encode("utf-8"),
                file_name=f"business_report_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary",
            )
        with dl_col2:
            st.download_button(
                "🌐 Download HTML Report",
                data=result["report_html"].encode("utf-8"),
                file_name=f"business_report_{datetime.now():%Y%m%d_%H%M}.html",
                mime="text/html",
                use_container_width=True,
            )

def render_chat_interface(client: OllamaClient):
    """Main chat interface with enhanced file handling"""
    user = st.session_state.user
    _init_file_queue()

    # Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h2 style="margin: 0;">💬 Chat</h2>
        <span class="badge badge-success">● {st.session_state.model}</span>
    </div>
    """, unsafe_allow_html=True)

    # Is there a generation currently running for this (user, session)?
    job = _get_job(user['id'], st.session_state.current_session)
    is_generating = job is not None and not job.done

    # Render historical messages
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            images = msg.get('images', [])
            UIComponents.render_message(
                msg['role'],
                msg['content'],
                msg.get('timestamp'),
                images if msg['role'] == 'user' else None,
                idx=i,
            )

        # Live streaming view
        if job is not None:
            _live_response_fragment(user)

    # Input area
    st.divider()

    # Display file queue
    has_queued_files = render_file_queue_display()

    # Modern compact uploader
    st.markdown("""
    <style>
    div[data-testid="stPopover"] > button {
        border-radius: 50%!important;
        width: 38px!important;
        height: 38px!important;
        min-width: 38px!important;
        padding: 0!important;
        font-size: 20px!important;
        font-weight: 300!important;
        background: #ffffff!important;
        border: 1px solid #e5e7eb!important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08)!important;
        color: #111827!important;
        margin-bottom: 6px;
    }
    div[data-testid="stPopover"] > button:hover {
        background: #f9fafb!important;
        border-color: #d1d5db!important;
    }
    div[data-testid="stPopoverBody"] {
        padding: 12px!important;
        min-width: 240px!important;
    }
    </style>
    """, unsafe_allow_html=True)

    input_cols = st.columns([0.06, 0.94], vertical_alignment="bottom")

    img_uploader = None
    with input_cols[0]:
        with st.popover("＋", help="Attach files"):
            st.markdown("**📎 Upload Files**")

            img_uploader = st.file_uploader(
                "📷 Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="img_upload",
                label_visibility="visible",
            )

            audio_file = st.file_uploader(
                "🎤 Audio",
                type=["wav", "mp3", "m4a"],
                key="audio_upload",
                label_visibility="visible",
            )
            if audio_file is not None:
                st.session_state['pending_audio'] = {
                    "name": audio_file.name,
                    "bytes": audio_file.getvalue(),
                }

            doc_uploader = st.file_uploader(
                "📄 Document",
                type=["xlsx", "xls", "csv", "py", "json", "txt", "md", "pdf"],
                key="doc_upload",
                label_visibility="visible",
                help="Upload PDF, Excel, CSV, Python files — AI will read and analyze them"
            )

            # ✅ FIX 1: Queue the file instead of replacing it
            if doc_uploader is not None:
                file_id = _add_file_to_queue(doc_uploader)
                st.success(f"✅ {doc_uploader.name} added to queue!")

    prompt: Optional[str] = None
    with input_cols[1]:
        if is_generating:
            if st.button("⏹️ Stop generating", key="stop_btn",
                         use_container_width=True, type="secondary"):
                job.stop_event.set()
                rag_used = bool(st.session_state.pop('_last_rag_used', False))
                _finalize_job(job, user, rag_used)
                st.session_state.stop_requested = True
                st.toast("Stopped")
                st.rerun()
        else:
            prompt = st.chat_input("Message Ollama... (files will be processed together)")

    # Store uploaded image
    if img_uploader and st.session_state.vision_enabled:
        img_data = MultimodalProcessor.process_image(img_uploader.getvalue())
        st.session_state.pending_images.append(img_data)
        st.toast(f"📷 Image ready ({img_data['size'][0]}x{img_data['size'][1]})")

    # ✅ FIX 2: Enhanced file + prompt handling
    if prompt and not is_generating:
        prompt = SecurityManager.sanitize_input(prompt)

        # Get unprocessed files from queue
        unprocessed_files = _get_unprocessed_files()

        # Check if user wants to process files
        clean_keywords = ['clean', 'saaf', 'saf', 'fix', 'theek', 'saaf karo', 'clean karo', 'theek karo', 'saf karo']
        has_clean_intent = any(kw in prompt.lower() for kw in clean_keywords)

        # ✅ FIX 3: Process queued files with prompt
        if unprocessed_files and has_clean_intent:
            # Process each queued file
            for file_item in unprocessed_files:
                doc_name = file_item.name
                doc_bytes = file_item.bytes_data

                # Add user message
                user_msg = {
                    'role': 'user',
                    'content': f"📎 **{doc_name}**\n\n{prompt}",
                    'timestamp': datetime.now().isoformat(),
                    'images': []
                }
                st.session_state.messages.append(user_msg)
                _mark_file_processed(file_item.file_id)

                # Process file
                with st.chat_message("assistant"):
                    with st.status(f"🔍 Processing {doc_name}...", expanded=True) as status:
                        try:
                            from core.file_doctor import FileDoctor

                            # Save temp file
                            suffix = Path(doc_name).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
                                tmp.write(doc_bytes)
                                tmp_path = tmp.name

                            doctor = FileDoctor(client, st.session_state.model)

                            if doc_name.endswith(('.xlsx', '.xls', '.csv')):
                                # Excel/CSV
                                diag = doctor.diagnose_excel(tmp_path)

                                if "error" in diag:
                                    st.error(f"❌ {diag['error']}")
                                else:
                                    st.write(f"📊 **Size:** {diag['rows']} rows × {diag['cols']} cols")
                                    st.write(f"🐛 **Problems:** {len(diag['issues'])}")

                                    if diag['issues']:
                                        with st.expander("Issues found"):
                                            for issue in diag['issues']:
                                                st.write(f"• {issue}")

                                    status.update(label="🧹 Cleaning...", state="running")
                                    cleaned_df, msg = doctor.clean_excel(diag)

                                    st.success(f"✅ {msg}")
                                    st.dataframe(cleaned_df.head(10), use_container_width=True)

                                    # Download
                                    output_name = f"cleaned_{doc_name}"
                                    if doc_name.endswith('.csv'):
                                        cleaned_df.to_csv(output_name, index=False)
                                        mime = "text/csv"
                                    else:
                                        cleaned_df.to_excel(output_name, index=False)
                                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                                    with open(output_name, 'rb') as f:
                                        st.download_button(
                                            f"⬇️ Download {output_name}",
                                            f,
                                            file_name=output_name,
                                            mime=mime,
                                            use_container_width=True,
                                            type="primary",
                                            key=f"dl_{int(time.time())}"
                                        )
                                    Path(output_name).unlink(missing_ok=True)

                            elif doc_name.endswith('.py'):
                                # Python
                                diag = doctor.diagnose_python(tmp_path)

                                st.write(f"📄 **Lines:** {diag['lines']}")
                                st.write(f"🐛 **Problems:** {len(diag['issues'])}")

                                if diag['issues']:
                                    with st.expander("Issues found"):
                                        for issue in diag['issues']:
                                            st.write(f"• {issue}")

                                status.update(label="🧹 Cleaning...", state="running")
                                cleaned_code, msg = doctor.clean_python(diag)

                                st.success(f"✅ {msg}")

                                tab1, tab2 = st.tabs(["✨ Cleaned", "📝 Original"])
                                with tab1:
                                    st.code(cleaned_code, language='python')
                                with tab2:
                                    st.code(diag['code'][:1500], language='python')

                                st.markdown("---")
                                st.download_button(
                                    label=f"⬇️ Download cleaned_{doc_name}",
                                    data=cleaned_code,
                                    file_name=f"cleaned_{doc_name}",
                                    mime="text/x-python",
                                    use_container_width=True,
                                    type="primary",
                                    key=f"py_dl_{int(time.time())}"
                                )

                            Path(tmp_path).unlink(missing_ok=True)
                            status.update(label="✅ Done!", state="complete")

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            status.update(label="❌ Failed", state="error")

            # Clear processed files
            _clear_processed_files()
            st.rerun()
            return

        # ✅ FIX 2: Process queued files with ANY prompt — not just "clean" keywords
        if unprocessed_files and prompt:
            clean_keywords = ['clean', 'saaf', 'saf', 'fix', 'theek', 'saaf karo', 'clean karo', 'theek karo', 'saf karo']
            has_clean_intent = any(kw in prompt.lower() for kw in clean_keywords)
            has_analyze_intent = any(kw in prompt.lower() for kw in [
                'analyze', 'analyse', 'summary', 'summarize', 'report', 'read', 'check',
                'what', 'tell me', 'explain', 'show', 'review', 'business', 'agents'
            ])

            for file_item in unprocessed_files:
                doc_name = file_item.name
                doc_bytes = file_item.bytes_data
                ext = Path(doc_name).suffix.lower()

                # ── A: Structured file cleaning (Excel / CSV / Python) ─────────
                if has_clean_intent and ext in ('.xlsx', '.xls', '.csv', '.py'):
                    user_msg = {
                        'role': 'user',
                        'content': f"📎 **{doc_name}**\n\n{prompt}",
                        'timestamp': datetime.now().isoformat(),
                        'images': []
                    }
                    st.session_state.messages.append(user_msg)
                    _mark_file_processed(file_item.file_id)

                    with st.chat_message("assistant"):
                        with st.status(f"🔍 Processing {doc_name}...", expanded=True) as status:
                            try:
                                from core.file_doctor import FileDoctor
                                suffix = ext
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
                                    tmp.write(doc_bytes)
                                    tmp_path = tmp.name

                                doctor = FileDoctor(client, st.session_state.model)

                                if ext in ('.xlsx', '.xls', '.csv'):
                                    diag = doctor.diagnose_excel(tmp_path)
                                    if "error" in diag:
                                        st.error(f"❌ {diag['error']}")
                                    else:
                                        st.write(f"📊 **Size:** {diag['rows']} rows × {diag['cols']} cols")
                                        st.write(f"🐛 **Problems:** {len(diag['issues'])}")
                                        if diag['issues']:
                                            with st.expander("Issues found"):
                                                for issue in diag['issues']:
                                                    st.write(f"• {issue}")
                                        status.update(label="🧹 Cleaning...", state="running")
                                        cleaned_df, msg = doctor.clean_excel(diag)
                                        st.success(f"✅ {msg}")
                                        st.dataframe(cleaned_df.head(10), use_container_width=True)
                                        output_name = f"cleaned_{doc_name}"
                                        if ext == '.csv':
                                            out_bytes = cleaned_df.to_csv(index=False).encode()
                                            mime = "text/csv"
                                        else:
                                            buf = io.BytesIO()
                                            cleaned_df.to_excel(buf, index=False)
                                            out_bytes = buf.getvalue()
                                            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        st.download_button(
                                            f"⬇️ Download {output_name}", out_bytes,
                                            file_name=output_name, mime=mime,
                                            use_container_width=True, type="primary",
                                            key=f"dl_{int(time.time())}"
                                        )

                                elif ext == '.py':
                                    diag = doctor.diagnose_python(tmp_path)
                                    st.write(f"📄 **Lines:** {diag['lines']}")
                                    st.write(f"🐛 **Problems:** {len(diag['issues'])}")
                                    if diag['issues']:
                                        with st.expander("Issues found"):
                                            for issue in diag['issues']:
                                                st.write(f"• {issue}")
                                    status.update(label="🧹 Cleaning...", state="running")
                                    cleaned_code, msg = doctor.clean_python(diag)
                                    st.success(f"✅ {msg}")
                                    tab1, tab2 = st.tabs(["✨ Cleaned", "📝 Original"])
                                    with tab1:
                                        st.code(cleaned_code, language='python')
                                    with tab2:
                                        st.code(diag['code'][:1500], language='python')
                                    st.download_button(
                                        label=f"⬇️ Download cleaned_{doc_name}",
                                        data=cleaned_code,
                                        file_name=f"cleaned_{doc_name}",
                                        mime="text/x-python",
                                        use_container_width=True, type="primary",
                                        key=f"py_dl_{int(time.time())}"
                                    )

                                Path(tmp_path).unlink(missing_ok=True)
                                status.update(label="✅ Done!", state="complete")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                status.update(label="❌ Failed", state="error")

                # ── B: PDF / TXT / MD / JSON / any readable file → inject as chat context ──
                else:
                    extracted_text = _extract_file_text(doc_name, doc_bytes)
                    if extracted_text:
                        # Inject file content into the prompt as context
                        file_context = (
                            f"\n\n---\n📎 **Attached file: {doc_name}**\n\n"
                            f"{extracted_text[:6000]}" # cap at 6000 chars to stay in context window
                            f"\n---\n"
                        )
                        augmented_prompt = prompt + file_context
                        _mark_file_processed(file_item.file_id)

                        user_msg = {
                            'role': 'user',
                            'content': f"📎 **{doc_name}** attached\n\n{prompt}",
                            'timestamp': datetime.now().isoformat(),
                            'images': []
                        }
                        st.session_state.messages.append(user_msg)

                        # Build messages with file context injected
                        ollama_msgs_with_file = _build_ollama_messages(
                            st.session_state.system_prompt,
                            st.session_state.messages[:-1] # history without the just-appended msg
                        )
                        ollama_msgs_with_file.append({
                            "role": "user",
                            "content": augmented_prompt
                        })

                        if st.session_state.current_session:
                            SessionManager.save_session(
                                st.session_state.current_session,
                                st.session_state.messages,
                                {"model": st.session_state.model},
                            )

                        _start_generation(
                            client, st.session_state.model,
                            ollama_msgs_with_file,
                            {
                                "temperature": st.session_state.temperature,
                                "top_p": st.session_state.top_p,
                                "num_ctx": st.session_state.num_ctx,
                                "num_predict": 1200,
                            },
                            user['id'], st.session_state.current_session,
                        )
                        _clear_processed_files()
                        st.rerun()
                        return
                    else:
                        st.warning(f"⚠️ Could not extract text from {doc_name}")

            _clear_processed_files()
            st.rerun()
            return

        # RAG context (only reached when NO queued files)
        context = ""
        rag_used = False
        if st.session_state.rag_enabled:
            if is_greeting(prompt):
                pass
            elif should_use_rag(prompt):
                rag = _init_global_rag()
                if rag:
                    hits = rag.query(prompt, None)
                    if hits:
                        bullets = []
                        try:
                            if hasattr(hits, 'hits'):
                                hits_list = list(hits.hits)
                            elif hasattr(hits, 'results'):
                                hits_list = list(hits.results)
                            elif hasattr(hits, 'documents'):
                                hits_list = list(hits.documents)
                            elif hasattr(hits, '__iter__') and not isinstance(hits, (str, bytes, dict)):
                                hits_list = list(hits)
                            else:
                                hits_list = [hits] if hits else []
                        except:
                            hits_list = []

                        for h in hits_list[:3]:
                            if hasattr(h, 'content'):
                                text = h.content
                            elif isinstance(h, dict):
                                text = h.get('text', '')
                            else:
                                text = str(h) if h else ''
                            sent = re.split(r'(?<=[.!?])\s+', text.strip())[0] if text else ""
                            sent = ' '.join(sent.split()[:12])
                            if sent:
                                bullets.append(f"- {sent}")
                        if bullets:
                            context = "\n\nRelevant context (use ONLY these bullets):\n" + "\n".join(bullets)
                            rag_used = True

        user_msg: Dict[str, Any] = {
            "role": "user",
            "content": prompt + context,
            "timestamp": datetime.now().isoformat(),
        }
        if st.session_state.pending_images:
            user_msg["images"] = [img["base64"] for img in st.session_state.pending_images]

        st.session_state.messages.append(user_msg)
        st.session_state['_last_rag_used'] = rag_used

        if st.session_state.current_session:
            SessionManager.save_session(
                st.session_state.current_session,
                st.session_state.messages,
                {"model": st.session_state.model},
            )

        ollama_msgs = _build_ollama_messages(st.session_state.system_prompt, st.session_state.messages)
        _start_generation(
            client,
            st.session_state.model,
            ollama_msgs,
            {
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "num_ctx": st.session_state.num_ctx,
                "num_predict": 600,
            },
            user['id'],
            st.session_state.current_session,
        )
        st.rerun()

def _fragment_or_plain(run_every=None):
    """Decorator that uses @st.fragment(run_every=...) if supported, else plain function."""
    def decorator(fn):
        if _FRAGMENT_SUPPORTED and run_every is not None:
            try:
                return st.fragment(run_every=run_every)(fn)
            except Exception:
                return fn
        return fn
    return decorator

@_fragment_or_plain(run_every=0.15)
def _live_response_fragment(user: Dict[str, Any]) -> None:
    """Auto-rerunning slice that mirrors the worker thread's progress."""
    job = _get_job(user['id'], st.session_state.current_session)
    if job is None:
        return

    with job.state_lock:
        text = job.full_response
        done = job.done

    if not text and not done:
        st.markdown("""
        <div style="display: flex; justify-content: flex-start;">
            <div style="margin-right: 0.5rem;">🦙</div>
            <div class="assistant-message">
                <div class="thinking-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cursor = "" if done else "▌"
        UIComponents.render_message("assistant", text + cursor, datetime.now())

    if done:
        rag_used = bool(st.session_state.pop('_last_rag_used', False))
        _finalize_job(job, user, rag_used)
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()

def render_agents_tab(client: OllamaClient):
    """Multi-agent workflow interface"""
    user = st.session_state.user
    if not isinstance(st.session_state.agents, dict):
        st.session_state.agents = {}

    if not SecurityManager.check_permission(user['role'], "agents"):
        st.error("Upgrade to Pro to use AI Agents")
        return

    st.header("🤖 AI Agents")
    st.markdown("Create and orchestrate multi-agent workflows")

    if AgentOrchestrator is None:
        st.error("Agent Orchestrator not available")
        return

    orchestrator = AgentOrchestrator(client)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Create Agent")
        with st.form("agent_form"):
            agent_name = st.text_input("Agent Name")
            agent_prompt = st.text_area("System Prompt", height=150)
            agent_tools = st.multiselect("Tools", list(orchestrator.AVAILABLE_TOOLS.keys()) if hasattr(orchestrator, 'AVAILABLE_TOOLS') else [])
            agent_model = st.selectbox("Model", [get_optimal_model(), "llama3.1", "llama3.2:3b", "gemma2:2b", "llama3.2:1b"], key="agent_model")

            if st.form_submit_button("Create Agent"):
                agent = orchestrator.create_agent(agent_name, agent_prompt, agent_tools, agent_model)
                st.session_state.agents[agent_name] = agent
                st.success(f"Agent '{agent_name}' created!")

        if st.session_state.agents:
            st.subheader("Your Agents")
            for name, agent in st.session_state.agents.items():
                with st.container():
                    st.markdown(f"""
                    <div class="agent-box">
                        <strong>{name}</strong><br>
                        <small>Tools: {', '.join(agent.tools if hasattr(agent, 'tools') else [])}</small>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        if MULTI_TASK_AVAILABLE:
            st.subheader("🚀 Run 5 Tasks Parallel")
            st.caption("Manager → splits task → 5 agents work simultaneously")

            multi_task_input = st.text_input(
                "Project name",
                placeholder="e.g., Build AI App with database",
                key="multi_task_input"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                run_multi = st.button("⚡ Run 5 Tasks", type="primary", use_container_width=True)
            with col_b:
                st.caption("Creates 5 parallel agents")

            if run_multi and multi_task_input:
                with st.spinner(f"Running 5 parallel tasks for '{multi_task_input}'..."):
                    try:
                        results, final = run_multi_task_project(multi_task_input)

                        st.success(f"✅ Completed {len(results)} tasks in parallel!")

                        tab1, tab2 = st.tabs(["📊 Results", "📝 Final Report"])

                        with tab1:
                            for i, r in enumerate(results, 1):
                                with st.expander(f"Task {i}: {r['task'][:50]}...", expanded=(i==1)):
                                    st.markdown(f"**Agent:** {r['agent']}")
                                    st.markdown(f"**Time:** {r.get('time', 'N/A')}s")
                                    st.markdown("**Result:**")
                                    st.write(r['result'][:500] + "..." if len(r['result']) > 500 else r['result'])

                        with tab2:
                            st.markdown("### Final Combined Report")
                            st.write(final)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Make sure you've created the features/multi_task folder with PowerShell script")

            st.divider()

        st.subheader("Run Workflow")
        task = st.text_area("Task Description", height=100,
                           placeholder="Describe the task you want agents to complete...")

        available_agents = list(st.session_state.agents.keys())
        selected_agents = st.multiselect("Select Agents", available_agents)

        if st.button("▶️ Run Workflow", type="primary") and task and selected_agents:
            result_placeholder = st.empty()
            full_output = ""

            for chunk in orchestrator.execute_workflow(task, selected_agents):
                full_output += chunk
                result_placeholder.markdown(full_output)

def render_analytics_tab():
    """Analytics dashboard"""
    user = st.session_state.user

    if not SecurityManager.check_permission(user['role'], "analytics"):
        st.error("Access denied")
        return

    st.header("📊 Analytics Dashboard")

    data = Analytics.get_dashboard_data(user['id'])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        UIComponents.metric_card(
            "Total Interactions",
            str(data['stats'].get('total_interactions', 0))
        )
    with col2:
        tokens = data['stats'].get('total_tokens', 0)
        UIComponents.metric_card(
            "Total Tokens",
            f"{tokens:,}" if tokens else "0"
        )
    with col3:
        latency = data['stats'].get('avg_latency', 0)
        UIComponents.metric_card(
            "Avg Latency",
            f"{latency:.0f}ms" if latency else "0ms"
        )
    with col4:
        sessions = len(SessionManager.list_sessions(user['id']))
        UIComponents.metric_card("Sessions", str(sessions))

    st.divider()

    if ANALYTICS_PRO_AVAILABLE:
        st.subheader("🤖 Pro Analytics Engine")
        st.caption("Upload CSV/Excel to compute domain KPIs")

        uploaded = st.file_uploader("Choose data file", type=['csv','xlsx','xls'], key="ana_upload")
        domain = st.selectbox("Domain", ["finance","sales","hr","supply_chain","operations","marketing",
                                       "ecommerce","logistics","healthcare","manufacturing","real_estate",
                                       "education","legal","general"], key="ana_domain")

        if uploaded and st.button("Compute KPIs", type="primary", key="compute_kpi"):
            with st.spinner("Computing..."):
                try:
                    # Save temp
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
                    tmp.write(uploaded.getvalue()); tmp.close()

                    agent = AnalyticsAgent()
                    agent.upload_file(tmp.name)
                    result = agent.compute(domain=domain)

                    # Store in session
                    st.session_state.last_analytics_result = result

                    st.success(f"Computed {len(result.kpis)} KPIs for {domain}")

                    # Show quick preview
                    kpi_df = pd.DataFrame([asdict(k) for k in result.kpis])
                    st.dataframe(kpi_df[['name','value','unit','flag']], use_container_width=True)

                    if result.insights:
                        st.info("**Top Insights:**\n" + "\n".join([f"• {i}" for i in result.insights[:3]]))

                    os.unlink(tmp.name)
                except Exception as e:
                    st.error(f"Compute failed: {e}")
    else:
        st.info("Install AnalyticsAgent Pro for KPI computation")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Usage")
        if data['daily']:
            df = pd.DataFrame(data['daily'])
            if px:
                fig = px.bar(df, x='day', y='tokens', title='Token Usage by Day')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index('day')['tokens'])
        else:
            st.info("No data available")

    with col2:
        st.subheader("Model Usage")
        if data['models']:
            df = pd.DataFrame(data['models'])
            if px:
                fig = px.pie(df, values='uses', names='model', title='Model Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No data available")

def render_sql_builder(client: OllamaClient):
    """Natural language to SQL"""
    user = st.session_state.user

    st.header("🛠️ SQL Builder")
    st.markdown("Convert natural language to SQL queries")

    col1, col2 = st.columns([1, 1])

    with col1:
        schema = st.text_area("Database Schema", height=300,
                             placeholder="CREATE TABLE users (id INT, name TEXT);\nCREATE TABLE orders (id INT, user_id INT,...)",
                             help="Paste your CREATE TABLE statements")

    with col2:
        question = st.text_input("Your Question",
                                placeholder="Find all users who made purchases last month")

        dialect = st.selectbox("SQL Dialect", ["PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle"])

        if st.button("Generate SQL", type="primary") and question and schema:
            prompt = f"""You are an expert SQL developer. Convert the following natural language question into {dialect} SQL.

Schema:
{schema}

Question: {question}

Requirements:
- Return ONLY the SQL query, no explanations
- Use proper {dialect} syntax
- Add comments for complex logic
- Use best practices for performance

SQL:"""

            with st.spinner("Generating..."):
                try:
                    sql, _ = client.chat_once(
                        st.session_state.model,
                        [{"role": "user", "content": prompt}],
                        {"temperature": 0.1}
                    )

                    sql = re.sub(r'^```sql?\s*', '', sql.strip())
                    sql = re.sub(r'```$', '', sql.strip())

                    st.code(sql.strip(), language="sql")

                    if st.button("🔍 Explain Query"):
                        explain_prompt = f"Explain this SQL query in plain English:\n\n{sql}"
                        explanation, _ = client.chat_once(
                            st.session_state.model,
                            [{"role": "user", "content": explain_prompt}],
                            {"temperature": 0.3}
                        )
                        st.info(explanation)

                except Exception as e:
                    st.error(f"Error: {e}")

def render_project_generator():
    """Code project generator from chat"""
    st.header("📁 Project Generator")

    FILE_RE = re.compile(r"###\s*FILE:\s*(.+?)\n```(\w+)?\n(.*?)```", re.DOTALL)

    files = {}
    for msg in reversed(st.session_state.messages):
        if msg['role'] == 'assistant':
            found = {fp.strip(): (lang, code.strip())
                    for fp, lang, code in FILE_RE.findall(msg['content'])}
            if found:
                files = found
                break

    if not files:
        st.info("No code files detected in recent messages. Ask the assistant to generate code with file markers like:\n\n`### FILE: main.py` followed by code in triple backticks.")
        return

    st.subheader(f"Detected {len(files)} Files")

    for filepath, (lang, code) in files.items():
        with st.expander(f"📄 {filepath}"):
            st.code(code, language=lang or "text")

    col1, col2 = st.columns(2)

    with col1:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp, (lang, code) in files.items():
                zf.writestr(fp, code)

        st.download_button(
            "⬇️ Download as ZIP",
            buf.getvalue(),
            f"project_{datetime.now():%Y%m%d_%H%M}.zip",
            mime="application/zip",
            use_container_width=True
        )

    with col2:
        req_lines = []
        for fp, (lang, code) in files.items():
            if "requirements" in fp.lower() or "package" in fp.lower():
                req_lines.append(code)

        if st.button("📋 Copy All to Clipboard", use_container_width=True):
            all_code = "\n\n".join([f"// {fp}\n{code}" for fp, (_, code) in files.items()])
            st.code(all_code)
            st.success("Ready to copy!")

def render_export_tab():
    """Export conversation"""
    st.header("📤 Export")

    if not st.session_state.messages:
        st.info("No messages to export")
        return

    col1, col2, col3 = st.columns(3)

    mgr = ExportManager()
    title = st.session_state.get('current_session', 'Conversation')
    metadata = {"model": st.session_state.model, "user": st.session_state.user['username']}

    with col1:
        md_result = mgr.to_markdown(st.session_state.messages, title)
        st.download_button(
            "📄 Markdown",
            data=md_result.as_bytes(),
            file_name=md_result.filename,
            mime=md_result.mime_type,
            use_container_width=True
        )

    with col2:
        json_result = mgr.to_json(st.session_state.messages, metadata)
        st.download_button(
            "📋 JSON",
            data=json_result.as_bytes(),
            file_name=json_result.filename,
            mime=json_result.mime_type,
            use_container_width=True
        )

    with col3:
        html_result = mgr.to_html(st.session_state.messages, title)
        st.download_button(
            "🌐 HTML",
            data=html_result.as_bytes(),
            file_name=html_result.filename,
            mime=html_result.mime_type,
            use_container_width=True
        )

    st.divider()
    st.subheader("📈 Analytics Export")

    if not ANALYTICS_PRO_AVAILABLE:
        st.info("AnalyticsAgent Pro not available - install core.rag_system_pro")
    elif not st.session_state.get('last_analytics_result'):
        st.info("No analytics computed yet. Run analysis in Analytics tab first.")
    else:
        result = st.session_state.last_analytics_result

        # Convert KPIs
        from dataclasses import asdict
        kpi_df = pd.DataFrame([asdict(k) for k in result.kpis]) if result.kpis else pd.DataFrame()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if not kpi_df.empty:
                csv = kpi_df.to_csv(index=False)
                st.download_button("📊 KPIs CSV", csv, f"kpis_{datetime.now():%Y%m%d}.csv",
                                  "text/csv", key=f"kpi_csv_{uuid.uuid4().hex[:6]}", use_container_width=True)

        with col2:
            if not kpi_df.empty:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    kpi_df.to_excel(writer, sheet_name="KPIs", index=False)
                    # Add trends sheet if available
                    if result.trends:
                        trend_data = []
                        for col, t in result.trends.items():
                            d = asdict(t); d['column']=col
                            trend_data.append(d)
                        pd.DataFrame(trend_data).to_excel(writer, sheet_name="Trends", index=False)
                    # Add anomalies
                    if result.anomalies:
                        anom_data = []
                        for col, a in result.anomalies.items():
                            if a.count>0:
                                d = asdict(a); d['column']=col
                                anom_data.append(d)
                        if anom_data:
                            pd.DataFrame(anom_data).to_excel(writer, sheet_name="Anomalies", index=False)
                st.download_button("📗 KPIs Excel", buf.getvalue(), f"kpis_{datetime.now():%Y%m%d}.xlsx",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  key=f"kpi_xls_{uuid.uuid4().hex[:6]}", use_container_width=True)

        with col3:
            # JSON export of full result
            result_dict = {
                "domain": result.domain,
                "computed_at": result.computed_at,
                "kpis": [asdict(k) for k in result.kpis],
                "insights": result.insights,
                "correlations": result.correlations
            }
            st.download_button("📋 Full JSON", json.dumps(result_dict, indent=2),
                              f"analytics_{datetime.now():%Y%m%d}.json", "application/json",
                              key=f"ana_js_{uuid.uuid4().hex[:6]}", use_container_width=True)

        with col4:
            # Insights as text
            insights_txt = "\n".join([f"• {i}" for i in result.insights]) if result.insights else "No insights"
            st.download_button("📝 Insights", insights_txt, f"insights_{datetime.now():%Y%m%d}.txt",
                              "text/plain", key=f"ins_{uuid.uuid4().hex[:6]}", use_container_width=True)

        with st.expander("Preview KPIs"):
            st.dataframe(kpi_df, use_container_width=True)

    if st.session_state.user.get('role') == 'admin':
        st.divider()
        st.subheader("Database Backup")
        if st.button("Create Backup", type="primary"):
            db_path = Path("data/ollama_pro_enterprise.db")
            if db_path.exists():
                st.download_button(
                    "⬇️ Download Database",
                    db_path.read_bytes(),
                    f"backup_{datetime.now():%Y%m%d_%H%M}.db",
                    mime="application/x-sqlite3",
                    use_container_width=True
                )

def render_admin_database():
    """Admin Database Viewer"""
    st.header("🗄️ Database Admin")

    tab1, tab2 = st.tabs(["RAG System DB", "App DB"])

    with tab1:
        db_path = "rag_system.db"
        if not os.path.exists(db_path):
            st.warning(f"{db_path} not found - upload a document first to create it")
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        col1, col2, col3, col4 = st.columns(4)

        def _safe_count(sql: str, label: str, target) -> None:
            try:
                row = conn.execute(sql).fetchone()
                target.metric(label, row['c'] if row else 0)
            except sqlite3.OperationalError:
                target.metric(label, 0)
            except sqlite3.Error as e:
                target.metric(label, 0)
                st.caption(f"⚠️ {label}: {e}")

        _safe_count("SELECT COUNT(*) as c FROM documents", "Documents", col1)
        _safe_count("SELECT COUNT(*) as c FROM chunks", "Chunks", col2)
        _safe_count("SELECT COUNT(*) as c FROM chat_history", "Chat Messages", col3)
        col4.metric("DB Size", f"{os.path.getsize(db_path)/1024/1024:.2f} MB")

        st.divider()

        table = st.selectbox("Select Table", ["documents", "chunks", "chat_history", "agent_runs"])
        limit = st.slider("Rows", 10, 500, 50)

        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {limit}", conn)

            if 'embedding' in df.columns:
                df = df.drop(columns=['embedding'])
            for col in ['content', 'input', 'output']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str[:200] + "..."

            st.dataframe(df, use_container_width=True, height=400)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"{table}.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading table: {e}")

        conn.close()

    with tab2:
        st.info("Main app database viewer - configure path in code if needed")

def main():
    """Main application"""
    UIComponents.apply_custom_theme()

    SessionManager.init_session()

    defaults = {
        'user': None,
        'messages': [],
        'model': get_optimal_model(),
        'temperature': 0.7,
        'top_p': 0.9,
        'num_ctx': 2048,
        'system_prompt': 'You are a helpful assistant. Follow these rules strictly:\n1. For greetings or casual questions (e.g. "hello", "how are you", "what is your name"): reply in 1 sentence only.\n2. For simple factual questions: reply in 1-2 sentences only.\n3. For technical questions (code, SQL, math, programming): give a full detailed answer with examples.\n4. Never add filler, never repeat yourself, never explain what you are about to do.',
        'rag_enabled': False,
        'vision_enabled': True,
        'tts_enabled': False,
        'show_token_count': False,
        'pending_images': [],
        'current_session': None,
        'agents': {},
        'stop_requested': False,
        'partial_response': '',
        'file_queue': {},
        'processed_files': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not isinstance(st.session_state.agents, dict):
        st.session_state.agents = {}

    # --- UPDATED FOR OLLAMA CLOUD ---
    ollama_host = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
    client = OllamaClient(ollama_host)

    if not st.session_state.user:
        login_screen()
        return

    with st.sidebar:
        render_sidebar(client)

    tab_icons = ["💬", "🤖", "🛠️", "📊", "🏢", "📁", "📤"]
    tab_labels = ["Chat", "Agents", "SQL Builder", "Analytics", "Business Analysis", "Project", "Export"]

    available_tabs = []
    tab_panels = []

    for icon, label in zip(tab_icons, tab_labels):
        feature_key = label.lower().replace(" ", "_")
        permission_key = feature_key
        if feature_key == "sql_builder":
            permission_key = "sql_builder"
        if feature_key in ["chat", "project", "export"] or SecurityManager.check_permission(
            st.session_state.user.get('role', 'guest'),
            permission_key
        ):
            available_tabs.append(f"{icon} {label}")

    if not available_tabs:
        available_tabs = ["💬 Chat"]

    tabs = st.tabs(available_tabs)

    for i, tab in enumerate(tabs):
        with tab:
            tab_name = available_tabs[i].split(" ", 1)[1].lower().replace(" ", "_")

            if tab_name == "chat":
                render_chat_interface(client)
            elif tab_name == "agents":
                render_agents_tab(client)
            elif tab_name == "sql_builder":
                render_sql_builder(client)
            elif tab_name == "analytics":
                render_analytics_tab()
            elif tab_name == "business_analysis":
                render_business_analysis_tab(client)
            elif tab_name == "project":
                render_project_generator()
            elif tab_name == "export":
                render_export_tab()

if __name__ == "__main__":
    main()
