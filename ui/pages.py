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
try:
    from core.rag_system_pro import RAGSystem
    from core.rag_system_pro.agents import (
        ExcelAgent, CodeAgent, DocWriterAgent,
        WebResearchAgent, SQLAgent, LLMClient,
        available_agents
    )
    PRO_AGENTS_AVAILABLE = True
except ImportError:
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

OllamaClient = OllamaClientCloud

# --- PATCH: remove timestamp HTML ---
def _clean_render_message(role: str, content: str, timestamp=None, images=None, idx=None, **kwargs):
    try:
        with st.chat_message(role if role in ("user", "assistant") else "assistant"):
            if images and role == "user":
                cols = st.columns(min(len(images), 4))
                for i, img in enumerate(images[:4]):
                    with cols[i % 4]:
                        try:
                            st.image(img, use_container_width=True)
                        except Exception:
                            pass
            st.markdown(content)
    except Exception:
        st.write(f"**{role}:** {content}")

UIComponents.render_message = staticmethod(_clean_render_message)

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
    if PRO_AGENTS_AVAILABLE:
        rag = RAGSystem(
            llm_model=get_optimal_model(),
            embed_model="nomic-embed-text",
            index_path="data/faiss.index"
        )
    else:
        if RAGSystem is None:
            return None
        rag = RAGSystem()
        kb_path = Path("data/knowledge")
        try:
            if kb_path.exists():
                files = list(kb_path.rglob("*.md")) + list(kb_path.rglob("*.txt"))
                for f in files:
                    if any(s in f.name.lower() for s in ["prompt", "rules", "system", "instruction", "guideline"]):
                        continue
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")
                        rag.add_document("global", f.name, text)
                    except Exception:
                        pass
        except Exception:
            pass
    GLOBAL_RAG = rag
    return rag

_init_global_rag()

try:
    from features.multi_task.orchestrator import run_multi_task_project
    MULTI_TASK_AVAILABLE = True
except ImportError:
    MULTI_TASK_AVAILABLE = False

if not hasattr(SecurityManager, 'sanitize_input'):
    def _sanitize_input(text: str) -> str:
        import re
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        return text[:10000]
    SecurityManager.sanitize_input = staticmethod(_sanitize_input)

@dataclass
class GenerationJob:
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

def _start_generation(client, model, ollama_msgs, options, user_id, session_id):
    stop_event = threading.Event()
    chunk_queue = _queue_module.Queue()
    job = GenerationJob(stop_event=stop_event, chunk_queue=chunk_queue)
    def _worker():
        try:
            for chunk in client.chat_stream(model, ollama_msgs, options):
                if stop_event.is_set():
                    break
                with job.state_lock:
                    job.full_response += chunk
                chunk_queue.put(chunk)
        except Exception as e:
            with job.state_lock:
                job.error = str(e)
        finally:
            with job.state_lock:
                job.done = True
            chunk_queue.put(None)
    thread = threading.Thread(target=_worker, daemon=True)
    job.thread = thread
    _set_job(user_id, session_id, job)
    thread.start()
    return job

@dataclass
class FileQueueItem:
    file_id: str
    name: str
    size: int
    file_type: str
    bytes_data: bytes
    uploaded_at: datetime
    processed: bool = False
    def get_size_mb(self): return self.size / (1024 * 1024)
    def get_size_display(self):
        mb = self.get_size_mb()
        if mb > 1: return f"{mb:.2f} MB"
        kb = self.size / 1024
        if kb > 1: return f"{kb:.2f} KB"
        return f"{self.size} bytes"

def _init_file_queue():
    if 'file_queue' not in st.session_state:
        st.session_state.file_queue = {}
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def _add_file_to_queue(file_obj):
    _init_file_queue()
    file_id = str(uuid.uuid4())[:8]
    st.session_state.file_queue[file_id] = FileQueueItem(
        file_id, file_obj.name, len(file_obj.getvalue()),
        file_obj.type or "unknown", file_obj.getvalue(), datetime.now()
    )
    return file_id

def _get_queued_files():
    _init_file_queue()
    return list(st.session_state.file_queue.values())

def _get_unprocessed_files():
    return [f for f in _get_queued_files() if not f.processed]

def _mark_file_processed(fid):
    if fid in st.session_state.file_queue:
        st.session_state.file_queue[fid].processed = True
        st.session_state.processed_files.append(st.session_state.file_queue[fid].name)

def _clear_processed_files():
    st.session_state.file_queue = {k:v for k,v in st.session_state.file_queue.items() if not v.processed}

def login_screen():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<h1 style="text-align:center">🦋 Ollama Pro v7.5</h1>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Sign In","Create Account"])
        with tab1:
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    user = AuthManager.authenticate(u,p)
                    if user:
                        SessionManager.persist_login(user)
                        st.rerun()
                    else:
                        st.error("Invalid")
        with tab2:
            with st.form("reg"):
                nu = st.text_input("Username")
                np = st.text_input("Password", type="password")
                cp = st.text_input("Confirm", type="password")
                if st.form_submit_button("Create"):
                    if np==cp and len(nu)>=3 and len(np)>=8:
                        if AuthManager.create_user(nu,np):
                            st.success("Created")
                        else:
                            st.error("Exists")

def render_sidebar(client):
    user = st.session_state.user
    st.markdown(f"<div style='padding:1rem'><h3>🦋 Pro v7.5</h3><p>@{user['username']}</p></div>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", use_container_width=True):
        SessionManager.logout(); st.rerun()
    st.divider()

    ok, err, models_info = client.health()
    models = []
    for m in models_info or []:
        if hasattr(m, "name"): models.append(m.name)
        elif isinstance(m, dict): models.append(m.get("name") or m.get("model"))
        elif isinstance(m, str): models.append(m)
    models = [m for m in models if m]

    if models:
        if st.session_state.model not in models:
            st.session_state.model = models[0]
        selected = st.selectbox("🤖 Model", models, index=models.index(st.session_state.model))
        st.session_state.model = selected
    else:
        st.warning("No models - check API key")
        if err: st.error(err)

    st.toggle("🔍 RAG", key="rag_enabled")
    st.toggle("🖼️ Vision", key="vision_enabled")
    st.toggle("🔊 TTS", key="tts_enabled")

    with st.expander("Parameters"):
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
        st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
        st.slider("Context", 512, 4096, key="num_ctx", step=256)
        st.text_area("System", key="system_prompt", height=80)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session = SessionManager.create_session(user['id'])
        st.rerun()

def _build_ollama_messages(sys, hist):
    msgs = []
    if sys: msgs.append({"role":"system","content":sys})
    for m in hist[-6:]:
        d = {"role":m["role"],"content":m["content"]}
        if m.get("images"): d["images"]=m["images"]
        msgs.append(d)
    return msgs

def render_chat_interface(client):
    user = st.session_state.user
    st.markdown(f"## 💬 Chat - {st.session_state.model}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Message..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            msgs = _build_ollama_messages(st.session_state.system_prompt, st.session_state.messages)
            for chunk in client.chat_stream(st.session_state.model, msgs, {"temperature":st.session_state.temperature}):
                full += chunk
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        st.session_state.messages.append({"role":"assistant","content":full})

def main():
    st.set_page_config(page_title="Ollama Pro", layout="wide")
    SessionManager.init_session()
    defaults = {'user':None,'messages':[],'model':get_optimal_model(),'temperature':0.7,'top_p':0.9,'num_ctx':2048,'system_prompt':'You are helpful','rag_enabled':False,'vision_enabled':True,'tts_enabled':False,'current_session':None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

    ollama_host = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
    client = OllamaClient(ollama_host)

    if not st.session_state.user:
        login_screen()
        return

    with st.sidebar:
        render_sidebar(client)

    render_chat_interface(client)

if __name__ == "__main__":
    main()
