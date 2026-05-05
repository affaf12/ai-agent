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

try:
    _st_version = tuple(int(x) for x in _get_version("streamlit").split(".")[:2])
    _FRAGMENT_SUPPORTED = _st_version >= (1, 35)
except Exception:
    _FRAGMENT_SUPPORTED = hasattr(st, "fragment")

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from core.ollama_client import OllamaClient as _BaseOllamaClient
except ImportError:
    from ollama_client import OllamaClient as _BaseOllamaClient
from core.session import SessionManager
from core.auth import AuthManager, SecurityManager
from core.analytics import Analytics
try:
    from core.rag_system_pro import RAGSystem
    from core.rag_system_pro.agents import ExcelAgent, CodeAgent, DocWriterAgent, WebResearchAgent, SQLAgent, LLMClient, available_agents
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

try:
    from ollama import Client as _OllamaOfficial
    _OFFICIAL_AVAILABLE = True
except ImportError:
    _OFFICIAL_AVAILABLE = False

class OllamaClientCloud:
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
        try:
            if self._use_official:
                stream = self.client.chat(model=model, messages=messages, options=options or {}, stream=True)
                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        yield content
            else:
                yield from self.client.chat_stream(model, messages, options)
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                yield "⚠️ **Authentication Failed**\n\nCreate a NEW API key at ollama.com/settings/keys (your old key was leaked)"
            elif "404" in error_msg or "not found" in error_msg.lower():
                yield f"⚠️ **Model '{model}' not found**\n\nUse one of these exact names: `llama3.1`, `gemma2`, `qwen2.5`, or `mistral`"
            else:
                yield f"⚠️ **Error:** {error_msg}"

    def chat_once(self, model, messages, options):
        try:
            if self._use_official:
                resp = self.client.chat(model=model, messages=messages, options=options or {}, stream=False)
                return resp['message']['content'], None
            else:
                return self.client.chat_once(model, messages, options)
        except Exception as e:
            return f"Error: {str(e)}", None

OllamaClient = OllamaClientCloud

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

#... [keeping all your helper functions the same]...

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

    if not models:
        # FIXED: Real Ollama Cloud model names
        models = ["llama3.1", "gemma2", "qwen2.5", "mistral"]
        st.info("Using Ollama Cloud defaults")

    if st.session_state.model not in models:
        st.session_state.model = models[0]

    selected = st.selectbox("🤖 Model", models, index=models.index(st.session_state.model))
    st.session_state.model = selected

    if not ok and err:
        st.error(f"API Error: {err}")

    st.toggle("🔍 RAG", key="rag_enabled")
    st.toggle("🖼️ Vision", key="vision_enabled")

    with st.expander("Parameters"):
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
        st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
        st.text_area("System", key="system_prompt", height=80)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session = SessionManager.create_session(user['id'])
        st.rerun()

def render_chat_interface(client):
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
            msgs = [{"role":"system","content":st.session_state.system_prompt}] + [{"role":m["role"],"content":m["content"]} for m in st.session_state.messages[-6:]]
            try:
                for chunk in client.chat_stream(st.session_state.model, msgs, {"temperature":st.session_state.temperature}):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
            except Exception as e:
                full = f"⚠️ Error: {str(e)}"
                placeholder.error(full)
        st.session_state.messages.append({"role":"assistant","content":full})

def main():
    st.set_page_config(page_title="Ollama Pro", layout="wide")
    SessionManager.init_session()
    # FIXED: Changed default from llama3.1:8b to llama3.1
    defaults = {'user':None,'messages':[],'model':'llama3.1','temperature':0.7,'top_p':0.9,'num_ctx':2048,'system_prompt':'You are helpful','rag_enabled':False,'vision_enabled':True,'tts_enabled':False,'current_session':None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

    ollama_host = st.secrets.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "https://ollama.com"))
    client = OllamaClient(ollama_host)

    if not st.session_state.user:
        # simplified login for brevity - use your full login_screen if needed
        st.title("Login")
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Login"):
            user = AuthManager.authenticate(u,p)
            if user:
                SessionManager.persist_login(user)
                st.rerun()
        return

    with st.sidebar:
        render_sidebar(client)
    render_chat_interface(client)

if __name__ == "__main__":
    main()
