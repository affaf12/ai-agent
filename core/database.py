import streamlit as st
import asyncio
import hashlib
import hmac
import json
import os
import re
import io
import zipfile
import uuid
import tempfile
import time
import base64
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Callable
from contextlib import contextmanager
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageOps
import requests


class DatabaseManager:
    """Thread-safe database operations with connection pooling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.db_path = Path("data/ollama_pro_enterprise.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialized = True
        self._init_db()
    
    @property
    def conn(self):
        """Thread-local connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema"""
        schema = """
        -- Users table with roles
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            salt BLOB NOT NULL,
            role TEXT DEFAULT 'user',
            api_key TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            settings TEXT DEFAULT '{}'
        );
        
        -- Sessions with encryption
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT,
            messages TEXT NOT NULL,  -- JSON
            model TEXT,
            settings TEXT,  -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_archived BOOLEAN DEFAULT 0,
            tags TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        
        -- Analytics & telemetry
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id INTEGER,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            latency_ms REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        
        -- RAG documents with vector metadata
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            content_hash TEXT UNIQUE,
            content TEXT,
            chunks TEXT,  -- JSON array of chunks
            metadata TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            usage_count INTEGER DEFAULT 0
        );
        
        -- Agent configurations
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            system_prompt TEXT,
            tools TEXT,  -- JSON
            model TEXT,
            temperature REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Collaboration: shared sessions
        CREATE TABLE IF NOT EXISTS shared_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            shared_by INTEGER,
            shared_with INTEGER,
            permissions TEXT,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_user ON analytics(user_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_time ON analytics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);
        """
        
        with self.conn:
            self.conn.executescript(schema)
        
        # Create default admin if not exists
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'"
        )
        if cursor.fetchone()[0] == 0:
            salt, pwd_hash = SecurityManager.hash_password("admin")
            self.conn.execute("""
                INSERT INTO users (username, password_hash, salt, role, api_key)
                VALUES (?, ?, ?, 'admin', ?)
            """, ("admin", pwd_hash, salt, SecurityManager.generate_api_key()))
            self.conn.commit()
    
    def close(self):
        """Close thread-local connection"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

# Global DB instance
db = DatabaseManager()


# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================


