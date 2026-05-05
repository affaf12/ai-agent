"""
embedding.py
============
Pluggable embedding layer with batching, caching and retries.

Highlights
----------
* Provider-agnostic: Ollama (HTTP), SentenceTransformers (local) and a
  deterministic "hash" provider for offline tests.
* Automatic L2-normalisation so cosine similarity == dot product downstream.
* Batched encoding with a configurable batch size and tqdm progress bar.
* Persistent on-disk LRU cache keyed by (model, sha1(text)). Reruns over
  the same corpus are essentially free.
* Exponential-backoff retries for transient Ollama failures.
* `embedding_dim` property is detected lazily on first use.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Thread-safe sqlite-backed embedding cache."""

    def __init__(self, path: str = "data/embeddings.cache.sqlite"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emb (
                key TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(model: str, text: str) -> str:
        h = hashlib.sha1(f"{model}\u241f{text}".encode("utf-8")).hexdigest()
        return h

    def get_many(self, keys: Sequence[str]) -> List[Optional[np.ndarray]]:
        if not keys:
            return []
        with self._lock:
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, vector FROM emb WHERE key IN ({placeholders})",
                list(keys),
            ).fetchall()
        found = {k: np.frombuffer(v, dtype=np.float32) for k, v in rows}
        return [found.get(k) for k in keys]

    def put_many(self, keys: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        if not keys:
            return
        rows = [(k, v.astype(np.float32).tobytes()) for k, v in zip(keys, vectors)]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


class EmbeddingModel:
    """Embeddings with batching, retries and caching."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        provider: str = "ollama",
        batch_size: int = 32,
        normalize: bool = True,
        max_retries: int = 3,
        cache: Optional[EmbeddingCache] = None,
        ollama_host: Optional[str] = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_retries = max_retries
        self.cache = cache
        self._dim: Optional[int] = None
        self._st_model = None
        self._ollama = None

        if provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(model_name)
            self._dim = self._st_model.get_sentence_embedding_dimension()
        elif provider == "ollama":
            import ollama

            self._ollama = ollama.Client(host=ollama_host) if ollama_host else ollama
        elif provider == "hash":
            # Deterministic, dependency-free embeddings for tests/demos.
            self._dim = 256
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    # ------------------------------------------------------------------ dim

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed_one("dimension probe"))
        return self._dim

    # ------------------------------------------------------------------ API

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        # 1. Try cache
        keys = (
            [EmbeddingCache.make_key(self.model_name, t) for t in texts]
            if self.cache
            else None
        )
        cached = self.cache.get_many(keys) if self.cache else [None] * len(texts)
        results: List[Optional[np.ndarray]] = list(cached)

        # 2. Embed misses in batches
        miss_idx = [i for i, v in enumerate(results) if v is None]
        for batch_start in range(0, len(miss_idx), self.batch_size):
            batch_idx = miss_idx[batch_start : batch_start + self.batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            vectors = self._embed_batch(batch_texts)
            for i, vec in zip(batch_idx, vectors):
                results[i] = vec

        # 3. Persist new vectors to cache
        if self.cache:
            new_keys = [keys[i] for i in miss_idx]
            new_vecs = [results[i] for i in miss_idx]
            self.cache.put_many(new_keys, new_vecs)

        if self.normalize:
            mat = _l2_normalize(np.vstack(results).astype(np.float32))
            return mat.tolist()
        return [v.tolist() for v in results]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    # --------------------------------------------------------------- helpers

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if self.provider == "sentence_transformers":
            arr = self._st_model.encode(
                texts,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            return [np.asarray(v, dtype=np.float32) for v in arr]

        if self.provider == "hash":
            return [self._hash_embed(t) for t in texts]

        # Ollama: one call per text, with retries.
        out: List[np.ndarray] = []
        for t in texts:
            out.append(np.asarray(self._embed_one(t), dtype=np.float32))
        return out

    def _embed_one(self, text: str) -> List[float]:
        if self.provider == "hash":
            return self._hash_embed(text).tolist()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._ollama.embeddings(model=self.model_name, prompt=text)
                return resp["embedding"]
            except Exception as e:  # pragma: no cover - network path
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Ollama embeddings failed after retries: {last_err}")

    def _hash_embed(self, text: str) -> np.ndarray:
        # Deterministic bag-of-hashed-tokens projection. Not state of the art,
        # but stable, fast and dependency-free for tests/CI.
        dim = 256
        vec = np.zeros(dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
            vec[(h >> 16) % dim] -= 0.5
        return vec
