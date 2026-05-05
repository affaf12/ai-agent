"""
vector_db.py
============
FAISS vector store with HNSW + metadata filtering + atomic persistence.

Highlights
----------
* Backends: HNSW (default, scalable, log-time search) or Flat (exact, small
  corpora). Choose at construction; both share the same on-disk format.
* Stable string IDs preserved across save/load via a parallel id <-> rowid map.
* Metadata filter callable so callers can search "where source == 'docs'" etc.
* Upserts and deletes implemented via tombstoning + lazy compaction so the
  index stays correct without full rebuilds on every change.
* Atomic, crash-safe persistence: writes go to a temp file then os.replace.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Sequence

import faiss
import numpy as np


_MetaFilter = Callable[[Dict], bool]


class VectorDB:
    def __init__(
        self,
        dim: int = 768,
        index_path: str = "data/faiss.index",
        backend: str = "hnsw",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = index_path + ".meta"
        self.backend = backend
        self._lock = threading.RLock()

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

        if os.path.exists(index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(index_path)
            with open(self.meta_path, "rb") as f:
                state = pickle.load(f)
            self.metadata: List[Dict] = state["metadata"]
            self.id_to_row: Dict[str, int] = state["id_to_row"]
            self.tombstones: set = set(state.get("tombstones", []))
            # Rehydrate ef_search for HNSW.
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = ef_search
        else:
            if backend == "hnsw":
                idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = ef_construction
                idx.hnsw.efSearch = ef_search
                self.index = idx
            elif backend == "flat":
                self.index = faiss.IndexFlatIP(dim)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            self.metadata = []
            self.id_to_row = {}
            self.tombstones = set()

    # ------------------------------------------------------------------ size

    def __len__(self) -> int:
        with self._lock:
            return len(self.metadata) - len(self.tombstones)

    # ----------------------------------------------------------------- write

    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict],
    ) -> None:
        """Insert or upsert by metadata['id']."""
        if not embeddings:
            return
        with self._lock:
            vecs = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(vecs)

            new_vecs: List[np.ndarray] = []
            for v, m in zip(vecs, metadatas):
                key = m.get("id")
                if key and key in self.id_to_row:
                    # Upsert: tombstone the old row, append the new vector.
                    self.tombstones.add(self.id_to_row[key])
                row = len(self.metadata)
                self.metadata.append(dict(m))
                if key:
                    self.id_to_row[key] = row
                new_vecs.append(v)

            self.index.add(np.vstack(new_vecs))
            self._maybe_compact()
            self.save()

    def delete(self, ids: Sequence[str]) -> int:
        """Soft-delete rows by id. Returns number tombstoned."""
        with self._lock:
            n = 0
            for key in ids:
                row = self.id_to_row.pop(key, None)
                if row is not None:
                    self.tombstones.add(row)
                    n += 1
            if n:
                self._maybe_compact()
                self.save()
            return n

    # ----------------------------------------------------------------- read

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        meta_filter: Optional[_MetaFilter] = None,
    ) -> List[Dict]:
        with self._lock:
            if len(self.metadata) == 0:
                return []
            q = np.asarray([query_embedding], dtype=np.float32)
            faiss.normalize_L2(q)
            # Over-fetch when we have tombstones or filters so we still return top_k.
            fetch = top_k * 4 if (self.tombstones or meta_filter) else top_k
            fetch = min(fetch, len(self.metadata))
            scores, idxs = self.index.search(q, fetch)
            results: List[Dict] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1 or idx in self.tombstones:
                    continue
                meta = self.metadata[idx]
                if meta_filter and not meta_filter(meta):
                    continue
                results.append({**meta, "score": float(score)})
                if len(results) >= top_k:
                    break
            return results

    # ----------------------------------------------------------- maintenance

    def _maybe_compact(self) -> None:
        # Compact when more than 30% of rows are tombstoned.
        total = len(self.metadata)
        if total < 1000 or len(self.tombstones) / total < 0.3:
            return
        live_meta: List[Dict] = []
        live_vecs: List[np.ndarray] = []
        # Reconstruct vectors from the existing index where supported.
        try:
            for row, meta in enumerate(self.metadata):
                if row in self.tombstones:
                    continue
                live_vecs.append(self.index.reconstruct(row))
                live_meta.append(meta)
        except Exception:
            return  # backend doesn't support reconstruct; skip compaction

        if self.backend == "hnsw":
            new_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = 200
            new_index.hnsw.efSearch = 64
        else:
            new_index = faiss.IndexFlatIP(self.dim)
        new_index.add(np.vstack(live_vecs).astype(np.float32))

        self.index = new_index
        self.metadata = live_meta
        self.tombstones = set()
        self.id_to_row = {
            m["id"]: i for i, m in enumerate(live_meta) if m.get("id")
        }

    # ------------------------------------------------------------ persistence

    def save(self) -> None:
        with self._lock:
            tmp_idx = self.index_path + ".tmp"
            tmp_meta = self.meta_path + ".tmp"
            faiss.write_index(self.index, tmp_idx)
            with open(tmp_meta, "wb") as f:
                pickle.dump(
                    {
                        "metadata": self.metadata,
                        "id_to_row": self.id_to_row,
                        "tombstones": list(self.tombstones),
                        "dim": self.dim,
                        "backend": self.backend,
                    },
                    f,
                )
            os.replace(tmp_idx, self.index_path)
            os.replace(tmp_meta, self.meta_path)

    def reset(self) -> None:
        """Wipe the index entirely (use with care)."""
        with self._lock:
            for p in (self.index_path, self.meta_path):
                if os.path.exists(p):
                    os.remove(p)
            self.__init__(
                dim=self.dim,
                index_path=self.index_path,
                backend=self.backend,
            )
