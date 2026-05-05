"""
retriever.py
============
Hybrid retrieval with Reciprocal Rank Fusion, MMR diversification and an
optional cross-encoder reranker.

Pipeline
--------
    query
      |
      |-- dense (FAISS, cosine)         --\\
      |                                    >--> Reciprocal Rank Fusion
      |-- sparse (BM25Okapi)            --/
      |
      v
    candidates --> MMR (relevance + diversity) --> cross-encoder rerank --> top_k

Why each stage
--------------
* Dense catches paraphrases ("monetary policy" ~ "interest rates").
* Sparse catches rare/exact terms (model numbers, names, code symbols).
* RRF fuses two ranked lists without needing score calibration.
* MMR drops near-duplicates so the LLM gets diverse evidence, not 8 copies of
  the same paragraph.
* Cross-encoder reranking (optional) gives a real query-document score
  instead of approximating with two independently-trained encoders.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# BM25 (lazy import, falls back to a tiny pure-Python implementation)
# ---------------------------------------------------------------------------


def _build_bm25(tokenized_corpus: List[List[str]]):
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:  # pragma: no cover - fallback path
        return _MiniBM25(tokenized_corpus)


class _MiniBM25:
    """Minimal Okapi BM25 fallback so the system still works without rank_bm25."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(corpus) else 0.0
        self.df: Dict[str, int] = {}
        for doc in corpus:
            for tok in set(doc):
                self.df[tok] = self.df.get(tok, 0) + 1
        n = len(corpus)
        self.idf = {
            t: float(np.log((n - df + 0.5) / (df + 0.5) + 1.0))
            for t, df in self.df.items()
        }

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        for i, doc in enumerate(self.corpus):
            tf: Dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            denom_norm = 1.0 - self.b + self.b * (self.doc_len[i] / max(self.avgdl, 1.0))
            for q in query:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                f = tf[q]
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * denom_norm)
        return scores


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Optional second-stage reranker using a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Dense + BM25 + RRF + MMR + optional cross-encoder."""

    def __init__(
        self,
        vector_db,
        embedding_model,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
        mmr_lambda: float = 0.6,
    ):
        self.vdb = vector_db
        self.emb = embedding_model
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda

        self.bm25 = None
        self.corpus_texts: List[str] = []
        self.corpus_metas: List[Dict] = []
        self.corpus_embs: Optional[np.ndarray] = None  # for MMR

    # ------------------------------------------------------------------ index

    def index(self, chunks: List[Dict]) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.emb.embed_documents(texts)

        # Persist into the vector DB. We pass the chunk dict itself as metadata
        # so retrieval results round-trip with full provenance.
        self.vdb.add(embeddings, chunks)

        # Append (rather than replace) so multiple ingest() calls accumulate.
        self.corpus_texts.extend(texts)
        self.corpus_metas.extend(chunks)
        self.corpus_embs = (
            np.asarray(embeddings, dtype=np.float32)
            if self.corpus_embs is None
            else np.vstack([self.corpus_embs, np.asarray(embeddings, dtype=np.float32)])
        )
        self.bm25 = _build_bm25([_tokenize(t) for t in self.corpus_texts])

    # ----------------------------------------------------------------- query

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 30,
        meta_filter: Optional[Callable[[Dict], bool]] = None,
        use_mmr: bool = True,
        use_reranker: bool = True,
    ) -> List[Dict]:
        if not self.corpus_texts:
            return []

        # 1) Dense
        q_emb = np.asarray(self.emb.embed_query(query), dtype=np.float32)
        dense_hits = self.vdb.search(
            q_emb.tolist(), top_k=candidate_pool, meta_filter=meta_filter
        )

        # 2) Sparse (BM25)
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
        sparse_hits: List[Dict] = []
        for i in top_idx:
            if bm25_scores[i] <= 0:
                continue
            meta = dict(self.corpus_metas[i])
            if meta_filter and not meta_filter(meta):
                continue
            meta["score"] = float(bm25_scores[i])
            sparse_hits.append(meta)

        # 3) Reciprocal Rank Fusion
        fused = self._rrf([dense_hits, sparse_hits])

        # 4) MMR for diversity (operates on the candidate pool)
        candidates = fused[: max(top_k * 4, candidate_pool)]
        if use_mmr and len(candidates) > top_k:
            candidates = self._mmr(query, q_emb, candidates, top_k=top_k * 2)

        # 5) Cross-encoder rerank (optional)
        if use_reranker and self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Add rank field for downstream citations.
        for rank, c in enumerate(candidates, start=1):
            c["rank"] = rank
        return candidates

    # ------------------------------------------------------------- internals

    def _rrf(self, ranked_lists: List[List[Dict]]) -> List[Dict]:
        scores: Dict[str, float] = {}
        items: Dict[str, Dict] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                key = item.get("id") or f"_anon::{id(item)}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                # Keep the richest copy of the metadata we've seen.
                if key not in items:
                    items[key] = dict(item)
                else:
                    items[key].update({k: v for k, v in item.items() if k not in items[key]})
        ordered = sorted(items.values(), key=lambda it: scores[it.get("id") or f"_anon::{id(it)}"], reverse=True)
        for it in ordered:
            it["fusion_score"] = scores[it.get("id") or f"_anon::{id(it)}"]
        return ordered

    def _mmr(
        self,
        query: str,
        q_emb: np.ndarray,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if not candidates:
            return []
        # Pull embeddings for the candidates from the corpus matrix where we
        # have them, otherwise embed on the fly.
        id_to_row: Dict[str, int] = {m.get("id"): i for i, m in enumerate(self.corpus_metas)}
        cand_embs: List[np.ndarray] = []
        missing: List[int] = []
        for i, c in enumerate(candidates):
            row = id_to_row.get(c.get("id"))
            if row is not None and self.corpus_embs is not None:
                cand_embs.append(self.corpus_embs[row])
            else:
                cand_embs.append(None)  # type: ignore[arg-type]
                missing.append(i)
        if missing:
            new_embs = self.emb.embed_documents([candidates[i]["text"] for i in missing])
            for i, vec in zip(missing, new_embs):
                cand_embs[i] = np.asarray(vec, dtype=np.float32)
        cand_mat = np.vstack(cand_embs)

        # Cosine similarities (everything is L2-normalised already).
        sim_to_query = cand_mat @ q_emb
        sim_matrix = cand_mat @ cand_mat.T

        selected: List[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda i: sim_to_query[i])
            else:
                def _score(i: int) -> float:
                    relevance = float(sim_to_query[i])
                    diversity = float(max(sim_matrix[i, j] for j in selected))
                    return self.mmr_lambda * relevance - (1 - self.mmr_lambda) * diversity

                best = max(remaining, key=_score)
            selected.append(best)
            remaining.remove(best)

        out = [candidates[i] for i in selected]
        for c, i in zip(out, selected):
            c["mmr_relevance"] = float(sim_to_query[i])
        return out
