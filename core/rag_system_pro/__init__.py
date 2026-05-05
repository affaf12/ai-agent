"""
rag_system_pro
==============
A production-grade Retrieval-Augmented Generation (RAG) toolkit for local LLMs.

Pipeline:
    documents -> chunking -> embeddings -> vector_db (FAISS HNSW)
                              |
                              v
    query -> rewrite -> hybrid retrieval (dense + BM25 with RRF)
          -> MMR diversification -> cross-encoder rerank
          -> prompt assembly -> LLM (Ollama) -> answer + citations

Public API:
    RAGSystem            - end-to-end orchestrator with conversation memory
    EmbeddingModel       - pluggable embeddings (Ollama / SentenceTransformers)
    VectorDB             - FAISS index with HNSW/Flat backends + metadata filters
    HybridRetriever      - dense + BM25 fusion + MMR + optional cross-encoder
    Chunker              - recursive, sentence-aware, token-budgeted chunker
    chunk_documents      - convenience wrapper for batch chunking
"""

from .chunking import Chunker, chunk_text, chunk_documents
from .embedding import EmbeddingModel, EmbeddingCache
from .vector_db import VectorDB
from .retriever import HybridRetriever, CrossEncoderReranker
from .rag import RAGSystem, RAGResponse, ConversationMemory

__all__ = [
    "RAGSystem",
    "RAGResponse",
    "ConversationMemory",
    "EmbeddingModel",
    "EmbeddingCache",
    "VectorDB",
    "HybridRetriever",
    "CrossEncoderReranker",
    "Chunker",
    "chunk_text",
    "chunk_documents",
]

__version__ = "1.0.0"
