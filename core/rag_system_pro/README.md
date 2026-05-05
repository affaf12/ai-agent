# rag_system_pro

A pro-level, drop-in upgrade for your local RAG stack. Same package layout as
your originals, same public function names where it matters — just much
smarter under the hood.

## What's new vs. the old `rag_system`

| Area | Old | New |
|---|---|---|
| Chunking | Fixed-size + sentence break | Recursive + token-budgeted + code-fence aware + dedup + rich metadata |
| Embeddings | Single Ollama call per text | Batched, retried, **disk-cached** (sqlite), pluggable providers (Ollama / SentenceTransformers / hash) |
| Vector DB | `IndexFlatIP` only | **HNSW** (default) or Flat, **upserts**, **deletes**, **metadata filters**, atomic crash-safe persistence, lazy compaction |
| Retrieval | Linear blend of dense + BM25 | **Reciprocal Rank Fusion** + **MMR diversity** + optional **cross-encoder reranking** |
| Querying | Single shot, no rewrites | **Query rewriting** (multi-query) + **conversation memory with rolling summarisation** + **streaming** |
| Prompting | Basic citation hint | Strict citation-aware prompt + `"I don't know"` guardrail + **confidence scoring** |
| Output | `dict` blob | Typed `RAGResponse` (answer, sources, citations, confidence, latency) |
| Ops | None | Query cache, embedding cache, structured logs, atomic saves |

## Install

```bash
pip install -r requirements.txt
# Plus, locally:
ollama pull nomic-embed-text
ollama pull llama3.1
```

`rank_bm25`, `tiktoken` and `sentence-transformers` are optional — the package
falls back to built-in implementations if any are missing.

## Quick start

```python
from rag_system_pro import RAGSystem

rag = RAGSystem(llm_model="llama3.1", embed_model="nomic-embed-text")

rag.ingest([
    {"id": "doc1", "text": open("docs/handbook.md").read(), "metadata": {"source": "handbook"}},
    {"id": "doc2", "text": open("docs/faq.md").read(),      "metadata": {"source": "faq"}},
])

resp = rag.query("How do I rotate my API keys?")
print(resp.answer)
print("Sources:", [s["id"] for s in resp.sources])
print("Citations:", resp.citations)
print("Confidence:", resp.confidence)
```

## Streaming

```python
for token in rag.stream_query("Summarise the security policy"):
    print(token, end="", flush=True)
```

## Metadata filters

```python
resp = rag.query(
    "What changed in v7.7?",
    meta_filter=lambda m: m.get("source") == "changelog",
)
```

## Cross-encoder reranking (best quality, slower)

```python
rag = RAGSystem(use_reranker=True)
```

## Files

- `chunking.py` — `Chunker`, `chunk_text`, `chunk_documents`
- `embedding.py` — `EmbeddingModel`, `EmbeddingCache`
- `vector_db.py` — `VectorDB` (HNSW / Flat, upserts, deletes, filters)
- `retriever.py` — `HybridRetriever`, `CrossEncoderReranker`
- `rag.py` — `RAGSystem`, `RAGResponse`, `ConversationMemory`
- `agents/excel_agent.py` — upload-clean-export Excel/CSV with an LLM-planned pipeline
- `agents/code_agent.py` — explain / review / refactor / fix / document / test / translate / optimize / generate / run
- `agents/sandbox.py` — restricted Python sandbox shared by both agents
- `agents/llm.py` — small Ollama wrapper used by the agents
- `streamlit_demo.py` — ready-to-run Streamlit UI tying RAG + Excel + Code together

## Agents — quick examples

```python
from rag_system_pro.agents import ExcelAgent, CodeAgent

# Clean a messy spreadsheet
excel = ExcelAgent()
result = excel.clean(
    "uploads/sales_q3.xlsx",
    instruction="snake_case headers, fix dates, drop blank rows, fill missing amounts with 0",
)
print(result.cleaned_path)   # data/agent_work/sales_q3.cleaned.xlsx
print(result.report_md)      # markdown change-log

# Do anything with code
code = CodeAgent()
print(code.explain(open("buggy.py").read()).output)
print(code.fix(open("buggy.py").read(), problem="recursion depth on long inputs").output)
print(code.translate(open("script.py").read(), target_language="typescript").output)
print(code.test(open("module.py").read(), framework="pytest").output)
```

## Demo UI

```bash
pip install streamlit pandas openpyxl
streamlit run rag_system_pro/streamlit_demo.py
```

Three tabs: knowledge chat, Excel cleaner, code agent.
