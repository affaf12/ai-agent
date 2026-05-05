# core/intent.py
"""Intent detection for Ollama Pro - prevents RAG on greetings"""

GREETINGS = {"hello", "hi", "hey", "how are you", "good morning", "good evening", "sup", "yo", "what's up"}
CODING_SIGNALS = {"error", "fix", "bug", "install", "python", "javascript", "code", "traceback", "exception", "import", "module", "syntax", "typeerror", "keyerror"}

def is_greeting(q: str) -> bool:
    """Returns True for casual greetings - bypass RAG completely"""
    if not q:
        return False
    ql = q.lower().strip()
    return any(g in ql for g in GREETINGS) and len(ql.split()) <= 6

def should_use_rag(q: str) -> bool:
    """Returns True only for technical queries that need knowledge base"""
    if is_greeting(q):
        return False
    if len(q.split()) <= 2:
        return False
    ql = q.lower()
    return any(s in ql for s in CODING_SIGNALS) or "how to" in ql or "?" in ql
