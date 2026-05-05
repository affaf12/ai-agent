"""
rag_system_pro.agents

Tool-using agents that go beyond Q&A.

* ExcelAgent       - inspect, clean and export Excel/CSV files
* CodeAgent        - explain, refactor, debug, test, translate any code
* DocWriterAgent   - draft reports, emails, proposals, blogs, READMEs from RAG context
* WebResearchAgent - autonomous web research with citations (no API keys needed)
* SQLAgent         - natural language to SQL, optimize, fix, and safely execute
* PythonSandbox    - safe(ish) Python execution sandbox they share

The agents talk to a local Ollama model. Recommended models:
    - deepseek-coder:6.7b   (code + data + SQL work)
    - llama3.1              (general reasoning, RAG, writing, research)
"""

from .sandbox import PythonSandbox, SandboxResult
from .llm import LLMClient

# Core agents
from .excel_agent import ExcelAgent, ExcelCleanResult
from .code_agent import CodeAgent, CodeTaskResult

# v3 agents
try:
    from .doc_writer_agent import DocWriterAgent
    _DOC_AVAILABLE = True
except ImportError:
    DocWriterAgent = None
    _DOC_AVAILABLE = False

try:
    from .web_research_agent import WebResearchAgent
    _WEB_AVAILABLE = True
except ImportError:
    WebResearchAgent = None
    _WEB_AVAILABLE = False

# v4 agent
try:
    from .sql_agent import SQLAgent, SQLResult
    _SQL_AVAILABLE = True
except ImportError:
    SQLAgent = None
    SQLResult = None
    _SQL_AVAILABLE = False

__all__ = [
    # Core
    "PythonSandbox",
    "SandboxResult",
    "LLMClient",
    "ExcelAgent",
    "ExcelCleanResult",
    "CodeAgent",
    "CodeTaskResult",
    # v3
    "DocWriterAgent",
    "WebResearchAgent",
    # v4
    "SQLAgent",
    "SQLResult",
]

# Convenience check
def available_agents():
    """Return dict of which agents are installed"""
    return {
        "excel": True,
        "code": True,
        "doc_writer": _DOC_AVAILABLE,
        "web_research": _WEB_AVAILABLE,
        "sql": _SQL_AVAILABLE,
    }