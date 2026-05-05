"""
/core/rag_system_pro/agents/__init__.py
---------------------------------------
Registers all agents so they can be imported cleanly.

Usage:
    from core.rag_system_pro.agents import CEOAgent, CFOAgent, ...
    from core.rag_system_pro.agents import ALL_AGENTS
"""

from .base_agent    import BaseAgent
from .ceo_agent     import CEOAgent
from .cfo_agent     import CFOAgent
from .coo_agent     import COOAgent
from .cto_agent     import CTOAgent
from .hr_agent      import HRAgent
from .sales_agent   import SalesAgent
from .pm_agent      import PMAgent

# Convenience: all business agents in one list
ALL_BUSINESS_AGENTS = [
    CEOAgent,
    CFOAgent,
    COOAgent,
    CTOAgent,
    HRAgent,
    SalesAgent,
    PMAgent,
]

# Keep your existing agents (add their imports back here if needed)
# from .analytics_agent      import AnalyticsAgent
# from .api_agent            import ApiAgent
# from .code_agent           import CodeAgent
# from .doc_writer_agent     import DocWriterAgent
# from .email_agent          import EmailAgent
# from .excel_agent          import ExcelAgent
# from .image_agent          import ImageAgent
# from .pdf_agent            import PDFAgent
# from .sql_agent            import SQLAgent
# from .translate_agent      import TranslateAgent
# from .viz_agent            import VizAgent
# from .web_research_agent   import WebResearchAgent

__all__ = [
    "BaseAgent",
    "CEOAgent",
    "CFOAgent",
    "COOAgent",
    "CTOAgent",
    "HRAgent",
    "SalesAgent",
    "PMAgent",
    "ALL_BUSINESS_AGENTS",
]
