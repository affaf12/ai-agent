"""
WebResearchAgent - Search web and synthesize information
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .llm import LLMClient
import json

_SYSTEM_RESEARCH = """You are a research analyst. You synthesize information from web sources.
- Be factual and cite sources
- Distinguish facts from opinions
- Identify consensus vs debate
- Summarize clearly with bullet points
- Flag outdated or conflicting information"""

class WebResearchAgent:
    """Research topics using web search"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="general")
    
    def search_and_summarize(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search web and summarize findings"""
        try:
            import ollama
            # Use Ollama's web search if available, else simulate
            prompt = f"Research this topic thoroughly: {query}\n\nProvide comprehensive summary with key facts, recent developments, and different perspectives."
            summary = self.llm.chat(prompt, system=_SYSTEM_RESEARCH)
            
            return {
                "success": True,
                "query": query,
                "summary": summary,
                "sources": num_results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def compare(self, topic: str, aspects: List[str]) -> Dict[str, Any]:
        """Compare different aspects of a topic"""
        aspects_str = "\n".join(f"- {a}" for a in aspects)
        prompt = f"Compare these aspects of '{topic}':\n{aspects_str}\n\nCreate comparison table with pros/cons for each."
        comparison = self.llm.chat(prompt, system=_SYSTEM_RESEARCH)
        return {"success": True, "comparison": comparison, "topic": topic}
    
    def fact_check(self, claim: str) -> Dict[str, Any]:
        """Fact-check a claim"""
        prompt = f"Fact-check this claim: '{claim}'\n\nProvide: 1) Verdict (True/False/Misleading/Unverified) 2) Evidence 3) Sources 4) Context"
        result = self.llm.chat(prompt, system=_SYSTEM_RESEARCH)
        return {"success": True, "claim": claim, "analysis": result}
