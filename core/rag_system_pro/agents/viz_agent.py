"""
VizAgent - Generate charts and visualizations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .llm import LLMClient

_SYSTEM_VIZ = """You are a data visualization expert. You create clear, effective charts.
- Choose the right chart type for the data
- Use matplotlib, plotly, or altair
- Include labels, titles, legends
- Make it publication-ready
- Return complete, runnable Python code"""

class VizAgent:
    """Generate data visualizations"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="code")
    
    def create_chart(self, data_description: str, chart_type: str = "auto", library: str = "matplotlib") -> Dict[str, Any]:
        """Generate chart code from description"""
        prompt = f"""Create a {library} visualization.

Data: {data_description}
Chart type: {chart_type}

Return complete Python code that:
1. Creates sample data (if not provided)
2. Generates the chart
3. Saves to 'chart.png'
4. Includes proper labels and styling

Return ONLY code in ```python block."""
        
        code = self.llm.chat_code(prompt, system=_SYSTEM_VIZ, language="python")
        
        return {
            "success": True,
            "code": code,
            "library": library,
            "chart_type": chart_type
        }
    
    def suggest_chart(self, data_sample: str) -> Dict[str, Any]:
        """Suggest best chart type for data"""
        prompt = f"What is the best chart type for this data?\n\n{data_sample}\n\nRecommend 3 options with reasons."
        suggestion = self.llm.chat(prompt, system=_SYSTEM_VIZ)
        return {"success": True, "suggestions": suggestion}
