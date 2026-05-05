"""
TranslateAgent - Translate text between languages
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from .llm import LLMClient

_SYSTEM_TRANSLATE = """You are a professional translator. 
- Preserve meaning, tone, and nuance
- Adapt idioms culturally
- Maintain formatting
- For code, translate comments only
- Return ONLY the translation, no explanations"""

class TranslateAgent:
    """Translate text between languages"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="general")
    
    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> Dict[str, Any]:
        """Translate text"""
        if source_lang == "auto":
            prompt = f"Translate to {target_lang}:\n\n{text}"
        else:
            prompt = f"Translate from {source_lang} to {target_lang}:\n\n{text}"
        
        translation = self.llm.chat(prompt, system=_SYSTEM_TRANSLATE, temperature=0.1)
        
        return {
            "success": True,
            "original": text,
            "translated": translation.strip(),
            "source": source_lang,
            "target": target_lang
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        prompt = f"What language is this? Respond with only the language name:\n\n{text[:200]}"
        lang = self.llm.chat(prompt, temperature=0.0)
        return {"success": True, "language": lang.strip(), "text": text[:100]}
    
    def translate_batch(self, texts: List[str], target_lang: str) -> Dict[str, Any]:
        """Translate multiple texts"""
        results = []
        for text in texts:
            result = self.translate(text, target_lang)
            results.append(result["translated"])
        return {"success": True, "translations": results, "count": len(results)}
