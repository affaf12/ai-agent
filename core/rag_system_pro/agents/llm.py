"""
Tiny shared Ollama wrapper used by all agents.
Adds retries, JSON-mode parsing, and RAM-aware model selection.
"""

from __future__ import annotations

import json
import re
import time
import os
from typing import Any, Dict, List, Optional


def get_system_ram():
    """Get total and available RAM in GB"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        return round(total_gb, 1), round(available_gb, 1)
    except:
        return 8.0, 4.0  # Default fallback


def get_optimal_model(task_type: str = "general") -> str:
    """
    Auto-select best model based on available RAM and task type.
    
    Task types:
    - general: chat, simple QA
    - code: programming, debugging
    - data: excel, sql, analysis
    - doc: writing, summarization
    - vision: image analysis
    """
    # Manual override via environment variable
    manual_model = os.environ.get("OLLAMA_MODEL")
    if manual_model:
        return manual_model
    
    total_ram, available_ram = get_system_ram()
    
    # Model selection by RAM and task
    if task_type == "vision":
        # Vision models are heavy
        if available_ram >= 8:
            return "moondream"  # 1.7GB, best for low RAM
        elif available_ram >= 6:
            return "moondream"
        else:
            return "moondream"  # Only option
    
    elif task_type == "code":
        # Code needs better models
        if available_ram >= 10:
            return "qwen2.5-coder:7b"
        elif available_ram >= 6:
            return "qwen2.5-coder:3b"
        elif available_ram >= 4:
            return "llama3.2:3b"
        else:
            return "llama3.2:1b"
    
    elif task_type == "data":
        # Data analysis needs reasoning
        if available_ram >= 8:
            return "llama3.2:3b"
        elif available_ram >= 4:
            return "llama3.2:3b"
        else:
            return "llama3.2:1b"
    
    elif task_type == "doc":
        # Document writing
        if available_ram >= 8:
            return "llama3.2:3b"
        elif available_ram >= 4:
            return "llama3.2:3b"
        else:
            return "llama3.2:1b"
    
    else:  # general
        if available_ram >= 6:
            return "llama3.2:3b"
        elif available_ram >= 3:
            return "llama3.2:1b"
        else:
            return "llama3.2:1b"


class LLMClient:
    def __init__(
        self,
        model: str = None,
        task_type: str = "general",
        host: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.2,
    ):
        import ollama

        # Auto-select model if not provided
        self.model = model or get_optimal_model(task_type)
        self.task_type = task_type
        self.temperature = temperature
        self.max_retries = max_retries
        self._ollama = ollama.Client(host=host) if host else ollama

    # ---------------------------------------------------------------- chat

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        
        # Handle images for vision models
        user_msg = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature if temperature is not None else self.temperature},
                )
                return resp["message"]["content"]
            except Exception as e:  # pragma: no cover - network path
                last_err = e
                # If model not found, try fallback
                if "not found" in str(e).lower() and attempt == 0:
                    print(f"Model {self.model} not found, trying fallback...")
                    self.model = "llama3.2:1b"
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"LLM call failed: {last_err}")

    # ------------------------------------------------------------- helpers

    def chat_json(self, prompt: str, system: Optional[str] = None) -> Any:
        """Ask the model for a JSON answer; tolerate fenced code blocks."""
        raw = self.chat(prompt, system=system, temperature=0.0)
        return self._parse_json(raw)

    def chat_code(self, prompt: str, system: Optional[str] = None, language: str = "python") -> str:
        """Ask the model for code; strip markdown fences if present."""
        raw = self.chat(prompt, system=system, temperature=0.1)
        return self._strip_code_fence(raw, language)

    @staticmethod
    def _parse_json(text: str) -> Any:
        # Try direct parse first.
        try:
            return json.loads(text)
        except Exception:
            pass
        # Look for a fenced JSON block.
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Last resort: find the first {...} or [...] block.
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"Could not parse JSON from model output: {text[:500]}")

    @staticmethod
    def _strip_code_fence(text: str, language: str) -> str:
        m = re.search(rf"```(?:{language})?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()


# Export for easy access
def get_model_info():
    """Get current model selection info"""
    total, available = get_system_ram()
    return {
        "total_ram_gb": total,
        "available_ram_gb": available,
        "general_model": get_optimal_model("general"),
        "code_model": get_optimal_model("code"),
        "data_model": get_optimal_model("data"),
        "doc_model": get_optimal_model("doc"),
        "vision_model": get_optimal_model("vision"),
    }
