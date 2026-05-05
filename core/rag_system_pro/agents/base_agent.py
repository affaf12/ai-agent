"""
base_agent.py
-------------
Base class for all role-based agents (CEO, CFO, COO, etc.)
All agents inherit from this and only override their ROLE and SYSTEM_PROMPT.
"""

import json
from typing import Optional


class BaseAgent:
    """
    Every role agent inherits from this.
    It handles the Ollama API call, error handling, and output formatting.
    """

    ROLE: str = "Base Agent"
    EMOJI: str = "🤖"
    SYSTEM_PROMPT: str = "You are a helpful assistant."

    def __init__(self, ollama_client, model: str = "llama3"):
        """
        Args:
            ollama_client: Your existing OllamaClient instance from core/ollama_client.py
            model: The Ollama model to use (e.g. 'llama3', 'mistral')
        """
        self.client = ollama_client
        self.model = model

    def analyze(self, context: str, extra_instructions: str = "") -> dict:
        """
        Run this agent's analysis on the given context.

        Args:
            context: The extracted text/chunks from the uploaded file
            extra_instructions: Optional extra task instructions

        Returns:
            dict with keys: role, emoji, analysis, success
        """
        prompt = self._build_prompt(context, extra_instructions)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )

            # Extract text from response (adjust if your client returns differently)
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                content = str(response)

            return {
                "role":     self.ROLE,
                "emoji":    self.EMOJI,
                "analysis": content.strip(),
                "success":  True,
            }

        except Exception as e:
            return {
                "role":     self.ROLE,
                "emoji":    self.EMOJI,
                "analysis": f"[Error during {self.ROLE} analysis: {e}]",
                "success":  False,
            }

    def _build_prompt(self, context: str, extra_instructions: str = "") -> str:
        """Combine the document context with role-specific instructions."""
        base = f"""
You are acting as the {self.ROLE} of a company.

Below is a document / report / data that has been uploaded for review:

--- DOCUMENT CONTEXT START ---
{context}
--- DOCUMENT CONTEXT END ---

Analyze this document strictly from your {self.ROLE} perspective.
Be specific, use bullet points where helpful, and keep your analysis professional.
"""
        if extra_instructions:
            base += f"\nAdditional instructions: {extra_instructions}"

        return base.strip()

    def __repr__(self):
        return f"<{self.ROLE} Agent>"
