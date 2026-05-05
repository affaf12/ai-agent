"""
EmailAgent - Draft professional emails
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from .llm import LLMClient

_SYSTEM_EMAIL = """You are an executive assistant who writes excellent emails.
- Clear subject lines
- Professional but warm tone
- Get to the point quickly
- Clear call-to-action
- Proper greeting and sign-off
- No fluff"""

class EmailAgent:
    """Draft and improve emails"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="doc")
    
    def draft(self, purpose: str, recipient: str = "", context: str = "", tone: str = "professional") -> Dict[str, Any]:
        """Draft an email"""
        prompt = f"""Draft an email.

Purpose: {purpose}
Recipient: {recipient or 'Not specified'}
Context: {context or 'None'}
Tone: {tone}

Format:
Subject: [clear subject]

[Email body]

Provide complete email."""
        
        email = self.llm.chat(prompt, system=_SYSTEM_EMAIL)
        
        return {
            "success": True,
            "email": email,
            "purpose": purpose,
            "tone": tone
        }
    
    def reply(self, original_email: str, reply_points: str, tone: str = "professional") -> Dict[str, Any]:
        """Draft a reply"""
        prompt = f"""Draft a reply to this email:

Original:
{original_email}

My points to cover:
{reply_points}

Tone: {tone}

Provide complete reply."""
        
        reply = self.llm.chat(prompt, system=_SYSTEM_EMAIL)
        return {"success": True, "reply": reply}
    
    def improve(self, draft: str, goal: str = "more professional") -> Dict[str, Any]:
        """Improve existing email"""
        prompt = f"Improve this email to be {goal}:\n\n{draft}\n\nReturn improved version only."
        improved = self.llm.chat(prompt, system=_SYSTEM_EMAIL)
        return {"success": True, "original": draft, "improved": improved}
