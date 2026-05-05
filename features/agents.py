"""
features/agents.py
Production-grade multi-agent orchestrator for Ollama-based workflows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    model: str = "llama3.1:8b"
    temperature: float = 0.7

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")

    def to_dict(self):
        return asdict(self)


class AgentOrchestrator:
    AVAILABLE_TOOLS = {
        "search": "Web search",
        "code_interpreter": "Execute Python",
        "file_reader": "Read files",
        "sql_query": "SQL",
        "calculator": "Math",
    }

    def __init__(self, client):
        self.client = client
        self.agents = {}

    def _stream(self, model, messages, options):
        if hasattr(self.client, "chat_stream"):
            return self.client.chat_stream(model, messages, options)
        if hasattr(self.client, "chat"):
            return self.client.chat(model=model, messages=messages, stream=True, options=options)
        raise AttributeError("Client needs chat_stream or chat")

    def create_agent(self, name, system_prompt, tools, model="llama3.1:8b", temperature=0.7):
        agent = Agent(name, system_prompt.strip(), tools, model, temperature)
        self.agents[name] = agent
        return agent

    def execute_workflow(self, task, agents, context=None):
        context = dict(context or {})
        for agent_name in agents:
            agent = self.agents.get(agent_name)
            if not agent:
                yield "\n[Agent '" + agent_name + "' not found]\n"
                continue
            yield "\n---\n**🤖 " + agent.name + "** is working...\n"
            prompt = agent.system_prompt + "\n\nTask: " + task + "\n\n"
            if context:
                prompt += "Context:\n" + json.dumps({k: str(v)[:1000] for k,v in context.items()}, indent=2) + "\n\n"
            messages = [{"role": "user", "content": prompt}]
            resp = ""
            try:
                for chunk in self._stream(agent.model, messages, {"temperature": agent.temperature}):
                    resp += str(chunk)
                    yield str(chunk)
            except Exception as e:
                yield "\n[Error: " + str(e) + "]\n"
            context[agent_name] = resp
        yield "\n---\n**Workflow Complete** ✅\n"
