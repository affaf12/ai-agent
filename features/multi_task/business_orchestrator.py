"""
business_orchestrator.py
-------------------------
The brain of the multi-agent system.
Runs all role-based agents, collects their outputs,
and passes them to the report builder.

Place this in: /features/multi_task/business_orchestrator.py
"""

import concurrent.futures
from typing import Optional

# Import all agents
from core.rag_system_pro.agents.ceo_agent   import CEOAgent
from core.rag_system_pro.agents.cfo_agent   import CFOAgent
from core.rag_system_pro.agents.coo_agent   import COOAgent
from core.rag_system_pro.agents.cto_agent   import CTOAgent
from core.rag_system_pro.agents.hr_agent    import HRAgent
from core.rag_system_pro.agents.sales_agent import SalesAgent
from core.rag_system_pro.agents.pm_agent    import PMAgent
from core.report_builder                    import ReportBuilder


class BusinessOrchestrator:
    """
    Coordinator / Manager Agent.
    Decides which agents run, collects outputs, builds final report.
    """

    # Default set of agents to run (you can customize per use case)
    DEFAULT_AGENTS = ["ceo", "cfo", "coo", "cto", "hr", "sales", "pm"]

    def __init__(
        self,
        ollama_client,
        model: str = "llama3",
        parallel: bool = False,
        agents_to_run: Optional[list] = None,
    ):
        """
        Args:
            ollama_client : Your existing OllamaClient instance
            model         : Ollama model name
            parallel      : Run all agents at the same time (faster) or one by one
            agents_to_run : Which agents to include. None = all defaults.
        """
        self.client   = ollama_client
        self.model    = model
        self.parallel = parallel
        self.selected = agents_to_run or self.DEFAULT_AGENTS

        # Instantiate all agents
        self._agent_map = {
            "ceo":   CEOAgent(ollama_client, model),
            "cfo":   CFOAgent(ollama_client, model),
            "coo":   COOAgent(ollama_client, model),
            "cto":   CTOAgent(ollama_client, model),
            "hr":    HRAgent(ollama_client, model),
            "sales": SalesAgent(ollama_client, model),
            "pm":    PMAgent(ollama_client, model),
        }

        self.report_builder = ReportBuilder()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run(self, context: str, file_name: str = "Uploaded Document") -> dict:
        """
        Main entry point.

        Args:
            context   : Extracted text from the uploaded file
            file_name : Original file name (for the report header)

        Returns:
            {
                "agent_outputs": [...],   # list of each agent's result dict
                "report_markdown": "...", # full combined markdown report
                "report_html":     "...", # HTML version (optional)
                "success":         True,
            }
        """
        print(f"[Orchestrator] Starting analysis of: {file_name}")
        print(f"[Orchestrator] Running agents: {self.selected}")

        agents = [self._agent_map[k] for k in self.selected if k in self._agent_map]

        if self.parallel:
            outputs = self._run_parallel(agents, context)
        else:
            outputs = self._run_sequential(agents, context)

        print(f"[Orchestrator] All agents done. Building report...")

        report_md   = self.report_builder.build_markdown(file_name, outputs)
        report_html = self.report_builder.build_html(file_name, outputs)

        return {
            "agent_outputs":   outputs,
            "report_markdown": report_md,
            "report_html":     report_html,
            "success":         True,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _run_parallel(self, agents: list, context: str) -> list:
        """Run all agents at the same time using thread pool."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {
                executor.submit(agent.analyze, context): agent
                for agent in agents
            }
            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                try:
                    result = future.result()
                    print(f"  ✅ {agent.ROLE} done")
                except Exception as e:
                    result = {
                        "role":     agent.ROLE,
                        "emoji":    agent.EMOJI,
                        "analysis": f"[Failed: {e}]",
                        "success":  False,
                    }
                    print(f"  ❌ {agent.ROLE} failed: {e}")
                results.append(result)

        # Sort back to original agent order
        order = [a.ROLE for a in agents]
        results.sort(key=lambda r: order.index(r["role"]) if r["role"] in order else 99)
        return results

    def _run_sequential(self, agents: list, context: str) -> list:
        """Run agents one by one (slower but safer for resource-limited machines)."""
        results = []
        for agent in agents:
            print(f"  ▶ Running {agent.ROLE}...")
            result = agent.analyze(context)
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {agent.ROLE} done")
            results.append(result)
        return results
