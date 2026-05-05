from core.ollama_client import OllamaClient
import os
client = OllamaClient(os.getenv("OLLAMA_HOST","http://localhost:11434"))
def get_agent(agent_type):
    """Get worker agent"""
    prompts = {
        "researcher": "You are researcher. Research thoroughly: ",
        "designer": "You are UI designer. Design: ",
        "coder": "You are coder. Write code for: ",
        "database": "You are DB expert. Create schema for: ",
        "tester": "You are QA tester. Test plan for: ",
        "manager": "You are project manager. Summarize: "
    }
    def agent_fn(task):
        prompt = prompts.get(agent_type, "") + task
        return client.chat(prompt, model="llama3")
    return agent_fn
