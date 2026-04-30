import subprocess, tempfile, time
class CodeExecutor:
    def run_python(self, code: str, timeout=8):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code); path = f.name
        try:
            r = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, r.stdout if r.returncode==0 else r.stderr
        except Exception as e:
            return False, str(e)
class DebugLoop:
    def __init__(self, rag_system, ollama_host="http://localhost:11434", model="llama3.1"):
        self.rag = rag_system; self.host = ollama_host; self.model = model; self.exec = CodeExecutor()
    def _ask(self, prompt):
        import requests
        r = requests.post(f"{self.host}/api/generate", json={"model":self.model,"prompt":prompt,"stream":False}, timeout=60)
        return r.json().get("response","")
    def fix(self, user_id, query):
        ctx = "\n\n".join([h.content for h in self.rag.retrieve(query, user_id, {"type":"error_fix"})[:3]])
        code = self._ask(f"Context:\n{ctx}\n\nFix this: {query}\nReturn ONLY python code:")
        for i in range(3):
            ok, out = self.exec.run_python(code)
            if ok: return {"success":True, "code":code, "output":out, "tries":i+1}
            code = self._ask(f"Code failed:\n{code}\nError:\n{out}\n\nFix using context:\n{ctx}\nReturn ONLY code:")
        return {"success":False, "code":code, "error":out}
