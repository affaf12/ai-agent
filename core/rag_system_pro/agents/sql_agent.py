"""
SQLAgent - Generate, optimize, and explain SQL queries
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from .llm import LLMClient


_SYSTEM_SQL = """You are a senior database engineer expert in SQL.
You write correct, efficient, and secure SQL for PostgreSQL, MySQL, SQLite, and SQL Server.
- Always use parameterized queries
- Prefer explicit JOINs
- Add comments for complex logic
- Consider indexes and performance
- Return ONLY SQL in fenced code block unless asked for explanation."""

class SQLAgent:
    """Generate and optimize SQL queries"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(task_type="data")
        self.dialect = "postgresql"
    
    def set_dialect(self, dialect: str):
        self.dialect = dialect.lower()
    
    def generate(self, description: str, schema: str = "", dialect: Optional[str] = None) -> Dict[str, Any]:
        dialect = dialect or self.dialect
        prompt = f"Generate {dialect} SQL for:\n\n{description}\n\n{f'Schema:\n{schema}' if schema else ''}\n\nReturn ONLY SQL in ```sql block."
        sql = self.llm.chat_code(prompt, system=_SYSTEM_SQL, language="sql")
        return {"success": True, "sql": sql, "dialect": dialect}
    
    def explain(self, sql: str) -> Dict[str, Any]:
        prompt = f"Explain this SQL:\n```sql\n{sql}\n```\n\nProvide: 1) What it does 2) Tables 3) Operations 4) Performance issues"
        explanation = self.llm.chat(prompt, system=_SYSTEM_SQL)
        return {"success": True, "explanation": explanation}
    
    def optimize(self, sql: str, schema: str = "") -> Dict[str, Any]:
        prompt = f"Optimize this SQL:\n```sql\n{sql}\n```\n{f'Schema: {schema}' if schema else ''}\n\nReturn optimized SQL and changes."
        result = self.llm.chat(prompt, system=_SYSTEM_SQL)
        optimized = self.llm._strip_code_fence(result, "sql")
        return {"success": True, "original": sql, "optimized": optimized, "explanation": result}
    
    def fix_error(self, sql: str, error: str) -> Dict[str, Any]:
        prompt = f"Fix SQL error:\nSQL:\n```sql\n{sql}\n```\nError: {error}\n\nReturn corrected SQL."
        fixed = self.llm.chat_code(prompt, system=_SYSTEM_SQL, language="sql")
        return {"success": True, "fixed_sql": fixed}
