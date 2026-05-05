"""
File Doctor - RAM ke hisab se auto model select
"""
import pandas as pd
import json
import re
import ast
from pathlib import Path
import streamlit as st
import psutil

class FileDoctor:
    def __init__(self, llm_client, model_name=None):
        self.llm = llm_client
        
        # Auto-select model based on RAM if not provided
        if model_name is None:
            self.model = self._select_model_by_ram()
        else:
            self.model = model_name
        
        # Show which model is being used
        try:
            st.toast(f"🧠 Using model: {self.model}", icon="🤖")
        except:
            pass
    
    def _select_model_by_ram(self) -> str:
        """RAM dekh ke best model choose karo"""
        try:
            # Get available RAM in GB
            ram_gb = psutil.virtual_memory().total / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            # Model mapping based on RAM
            # llama3.2:1b ~ 1.3GB, llama3.2:3b ~ 2GB, llama3:8b ~ 4.7GB, etc.
            if available_gb < 2:
                model = "llama3.2:1b"
            elif available_gb < 4:
                model = "llama3.2:3b"
            elif available_gb < 6:
                model = "llama3:8b"
            elif available_gb < 12:
                model = "llama3.1:8b"
            elif available_gb < 16:
                model = "mistral:7b"
            else:
                model = "llama3.1:70b"
            
            # Try to verify model exists, fallback if not
            try:
                # Check if model is available in Ollama
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                if model not in result.stdout:
                    # Fallback to smallest available
                    if 'llama3.2:1b' in result.stdout:
                        model = 'llama3.2:1b'
                    elif 'llama3.2' in result.stdout:
                        model = 'llama3.2'
                    elif 'llama3' in result.stdout:
                        model = 'llama3'
                    else:
                        # Use first available model
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header
                        if lines:
                            model = lines[0].split()[0]
            except:
                pass  # Use selected model anyway
            
            return model
            
        except Exception as e:
            # Fallback to safe default
            return "llama3.2:1b"
    
    def get_system_info(self) -> dict:
        """System RAM info dikhao"""
        try:
            vm = psutil.virtual_memory()
            return {
                "total_gb": round(vm.total / (1024**3), 1),
                "available_gb": round(vm.available / (1024**3), 1),
                "used_percent": vm.percent,
                "model_selected": self.model
            }
        except:
            return {"model_selected": self.model}

    def diagnose_excel(self, file_path: str) -> dict:
        """Excel file ko read karke problems dhoondo"""
        try:
            # Try different engines
            for engine in ['openpyxl', 'xlrd', None]:
                try:
                    df = pd.read_excel(file_path, header=None, engine=engine)
                    break
                except:
                    continue

            issues = []
            sample = df.head(15).to_string()

            # Check problems
            if df.isnull().all().all():
                issues.append("File completely empty")
            if df.shape[0] < 2:
                issues.append("Only 1 row found")
            if df.isnull().sum().sum() > df.size * 0.5:
                issues.append("More than 50% empty cells")

            return {
                "type": "excel",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "issues": issues,
                "sample": sample,
                "df": df
            }
        except Exception as e:
            return {"error": str(e)}

    def clean_excel(self, diagnosis: dict) -> tuple:
        """AI se Excel clean karwao - RAM optimized"""
        if "error" in diagnosis:
            return None, diagnosis["error"]

        df = diagnosis["df"]
        sample = diagnosis["sample"]

        # Shorter prompt for low RAM models
        is_small_model = any(x in self.model.lower() for x in ['1b', '3b', 'tiny'])
        
        if is_small_model:
            prompt = f"""Clean this Excel data. Return JSON only:
Data: {sample[:500]}
Issues: {', '.join(diagnosis['issues'][:2])}
JSON format: {{"has_header":true,"skip_rows":[],"drop_empty_rows":true}}"""
        else:
            prompt = f"""You are an Excel data cleaner. Analyze this messy Excel data and provide cleaning instructions.

DATA SAMPLE (first 15 rows):
{sample}

Current shape: {diagnosis['rows']} rows x {diagnosis['cols']} columns
Issues found: {', '.join(diagnosis['issues'])}

Provide JSON ONLY with cleaning steps:
{{
  "has_header": true/false,
  "header_row": 0,
  "skip_rows": [0,1],
  "drop_empty_rows": true,
  "drop_empty_cols": true,
  "fill_method": "none" or "forward" or "zero",
  "column_names": ["Col1", "Col2"] or null
}}

Return ONLY the JSON, no explanation."""

        try:
            response, _ = self.llm.chat_once(self.model, [{"role": "user", "content": prompt}], {"temperature": 0.1})
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                instructions = json.loads(json_match.group())

                # Apply cleaning
                if instructions.get("skip_rows"):
                    df = df.drop(index=instructions["skip_rows"], errors='ignore')

                if instructions.get("has_header") and "header_row" in instructions:
                    header_idx = instructions["header_row"]
                    if header_idx < len(df):
                        df.columns = df.iloc[header_idx]
                        df = df.drop(index=header_idx)

                if instructions.get("drop_empty_rows", True):
                    df = df.dropna(how='all')
                if instructions.get("drop_empty_cols", True):
                    df = df.dropna(axis=1, how='all')

                if instructions.get("column_names"):
                    cols = instructions["column_names"]
                    if len(cols) <= len(df.columns):
                        df.columns = cols + list(df.columns[len(cols):])

                # Reset index
                df = df.reset_index(drop=True)

                return df, f"✅ Excel cleaned with {self.model}!"
        except Exception as e:
            # Fallback: basic cleaning
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df = df.reset_index(drop=True)
            return df, f"⚠️ Basic cleaning done (AI: {str(e)[:40]})"

        return df, "Cleaned"

    def diagnose_python(self, file_path: str) -> dict:
        """Python file ko check karo"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        issues = []

        # Check syntax
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError as e:
            syntax_ok = False
            issues.append(f"Syntax error line {e.lineno}: {e.msg}")

        # Check common issues
        if '    ' in code and '\t' in code:
            issues.append("Mixed tabs and spaces")
        if code.count('\n\n\n') > 0:
            issues.append("Too many blank lines")
        if 'import *' in code:
            issues.append("Wildcard imports found")

        lines = code.split('\n')
        long_lines = [i+1 for i, l in enumerate(lines) if len(l) > 100]

        if long_lines:
            issues.append(f"{len(long_lines)} lines over 100 chars")

        return {
            "type": "python",
            "lines": len(lines),
            "issues": issues,
            "syntax_ok": syntax_ok,
            "code": code,
            "sample": '\n'.join(lines[:30])
        }

    def clean_python(self, diagnosis: dict) -> tuple:
        """AI se Python clean karwao - RAM optimized"""
        code = diagnosis["code"]
        
        # Check model size for prompt optimization
        is_small_model = any(x in self.model.lower() for x in ['1b', '3b', 'tiny', '1.5b'])
        
        if is_small_model:
            # Shorter prompt for small models
            prompt = f"""Fix Python code. Return clean code only:

Issues: {', '.join(diagnosis['issues'][:3])}
Code:
{code[:2000]}

Clean code:"""
        else:
            prompt = f"""Fix and format this Python code. Return ONLY the cleaned code, no explanations, no markdown.

ISSUES FOUND: {', '.join(diagnosis['issues'])}

CODE TO FIX:
{code}

REQUIREMENTS:
1. Fix indentation (use 4 spaces)
2. Remove extra blank lines (max 2)
3. Organize imports (stdlib, third-party, local)
4. Fix syntax errors if any
5. Follow PEP8
6. Keep all functionality

Return ONLY clean Python code:"""

        try:
            cleaned, _ = self.llm.chat_once(self.model, [{"role": "user", "content": prompt}], {"temperature": 0.1})

            # Remove markdown
            cleaned = re.sub(r'^```python\s*', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            # Validate
            try:
                ast.parse(cleaned)
                return cleaned, f"✅ Python cleaned with {self.model}!"
            except:
                return cleaned, f"⚠️ Cleaned with {self.model} (check syntax)"
        except Exception as e:
            # Fallback: basic formatting without AI
            try:
                # Simple cleanup
                lines = code.split('\n')
                cleaned_lines = []
                blank_count = 0
                
                for line in lines:
                    # Remove trailing whitespace
                    line = line.rstrip()
                    
                    # Limit blank lines to 2
                    if not line.strip():
                        blank_count += 1
                        if blank_count <= 2:
                            cleaned_lines.append('')
                    else:
                        blank_count = 0
                        cleaned_lines.append(line)
                
                cleaned = '\n'.join(cleaned_lines)
                return cleaned, f"⚠️ Basic cleanup (AI failed: {self.model} not available)"
            except:
                return code, f"❌ Failed: {str(e)[:50]}"
