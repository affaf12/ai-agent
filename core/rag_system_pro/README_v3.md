# rag_system_pro

A pro-level, drop-in upgrade for your local RAG stack.  
Now with **five AI agents** — all running on `deepseek-coder:6.7b` (or any Ollama model).

---

## What's in the package

```
rag_system_pro/
├── agents/
│   ├── __init__.py
│   ├── llm_client.py          ← Ollama REST wrapper
│   ├── sandbox.py             ← safe Python executor
│   ├── excel_agent.py         ← Excel / CSV cleaner
│   ├── code_agent.py          ← 10 coding tasks, 20+ languages
│   ├── doc_writer_agent.py    ← 10 document types  ← NEW
│   └── web_research_agent.py  ← autonomous web research ← NEW
└── streamlit_demo.py          ← 5-tab UI (run with streamlit)
```

---

## Installation

```bash
pip install pandas openpyxl streamlit requests beautifulsoup4
# optional — enables .docx export from DocWriterAgent
pip install python-docx
```

Make sure Ollama is running and your model is pulled:

```bash
ollama pull deepseek-coder:6.7b
ollama serve
```

---

## Quick start — Streamlit UI

```bash
streamlit run core\rag_system_pro\streamlit_demo.py
```

Opens five tabs in your browser:

| Tab | What it does |
|-----|-------------|
| 💬 Knowledge Chat | RAG Q&A over your docs |
| 🧹 Excel Cleaner | Upload messy sheet → clean sheet + change log |
| 💻 Code Agent | Explain / review / refactor / fix / test / translate / run |
| 📄 Doc Writer | Draft any document from topic + optional context |
| 🔍 Web Research | Autonomous search → cited report |

---

## Agent API reference

### LLMClient

```python
from rag_system_pro.agents import LLMClient

llm = LLMClient(model="deepseek-coder:6.7b", temperature=0.2)
response = llm.ask("You are a helpful assistant.", "What is RAG?")
```

---

### ExcelAgent

```python
from rag_system_pro.agents import ExcelAgent

agent = ExcelAgent(llm=llm)
result = agent.clean("sales_data.xlsx",
                     instruction="snake_case headers, fix date columns, drop dupes")

print(result.cleaned_path)   # path to cleaned .xlsx
print(result.report_md)      # markdown change log
print(result.steps_run)      # list of steps applied
```

---

### CodeAgent

```python
from rag_system_pro.agents import CodeAgent

agent = CodeAgent(llm=llm)

# Explain
result = agent.explain(my_code)

# Fix a bug
result = agent.fix(buggy_code, problem="off-by-one error in the loop")

# Translate Python → TypeScript
result = agent.translate(py_code, target_language="typescript")

# Generate from spec
result = agent.generate("A REST API for a todo list", language="go")

# Run in sandbox
result = agent.run("import pandas as pd; print(pd.Series([1,2,3]).sum())")
print(result.stdout)

# All tasks: explain, review, refactor, fix, document, test,
#            translate, optimize, generate, run
```

Supported languages: python, javascript, typescript, go, rust, java, c, c++, c#,
ruby, php, swift, kotlin, scala, sql, html, css, yaml, json, bash.

---

### DocWriterAgent  ← NEW

```python
from rag_system_pro.agents import DocWriterAgent

agent = DocWriterAgent(llm=llm)

# Draft a report using RAG context
result = agent.report(
    topic="Q2 sales performance",
    context="<paste your RAG-retrieved chunks here>",
    tone="formal",
    word_limit=600,
)
print(result.markdown)
result.save("Q2_report.md")       # or .docx if python-docx is installed

# Other shorthands
agent.email(topic="Follow-up after client meeting", tone="friendly")
agent.proposal(topic="Cloud migration project", context=notes)
agent.summary(source_text=long_article)
agent.blog_post(topic="Why RAG beats fine-tuning")
agent.readme("A CLI tool that converts Markdown to PDF")
agent.cover_letter(job_spec="Senior ML Engineer at Acme", candidate_info=cv_text)
agent.meeting_notes(raw_notes=transcript)

# Generic — any document type you describe
agent.draft(
    topic="Internal audit findings",
    doc_type="custom",
    instructions="Format as a table of findings with severity and remediation columns",
    tone="technical",
)
```

**Supported document types:**  
`report`, `email`, `proposal`, `summary`, `memo`, `blog_post`,
`README`, `cover_letter`, `meeting_notes`, `custom`

**Tones:** `formal`, `friendly`, `technical`, `persuasive`, `casual`

---

### WebResearchAgent  ← NEW

```python
from rag_system_pro.agents import WebResearchAgent

agent = WebResearchAgent(llm=llm)

# Standard research — 3 sub-queries, 3 pages each
result = agent.research(
    question="What are the latest advances in retrieval-augmented generation?",
    depth=3,
    results_per_query=3,
)
print(result.report_md)
for src in result.sources:
    print(src.url, "—", src.title)
result.save("rag_research.md")

# Convenience shorthands
agent.quick("What is LangChain?")                          # 1 query, fast
agent.deep("How do transformers work?")                    # 5 queries, thorough
agent.competitive_analysis("vector database market")
agent.market_research("enterprise AI software 2025")
agent.tech_overview("mixture of experts models")
```

The agent:
1. Decomposes your question into `depth` targeted sub-queries (via LLM)
2. Searches DuckDuckGo for each (no API key needed)
3. Fetches and strips the top pages
4. Synthesises everything into a structured Markdown report
5. Includes inline `[Source N]` citations with URLs

**Dependencies:** `requests`, `beautifulsoup4` (falls back to regex if bs4 is absent)

---

## Combining agents

```python
# Research a topic, then write a polished report from the findings
research = web_agent.research("best practices for Python API design", depth=4)
doc = doc_agent.report(
    topic="Python API Design Best Practices",
    context=research.report_md,
    tone="technical",
)
doc.save("api_design_report.docx")

# Review and fix code, then write documentation for it
fix_result = code_agent.fix(buggy_code, problem="memory leak")
doc_result = code_agent.document(fix_result.output)
readme = doc_agent.readme(f"This module: {doc_result.output[:500]}")
```

---

## Configuration

All agents accept any `LLMClient` instance — swap models at any time:

```python
LLMClient(model="llama3")
LLMClient(model="mistral:7b")
LLMClient(model="codellama:13b")
LLMClient(model="deepseek-coder:6.7b", temperature=0.0)  # deterministic
```

---

## Licence

MIT — do whatever you want with it.
