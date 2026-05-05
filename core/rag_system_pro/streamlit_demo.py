"""
Streamlit demo wiring everything together:

    * RAG Q&A over your docs/         (knowledge)
    * Excel cleaner agent             (skill)
    * Code agent (explain/fix/review/etc) (skill)

Run with:
    streamlit run streamlit_demo.py

Requires the package to be importable. If your project layout is
    project_root/
        core/rag_system_pro/
        streamlit_demo.py
add this BEFORE the imports below:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "core"))
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from rag_system_pro import RAGSystem
from rag_system_pro.agents import CodeAgent, ExcelAgent, LLMClient


st.set_page_config(page_title="Ollama Pro", layout="wide")
st.title("Ollama Pro")

# ----------------------------------------------------------------------- shared

@st.cache_resource
def get_rag(chat_model: str, embed_model: str) -> RAGSystem:
    return RAGSystem(
        llm_model=chat_model,
        embed_model=embed_model,
        index_path="data/faiss.index",
    )


@st.cache_resource
def get_excel_agent(model: str) -> ExcelAgent:
    return ExcelAgent(llm=LLMClient(model=model))


@st.cache_resource
def get_code_agent(model: str) -> CodeAgent:
    return CodeAgent(llm=LLMClient(model=model))


# --------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    chat_model = st.selectbox(
        "Chat model",
        ["llama3:instruct", "gemma:2b", "deepseek-coder:6.7b", "phi:latest", "tinyllama:latest"],
        index=0,
    )
    embed_model = st.selectbox("Embedding model", ["nomic-embed-text"], index=0)
    code_model = st.selectbox(
        "Code/agent model",
        ["deepseek-coder:6.7b", "llama3:instruct", "gemma:2b"],
        index=0,
    )

# ------------------------------------------------------------------------- tabs

tab_chat, tab_excel, tab_code = st.tabs(["Knowledge Chat", "Excel Cleaner", "Code Agent"])

# -------------------------------------------------------------- KNOWLEDGE CHAT

with tab_chat:
    st.subheader("Ask your knowledge base")
    rag = get_rag(chat_model, embed_model)
    q = st.chat_input("Ask anything about your indexed documents")
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            for tok in rag.stream_query(q, top_k=5):
                full += tok
                placeholder.markdown(full)
            resp = rag.query(q, top_k=5)  # cached -> instant
            with st.expander(f"Sources (confidence: {resp.confidence:.0%})"):
                for i, src in enumerate(resp.sources, 1):
                    label = src.get("metadata", {}).get("source", src.get("id", "?"))
                    st.markdown(f"**[{i}] {label}**")
                    st.caption(src["text"][:400] + "...")

# ---------------------------------------------------------------- EXCEL AGENT

with tab_excel:
    st.subheader("Upload a messy Excel/CSV — get a clean one back")
    agent = get_excel_agent(code_model)
    file = st.file_uploader("Drop an .xlsx or .csv", type=["xlsx", "xls", "csv"])
    instruction = st.text_area(
        "Optional instruction",
        placeholder="e.g. drop duplicates, fix the date column, snake_case headers, fill blank amounts with 0",
        height=80,
    )
    if file and st.button("Clean it", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getbuffer())
            in_path = tmp.name
        with st.spinner("Inspecting + planning + cleaning..."):
            result = agent.clean(in_path, instruction=instruction)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows before", result.rows_before)
        c2.metric("Rows after", result.rows_after, result.rows_after - result.rows_before)
        c3.metric("Cols before", result.cols_before)
        c4.metric("Cols after", result.cols_after, result.cols_after - result.cols_before)
        st.markdown(result.report_md)
        with open(result.cleaned_path, "rb") as f:
            st.download_button(
                "Download cleaned file",
                f.read(),
                file_name=os.path.basename(result.cleaned_path),
            )

# ----------------------------------------------------------------- CODE AGENT

with tab_code:
    st.subheader("Do anything with code")
    agent = get_code_agent(code_model)
    task = st.selectbox(
        "Task",
        ["explain", "review", "refactor", "fix", "document", "test", "translate", "optimize", "generate", "run"],
    )

    if task == "generate":
        spec = st.text_area("Describe what to build", height=200)
        language = st.selectbox("Language", ["python", "typescript", "javascript", "go", "rust", "java", "sql"])
        if st.button("Generate", type="primary") and spec.strip():
            with st.spinner("Generating..."):
                res = agent.generate(spec, language=language)
            st.code(res.output, language=res.language)
    else:
        code = st.text_area("Paste code", height=300)
        extra = ""
        target = "python"
        if task in {"refactor", "fix", "translate"}:
            extra = st.text_input(
                "Instruction / problem / target language",
                placeholder="e.g. 'remove duplicate logic' or 'fix the off-by-one bug' or 'translate to TypeScript'",
            )
            if task == "translate":
                target = extra or "typescript"

        if st.button("Run task", type="primary") and code.strip():
            with st.spinner(f"Running {task}..."):
                if task == "explain":
                    res = agent.explain(code)
                elif task == "review":
                    res = agent.review(code)
                elif task == "refactor":
                    res = agent.refactor(code, instruction=extra)
                elif task == "fix":
                    res = agent.fix(code, problem=extra or "general bug fix")
                elif task == "document":
                    res = agent.document(code)
                elif task == "test":
                    res = agent.test(code)
                elif task == "translate":
                    res = agent.translate(code, target_language=target)
                elif task == "optimize":
                    res = agent.optimize(code, goal=extra or "speed")
                elif task == "run":
                    res = agent.run(code)
                else:
                    st.error("Unknown task")
                    res = None

            if res is not None:
                if task in {"explain", "review"}:
                    st.markdown(res.output)
                else:
                    st.code(res.output, language=res.language)
                    if res.diff:
                        with st.expander("Diff"):
                            st.code(res.diff, language="diff")
                    if res.explanation:
                        with st.expander("Explanation"):
                            st.markdown(res.explanation)
