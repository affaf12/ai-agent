"""
CodeAgent - a "do anything with code" assistant.

Tasks supported:
    - explain         walk through what code does
    - review          find bugs, smells, security issues, perf problems
    - refactor        clean up while preserving behavior
    - fix             apply specific fixes (bugs, lint, types)
    - document        add docstrings/comments/JSDoc/etc.
    - test            generate unit tests for a target framework
    - translate       convert between languages (py <-> js <-> ts <-> go ...)
    - optimize        improve performance / memory
    - generate        write new code from a spec
    - run             execute Python code in the sandbox and report results

Each task returns a CodeTaskResult with the output code, an explanation,
optional diff, and optional sandbox run.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm import LLMClient
from .sandbox import PythonSandbox, SandboxResult


_LANGS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala", ".sh": "bash",
    ".sql": "sql", ".html": "html", ".css": "css", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".toml": "toml", ".md": "markdown",
}


def detect_language(path_or_code: str) -> str:
    if "\n" not in path_or_code and "." in path_or_code:
        ext = os.path.splitext(path_or_code)[1].lower()
        if ext in _LANGS:
            return _LANGS[ext]
    return "python"


_SYSTEM_BASE = """You are an expert polyglot software engineer. You write \
production-quality code: correct, idiomatic, well-named, well-typed, and as \
small as it can be while still being clear. You never invent APIs. When asked \
for code, return ONLY a single fenced code block with the right language tag \
unless the user asked for prose explanation."""


@dataclass
class CodeTaskResult:
    task: str
    language: str
    output: str
    explanation: str = ""
    diff: str = ""
    run: Optional[SandboxResult] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class CodeAgent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        sandbox: Optional[PythonSandbox] = None,
        work_dir: str = "data/agent_work",
    ):
        self.llm = llm or LLMClient(task_type="code")
        os.makedirs(work_dir, exist_ok=True)
        self.sandbox = sandbox or PythonSandbox(allowed_dir=work_dir, timeout_seconds=30)

    # ---------------------------------------------------------------- explain

    def explain(self, code: str, language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"Explain the following {lang} code clearly. Cover what it does, "
            f"its inputs/outputs, side effects, edge cases, and any pitfalls. "
            f"Use short sections with headings. No code in the answer.\n\n"
            f"```{lang}\n{code}\n```"
        )
        text = self.llm.chat(prompt, system=_SYSTEM_BASE, temperature=0.3)
        return CodeTaskResult(task="explain", language=lang, output=text, explanation=text)

    # ----------------------------------------------------------------- review

    def review(self, code: str, language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"Review the following {lang} code as a senior engineer. Produce a "
            f"prioritised list with sections: BUGS, SECURITY, PERFORMANCE, "
            f"STYLE, TESTABILITY. For each item give a 1-line title, the "
            f"affected line(s), why it matters, and a concrete suggested fix. "
            f"Be specific. No code rewrite, just findings.\n\n"
            f"```{lang}\n{code}\n```"
        )
        text = self.llm.chat(prompt, system=_SYSTEM_BASE, temperature=0.2)
        return CodeTaskResult(task="review", language=lang, output=text, explanation=text)

    # --------------------------------------------------------------- refactor

    def refactor(self, code: str, instruction: str = "", language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        intent = instruction.strip() or "Improve clarity and structure while keeping behavior identical."
        prompt = (
            f"Refactor the following {lang} code. {intent}\n"
            f"Preserve public API and behavior unless I asked otherwise. "
            f"Return ONLY the refactored code in a single fenced block.\n\n"
            f"```{lang}\n{code}\n```"
        )
        new_code = self.llm.chat_code(prompt, system=_SYSTEM_BASE, language=lang)
        return CodeTaskResult(
            task="refactor",
            language=lang,
            output=new_code,
            diff=_unified_diff(code, new_code),
        )

    # --------------------------------------------------------------------- fix

    def fix(self, code: str, problem: str, language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"The following {lang} code has this problem:\n{problem}\n\n"
            f"Fix it. Return ONLY the corrected code in a single fenced block, "
            f"followed by a one-paragraph explanation after the block.\n\n"
            f"```{lang}\n{code}\n```"
        )
        raw = self.llm.chat(prompt, system=_SYSTEM_BASE, temperature=0.2)
        new_code = self.llm._strip_code_fence(raw, lang)
        explanation = raw.split("```")[-1].strip() if "```" in raw else ""
        return CodeTaskResult(
            task="fix",
            language=lang,
            output=new_code,
            explanation=explanation,
            diff=_unified_diff(code, new_code),
        )

    # --------------------------------------------------------------- document

    def document(self, code: str, language: Optional[str] = None, style: str = "auto") -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"Add high-quality {style} docstrings/comments to the following "
            f"{lang} code. Document every public function/class with purpose, "
            f"params, returns, raises and one-line examples where useful. "
            f"Do not change behavior or names. Return ONLY the documented code "
            f"in a single fenced block.\n\n"
            f"```{lang}\n{code}\n```"
        )
        new_code = self.llm.chat_code(prompt, system=_SYSTEM_BASE, language=lang)
        return CodeTaskResult(
            task="document",
            language=lang,
            output=new_code,
            diff=_unified_diff(code, new_code),
        )

    # --------------------------------------------------------------------- test

    def test(self, code: str, framework: str = "pytest", language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"Write thorough {framework} tests for the following {lang} code. "
            f"Cover happy paths, edge cases, and error paths. Use clear names. "
            f"Return ONLY the test file in a single fenced block.\n\n"
            f"```{lang}\n{code}\n```"
        )
        tests = self.llm.chat_code(prompt, system=_SYSTEM_BASE, language=lang)
        return CodeTaskResult(task="test", language=lang, output=tests)

    # --------------------------------------------------------------- translate

    def translate(self, code: str, target_language: str, source_language: Optional[str] = None) -> CodeTaskResult:
        src = source_language or detect_language(code)
        prompt = (
            f"Translate the following {src} code into idiomatic {target_language}. "
            f"Preserve behavior. Use the conventions of {target_language} "
            f"(naming, error handling, types). Return ONLY the translated code "
            f"in a single fenced block.\n\n"
            f"```{src}\n{code}\n```"
        )
        out = self.llm.chat_code(prompt, system=_SYSTEM_BASE, language=target_language)
        return CodeTaskResult(
            task="translate",
            language=target_language,
            output=out,
            metadata={"from": src, "to": target_language},
        )

    # ---------------------------------------------------------------- optimize

    def optimize(self, code: str, goal: str = "speed", language: Optional[str] = None) -> CodeTaskResult:
        lang = language or detect_language(code)
        prompt = (
            f"Optimise the following {lang} code for {goal}. Keep behavior "
            f"identical. After the code, briefly justify the changes and the "
            f"expected complexity improvement. Return the optimised code in a "
            f"fenced block FIRST, then the explanation.\n\n"
            f"```{lang}\n{code}\n```"
        )
        raw = self.llm.chat(prompt, system=_SYSTEM_BASE, temperature=0.2)
        new_code = self.llm._strip_code_fence(raw, lang)
        explanation = raw.split("```")[-1].strip() if "```" in raw else ""
        return CodeTaskResult(
            task="optimize",
            language=lang,
            output=new_code,
            explanation=explanation,
            diff=_unified_diff(code, new_code),
        )

    # --------------------------------------------------------------- generate

    def generate(self, spec: str, language: str = "python") -> CodeTaskResult:
        prompt = (
            f"Write {language} code for the following specification. Aim for "
            f"production quality: types, error handling, small functions, no "
            f"unnecessary deps. Return ONLY the code in a single fenced block.\n\n"
            f"SPEC:\n{spec}"
        )
        out = self.llm.chat_code(prompt, system=_SYSTEM_BASE, language=language)
        return CodeTaskResult(task="generate", language=language, output=out)

    # -------------------------------------------------------------------- run

    def run(self, code: str) -> CodeTaskResult:
        """Execute Python code in the sandbox and report stdout/value/errors."""
        result = self.sandbox.run(code)
        summary = (
            f"ok: {result.ok}\n"
            f"duration: {result.duration_ms} ms\n"
            f"stdout:\n{result.stdout}\n"
            + (f"stderr:\n{result.stderr}\n" if result.stderr else "")
            + (f"error:\n{result.error}\n" if result.error else "")
            + (f"value:\n{result.value!r}\n" if result.value is not None else "")
        )
        return CodeTaskResult(
            task="run",
            language="python",
            output=summary,
            run=result,
        )


def _unified_diff(before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
