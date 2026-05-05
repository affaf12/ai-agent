"""
Lightweight Python execution sandbox for tool-using agents.

This is intentionally simple, NOT a security boundary. It blocks the most
common foot-guns (network, subprocess, filesystem traversal) but a determined
attacker could still escape. Use it on code you, the operator, would have run
yourself.

Features:
* Captures stdout, stderr, and the value of a final expression.
* Hard timeout (best-effort, thread-based).
* Pre-imported pandas / numpy / matplotlib for data work.
* Persistent namespace across calls (`run_again` re-uses globals) so the agent
  can take multi-step actions without losing state.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_BLOCKED_MODULES = {
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "asyncio.subprocess",
}

_BLOCKED_BUILTINS = {"exec", "eval", "compile", "open", "__import__"}
# We re-add a controlled `open` and `__import__` below.


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _BLOCKED_MODULES or name in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is blocked in the sandbox")
    return __import__(name, globals, locals, fromlist, level)


def _safe_open_factory(allowed_dir: Optional[str]):
    import os

    real_open = builtins.open

    def _safe_open(file, mode="r", *args, **kwargs):
        path = os.path.abspath(str(file))
        if allowed_dir is not None:
            allowed = os.path.abspath(allowed_dir)
            if not path.startswith(allowed):
                raise PermissionError(
                    f"open() restricted to {allowed}; got {path}"
                )
        if any(c in mode for c in "wax+"):
            # writes allowed only inside allowed_dir
            if allowed_dir is None:
                raise PermissionError("Writes are disabled in this sandbox")
        return real_open(file, mode, *args, **kwargs)

    return _safe_open


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    value: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    namespace: Dict[str, Any] = field(default_factory=dict)


class PythonSandbox:
    def __init__(
        self,
        allowed_dir: Optional[str] = None,
        timeout_seconds: float = 30.0,
        preload_data_libs: bool = True,
    ):
        self.allowed_dir = allowed_dir
        self.timeout = timeout_seconds
        self._namespace: Dict[str, Any] = {}
        self._build_namespace(preload_data_libs)

    # -------------------------------------------------------------- namespace

    def _build_namespace(self, preload_data_libs: bool) -> None:
        safe_builtins = {
            k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS
        }
        safe_builtins["__import__"] = _safe_import
        safe_builtins["open"] = _safe_open_factory(self.allowed_dir)

        ns: Dict[str, Any] = {"__builtins__": safe_builtins}

        if preload_data_libs:
            try:
                import pandas as pd  # type: ignore
                import numpy as np  # type: ignore

                ns["pd"] = pd
                ns["np"] = np
            except Exception:
                pass
            try:
                import matplotlib  # type: ignore

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # type: ignore

                ns["plt"] = plt
            except Exception:
                pass

        self._namespace = ns

    # ----------------------------------------------------------------- run

    def inject(self, **kwargs: Any) -> None:
        """Add variables to the sandbox namespace before running code."""
        self._namespace.update(kwargs)

    def get(self, name: str, default: Any = None) -> Any:
        return self._namespace.get(name, default)

    def reset(self) -> None:
        self._build_namespace(preload_data_libs=True)

    def run(self, code: str) -> SandboxResult:
        import time

        out = io.StringIO()
        err = io.StringIO()
        result_holder: Dict[str, Any] = {"value": None, "error": None}

        def _exec() -> None:
            try:
                # Compile as Interactive so a final expression returns its value.
                # We split out the last line and try eval() on it.
                lines = code.strip().splitlines()
                body = "\n".join(lines[:-1]) if len(lines) > 1 else ""
                tail = lines[-1] if lines else ""

                if body.strip():
                    exec(compile(body, "<sandbox>", "exec"), self._namespace)
                try:
                    result_holder["value"] = eval(
                        compile(tail, "<sandbox>", "eval"), self._namespace
                    )
                except SyntaxError:
                    exec(compile(tail, "<sandbox>", "exec"), self._namespace)
            except Exception:
                result_holder["error"] = traceback.format_exc()

        started = time.time()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            t = threading.Thread(target=_exec, daemon=True)
            t.start()
            t.join(self.timeout)
            timed_out = t.is_alive()

        duration_ms = int((time.time() - started) * 1000)

        if timed_out:
            return SandboxResult(
                ok=False,
                stdout=out.getvalue(),
                stderr=err.getvalue(),
                error=f"Timed out after {self.timeout}s",
                duration_ms=duration_ms,
                namespace=self._namespace,
            )

        return SandboxResult(
            ok=result_holder["error"] is None,
            stdout=out.getvalue(),
            stderr=err.getvalue(),
            value=result_holder["value"],
            error=result_holder["error"],
            duration_ms=duration_ms,
            namespace=self._namespace,
        )
