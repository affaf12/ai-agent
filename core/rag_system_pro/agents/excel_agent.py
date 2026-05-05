"""
ExcelAgent - inspect, clean and export messy spreadsheets.

Workflow:
    1.  inspect(path) -> structured profile (sheets, dtypes, missing %, dups, samples)
    2.  plan(profile, instruction) -> ordered list of cleaning steps (JSON)
    3.  execute(plan) -> runs each step in the sandbox via pandas
    4.  export() -> writes a cleaned .xlsx and a markdown change report

Cleaning steps the AI is allowed to plan (the executor enforces this list):
    - drop_duplicates
    - drop_empty_rows / drop_empty_columns
    - rename_columns                  {"map": {"Old": "new"}}
    - strip_whitespace                ["col1", "col2"]
    - lowercase_columns               (snake_case headers)
    - to_datetime                     {"col": "date_col", "format": "%Y-%m-%d"}
    - to_numeric                      {"col": "amount", "errors": "coerce"}
    - fill_na                         {"col": "x", "value": 0}     or "median" / "mean" / "mode"
    - clip_outliers                   {"col": "x", "lower_quantile": 0.01, "upper_quantile": 0.99}
    - replace_values                  {"col": "x", "map": {"NA": null}}
    - filter_rows                     {"query": "amount > 0"}
    - sort_values                     {"by": ["date"], "ascending": true}
    - run_pandas                      {"code": "df = df.assign(...)"}   <- escape hatch
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .sandbox import PythonSandbox


_PLAN_SYSTEM = """You are a senior data engineer. You receive a profile of a \
messy spreadsheet and a user instruction. You must respond with ONLY a JSON \
object of the form:

{
  "summary": "<1 sentence describing the cleaning strategy>",
  "steps": [
    {"op": "<one of the allowed ops>", "args": {...}, "why": "<short reason>"},
    ...
  ]
}

Allowed ops:
  drop_duplicates, drop_empty_rows, drop_empty_columns, rename_columns,
  strip_whitespace, lowercase_columns, to_datetime, to_numeric, fill_na,
  clip_outliers, replace_values, filter_rows, sort_values, run_pandas

Rules:
- Only output JSON. No prose, no markdown fences.
- Be conservative: do not drop columns the user might want.
- Prefer multiple small steps over one giant `run_pandas`.
- Reference real column names from the profile.
- If the user gave no specific instruction, do a sensible default cleanup:
  trim whitespace, snake_case headers, drop fully empty rows/cols, drop exact
  duplicate rows, coerce obvious date and numeric columns, fill blanks
  conservatively.
"""


@dataclass
class ExcelCleanResult:
    cleaned_path: str
    report_md: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    steps_run: List[Dict[str, Any]] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)


class ExcelAgent:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        sandbox: Optional[PythonSandbox] = None,
        work_dir: str = "data/agent_work",
    ):
        self.llm = llm or LLMClient(task_type="data")
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.sandbox = sandbox or PythonSandbox(allowed_dir=work_dir, timeout_seconds=60)

    # =================================================================== INSPECT

    def inspect(self, path: str, sheet: Optional[str] = None) -> Dict[str, Any]:
        import pandas as pd

        if path.lower().endswith(".csv"):
            sheets = {"Sheet1": pd.read_csv(path)}
        else:
            xl = pd.ExcelFile(path)
            sheets = {
                name: xl.parse(name)
                for name in (xl.sheet_names if sheet is None else [sheet])
            }

        profile_sheets: Dict[str, Any] = {}
        for name, df in sheets.items():
            profile_sheets[name] = {
                "shape": list(df.shape),
                "columns": [
                    {
                        "name": str(c),
                        "dtype": str(df[c].dtype),
                        "missing_pct": round(float(df[c].isna().mean() * 100), 2),
                        "n_unique": int(df[c].nunique(dropna=True)),
                        "sample": [self._jsonable(v) for v in df[c].dropna().head(5).tolist()],
                    }
                    for c in df.columns
                ],
                "duplicate_rows": int(df.duplicated().sum()),
                "head": [
                    {str(k): self._jsonable(v) for k, v in row.items()}
                    for row in df.head(5).to_dict(orient="records")
                ],
            }
        return {"path": path, "sheets": profile_sheets}

    # =================================================================== PLAN

    def plan(self, profile: Dict[str, Any], instruction: str = "") -> Dict[str, Any]:
        instruction = instruction.strip() or "Do a sensible default cleanup."
        prompt = (
            f"USER INSTRUCTION:\n{instruction}\n\n"
            f"SPREADSHEET PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
            "Respond with the cleaning plan JSON now."
        )
        return self.llm.chat_json(prompt, system=_PLAN_SYSTEM)

    # =================================================================== EXECUTE

    def clean(
        self,
        path: str,
        instruction: str = "",
        sheet: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> ExcelCleanResult:
        import pandas as pd

        # 1. Load
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
        rows_before, cols_before = df.shape

        # 2. Profile + plan
        profile = self.inspect(path, sheet=sheet)
        plan = self.plan(profile, instruction=instruction)

        # 3. Execute steps in the sandbox so the agent's `run_pandas` escape
        #    hatch is contained. The dataframe is injected as `df`.
        self.sandbox.inject(df=df, pd=pd)
        report_lines: List[str] = [
            f"# Cleaning report for `{os.path.basename(path)}`",
            "",
            f"**Strategy:** {plan.get('summary', 'n/a')}",
            "",
            "## Steps",
            "",
        ]
        steps_run: List[Dict[str, Any]] = []
        for i, step in enumerate(plan.get("steps", []), 1):
            op = step.get("op")
            args = step.get("args", {}) or {}
            why = step.get("why", "")
            code = self._compile_step(op, args)
            if not code:
                report_lines.append(f"{i}. SKIPPED unknown op `{op}`")
                continue
            res = self.sandbox.run(code)
            if not res.ok:
                report_lines.append(f"{i}. **{op}** failed: `{res.error and res.error.splitlines()[-1]}`")
                continue
            steps_run.append({"op": op, "args": args, "why": why})
            df = self.sandbox.get("df")  # refresh after step
            report_lines.append(
                f"{i}. **{op}** — {why or '(no reason given)'}  \n"
                f"   shape now: {df.shape}"
            )

        rows_after, cols_after = df.shape
        report_lines += [
            "",
            "## Summary",
            f"- Rows: {rows_before} -> {rows_after} (delta {rows_after - rows_before})",
            f"- Columns: {cols_before} -> {cols_after} (delta {cols_after - cols_before})",
        ]

        # 4. Export
        if output_path is None:
            base = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(self.work_dir, f"{base}.cleaned.xlsx")
        df.to_excel(output_path, index=False)
        report_path = os.path.splitext(output_path)[0] + ".report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        return ExcelCleanResult(
            cleaned_path=output_path,
            report_md="\n".join(report_lines),
            rows_before=rows_before,
            rows_after=rows_after,
            cols_before=cols_before,
            cols_after=cols_after,
            steps_run=steps_run,
            profile=profile,
        )

    # =================================================================== STEP COMPILER

    def _compile_step(self, op: str, args: Dict[str, Any]) -> Optional[str]:
        """Translate a JSON step into safe pandas code."""
        if op == "drop_duplicates":
            subset = args.get("subset")
            if subset:
                return f"df = df.drop_duplicates(subset={subset!r})"
            return "df = df.drop_duplicates()"

        if op == "drop_empty_rows":
            return "df = df.dropna(how='all')"

        if op == "drop_empty_columns":
            return "df = df.dropna(axis=1, how='all')"

        if op == "rename_columns":
            mapping = args.get("map") or {}
            return f"df = df.rename(columns={mapping!r})"

        if op == "strip_whitespace":
            cols = args.get("columns") or args.get("cols") or []
            if not cols:
                return (
                    "for _c in df.select_dtypes(include='object').columns:\n"
                    "    df[_c] = df[_c].astype(str).str.strip()\n"
                    "df"
                )
            return "\n".join(f"df[{c!r}] = df[{c!r}].astype(str).str.strip()" for c in cols) + "\ndf"

        if op == "lowercase_columns":
            return (
                "import re\n"
                "df.columns = [re.sub(r'[^0-9a-zA-Z]+', '_', str(c)).strip('_').lower() "
                "for c in df.columns]\n"
                "df"
            )

        if op == "to_datetime":
            col = args["col"]
            fmt = args.get("format")
            if fmt:
                return f"df[{col!r}] = pd.to_datetime(df[{col!r}], format={fmt!r}, errors='coerce')"
            return f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce')"

        if op == "to_numeric":
            col = args["col"]
            errors = args.get("errors", "coerce")
            return f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors={errors!r})"

        if op == "fill_na":
            col = args["col"]
            value = args.get("value")
            if value in ("median", "mean", "mode"):
                if value == "mode":
                    return f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].mode().iloc[0])"
                return f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].{value}())"
            return f"df[{col!r}] = df[{col!r}].fillna({value!r})"

        if op == "clip_outliers":
            col = args["col"]
            lo = args.get("lower_quantile", 0.01)
            hi = args.get("upper_quantile", 0.99)
            return (
                f"_lo = df[{col!r}].quantile({lo})\n"
                f"_hi = df[{col!r}].quantile({hi})\n"
                f"df[{col!r}] = df[{col!r}].clip(_lo, _hi)"
            )

        if op == "replace_values":
            col = args["col"]
            mapping = args.get("map") or {}
            return f"df[{col!r}] = df[{col!r}].replace({mapping!r})"

        if op == "filter_rows":
            query = args["query"]
            return f"df = df.query({query!r})"

        if op == "sort_values":
            by = args["by"]
            asc = args.get("ascending", True)
            return f"df = df.sort_values(by={by!r}, ascending={asc!r})"

        if op == "run_pandas":
            # Escape hatch: trust but verify. We at least block obvious badness.
            code = args.get("code", "")
            forbidden = ["import os", "import sys", "subprocess", "open(", "__"]
            if any(f in code for f in forbidden):
                return None
            return code

        return None

    # =================================================================== utils

    @staticmethod
    def _jsonable(v: Any) -> Any:
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)
