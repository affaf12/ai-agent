"""
prompt.py — Pro-level prompt engineering module.

Provides structured, validated, and configurable prompt builders
for use with any LLM API (Anthropic, OpenAI, etc.).

Upgrades over v1
----------------
* New ResponseFormats: TABLE, MARKDOWN, XML
* CoTMode enum (SILENT | VERBOSE | NONE) — replaces bool flag
* PromptConfig.__post_init__ validation — catches bad config at construction time
* Prompt-injection defense — scans query/context for jailbreak patterns
* Format-safe substitution — uses string.Template to survive user-supplied braces
* XML-tag wrapping (Anthropic best practice) — structured context injection
* Token budget guard — raises before you waste an API call
* ConversationBuilder — multi-turn message list with system pinning
* OutputValidator — validates raw LLM text against the requested format
* PromptTemplate — named, reusable templates with typed variable slots
* build_agent_prompt — tool-calling / agentic system prompt builder
* Persona registry — swap expert personas without rewriting prompts
* Full type annotations on every public symbol
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResponseFormat(str, Enum):
    """Supported output formats the model is instructed to follow."""
    BULLET       = "bullet"
    STRUCTURED   = "structured"
    JSON         = "json"
    STEP_BY_STEP = "step_by_step"
    TABLE        = "table"       # Markdown table output
    MARKDOWN     = "markdown"    # Free-form markdown (headers, bold, code blocks)
    XML          = "xml"         # XML-tagged output for downstream parsing


class CoTMode(str, Enum):
    """Chain-of-thought behaviour."""
    NONE    = "none"    # No CoT instruction
    SILENT  = "silent"  # Reason internally; output only final answer
    VERBOSE = "verbose" # Output reasoning AND final answer (useful for debugging)


class Persona(str, Enum):
    """
    Built-in expert personas.  Add your own to PERSONA_PROMPTS below.
    """
    GENERIC    = "generic"
    ANALYST    = "analyst"
    ENGINEER   = "engineer"
    RESEARCHER = "researcher"
    TUTOR      = "tutor"
    LEGAL      = "legal"
    MEDICAL    = "medical"


PERSONA_PROMPTS: dict[Persona, str] = {
    Persona.GENERIC:    "You are a precise, expert-level AI assistant. "
                        "Your answers are factual, concise, and free of filler language.",
    Persona.ANALYST:    "You are a senior data analyst. Prioritise quantitative evidence, "
                        "cite assumptions explicitly, and flag uncertainty.",
    Persona.ENGINEER:   "You are a principal software engineer. Prefer concrete code examples, "
                        "highlight edge cases, and call out performance trade-offs.",
    Persona.RESEARCHER: "You are an academic researcher. Cite mechanisms over conclusions, "
                        "acknowledge conflicting evidence, and be epistemically humble.",
    Persona.TUTOR:      "You are a patient, encouraging tutor. Build intuition before detail, "
                        "use analogies, and check understanding at each step.",
    Persona.LEGAL:      "You are a legal analyst (NOT a licensed attorney). Identify relevant "
                        "principles and jurisdiction-specific nuances; always recommend "
                        "qualified counsel for actual decisions.",
    Persona.MEDICAL:    "You are a medical information assistant (NOT a licensed clinician). "
                        "Present evidence-based information and always advise consulting a "
                        "healthcare professional for personal medical decisions.",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PromptConfig:
    """
    Central config for all prompt behaviour.

    Attributes
    ----------
    max_points            Max bullet / step / table-row items returned.
    context_char_limit    Hard cap on injected context (chars).
    response_format       Output structure the model must follow.
    language              ISO-639-1 code for response language.
    confidence_threshold  If set, model flags low-confidence points with ⚠️.
    cot_mode              Chain-of-thought behaviour (NONE | SILENT | VERBOSE).
    few_shot_examples     (question, answer) tuples injected as style examples.
    strict_mode           Refuse off-topic queries instead of guessing.
    persona               Expert role the model should adopt.
    token_budget          If set, raises TokenBudgetExceeded when exceeded.
    wrap_context_xml      Wrap injected context in <context> XML tags.
    allow_prompt_injection Disable injection-pattern scanning (testing only).
    """
    max_points:             int                   = 5
    context_char_limit:     int                   = 2000
    response_format:        ResponseFormat        = ResponseFormat.BULLET
    language:               str                   = "en"
    confidence_threshold:   Optional[float]       = None
    cot_mode:               CoTMode               = CoTMode.NONE
    few_shot_examples:      list[tuple[str, str]] = field(default_factory=list)
    strict_mode:            bool                  = True
    persona:                Persona               = Persona.GENERIC
    token_budget:           Optional[int]         = None   # total tokens (system + user)
    wrap_context_xml:       bool                  = True
    allow_prompt_injection: bool                  = False  # set True only in tests

    def __post_init__(self) -> None:
        """Validate all fields at construction time."""
        if self.max_points < 1:
            raise ValueError(f"max_points must be ≥ 1, got {self.max_points}")
        if self.context_char_limit < 100:
            raise ValueError(
                f"context_char_limit must be ≥ 100, got {self.context_char_limit}"
            )
        if self.confidence_threshold is not None:
            if not (0.0 < self.confidence_threshold < 1.0):
                raise ValueError(
                    "confidence_threshold must be in (0, 1), "
                    f"got {self.confidence_threshold}"
                )
        if self.token_budget is not None and self.token_budget < 50:
            raise ValueError(
                f"token_budget must be ≥ 50 tokens, got {self.token_budget}"
            )
        if len(self.language) not in (2, 5):
            raise ValueError(
                f"language must be an ISO-639-1 code (e.g. 'en', 'fr', 'zh-CN'), "
                f"got '{self.language}'"
            )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TokenBudgetExceeded(RuntimeError):
    """Raised when the combined prompt exceeds config.token_budget."""


class PromptInjectionDetected(ValueError):
    """Raised when a query or context contains suspected injection patterns."""


# ---------------------------------------------------------------------------
# Security: prompt-injection scanner
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)\b", re.I),
    re.compile(r"\bforget\s+(everything|all)\b", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|a\b)", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bDAN\b"),
    re.compile(r"<\s*/?system\s*>", re.I),   # XML system-tag injection
    re.compile(r"\[\s*system\s*\]", re.I),    # Bracket-style injection
]


def _check_injection(text: str, label: str = "input") -> None:
    """
    Raise :class:`PromptInjectionDetected` if *text* matches known
    jailbreak / injection patterns.

    Args:
        text:  The string to scan.
        label: Human-readable field name for the error message.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise PromptInjectionDetected(
                f"Suspected prompt injection in {label!r}: "
                f"matched pattern /{pattern.pattern}/i"
            )


# ---------------------------------------------------------------------------
# Format instructions  (Template-safe — no raw .format() on user content)
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTIONS: dict[ResponseFormat, str] = {
    ResponseFormat.BULLET: textwrap.dedent("""\
        STRICT OUTPUT FORMAT — BULLET LIST (MANDATORY):
        - ONLY bullet points allowed
        - Maximum $max_points bullets
        - Each bullet: one short line (≤ 12 words)
        - NO paragraphs, NO introduction, NO explanation, NO greetings

        Violating this format renders the answer invalid."""),

    ResponseFormat.STRUCTURED: textwrap.dedent("""\
        Output format — STRUCTURED SECTIONS:
        **Summary**: One sentence.
        **Key Points**: Up to $max_points labelled points.
        **Caveats**: Important limitations or unknowns (omit section if none)."""),

    ResponseFormat.JSON: textwrap.dedent("""\
        Output format — JSON ONLY (no markdown fences, no commentary):
        {
          "summary": "<one sentence>",
          "points": ["<point1>", "…", "<point$max_points>"],
          "confidence": "<high|medium|low>",
          "caveats": "<string or null>"
        }"""),

    ResponseFormat.STEP_BY_STEP: textwrap.dedent("""\
        Output format — NUMBERED STEPS:
        1. Step one
        2. Step two
        … up to $max_points steps.
        Conclude with a single **Result**: line."""),

    ResponseFormat.TABLE: textwrap.dedent("""\
        Output format — MARKDOWN TABLE (mandatory):
        | Column A | Column B | Column C |
        |----------|----------|----------|
        | value    | value    | value    |
        - Maximum $max_points data rows (excluding header)
        - No text before or after the table."""),

    ResponseFormat.MARKDOWN: textwrap.dedent("""\
        Output format — STRUCTURED MARKDOWN:
        - Use ## headers to organise sections
        - Use **bold** for key terms
        - Use ``` code blocks ``` for code or commands
        - Maximum $max_points top-level sections
        - No filler sentences."""),

    ResponseFormat.XML: textwrap.dedent("""\
        Output format — XML (no markdown, no prose outside tags):
        <response>
          <summary>One sentence.</summary>
          <points>
            <point>…</point>
            <!-- up to $max_points <point> elements -->
          </points>
          <confidence>high|medium|low</confidence>
          <caveats>string or empty</caveats>
        </response>"""),
}

_STOP_HINTS: dict[ResponseFormat, str] = {
    ResponseFormat.BULLET:       "Stop after the final bullet point.",
    ResponseFormat.STEP_BY_STEP: "Stop after the **Result**: line.",
    ResponseFormat.STRUCTURED:   "Stop after the **Caveats** section (or **Key Points** if omitted).",
    ResponseFormat.JSON:         "Stop after the closing `}`.",
    ResponseFormat.TABLE:        "Stop after the final table row.",
    ResponseFormat.MARKDOWN:     "Stop after the final section.",
    ResponseFormat.XML:          "Stop after the closing `</response>` tag.",
}

_STRICT_FALLBACKS: dict[str, str] = {
    "bullet":       "If the answer cannot be determined, return exactly:\n- Not found",
    "structured":   "If the answer cannot be determined, return exactly:\n**Summary**: Not found",
    "step_by_step": "If the answer cannot be determined, return exactly:\n1. Not found\n**Result**: Not found",
    "json":         (
        "If the answer cannot be determined, return exactly:\n"
        '{"summary": "Not found", "points": [], "confidence": "low", "caveats": null}'
    ),
    "table":        "If the answer cannot be determined, return a table with a single row: | Not found | — | — |",
    "markdown":     "If the answer cannot be determined, return exactly:\n## Not found",
    "xml":          (
        "If the answer cannot be determined, return exactly:\n"
        "<response><summary>Not found</summary><points/>"
        "<confidence>low</confidence><caveats/></response>"
    ),
}

_COT_INSTRUCTIONS: dict[CoTMode, str] = {
    CoTMode.SILENT: (
        "Before answering, reason through the problem step by step internally. "
        "Do NOT output your reasoning — output ONLY the final answer in the required format."
    ),
    CoTMode.VERBOSE: (
        "Reason through the problem step by step. "
        "Wrap your reasoning in <thinking>…</thinking> tags, "
        "then output the final answer in the required format after the closing tag."
    ),
}

_CONFIDENCE_INSTRUCTION = Template(
    "If your confidence in any point is below $threshold, "
    "prefix that point with ⚠️ and briefly state why."
)

_USER_INSTRUCTIONS: dict[ResponseFormat, str] = {
    ResponseFormat.BULLET: (
        "### Instructions\n"
        "- Output ONLY bullet points\n"
        "- Maximum $max_points bullets\n"
        "- No paragraphs, no intro, no outro\n\n"
        "Format:\n- point 1\n- point 2"
    ),
    ResponseFormat.STEP_BY_STEP: (
        "### Instructions\n"
        "Output ONLY a numbered list of at most $max_points steps "
        "followed by a single **Result**: line. No intro, no outro."
    ),
    ResponseFormat.STRUCTURED: (
        "### Instructions\n"
        "Output ONLY **Summary**, **Key Points** (max $max_points), "
        "and optional **Caveats**. Nothing else."
    ),
    ResponseFormat.JSON: (
        "### Instructions\n"
        "Output ONLY a single JSON object as specified in the system prompt. "
        "No markdown fences, no commentary. `points` max $max_points items."
    ),
    ResponseFormat.TABLE: (
        "### Instructions\n"
        "Output ONLY a markdown table with at most $max_points data rows. "
        "No text before or after the table."
    ),
    ResponseFormat.MARKDOWN: (
        "### Instructions\n"
        "Output ONLY structured markdown with at most $max_points ## sections. "
        "No filler sentences."
    ),
    ResponseFormat.XML: (
        "### Instructions\n"
        "Output ONLY a well-formed XML response as specified in the system prompt. "
        "No markdown, no prose outside tags. max $max_points <point> elements."
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Strip control characters and normalise whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _truncate_context(context: str, limit: int) -> str:
    """Truncate *context* to *limit* chars at a sentence boundary."""
    if len(context) <= limit:
        return context
    truncated = context[:limit]
    last_period = truncated.rfind(".")
    if last_period > limit * 0.75:
        truncated = truncated[: last_period + 1]
    return truncated + " […truncated]"


def _substitute(template_str: str, **kwargs: Any) -> str:
    """
    Safe Template substitution — survives user-supplied `{` / `}` characters
    that would explode with str.format().
    """
    return Template(template_str).safe_substitute(**kwargs)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Estimate token count via the ~4 chars/token heuristic.
    Use ``tiktoken`` for exact counts.

    Returns:
        int: Estimated token count (minimum 1).
    """
    return max(1, len(text) // 4)


def _check_budget(system: str, user: str, budget: Optional[int]) -> None:
    """Raise :class:`TokenBudgetExceeded` if combined tokens exceed *budget*."""
    if budget is None:
        return
    total = estimate_tokens(system) + estimate_tokens(user)
    if total > budget:
        raise TokenBudgetExceeded(
            f"Combined prompt is ~{total} tokens, which exceeds "
            f"the configured budget of {budget} tokens. "
            "Reduce context, max_points, or few-shot examples."
        )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(config: Optional[PromptConfig] = None) -> str:
    """
    Build a complete, parameterised system prompt.

    Args:
        config: A :class:`PromptConfig` instance. Defaults to a fresh one per call.

    Returns:
        str: Ready-to-use system prompt.
    """
    if config is None:
        config = PromptConfig()

    parts: list[str] = []

    # Persona / role declaration
    parts.append(PERSONA_PROMPTS[config.persona])

    # Language
    if config.language not in ("en", "en-US", "en-GB"):
        parts.append(f"Respond in language: {config.language} (ISO-639-1).")

    # Chain-of-thought
    if config.cot_mode != CoTMode.NONE:
        parts.append(_COT_INSTRUCTIONS[config.cot_mode])

    # Strict mode with per-format fallback
    if config.strict_mode:
        fallback = _STRICT_FALLBACKS.get(
            config.response_format.value,
            _STRICT_FALLBACKS["bullet"],
        )
        parts.append(fallback)

    # Confidence flagging
    if config.confidence_threshold is not None:
        parts.append(
            _CONFIDENCE_INSTRUCTION.substitute(
                threshold=f"{config.confidence_threshold:.0%}"
            )
        )

    # Output format
    fmt_raw = _FORMAT_INSTRUCTIONS[config.response_format]
    parts.append(_substitute(fmt_raw, max_points=config.max_points))

    # Stop hint
    stop = _STOP_HINTS.get(config.response_format)
    if stop:
        parts.append(stop)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    query: str,
    context: str = "",
    config: Optional[PromptConfig] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build a structured user-turn prompt.

    Args:
        query:    The user's question (required, non-empty after sanitisation).
        context:  Optional background text.
        config:   Shared :class:`PromptConfig` (should match system prompt).
        metadata: Optional key-value pairs appended as structured hints.

    Returns:
        str: Formatted prompt string.

    Raises:
        ValueError:              If *query* is empty after sanitisation.
        PromptInjectionDetected: If query or context contains injection patterns
                                 and ``config.allow_prompt_injection`` is False.
    """
    if config is None:
        config = PromptConfig()

    query = _sanitize(query)
    if not query:
        raise ValueError("`query` must not be empty after sanitisation.")

    context = _sanitize(context)

    # Injection guard
    if not config.allow_prompt_injection:
        _check_injection(query, label="query")
        if context:
            _check_injection(context, label="context")

    context = _truncate_context(context, config.context_char_limit)

    lines: list[str] = [f"### Question\n{query}"]

    if context:
        if config.wrap_context_xml:
            # XML wrapping is an Anthropic best-practice: it separates
            # user-supplied data from instructions, reducing injection risk.
            lines.append(f"### Context\n<context>\n{context}\n</context>")
        else:
            lines.append(f"### Context\n{context}")

    if metadata:
        meta_block = "\n".join(f"- **{k}**: {v}" for k, v in metadata.items())
        lines.append(f"### Metadata\n{meta_block}")

    # Format-aware reminder
    instr_raw = _USER_INSTRUCTIONS.get(
        config.response_format,
        _USER_INSTRUCTIONS[ResponseFormat.BULLET],
    )
    lines.append(_substitute(instr_raw, max_points=config.max_points))

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot block builder
# ---------------------------------------------------------------------------

def build_few_shot_block(examples: list[tuple[str, str]]) -> str:
    """
    Render few-shot examples as a formatted string.

    Args:
        examples: List of (question, ideal_answer) tuples.

    Returns:
        str: Formatted few-shot block, or empty string if *examples* is empty.
    """
    if not examples:
        return ""
    blocks = ["### Examples (follow this style exactly)"]
    for i, (q, a) in enumerate(examples, 1):
        blocks.append(f"**Example {i}**\nQ: {q}\nA: {a}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Agent prompt builder
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """
    Minimal description of a tool available to an agent.

    Attributes:
        name:        Tool identifier (snake_case).
        description: What the tool does (one sentence).
        parameters:  Parameter names and their descriptions.
        required:    Which parameters are mandatory.
    """
    name:        str
    description: str
    parameters:  dict[str, str]  = field(default_factory=dict)
    required:    list[str]       = field(default_factory=list)

    def render(self) -> str:
        lines = [f"**{self.name}**: {self.description}"]
        for param, desc in self.parameters.items():
            req_marker = " *(required)*" if param in self.required else ""
            lines.append(f"  - `{param}`{req_marker}: {desc}")
        return "\n".join(lines)


def build_agent_prompt(
    tools: list[ToolSpec],
    objective: str,
    config: Optional[PromptConfig] = None,
    max_iterations: int = 5,
) -> str:
    """
    Build a system prompt for a tool-calling / agentic workflow.

    Args:
        tools:          Available tools the agent may invoke.
        objective:      High-level goal the agent must achieve.
        config:         :class:`PromptConfig` for base behaviour.
        max_iterations: Hard cap on reasoning / tool-call cycles.

    Returns:
        str: Agent system prompt.
    """
    if config is None:
        config = PromptConfig()

    tool_block = "\n\n".join(t.render() for t in tools)

    return textwrap.dedent(f"""\
        {PERSONA_PROMPTS[config.persona]}

        ## Objective
        {objective}

        ## Available Tools
        {tool_block}

        ## Agent Rules (MANDATORY)
        1. Think before acting — reason step by step inside <thinking> tags.
        2. Call at most ONE tool per iteration.
        3. Never fabricate tool results — use only real outputs.
        4. Stop after at most {max_iterations} iterations even if the goal is unmet.
        5. Conclude with a <final_answer> block summarising your result.
        6. If a required tool is unavailable, explain the limitation and stop.

        ## Output Shape
        <thinking>…your reasoning…</thinking>
        <tool_call>{{ "tool": "<name>", "params": {{…}} }}</tool_call>
        …(repeat as needed)…
        <final_answer>…concise result…</final_answer>
    """).strip()


# ---------------------------------------------------------------------------
# Output validator
# ---------------------------------------------------------------------------

class OutputValidator:
    """
    Validates raw LLM output against the expected :class:`ResponseFormat`.

    Usage::

        validator = OutputValidator(ResponseFormat.JSON)
        result = validator.validate(llm_output)
        if not result.valid:
            print(result.errors)
    """

    @dataclass
    class Result:
        valid:  bool
        errors: list[str] = field(default_factory=list)
        parsed: Any = None  # Populated for JSON / XML if parse succeeds

    def __init__(self, fmt: ResponseFormat) -> None:
        self.fmt = fmt

    def validate(self, text: str) -> "OutputValidator.Result":
        """
        Validate *text* and return a :class:`Result`.

        Args:
            text: Raw model output string.

        Returns:
            OutputValidator.Result: Validation outcome with errors list.
        """
        text = text.strip()
        errors: list[str] = []

        if not text:
            return self.Result(valid=False, errors=["Output is empty."])

        if self.fmt == ResponseFormat.JSON:
            return self._validate_json(text)

        if self.fmt == ResponseFormat.XML:
            return self._validate_xml(text)

        if self.fmt == ResponseFormat.BULLET:
            lines = [l for l in text.splitlines() if l.strip()]
            non_bullets = [l for l in lines if not l.strip().startswith("-")]
            if non_bullets:
                errors.append(
                    f"Non-bullet lines found: {non_bullets[:3]} …"
                )

        if self.fmt == ResponseFormat.STEP_BY_STEP:
            if not re.search(r"\*\*Result\*\*", text):
                errors.append("Missing required **Result**: line.")

        if self.fmt == ResponseFormat.TABLE:
            if "|" not in text:
                errors.append("No markdown table found (missing `|` characters).")

        return self.Result(valid=len(errors) == 0, errors=errors)

    @staticmethod
    def _validate_json(text: str) -> "OutputValidator.Result":
        # Strip accidental markdown fences before parsing.
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
        try:
            parsed = json.loads(clean)
            return OutputValidator.Result(valid=True, parsed=parsed)
        except json.JSONDecodeError as exc:
            return OutputValidator.Result(
                valid=False,
                errors=[f"JSON parse error: {exc}"],
            )

    @staticmethod
    def _validate_xml(text: str) -> "OutputValidator.Result":
        if not text.strip().startswith("<response>"):
            return OutputValidator.Result(
                valid=False,
                errors=["XML output must start with <response>."],
            )
        if not text.strip().endswith("</response>"):
            return OutputValidator.Result(
                valid=False,
                errors=["XML output must end with </response>."],
            )
        return OutputValidator.Result(valid=True, parsed=text)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """
    A named, reusable prompt template with typed variable slots.

    Example::

        tmpl = PromptTemplate(
            name="company_summary",
            template="Summarise the company $company in the $industry industry.",
            required_vars=["company", "industry"],
        )
        query = tmpl.render(company="Acme Corp", industry="SaaS")
    """
    name:          str
    template:      str
    required_vars: list[str] = field(default_factory=list)
    description:   str = ""

    def render(self, **kwargs: Any) -> str:
        """
        Substitute *kwargs* into the template.

        Raises:
            ValueError: If any required variable is missing.
        """
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing required variable(s): {missing}"
            )
        return _substitute(self.template, **kwargs)


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------

class ConversationBuilder:
    """
    Incrementally builds a multi-turn message list for the LLM API,
    pinning the system prompt across every call.

    Usage::

        cb = ConversationBuilder(system_prompt=build_system_prompt(cfg))
        cb.add_user("What is 2 + 2?")
        cb.add_assistant("- 4")
        cb.add_user("Multiply by 3.")
        messages = cb.build()   # pass to API
    """

    def __init__(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._turns:  list[dict[str, str]] = []

    def add_user(self, content: str, allow_injection: bool = False) -> "ConversationBuilder":
        """Append a user turn. Scans for injection by default."""
        content = _sanitize(content)
        if not allow_injection:
            _check_injection(content, label="user turn")
        self._turns.append({"role": "user", "content": content})
        return self

    def add_assistant(self, content: str) -> "ConversationBuilder":
        """Append an assistant turn."""
        self._turns.append({"role": "assistant", "content": _sanitize(content)})
        return self

    def build(self) -> dict[str, Any]:
        """
        Return a dict with ``system`` and ``messages`` keys,
        compatible with the Anthropic and OpenAI APIs.
        """
        return {"system": self._system, "messages": list(self._turns)}

    def reset(self) -> "ConversationBuilder":
        """Clear turn history while keeping the system prompt."""
        self._turns.clear()
        return self

    def __len__(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# Convenience: full prompt pair
# ---------------------------------------------------------------------------

def build_prompt_pair(
    query: str,
    context: str = "",
    config: Optional[PromptConfig] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """
    Return a ``{"system": …, "user": …}`` dict ready for any LLM API.

    Also enforces the token budget if one is set in *config*.

    Args:
        query:    User question.
        context:  Optional background text.
        config:   :class:`PromptConfig` instance.
        metadata: Optional key-value metadata pairs.

    Returns:
        dict[str, str]: Keys ``system`` and ``user``.

    Raises:
        TokenBudgetExceeded: If the combined prompt exceeds ``config.token_budget``.
    """
    cfg = config or PromptConfig()
    system = build_system_prompt(cfg)

    if cfg.few_shot_examples:
        few_shot = build_few_shot_block(cfg.few_shot_examples)
        system = system + "\n\n" + few_shot

    user = build_user_prompt(query, context, cfg, metadata)

    _check_budget(system, user, cfg.token_budget)

    return {"system": system, "user": user}
