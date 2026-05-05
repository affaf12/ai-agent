"""
core/brain.py — Universal Q&A Control Layer  (pro edition)

Pipeline: Query → Sanitize → Classify → Rewrite → Route → RAG? → Generate → Clean → Trace

Upgrades over v1
----------------
* TaskType enum — 16 task types vs the original 4; each maps to a specialised agent
* IntentClassifier — weighted signal scoring with confidence; returns ranked candidates
* AgentSpec + AgentRegistry — pluggable, self-describing agent registry
* Router — resolves TaskType → AgentSpec with a priority fallback chain
* QueryRewriter — rephrases the query for better RAG recall before retrieval
* ContextBuilder — deduplicates, scores, and ranks RAG docs; smart sentence-boundary truncation
* OutputCleaner — extensible bad-token registry with log-line and artefact stripping
* BrainConfig — single dataclass controlling every tunable parameter
* BrainResult + ExecutionTrace — full audit trail: intent, confidence, agent, RAG hit count, latency
* Brain — main orchestrator with pre/post middleware hooks and proper exception surfaces
* Protocol types — RAGSystem and LLMGenerator are typed; no more untyped callables
* Errors never swallowed silently — all exceptions are re-raised or captured in the trace
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """
    Fine-grained task types the classifier can assign.
    Each maps to a different agent and RAG strategy.
    """
    GREETING       = "greeting"       # "hi", "hello", "how are you"
    CODING         = "coding"         # write/fix/explain code
    CODE_REVIEW    = "code_review"    # review / audit existing code
    DEBUG          = "debug"          # traceback / runtime error diagnosis
    MATH           = "math"           # arithmetic, algebra, calculus
    DATA_ANALYSIS  = "data_analysis"  # stats, trends, datasets, charts
    EXCEL          = "excel"          # spreadsheet formulas, pivot tables
    SUMMARIZE      = "summarize"      # TL;DR / summarise / condense
    TRANSLATE      = "translate"      # language translation
    CREATIVE       = "creative"       # story, poem, creative writing
    COMPARISON     = "comparison"     # compare X vs Y
    STEP_BY_STEP   = "step_by_step"   # "how do I", "walk me through"
    FACTUAL_QA     = "factual_qa"     # who/what/when/where factual lookup
    ANALYSIS       = "analysis"       # deep reasoning / multi-part questions
    GENERAL        = "general"        # catch-all
    UNKNOWN        = "unknown"        # unclassifiable; route to safe fallback


# ---------------------------------------------------------------------------
# Weighted signal tables
# ---------------------------------------------------------------------------

# Each entry: signal_substring → (TaskType, weight)
# Higher weight = stronger signal for that task type.
# Multiple matches accumulate; the highest-scoring type wins.

_SIGNALS: list[tuple[str, TaskType, float]] = [
    # --- Greeting ---
    ("hello",           TaskType.GREETING,      3.0),
    ("hi ",             TaskType.GREETING,      3.0),
    ("hey",             TaskType.GREETING,      2.5),
    ("good morning",    TaskType.GREETING,      3.0),
    ("good evening",    TaskType.GREETING,      3.0),
    ("how are you",     TaskType.GREETING,      3.0),
    ("what's up",       TaskType.GREETING,      2.5),
    ("sup",             TaskType.GREETING,      2.0),
    ("howdy",           TaskType.GREETING,      2.0),
    ("greetings",       TaskType.GREETING,      2.5),

    # --- Debug ---
    ("traceback",       TaskType.DEBUG,         4.0),
    ("exception",       TaskType.DEBUG,         3.5),
    ("error",           TaskType.DEBUG,         2.5),
    ("stack trace",     TaskType.DEBUG,         4.0),
    ("keyerror",        TaskType.DEBUG,         4.0),
    ("typeerror",       TaskType.DEBUG,         4.0),
    ("valueerror",      TaskType.DEBUG,         4.0),
    ("attributeerror",  TaskType.DEBUG,         4.0),
    ("modulenotfound",  TaskType.DEBUG,         4.0),
    ("segfault",        TaskType.DEBUG,         4.0),
    ("null pointer",    TaskType.DEBUG,         4.0),
    ("why is my",       TaskType.DEBUG,         2.0),
    ("not working",     TaskType.DEBUG,         2.0),
    ("broken",          TaskType.DEBUG,         1.5),

    # --- Coding ---
    ("write a function",TaskType.CODING,        4.0),
    ("write code",      TaskType.CODING,        4.0),
    ("implement",       TaskType.CODING,        3.0),
    ("python",          TaskType.CODING,        2.0),
    ("javascript",      TaskType.CODING,        2.0),
    ("typescript",      TaskType.CODING,        2.0),
    ("sql",             TaskType.CODING,        2.0),
    ("bash",            TaskType.CODING,        2.0),
    ("class ",          TaskType.CODING,        1.5),
    ("function",        TaskType.CODING,        1.5),
    ("import ",         TaskType.CODING,        1.5),
    ("pip install",     TaskType.CODING,        3.0),
    ("npm install",     TaskType.CODING,        3.0),
    ("compile",         TaskType.CODING,        2.0),
    ("syntax",          TaskType.CODING,        2.5),

    # --- Code review ---
    ("review this",     TaskType.CODE_REVIEW,   4.0),
    ("audit",           TaskType.CODE_REVIEW,   3.0),
    ("refactor",        TaskType.CODE_REVIEW,   3.5),
    ("improve my code", TaskType.CODE_REVIEW,   4.0),
    ("code smell",      TaskType.CODE_REVIEW,   4.0),
    ("best practice",   TaskType.CODE_REVIEW,   2.5),
    ("clean up",        TaskType.CODE_REVIEW,   2.0),

    # --- Math ---
    ("calculate",       TaskType.MATH,          4.0),
    ("compute",         TaskType.MATH,          3.5),
    ("integral",        TaskType.MATH,          4.0),
    ("derivative",      TaskType.MATH,          4.0),
    ("equation",        TaskType.MATH,          3.0),
    ("algebra",         TaskType.MATH,          3.5),
    ("geometry",        TaskType.MATH,          3.5),
    ("probability",     TaskType.MATH,          3.5),
    ("statistics",      TaskType.MATH,          3.0),
    ("how much is",     TaskType.MATH,          2.0),
    ("percent",         TaskType.MATH,          2.0),

    # --- Data analysis ---
    ("dataset",         TaskType.DATA_ANALYSIS, 4.0),
    ("dataframe",       TaskType.DATA_ANALYSIS, 4.0),
    ("csv",             TaskType.DATA_ANALYSIS, 3.5),
    ("pandas",          TaskType.DATA_ANALYSIS, 4.0),
    ("plot",            TaskType.DATA_ANALYSIS, 3.0),
    ("chart",           TaskType.DATA_ANALYSIS, 3.0),
    ("trend",           TaskType.DATA_ANALYSIS, 2.5),
    ("correlation",     TaskType.DATA_ANALYSIS, 3.5),
    ("regression",      TaskType.DATA_ANALYSIS, 3.5),
    ("analyse data",    TaskType.DATA_ANALYSIS, 4.0),
    ("analyze data",    TaskType.DATA_ANALYSIS, 4.0),

    # --- Excel ---
    ("excel",           TaskType.EXCEL,         4.0),
    ("spreadsheet",     TaskType.EXCEL,         4.0),
    ("vlookup",         TaskType.EXCEL,         4.0),
    ("pivot table",     TaskType.EXCEL,         4.0),
    ("formula",         TaskType.EXCEL,         3.0),
    ("google sheets",   TaskType.EXCEL,         4.0),
    ("cell reference",  TaskType.EXCEL,         3.5),
    ("sum if",          TaskType.EXCEL,         3.5),

    # --- Summarize ---
    ("summarize",       TaskType.SUMMARIZE,     4.0),
    ("summarise",       TaskType.SUMMARIZE,     4.0),
    ("tl;dr",           TaskType.SUMMARIZE,     4.0),
    ("condense",        TaskType.SUMMARIZE,     3.5),
    ("brief overview",  TaskType.SUMMARIZE,     3.0),
    ("key points",      TaskType.SUMMARIZE,     2.5),
    ("main idea",       TaskType.SUMMARIZE,     2.5),

    # --- Translate ---
    ("translate",       TaskType.TRANSLATE,     4.0),
    ("in french",       TaskType.TRANSLATE,     4.0),
    ("in spanish",      TaskType.TRANSLATE,     4.0),
    ("in arabic",       TaskType.TRANSLATE,     4.0),
    ("in urdu",         TaskType.TRANSLATE,     4.0),
    ("in chinese",      TaskType.TRANSLATE,     4.0),
    ("in german",       TaskType.TRANSLATE,     4.0),

    # --- Creative ---
    ("write a story",   TaskType.CREATIVE,      4.0),
    ("write a poem",    TaskType.CREATIVE,      4.0),
    ("poem",            TaskType.CREATIVE,      3.5),  # catches "short poem", "write me a poem"
    ("poetry",          TaskType.CREATIVE,      3.5),
    ("short story",     TaskType.CREATIVE,      4.0),
    ("haiku",           TaskType.CREATIVE,      4.0),
    ("limerick",        TaskType.CREATIVE,      4.0),
    ("creative",        TaskType.CREATIVE,      3.0),
    ("fiction",         TaskType.CREATIVE,      3.5),
    ("imagine",         TaskType.CREATIVE,      2.0),
    ("write me a",      TaskType.CREATIVE,      1.5),

    # --- Comparison ---
    (" vs ",            TaskType.COMPARISON,    4.0),
    ("versus",          TaskType.COMPARISON,    3.5),
    ("compare",         TaskType.COMPARISON,    4.0),
    ("difference between", TaskType.COMPARISON, 4.0),
    ("better than",     TaskType.COMPARISON,    3.0),
    ("pros and cons",   TaskType.COMPARISON,    3.5),
    ("trade-off",       TaskType.COMPARISON,    3.0),
    ("tradeoff",        TaskType.COMPARISON,    3.0),

    # --- Step-by-step ---
    ("how do i",        TaskType.STEP_BY_STEP,  3.5),
    ("how to",          TaskType.STEP_BY_STEP,  3.0),
    ("walk me through", TaskType.STEP_BY_STEP,  4.0),
    ("step by step",    TaskType.STEP_BY_STEP,  4.0),
    ("guide me",        TaskType.STEP_BY_STEP,  3.0),
    ("tutorial",        TaskType.STEP_BY_STEP,  3.5),

    # --- Factual Q&A ---
    ("what is",         TaskType.FACTUAL_QA,    2.0),
    ("who is",          TaskType.FACTUAL_QA,    2.5),
    ("when did",        TaskType.FACTUAL_QA,    2.5),
    ("where is",        TaskType.FACTUAL_QA,    2.5),
    ("define",          TaskType.FACTUAL_QA,    3.0),
    ("history of",      TaskType.FACTUAL_QA,    3.0),
    ("tell me about",   TaskType.FACTUAL_QA,    2.0),
    ("explain",         TaskType.FACTUAL_QA,    2.0),

    # --- Deep analysis ---
    ("analyse",         TaskType.ANALYSIS,      3.0),
    ("analyze",         TaskType.ANALYSIS,      3.0),
    ("evaluate",        TaskType.ANALYSIS,      3.0),
    ("implications",    TaskType.ANALYSIS,      3.5),
    ("impact of",       TaskType.ANALYSIS,      3.0),
    ("assess",          TaskType.ANALYSIS,      3.0),
    ("critique",        TaskType.ANALYSIS,      3.5),
]


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Output of :class:`IntentClassifier`.

    Attributes
    ----------
    primary       Highest-scoring :class:`TaskType`.
    confidence    Normalised score in [0, 1].  ≥ 0.5 is considered reliable.
    candidates    All (TaskType, score) pairs, sorted descending.
    signals_hit   Raw signal substrings that fired.
    """
    primary:     TaskType
    confidence:  float
    candidates:  list[tuple[TaskType, float]]
    signals_hit: list[str]


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Weighted multi-signal intent classifier.

    Scoring
    -------
    Each matching signal adds its weight to that TaskType's running total.
    The raw scores are normalised to [0, 1] by dividing by the maximum.
    Ties go to the first match in priority order (debug > coding > math …).

    Length penalty
    --------------
    Very short queries (< 3 words) that score only 1–2 signals are penalised
    to avoid over-confident routing on ambiguous fragments.
    """

    # Minimum confidence to trust a classification; below this → GENERAL
    CONFIDENCE_FLOOR: float = 0.15
    # Greeting must be short; long "hi, can you summarise…" is not a greeting
    GREETING_MAX_WORDS: int = 7

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify *query* and return a :class:`ClassificationResult`.

        Args:
            query: Raw user query string.

        Returns:
            ClassificationResult with primary task type and confidence.
        """
        if not query or not query.strip():
            return ClassificationResult(
                primary=TaskType.UNKNOWN,
                confidence=0.0,
                candidates=[],
                signals_hit=[],
            )

        q_lower = query.lower().strip()
        word_count = len(q_lower.split())
        scores: dict[TaskType, float] = {}
        hits: list[str] = []

        for signal, task, weight in _SIGNALS:
            if signal in q_lower:
                scores[task] = scores.get(task, 0.0) + weight
                hits.append(signal)

        if not scores:
            return ClassificationResult(
                primary=TaskType.GENERAL,
                confidence=self.CONFIDENCE_FLOOR,
                candidates=[(TaskType.GENERAL, self.CONFIDENCE_FLOOR)],
                signals_hit=[],
            )

        # Greeting guard: only accept if query is short enough
        if TaskType.GREETING in scores and word_count > self.GREETING_MAX_WORDS:
            scores.pop(TaskType.GREETING)

        # Length penalty for very short ambiguous queries
        if word_count < 3 and len(scores) > 1:
            for k in scores:
                scores[k] *= 0.7

        max_score = max(scores.values())
        norm: list[tuple[TaskType, float]] = sorted(
            ((t, s / max_score) for t, s in scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )

        primary, confidence = norm[0]
        if confidence < self.CONFIDENCE_FLOOR:
            primary = TaskType.GENERAL

        return ClassificationResult(
            primary=primary,
            confidence=round(confidence, 3),
            candidates=norm,
            signals_hit=list(dict.fromkeys(hits)),  # deduplicated, order-preserved
        )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """
    Self-describing specification for one agent.

    Attributes
    ----------
    name          Unique identifier (snake_case).
    description   What the agent does (one sentence).
    task_types    TaskTypes this agent handles.
    requires_rag  Whether this agent benefits from RAG context.
    system_hint   Optional extra instruction injected into the LLM prompt.
    priority      Lower = tried first when multiple agents match.
    """
    name:        str
    description: str
    task_types:  list[TaskType]
    requires_rag: bool = False
    system_hint: str   = ""
    priority:    int   = 10


class AgentRegistry:
    """
    Central registry of all available agents.

    Usage::

        registry = AgentRegistry()
        registry.register(AgentSpec(...))
        spec = registry.resolve(TaskType.MATH)
    """

    def __init__(self) -> None:
        self._agents: list[AgentSpec] = []

    def register(self, spec: AgentSpec) -> "AgentRegistry":
        """Register an agent. Fluent — returns self."""
        if any(a.name == spec.name for a in self._agents):
            raise ValueError(f"Agent '{spec.name}' is already registered.")
        self._agents.append(spec)
        self._agents.sort(key=lambda a: a.priority)
        return self

    def resolve(self, task: TaskType) -> Optional[AgentSpec]:
        """Return the highest-priority agent that handles *task*, or None."""
        for agent in self._agents:
            if task in agent.task_types:
                return agent
        return None

    def all_agents(self) -> list[AgentSpec]:
        return list(self._agents)

    def __len__(self) -> int:
        return len(self._agents)


def _build_default_registry() -> AgentRegistry:
    """Build the default agent registry used when Brain is instantiated."""
    reg = AgentRegistry()
    reg.register(AgentSpec(
        name="greeting_agent",
        description="Responds warmly to conversational openers.",
        task_types=[TaskType.GREETING],
        requires_rag=False,
        system_hint="Be warm, brief, and invite a follow-up question.",
        priority=1,
    ))
    reg.register(AgentSpec(
        name="debug_agent",
        description="Diagnoses errors, tracebacks, and runtime failures.",
        task_types=[TaskType.DEBUG],
        requires_rag=True,
        system_hint=(
            "Identify the root cause first. "
            "Then provide a minimal reproducible fix. "
            "List any caveats or edge cases."
        ),
        priority=2,
    ))
    reg.register(AgentSpec(
        name="coding_agent",
        description="Writes, explains, and fixes source code.",
        task_types=[TaskType.CODING, TaskType.CODE_REVIEW],
        requires_rag=True,
        system_hint=(
            "Provide complete, runnable code. "
            "Add inline comments for non-obvious logic. "
            "State language and version assumptions."
        ),
        priority=3,
    ))
    reg.register(AgentSpec(
        name="math_agent",
        description="Solves mathematical problems step by step.",
        task_types=[TaskType.MATH],
        requires_rag=False,
        system_hint=(
            "Show every calculation step. "
            "State units and assumptions. "
            "Verify the answer at the end."
        ),
        priority=4,
    ))
    reg.register(AgentSpec(
        name="data_agent",
        description="Analyses datasets, statistics, and data trends.",
        task_types=[TaskType.DATA_ANALYSIS],
        requires_rag=True,
        system_hint=(
            "Interpret data distributions and outliers. "
            "Recommend appropriate visualisations. "
            "Call out data quality issues if visible."
        ),
        priority=5,
    ))
    reg.register(AgentSpec(
        name="excel_agent",
        description="Handles spreadsheet formulas, pivot tables, and automation.",
        task_types=[TaskType.EXCEL],
        requires_rag=False,
        system_hint=(
            "Provide exact Excel / Google Sheets syntax. "
            "Explain each argument. "
            "Offer an alternative formula if one exists."
        ),
        priority=6,
    ))
    reg.register(AgentSpec(
        name="summarize_agent",
        description="Condenses long content into concise summaries.",
        task_types=[TaskType.SUMMARIZE],
        requires_rag=True,
        system_hint=(
            "Extract the three most important points. "
            "Preserve key figures and proper nouns. "
            "Never invent facts not present in the source."
        ),
        priority=7,
    ))
    reg.register(AgentSpec(
        name="translate_agent",
        description="Translates text between languages.",
        task_types=[TaskType.TRANSLATE],
        requires_rag=False,
        system_hint=(
            "Provide the translation only (no explanation unless asked). "
            "Note dialectal choices where relevant."
        ),
        priority=8,
    ))
    reg.register(AgentSpec(
        name="creative_agent",
        description="Produces creative writing: stories, poems, scripts.",
        task_types=[TaskType.CREATIVE],
        requires_rag=False,
        system_hint="Prioritise originality and vivid language. Follow any style constraints.",
        priority=9,
    ))
    reg.register(AgentSpec(
        name="comparison_agent",
        description="Compares two or more options with structured trade-off analysis.",
        task_types=[TaskType.COMPARISON],
        requires_rag=True,
        system_hint=(
            "Use a table or labelled sections. "
            "Cover pros, cons, and use-case fit. "
            "End with a clear recommendation if possible."
        ),
        priority=10,
    ))
    reg.register(AgentSpec(
        name="howto_agent",
        description="Provides step-by-step instructions and tutorials.",
        task_types=[TaskType.STEP_BY_STEP],
        requires_rag=True,
        system_hint=(
            "Number every step. "
            "State prerequisites at the top. "
            "Add a troubleshooting note at the end."
        ),
        priority=11,
    ))
    reg.register(AgentSpec(
        name="analysis_agent",
        description="Performs deep multi-faceted reasoning and evaluation.",
        task_types=[TaskType.ANALYSIS, TaskType.FACTUAL_QA],
        requires_rag=True,
        system_hint=(
            "Structure your answer with clear sections. "
            "Distinguish facts from interpretations. "
            "Acknowledge uncertainty explicitly."
        ),
        priority=12,
    ))
    reg.register(AgentSpec(
        name="general_agent",
        description="Catch-all agent for unclassified queries.",
        task_types=[TaskType.GENERAL, TaskType.UNKNOWN],
        requires_rag=False,
        system_hint="Be helpful, honest, and concise.",
        priority=99,
    ))
    return reg


# ---------------------------------------------------------------------------
# Protocols for external dependencies
# ---------------------------------------------------------------------------

@runtime_checkable
class RAGSystem(Protocol):
    """Minimal interface expected from a retrieval system."""
    def retrieve(self, query: str, user_id: str = "global", top_k: int = 3) -> list[Any]: ...


@runtime_checkable
class LLMGenerator(Protocol):
    """Minimal interface expected from an LLM wrapper."""
    def generate(self, prompt: str, system: str = "") -> str: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BrainConfig:
    """
    Controls all tuneable parameters for :class:`Brain`.

    Attributes
    ----------
    max_rag_docs          Max documents retrieved from RAG per query.
    context_char_limit    Max chars of RAG context injected into the prompt.
    confidence_floor      Classification confidence below which → GENERAL.
    greeting_max_words    Max words for a query to be treated as a greeting.
    bad_output_tokens     Extra junk tokens to strip from LLM output.
    rag_user_id           user_id passed to rag_system.retrieve().
    enable_query_rewrite  Rewrite query before RAG retrieval for better recall.
    fallback_answer       Returned when all routes fail and no LLM is available.
    """
    max_rag_docs:         int        = 3
    context_char_limit:   int        = 2000
    confidence_floor:     float      = 0.15
    greeting_max_words:   int        = 7
    bad_output_tokens:    list[str]  = field(default_factory=lambda: [
        "[RAG INIT]", "contentType", "revision", "Primer_Brand",
        "data-color-mode", "BEGIN", "END", "---",
    ])
    rag_user_id:          str        = "global"
    enable_query_rewrite: bool       = True
    fallback_answer:      str        = "I'm sorry, I couldn't generate a response. Please try again."

    def __post_init__(self) -> None:
        if self.max_rag_docs < 1:
            raise ValueError("max_rag_docs must be ≥ 1")
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise ValueError("confidence_floor must be in [0, 1]")


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Rewrites a query into a form that improves RAG recall.

    Transformations applied
    -----------------------
    * Remove first-person filler ("can you", "please", "I need you to")
    * Expand contractions ("don't" → "do not")
    * Append TaskType-specific context hints so that embedding similarity
      pulls the right document cluster.
    """

    _FILLER = re.compile(
        r"\b(please|can you|could you|i need you to|i want you to|"
        r"help me|would you|kindly|just)\b",
        re.I,
    )
    _CONTRACTIONS = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "can't": "cannot", "i'm": "i am",
        "it's": "it is", "that's": "that is", "there's": "there is",
    }
    _TASK_HINTS: dict[TaskType, str] = {
        TaskType.DEBUG:         "error diagnosis fix",
        TaskType.CODING:        "source code implementation",
        TaskType.CODE_REVIEW:   "code quality review refactor",
        TaskType.DATA_ANALYSIS: "data statistics analysis",
        TaskType.EXCEL:         "spreadsheet formula Excel",
        TaskType.MATH:          "mathematical calculation",
        TaskType.COMPARISON:    "comparison trade-off analysis",
        TaskType.STEP_BY_STEP:  "step-by-step tutorial guide",
        TaskType.ANALYSIS:      "in-depth evaluation reasoning",
    }

    def rewrite(self, query: str, task: TaskType) -> str:
        """
        Return a rewritten version of *query* optimised for retrieval.

        Args:
            query: Original user query.
            task:  Classified task type.

        Returns:
            str: Rewritten query (never empty; falls back to original).
        """
        q = query.strip()

        # Expand contractions
        for contraction, expansion in self._CONTRACTIONS.items():
            q = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, q, flags=re.I)

        # Strip filler words
        q = self._FILLER.sub("", q)
        q = re.sub(r"\s{2,}", " ", q).strip()

        # Append task hint (only for RAG-heavy tasks)
        hint = self._TASK_HINTS.get(task, "")
        if hint and hint.split()[0].lower() not in q.lower():
            q = f"{q} [{hint}]"

        return q or query  # never return empty


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

@dataclass
class ScoredDoc:
    content:  str
    score:    float  # relevance score, higher = better
    source:   str    = ""


class ContextBuilder:
    """
    Builds a clean, de-duplicated, relevance-ranked context block
    from raw RAG documents.
    """

    def __init__(self, config: BrainConfig) -> None:
        self._config = config

    def build(self, docs: list[Any]) -> tuple[str, int]:
        """
        Process *docs* into a context string.

        Args:
            docs: Raw documents from the RAG system.  Each doc is expected to
                  have a ``.content`` attribute and optionally ``.score`` and
                  ``.source``.  Plain strings are also accepted.

        Returns:
            Tuple of (context_string, doc_count_used).
        """
        scored: list[ScoredDoc] = []
        seen_hashes: set[int] = set()

        for doc in docs:
            content = doc.content if hasattr(doc, "content") else str(doc)
            score   = getattr(doc, "score", 0.5)
            source  = getattr(doc, "source", "")

            content = content.strip()

            # Skip junk docs
            if len(content) < 20:
                continue
            if any(bad.lower() in content.lower() for bad in self._config.bad_output_tokens):
                continue
            if "contenttype" in content.lower():
                continue

            # Deduplicate by content hash
            h = hash(content[:200])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            scored.append(ScoredDoc(content=content, score=score, source=source))

        # Sort by relevance descending
        scored.sort(key=lambda d: d.score, reverse=True)

        # Build context block, respecting char limit
        parts: list[str] = []
        total = 0
        for doc in scored:
            if total >= self._config.context_char_limit:
                break
            remaining = self._config.context_char_limit - total
            chunk = self._truncate(doc.content, remaining)
            if doc.source:
                parts.append(f"[Source: {doc.source}]\n{chunk}")
            else:
                parts.append(chunk)
            total += len(chunk)

        context = "\n\n".join(parts)
        return context, len(scored)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit]
        last_period = cut.rfind(".")
        if last_period > limit * 0.75:
            cut = cut[: last_period + 1]
        return cut + " […]"


# ---------------------------------------------------------------------------
# Output cleaner
# ---------------------------------------------------------------------------

class OutputCleaner:
    """
    Strips system artefacts, log lines, and junk tokens from LLM output.
    Extensible via the ``extra_bad_tokens`` parameter.
    """

    _LOG_PATTERNS = re.compile(
        r"^\s*(\[rag|indexed:|loading knowledge|retriev|embedding)",
        re.I | re.M,
    )
    _EXCESS_NEWLINES = re.compile(r"\n{3,}")
    _TRAILING_FENCE  = re.compile(r"```\s*$", re.M)

    def __init__(self, bad_tokens: list[str]) -> None:
        self._bad_tokens = bad_tokens

    def clean(self, text: str) -> str:
        """
        Clean *text* and return the sanitised string.

        Args:
            text: Raw LLM output.

        Returns:
            str: Cleaned output, never None.
        """
        if not text:
            return ""

        # Remove known bad tokens
        for token in self._bad_tokens:
            text = text.replace(token, "")

        # Remove lines that look like internal logs
        lines = [
            line for line in text.splitlines()
            if not self._LOG_PATTERNS.match(line)
        ]
        text = "\n".join(lines)

        # Collapse excessive blank lines
        text = self._EXCESS_NEWLINES.sub("\n\n", text)

        # Remove dangling code fences
        text = self._TRAILING_FENCE.sub("", text)

        return text.strip()


# ---------------------------------------------------------------------------
# Routing decision + trace
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    """The resolved routing plan before execution."""
    agent:          AgentSpec
    classification: ClassificationResult
    rewritten_query: str
    use_rag:        bool


@dataclass
class ExecutionTrace:
    """Full audit trail of a single Brain.answer() call."""
    query:           str
    classification:  ClassificationResult
    agent_name:      str
    rewritten_query: str
    rag_used:        bool
    rag_doc_count:   int
    latency_ms:      float
    error:           Optional[str]  = None


@dataclass
class BrainResult:
    """Return type of :meth:`Brain.answer`."""
    answer: str
    trace:  ExecutionTrace


# ---------------------------------------------------------------------------
# Middleware types
# ---------------------------------------------------------------------------

PreHook  = Callable[[str, BrainConfig], str]          # (query, config) → query
PostHook = Callable[[str, ExecutionTrace], str]        # (answer, trace) → answer


# ---------------------------------------------------------------------------
# Brain — main orchestrator
# ---------------------------------------------------------------------------

class Brain:
    """
    Universal Q&A orchestrator.

    Pipeline
    --------
    1. Pre-hooks (sanitise, inject context, rate-limit, …)
    2. Classify intent → ClassificationResult
    3. Resolve agent via AgentRegistry
    4. Optionally rewrite query for RAG
    5. Retrieve RAG docs if agent.requires_rag
    6. Build context block
    7. Generate answer (agent system_hint + context + query)
    8. Clean output
    9. Post-hooks (PII redaction, logging, …)
    10. Return BrainResult

    Usage::

        brain = Brain(rag_system=my_rag, llm=my_llm)
        result = brain.answer("Why is my pandas merge returning NaN?")
        print(result.answer)
        print(result.trace)
    """

    def __init__(
        self,
        rag_system:  Optional[RAGSystem]   = None,
        llm:         Optional[LLMGenerator]= None,
        config:      Optional[BrainConfig] = None,
        registry:    Optional[AgentRegistry] = None,
    ) -> None:
        self._rag    = rag_system
        self._llm    = llm
        self._config = config or BrainConfig()
        self._registry   = registry or _build_default_registry()
        self._classifier = IntentClassifier()
        self._rewriter   = QueryRewriter()
        self._cleaner    = OutputCleaner(self._config.bad_output_tokens)
        self._ctx_builder = ContextBuilder(self._config)
        self._pre_hooks:  list[PreHook]  = []
        self._post_hooks: list[PostHook] = []

    # ------------------------------------------------------------------
    # Middleware registration
    # ------------------------------------------------------------------

    def add_pre_hook(self, hook: PreHook) -> "Brain":
        """
        Register a pre-processing hook.  Hooks run in registration order.

        Args:
            hook: Callable ``(query: str, config: BrainConfig) → str``.

        Returns:
            Brain: self (fluent).
        """
        self._pre_hooks.append(hook)
        return self

    def add_post_hook(self, hook: PostHook) -> "Brain":
        """
        Register a post-processing hook.  Hooks run in registration order.

        Args:
            hook: Callable ``(answer: str, trace: ExecutionTrace) → str``.

        Returns:
            Brain: self (fluent).
        """
        self._post_hooks.append(hook)
        return self

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def answer(self, query: str) -> BrainResult:
        """
        Process *query* end-to-end and return a :class:`BrainResult`.

        Args:
            query: Raw user question.

        Returns:
            BrainResult: Contains the final answer and a full ExecutionTrace.
        """
        t0 = time.perf_counter()

        # 1. Pre-hooks
        for hook in self._pre_hooks:
            query = hook(query, self._config)

        # 2. Classify
        clf = self._classifier.classify(query)
        logger.debug("Classified '%s' → %s (confidence=%.2f)", query[:60], clf.primary, clf.confidence)

        # 3. Resolve agent
        agent = self._registry.resolve(clf.primary)
        if agent is None:
            agent = self._registry.resolve(TaskType.GENERAL)

        # Greeting shortcut — no LLM needed
        if clf.primary == TaskType.GREETING:
            answer = "Hello! How can I help you today?"
            trace  = ExecutionTrace(
                query=query, classification=clf, agent_name=agent.name,
                rewritten_query=query, rag_used=False, rag_doc_count=0,
                latency_ms=_ms(t0),
            )
            return BrainResult(answer=answer, trace=trace)

        # 4. Query rewrite
        rewritten = (
            self._rewriter.rewrite(query, clf.primary)
            if self._config.enable_query_rewrite
            else query
        )

        # 5–6. RAG retrieval + context
        context    = ""
        rag_used   = False
        doc_count  = 0
        error_note = None

        if agent.requires_rag and self._rag is not None:
            try:
                docs = self._rag.retrieve(
                    rewritten,
                    user_id=self._config.rag_user_id,
                    top_k=self._config.max_rag_docs,
                )
                context, doc_count = self._ctx_builder.build(docs)
                rag_used = doc_count > 0
            except Exception as exc:
                error_note = f"RAG retrieval failed: {exc}"
                logger.warning(error_note)
                # Continue without RAG rather than dying

        # 7. Generate
        answer = self._generate(query, context, agent)

        # 8. Clean
        answer = self._cleaner.clean(answer)

        if not answer:
            answer = self._config.fallback_answer

        # 9. Post-hooks
        trace = ExecutionTrace(
            query=query, classification=clf, agent_name=agent.name,
            rewritten_query=rewritten, rag_used=rag_used, rag_doc_count=doc_count,
            latency_ms=_ms(t0), error=error_note,
        )
        for hook in self._post_hooks:
            answer = hook(answer, trace)

        logger.info(
            "answered | agent=%s rag=%s docs=%d conf=%.2f latency=%.0fms",
            agent.name, rag_used, doc_count, clf.confidence, trace.latency_ms,
        )

        return BrainResult(answer=answer, trace=trace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, query: str, context: str, agent: AgentSpec) -> str:
        """Build the prompt and call the LLM, or fall back gracefully."""
        if self._llm is None:
            # No LLM configured — return context or fallback
            return context if context else self._config.fallback_answer

        system = agent.system_hint or "Answer the user's question clearly and concisely."

        if context:
            prompt = (
                f"<context>\n{context}\n</context>\n\n"
                f"Question: {query}"
            )
        else:
            prompt = query

        return self._llm.generate(prompt, system=system)

    def register_agent(self, spec: AgentSpec) -> "Brain":
        """Register an additional agent at runtime. Fluent."""
        self._registry.register(spec)
        return self


# ---------------------------------------------------------------------------
# Standalone helpers  (backwards-compatible surface)
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> TaskType:
    """
    Convenience wrapper — returns the primary :class:`TaskType` for *query*.
    For full confidence and candidates, use :class:`IntentClassifier` directly.
    """
    return IntentClassifier().classify(query).primary


def clean_output(text: str, extra_bad_tokens: Optional[list[str]] = None) -> str:
    """
    Convenience wrapper — cleans *text* using the default bad-token list.

    Args:
        text:             Raw LLM output.
        extra_bad_tokens: Additional tokens to strip.

    Returns:
        str: Cleaned output.
    """
    tokens = BrainConfig().bad_output_tokens + (extra_bad_tokens or [])
    return OutputCleaner(tokens).clean(text)


def should_use_rag(task: TaskType) -> bool:
    """Return True if the default registry's agent for *task* requires RAG."""
    reg  = _build_default_registry()
    spec = reg.resolve(task)
    return spec.requires_rag if spec else False


def build_context(docs: list[Any], max_chars: int = 2000) -> str:
    """
    Convenience wrapper around :class:`ContextBuilder`.

    Args:
        docs:      Raw RAG documents.
        max_chars: Max chars of context to return.

    Returns:
        str: Context string.
    """
    cfg = BrainConfig(context_char_limit=max_chars)
    ctx, _ = ContextBuilder(cfg).build(docs)
    return ctx


def answer_user(
    query: str,
    rag_system: Optional[Any] = None,
    llm_generate: Optional[Callable[[str], str]] = None,
) -> tuple[str, bool]:
    """
    Backwards-compatible shim.  Wraps :class:`Brain` with a callable LLM.

    Args:
        query:        User question.
        rag_system:   Optional RAG system with a ``.retrieve()`` method.
        llm_generate: Optional callable ``(prompt: str) → str``.

    Returns:
        Tuple of (answer_str, rag_used_bool).
    """
    # Adapt plain callable to LLMGenerator protocol
    llm_adapter: Optional[LLMGenerator] = None
    if llm_generate is not None:
        class _Adapter:
            def generate(self, prompt: str, system: str = "") -> str:
                return llm_generate(prompt)
        llm_adapter = _Adapter()

    brain  = Brain(rag_system=rag_system, llm=llm_adapter)
    result = brain.answer(query)
    return result.answer, result.trace.rag_used


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _ms(t0: float) -> float:
    """Elapsed milliseconds since *t0* (from time.perf_counter())."""
    return round((time.perf_counter() - t0) * 1000, 2)
