"""
formatter.py — Pro-level output formatting utilities.

Handles bullet extraction, text normalisation, truncation, multi-format
rendering, and structured output validation for LLM responses.

Supports:
  • All common bullet styles: -, *, •, >, 1., (1), [1]
  • Numbered list detection and re-numbering
  • Plain-prose → bullet conversion
  • JSON / Markdown / plain-text rendering
  • Unicode normalisation and control-char sanitisation
  • Configurable truncation with smart sentence-boundary detection
"""

from __future__ import annotations

import html as _html
import json as _json
import re
import unicodedata
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Union


# ───────────────────────────────────────────────────────────���─────────────────
# Enums & Config
# ─────────────────────────────────────────────────────────────────────────────

class OutputFormat(str, Enum):
    """Supported rendering targets."""
    PLAIN    = "plain"
    MARKDOWN = "markdown"
    HTML     = "html"
    NUMBERED = "numbered"
    JSON     = "json"


@dataclass
class FormatConfig:
    """Controls all formatting behaviour."""
    max_points:        int                        = 5
    max_line_length:   int                        = 0
    output_format:     OutputFormat               = OutputFormat.PLAIN
    deduplicate:       bool                       = True
    strip_empty:       bool                       = True
    normalise_unicode: bool                       = True
    min_line_length:   int                        = 3
    transform:         Optional[Callable[[str], str]] = field(
        default=None, repr=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns (Compiled Once)
# ───────────────────────────────────────────────────────────────��─────────────

_BULLET_RE = re.compile(
    r"""
    ^                       # start of line
    \s*                     # optional leading whitespace
    (?:
        [-–—•*›»▸▶○●◆◇]     # symbolic bullets
      | (?:\d+|[a-zA-Z])    # number or letter …
        [\.\)\]:\-]         # … followed by . ) ] : -
      | \((?:\d+|[a-zA-Z])\)  # (1) or (a) style
      | \[(?:\d+|[a-zA-Z])\]  # [1] or [a] style
    )
    \s*                     # optional space after marker
    """,
    re.VERBOSE,
)

_NUMBERED_RE = re.compile(
    r"^\s*(?:\d+|[a-zA-Z])[\.\)\]\:\-]\s*"
)

# ✅ FIX: Cache section header regex
_SECTION_HEADER_RE = re.compile(r"^#{1,4}\s+(.+)$")


def is_bullet_line(line: str) -> bool:
    """Return True if *line* starts with any recognised bullet marker."""
    return bool(_BULLET_RE.match(line))


def strip_bullet_marker(line: str) -> str:
    """Remove the leading bullet/number marker and surrounding whitespace."""
    cleaned = _BULLET_RE.sub("", line)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def sanitise(text: str) -> str:
    """Strip ASCII control characters and apply NFKC Unicode normalisation."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def truncate_line(line: str, max_length: int, ellipsis: str = "…") -> str:
    """Truncate *line* to *max_length* characters, breaking at word boundary."""
    if max_length <= 0 or len(line) <= max_length:
        return line
    cut = line[: max_length - len(ellipsis)]
    last_space = cut.rfind(" ")
    if last_space > max_length * 0.6:
        cut = cut[:last_space]
    return cut.rstrip() + ellipsis


def truncate_context(
    text: str,
    max_chars: int = 2000,
    ellipsis: str = " […]",
) -> str:
    """Truncate *text* to *max_chars*, preferring a sentence boundary."""
    if not text or len(text) <= max_chars:
        return text or ""
    cut        = text[:max_chars]
    last_stop  = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_stop > max_chars * 0.7:
        cut = cut[: last_stop + 1]
    return cut + ellipsis


# ─────────────────────────────────────────────────────────────────────────────
# Core Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_bullets(
    text: str,
    config: Optional[FormatConfig] = None,
) -> List[str]:
    """Extract and clean bullet points from *text*."""
    if not isinstance(text, str):
        raise TypeError(f"extract_bullets() expects str, got {type(text).__name__!r}.")

    cfg = config or FormatConfig()
    if cfg.max_points < 1:
        raise ValueError(f"max_points must be ≥ 1, got {cfg.max_points}.")

    if cfg.normalise_unicode:
        text = sanitise(text)

    lines = text.splitlines()
    bullet_lines = [ln for ln in lines if is_bullet_line(ln)]

    if bullet_lines:
        raw_points = [strip_bullet_marker(ln) for ln in bullet_lines]
    else:
        raw_points = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

    points: List[str] = []
    seen:   set       = set()

    for point in raw_points:
        if cfg.strip_empty and not point:
            continue
        if len(point) < cfg.min_line_length:
            continue
        if cfg.max_line_length > 0:
            point = truncate_line(point, cfg.max_line_length)
        if cfg.transform:
            point = cfg.transform(point)
        if cfg.deduplicate:
            key = point.lower().strip()
            if key in seen:
                continue
            seen.add(key)
        points.append(point)

    return points[: cfg.max_points]


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_plain(points: List[str]) -> str:
    return "\n".join(f"- {p}" for p in points)


def _render_markdown(points: List[str]) -> str:
    return "\n".join(f"- {p}" for p in points)


def _render_numbered(points: List[str]) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))


def _render_html(points: List[str]) -> str:
    """✅ FIX: HTML-escape all point content for XSS safety."""
    items = "\n".join(f"  <li>{_html.escape(p, quote=True)}</li>" for p in points)
    return f"<ul>\n{items}\n</ul>"


def _render_json(points: List[str]) -> str:
    return _json.dumps(points, ensure_ascii=False, indent=2)


# ✅ FIX: Add proper type hint
_RENDERERS: Dict[OutputFormat, Callable[[List[str]], str]] = {
    OutputFormat.PLAIN:    _render_plain,
    OutputFormat.MARKDOWN: _render_markdown,
    OutputFormat.NUMBERED: _render_numbered,
    OutputFormat.HTML:     _render_html,
    OutputFormat.JSON:     _render_json,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def format_output(
    text: str,
    max_points: int = 5,
    output_format: Union[OutputFormat, str] = OutputFormat.PLAIN,
    config: Optional[FormatConfig] = None,
) -> str:
    """
    Extract bullet points from *text* and render them in the requested format.

    This is the primary public entry point.

    Args:
        text:          Raw LLM response or any text.
        max_points:    Maximum points to include (ignored if *config* provided).
        output_format: One of ``plain``, ``markdown``, ``html``, ``numbered``,
                       ``json`` — or an :class:`OutputFormat` enum value.
        config:        Full :class:`FormatConfig` (overrides *max_points*).

    Returns:
        Formatted string.
    """
    if config is None:
        config = FormatConfig(max_points=max_points)

    if isinstance(output_format, str):
        try:
            output_format = OutputFormat(output_format.lower())
        except ValueError:
            valid = ", ".join(f.value for f in OutputFormat)
            raise ValueError(
                f"Unknown output_format {output_format!r}. Valid: {valid}"
            )

    effective_cfg = _dc_replace(config, output_format=output_format)
    points = extract_bullets(text, effective_cfg)

    # ✅ FIX: Return format-appropriate empty value
    if not points:
        if output_format == OutputFormat.HTML:
            return "<ul>\n</ul>"
        elif output_format == OutputFormat.JSON:
            return "[]"
        return ""

    renderer = _RENDERERS[output_format]
    return renderer(points)


def enforce_bullets(
    text: str,
    max_points: int = 5,
    config: Optional[FormatConfig] = None,
) -> str:
    """
    Backward-compatible wrapper: extract and return plain-text bullet lines.

    Args:
        text:       Input text.
        max_points: Maximum bullet points to return.
        config:     Optional full config (overrides *max_points*).

    Returns:
        Newline-joined bullet lines prefixed with ``-``.
    """
    if config is None:
        cfg = FormatConfig(max_points=max_points, output_format=OutputFormat.PLAIN)
    else:
        cfg = _dc_replace(config, output_format=OutputFormat.PLAIN)
    return format_output(text, config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_bullet_response(
    text: str,
    expected_points: int = 5,
    strict: bool = False,
) -> Dict[str, Union[bool, int, List[str]]]:
    """
    Validate that an LLM response contains the expected number of bullet points.

    Args:
        text:             LLM response string.
        expected_points:  Number of bullet points expected.
        strict:           If True, exact count required; otherwise ≥ 1 is ok.

    Returns:
        Dict with keys ``valid`` (bool), ``found`` (int), ``issues`` (list[str]).
    """
    # ✅ FIX: More specific error messages
    if not isinstance(text, str):
        return {
            "valid": False,
            "found": 0,
            "issues": ["Input is not a string."],
        }
    if not text.strip():
        return {
            "valid": False,
            "found": 0,
            "issues": ["Empty string provided."],
        }

    points = extract_bullets(
        text,
        FormatConfig(max_points=100, min_line_length=1),
    )
    found  = len(points)
    issues = []

    if strict and found != expected_points:
        issues.append(
            f"Expected exactly {expected_points} point(s), found {found}."
        )
    elif not strict and found < 1:
        issues.append("No bullet points detected.")

    if found > expected_points * 2:
        issues.append(
            f"Unusually high point count ({found}); possible formatting issue."
        )

    return {"valid": len(issues) == 0, "found": found, "issues": issues}


def extract_sections(text: str) -> Dict[str, List[str]]:
    """
    Parse a structured LLM response with Markdown-style ``## Section`` headers
    into a dict mapping section name → list of bullet points.

    Args:
        text: Multi-section LLM response.

    Returns:
        ``{"Section Name": ["point 1", "point 2", …], …}``
    """
    sections: Dict[str, List[str]] = {}
    current_header = "__default__"
    current_lines:  List[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        # ✅ FIX: Use pre-compiled regex
        header_match = _SECTION_HEADER_RE.match(stripped)
        if header_match:
            if current_lines:
                sections[current_header] = extract_bullets(
                    "\n".join(current_lines),
                    FormatConfig(max_points=50),
                )
            current_header = header_match.group(1).strip()
            current_lines  = []
        else:
            current_lines.append(raw_line)

    if current_lines:
        extracted = extract_bullets(
            "\n".join(current_lines),
            FormatConfig(max_points=50),
        )
        if extracted:
            sections[current_header] = extracted

    sections.pop("__default__", None)
    return sections


# ────────────────────────────────────────────────────���────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SAMPLE = dedent("""
        Here are the key findings:
        - Revenue grew by 18% YoY
        • Customer churn dropped to 4.2%
        * Three new markets were entered
        1. Headcount increased by 22 FTEs
        (5) EBITDA margin held at 31%
        - Revenue grew by 18% YoY   ← duplicate, should be removed
    """).strip()

    print("=== PLAIN ===")
    print(format_output(_SAMPLE, max_points=5))

    print("\n=== NUMBERED ===")
    print(format_output(_SAMPLE, output_format="numbered"))

    print("\n=== HTML ===")
    print(format_output(_SAMPLE, output_format="html"))

    print("\n=== JSON ===")
    print(format_output(_SAMPLE, output_format="json"))

    print("\n=== VALIDATION ===")
    print(validate_bullet_response(_SAMPLE, expected_points=5))

    print("\n=== SECTIONS ===")
    _MULTI = "## Strengths\n- Fast\n- Scalable\n## Weaknesses\n- Expensive\n- Complex"
    print(extract_sections(_MULTI))

    print("\n=== TRUNCATE CONTEXT ===")
    long_text = "The quick brown fox. " * 200
    print(repr(truncate_context(long_text, max_chars=100)))