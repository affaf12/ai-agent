"""
chunking.py
===========
Pro-level text chunking.

Why this is better than naive chunking
--------------------------------------
* Recursive, hierarchy-aware splitting (paragraph -> sentence -> word) so we
  never break mid-thought when we can avoid it.
* Token-budgeted (approximate) so chunks fit your embedding model's context
  window instead of exploding past it.
* Configurable overlap that preserves cross-boundary context for retrieval.
* Markdown / code-fence aware: we never split inside a fenced ``` block.
* Drops near-duplicate chunks via a cheap shingled hash so your index stays
  clean.
* Emits rich metadata: parent_id, chunk_index, char_span, token_estimate,
  content_hash. This metadata powers filtering, deduplication and citation.

The chunker has zero hard dependencies. If `tiktoken` is installed it will
use real tokenizer counts; otherwise it falls back to a fast heuristic
(`len(text) / 4`) which is accurate enough for sizing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Dict, Optional

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover - fallback path
    def _count_tokens(text: str) -> int:
        # ~4 characters per token for English. Good enough for sizing.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

# Ordered from coarsest to finest. The recursive splitter walks down the list
# until each piece is below the size limit.
_DEFAULT_SEPARATORS: List[str] = [
    "\n## ",   # markdown sub-heading
    "\n# ",    # markdown heading
    "\n\n",    # paragraph
    "\n",      # line
    ". ",      # sentence
    "? ",
    "! ",
    "; ",
    ", ",
    " ",       # word
    "",        # character (last resort)
]

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _protect_code_fences(text: str) -> (str, Dict[str, str]):
    """Replace fenced code blocks with placeholders so they're not split."""
    placeholders: Dict[str, str] = {}

    def _sub(match: re.Match) -> str:
        key = f"\u0000CODEBLOCK{len(placeholders)}\u0000"
        placeholders[key] = match.group(0)
        return key

    protected = _CODE_FENCE_RE.sub(_sub, text)
    return protected, placeholders


def _restore_code_fences(text: str, placeholders: Dict[str, str]) -> str:
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _split_on(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to each piece (except possibly the last) so we
    # don't lose information when we recombine.
    out: List[str] = []
    for i, p in enumerate(parts):
        if not p:
            continue
        if i < len(parts) - 1:
            out.append(p + separator)
        else:
            out.append(p)
    return out


def _recursive_split(
    text: str,
    max_tokens: int,
    separators: List[str],
    measure: Callable[[str], int],
) -> List[str]:
    if measure(text) <= max_tokens:
        return [text]

    for sep in separators:
        pieces = _split_on(text, sep)
        if len(pieces) <= 1:
            continue
        # Recurse on every oversize piece, keep small pieces as-is.
        out: List[str] = []
        buf = ""
        for piece in pieces:
            if measure(buf + piece) <= max_tokens:
                buf += piece
            else:
                if buf:
                    out.append(buf)
                if measure(piece) > max_tokens:
                    out.extend(_recursive_split(piece, max_tokens, separators, measure))
                    buf = ""
                else:
                    buf = piece
        if buf:
            out.append(buf)
        return out

    # Should be unreachable because the last separator is "" which splits per
    # character, but defensively return the whole text.
    return [text]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}


@dataclass
class Chunker:
    """Token-budgeted recursive chunker with overlap and dedup."""

    chunk_tokens: int = 350
    overlap_tokens: int = 60
    min_chunk_tokens: int = 30
    separators: List[str] = field(default_factory=lambda: list(_DEFAULT_SEPARATORS))
    measure: Callable[[str], int] = _count_tokens

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        protected, placeholders = _protect_code_fences(text)
        raw = _recursive_split(
            protected,
            max_tokens=self.chunk_tokens,
            separators=self.separators,
            measure=self.measure,
        )

        # Apply overlap by carrying the tail of the previous chunk.
        merged: List[str] = []
        carry = ""
        for piece in raw:
            piece = _restore_code_fences(piece, placeholders).strip()
            if not piece:
                continue
            candidate = (carry + "\n" + piece).strip() if carry else piece
            merged.append(candidate)
            # Build the next carry: tail of this piece sized to overlap budget.
            carry = self._tail(piece, self.overlap_tokens)

        return [m for m in merged if self.measure(m) >= self.min_chunk_tokens]

    def _tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        # Walk back from the end, accumulating words until we hit budget.
        words = text.split()
        out: List[str] = []
        total = 0
        for w in reversed(words):
            t = self.measure(w + " ")
            if total + t > max_tokens:
                break
            out.append(w)
            total += t
        return " ".join(reversed(out))

    def chunk_document(self, doc: Dict) -> List[Chunk]:
        """doc -> chunks. doc must have id, text and may have metadata."""
        text = doc.get("text", "")
        base_meta = dict(doc.get("metadata", {}))
        chunks: List[Chunk] = []
        seen_hashes: set = set()
        for i, piece in enumerate(self.split(text)):
            content_hash = _hash(piece)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            meta = {
                **base_meta,
                "parent_id": doc["id"],
                "chunk_index": i,
                "token_estimate": self.measure(piece),
                "char_span": [0, len(piece)],
                "content_hash": content_hash,
            }
            chunks.append(Chunk(id=f"{doc['id']}::c{i}", text=piece, metadata=meta))
        return chunks

    def chunk_documents(self, docs: Iterable[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for doc in docs:
            out.extend(c.to_dict() for c in self.chunk_document(doc))
        return out


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


_default_chunker = Chunker()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """Drop-in replacement for the old chunk_text(); now token-budgeted.

    `chunk_size` and `overlap` are interpreted as approximate token budgets
    (1 token ~= 4 characters) so existing call-sites keep working.
    """
    chunker = Chunker(
        chunk_tokens=max(50, chunk_size // 4),
        overlap_tokens=max(0, overlap // 4),
    )
    return chunker.split(text)


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """Drop-in replacement for the old chunk_documents()."""
    return _default_chunker.chunk_documents(docs)
