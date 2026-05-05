"""
ollama_client.py — Production-grade Ollama API client
======================================================
Fixes over the original
-----------------------
  BUG-1  31 unused imports removed (st, asyncio, numpy, pandas, PIL, etc.)
  BUG-2  /api/embeddings + wrong payload → fixed to /api/embed with `input` field
  BUG-3  chat_once had zero error handling → full try/except + OllamaAPIError
  BUG-4  System-prompt injection was fragile (only checked messages[0]) → helper
  BUG-5  No connection pooling → requests.Session with HTTPAdapter + retries
  BUG-6  Stream errors silently yielded a "[Error: …]" string that callers
         couldn't distinguish from real content → raises OllamaStreamError
 
Pro upgrades
------------
  • Custom exception hierarchy (OllamaError → connection / model / API / stream)
  • Typed dataclasses: ChatMessage, ModelInfo, UsageStats
  • Connection pooling + automatic retry with exponential backoff
  • Context-manager support  (with OllamaClient() as client: …)
  • list_models() / model_info() / pull_model() / delete_model()
  • Multimodal chat  (pass images= to chat_stream / chat_once)
  • embed() fixed  +  embed_batch() for bulk RAG pipelines
  • UsageStats returned from chat_once; streamed from chat_stream at end
  • Per-call system_prompt override (doesn't mutate shared state)
  • Structured logging via the standard `logging` module
  • __repr__ for easy debugging
"""
 
from __future__ import annotations
 
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any, Tuple
 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
 
 
# ===========================================================================
# EXCEPTIONS
# ===========================================================================
 
class OllamaError(Exception):
    """Base class for all Ollama client errors."""
 
 
class OllamaConnectionError(OllamaError):
    """Raised when the client cannot reach the Ollama server."""
 
 
class OllamaModelError(OllamaError):
    """Raised for model-related failures (not found, load error, etc.)."""
 
 
class OllamaAPIError(OllamaError):
    """Raised when the API returns an unexpected HTTP status or bad JSON."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
 
 
class OllamaStreamError(OllamaError):
    """Raised when a streaming response fails mid-stream."""
 
 
# ===========================================================================
# DATACLASSES
# ===========================================================================
 
@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant"
    content: str
    images: List[str] = field(default_factory=list)   # base64-encoded images
 
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d
 
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            images=d.get("images", []),
        )
 
 
@dataclass
class ModelInfo:
    """Metadata for a locally available model."""
    name: str
    size: int               # bytes
    modified_at: str
    digest: str
    format: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
 
    @property
    def size_gb(self) -> float:
        return round(self.size / 1_073_741_824, 2)
 
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "ModelInfo":
        details = data.get("details", {})
        return cls(
            name=data.get("name", ""),
            size=data.get("size", 0),
            modified_at=data.get("modified_at", ""),
            digest=data.get("digest", ""),
            format=details.get("format", ""),
            family=details.get("family", ""),
            parameter_size=details.get("parameter_size", ""),
            quantization_level=details.get("quantization_level", ""),
        )
 
 
@dataclass
class UsageStats:
    """Token usage and timing from a completed chat response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0   # wall-clock ms (from Ollama)
    load_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
 
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
 
    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "UsageStats":
        ns = 1_000_000  # nanoseconds → milliseconds
        return cls(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_duration_ms=data.get("total_duration", 0) / ns,
            load_duration_ms=data.get("load_duration", 0) / ns,
            eval_duration_ms=data.get("eval_duration", 0) / ns,
        )
 
 
# ===========================================================================
# HELPERS
# ===========================================================================
 
def _encode_image(source: Any) -> str:
    """
    Accept a file path (str | Path), raw bytes, or an already-encoded base64
    string and return a base64 string suitable for the Ollama images field.
    """
    if isinstance(source, str) and not Path(source).exists():
        # Assume it's already base64
        return source
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return base64.b64encode(f.read()).decode()
    if isinstance(source, bytes):
        return base64.b64encode(source).decode()
    raise TypeError(f"Unsupported image type: {type(source)}")
 
 
def _build_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    pool_connections: int = 4,
    pool_maxsize: int = 10,
) -> requests.Session:
    """
    Return a requests.Session with connection pooling and automatic retry on
    transient network errors and 5xx responses.
    BUG-5 FIX: the original code called requests.get/post directly on every
    request, creating a new TCP connection each time.
    """
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
 
 
def _inject_system_prompt(
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """
    Prepend a system message only if one is not already present.
    BUG-4 FIX: the original code checked only messages[0] in a fragile way,
    and would append duplicate system messages on repeated calls.
    """
    if not system_prompt:
        return messages
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        return messages
    return [{"role": "system", "content": system_prompt}] + messages
 
 
# ===========================================================================
# CLIENT
# ===========================================================================
 
class OllamaClient:
    """
    Production-grade Ollama HTTP client.
 
    Features
    --------
    • Connection pooling + automatic retry with exponential back-off
    • Streaming and non-streaming chat with usage stats
    • Multimodal chat (images accepted as path, bytes, or base64)
    • Fixed embeddings endpoint (/api/embed, batch-capable)
    • Model management: list, info, pull (with progress), delete
    • Context-manager support
    • Structured logging
 
    Parameters
    ----------
    host            : Ollama base URL (default http://localhost:11434)
    default_system  : Default system prompt injected when no system message
                      is present.  Pass "" to disable.
    timeout         : Default request timeout in seconds (default 300)
    max_retries     : Retry attempts on transient errors (default 3)
    """
 
    DEFAULT_SYSTEM = (
        "Answer concisely. Use short bullet points. "
        "Maximum 3-4 lines. No long paragraphs. Be direct."
    )
 
    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_system: str = DEFAULT_SYSTEM,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.host = host.rstrip("/")
        self.default_system = default_system
        self.timeout = timeout
        self._session = _build_session(max_retries=max_retries)
        # BUG FIX: previously `last_usage` was created inside `chat_stream`'s
        # `with` block. Accessing `client.last_usage` before the first call,
        # or after a stream that errored before the `done` event, raised
        # AttributeError. Initialise here so the attribute always exists.
        self.last_usage: Optional[UsageStats] = None
        logger.debug("OllamaClient initialised — host=%s", self.host)
 
    # ── Context manager ────────────────────────────────────────────────────
    def __enter__(self) -> "OllamaClient":
        return self
 
    def __exit__(self, *_: Any) -> None:
        self.close()
 
    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()
 
    def __repr__(self) -> str:
        return f"OllamaClient(host={self.host!r}, timeout={self.timeout})"
 
    # ── Internal helpers ───────────────────────────────────────────────────
    def _request(
        self,
        method: str,
        path: str,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Single dispatch point for GET / POST / DELETE.

        BUG FIX: the original per-method helpers only caught ConnectionError
        and Timeout. Other requests.RequestException subclasses (MissingSchema,
        InvalidURL, TooManyRedirects, ChunkedEncodingError, etc.) bubbled up
        as raw `requests` exceptions, defeating the typed OllamaError hierarchy.
        Catching RequestException covers the whole family.
        """
        url = f"{self.host}{path}"
        eff_timeout = timeout or self.timeout
        try:
            return self._session.request(method, url, timeout=eff_timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Request timed out after {eff_timeout}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.host}: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"HTTP request failed: {exc}") from exc
 
    def _get(self, path: str, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
        return self._request("GET", path, timeout=timeout, **kwargs)
 
    def _post(self, path: str, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
        return self._request("POST", path, timeout=timeout, **kwargs)
 
    def _delete(self, path: str, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
        return self._request("DELETE", path, timeout=timeout, **kwargs)
 
    def _raise_for_api_error(self, resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise OllamaAPIError(
                f"Ollama API error {resp.status_code}: {detail}",
                status_code=resp.status_code,
            )
 
    # ── Health ─────────────────────────────────────────────────────────────
    def health(self) -> Tuple[bool, Optional[str], List[ModelInfo]]:
        """
        Check server health.
 
        Returns
        -------
        (is_healthy, error_message_or_None, list_of_models)
        """
        try:
            resp = self._get("/api/tags", timeout=10)
            self._raise_for_api_error(resp)
            models = [ModelInfo.from_api(m) for m in resp.json().get("models", [])]
            logger.debug("Health OK — %d model(s) available", len(models))
            return True, None, models
        except OllamaError as exc:
            logger.warning("Health check failed: %s", exc)
            return False, str(exc), []
        except Exception as exc:
            logger.warning("Health check failed (unexpected): %s", exc)
            return False, str(exc), []
 
    # ── Model management ───────────────────────────────────────────────────
    def list_models(self) -> List[ModelInfo]:
        """Return all locally available models."""
        resp = self._get("/api/tags", timeout=10)
        self._raise_for_api_error(resp)
        return [ModelInfo.from_api(m) for m in resp.json().get("models", [])]
 
    def model_info(self, model: str) -> Dict[str, Any]:
        """Return detailed information about a model (architecture, parameters, etc.)."""
        resp = self._post("/api/show", json={"name": model}, timeout=30)
        self._raise_for_api_error(resp)
        return resp.json()
 
    def pull_model(self, model: str) -> Generator[Dict[str, Any], None, None]:
        """
        Pull a model from the Ollama registry, streaming progress events.
 
        Usage
        -----
        for event in client.pull_model("llama3"):
            print(event.get("status"), event.get("completed"), "/", event.get("total"))
        """
        # BUG FIX: previously the streaming response was not used as a
        # context manager, so abandoning the generator (early break, raised
        # exception, GC) leaked the underlying connection. Also, server-side
        # `{"error": "..."}` events were yielded silently — callers had no
        # easy way to distinguish failure from progress. Both are now handled.
        with self._post(
            "/api/pull",
            json={"name": model, "stream": True},
            stream=True,
            timeout=3600,       # pulling can take a long time
        ) as resp:
            self._raise_for_api_error(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    raise OllamaModelError(
                        f"Failed to pull '{model}': {data['error']}"
                    )
                yield data
                if data.get("status") == "success":
                    logger.info("Model '%s' pulled successfully.", model)
                    break
 
    def delete_model(self, model: str) -> None:
        """Delete a locally stored model."""
        resp = self._delete("/api/delete", json={"name": model}, timeout=30)
        self._raise_for_api_error(resp)
        logger.info("Model '%s' deleted.", model)
 
    # ── Chat ───────────────────────────────────────────────────────────────
    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        images: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalise messages list:
        • Inject system prompt if needed (BUG-4 fix)
        • Attach encoded images to the last user message if provided
        """
        effective_system = self.default_system if system_prompt is None else system_prompt
        msgs = _inject_system_prompt(list(messages), effective_system)
 
        if images:
            encoded = [_encode_image(img) for img in images]
            # Attach to the last user message — standard Ollama multimodal convention
            for i in reversed(range(len(msgs))):
                if msgs[i].get("role") == "user":
                    msgs[i] = dict(msgs[i])
                    msgs[i]["images"] = encoded
                    break
 
        return msgs
 
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream assistant tokens one chunk at a time.
 
        The final item yielded is never a content chunk — after the stream
        ends, ``usage_stats`` is populated on this client instance so callers
        can read it:  ``client.last_usage``
 
        Parameters
        ----------
        model         : Model name, e.g. "llama3"
        messages      : OpenAI-style message list (role / content dicts)
        options       : Ollama generation options (temperature, top_p, etc.)
        system_prompt : Override the default system prompt for this call only.
                        Pass "" to send no system message.
        images        : Image sources for multimodal models (path / bytes / b64)
 
        Raises
        ------
        OllamaConnectionError  – server unreachable
        OllamaModelError       – model not found or failed to load
        OllamaStreamError      – stream broken mid-response
        """
        msgs = self._prepare_messages(messages, system_prompt, images)
        payload = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "options": options or {},
        }
 
        logger.debug("chat_stream → model=%s, messages=%d", model, len(msgs))
        t0 = time.monotonic()
 
        try:
            with self._session.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaModelError(f"Model '{model}' not found on this server.")
                self._raise_for_api_error(resp)
 
                # Reset for this call. Initialised in __init__ so the attribute
                # always exists even if the stream errors before completion.
                self.last_usage = None
 
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        # BUG-6 FIX: raise instead of silently yielding error string
                        raise OllamaStreamError(f"Malformed JSON in stream: {line!r}") from exc
 
                    if "error" in data:
                        raise OllamaStreamError(data["error"])
 
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
 
                    if data.get("done"):
                        self.last_usage = UsageStats.from_api(data)
                        elapsed = time.monotonic() - t0
                        logger.debug(
                            "chat_stream done — %d tokens in %.2fs",
                            self.last_usage.total_tokens, elapsed,
                        )
                        break
 
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(str(exc)) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaStreamError("Stream timed out mid-response.") from exc
        except (OllamaError, StopIteration):
            raise
        except Exception as exc:
            # BUG-6 FIX: never swallow unknown errors as a yield
            raise OllamaStreamError(f"Unexpected stream error: {exc}") from exc
 
    def chat_once(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
    ) -> Tuple[str, UsageStats]:
        """
        Non-streaming chat.  Returns (response_text, usage_stats).
 
        BUG-3 FIX: the original had no error handling at all — any network
        or API failure would raise an unhandled exception to the caller.
        Now raises typed OllamaError subclasses.
 
        Parameters
        ----------
        Same as chat_stream.
 
        Returns
        -------
        (content, UsageStats)
        """
        msgs = self._prepare_messages(messages, system_prompt, images)
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "options": options or {},
        }
 
        logger.debug("chat_once → model=%s, messages=%d", model, len(msgs))
        t0 = time.monotonic()
 
        try:
            resp = self._post("/api/chat", json=payload)
        except OllamaConnectionError:
            raise
        except Exception as exc:
            raise OllamaConnectionError(f"Request failed: {exc}") from exc
 
        if resp.status_code == 404:
            raise OllamaModelError(f"Model '{model}' not found on this server.")
        self._raise_for_api_error(resp)
 
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise OllamaAPIError(f"Server returned non-JSON: {resp.text[:200]}") from exc
 
        content = data.get("message", {}).get("content", "")
        usage = UsageStats.from_api(data)
        logger.debug(
            "chat_once done — %d tokens in %.2fs",
            usage.total_tokens, time.monotonic() - t0,
        )
        return content, usage
 
    # ── Embeddings ─────────────────────────────────────────────────────────
    def embed(self, model: str, text: str) -> List[float]:
        """
        Generate a single embedding vector.
 
        BUG-2 FIX:
        • Old endpoint  /api/embeddings   (deprecated, removed in Ollama ≥0.2)
        • Old payload   {"prompt": text, "options": {"temperature": 0}}
                        — the options field is not accepted by the embed endpoint
        • New endpoint  /api/embed
        • New payload   {"input": text}
        """
        resp = self._post(
            "/api/embed",
            json={"model": model, "input": text},
            timeout=60,
        )
        self._raise_for_api_error(resp)
        data = resp.json()
        # Ollama returns either {"embeddings": [[...]]} or {"embedding": [...]}
        embeddings = data.get("embeddings") or [data.get("embedding", [])]
        if not embeddings or not embeddings[0]:
            raise OllamaAPIError("Empty embedding returned from server.")
        return embeddings[0]
 
    def embed_batch(self, model: str, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in a single API call.
        Much more efficient than calling embed() in a loop.
 
        Returns a list of vectors in the same order as the input texts.
        """
        if not texts:
            return []
        resp = self._post(
            "/api/embed",
            json={"model": model, "input": texts},
            timeout=120,
        )
        self._raise_for_api_error(resp)
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise OllamaAPIError("Empty batch embedding returned from server.")
        return embeddings
 
 
# ===========================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ===========================================================================
 
def quick_chat(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    system: str = "",
) -> str:
    """
    One-shot helper for simple scripts — no class boilerplate required.
 
    Returns the assistant's reply as a plain string.
    Raises OllamaError subclasses on failure.
    """
    with OllamaClient(host=host, default_system=system) as client:
        content, _ = client.chat_once(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    return content
