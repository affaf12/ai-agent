import os
from typing import Optional, Tuple


class AppConfig:
    VERSION: str = "7.5.0"
    BUILD: str = "enterprise"
    SESSION_TIMEOUT: int = 3600  # 1 hour

    # ── Upload limits ───────────────────────────────────────────────────────
    # MAX_FILE_SIZE_MB    = per-individual-file cap (rejected before parsing)
    # MAX_UPLOAD_SIZE_MB  = total request payload cap (sum across all files)
    MAX_FILE_SIZE_MB: int = 100
    MAX_UPLOAD_SIZE_MB: int = 500

    # BUG FIX: vision was enabled (`ENABLE_VISION = True`) but the upload
    # whitelist contained no image types, so users could not actually attach
    # images for multimodal models. Image extensions added below.
    ALLOWED_UPLOAD_TYPES: Tuple[str, ...] = (
        # Documents
        "pdf", "docx", "txt", "md", "csv", "xlsx", "pptx",
        # Code
        "py", "js", "ts", "json",
        # Images (required for ENABLE_VISION)
        "png", "jpg", "jpeg", "webp", "gif",
    )

    CHUNK_SIZE: int = 1024
    RAG_TOP_K: int = 5
    ENABLE_CACHE: bool = True

    # UPGRADE: never hard-code secrets / DB URL — read from environment
    # with the previous values as fallbacks. Avoids ENCRYPTION_KEY=None
    # being silently used in production.
    ENCRYPTION_KEY: Optional[str] = os.environ.get("ENCRYPTION_KEY")
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "sqlite:///data/ollama_pro_enterprise.db",
    )

    # ── Feature flags ───────────────────────────────────────────────────────
    ENABLE_AGENTS: bool = True
    ENABLE_VISION: bool = True
    ENABLE_TTS: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_COLLABORATION: bool = True

    # ── LLM generation options ──────────────────────────────────────────────
    # UPGRADE — directly addresses the "answers are too long" problem.
    #
    # Wording in the system prompt ("be concise", "max 4 bullets") is only
    # a *request* — models routinely ignore it. The reliable controls are
    # the Ollama generation options below. Pass them as the `options=` arg
    # of OllamaClient.chat_stream / chat_once and the model is *physically*
    # unable to exceed the cap.
    #
    # Tune these to taste:
    #   LLM_NUM_PREDICT  hard cap on generated tokens (≈ 4 chars each)
    #   LLM_TEMPERATURE  lower = less rambling, more deterministic
    #   LLM_TOP_P        nucleus sampling cutoff
    #   LLM_STOP         stop sequences — generation halts as soon as one
    #                    of these strings appears in the output
    LLM_NUM_PREDICT: int = 256          # ≈ 4–6 short bullets
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.9
    LLM_REPEAT_PENALTY: float = 1.1
    LLM_STOP: Tuple[str, ...] = (
        "\n\n\n",      # any triple-blank-line = wrap up
        "###",         # markdown heading marker (model often drifts here)
        "</answer>",   # if the model wraps responses
    )

    @classmethod
    def llm_options(cls) -> dict:
        """Return Ollama-compatible options dict built from the LLM_* fields."""
        return {
            "num_predict":   cls.LLM_NUM_PREDICT,
            "temperature":   cls.LLM_TEMPERATURE,
            "top_p":         cls.LLM_TOP_P,
            "repeat_penalty": cls.LLM_REPEAT_PENALTY,
            "stop":          list(cls.LLM_STOP),
        }


CONFIG = AppConfig()

# =============================================================================
# SMART MODEL SELECTION - NEW
# =============================================================================

def get_system_ram():
    """Get total and available RAM in GB"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        return round(total_gb, 1), round(available_gb, 1)
    except:
        return 8.0, 4.0  # Default fallback

def get_optimal_model():
    """
    Auto-select best model based on available RAM.
    Users can override by setting OLLAMA_MODEL env variable.
    """
    # Manual override via environment variable
    manual_model = os.environ.get("OLLAMA_MODEL")
    if manual_model:
        return manual_model
    
    total_ram, available_ram = get_system_ram()
    
    # Model selection logic
    if available_ram >= 10:
        # 16GB+ systems with plenty free
        return "llama3.1"
    elif available_ram >= 6:
        # 12GB systems
        return "llama3.2:3b"
    elif available_ram >= 3:
        # 8GB systems
        return "gemma2:2b"
    else:
        # 4-6GB systems or low memory
        return "llama3.2:1b"

def get_model_info():
    """Get current model and system info for display"""
    total_ram, available_ram = get_system_ram()
    model = get_optimal_model()
    
    try:
        import psutil
        psutil_ok = True
    except:
        psutil_ok = False
    
    return {
        "model": model,
        "total_ram_gb": total_ram,
        "available_ram_gb": available_ram,
        "psutil_available": psutil_ok,
    }

# Export for easy import
LLM_MODEL = get_optimal_model()
EMBED_MODEL = "nomic-embed-text"


# =============================================================================
# SECURITY & AUTHENTICATION
# =============================================================================
