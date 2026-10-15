"""Process-wide memory runtime construction from environment configuration.

All realtime pipeline units share one runtime so facts are stored and recalled
under a single SQLite file. Memory is disabled unless ``PICARX_MEMORY_DB`` is set.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
from pathlib import Path
from typing import Callable, Mapping

from speech_to_speech.memory.extractor import GenerateFn, MemoryExtractor
from speech_to_speech.memory.service import MemoryRuntime
from speech_to_speech.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_shared: MemoryRuntime | None = None


def _safe_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, "").strip())
    except (TypeError, ValueError):
        return default


def _safe_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, "").strip())
    except (TypeError, ValueError):
        return default


def load_generate_fn(module_name: str) -> GenerateFn | None:
    """Load a ``generate(system, user) -> str`` callable from a module, if named."""
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.warning("Memory extractor module %r could not be imported", module_name, exc_info=True)
        return None
    generate = getattr(module, "generate", None)
    if not callable(generate):
        logger.warning("Memory extractor module %r has no callable generate(system, user)", module_name)
        return None
    return generate


def build_memory_runtime(env: Mapping[str, str] | None = None) -> MemoryRuntime | None:
    """Build a runtime from env, or ``None`` when memory is disabled."""
    env = os.environ if env is None else env
    db_path = env.get("PICARX_MEMORY_DB", "").strip()
    if not db_path:
        return None

    key = env.get("PICARX_MEMORY_KEY", "aria").strip() or "aria"
    try:
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Memory directory for %r could not be created", db_path, exc_info=True)

    store = MemoryStore(
        db_path,
        half_life_days=_safe_float(env, "PICARX_MEMORY_HALF_LIFE_DAYS", 30.0),
        prune_min_age_days=_safe_float(env, "PICARX_MEMORY_PRUNE_MIN_AGE_DAYS", 60.0),
    )
    if not store.initialize():
        return None
    generate_fn: Callable[[str, str], str] | None = load_generate_fn(
        env.get("PICARX_MEMORY_EXTRACT_MODULE", "").strip()
    )
    extractor: MemoryExtractor | None = None
    if generate_fn is not None:
        extractor = MemoryExtractor(
            store,
            key,
            generate_fn,
            min_confidence=_safe_float(env, "PICARX_MEMORY_MIN_CONFIDENCE", 0.6),
        )
        if not extractor.start():
            extractor = None
    else:
        logger.info("Memory extraction disabled (no PICARX_MEMORY_EXTRACT_MODULE); recall-only mode")

    runtime = MemoryRuntime(
        store,
        extractor,
        key,
        top_k=_safe_int(env, "PICARX_MEMORY_TOP_K", 5),
        max_chars=_safe_int(env, "PICARX_MEMORY_MAX_CHARS", 600),
    )
    logger.info("Long-term memory enabled (profile=%s, db=%s, extraction=%s)", key, db_path, extractor is not None)
    return runtime


def get_shared_memory_runtime() -> MemoryRuntime | None:
    """Return the process-wide runtime, building it once. Idempotent."""
    global _shared
    if _shared is None:
        _shared = build_memory_runtime()
        if _shared is not None:
            atexit.register(_shared.close)
    return _shared


def reset_shared_memory_runtime() -> None:
    """Testing helper: drop the cached runtime without closing it."""
    global _shared
    _shared = None
