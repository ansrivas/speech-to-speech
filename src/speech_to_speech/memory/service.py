"""Request-path and lifecycle glue for long-term memory."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from speech_to_speech.memory.extractor import MemoryExtractor
from speech_to_speech.memory.prompt import format_memory_block
from speech_to_speech.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def last_turn_messages(messages: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """Return ``(user_text, assistant_text)`` from the tail of a chat snapshot."""
    user_text = ""
    assistant_text = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content
    return user_text, assistant_text


class MemoryRuntime:
    """Owns the store and extractor for one process."""

    def __init__(
        self,
        store: MemoryStore,
        extractor: MemoryExtractor | None,
        profile_key: str,
        *,
        top_k: int = 5,
        max_chars: int = 600,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._profile_key = profile_key
        self._top_k = top_k
        self._max_chars = max_chars

    def context(self) -> str:
        """Read-only, bounded memory block for the system prompt."""
        try:
            facts = self._store.recall(self._profile_key, top_k=self._top_k, max_chars=self._max_chars)
        except Exception:  # noqa: BLE001 - memory must never break a turn
            logger.warning("Memory context build failed", exc_info=True)
            return ""
        return format_memory_block([fact.text for fact in facts], self._max_chars)

    def record_turn(self, user_text: str, assistant_text: str) -> bool:
        """Queue a finished turn for background extraction. Never blocks."""
        if self._extractor is None:
            return False
        if not user_text.strip() and not assistant_text.strip():
            return False
        text = f"User: {user_text}\nAssistant: {assistant_text}".strip()
        try:
            return self._extractor.enqueue(text)
        except Exception:  # noqa: BLE001
            logger.warning("Memory turn enqueue failed", exc_info=True)
            return False

    def record_latest_turn(self, chat: Any) -> bool:
        """Queue the tail turn of a chat snapshot for background extraction."""
        try:
            messages = chat.to_transformers_chat()
        except Exception:  # noqa: BLE001
            logger.warning("Memory chat snapshot failed", exc_info=True)
            return False
        return self.record_turn(*last_turn_messages(messages))

    def close(self) -> None:
        """Stop the extractor first, then close the store."""
        if self._extractor is not None:
            self._extractor.stop()
        self._store.close()
