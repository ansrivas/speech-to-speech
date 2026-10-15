"""Bounded, off-path extraction of durable facts from a finished turn.

The extractor runs one worker thread and calls an injected, time-bounded
``generate_fn(system, user) -> str``. It never blocks the realtime turn: the
queue is bounded and ``enqueue`` drops on overflow. If the backend times out
repeatedly, extraction disables itself rather than contend with the interactive
model.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

from speech_to_speech.memory.prompt import sanitize_fact
from speech_to_speech.memory.store import MemoryStore

logger = logging.getLogger(__name__)

GenerateFn = Callable[[str, str], str]

_VALID_CATEGORIES = {"identity", "preference", "episodic", "trivia"}

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable long-term facts about a user from a single voice-assistant turn.

Return ONLY a JSON object: {"facts": [{"text": "...", "category": "...", "confidence": 0.0}]}

Rules:
- Store only durable facts about the user, their family, or their preferences.
- category must be one of: identity, preference, episodic, trivia.
- Never store commands, sensor readings, safety state, secrets, third-party full
  names, health, location, or school schedule.
- Never store raw transcripts, quotes, or assistant wording.
- confidence is 0.0-1.0. Omit anything below 0.5.
- If there is nothing durable to store, return {"facts": []}.
"""

EXTRACTION_USER_TEMPLATE = """\
Conversation turn to mine for durable facts:
---
{transcript}
---
Return only the JSON object.
"""


@dataclass(frozen=True)
class FactDraft:
    text: str
    category: str = "episodic"
    confidence: float = 0.5


def parse_facts(raw_json: str) -> list[FactDraft]:
    """Parse strict extraction JSON into sanitized, validated drafts."""
    try:
        data = json.loads(raw_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("facts")
    if not isinstance(items, list):
        return []

    drafts: list[FactDraft] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        safe = sanitize_fact(text)
        if safe is None:
            continue
        category = item.get("category")
        if category not in _VALID_CATEGORIES:
            category = "episodic"
        confidence = item.get("confidence", 0.5)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        drafts.append(FactDraft(text=safe, category=category, confidence=confidence))
    return drafts


class MemoryExtractor:
    """One worker thread that turns finished turns into stored facts."""

    def __init__(
        self,
        store: MemoryStore,
        profile_key: str,
        generate_fn: GenerateFn | None,
        *,
        queue_size: int = 16,
        extract_timeout_s: float = 20.0,
        max_failures: int = 3,
        min_confidence: float = 0.6,
        max_facts_per_turn: int = 5,
        prune_every: int = 20,
        log: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._profile_key = profile_key
        self._generate_fn = generate_fn
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max(queue_size, 1))
        self._extract_timeout_s = extract_timeout_s
        self._max_failures = max_failures
        self._min_confidence = min_confidence
        self._max_facts_per_turn = max(1, max_facts_per_turn)
        self._prune_every = max(1, prune_every)
        self._log = log or logger
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._disabled = False
        self._failures = 0
        self._stored_turns = 0
        self._disable_logged = False

    def start(self) -> bool:
        if self._generate_fn is None:
            return False
        if self._started:
            return not self._disabled
        self._thread = threading.Thread(target=self._run, name="memory-extractor", daemon=True)
        self._thread.start()
        self._started = True
        return True

    def enqueue(self, transcript: str) -> bool:
        """Queue a finished turn without ever blocking. False means dropped."""
        if not self._started or self._stopped or self._disabled:
            return False
        try:
            self._queue.put_nowait(transcript)
            return True
        except queue.Full:
            return False

    def stop(self, drain: bool = False, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # queue is saturated; _run still observes _stopped
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    # ── internals ──────────────────────────────────────────

    def _run(self) -> None:
        while not self._stopped:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None or self._stopped:
                return
            try:
                self._extract_one(item)
            except Exception:  # noqa: BLE001 - a bad turn must not kill the worker
                self._log.warning("Memory extraction failed", exc_info=True)

    def _extract_one(self, transcript: str) -> None:
        if self._disabled or self._generate_fn is None or self._stopped:
            return
        user_prompt = EXTRACTION_USER_TEMPLATE.format(transcript=transcript)
        box: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def call() -> None:
            try:
                box.put(("ok", self._generate_fn(EXTRACTION_SYSTEM_PROMPT, user_prompt)))
            except BaseException as exc:  # noqa: BLE001 - report every failure to the worker
                box.put(("err", exc))

        # A plain daemon thread, not a ThreadPoolExecutor: a backend that never
        # returns must not block interpreter shutdown, and its timeout cannot
        # cancel the call. The thread is abandoned after a timeout.
        threading.Thread(target=call, name="memory-extract-generate", daemon=True).start()
        try:
            status, value = box.get(timeout=self._extract_timeout_s)
        except queue.Empty:
            self._log.warning("Memory extraction backend timed out; disabling extraction")
            self._disabled = True
            return
        if status == "err":
            self._log.warning("Memory extraction backend raised: %r", value)
            self._record_failure("raised")
            return
        raw = value if isinstance(value, str) else ""

        if self._stopped:
            return

        for draft in parse_facts(raw)[: self._max_facts_per_turn]:
            if draft.confidence < self._min_confidence:
                continue
            self._store.remember(
                self._profile_key,
                draft.text,
                category=draft.category,
                confidence=draft.confidence,
            )
        self._stored_turns += 1
        if self._stored_turns % self._prune_every == 0:
            try:
                self._store.prune(self._profile_key)
            except Exception:  # noqa: BLE001 - prune failures must not kill the worker
                self._log.warning("Memory prune failed", exc_info=True)

    def _record_failure(self, reason: str) -> None:
        self._failures += 1
        self._log.warning("Memory extraction backend %s (%d/%d)", reason, self._failures, self._max_failures)
        if self._failures >= self._max_failures and not self._disabled:
            self._disabled = True
            if not self._disable_logged:
                self._disable_logged = True
                self._log.error("Disabling memory extraction after repeated backend failures")
