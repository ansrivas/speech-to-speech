"""Tests for the bounded, off-path memory extractor."""

from __future__ import annotations

import json
import threading
import time

from speech_to_speech.memory.extractor import MemoryExtractor, parse_facts
from speech_to_speech.memory.store import MemoryStore


def _store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_parse_facts_valid_and_invalid():
    raw = json.dumps(
        {
            "facts": [
                {"text": "Aria is four.", "category": "identity", "confidence": 0.9},
                {"text": "", "confidence": 0.9},
                {"text": "bad <tag>", "confidence": 0.9},
                {"text": "no confidence"},
                "not a dict",
            ]
        }
    )
    drafts = parse_facts(raw)
    assert [d.text for d in drafts] == ["Aria is four.", "no confidence"]

    assert parse_facts("not json") == []
    assert parse_facts('{"facts": "nope"}') == []
    assert parse_facts('{"other": []}') == []


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_enqueue_runs_generate_and_persists(tmp_path):
    store = _store(tmp_path)
    calls: list[tuple[str, str]] = []

    def generate(system: str, user: str) -> str:
        calls.append((system, user))
        return json.dumps({"facts": [{"text": "Vikki has one eye.", "category": "identity", "confidence": 0.9}]})

    extractor = MemoryExtractor(store, "aria", generate, min_confidence=0.5)
    assert extractor.start() is True
    try:
        assert extractor.enqueue("user: my cat vikki has one eye") is True
        assert _wait_for(lambda: bool(store.list_facts("aria")))
        assert calls and "my cat vikki" in calls[0][1]
        assert [f.text for f in store.list_facts("aria")] == ["Vikki has one eye."]
    finally:
        extractor.stop()
        store.close()


def test_enqueue_drops_when_full_or_stopped(tmp_path):
    store = _store(tmp_path)
    release = threading.Event()
    processing = threading.Event()

    def generate(system: str, user: str) -> str:
        processing.set()
        release.wait(timeout=3.0)
        return json.dumps({"facts": []})

    extractor = MemoryExtractor(store, "aria", generate, queue_size=1)
    assert extractor.start() is True
    try:
        assert extractor.enqueue("first") is True
        assert processing.wait(timeout=2.0)          # worker is now blocked in generate
        assert extractor.enqueue("second") is True  # fills the queue
        assert extractor.enqueue("third") is False  # queue full -> dropped
    finally:
        release.set()
        extractor.stop()
    assert extractor.enqueue("after stop") is False
    store.close()


def test_generate_failure_and_timeout_are_swallowed(tmp_path):
    store = _store(tmp_path)

    def boom(system: str, user: str) -> str:
        raise RuntimeError("model exploded")

    extractor = MemoryExtractor(store, "aria", boom)
    assert extractor.start() is True
    try:
        assert extractor.enqueue("turn") is True
        time.sleep(0.1)
    finally:
        extractor.stop()

    def slow(system: str, user: str) -> str:
        time.sleep(1.0)
        return json.dumps({"facts": []})

    extractor2 = MemoryExtractor(store, "aria", slow, extract_timeout_s=0.05)
    assert extractor2.start() is True
    try:
        assert extractor2.enqueue("turn") is True
        time.sleep(0.2)
    finally:
        extractor2.stop()

    assert store.list_facts("aria") == []
    store.close()


def test_start_false_without_backend(tmp_path):
    store = _store(tmp_path)
    extractor = MemoryExtractor(store, "aria", None)
    assert extractor.start() is False
    assert extractor.enqueue("turn") is False
    extractor.stop()
    store.close()


def test_timeout_disables_and_shuts_executor(tmp_path):
    store = _store(tmp_path)

    def slow(system: str, user: str) -> str:
        time.sleep(0.5)
        return json.dumps({"facts": []})

    extractor = MemoryExtractor(store, "aria", slow, extract_timeout_s=0.05)
    assert extractor.start() is True
    try:
        assert extractor.enqueue("turn") is True
        assert _wait_for(lambda: extractor._disabled)
        assert extractor.enqueue("another") is False
    finally:
        extractor.stop()
        store.close()


def test_stop_is_idempotent_and_not_running_after(tmp_path):
    store = _store(tmp_path)
    extractor = MemoryExtractor(store, "aria", lambda s, u: json.dumps({"facts": []}))
    assert extractor.start() is True

    extractor.stop()
    extractor.stop()
    assert extractor.is_running is False
    assert extractor.enqueue("turn") is False
    store.close()
