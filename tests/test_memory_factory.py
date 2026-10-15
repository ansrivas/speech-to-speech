"""Tests for environment-driven memory runtime construction."""

from __future__ import annotations

from speech_to_speech.memory.factory import (
    build_memory_runtime,
    get_shared_memory_runtime,
    load_generate_fn,
    reset_shared_memory_runtime,
)


def test_disabled_without_db():
    assert build_memory_runtime({}) is None
    assert build_memory_runtime({"PICARX_MEMORY_DB": "   "}) is None


def test_builds_recall_only_runtime(tmp_path):
    db = tmp_path / "mem" / "memory.db"
    runtime = build_memory_runtime(
        {
            "PICARX_MEMORY_DB": str(db),
            "PICARX_MEMORY_KEY": "aria",
            "PICARX_MEMORY_TOP_K": "3",
            "PICARX_MEMORY_MAX_CHARS": "200",
        }
    )
    assert runtime is not None
    assert runtime.context() == ""
    assert db.exists()
    runtime.close()


def test_bad_numeric_env_falls_back(tmp_path):
    runtime = build_memory_runtime(
        {
            "PICARX_MEMORY_DB": str(tmp_path / "memory.db"),
            "PICARX_MEMORY_TOP_K": "not-a-number",
            "PICARX_MEMORY_HALF_LIFE_DAYS": "nope",
        }
    )
    assert runtime is not None
    runtime.close()


def test_load_generate_fn_requires_callable():
    assert load_generate_fn("") is None
    assert load_generate_fn("speech_to_speech.memory.prompt") is None
    assert load_generate_fn("speech_to_speech.memory.factory") is None
    assert callable(load_generate_fn("examples.realtime_web_search_tool")) is False


def test_shared_runtime_caches(tmp_path, monkeypatch):
    reset_shared_memory_runtime()
    monkeypatch.setenv("PICARX_MEMORY_DB", str(tmp_path / "shared.db"))
    first = get_shared_memory_runtime()
    second = get_shared_memory_runtime()
    assert first is not None and first is second
    reset_shared_memory_runtime()
    monkeypatch.delenv("PICARX_MEMORY_DB")
    assert get_shared_memory_runtime() is None
    reset_shared_memory_runtime()
