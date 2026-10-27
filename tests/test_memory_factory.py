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


def test_uses_chat_completions_backend_by_default(tmp_path, monkeypatch):
    from speech_to_speech.memory import factory

    calls = []

    def generate(system: str, user: str) -> str:
        calls.append((system, user))
        return '{"facts": [{"text": "Aria likes pancakes.", "category": "preference", "confidence": 0.9}]}'

    def build_generate(base_url, api_key, model_name, *, disable_thinking, reasoning_effort):
        assert base_url == "http://127.0.0.1:8080/v1"
        assert api_key == ""
        assert model_name == "google/gemma-4-12B-it-QAT-Q4_0.gguf"
        assert disable_thinking is False
        assert reasoning_effort == "low"
        return generate

    monkeypatch.setattr(factory, "create_chat_completions_generate_fn", build_generate)
    runtime = build_memory_runtime(
        {"PICARX_MEMORY_DB": str(tmp_path / "memory.db")},
        llm_backend="chat-completions",
        llm_config={
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "",
            "model_name": "google/gemma-4-12B-it-QAT-Q4_0.gguf",
            "disable_thinking": False,
            "reasoning_effort": "low",
        },
    )
    assert runtime is not None
    try:
        assert runtime.record_turn("I like pancakes.", "I'll remember that.") is True
        import time

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not runtime.context():
            time.sleep(0.01)
        assert "Aria likes pancakes." in runtime.context()
        assert calls
    finally:
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
