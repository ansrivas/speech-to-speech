"""Exercise the real extraction SDK, background worker, and persistent fact store."""

from __future__ import annotations

import json
import time

import httpx
import openai
import pytest

from speech_to_speech.memory.factory import build_memory_runtime
from speech_to_speech.memory.store import MemoryStore


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace only HTTP I/O; keep SDK serialization and all memory code real."""
    requests = []
    responses = []
    clients = []
    real_openai = openai.OpenAI

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "gemma-4-12B-it-QAT-GGUF"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] == 512
        assert body["stream"] is False
        reply = responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(
            200,
            json={
                "id": "memory-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": reply.get("finish_reason", "stop"),
                        "message": {"role": "assistant", "content": reply["content"]},
                    }
                ],
            },
        )

    def create_client(**kwargs):
        client = real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", create_client)
    yield requests, responses
    for client in clients:
        client.close()


def _runtime(db):
    runtime = build_memory_runtime(
        {"PICARX_MEMORY_DB": str(db), "PICARX_MEMORY_KEY": "aria"},
        llm_backend="chat-completions",
        llm_config={
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "",
            "model_name": "gemma-4-12B-it-QAT-GGUF",
        },
    )
    assert runtime is not None
    return runtime


def _wait_for_facts(store, count):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        facts = store.list_facts("aria")
        if len(facts) == count:
            return facts
        time.sleep(0.01)
    pytest.fail(f"Expected {count} committed facts; found {store.list_facts('aria')!r}")


def _facts_reply(texts):
    return {
        "content": json.dumps(
            {"facts": [{"text": text, "category": "preference", "confidence": 1.0} for text in texts]},
            indent=2,
        )
    }


def test_multiple_completed_turns_are_recalled_and_survive_reopening(tmp_path, mock_llm):
    requests, responses = mock_llm
    first = [
        "User prefers vegetarian food for lunch, specifically lentil soup or roasted vegetables.",
        "User does not enjoy very spicy dishes.",
        "User enjoys growing herbs in small pots as a hobby.",
        "User enjoys reading science fiction.",
    ]
    second = ["User prefers metric units.", "User prefers a 24-hour clock."]
    responses.extend([_facts_reply(first), _facts_reply(second)])
    db = tmp_path / "aria-memory.sqlite3"
    runtime = _runtime(db)
    reader = MemoryStore(db)
    try:
        assert runtime.record_turn(
            "I usually pick vegetarian lunches, especially lentil soup or roasted vegetables. "
            "I do not enjoy spicy dishes. At weekends I enjoy growing herbs and reading science fiction.",
            "I can use those preferences when suggesting lunches and relaxing activities.",
        )
        assert runtime.record_turn(
            "I also prefer metric units and a 24-hour clock in examples, not a particular appointment.",
            "I will use metric units and a 24-hour clock when relevant.",
        )
        facts = _wait_for_facts(reader, 6)
        assert {fact.text for fact in facts} == set(first + second)
        assert reader.list_facts("other-profile") == []
        assert "User prefers metric units." in runtime.context()
        assert len(requests) == 2
        assert "User: I usually pick vegetarian lunches" in requests[0]["messages"][1]["content"]
    finally:
        runtime.close()
        reader.close()

    reopened = MemoryStore(db)
    try:
        assert {fact.text for fact in reopened.recall("aria", top_k=10, max_chars=2000)} == set(first + second)
    finally:
        reopened.close()


@pytest.mark.parametrize("bad_reply", ["fenced", "truncated", "timeout", "empty"])
def test_unusable_extraction_does_not_prevent_next_turn_from_persisting(tmp_path, mock_llm, bad_reply, caplog):
    requests, responses = mock_llm
    bad_content = _facts_reply(["This incomplete result must not be remembered."])["content"]
    responses.append(
        {
            "fenced": {"content": f"```json\n{bad_content}\n```"},
            "truncated": {"content": bad_content, "finish_reason": "length"},
            "timeout": httpx.ReadTimeout("synthetic timeout"),
            "empty": {"content": '{"facts": []}'},
        }[bad_reply]
    )
    responses.append(_facts_reply(["User enjoys watercolor painting."]))
    runtime = _runtime(tmp_path / "aria-memory.sqlite3")
    reader = MemoryStore(tmp_path / "aria-memory.sqlite3")
    try:
        assert runtime.record_turn("First turn for the failure case.", "Acknowledged.")
        assert runtime.record_turn("I enjoy watercolor painting.", "I will remember your hobby.")
        assert [fact.text for fact in _wait_for_facts(reader, 1)] == ["User enjoys watercolor painting."]
        assert len(requests) == 2
        if bad_reply == "truncated":
            assert "token limit" in caplog.text
    finally:
        runtime.close()
        reader.close()
