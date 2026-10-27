"""Tests for the local OpenAI-compatible memory extraction backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from speech_to_speech.memory.chat_completions import create_chat_completions_generate_fn


class _Completions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_generate_calls_configured_chat_completions_model():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"facts": []}'))])
    completions = _Completions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generate = create_chat_completions_generate_fn(
        "http://127.0.0.1:8080/v1",
        "",
        "google/gemma-4-12B-it-QAT-Q4_0.gguf",
        client=client,
    )

    result = generate("extract facts", "user turn")

    assert result == '{"facts": []}'
    assert completions.calls == [
        {
            "model": "google/gemma-4-12B-it-QAT-Q4_0.gguf",
            "messages": [
                {"role": "system", "content": "extract facts"},
                {"role": "user", "content": "user turn"},
            ],
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ]


@pytest.mark.parametrize(
    ("base_url", "disable_thinking", "reasoning_effort", "expected"),
    [
        ("http://127.0.0.1:8080/v1", False, None, None),
        ("http://127.0.0.1:8080/v1", True, "none", {"reasoning_effort": "none"}),
        ("http://127.0.0.1:8080/v1", False, "low", {"reasoning_effort": "low"}),
        ("https://api.openai.com/v1", True, None, None),
        ("https://api.openai.com/v1/", True, None, None),
        ("https://api.openai.com/v1", True, "low", {"reasoning_effort": "low"}),
    ],
)
def test_generate_respects_configured_reasoning_and_official_provider(
    base_url, disable_thinking, reasoning_effort, expected
):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"facts": []}'))])
    completions = _Completions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generate = create_chat_completions_generate_fn(
        base_url,
        "test-key",
        "model",
        client=client,
        disable_thinking=disable_thinking,
        reasoning_effort=reasoning_effort,
    )

    assert generate("Return JSON", "turn") == '{"facts": []}'
    assert completions.calls[0]["extra_body"] == expected


def test_generate_rejects_truncated_response_without_logging_private_content(caplog):
    content = '{"facts": [{"text": "private preference"}]}'
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=content))]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(response)))
    generate = create_chat_completions_generate_fn("http://127.0.0.1:8080/v1", "", "model", client=client)

    assert generate("Return JSON", "private transcript") == ""
    assert "token limit" in caplog.text
    assert "private preference" not in caplog.text
    assert "private transcript" not in caplog.text


def test_generate_returns_empty_string_for_non_text_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(response)))
    generate = create_chat_completions_generate_fn("http://127.0.0.1:8080/v1", "", "model", client=client)

    assert generate("system", "user") == ""


def test_openai_client_targets_local_server_without_forwarding_env_key(monkeypatch):
    received = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            received.update(kwargs)

    module = ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")

    create_chat_completions_generate_fn("http://127.0.0.1:8080/v1", "", "model")

    assert received == {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8080/v1",
        "timeout": 8.0,
        "max_retries": 0,
    }
