"""Memory extraction through an existing OpenAI-compatible chat-completions server."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from speech_to_speech.memory.extractor import GenerateFn

MAX_EXTRACTION_INPUT_CHARS = 8_000
MAX_EXTRACTION_OUTPUT_TOKENS = 512
DEFAULT_TIMEOUT_S = 8.0

logger = logging.getLogger(__name__)


def create_chat_completions_generate_fn(
    base_url: str,
    api_key: str | None,
    model_name: str,
    *,
    client: Any | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    disable_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> GenerateFn:
    """Build a bounded generator using the configured chat-completions endpoint.

    An injected client keeps tests independent of the optional local runtime
    environment. Production uses the project's existing ``openai`` dependency.
    """
    if client is None:
        from openai import OpenAI

        if not api_key and _is_loopback(base_url):
            api_key = "none"
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
        )

    # Match the conversation backend's precedence without importing its handlers.
    extra_body: dict[str, Any] | None = None
    if reasoning_effort:
        extra_body = {"reasoning_effort": reasoning_effort}
    elif disable_thinking and base_url.rstrip("/") != "https://api.openai.com/v1":
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def generate(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user[:MAX_EXTRACTION_INPUT_CHARS]},
            ],
            temperature=0,
            max_tokens=MAX_EXTRACTION_OUTPUT_TOKENS,
            stream=False,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        if getattr(choices[0], "finish_reason", None) == "length":
            logger.warning(
                "Memory extraction reached its %s-token limit; skipping incomplete output", MAX_EXTRACTION_OUTPUT_TOKENS
            )
            return ""
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return content if isinstance(content, str) else ""

    return generate


def _is_loopback(base_url: str) -> bool:
    host = urlparse(base_url).hostname
    if host is None:
        return False
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
