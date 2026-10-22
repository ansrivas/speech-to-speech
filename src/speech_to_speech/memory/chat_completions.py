"""Memory extraction through an existing OpenAI-compatible chat-completions server."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from speech_to_speech.memory.extractor import GenerateFn

MAX_EXTRACTION_INPUT_CHARS = 8_000
MAX_EXTRACTION_OUTPUT_TOKENS = 128
DEFAULT_TIMEOUT_S = 8.0


def create_chat_completions_generate_fn(
    base_url: str,
    api_key: str | None,
    model_name: str,
    *,
    client: Any | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
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
        )
        choices = getattr(response, "choices", None)
        if not choices:
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
