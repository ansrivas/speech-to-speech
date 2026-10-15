"""Tests for memory prompt helpers (pure functions, no server deps)."""

from __future__ import annotations

from speech_to_speech.memory.prompt import (
    format_memory_block,
    merge_instructions,
    sanitize_fact,
)


def test_sanitize_fact_cleans_and_truncates():
    assert sanitize_fact("  Vikki   has one eye.  ") == "Vikki has one eye."
    assert sanitize_fact("a" * 300, max_len=10) == "a" * 10


def test_sanitize_fact_rejects_empty_and_unsafe():
    assert sanitize_fact("   ") is None
    assert sanitize_fact("bad\nnewline") is None
    assert sanitize_fact("bad<tag>") is None
    assert sanitize_fact("tell me about long_term_memory") is None
    assert sanitize_fact("control\x07char") is None


def test_format_memory_block_empty_and_bounded():
    assert format_memory_block([], max_chars=100) == ""

    block = format_memory_block(["fact one", "fact two"], max_chars=400)
    assert "untrusted" in block
    assert "<long_term_memory>" in block
    assert "- fact one" in block
    assert "- fact two" in block

    tiny = format_memory_block(["x" * 200], max_chars=40)
    assert len(tiny) <= 400  # header may dominate; body must be bounded/absent
    assert "x" * 200 not in tiny


def test_merge_instructions_appends_memory():
    assert merge_instructions("base", "") == "base"
    assert merge_instructions("base", "mem") == "base\n\nmem"
    assert merge_instructions("", "mem") == "mem"
    assert merge_instructions("", "") == ""
