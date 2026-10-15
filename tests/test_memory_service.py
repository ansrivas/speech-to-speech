"""Tests for the memory runtime glue (request path and lifecycle)."""

from __future__ import annotations

from types import SimpleNamespace

from speech_to_speech.memory.service import MemoryRuntime, last_turn_messages


class FakeStore:
    def __init__(self, facts=None, raise_on_recall=False):
        self._facts = facts or []
        self._raise = raise_on_recall
        self.closed = False

    def recall(self, profile_key, *, top_k=5, max_chars=600):
        if self._raise:
            raise RuntimeError("db down")
        return self._facts

    def close(self):
        self.closed = True


class FakeExtractor:
    def __init__(self):
        self.enqueued: list[str] = []
        self.stopped = False

    def enqueue(self, transcript: str) -> bool:
        self.enqueued.append(transcript)
        return True

    def stop(self, *args, **kwargs) -> None:
        self.stopped = True


class FakeChat:
    def __init__(self, messages):
        self._messages = messages

    def to_transformers_chat(self):
        return self._messages


def test_last_turn_messages_picks_latest_user_and_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": [{"type": "input_text", "text": "non-string"}]},
    ]
    assert last_turn_messages(messages) == ("new question", "new answer")


def test_context_builds_block_and_is_empty_without_facts():
    runtime = MemoryRuntime(FakeStore(), None, "aria", max_chars=400)
    assert runtime.context() == ""

    facts = [SimpleNamespace(text="Aria is four."), SimpleNamespace(text="Likes pancakes.")]
    runtime = MemoryRuntime(FakeStore(facts), None, "aria", max_chars=400)
    block = runtime.context()
    assert "<long_term_memory>" in block
    assert "- Aria is four." in block


def test_context_fails_open():
    runtime = MemoryRuntime(FakeStore(raise_on_recall=True), None, "aria")
    assert runtime.context() == ""


def test_record_turn_enqueues_formatted_transcript():
    extractor = FakeExtractor()
    runtime = MemoryRuntime(FakeStore(), extractor, "aria")
    assert runtime.record_turn("hi", "hello") is True
    assert extractor.enqueued == ["User: hi\nAssistant: hello"]
    assert runtime.record_turn("", "") is False


def test_record_latest_turn_reads_chat():
    extractor = FakeExtractor()
    runtime = MemoryRuntime(FakeStore(), extractor, "aria")
    chat = FakeChat(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what is my cat called"},
            {"role": "assistant", "content": "Vikki"},
        ]
    )
    assert runtime.record_latest_turn(chat) is True
    assert extractor.enqueued == ["User: what is my cat called\nAssistant: Vikki"]


def test_close_stops_extractor_then_store():
    store = FakeStore()
    extractor = FakeExtractor()
    runtime = MemoryRuntime(store, extractor, "aria")
    runtime.close()
    assert extractor.stopped is True
    assert store.closed is True
