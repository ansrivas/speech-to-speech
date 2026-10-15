"""Tests for the SQLite-backed memory store."""

from __future__ import annotations

from speech_to_speech.memory.store import MemoryStore

DAY = 86400.0


def _clock(start: float = 1000.0):
    state = {"t": start}
    return state, (lambda: state["t"])


def _store(tmp_path, clock, **kwargs):
    return MemoryStore(tmp_path / "memory.db", now=clock, **kwargs)


def test_remember_then_recall_roundtrip(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    assert store.remember("aria", "Vikki is a cat with one eye.") is not None

    facts = store.recall("aria")
    assert [f.text for f in facts] == ["Vikki is a cat with one eye."]
    assert store.recall("other") == []


def test_dedupe_reinforces_existing_fact(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    first = store.remember("aria", "Likes pancakes.")
    second = store.remember("aria", "  likes   pancakes  ")

    facts = store.list_facts("aria")
    assert len(facts) == 1
    assert first == second == facts[0].id
    assert facts[0].weight == 1.0  # already capped

    store.remember("aria", "Likes pancakes.")
    store.touch("aria", [facts[0].id])
    facts = store.list_facts("aria")
    assert facts[0].use_count >= 1


def test_decay_lowers_unused_fact_score(tmp_path):
    state, now = _clock()
    store = _store(tmp_path, now, half_life_days=1.0)
    store.remember("aria", "old fact")
    state["t"] += DAY * 10
    store.remember("aria", "fresh fact")

    facts = store.recall("aria")
    assert facts[0].text == "fresh fact"
    old = next(f for f in facts if f.text == "old fact")
    assert old.effective < 0.01


def test_pinned_fact_never_decays(tmp_path):
    state, now = _clock()
    store = _store(tmp_path, now, half_life_days=1.0)
    store.remember("aria", "Aria is four years old.", category="identity")
    state["t"] += DAY * 100

    facts = store.recall("aria")
    pinned = next(f for f in facts if f.category == "identity")
    assert pinned.pinned is True
    assert pinned.effective == 1.0


def test_recall_is_read_only(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    store.remember("aria", "Read-only check.")
    before = store.list_facts("aria")[0]

    store.recall("aria", query="read only check")

    after = store.list_facts("aria")[0]
    assert (after.weight, after.use_count, after.last_used_at) == (
        before.weight,
        before.use_count,
        before.last_used_at,
    )


def test_prune_removes_old_faded_unpinned_only(tmp_path):
    state, now = _clock()
    store = _store(tmp_path, now, half_life_days=1.0, prune_min_age_days=1)
    store.remember("aria", "faded trivia")
    store.remember("aria", "Aria likes blue.", category="preference")
    state["t"] += DAY * 30

    removed = store.prune("aria")
    assert removed == 1
    remaining = {f.text for f in store.list_facts("aria")}
    assert remaining == {"Aria likes blue."}


def test_recall_respects_max_chars(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    for i in range(5):
        store.remember("aria", f"fact number {i} with some padding text")

    facts = store.recall("aria", max_chars=80)
    assert facts
    assert sum(len(f.text) for f in facts) <= 80


def test_wipe_and_forget(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    keep = store.remember("aria", "Keep me.")
    drop = store.remember("aria", "Drop me.")

    assert store.forget("aria", drop) is True
    assert store.forget("aria", drop) is False
    assert [f.id for f in store.list_facts("aria")] == [keep]

    assert store.wipe("aria") == 1
    assert store.list_facts("aria") == []


def test_persists_across_reopen(tmp_path):
    _state, now = _clock()
    path = tmp_path / "memory.db"
    store = MemoryStore(path, now=now)
    store.remember("aria", "Survives restart.")
    store.close()

    reopened = MemoryStore(path, now=now)
    assert [f.text for f in reopened.recall("aria")] == ["Survives restart."]


def test_recall_fails_open_on_broken_db(tmp_path):
    _state, now = _clock()
    store = MemoryStore(tmp_path, now=now)  # a directory, not a file
    assert store.recall("aria") == []


def test_recall_does_not_create_or_write_db(tmp_path):
    _state, now = _clock()
    path = tmp_path / "fresh.db"
    store = MemoryStore(path, now=now)

    assert store.recall("aria") == []
    assert not path.exists()  # a read must never create the database


def test_writes_after_close_are_noops(tmp_path):
    _state, now = _clock()
    store = _store(tmp_path, now)
    assert store.remember("aria", "before close") is not None
    store.close()

    assert store.remember("aria", "after close") is None
    assert store.prune("aria") == 0
    assert store.forget("aria", 1) is False
    assert store.wipe("aria") == 0


def test_initialize_creates_schema_once(tmp_path):
    _state, now = _clock()
    path = tmp_path / "memory.db"
    store = MemoryStore(path, now=now)

    assert store.initialize() is True
    assert path.exists()
    assert store.initialize() is True  # idempotent
