"""Durable, SQLite-backed long-term memory for one robot profile.

Design notes (see MEMORY_REPORT.md):
- Request-path ``recall`` is strictly read-only and fails open.
- Writes (``remember``/``touch``/``prune``/``forget``/``wipe``) are serialized by
  a process lock and are meant to run from the maintenance worker.
- One SQLite connection per thread so the extractor worker and the request
  thread never share a connection. WAL lets readers proceed during writes.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from speech_to_speech.memory.prompt import sanitize_fact

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0
_PINNED_CATEGORIES = {"identity", "preference"}
_VALID_CATEGORIES = {"identity", "preference", "episodic", "trivia"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_key  TEXT NOT NULL,
    text         TEXT NOT NULL,
    norm_text    TEXT NOT NULL,
    category     TEXT NOT NULL DEFAULT 'episodic',
    pinned       INTEGER NOT NULL DEFAULT 0,
    weight       REAL NOT NULL DEFAULT 1.0,
    use_count    INTEGER NOT NULL DEFAULT 0,
    confidence   REAL NOT NULL DEFAULT 1.0,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_dedupe ON facts(profile_key, norm_text);
CREATE INDEX IF NOT EXISTS idx_facts_profile ON facts(profile_key);
"""

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass(frozen=True)
class Fact:
    id: int
    profile_key: str
    text: str
    category: str
    pinned: bool
    weight: float
    use_count: int
    confidence: float
    created_at: float
    last_used_at: float
    effective: float


class MemoryStore:
    """SQLite fact store with lazy decay, reinforcement, and pinning."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        half_life_days: float = 30.0,
        reinforcement_bonus: float = 0.2,
        prune_below: float = 0.05,
        prune_min_age_days: float = 60.0,
        recall_timeout_s: float = 0.05,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.half_life_seconds = max(half_life_days, 0.0) * _SECONDS_PER_DAY
        self.reinforcement_bonus = reinforcement_bonus
        self.prune_below = prune_below
        self.prune_min_age_seconds = max(prune_min_age_days, 0.0) * _SECONDS_PER_DAY
        self.recall_timeout_s = recall_timeout_s
        self._now = now or time.time
        self._lock = threading.Lock()
        self._local = threading.local()
        self._closed = False

    # ── connection handling ────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("MemoryStore is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=250")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._local.conn = conn
        return conn

    def _read_conn(self) -> sqlite3.Connection:
        """A connection that can only read and never creates or migrates the DB."""
        conn = getattr(self._local, "read_conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{quote(self.db_path)}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=250")
            self._local.read_conn = conn
        return conn

    def initialize(self) -> bool:
        """Create the schema up front so the request path never migrates. Fails open."""
        if self._closed:
            return False
        try:
            self._conn().commit()
            return True
        except Exception:  # noqa: BLE001 - an unusable DB must not break startup
            logger.warning("Memory database could not be initialized at %r", self.db_path, exc_info=True)
            return False

    def close(self) -> None:
        """Close this thread's connections. Safe to call more than once."""
        with self._lock:
            self._closed = True
            for name in ("conn", "read_conn"):
                conn = getattr(self._local, name, None)
                if conn is not None:
                    conn.close()
                    setattr(self._local, name, None)

    # ── scoring ────────────────────────────────────────────

    def _effective(self, weight: float, last_used_at: float, now: float) -> float:
        if self.half_life_seconds <= 0:
            return weight
        age = max(0.0, now - last_used_at)
        return weight * (0.5 ** (age / self.half_life_seconds))

    def _row_to_fact(self, row: sqlite3.Row, now: float) -> Fact:
        pinned = bool(row["pinned"])
        effective = (
            float(row["weight"]) if pinned else self._effective(float(row["weight"]), row["last_used_at"], now)
        )
        return Fact(
            id=row["id"],
            profile_key=row["profile_key"],
            text=row["text"],
            category=row["category"],
            pinned=pinned,
            weight=float(row["weight"]),
            use_count=row["use_count"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            effective=effective,
        )

    # ── request path (read-only, fails open) ───────────────

    def recall(
        self,
        profile_key: str,
        *,
        query: str = "",
        top_k: int = 5,
        max_chars: int = 600,
    ) -> list[Fact]:
        """Top facts for *profile_key*. Never writes, never raises."""
        if self._closed:
            return []
        acquired = self._lock.acquire(timeout=self.recall_timeout_s)
        if not acquired:
            return []
        try:
            if self._closed:
                return []
            now = self._now()
            rows = self._read_conn().execute(
                "SELECT * FROM facts WHERE profile_key = ?",
                (profile_key,),
            ).fetchall()
            query_tokens = _tokens(query)
            scored: list[tuple[float, float, Fact]] = []
            for row in rows:
                fact = self._row_to_fact(row, now)
                lexical = 1.0
                if query_tokens:
                    overlap = len(query_tokens & _tokens(fact.text))
                    lexical = overlap / len(query_tokens)
                scored.append((fact.effective * lexical, fact.effective, fact))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

            chosen: list[Fact] = []
            used = 0
            for _score, _effective, fact in scored[: max(top_k, 1) * 4]:
                if len(chosen) >= top_k:
                    break
                if used + len(fact.text) > max_chars:
                    continue
                chosen.append(fact)
                used += len(fact.text)
            return chosen
        except Exception:  # noqa: BLE001 - memory must never break a turn
            logger.warning("Memory recall failed for profile %r", profile_key, exc_info=True)
            return []
        finally:
            self._lock.release()

    # ── maintenance path (writes) ──────────────────────────

    def remember(
        self,
        profile_key: str,
        text: str,
        *,
        category: str = "episodic",
        pinned: bool | None = None,
        confidence: float = 1.0,
    ) -> int | None:
        """Store a fact, deduping on normalized text. Returns its row id."""
        if self._closed:
            return None
        safe = sanitize_fact(text)
        if safe is None:
            return None
        if category not in _VALID_CATEGORIES:
            category = "episodic"
        if pinned is None:
            pinned = category in _PINNED_CATEGORIES
        norm = _normalize_text(safe)
        if not norm:
            return None

        now = self._now()
        with self._lock:
            if self._closed:
                return None
            conn = self._conn()
            existing = conn.execute(
                "SELECT id, weight FROM facts WHERE profile_key = ? AND norm_text = ?",
                (profile_key, norm),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE facts SET text = ?, weight = ?, use_count = use_count + 1, "
                    "confidence = MAX(confidence, ?), last_used_at = ?, pinned = MAX(pinned, ?) "
                    "WHERE id = ?",
                    (
                        safe,
                        min(1.0, float(existing["weight"]) + self.reinforcement_bonus),
                        confidence,
                        now,
                        int(bool(pinned)),
                        existing["id"],
                    ),
                )
                conn.commit()
                return int(existing["id"])

            cursor = conn.execute(
                "INSERT INTO facts (profile_key, text, norm_text, category, pinned, weight, "
                "use_count, confidence, created_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?, 1.0, 0, ?, ?, ?)",
                (profile_key, safe, norm, category, int(bool(pinned)), confidence, now, now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def touch(self, profile_key: str, fact_ids: list[int]) -> None:
        """Reinforce facts that were selected for injection."""
        ids = [int(fid) for fid in fact_ids]
        if not ids or self._closed:
            return
        now = self._now()
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            if self._closed:
                return
            conn = self._conn()
            conn.execute(
                f"UPDATE facts SET weight = MIN(1.0, weight + ?), use_count = use_count + 1, "
                f"last_used_at = ? WHERE profile_key = ? AND id IN ({placeholders})",
                (self.reinforcement_bonus, now, profile_key, *ids),
            )
            conn.commit()

    def prune(self, profile_key: str) -> int:
        """Delete old, faded, unpinned facts. Returns the number removed."""
        if self._closed:
            return 0
        now = self._now()
        with self._lock:
            if self._closed:
                return 0
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM facts WHERE profile_key = ? AND pinned = 0",
                (profile_key,),
            ).fetchall()
            doomed = [
                row["id"]
                for row in rows
                if now - row["created_at"] > self.prune_min_age_seconds
                and self._effective(float(row["weight"]), row["last_used_at"], now) < self.prune_below
            ]
            if doomed:
                placeholders = ",".join("?" for _ in doomed)
                conn.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", doomed)
                conn.commit()
            return len(doomed)

    def list_facts(self, profile_key: str) -> list[Fact]:
        if self._closed:
            return []
        now = self._now()
        with self._lock:
            if self._closed:
                return []
            rows = self._conn().execute(
                "SELECT * FROM facts WHERE profile_key = ? ORDER BY id",
                (profile_key,),
            ).fetchall()
        return [self._row_to_fact(row, now) for row in rows]

    def forget(self, profile_key: str, fact_id: int) -> bool:
        if self._closed:
            return False
        with self._lock:
            if self._closed:
                return False
            conn = self._conn()
            cursor = conn.execute(
                "DELETE FROM facts WHERE profile_key = ? AND id = ?",
                (profile_key, fact_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def wipe(self, profile_key: str) -> int:
        if self._closed:
            return 0
        with self._lock:
            if self._closed:
                return 0
            conn = self._conn()
            cursor = conn.execute("DELETE FROM facts WHERE profile_key = ?", (profile_key,))
            conn.commit()
            return cursor.rowcount
