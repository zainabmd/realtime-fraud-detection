"""
Event broker abstraction with two interchangeable backends.

``SQLiteBroker`` – a zero-dependency, multi-process-safe *partitioned
commit log* that mirrors the Kafka semantics this pipeline relies on:

    * topics split into N partitions, records routed by key hash, so all events
      for one card land in one partition and stay ordered;
    * monotonically increasing per-partition offsets;
    * consumer groups with explicitly committed offsets (at-least-once);
    * end-offset queries for consumer-lag monitoring;
    * time-based retention.

``KafkaBroker`` – the same interface on top of ``confluent-kafka`` for
production. Select with ``BROKER_BACKEND=kafka``.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import (BROKER_BACKEND, BROKER_DB, KAFKA_BOOTSTRAP, NUM_PARTITIONS, get_logger)

log = get_logger("broker")


@dataclass(frozen=True)
class Message:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: str
    timestamp: float


def partition_for(key: str | None, n_partitions: int) -> int:
    """Stable key -> partition routing (crc32 is identical across processes, unlike hash())."""
    if key is None:
        return 0
    return zlib.crc32(key.encode()) % n_partitions


class Broker(ABC):
    """Minimal producer/consumer contract used by the simulator and the scoring service."""

    @abstractmethod
    def produce_batch(self, topic: str, records: Iterable[tuple[str | None, str]]) -> int: ...

    @abstractmethod
    def subscribe(self, topic: str, group: str) -> None: ...

    @abstractmethod
    def poll(self, max_records: int = 1000) -> list[Message]: ...

    @abstractmethod
    def commit(self) -> None: ...

    @abstractmethod
    def lag(self) -> dict[int, int]:
        """Per-partition consumer lag (end offset - committed offset) for the subscribed topic."""

    def close(self) -> None:  # pragma: no cover - optional
        pass


# =========================================================================== #
# SQLite commit log
# =========================================================================== #
class SQLiteBroker(Broker):
    _DDL = """
    CREATE TABLE IF NOT EXISTS messages (
        topic     TEXT    NOT NULL,
        partition INTEGER NOT NULL,
        offset    INTEGER NOT NULL,
        key       TEXT,
        value     TEXT    NOT NULL,
        ts        REAL    NOT NULL,
        PRIMARY KEY (topic, partition, offset)
    ) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS consumer_offsets (
        grp       TEXT    NOT NULL,
        topic     TEXT    NOT NULL,
        partition INTEGER NOT NULL,
        offset    INTEGER NOT NULL,          -- next offset to read
        updated   REAL    NOT NULL,
        PRIMARY KEY (grp, topic, partition)
    );
    CREATE INDEX IF NOT EXISTS ix_messages_ts ON messages (topic, ts);
    """

    def __init__(self, path: Path = BROKER_DB, n_partitions: int = NUM_PARTITIONS):
        self.path = Path(path)
        self.n_partitions = n_partitions
        self._local = threading.local()
        self._topic: str | None = None
        self._group: str | None = None
        self._positions: dict[int, int] = {}
        self._rr = 0  # round-robin cursor so no partition starves
        with self._conn() as c:
            c.executescript(self._DDL)

    # ---- connection handling ------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        """One connection per thread; WAL lets readers and the writer work concurrently."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _retry(self, fn, attempts: int = 5):
        """Retry on transient 'database is locked' errors with exponential backoff."""
        for i in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or i == attempts - 1:
                    raise
                time.sleep(0.05 * 2 ** i)

    # ---- producer -------------------------------------------------------------
    def produce_batch(self, topic: str, records: Iterable[tuple[str | None, str]]) -> int:
        records = list(records)
        if not records:
            return 0

        def _tx() -> int:
            c = self._conn()
            c.execute("BEGIN IMMEDIATE")  # serialises offset assignment across producers
            try:
                nxt = self._heads(c, topic)
                now = time.time()
                rows = []
                for key, value in records:
                    p = partition_for(key, self.n_partitions)
                    off = nxt.get(p, 0)
                    nxt[p] = off + 1
                    rows.append((topic, p, off, key, value, now))
                c.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
                c.execute("COMMIT")
                return len(rows)
            except Exception:
                c.execute("ROLLBACK")
                raise

        return self._retry(_tx)

    # ---- consumer -------------------------------------------------------------
    def subscribe(self, topic: str, group: str) -> None:
        self._topic, self._group = topic, group
        c = self._conn()
        committed = dict(c.execute(
            "SELECT partition, offset FROM consumer_offsets WHERE grp = ? AND topic = ?", (group, topic)).fetchall())
        self._positions = {p: committed.get(p, 0) for p in range(self.n_partitions)}
        log.info("Subscribed group=%s topic=%s positions=%s", group, topic, self._positions)

    def poll(self, max_records: int = 1000) -> list[Message]:
        if self._topic is None:
            raise RuntimeError("subscribe() before poll()")
        c = self._conn()
        out: list[Message] = []
        per_part = max(1, max_records // self.n_partitions)
        for i in range(self.n_partitions):
            p = (self._rr + i) % self.n_partitions
            rows = c.execute(
                "SELECT partition, offset, key, value, ts FROM messages "
                "WHERE topic = ? AND partition = ? AND offset >= ? ORDER BY offset LIMIT ?",
                (self._topic, p, self._positions[p], per_part)).fetchall()
            for part, off, key, value, ts in rows:
                out.append(Message(self._topic, part, off, key, value, ts))
            if rows:
                self._positions[p] = rows[-1][1] + 1
        self._rr = (self._rr + 1) % self.n_partitions
        return out

    def commit(self) -> None:
        if self._topic is None:
            return
        now = time.time()
        rows = [(self._group, self._topic, p, off, now) for p, off in self._positions.items()]

        def _tx():
            self._conn().executemany(
                "INSERT INTO consumer_offsets VALUES (?,?,?,?,?) "
                "ON CONFLICT(grp, topic, partition) DO UPDATE SET offset = excluded.offset, updated = excluded.updated",
                rows)

        self._retry(_tx)

    def seek_to_committed(self) -> None:
        """Rewind in-memory positions to the last commit (used after a failed batch)."""
        if self._topic and self._group:
            self.subscribe(self._topic, self._group)

    def _heads(self, c: sqlite3.Connection, topic: str) -> dict[int, int]:
        """Next offset per partition.

        One MAX() per partition with equality on (topic, partition) is an O(log n) seek on the
        primary key. A single `GROUP BY partition` looks equivalent but scans the whole topic:
        at ~700k messages that scan ran under the write lock on every produce and starved the
        consumer (batches went from 0.7 s to 25 s). Found in load testing; keep it this way.
        """
        heads = {}
        for p in range(self.n_partitions):
            row = c.execute("SELECT MAX(offset) FROM messages WHERE topic = ? AND partition = ?", (topic, p)).fetchone()
            heads[p] = 0 if row[0] is None else row[0] + 1
        return heads

    def end_offsets(self, topic: str) -> dict[int, int]:
        return self._heads(self._conn(), topic)

    def lag(self) -> dict[int, int]:
        if self._topic is None:
            return {}
        ends = self.end_offsets(self._topic)
        committed = dict(self._conn().execute(
            "SELECT partition, offset FROM consumer_offsets WHERE grp = ? AND topic = ?",
            (self._group, self._topic)).fetchall())
        return {p: max(0, ends[p] - committed.get(p, 0)) for p in ends}

    def apply_retention(self, topic: str, max_age_s: float) -> int:
        """Delete records older than max_age_s that every consumer group has already read."""
        c = self._conn()
        cutoff = time.time() - max_age_s
        deleted = 0
        for p in range(self.n_partitions):
            row = c.execute("SELECT MIN(offset) FROM consumer_offsets WHERE topic = ? AND partition = ?",
                            (topic, p)).fetchone()
            head = c.execute("SELECT MAX(offset) FROM messages WHERE topic = ? AND partition = ?",
                             (topic, p)).fetchone()[0]
            if head is None:
                continue
            # Never delete the newest record: offsets are derived from MAX(offset), so an empty
            # partition would restart at 0 and break every consumer's position.
            safe = min(row[0] if row and row[0] is not None else 0, head)
            cur = c.execute("DELETE FROM messages WHERE topic = ? AND partition = ? AND ts < ? AND offset < ?",
                            (topic, p, cutoff, safe))
            deleted += cur.rowcount
        return deleted

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# =========================================================================== #
# Kafka adapter
# =========================================================================== #
class KafkaBroker(Broker):
    """Same contract on Apache Kafka via confluent-kafka (librdkafka)."""

    def __init__(self, bootstrap: str = KAFKA_BOOTSTRAP):
        try:
            from confluent_kafka import Consumer, Producer  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("BROKER_BACKEND=kafka requires `pip install confluent-kafka`") from exc
        from confluent_kafka import Producer

        self._bootstrap = bootstrap
        self._producer = Producer({
            "bootstrap.servers": bootstrap,
            "enable.idempotence": True,      # exactly-once on the produce side
            "acks": "all",
            "linger.ms": 20,                  # small batching window -> higher throughput
            "compression.type": "lz4",
        })
        self._consumer = None
        self._topic: str | None = None

    def produce_batch(self, topic: str, records: Iterable[tuple[str | None, str]]) -> int:
        n = 0
        for key, value in records:
            while True:
                try:
                    self._producer.produce(topic, key=key, value=value)
                    break
                except BufferError:  # local queue full: serve delivery callbacks, then retry
                    self._producer.poll(0.1)
            n += 1
        self._producer.flush(10)
        return n

    def subscribe(self, topic: str, group: str) -> None:
        from confluent_kafka import Consumer

        self._consumer = Consumer({
            "bootstrap.servers": self._bootstrap,
            "group.id": group,
            "enable.auto.commit": False,      # commit only after the lake write succeeds
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 300000,
        })
        self._consumer.subscribe([topic])
        self._topic = topic

    def poll(self, max_records: int = 1000) -> list[Message]:
        msgs = self._consumer.consume(num_messages=max_records, timeout=1.0)
        out = []
        for m in msgs:
            if m.error():
                log.warning("Kafka error: %s", m.error())
                continue
            key = m.key().decode() if m.key() else None
            out.append(Message(m.topic(), m.partition(), m.offset(), key, m.value().decode(), m.timestamp()[1] / 1000))
        return out

    def commit(self) -> None:
        if self._consumer is not None:
            self._consumer.commit(asynchronous=False)

    def lag(self) -> dict[int, int]:
        from confluent_kafka import TopicPartition

        if self._consumer is None:
            return {}
        out = {}
        for tp in self._consumer.assignment():
            lo, hi = self._consumer.get_watermark_offsets(tp, timeout=5)
            committed = self._consumer.committed([TopicPartition(tp.topic, tp.partition)], timeout=5)[0].offset
            out[tp.partition] = hi - (committed if committed >= 0 else lo)
        return out

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
        self._producer.flush(10)


def get_broker() -> Broker:
    """Factory: pick the backend from BROKER_BACKEND."""
    if BROKER_BACKEND == "kafka":
        return KafkaBroker()
    return SQLiteBroker()
