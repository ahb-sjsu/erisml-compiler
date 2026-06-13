"""SQLite-backed habit store (production-grade replacement for JSON Lines).

The v1 `HabitStore` in `habit_store.py` appends to JSON Lines files —
simple, audit-friendly, but no concurrency safety, no indexed reads,
no schema migrations. This module replaces it with SQLite (stdlib,
no extra deps) running in WAL mode for concurrent readers + a single
writer.

Schema (versioned):

    CREATE TABLE schema_version (
        version INTEGER PRIMARY KEY
    );

    CREATE TABLE act_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id        TEXT    NOT NULL,
        case_id         TEXT    NOT NULL,
        timestamp_utc   TEXT    NOT NULL,
        action_kind     TEXT    NOT NULL,
        virtue_axis     TEXT    NOT NULL,
        polarity        INTEGER NOT NULL,
        case_severity   TEXT,
        maxim_description TEXT,
        case_severity_weight REAL DEFAULT 1.0,
        UNIQUE(agent_id, case_id, timestamp_utc)
    );

    CREATE INDEX idx_records_agent_time
        ON act_records(agent_id, timestamp_utc);

WAL mode (`PRAGMA journal_mode=WAL`) allows concurrent readers
without blocking the writer. Atomic transactions guarantee that
either the record is fully appended or not at all.

The store is **API-compatible** with the JSON Lines store — same
`append`, `read_history`, `known_agents`, `assess` methods — so
code switching from one to the other doesn't need to change.

When the user wants to migrate from a JSON Lines store, the
`migrate_from_jsonl(jsonl_root, sqlite_path)` helper bulk-loads
the old data into the new schema.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from erisml_compiler.history.habit_store import (
    ActRecord,
    HabitStore,
    VirtueAssessment,
    assess_virtue_history,
)

_CURRENT_SCHEMA_VERSION = 1


_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS act_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id             TEXT    NOT NULL,
    case_id              TEXT    NOT NULL,
    timestamp_utc        TEXT    NOT NULL,
    action_kind          TEXT    NOT NULL,
    virtue_axis          TEXT    NOT NULL,
    polarity             INTEGER NOT NULL,
    case_severity        TEXT,
    case_severity_weight REAL    NOT NULL DEFAULT 1.0,
    maxim_description    TEXT,
    UNIQUE(agent_id, case_id, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_records_agent_time
    ON act_records(agent_id, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_records_axis
    ON act_records(agent_id, virtue_axis);
"""


class SqliteHabitStore:
    """SQLite-backed agent history. API-compatible with HabitStore."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---------------------------------------------------- connection

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit off — manage txns explicitly
            timeout=10.0,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _txn(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    # ---------------------------------------------------- schema

    def _init_schema(self) -> None:
        # executescript commits its own transactions, so we don't wrap
        # it in our explicit BEGIN/COMMIT.
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_V1)
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)",
                    (_CURRENT_SCHEMA_VERSION,),
                )
        finally:
            conn.close()

    def schema_version(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            return int(row["version"]) if row else 0

    # ---------------------------------------------------- mutation

    def append(self, record: ActRecord) -> None:
        """Append one record. Atomic; uses UPSERT to handle re-runs
        idempotently (same agent_id+case_id+timestamp updates the row
        rather than producing a duplicate)."""
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO act_records (
                    agent_id, case_id, timestamp_utc, action_kind,
                    virtue_axis, polarity, case_severity,
                    case_severity_weight, maxim_description
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(agent_id, case_id, timestamp_utc)
                DO UPDATE SET
                    action_kind=excluded.action_kind,
                    virtue_axis=excluded.virtue_axis,
                    polarity=excluded.polarity,
                    case_severity=excluded.case_severity,
                    case_severity_weight=excluded.case_severity_weight,
                    maxim_description=excluded.maxim_description
                """,
                (
                    record.agent_id,
                    record.case_id,
                    record.timestamp_utc,
                    record.action_kind,
                    record.virtue_axis,
                    int(record.polarity),
                    record.case_severity,
                    _severity_weight(record.case_severity),
                    record.maxim_description,
                ),
            )

    def append_many(self, records: Iterable[ActRecord]) -> None:
        """Bulk append in a single transaction."""
        rows = [
            (
                r.agent_id,
                r.case_id,
                r.timestamp_utc,
                r.action_kind,
                r.virtue_axis,
                int(r.polarity),
                r.case_severity,
                _severity_weight(r.case_severity),
                r.maxim_description,
            )
            for r in records
        ]
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT INTO act_records (
                    agent_id, case_id, timestamp_utc, action_kind,
                    virtue_axis, polarity, case_severity,
                    case_severity_weight, maxim_description
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(agent_id, case_id, timestamp_utc) DO NOTHING
                """,
                rows,
            )

    # ---------------------------------------------------- read

    def read_history(self, agent_id: str) -> list[ActRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_id, case_id, timestamp_utc, action_kind,
                       virtue_axis, polarity, case_severity,
                       maxim_description
                FROM act_records
                WHERE agent_id = ?
                ORDER BY timestamp_utc, case_id
                """,
                (agent_id,),
            ).fetchall()
        return [
            ActRecord(
                agent_id=r["agent_id"],
                case_id=r["case_id"],
                timestamp_utc=r["timestamp_utc"],
                action_kind=r["action_kind"],
                virtue_axis=r["virtue_axis"],
                polarity=int(r["polarity"]),
                maxim_description=r["maxim_description"] or "",
                case_severity=r["case_severity"],
            )
            for r in rows
        ]

    def known_agents(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT agent_id FROM act_records ORDER BY agent_id"
            ).fetchall()
        return [r["agent_id"] for r in rows]

    def assess(self, agent_id: str) -> VirtueAssessment:
        records = self.read_history(agent_id)
        return assess_virtue_history(agent_id, records)

    def n_records(self, agent_id: str | None = None) -> int:
        with self._connect() as conn:
            if agent_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM act_records WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM act_records").fetchone()
            return int(row["n"])


# ----------------------------------------------------------------- severity


_SEVERITY_WEIGHT_MAP = {
    "minor": 0.5,
    "moderate": 1.0,
    "grave": 2.0,
    "catastrophic": 4.0,
}


def _severity_weight(severity: str | None) -> float:
    if severity is None:
        return 1.0
    return _SEVERITY_WEIGHT_MAP.get(severity.lower(), 1.0)


# ----------------------------------------------------------------- migration


def migrate_from_jsonl(jsonl_root: str | Path, sqlite_path: str | Path) -> int:
    """Bulk-load every JSON Lines store under `jsonl_root` into a new
    SQLite store. Returns the number of records imported."""
    src = HabitStore(jsonl_root)
    dst = SqliteHabitStore(sqlite_path)
    total = 0
    for agent_id in src.known_agents():
        records = src.read_history(agent_id)
        dst.append_many(records)
        total += len(records)
    return total
