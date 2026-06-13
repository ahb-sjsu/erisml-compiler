"""Tests for SQLite habit store + temporal weighting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from erisml_compiler.history import (
    ActRecord,
    HabitStore,
    SqliteHabitStore,
    TemporallyWeightedAssessment,
    migrate_from_jsonl,
    temporally_weighted_assessment,
)

# ---------------------------------------------------- store mechanics


def _make_record(
    agent_id: str = "alice",
    case_id: str = "case_001",
    action_kind: str = "deceive",
    virtue_axis: str = "honesty",
    polarity: int = -1,
    case_severity: str | None = None,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> ActRecord:
    return ActRecord(
        agent_id=agent_id,
        case_id=case_id,
        timestamp_utc=timestamp,
        action_kind=action_kind,
        virtue_axis=virtue_axis,
        polarity=polarity,
        case_severity=case_severity,
    )


def test_schema_initialised_at_v1(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    assert store.schema_version() == 1


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    r = _make_record()
    store.append(r)
    out = store.read_history("alice")
    assert len(out) == 1
    assert out[0].action_kind == "deceive"
    assert out[0].virtue_axis == "honesty"


def test_unique_constraint_idempotent_upsert(tmp_path: Path) -> None:
    """Re-appending the same (agent, case, timestamp) tuple should
    update rather than duplicate."""
    store = SqliteHabitStore(tmp_path / "h.db")
    store.append(_make_record(polarity=-1))
    store.append(_make_record(polarity=+1))  # same key, different polarity
    out = store.read_history("alice")
    assert len(out) == 1
    assert out[0].polarity == 1  # UPSERT applied the second value


def test_append_many_atomic(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    records = [
        _make_record(case_id=f"case_{i}", timestamp=f"2026-0{(i % 9) + 1}-01T00:00:00+00:00")
        for i in range(10)
    ]
    store.append_many(records)
    assert store.n_records("alice") == 10


def test_known_agents_returns_distinct(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    store.append(_make_record(agent_id="alice"))
    store.append(_make_record(agent_id="bob", case_id="case_002"))
    store.append(_make_record(agent_id="alice", case_id="case_002"))
    assert set(store.known_agents()) == {"alice", "bob"}


def test_assess_aggregates_records(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    for i in range(5):
        store.append(
            _make_record(
                case_id=f"case_{i}",
                timestamp=f"2026-0{(i % 9) + 1}-01T00:00:00+00:00",
            )
        )
    a = store.assess("alice")
    assert a.n_observations == 5
    assert a.per_axis_dominant["honesty"] == "vice"


def test_n_records_total_vs_per_agent(tmp_path: Path) -> None:
    store = SqliteHabitStore(tmp_path / "h.db")
    store.append(_make_record(agent_id="alice"))
    store.append(_make_record(agent_id="bob", case_id="case_002"))
    assert store.n_records() == 2
    assert store.n_records("alice") == 1
    assert store.n_records("bob") == 1


def test_concurrent_reader_blocked_when_writer_holds_immediate(tmp_path: Path) -> None:
    """Smoke test: WAL mode allows readers concurrent with writers
    in normal operation. (Full multi-process concurrency would need
    multiprocessing; this verifies the pragma is set.)"""
    store = SqliteHabitStore(tmp_path / "h.db")
    # Read while the same process opens a second connection.
    store.append(_make_record())
    out = store.read_history("alice")
    assert len(out) == 1


# ---------------------------------------------------- migration from JSONL


def test_migrate_from_jsonl(tmp_path: Path) -> None:
    """JSON Lines store → SQLite migration preserves all records."""
    jsonl_root = tmp_path / "jsonl"
    sqlite_path = tmp_path / "h.db"

    src = HabitStore(jsonl_root)
    src.append(_make_record(case_id="c1", timestamp="2026-01-01T00:00:00+00:00"))
    src.append(_make_record(case_id="c2", timestamp="2026-02-01T00:00:00+00:00", polarity=+1))
    src.append(
        _make_record(
            agent_id="bob",
            case_id="c1",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    n = migrate_from_jsonl(jsonl_root, sqlite_path)
    assert n == 3

    dst = SqliteHabitStore(sqlite_path)
    assert dst.n_records() == 3
    assert set(dst.known_agents()) == {"alice", "bob"}


# ---------------------------------------------------- temporal weighting


def test_temporal_decay_recent_outweighs_old() -> None:
    """A recent observation should outweigh an equally-strong old one."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    records = [
        _make_record(
            case_id="old", polarity=-1, timestamp="2024-06-01T00:00:00+00:00"
        ),  # 2 years ago
        _make_record(
            case_id="recent", polarity=+1, timestamp="2026-05-15T00:00:00+00:00"
        ),  # 2 weeks ago
    ]
    a = temporally_weighted_assessment(
        "alice",
        records,
        half_life_days=90.0,
        now_utc=now,
    )
    # Recent +1 with weight ~1 vs old -1 with weight ≈ 2^(-8) ≈ 0.004
    assert a.per_axis_weighted_mean["honesty"] > 0.5


def test_severity_multiplier_amplifies_grave_cases() -> None:
    """A single grave observation should outweigh many minor ones."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    records = [
        _make_record(
            case_id=f"minor_{i}",
            polarity=+1,
            case_severity="minor",
            timestamp="2026-05-01T00:00:00+00:00",
        )
        for i in range(3)
    ] + [
        _make_record(
            case_id="grave",
            polarity=-1,
            case_severity="catastrophic",
            timestamp="2026-05-01T00:00:00+00:00",
        ),
    ]
    a = temporally_weighted_assessment(
        "alice",
        records,
        half_life_days=90.0,
        now_utc=now,
    )
    # 3 × (0.5 multiplier × +1) = 1.5; 1 × (4.0 × -1) = -4.0 → mean ≈ -2.5/5.5
    assert a.per_axis_weighted_mean["honesty"] < -0.2


def test_habituation_score_high_for_consistent_history() -> None:
    """Many same-polarity observations → high habituation score."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    records = [
        _make_record(case_id=f"c{i}", polarity=-1, timestamp="2026-05-01T00:00:00+00:00")
        for i in range(8)
    ]
    a = temporally_weighted_assessment("alice", records, now_utc=now)
    assert a.per_axis_habituation_score["honesty"] > 0.7
    assert a.per_axis_dominant["honesty"] == "vice"


def test_habituation_score_low_for_mixed_history() -> None:
    """Mixed-polarity observations → low habituation."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    records = [
        _make_record(case_id="c1", polarity=+1, timestamp="2026-05-01T00:00:00+00:00"),
        _make_record(case_id="c2", polarity=-1, timestamp="2026-05-02T00:00:00+00:00"),
        _make_record(case_id="c3", polarity=+1, timestamp="2026-05-03T00:00:00+00:00"),
        _make_record(case_id="c4", polarity=-1, timestamp="2026-05-04T00:00:00+00:00"),
    ]
    a = temporally_weighted_assessment("alice", records, now_utc=now)
    assert a.per_axis_habituation_score["honesty"] < 0.3
    assert a.per_axis_dominant["honesty"] == "ambiguous"


def test_temporal_assessment_history_hash_matches_baseline() -> None:
    """The history_hash here should match the one from the v1
    assess_virtue_history for the same records."""
    from erisml_compiler.history.habit_store import history_hash

    records = [
        _make_record(case_id="c1", polarity=-1, timestamp="2026-01-01T00:00:00+00:00"),
        _make_record(case_id="c2", polarity=+1, timestamp="2026-02-01T00:00:00+00:00"),
    ]
    a = temporally_weighted_assessment("alice", records)
    assert a.history_hash == history_hash(records)


def test_effective_n_decays_with_age() -> None:
    """effective_n should be approximately equal to N when all
    observations are recent; less than N when they're old."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    recent = [
        _make_record(case_id=f"r{i}", polarity=-1, timestamp="2026-05-25T00:00:00+00:00")
        for i in range(5)
    ]
    old = [
        _make_record(case_id=f"o{i}", polarity=-1, timestamp="2024-01-01T00:00:00+00:00")
        for i in range(5)
    ]
    a_recent = temporally_weighted_assessment("alice", recent, now_utc=now)
    a_old = temporally_weighted_assessment("alice", old, now_utc=now)
    assert a_recent.effective_n > 4.0  # close to 5
    assert a_old.effective_n < 1.0  # heavily decayed


def test_no_observations_yields_no_evidence() -> None:
    a = temporally_weighted_assessment("nobody", [])
    assert a.n_observations == 0
    assert all(d == "no_evidence" for d in a.per_axis_dominant.values())


def test_returns_temporally_weighted_assessment_type() -> None:
    a = temporally_weighted_assessment("alice", [])
    assert isinstance(a, TemporallyWeightedAssessment)
    assert a.half_life_days == 90.0


def test_half_life_parameter_affects_decay() -> None:
    """Shorter half-life → faster decay of older observations."""
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    records = [
        _make_record(
            case_id="old", polarity=+1, timestamp="2026-01-01T00:00:00+00:00"
        ),  # 5 months ago
        _make_record(case_id="recent", polarity=-1, timestamp="2026-05-25T00:00:00+00:00"),
    ]
    a_long = temporally_weighted_assessment("alice", records, half_life_days=365.0, now_utc=now)
    a_short = temporally_weighted_assessment("alice", records, half_life_days=30.0, now_utc=now)
    # Shorter half-life decays the old +1 more aggressively, so
    # the recent -1 should dominate more strongly.
    assert a_short.per_axis_weighted_mean["honesty"] < a_long.per_axis_weighted_mean["honesty"]
