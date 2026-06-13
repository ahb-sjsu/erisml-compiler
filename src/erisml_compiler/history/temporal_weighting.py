"""Temporally-weighted virtue assessment.

The v1 `assess_virtue_history` in `habit_store.py` uses a running
mean across all observations. That's defensible as a baseline but
ignores three things virtue ethics actually cares about:

  1. **Recency.** A pattern from ten years ago says less about
     current character than a pattern from last week. Aristotle:
     character is built by present habituation, not fossilised
     into a permanent score.
  2. **Severity.** A minor lie in passing is weak evidence; a
     grave deception over a high-stakes case is strong evidence.
     The case_severity (when known) modulates the contribution.
  3. **Habituation vs one-off.** High consistency in a direction
     is genuine habit; high dispersion is mixed-signal that
     should produce a less confident verdict, not an averaged-out
     "ambiguous" reading.

This module produces a `TemporallyWeightedAssessment` that
substitutes for the v1 aggregation. The decay/severity formulas
are pure functions with no SQL or filesystem dependency, so they
work with both the JSON Lines store and the SQLite store.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from erisml_compiler.history.habit_store import (
    VIRTUE_AXES,
    ActRecord,
    history_hash,
)

# Severity multipliers — matches the SqliteHabitStore's severity weight.
_SEVERITY_MULTIPLIER = {
    None: 1.0,
    "minor": 0.5,
    "moderate": 1.0,
    "grave": 2.0,
    "catastrophic": 4.0,
}


@dataclass(frozen=True)
class TemporallyWeightedAssessment:
    """Temporally-weighted aggregate over an agent's history."""

    agent_id: str
    n_observations: int
    effective_n: float
    """Sum of weights — measures 'how much evidence after weighting'."""

    per_axis_weighted_mean: dict[str, float]
    """Weight-aware mean polarity. Recent + severe observations
    contribute more."""

    per_axis_weighted_dispersion: dict[str, float]
    """Weight-aware dispersion — closer to zero = entrenched habit;
    closer to 1 = mixed signal."""

    per_axis_habituation_score: dict[str, float]
    """In [0, 1]. High → consistent direction across many observations
    (entrenched habit). Low → either ambiguous or low-evidence."""

    per_axis_dominant: dict[str, str]
    """axis → 'virtue' | 'vice' | 'ambiguous' | 'no_evidence'."""

    half_life_days: float
    """The exponential-decay half-life used for this assessment."""

    history_hash: str
    """SHA-256 over the canonical history."""


def temporally_weighted_assessment(
    agent_id: str,
    records: list[ActRecord],
    *,
    half_life_days: float = 90.0,
    now_utc: datetime | None = None,
    habit_threshold: float = 0.3,
) -> TemporallyWeightedAssessment:
    """Compute a temporally-weighted assessment of agent character.

    Args:
        agent_id: who is being assessed.
        records: the agent's act history.
        half_life_days: exponential decay half-life. Default 90 days
            — recent quarter weighs ≈2× the quarter before.
        now_utc: current time (used for the recency baseline). When
            None, uses `datetime.now(timezone.utc)`.
        habit_threshold: |weighted_mean| > threshold AND dispersion
            below the dispersion cap is required to mark a virtue/vice.
    """
    now = now_utc if now_utc is not None else datetime.now(timezone.utc)

    axis_buckets: dict[str, list[tuple[float, int]]] = {v: [] for v, _ in VIRTUE_AXES}
    for r in records:
        if r.virtue_axis not in axis_buckets:
            continue
        w = _weight(r, now=now, half_life_days=half_life_days)
        axis_buckets[r.virtue_axis].append((w, r.polarity))

    weighted_mean: dict[str, float] = {}
    weighted_dispersion: dict[str, float] = {}
    habituation: dict[str, float] = {}
    dominant: dict[str, str] = {}

    effective_n_total = 0.0

    for axis, observations in axis_buckets.items():
        if not observations:
            weighted_mean[axis] = 0.0
            weighted_dispersion[axis] = 0.0
            habituation[axis] = 0.0
            dominant[axis] = "no_evidence"
            continue

        weights = [w for w, _ in observations]
        values = [p for _, p in observations]
        total_w = sum(weights)
        effective_n_total += total_w

        if total_w < 1e-9:
            weighted_mean[axis] = 0.0
            weighted_dispersion[axis] = 0.0
            habituation[axis] = 0.0
            dominant[axis] = "no_evidence"
            continue

        mean = sum(w * v for w, v in observations) / total_w

        # Weighted population variance.
        var = sum(w * (v - mean) ** 2 for w, v in observations) / total_w
        std = math.sqrt(var)

        # Habituation: high mean magnitude + low dispersion + enough weight.
        # Smooth saturation function in [0, 1].
        evidence_factor = 1.0 - math.exp(-total_w / 3.0)  # ~63% at total_w=3
        consistency = 1.0 - min(std, 1.0)  # 1.0 - dispersion
        magnitude = min(abs(mean), 1.0)
        hab = round(magnitude * consistency * evidence_factor, 4)

        weighted_mean[axis] = round(mean, 4)
        weighted_dispersion[axis] = round(std, 4)
        habituation[axis] = hab

        # Dominant classification: weighted mean above threshold AND
        # enough effective evidence (total_w >= 1.0 ≈ at least one
        # full-weight observation OR several decayed ones).
        if total_w < 1.0:
            dominant[axis] = "no_evidence" if mean == 0 else "ambiguous"
        elif mean > habit_threshold:
            dominant[axis] = "virtue"
        elif mean < -habit_threshold:
            dominant[axis] = "vice"
        else:
            dominant[axis] = "ambiguous"

    return TemporallyWeightedAssessment(
        agent_id=agent_id,
        n_observations=len(records),
        effective_n=round(effective_n_total, 4),
        per_axis_weighted_mean=weighted_mean,
        per_axis_weighted_dispersion=weighted_dispersion,
        per_axis_habituation_score=habituation,
        per_axis_dominant=dominant,
        half_life_days=half_life_days,
        history_hash=history_hash(records),
    )


def _weight(
    record: ActRecord,
    *,
    now: datetime,
    half_life_days: float,
) -> float:
    """Per-record weight = exponential time-decay × severity multiplier."""
    age_days = _age_in_days(record.timestamp_utc, now=now)
    # exp(-ln(2) * age / half_life) → halves each half_life_days.
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    severity_mult = _SEVERITY_MULTIPLIER.get((record.case_severity or "").lower() or None, 1.0)
    return decay * severity_mult


def _age_in_days(timestamp_utc: str, *, now: datetime) -> float:
    """Compute the age of an ISO-8601 UTC timestamp in days. Returns
    0.0 for malformed timestamps (failsafe: treats as current)."""
    try:
        t = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    delta = now - t
    return max(0.0, delta.total_seconds() / 86400.0)
