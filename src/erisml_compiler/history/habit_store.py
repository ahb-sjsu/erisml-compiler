"""Filesystem-backed append-only history of an agent's acts.

Each `ActRecord` captures one observation of an agent: the maxim's
action_kind, the virtue/vice axis it sits on, the polarity
(`+1` virtuous, `0` ambiguous, `-1` vicious), the source case_id,
and an iso-8601 timestamp. The store reads/writes JSON Lines so
the file is human-inspectable + audit-friendly.

Aggregation across the history produces a `VirtueAssessment`:
  - `n_observations`
  - per-axis mean polarity (a coarse "trait score")
  - per-axis dispersion (a coarse "trait consistency")
  - the dominant virtue / vice on each axis if the trait score is
    far enough from zero

The trait scores are intentionally simple. A richer implementation
would weight recent observations more heavily, account for situation
severity, and distinguish habituation from one-off action. v1 is a
running mean — defensible as a baseline, transparent as a method.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Virtue axes (canonical 6) — also used by VirtueProjection's
# action-kind → axis mapping.
VIRTUE_AXES: tuple[tuple[str, str], ...] = (
    ("honesty", "deception"),
    ("courage", "cowardice"),
    ("justice", "injustice"),
    ("fidelity", "perfidy"),
    ("care", "callousness"),
    ("prudence", "imprudence"),
)


# Maps maxim action_kinds to (virtue, vice) axes — mirrors the
# `_ACTION_VIRTUE_AXES` in projections/virtue.py but extended for
# the larger action_kind set the prose-based maxim extractor emits.
_ACTION_KIND_TO_AXIS: dict[str, tuple[str, str, int]] = {
    # action_kind → (virtue_pole, vice_pole, polarity_evidence)
    # polarity_evidence: +1 if the act expresses the virtue,
    # -1 if it expresses the vice, 0 if ambiguous-without-context.
    "deceive":                   ("honesty",  "deception",   -1),
    "break_commitment":          ("fidelity", "perfidy",     -1),
    "make_or_keep_commitment":   ("fidelity", "perfidy",     +1),
    "protect":                   ("care",     "callousness", +1),
    "help":                      ("care",     "callousness", +1),
    "refuse":                    ("care",     "callousness", -1),
    "inflict_harm":              ("care",     "callousness", -1),
    "coerce":                    ("courage",  "cowardice",   -1),  # coercion is a cowardly use of force
    "coerce_or_be_coerced":      ("courage",  "cowardice",    0),
    "impose_externality":        ("justice",  "injustice",   -1),
    "cheat":                     ("justice",  "injustice",   -1),
    "disclose":                  ("courage",  "cowardice",   +1),  # whistleblower reading
    "use_as_means":              ("justice",  "injustice",   -1),
    "act_under_norm":            ("prudence", "imprudence",  +1),
    "act_under_authority":      ("prudence", "imprudence",   0),
}


@dataclass(frozen=True)
class ActRecord:
    """One observation of an agent's act, anchored to a case."""

    agent_id: str
    case_id: str
    """Stable identifier for the case (typically the IR's source_text_hash
    or the document doc_id)."""
    timestamp_utc: str
    action_kind: str
    virtue_axis: str  # one of {honesty, courage, justice, fidelity, care, prudence}
    polarity: int  # +1 virtuous, 0 ambiguous, -1 vicious
    maxim_description: str = ""
    case_severity: str | None = None  # passed through if known


@dataclass(frozen=True)
class VirtueAssessment:
    """Aggregated trait reading for one agent across N observations."""

    agent_id: str
    n_observations: int
    per_axis_mean_polarity: dict[str, float]
    per_axis_dispersion: dict[str, float]
    per_axis_dominant: dict[str, str]
    """axis → "virtue" if mean > +0.3, "vice" if < -0.3, else "ambiguous"."""

    history_hash: str
    """SHA-256 over the canonical history JSON. Recorded in audit so
    a reviewer can confirm which history the assessment was computed
    against."""


# ---------------------------------------------------------------------- store


class HabitStore:
    """Append-only JSON-Lines store keyed by agent_id.

    Each agent's history lives at `<root>/<agent_id>.jsonl`. The
    store is filesystem-backed because v1 wants the simplest
    auditable persistence; future versions can swap in SQLite.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------- mutation

    def append(self, record: ActRecord) -> None:
        path = self._path_for(record.agent_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_dict(record), separators=(",", ":")) + "\n")

    def append_many(self, records: Iterable[ActRecord]) -> None:
        for r in records:
            self.append(r)

    # --------------------------------------- read

    def read_history(self, agent_id: str) -> list[ActRecord]:
        path = self._path_for(agent_id)
        if not path.exists():
            return []
        out: list[ActRecord] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(_from_dict(json.loads(line)))
        return out

    def known_agents(self) -> list[str]:
        return sorted(
            p.stem for p in self.root.glob("*.jsonl") if p.is_file()
        )

    # --------------------------------------- aggregation

    def assess(self, agent_id: str) -> VirtueAssessment:
        records = self.read_history(agent_id)
        return assess_virtue_history(agent_id, records)

    # --------------------------------------- utility

    def _path_for(self, agent_id: str) -> Path:
        safe = "".join(c for c in agent_id if c.isalnum() or c in "-_.")
        return self.root / f"{safe}.jsonl"


# ---------------------------------------------------------------------- agg


def assess_virtue_history(agent_id: str, records: list[ActRecord]) -> VirtueAssessment:
    """Aggregate `records` into a `VirtueAssessment`."""
    axis_buckets: dict[str, list[int]] = {v: [] for v, _ in VIRTUE_AXES}
    for r in records:
        if r.virtue_axis in axis_buckets:
            axis_buckets[r.virtue_axis].append(r.polarity)

    per_axis_mean: dict[str, float] = {}
    per_axis_dispersion: dict[str, float] = {}
    per_axis_dominant: dict[str, str] = {}
    for axis, polarities in axis_buckets.items():
        if not polarities:
            per_axis_mean[axis] = 0.0
            per_axis_dispersion[axis] = 0.0
            per_axis_dominant[axis] = "no_evidence"
            continue
        m = statistics.fmean(polarities)
        d = statistics.pstdev(polarities) if len(polarities) > 1 else 0.0
        per_axis_mean[axis] = round(m, 4)
        per_axis_dispersion[axis] = round(d, 4)
        if m > 0.3:
            per_axis_dominant[axis] = "virtue"
        elif m < -0.3:
            per_axis_dominant[axis] = "vice"
        else:
            per_axis_dominant[axis] = "ambiguous"

    return VirtueAssessment(
        agent_id=agent_id,
        n_observations=len(records),
        per_axis_mean_polarity=per_axis_mean,
        per_axis_dispersion=per_axis_dispersion,
        per_axis_dominant=per_axis_dominant,
        history_hash=history_hash(records),
    )


def history_hash(records: list[ActRecord]) -> str:
    """Canonical SHA-256 over the history. Sorted by (timestamp,
    case_id) for determinism."""
    canon = sorted(
        (_to_dict(r) for r in records),
        key=lambda d: (d.get("timestamp_utc", ""), d.get("case_id", "")),
    )
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------- helpers


def record_from_compile(agent_id: str, ir, case_id: str | None = None) -> ActRecord | None:
    """Construct an ActRecord from a just-compiled IR.

    Returns None when the IR doesn't surface a maxim's action_kind we
    know how to classify on the virtue axes.
    """
    from erisml_compiler.ir.graph import NodeKind

    if ir.graph is None:
        return None
    maxim_nodes = ir.graph.nodes_of_kind(NodeKind.MAXIM)
    if not maxim_nodes:
        return None
    payload = maxim_nodes[0].payload or {}
    action_kind = payload.get("action_kind") or ""
    if action_kind not in _ACTION_KIND_TO_AXIS:
        return None
    virtue, _vice, polarity = _ACTION_KIND_TO_AXIS[action_kind]
    case_id = case_id or (
        ir.audit.source_text_hash[:16] if ir.audit else ir.document.doc_id
    )
    return ActRecord(
        agent_id=agent_id,
        case_id=case_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        action_kind=action_kind,
        virtue_axis=virtue,
        polarity=polarity,
        maxim_description=payload.get("description") or "",
    )


def _to_dict(r: ActRecord) -> dict:
    return {
        "agent_id": r.agent_id,
        "case_id": r.case_id,
        "timestamp_utc": r.timestamp_utc,
        "action_kind": r.action_kind,
        "virtue_axis": r.virtue_axis,
        "polarity": r.polarity,
        "maxim_description": r.maxim_description,
        "case_severity": r.case_severity,
    }


def _from_dict(d: dict) -> ActRecord:
    return ActRecord(
        agent_id=d["agent_id"],
        case_id=d["case_id"],
        timestamp_utc=d["timestamp_utc"],
        action_kind=d["action_kind"],
        virtue_axis=d["virtue_axis"],
        polarity=int(d["polarity"]),
        maxim_description=d.get("maxim_description", ""),
        case_severity=d.get("case_severity"),
    )
