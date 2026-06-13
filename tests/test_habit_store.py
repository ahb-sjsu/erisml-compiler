"""Tests for the longitudinal habit store + LongitudinalVirtueProjection."""

from __future__ import annotations

from pathlib import Path

import pytest

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.history import (
    ActRecord,
    HabitStore,
    LongitudinalVirtueProjection,
    VirtueAssessment,
)
from erisml_compiler.history.habit_store import (
    _ACTION_KIND_TO_AXIS,
    VIRTUE_AXES,
    assess_virtue_history,
    history_hash,
    record_from_compile,
)
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.projections import substrate_from_ir
from erisml_compiler.tiers import CompilerTier


REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------- knowledge base


def test_all_maxim_extractor_action_kinds_have_axis_mapping() -> None:
    """Every action_kind the maxim extractor can emit must map to an
    axis so longitudinal tracking has somewhere to put it."""
    from erisml_compiler.annotation.maxim_extractor import _VERB_PATTERNS

    emitted = {kind for _, kind in _VERB_PATTERNS}
    mapped = set(_ACTION_KIND_TO_AXIS.keys())
    missing = emitted - mapped
    assert not missing, f"Action kinds missing axis mapping: {missing}"


# ----------------------------------------------------- store mechanics


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    r = ActRecord(
        agent_id="alice",
        case_id="case_001",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        action_kind="deceive",
        virtue_axis="honesty",
        polarity=-1,
    )
    store.append(r)
    out = store.read_history("alice")
    assert len(out) == 1
    assert out[0].action_kind == "deceive"


def test_read_history_missing_agent_returns_empty(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    assert store.read_history("nobody") == []


def test_known_agents(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    store.append(ActRecord("alice", "c", "t", "protect", "care", +1))
    store.append(ActRecord("bob", "c", "t", "deceive", "honesty", -1))
    assert set(store.known_agents()) == {"alice", "bob"}


def test_history_hash_deterministic() -> None:
    records = [
        ActRecord("a", "c1", "2026-01-01T00:00:00", "protect", "care", +1),
        ActRecord("a", "c2", "2026-01-02T00:00:00", "deceive", "honesty", -1),
    ]
    h1 = history_hash(records)
    h2 = history_hash(list(reversed(records)))
    assert h1 == h2
    assert len(h1) == 64


def test_history_hash_changes_with_content() -> None:
    base = [ActRecord("a", "c1", "t", "protect", "care", +1)]
    h1 = history_hash(base)
    h2 = history_hash(base + [ActRecord("a", "c2", "t", "deceive", "honesty", -1)])
    assert h1 != h2


# ----------------------------------------------------- aggregation


def test_assess_empty_history() -> None:
    a = assess_virtue_history("nobody", [])
    assert a.n_observations == 0
    assert all(v == "no_evidence" for v in a.per_axis_dominant.values())


def test_assess_classifies_entrenched_vice() -> None:
    records = [
        ActRecord("a", f"c{i}", f"2026-01-0{i+1}T00:00:00", "deceive", "honesty", -1)
        for i in range(5)
    ]
    a = assess_virtue_history("a", records)
    assert a.n_observations == 5
    assert a.per_axis_dominant["honesty"] == "vice"
    assert a.per_axis_mean_polarity["honesty"] == -1.0
    assert a.per_axis_dispersion["honesty"] == 0.0


def test_assess_classifies_established_virtue() -> None:
    records = [
        ActRecord("a", f"c{i}", f"2026-01-0{i+1}T00:00:00", "protect", "care", +1)
        for i in range(4)
    ]
    a = assess_virtue_history("a", records)
    assert a.per_axis_dominant["care"] == "virtue"


def test_assess_handles_ambiguity() -> None:
    records = [
        ActRecord("a", "c1", "t1", "protect", "care", +1),
        ActRecord("a", "c2", "t2", "refuse",  "care", -1),
        ActRecord("a", "c3", "t3", "help",    "care", +1),
        ActRecord("a", "c4", "t4", "refuse",  "care", -1),
    ]
    a = assess_virtue_history("a", records)
    # Mean ~ 0; dominant should be "ambiguous"
    assert a.per_axis_dominant["care"] == "ambiguous"


# ----------------------------------------------------- record_from_compile


def _compile(name: str):
    return compile_document(
        REPO_ROOT / "examples" / f"{name}.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
        ),
    )


def test_record_from_compile_extracts_action_kind() -> None:
    ir = _compile("nazi_attic")
    rec = record_from_compile("speaker", ir)
    assert rec is not None
    assert rec.action_kind == "deceive"
    assert rec.virtue_axis == "honesty"
    assert rec.polarity == -1


# ----------------------------------------------------- LongitudinalVirtueProjection


def test_projection_writes_on_first_run(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    sub = substrate_from_ir(ir)
    proj = LongitudinalVirtueProjection(store)
    res = proj.project(sub, graph=ir.graph, ir=ir)
    assert res.framework == "virtue_longitudinal"
    # Store should now contain 1 record.
    assert store.read_history("self")  # 'self' is the agent id


def test_projection_accumulates_across_runs(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    proj = LongitudinalVirtueProjection(store)
    for _ in range(3):
        sub = substrate_from_ir(ir)
        proj.project(sub, graph=ir.graph, ir=ir)
    assert len(store.read_history("self")) == 3


def test_projection_read_only_mode_does_not_write(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    sub = substrate_from_ir(ir)
    proj = LongitudinalVirtueProjection(store, read_only=True)
    res = proj.project(sub, graph=ir.graph, ir=ir)
    assert store.read_history("self") == []
    # And the result should still flag no prior observations.
    by_name = {f["name"]: f for f in [f.model_dump() for f in res.findings]}
    assert by_name["longitudinal_pattern"]["passed"] is True


def test_projection_assesses_pattern_after_enough_observations(tmp_path: Path) -> None:
    """After 3+ deceive observations, the longitudinal pattern gate
    should fire 'vice' on the honesty axis."""
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    proj = LongitudinalVirtueProjection(store)
    for _ in range(5):
        sub = substrate_from_ir(ir)
        res = proj.project(sub, graph=ir.graph, ir=ir)
    by_name = {f.name: f for f in res.findings}
    pat = by_name["longitudinal_pattern"]
    assert pat.passed is False
    assert "honesty" in pat.reason.lower()


def test_evidence_sufficient_gate_flags_low_n(tmp_path: Path) -> None:
    """One swallow doesn't make a summer."""
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    sub = substrate_from_ir(ir)
    proj = LongitudinalVirtueProjection(store)
    res = proj.project(sub, graph=ir.graph, ir=ir)
    by_name = {f.name: f for f in res.findings}
    sufficient = by_name["evidence_sufficient"]
    assert sufficient.passed is False
    assert "swallow" in sufficient.reason.lower() or "1" in sufficient.reason


def test_history_hash_recorded_in_result(tmp_path: Path) -> None:
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    sub = substrate_from_ir(ir)
    proj = LongitudinalVirtueProjection(store)
    res = proj.project(sub, graph=ir.graph, ir=ir)
    la = res.framework_specific["longitudinal_assessment"]
    assert "history_hash" in la
    assert len(la["history_hash"]) == 64


def test_projection_combined_verdict_uses_longitudinal_when_n_sufficient(tmp_path: Path) -> None:
    """With ≥3 vice observations on the same axis, combined verdict
    should be 'vicious' (forbid polarity)."""
    store = HabitStore(tmp_path)
    ir = _compile("nazi_attic")
    proj = LongitudinalVirtueProjection(store)
    for _ in range(5):
        sub = substrate_from_ir(ir)
        res = proj.project(sub, graph=ir.graph, ir=ir)
    assert res.verdict == "vicious"
    assert res.polarity == "forbid"


def test_per_agent_history_separation(tmp_path: Path) -> None:
    """Two distinct agents should accumulate independently."""
    store = HabitStore(tmp_path)
    store.append(ActRecord("alice", "c1", "t", "deceive", "honesty", -1))
    store.append(ActRecord("alice", "c2", "t", "deceive", "honesty", -1))
    store.append(ActRecord("bob", "c1", "t", "protect", "care", +1))
    a = store.assess("alice")
    b = store.assess("bob")
    assert a.n_observations == 2
    assert b.n_observations == 1
    assert a.per_axis_dominant["honesty"] == "vice"
    assert b.per_axis_dominant["care"] in ("virtue", "ambiguous")
