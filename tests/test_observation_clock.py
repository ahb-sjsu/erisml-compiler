"""Vector clocks on a scene's history: causal order across producers, causal extension on rebuild,
observation patches concurrent with what was appended meanwhile, and ledgers merged from other
parties (the open problem of the gtc-prototype monograph, chapter 14)."""

from __future__ import annotations

import json

import pytest

from erisml_compiler.annotation.llm_extractor import MockLLMAdapter
from erisml_compiler.audit.hash_chain import compute_ir_hash
from erisml_compiler.ir.schemas import CompilerIR, Document, Event, Stakeholder
from erisml_compiler.observation import (
    Observation,
    ObservationCompiler,
    PatchNotAllowed,
    SceneHistoryError,
    append_event,
    apply_observation_patch,
    merge_ledger,
    rebuild_runtime,
)
from erisml_compiler.observation import clock as vc
from erisml_compiler.runtime import SceneRuntime

ACTS = {"check_in": "ask", "call_emergency_services": "call"}
HAZARD = {"events": [{"type": "hazard", "target": "margaret", "content": "emerging"}]}


def scene():
    return CompilerIR(
        document=Document(doc_id="home"),
        stakeholders=[Stakeholder(id="margaret", label="M", type="individual")],
        events=[Event(id="e0", time_index=0, type="role_assigned")],
    )


def patch_on(ir):
    return ObservationCompiler(
        MockLLMAdapter({"You update the structured model": json.dumps(HAZARD)})
    ).compile(ir, [Observation("speech", "margaret", "ow")], ACTS)


def test_the_clock_relations():
    a, b = {"robot": 1}, {"robot": 2}
    c = {"observer": 1}
    assert vc.before(a, b) and not vc.before(b, a) and vc.concurrent(a, c)
    assert vc.before({}, a) and vc.merge(a, c) == {"robot": 1, "observer": 1}
    assert vc.consistent([{}, a, c, b]) and not vc.consistent([b, a])


def test_appended_events_tick_their_producer_on_everything_seen():
    ir = append_event(append_event(scene(), {"type": "x"}, "robot"), {"type": "y"}, "centre")
    assert [e.clock for e in ir.events[1:]] == [{"robot": 1}, {"robot": 1, "centre": 1}]
    assert vc.before(ir.events[1].clock, ir.events[2].clock)


def test_an_observation_is_concurrent_with_what_was_appended_while_it_compiled():
    base = scene()
    patch = patch_on(base)
    ir = append_event(
        append_event(base, {"type": "governor_ruling"}, "robot"),
        {"type": "attack_measured"},
        "robot",
    )
    new = apply_observation_patch(ir, patch)
    obs = next(e for e in new.events if e.type == "hazard")
    rulings = [e for e in new.events if e.type in ("governor_ruling", "attack_measured")]
    assert all(vc.concurrent(obs.clock, r.clock) for r in rulings)
    assert obs.clock == {"observer": 1} and obs.time_index > max(r.time_index for r in rulings)


def test_the_observer_s_own_events_stay_in_order_across_patches():
    ir = apply_observation_patch(scene(), patch_on(scene()))
    ir = apply_observation_patch(ir, patch_on(ir))
    clocks = [e.clock for e in ir.events if e.type == "hazard"]
    assert vc.before(clocks[0], clocks[1])


def test_a_patch_from_a_history_this_ir_has_not_seen_is_refused():
    elsewhere = append_event(scene(), {"type": "x"}, "robot")
    patch = patch_on(elsewhere)  # compiled having seen robot:1
    with pytest.raises(PatchNotAllowed, match="has not seen"):
        apply_observation_patch(scene(), patch)


def test_rebuild_accepts_a_causal_extension_and_refuses_a_reordering():
    ir = append_event(scene(), {"type": "a"}, "robot")
    rt = SceneRuntime(ir)
    grown = apply_observation_patch(append_event(ir, {"type": "b"}, "robot"), patch_on(ir))
    assert [e.type for e in rebuild_runtime(rt, grown).events][-2:] == ["b", "hazard"]
    swapped = grown.model_copy(deep=True)
    a, b = (e for e in swapped.events if e.type in ("a", "b"))
    a.__dict__["time_index"], b.__dict__["time_index"] = (
        b.time_index,
        a.time_index,
    )  # b now before a
    rt_b = SceneRuntime(append_event(ir, {"type": "b"}, "robot"))
    with pytest.raises(SceneHistoryError):
        rebuild_runtime(rt_b, swapped)


def test_rebuild_refuses_a_dropped_event_or_one_stepped_outside_the_ir():
    ir = append_event(scene(), {"type": "a"}, "robot")
    rt = SceneRuntime(ir)
    with pytest.raises(SceneHistoryError):
        rebuild_runtime(rt, scene())  # 'a' dropped
    rt.step({"type": "live", "content": "never recorded"})
    with pytest.raises(SceneHistoryError):
        rebuild_runtime(rt, ir)


def test_a_ledger_from_another_party_merges_when_causally_consistent():
    ir = append_event(scene(), {"type": "referral"}, "robot")
    seen = ir.events[-1].clock
    centre = [
        {"id": "c1", "type": "client_spoke", "clock": vc.tick(seen, "centre")},
        {
            "id": "c2",
            "type": "send_emergency_services",
            "clock": vc.tick(vc.tick(seen, "centre"), "centre"),
        },
    ]
    merged = merge_ledger(ir, centre)
    assert [e.id for e in merged.events][-2:] == ["c1", "c2"]
    assert merge_ledger(merged, centre) is merged  # nothing new


def test_a_ledger_event_that_happened_before_the_history_is_refused():
    ir = append_event(append_event(scene(), {"type": "a"}, "robot"), {"type": "b"}, "robot")
    stale = [{"id": "x", "type": "late", "clock": {}}]  # precedes every clocked event
    with pytest.raises(SceneHistoryError):
        merge_ledger(ir, stale)


def test_an_ir_without_clocks_keeps_its_hash():
    ir = scene()
    assert "clock" not in json.dumps(ir.model_dump(mode="json"))
    assert compute_ir_hash(ir) == compute_ir_hash(
        CompilerIR.model_validate(ir.model_dump(mode="json"))
    )
