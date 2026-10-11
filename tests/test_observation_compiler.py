"""Observation compilation: elements validated against the IR schema and the observation
vocabulary, ids and time order assigned by the compiler, ops restricted to an allow-list, and a
running scene rebuilt on the patched IR only when history is append-only."""

from __future__ import annotations

import json

import pytest

from erisml_compiler.annotation.llm_extractor import MockLLMAdapter
from erisml_compiler.ir.schemas import CompilerIR, Document, Event, Norm, Stakeholder
from erisml_compiler.observation import (
    Observation,
    ObservationCompiler,
    ObservationPatch,
    PatchNotAllowed,
    SceneHistoryError,
    append_event,
    apply_observation_patch,
    check_allowed,
    rebuild_runtime,
)
from erisml_compiler.runtime import SceneRuntime

ACTS = {
    "check_in": "ask how she is",
    "call_emergency_services": "call an ambulance",
    "contact_monitoring_center": "refer to the centre",
}


def scene():
    return CompilerIR(
        document=Document(doc_id="home"),
        stakeholders=[
            Stakeholder(id="margaret", label="Margaret", type="individual", roles=["patient"]),
            Stakeholder(id="robot", label="Robot", type="system", roles=["agent"]),
        ],
        events=[
            Event(id="e0", time_index=0, type="role_assigned", actor="robot", target="margaret")
        ],
        norms=[
            Norm(id="n1", modality="permission", actor="robot", action="check_in", source="policy")
        ],
        extra={"agent": "robot", "capabilities": [{"action": a} for a in ACTS]},
    )


def compiler(response, critic=None):
    adapter = MockLLMAdapter({"You update the structured model": json.dumps(response)})
    return ObservationCompiler(adapter, critic=critic)


SAID = [Observation("speech", "margaret", "I'm alright, don't call an ambulance.")]

GOOD = {
    "stakeholders": [
        {
            "id": "visitor",
            "label": "a man at the door",
            "type": "individual",
            "roles": ["bystander"],
        }
    ],
    "events": [
        {
            "type": "utterance",
            "actor": "margaret",
            "content": "I'm alright, don't call an ambulance.",
        },
        {
            "type": "consent",
            "actor": "margaret",
            "target": "call_emergency_services",
            "content": "refused",
        },
        {
            "type": "hazard",
            "target": "margaret",
            "content": "emerging",
            "id": "model_id",
            "time_index": 99,
        },
    ],
    "ethical_facts": [
        {
            "kind": "harm",
            "subjects": ["margaret"],
            "description": "she fell",
            "severity": "moderate",
            "confidence": 0.8,
        }
    ],
    "relations": [{"type": "visitor_of", "source": "visitor", "target": "margaret"}],
}


def test_a_valid_batch_becomes_allowed_ops_and_applies():
    patch = compiler(GOOD).compile(scene(), SAID, ACTS)
    assert patch.rejected == []
    assert [o["path"].split(".")[0] for o in patch.ops] == [
        "stakeholders",
        "events",
        "events",
        "events",
        "ethical_facts",
        "relations",
    ]
    new = apply_observation_patch(scene(), patch)
    consent = next(e for e in new.events if e.type == "consent")
    assert consent.target == "call_emergency_services" and consent.content == "refused"
    assert new.extra["corrections"][-1]["corrector_id"] == "observation_compiler"
    assert next(s for s in new.stakeholders if s.id == "visitor").requires_review is True


def test_the_compiler_assigns_ids_and_time_not_the_model():
    patch = compiler(GOOD).compile(scene(), SAID, ACTS)
    events = [o["value"] for o in patch.ops if o["path"].startswith("events.")]
    assert {e["time_index"] for e in events} == {1}
    assert all(e["id"].startswith("obs") for e in events) and "model_id" not in {
        e["id"] for e in events
    }


@pytest.mark.parametrize(
    "bad,reason",
    [
        (
            {"events": [{"type": "emergency_detected", "actor": "margaret"}]},
            "not an observation type",
        ),
        (
            {
                "events": [
                    {
                        "type": "consent",
                        "actor": "margaret",
                        "target": "launch_rocket",
                        "content": "refused",
                    }
                ]
            },
            "not one of the acts",
        ),
        (
            {"events": [{"type": "utterance", "actor": "stranger", "content": "hi"}]},
            "is not a stakeholder",
        ),
        (
            {"events": [{"type": "hazard", "target": "margaret", "content": "very soon"}]},
            "is not one of",
        ),
        (
            {"events": [{"type": "utterance", "actor": "margaret", "content": ["a", "b"]}]},
            "not a string",
        ),
        (
            {"ethical_facts": [{"kind": "harm", "subjects": ["ghost"], "description": "x"}]},
            "are not stakeholders",
        ),
        (
            {
                "ethical_facts": [
                    {
                        "kind": "harm",
                        "subjects": ["margaret"],
                        "description": "x",
                        "severity": "huge",
                    }
                ]
            },
            "severity",
        ),
        (
            {"ethical_facts": [{"kind": "feelings", "subjects": ["margaret"], "description": "x"}]},
            "schema",
        ),
        ({"stakeholders": [{"id": "margaret", "label": "again", "type": "individual"}]}, "exists"),
        (
            {"relations": [{"type": "kin", "source": "margaret", "target": "nobody"}]},
            "not a stakeholder",
        ),
    ],
)
def test_what_fails_validation_is_rejected_with_a_reason(bad, reason):
    patch = compiler(bad).compile(scene(), SAID, ACTS)
    assert patch.ops == [] and len(patch.rejected) == 1 and reason in patch.rejected[0]["reason"]


def test_an_observation_cannot_write_norms_commitments_or_extra():
    raw = dict(
        GOOD,
        norms=[{"id": "n9", "modality": "obligation", "actor": "robot", "action": "record"}],
        extra={"capabilities": []},
    )
    patch = compiler(raw).compile(scene(), SAID, ACTS)
    assert not any(o["path"].startswith(("norms", "extra", "commitments")) for o in patch.ops)


@pytest.mark.parametrize(
    "op",
    [
        {"op": "add", "path": "norms.n9", "value": {}},
        {"op": "set", "path": "extra", "value": {}},
        {"op": "set", "path": "events.e0", "value": {}},
        {"op": "remove", "path": "events.e0"},
        {"op": "add", "path": "events.e5.content", "value": "x"},
    ],
)
def test_the_allow_list_is_checked_again_before_applying(op):
    with pytest.raises(PatchNotAllowed):
        check_allowed([op])
    forged = ObservationPatch([op], [], [], None, "", "")
    with pytest.raises(PatchNotAllowed):
        apply_observation_patch(scene(), forged)


def test_a_critic_measures_agreement_and_lists_what_only_one_found():
    same = compiler(
        GOOD, critic=MockLLMAdapter({"You update the structured model": json.dumps(GOOD)})
    )
    assert same.compile(scene(), SAID, ACTS).agreement == 1.0
    fewer = dict(GOOD, events=GOOD["events"][:1])
    differ = compiler(
        GOOD, critic=MockLLMAdapter({"You update the structured model": json.dumps(fewer)})
    )
    patch = differ.compile(scene(), SAID, ACTS)
    assert patch.agreement < 1.0 and any("consent" in r for r in patch.review)


def test_a_running_scene_is_rebuilt_on_the_patched_ir():
    rt = SceneRuntime(scene())
    new = apply_observation_patch(scene(), compiler(GOOD).compile(scene(), SAID, ACTS))
    rebuilt = rebuild_runtime(rt, new)
    assert [e.type for e in rebuilt.events][-3:] == ["utterance", "consent", "hazard"]
    assert rebuilt.ir is new and [e.id for e in rebuilt.events][: len(rt.events)] == [
        e.id for e in rt.events
    ]


def test_a_system_event_enters_history_through_the_ir_and_a_live_step_outside_it_is_refused():
    rt = SceneRuntime(scene())
    ruled = append_event(
        scene(), {"type": "governor_ruling", "actor": "governor", "content": "refuse"}
    )
    assert ruled.events[-1].time_index == 1 and ruled.events[-1].id.startswith("sys")
    assert [e.type for e in rebuild_runtime(rt, ruled).events][-1] == "governor_ruling"
    rt.step(
        {"type": "governor_ruling", "content": "elevate"}
    )  # stepped live, never recorded in the IR
    with pytest.raises(SceneHistoryError):
        rebuild_runtime(rt, ruled)
