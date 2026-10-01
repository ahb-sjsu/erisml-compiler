"""Scene runtime: moral state machines stepped on a live event stream."""

import json

import pytest

from erisml_compiler.ingestion.structured_loader import load_structured_input
from erisml_compiler.ir.graph.promote import graph_from_flat
from erisml_compiler.ir.graph.schema import NodeKind
from erisml_compiler.ir.schemas import Commitment, CompilerIR, Document, Norm, Stakeholder
from erisml_compiler.runtime import SceneRuntime


def home(**extra_over):
    extra = {
        "agent": "robot",
        "capabilities": [
            {"action": "chores", "elevated": False},
            {"action": "check_in", "elevated": False},
            {"action": "call_emergency_services", "elevated": True},
            {"action": "record", "elevated": True},
        ],
        "conditions": {"emergency": ["latest:governor_ruling=elevate"]},
        "oversight": ["oversight_restored"],
    }
    extra.update(extra_over)
    return CompilerIR(
        document=Document(doc_id="home"),
        stakeholders=[
            Stakeholder(
                id="margaret",
                label="M",
                type="individual",
                roles=["patient"],
                consent_status="not_obtained",
            ),
            Stakeholder(id="robot", label="R", type="system", roles=["agent", "protector"]),
        ],
        commitments=[
            Commitment(
                id="privacy",
                type="promise",
                holder="robot",
                beneficiary="margaret",
                content="no recording",
                defeasibility_conditions=["emergency"],
            ),
        ],
        norms=[
            Norm(
                id="n_privacy",
                modality="prohibition",
                actor="robot",
                action="record",
                source="commitment:privacy",
            ),
            Norm(
                id="n_auth",
                modality="prohibition",
                actor="robot",
                action="elevated_action",
                defeasible=False,
                source="delegation",
                priority_tier=0,
                conditions=["not:latest:governor_ruling=elevate"],
            ),
            Norm(
                id="n_check",
                modality="obligation",
                actor="robot",
                action="check_in",
                source="duty",
                conditions=["event:inactivity_exceeded"],
            ),
            Norm(
                id="n_chores",
                modality="permission",
                actor="robot",
                action="chores",
                source="duty",
                priority_tier=3,
            ),
        ],
        extra=extra,
    )


def test_elevated_actions_follow_the_latest_ruling():
    rt = SceneRuntime(home())
    s = rt.snapshot()
    assert "call_emergency_services" in s.prohibited and "record" in s.prohibited
    s = rt.step({"type": "governor_ruling", "content": "elevate"})
    assert "call_emergency_services" in s.allowed
    s = rt.step({"type": "governor_ruling", "content": "refuse"})
    assert "call_emergency_services" in s.prohibited  # one elevate is not standing authority


def test_privacy_is_overridden_in_an_emergency_and_restored_only_by_oversight():
    rt = SceneRuntime(home())
    assert rt.snapshot().machines["commitment:privacy"] == "active"
    s = rt.step({"type": "governor_ruling", "content": "elevate"})
    assert s.machines["commitment:privacy"] == "active_but_defeasible"
    assert "record" in s.allowed
    s = rt.step({"type": "governor_ruling", "content": "refuse"})
    # the emergency ruling has passed, but nothing the robot observes restores privacy
    assert s.machines["commitment:privacy"] == "active_but_defeasible"
    s = rt.step({"type": "oversight_restored", "content": "privacy"})
    assert s.machines["commitment:privacy"] == "active"
    assert "record" in s.prohibited
    # the old elevate ruling does not override it again; a new one does
    s = rt.step({"type": "time_passed"})
    assert s.machines["commitment:privacy"] == "active"
    s = rt.step({"type": "governor_ruling", "content": "elevate"})
    assert s.machines["commitment:privacy"] == "active_but_defeasible"


def test_obligation_triggers_and_is_discharged_by_the_action():
    rt = SceneRuntime(home())
    assert "check_in" not in rt.snapshot().obliged
    s = rt.step({"type": "inactivity_exceeded", "content": "seated"})
    assert s.obliged == ["check_in"]
    s = rt.step({"type": "action_performed", "content": "check_in"})
    assert s.obliged == []
    s = rt.step({"type": "inactivity_exceeded", "content": "seated"})
    assert s.obliged == ["check_in"]


def test_permission_does_not_lift_a_higher_priority_prohibition():
    ir = home()
    ir.norms = ir.norms + [
        Norm(
            id="n_no_chores",
            modality="prohibition",
            actor="robot",
            action="chores",
            source="x",
            priority_tier=1,
        )
    ]
    rt = SceneRuntime(ir)
    assert "chores" in rt.snapshot().prohibited


def test_consent_machine_steps_on_consent_events():
    rt = SceneRuntime(home())
    assert rt.snapshot().machines["consent:margaret"] == "not_obtained"
    s = rt.step({"type": "consent_given", "actor": "margaret"})
    assert s.machines["consent:margaret"] == "obtained"


def test_undefined_condition_is_an_error_not_a_silent_false():
    ir = home()
    ir.norms = ir.norms + [
        Norm(
            id="bad",
            modality="obligation",
            actor="robot",
            action="check_in",
            source="x",
            conditions=["cond:nope"],
        )
    ]
    with pytest.raises(KeyError):
        SceneRuntime(ir).snapshot()


def test_norm_without_conditions_keeps_its_graph_payload():
    n = Norm(id="n1", modality="prohibition", actor="a", action="b", source="s")
    plain = CompilerIR(document=Document(doc_id="d"), norms=[n])
    payload = graph_from_flat(plain).nodes_of_kind(NodeKind.NORM)[0].payload
    # identical to the payload before the field existed, so canonical hashes do not move
    assert payload == {k: v for k, v in n.model_dump().items() if k != "conditions"}
    with_cond = CompilerIR(
        document=Document(doc_id="d"), norms=[n.model_copy(update={"conditions": ["event:x"]})]
    )
    assert graph_from_flat(with_cond).nodes_of_kind(NodeKind.NORM)[0].payload["conditions"] == [
        "event:x"
    ]


def test_erisml_source_loads_as_tier1(tmp_path):
    src = tmp_path / "scene.erisml"
    src.write_text(
        "document: {doc_id: s, title: S, raw_text: a scene}\n"
        "stakeholders: [{id: m, label: M, type: individual, roles: [patient], consent_status: not_obtained}]\n"
        "norms: [{id: n1, modality: obligation, actor: r, action: check_in, source: duty, conditions: ['event:x']}]\n"
        "relations: [{id: r1, type: cares_for, source: r, target: m}]\n"
        "extra: {agent: r}\n",
        encoding="utf-8",
    )
    ir = load_structured_input(src)
    assert (
        ir.norms[0].conditions == ["event:x"]
        and ir.relations[0].type == "cares_for"
        and ir.extra["agent"] == "r"
    )
    assert json.loads(ir.model_dump_json())["document"]["doc_id"] == "s"
