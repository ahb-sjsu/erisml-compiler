"""Each candidate act's consequences projected into its own IR, for DEME to judge per option.

Before, the compiler described only the situation; every candidate act was judged on the same
facts, so harm and benefit could not tell the acts apart.
"""

from __future__ import annotations

import json

import pytest

from erisml_compiler.annotation.llm_extractor import MockLLMAdapter
from erisml_compiler.ir.schemas import CompilerIR, Document, EthicalFact, Event, Stakeholder
from erisml_compiler.observation import OptionProjector, consents_on_record, option_ir

ACTS = {
    "wait": "Carry on with chores.",
    "help_up": "Help Margaret up off the floor.",
    "call_centre": "Call the monitoring centre for help.",
}


def scene():
    return CompilerIR(
        document=Document(doc_id="home", raw_text="Margaret lives alone."),
        stakeholders=[Stakeholder(id="margaret", label="M", type="individual")],
        events=[
            Event(
                id="e0",
                time_index=0,
                type="consent",
                actor="margaret",
                target="call_centre",
                content="consented",
            ),
            Event(
                id="e1",
                time_index=1,
                type="consent",
                actor="margaret",
                target="help_up",
                content="refused",
            ),
        ],
        ethical_facts=[
            EthicalFact(
                id="f0",
                kind="harm",
                subjects=["margaret"],
                description="She has fallen.",
                severity="grave",
            ),
            EthicalFact(
                id="f1",
                kind="vulnerability",
                subjects=["margaret"],
                description="She is frail.",
                severity="moderate",
            ),
        ],
    )


def fact(kind, severity, **kw):
    return {"kind": kind, "subjects": ["margaret"], "description": "x", "severity": severity, **kw}


ANSWER = {
    "urgency": {"description": "She may be hurt.", "severity": "grave", "subjects": ["margaret"]},
    "options": {
        "wait": [fact("harm", "grave")],
        "help_up": [
            fact("benefit", "grave"),
            fact("harm", "minor"),
            fact("consent", "grave"),
        ],  # the model's consent claim; the record decides
        "call_centre": [fact("benefit", "grave"), fact("harm", "minor")],
        "fly_away": [fact("benefit", "grave")],
    },
}


def projector(answer=ANSWER):
    return OptionProjector(MockLLMAdapter({"You project the consequences": json.dumps(answer)}))


def test_the_projection_is_validated_and_ids_are_the_compiler_s():
    bad = dict(
        ANSWER,
        options=dict(
            ANSWER["options"],
            wait=[
                fact("harm", "apocalyptic"),
                fact("urgency", "grave"),
                {**fact("harm", "grave"), "subjects": ["ghost"]},
            ],
        ),
    )
    p = projector(bad).project(scene(), ACTS)
    reasons = [r["reason"] for r in p.rejected]
    assert any("not one of the acts" in r for r in reasons)  # fly_away
    assert any("severity" in r for r in reasons)
    assert any("urgency belongs to the situation" in r for r in reasons)
    assert any("not stakeholders" in r for r in reasons)
    assert p.unprojected == ["wait"]
    assert all(f.id.startswith("proj_") for fs in p.facts.values() for f in fs)


def test_an_option_ir_replaces_the_present_outcome_with_the_act_s_and_keeps_what_stands():
    p = projector().project(scene(), ACTS)
    ir = option_ir(scene(), p, "call_centre", ACTS["call_centre"])
    kinds = sorted(f.kind for f in ir.ethical_facts)
    assert kinds == ["benefit", "consent", "harm", "urgency", "vulnerability"]
    harm = next(f for f in ir.ethical_facts if f.kind == "harm")
    assert harm.severity == "minor"  # the act's projected harm, not the scene's grave one
    assert ir.document.raw_text == ACTS["call_centre"]


def test_consent_comes_from_the_record_and_overrides_the_model():
    p = projector().project(scene(), ACTS)
    help_up = option_ir(scene(), p, "help_up", ACTS["help_up"])
    consent = [f for f in help_up.ethical_facts if f.kind == "consent"]
    assert [(f.id, f.severity) for f in consent] == [("rec_consent_help_up_margaret", "grave")]
    assert consents_on_record(scene(), ACTS) == {
        "call_centre": {"margaret": "consented"},
        "help_up": {"margaret": "refused"},
    }


def test_per_option_facts_through_the_bridge_and_the_guide_s_hard_shell():
    pytest.importorskip("erisml.ethics.layers.composition")
    from erisml.ethics.governance.config_v2 import GovernanceConfigV2
    from erisml.ethics.layers.pipeline import DEMEPipeline, PipelineConfig
    from erisml.ethics.modules.tier0.geneva_em import GenevaEMV2
    from erisml.ethics.modules.tier1.core_physical_safety_em import CorePhysicalSafetyEMV2
    from erisml.ethics.modules.tier2.autonomy_consent_em import AutonomyConsentEMV2
    from erisml.ethics.modules.tier3.beneficence_em import BeneficenceEMV2

    from erisml_compiler.erisml_backend.v3_bridge import ir_to_v2_facts

    ir = scene()
    p = projector().project(ir, ACTS)
    options = [ir_to_v2_facts(option_ir(ir, p, a, d), option_id=a) for a, d in ACTS.items()]
    pipeline = DEMEPipeline(
        ems=[GenevaEMV2(), CorePhysicalSafetyEMV2(), AutonomyConsentEMV2(), BeneficenceEMV2()],
        config=PipelineConfig(
            composition="hard_shell",
            governance=GovernanceConfigV2(veto_ems=["geneva_constitutional"]),
        ),
    )
    result = pipeline.decide(options, baseline_option_id="wait")
    assert result.selected_option_id == "call_centre"
    stages = {s["stage"]: s["vetoed"] for s in result.stages}
    assert stages == {"shell": ["wait"], "inner": ["help_up"]}  # unsafe; refused
