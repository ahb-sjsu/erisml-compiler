"""Benefit and urgency reach DEME, and facts can be built per candidate option.

Before, the IR had no fact kind for what an option secures or how soon it must, so the bridge
left V2 Consequences.expected_benefit and .urgency at zero and V3 urgency was a stand-in (the
gravest harm). Every option then looked equally beneficial and equally urgent to DEME, which
biases selection toward doing nothing. The option id was always the document id, so per-option
facts for DEMEPipeline.decide could not be told apart.
"""

from __future__ import annotations

import pytest


from erisml_compiler.erisml_backend.v3_bridge import ir_to_v2_facts
from erisml_compiler.erisml_backend.v3_facts_direct import ir_to_v3_facts
from erisml_compiler.ir.schemas import CompilerIR, Document, EthicalFact, Stakeholder


def scene(*facts):
    return CompilerIR(
        document=Document(doc_id="home"),
        stakeholders=[
            Stakeholder(id="margaret", label="M", type="individual"),
            Stakeholder(id="neighbour", label="N", type="individual"),
        ],
        ethical_facts=list(facts),
    )


AVERTS_A_GRAVE_FALL = EthicalFact(
    id="b1", kind="benefit", subjects=["margaret"], description="x", severity="grave"
)
IMMINENT = EthicalFact(id="u1", kind="urgency", description="x", severity="catastrophic")
MINOR_HARM = EthicalFact(
    id="h1", kind="harm", subjects=["margaret"], description="x", severity="minor"
)


def test_v2_facts_carry_benefit_and_urgency():
    facts = ir_to_v2_facts(scene(AVERTS_A_GRAVE_FALL, IMMINENT))
    assert facts.consequences.expected_benefit == 0.75
    assert facts.consequences.urgency == 1.0


def test_v2_facts_without_them_stay_at_zero():
    facts = ir_to_v2_facts(scene(MINOR_HARM))
    assert facts.consequences.expected_benefit == 0.0 and facts.consequences.urgency == 0.0


def test_v3_benefit_goes_to_the_party_it_serves():
    facts = ir_to_v3_facts(scene(AVERTS_A_GRAVE_FALL))
    per = {p.party_id: p.expected_benefit for p in facts.consequences.per_party}
    assert per == {"margaret": 0.75, "neighbour": 0.0}


def test_v3_urgency_is_stated_when_stated_and_the_old_stand_in_otherwise():
    assert ir_to_v3_facts(scene(MINOR_HARM, IMMINENT)).consequences.urgency == 1.0
    assert ir_to_v3_facts(scene(MINOR_HARM)).consequences.urgency == 0.25


def test_facts_are_built_per_option():
    ir = scene(AVERTS_A_GRAVE_FALL)
    assert ir_to_v2_facts(ir, option_id="call_for_help").option_id == "call_for_help"
    assert ir_to_v3_facts(ir, option_id="wait").option_id == "wait"
    assert ir_to_v2_facts(ir).option_id == "home"
