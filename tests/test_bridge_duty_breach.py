"""A breached duty reaches DEME.

Before, a `role_duty` fact set only `role_duty_conflict`, which no DEME dimension reads, and the
bridge never read commitments. Harming one's own dependant then scored like harming a stranger,
and a promise marked violated scored like nothing: locking a grandfather's wife out of his
hospital room compiled to virtue_care 1.0 and rights_respect 1.0. Now a duty owed to a party (a
role duty naming them, or a commitment the agent holds for them) and breached (a harm, coercion
or deception falling on them, or the commitment marked violated) betrays trust and withholds
care, and at grave severity violates the party's rights.
"""

from __future__ import annotations

from erisml.ethics.moral_vector import MoralVector

from erisml_compiler.erisml_backend.v3_bridge import breached_duties, ir_to_v2_facts
from erisml_compiler.erisml_backend.v3_facts_direct import ir_to_v3_facts
from erisml_compiler.ir.schemas import Commitment, CompilerIR, Document, EthicalFact, Stakeholder


def scene(*facts, commitments=()):
    return CompilerIR(
        document=Document(doc_id="act"),
        stakeholders=[
            Stakeholder(id="narrator", label="I", type="individual", roles=["agent"]),
            Stakeholder(id="grandmother", label="G", type="individual", roles=["patient"]),
            Stakeholder(id="stranger", label="S", type="individual", roles=["patient"]),
        ],
        ethical_facts=list(facts),
        commitments=list(commitments),
    )


def fact(i, kind, subjects, severity="moderate"):
    return EthicalFact(id=i, kind=kind, subjects=subjects, description="x", severity=severity)


DUTY = fact("d", "role_duty", ["narrator", "grandmother"])


def vector(ir):
    return MoralVector.from_ethical_facts(ir_to_v2_facts(ir))


def test_harm_under_a_duty_lowers_care_and_harm_to_a_stranger_does_not():
    to_kin = vector(scene(DUTY, fact("h", "harm", ["grandmother"])))
    to_stranger = vector(scene(DUTY, fact("h", "harm", ["stranger"])))
    assert to_kin.virtue_care < to_stranger.virtue_care == 1.0
    assert to_kin.physical_harm == to_stranger.physical_harm


def test_a_duty_owed_and_kept_changes_nothing():
    assert breached_duties(scene(DUTY)) == {}
    assert vector(scene(DUTY)).virtue_care == 1.0


def test_only_a_grave_breach_violates_rights():
    moderate = ir_to_v2_facts(scene(DUTY, fact("h", "harm", ["grandmother"], "moderate")))
    grave = ir_to_v2_facts(scene(DUTY, fact("h", "coercion", ["grandmother"], "grave")))
    assert not moderate.rights_and_duties.violates_rights
    assert grave.rights_and_duties.violates_rights


def test_the_gravest_wrong_is_recorded():
    ir = scene(
        DUTY,
        fact("a", "harm", ["grandmother"], "minor"),
        fact("b", "deception", ["grandmother"], "grave"),
    )
    assert breached_duties(ir) == {"grandmother": "grave"}
    assert ir_to_v2_facts(ir).metadata["duties_breached"] == {"grandmother": "grave"}


def test_a_violated_commitment_held_by_the_agent_is_a_breach():
    promise = Commitment(
        id="p",
        type="promise",
        holder="narrator",
        beneficiary="stranger",
        content="repay",
        status="violated",
    )
    theirs = promise.model_copy(update={"holder": "grandmother"})
    kept = promise.model_copy(update={"status": "active"})
    assert breached_duties(scene(commitments=[promise])) == {"stranger": "moderate"}
    assert breached_duties(scene(commitments=[theirs])) == {}
    assert breached_duties(scene(commitments=[kept])) == {}


def test_the_agent_is_not_owed_its_own_duty():
    assert breached_duties(scene(DUTY, fact("h", "harm", ["narrator"]))) == {}


def test_per_party_records_mark_only_the_party_wronged():
    facts = ir_to_v3_facts(
        scene(
            DUTY,
            fact("h", "harm", ["grandmother"], "grave"),
            fact("k", "harm", ["stranger"], "grave"),
        )
    )
    virtue = {p.party_id: p for p in facts.virtue_and_care.per_party}
    rights = {p.party_id: p for p in facts.rights_and_duties.per_party}
    assert (
        not virtue["grandmother"].trust_preserved and not virtue["grandmother"].receives_compassion
    )
    assert virtue["stranger"].trust_preserved and virtue["stranger"].receives_compassion
    assert rights["grandmother"].rights_violated and rights["grandmother"].duty_owed
