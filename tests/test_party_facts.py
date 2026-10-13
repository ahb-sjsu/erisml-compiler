"""The option as each party bears it, for DEME at rank 2 (erisml-lib EthicalFactsV3.for_party)."""

from __future__ import annotations

import pytest

pytest.importorskip("erisml.ethics.facts_v3")

from erisml_compiler.erisml_backend.v3_bridge import ir_to_party_facts, ir_to_v2_facts
from erisml_compiler.ir.schemas import CompilerIR, Document, EthicalFact, Stakeholder


def scene():
    return CompilerIR(
        document=Document(doc_id="d", raw_text="Help Margaret up."),
        stakeholders=[
            Stakeholder(id="margaret", label="M", type="individual"),
            Stakeholder(id="tom", label="T", type="individual"),
        ],
        ethical_facts=[
            EthicalFact(id="f1", kind="harm", subjects=["tom"], description="x", severity="grave"),
            EthicalFact(
                id="f2", kind="benefit", subjects=["margaret"], description="y", severity="moderate"
            ),
            EthicalFact(
                id="f3",
                kind="consent",
                subjects=["margaret"],
                description="refused",
                severity="grave",
            ),
            EthicalFact(
                id="f4", kind="uncertainty", subjects=[], description="z", severity="minor"
            ),
        ],
    )


def test_each_party_bears_the_facts_that_name_it():
    f = ir_to_party_facts(scene(), "help_up")
    margaret, tom = f.for_party("margaret"), f.for_party("tom")
    assert (margaret.consequences.expected_harm, margaret.consequences.expected_benefit) == (
        0.0,
        0.5,
    )
    assert (tom.consequences.expected_harm, tom.consequences.expected_benefit) == (0.75, 0.0)
    assert margaret.rights_and_duties.has_valid_consent is False
    assert tom.rights_and_duties.has_valid_consent is True


def test_the_aggregate_maxim_and_unattributed_facts_are_the_whole_scenes():
    ir = scene()
    f, whole = ir_to_party_facts(ir, "help_up"), ir_to_v2_facts(ir, "help_up")
    assert f.to_v2().consequences == whole.consequences and f.option_id == "help_up"
    assert f.maxim == whole.maxim and f.maxim.action_kind == "help"
    assert f.for_party("tom").epistemic_status == f.for_party("margaret").epistemic_status
    assert sorted(f.party_ids) == ["margaret", "tom"]
