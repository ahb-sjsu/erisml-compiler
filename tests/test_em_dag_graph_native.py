"""EM-DAG graph-native port: regression + behavioural tests.

The port redirects the EM helpers (`facts_of_kind`, `active_commitments`,
`stakeholders_with_role`, `vulnerable_stakeholders`,
`nonconsenting_third_party_ids`) to read the MoralGraph when one is
attached, falling back to flat fields when not.

These tests verify:
  - EM outputs are byte-identical against a golden baseline captured
    from the flat-field implementation on the 3 bundled scenarios
    (verdict + every per-module value/confidence)
  - The helpers genuinely read the graph when ir.graph is set (proven
    by clearing the flat lists; reads still return the right data)
  - Flat-only IRs (no graph) still work via the fallback path
"""

from __future__ import annotations

import json
from pathlib import Path

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.em_dag.modules._helpers import (
    active_commitments,
    facts_of_kind,
    nonconsenting_third_party_ids,
    stakeholders_with_role,
    vulnerable_stakeholders,
)
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = REPO_ROOT / "tests" / "golden_em_dag_flat.json"
EXAMPLES_DIR = REPO_ROOT / "examples"


def _compile(name: str):
    return compile_document(
        EXAMPLES_DIR / f"{name}.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="rule",
                       canonicalizer=RegistryCanonicalizer(), tensor_rank=2),
    )


# ----------------------------------------------------- golden baseline


def _load_golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def test_nazi_attic_em_outputs_match_golden() -> None:
    g = _load_golden()["nazi_attic"]
    ir = _compile("nazi_attic")
    assert ir.deme_verdict.verdict == g["deme_verdict"]
    for name, em in ir.em_outputs.items():
        gem = g["em_outputs"][name]
        assert em.score.value == gem["value"], f"{name} value drifted"
        assert em.score.confidence == gem["confidence"], f"{name} confidence drifted"


def test_medical_confidentiality_em_outputs_match_golden() -> None:
    g = _load_golden()["medical_confidentiality"]
    ir = _compile("medical_confidentiality")
    assert ir.deme_verdict.verdict == g["deme_verdict"]
    for name, em in ir.em_outputs.items():
        gem = g["em_outputs"][name]
        assert em.score.value == gem["value"], f"{name} value drifted"
        assert em.score.confidence == gem["confidence"], f"{name} confidence drifted"


def test_whistleblower_em_outputs_match_golden() -> None:
    g = _load_golden()["whistleblower"]
    ir = _compile("whistleblower")
    assert ir.deme_verdict.verdict == g["deme_verdict"]
    for name, em in ir.em_outputs.items():
        gem = g["em_outputs"][name]
        assert em.score.value == gem["value"], f"{name} value drifted"
        assert em.score.confidence == gem["confidence"], f"{name} confidence drifted"


# ----------------------------------------------------- graph-native reads


def test_facts_of_kind_reads_graph_when_flat_cleared() -> None:
    """With graph attached and flat lists explicitly cleared, the helper
    must still find facts via graph FACT nodes."""
    ir = _compile("nazi_attic")
    assert ir.graph is not None

    ir.ethical_facts = []  # blank the flat list
    facts = facts_of_kind(ir, "externality")
    assert facts, "graph path returned no externality facts after flat clear"
    assert facts[0].kind == "externality"


def test_active_commitments_reads_graph_when_flat_cleared() -> None:
    ir = _compile("nazi_attic")
    ir.commitments = []
    comms = active_commitments(ir)
    assert comms, "graph path returned no commitments after flat clear"


def test_nonconsenting_third_party_ids_via_graph() -> None:
    ir = _compile("nazi_attic")
    ir.stakeholders = []
    sids = nonconsenting_third_party_ids(ir)
    # The village (collective target) is the canonical non-consenting party
    # in nazi_attic.
    assert any("village" in s for s in sids)


def test_vulnerable_stakeholders_via_graph() -> None:
    ir = _compile("nazi_attic")
    ir.stakeholders = []
    vs = vulnerable_stakeholders(ir)
    # The village is marked high vulnerability by the rule extractor.
    assert any("village" in s.id for s in vs)


def test_stakeholders_with_role_via_graph() -> None:
    ir = _compile("nazi_attic")
    ir.stakeholders = []
    agents = stakeholders_with_role(ir, "agent")
    # `self` is the document narrator marked agent.
    assert any(s.id == "self" for s in agents)


# ----------------------------------------------------- flat fallback path


def test_helpers_work_without_graph_attached() -> None:
    """An IR with no graph should still produce correct EM outputs
    via the flat-field fallback."""
    from erisml_compiler.ir.schemas import (
        Commitment, CompilerIR, Document, EthicalFact, Stakeholder,
    )

    doc = Document(doc_id="t", title="t", raw_text="t")
    ir = CompilerIR(
        document=doc,
        stakeholders=[
            Stakeholder(id="self", label="x", type="individual", roles=["agent"]),
            Stakeholder(id="v", label="village", type="community",
                        roles=["nonconsenting_third_party"],
                        vulnerability="high",
                        consent_status="not_obtained"),
        ],
        commitments=[
            Commitment(id="c1", type="vow", holder="self", content="x",
                       status="active_but_defeasible"),
        ],
        ethical_facts=[
            EthicalFact(id="f1", kind="externality", subjects=["v"],
                        description="x", severity="grave"),
        ],
        graph=None,
    )
    assert ir.graph is None  # confirm we're testing the fallback

    facts = facts_of_kind(ir, "externality")
    assert len(facts) == 1
    comms = active_commitments(ir)
    assert len(comms) == 1
    sids = nonconsenting_third_party_ids(ir)
    assert "v" in sids
