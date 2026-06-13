"""Tests for the pattern-based maxim extractor (release-planning-06 v1)."""

from __future__ import annotations

from pathlib import Path

from erisml_compiler.annotation.maxim_extractor import extract_maxim
from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.ir.graph import NodeKind
from erisml_compiler.ir.schemas import Stakeholder
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------- direct extractor


def test_deceive_verb_recognised() -> None:
    text = "I had to lie to keep the family safe."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.action_kind == "deceive"
    assert ev.matched_verb == "lie"


def test_promise_verb_recognised() -> None:
    text = "She promised to return by sundown."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.action_kind == "make_or_keep_commitment"


def test_protect_verb_recognised() -> None:
    text = "He shielded the children from the falling debris."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.action_kind == "protect"


def test_coerce_verb_recognised() -> None:
    text = "They threatened to fire her if she spoke up."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.action_kind == "coerce"


def test_disclose_verb_recognised() -> None:
    text = "Eventually she decided to disclose the fraud to regulators."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.action_kind == "disclose"


def test_no_verb_no_fallback_returns_none() -> None:
    text = "The weather is pleasant today."
    m, ev = extract_maxim(text, prefer_srl=False, fallback_action_kind=None)
    assert m is None


def test_no_verb_with_fallback_uses_fallback() -> None:
    text = "The weather is pleasant today."
    m, ev = extract_maxim(text, prefer_srl=False, fallback_action_kind="deceive")
    assert m is not None
    assert m.action_kind == "deceive"
    assert ev.matched_verb is None


def test_purpose_clause_extracted() -> None:
    text = "He hid the documents to protect his client."
    m, ev = extract_maxim(text, prefer_srl=False)
    assert m is not None
    assert m.purpose is not None
    assert "protect" in (m.purpose or "").lower() or m.purpose == "protect"


def test_first_person_agent_identified() -> None:
    text = "I vowed to keep the secret."
    m, ev = extract_maxim(
        text,
        prefer_srl=False,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert m.agent_id == "self"
    assert ev.agent_evidence == "first_person"


def test_third_person_falls_back_to_agent_role() -> None:
    text = "The doctor decided to disclose the diagnosis."
    m, ev = extract_maxim(
        text,
        prefer_srl=False,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    # No stakeholder labelled "doctor" → falls through to agent-role stakeholder.
    assert m.agent_id == "self"
    assert "agent-role" in (ev.agent_evidence or "")


def test_mere_means_pattern_detected() -> None:
    text = "He used her as a tool to advance his own career."
    m, ev = extract_maxim(
        text,
        prefer_srl=False,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert ev.mere_means_hits  # at least one match


# ---------------------------------------------------- via compile pipeline


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


def test_compile_produces_maxim_node_on_nazi_attic() -> None:
    """SRL picks 'vow' as the main verb of the first sentence (correct
    subject-of-verb attribution); the substrate's other gates still
    trip 'forbidden' verdict via the externality + mere_means findings."""
    ir = _compile("nazi_attic")
    maxim_nodes = ir.graph.nodes_of_kind(NodeKind.MAXIM)
    assert len(maxim_nodes) == 1
    m = maxim_nodes[0]
    # With SRL enabled, the action_kind is whichever moral verb scores
    # highest. Could be "deceive" (regex v1 sees "lie") or
    # "make_or_keep_commitment" (SRL sees "vowed" as the main verb).
    # Either is a legitimate maxim reading of the same scene.
    assert m.payload["action_kind"] in ("deceive", "make_or_keep_commitment")
    assert m.payload["extraction_evidence"]["matched_verb"] is not None


def test_compile_produces_maxim_node_on_medical() -> None:
    ir = _compile("medical_confidentiality")
    maxim_nodes = ir.graph.nodes_of_kind(NodeKind.MAXIM)
    assert len(maxim_nodes) == 1


def test_compile_produces_maxim_node_on_whistleblower() -> None:
    ir = _compile("whistleblower")
    maxim_nodes = ir.graph.nodes_of_kind(NodeKind.MAXIM)
    assert len(maxim_nodes) == 1
    m = maxim_nodes[0]
    # SRL may pick "expose" (the firm's act on the public) or
    # "disclose" (the whistleblower's contemplated act). Both are
    # surfaced in the prose; we accept either.
    assert m.payload["action_kind"] in ("disclose", "impose_externality")


def test_evidence_block_records_matched_verb_in_audit_chain() -> None:
    """The maxim node's extraction_evidence sub-dict is recorded in
    the graph payload and propagates into the audit's graph_hash."""
    ir = _compile("nazi_attic")
    m = ir.graph.nodes_of_kind(NodeKind.MAXIM)[0]
    ev = m.payload.get("extraction_evidence", {})
    assert "matched_verb" in ev
    assert "agent_evidence" in ev
    assert "fallback_used" in ev


def test_deontic_verdict_still_forbidden_on_nazi_attic_post_prose() -> None:
    """Regression: prose-extracted maxim still trips the universalizability gate."""
    ir = _compile("nazi_attic")
    assert ir.projections["deontic_kantian"]["verdict"] == "forbidden"
