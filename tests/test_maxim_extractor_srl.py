"""Tests for the SRL-based maxim extractor (spaCy dependency parser)."""

from __future__ import annotations

import pytest

from erisml_compiler.annotation.maxim_extractor_srl import (
    _disambiguate_expose,
    extract_maxim_srl,
    is_srl_available,
)
from erisml_compiler.ir.schemas import Stakeholder

# Skip the whole module if spaCy isn't installed.
pytestmark = pytest.mark.skipif(
    not is_srl_available(), reason="spaCy + en_core_web_sm not installed"
)


# ----------------------------------------------------- subject-of-verb


def test_srl_resolves_first_person_subject() -> None:
    text = "I lied to protect the family from the soldiers."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert m.action_kind == "deceive"
    assert m.agent_id == "self"
    assert ev.subject_text.lower() in ("i",)


def test_srl_resolves_third_person_subject_to_labeled_stakeholder() -> None:
    text = "Alice deceived her brother."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="alice", label="Alice", type="individual", roles=["agent"]),
            Stakeholder(id="brother", label="brother", type="individual"),
        ],
    )
    assert m is not None
    assert m.action_kind == "deceive"
    assert m.agent_id == "alice"


def test_srl_picks_main_action_not_subordinate() -> None:
    """For 'I decided to lie to protect the family', the moral action
    is 'lie' (which the agent's deliberation centres on), not 'decide'.
    Our scoring promotes meaningful action over meta-verbs."""
    text = "I decided to lie to protect the family."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    # 'decide' isn't in our action_kind library; the SRL skips it and
    # finds 'lie' under xcomp.
    assert m.action_kind == "deceive"


def test_srl_distinguishes_agent_from_patient() -> None:
    """The medical-confidentiality-style sentence has 'patient
    revealed a plan to harm'. The act being deliberated is the
    doctor's potential warning, not the patient's planned harm.
    The SRL should NOT attribute 'harm' to the doctor."""
    text = "The patient revealed a plan to seriously harm a former colleague."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="doctor", type="individual", roles=["agent"]),
            Stakeholder(id="patient_001", label="patient", type="individual"),
        ],
    )
    # The SRL picks 'reveal' (the patient's actual action in this
    # sentence) and attributes it to the patient — NOT to the doctor.
    # This is the key correctness property the v1 regex missed.
    assert m is not None
    assert (
        m.agent_id != "self"
    ), f"SRL incorrectly attributed verb to the doctor; got agent_id={m.agent_id}"


# ----------------------------------------------------- polysemy


def test_expose_polysemy_disclose_reading() -> None:
    """`expose wrongdoing` is whistleblower-style disclosure."""
    text = "I plan to expose the corporate fraud to regulators."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert (
        m.action_kind == "disclose"
    ), f"Expected expose+wrongdoing -> disclose; got {m.action_kind}"


def test_expose_polysemy_risk_reading() -> None:
    """`expose X to risk` is impose_externality."""
    text = "She exposed the public to losses they did not consent to."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert m.action_kind == "impose_externality"


# ----------------------------------------------------- contextual gates


def test_break_only_matches_promise_break() -> None:
    """'break the window' is not break_commitment."""
    text = "She broke the window with a rock."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    # 'break' should not match — no commitment-noun in dobj.
    # m may be None if no other verb matches in this text.
    if m is not None:
        assert m.action_kind != "break_commitment"


def test_break_matches_promise_break() -> None:
    text = "He broke his promise to attend the meeting."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert m.action_kind == "break_commitment"


def test_use_only_matches_instrumental_use() -> None:
    """'use a hammer' isn't use_as_means."""
    text = "She used a hammer to drive the nail."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    if m is not None:
        assert m.action_kind != "use_as_means"


def test_use_as_means_pattern() -> None:
    text = "He used the intern as a tool to advance his career."
    m, ev = extract_maxim_srl(
        text,
        stakeholders=[
            Stakeholder(id="self", label="narrator", type="individual", roles=["agent"]),
        ],
    )
    assert m is not None
    assert m.action_kind == "use_as_means"


# ----------------------------------------------------- empty / boundary


def test_empty_text_returns_none() -> None:
    m, ev = extract_maxim_srl("", stakeholders=[])
    assert m is None


def test_no_moral_verb_returns_none() -> None:
    text = "The weather is fine today."
    m, ev = extract_maxim_srl(text, stakeholders=[])
    assert m is None


# ----------------------------------------------------- _disambiguate_expose unit


def test_disambiguate_expose_helper_directly() -> None:
    """Direct unit test that doesn't need a stakeholder list."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    for text, expected in [
        ("She exposed the fraud.", "disclose"),
        ("They exposed the public to losses.", "impose_externality"),
        ("He exposed misconduct in the firm.", "disclose"),
    ]:
        doc = nlp(text)
        expose_tok = next(t for t in doc if t.lemma_.lower() == "expose")
        assert _disambiguate_expose(expose_tok) == expected, f"{text!r} expected {expected}"
