"""DEME interpretation of Maxim.polarity.

A negated maxim ("did not promise", "refused to lie") must be evaluated as the
maxim of *not performing* the action, not as the affirmed action. These tests
cover the universalizability KB, the deontic gate, and the virtue gate.
"""
from __future__ import annotations

from erisml_compiler.delta.universalizability import (
    test_universalizability as check_universalizability,
)
from erisml_compiler.ir.schemas import Document
from erisml_compiler.projections import Maxim, MoralSubstrate
from erisml_compiler.projections.deontic import DeonticProjection
from erisml_compiler.projections.virtue import VirtueProjection


def _sub(action_kind: str, polarity: str) -> MoralSubstrate:
    return MoralSubstrate(
        document=Document(doc_id="t", title="t", raw_text="t"),
        maxim=Maxim(description=action_kind, action_kind=action_kind, polarity=polarity),
    )


# ----------------------------------------------------- universalizability KB


def test_negating_a_prohibition_passes():
    assert check_universalizability("deceive").passes is False
    neg = check_universalizability("deceive", polarity="negated")
    assert neg.passes is True
    assert neg.action_kind == "not:deceive"


def test_negating_an_imperfect_duty_fails_in_will():
    assert check_universalizability("help").passes is True
    neg = check_universalizability("help", polarity="negated")
    assert neg.passes is False
    assert neg.contradiction_type == "contradiction_in_will"


def test_negating_a_permissible_act_passes():
    # Promising is permissible, not a duty — so NOT promising is fine.
    assert check_universalizability("make_or_keep_commitment").passes is True
    assert check_universalizability("make_or_keep_commitment", polarity="negated").passes is True


def test_negating_unknown_action_is_undetermined():
    neg = check_universalizability("mystery_action", polarity="negated")
    assert neg.contradiction_type == "undetermined"


# ----------------------------------------------------- deontic gate


def test_deontic_gate_flips_for_negated_prohibition():
    affirmed = DeonticProjection()._gate_universalizability(_sub("deceive", "affirmed"))
    negated = DeonticProjection()._gate_universalizability(_sub("deceive", "negated"))
    assert affirmed.passed is False  # lying fails
    assert negated.passed is True    # not lying passes
    assert negated.detail["polarity"] == "negated"


def test_deontic_gate_flips_for_negated_imperfect_duty():
    affirmed = DeonticProjection()._gate_universalizability(_sub("protect", "affirmed"))
    negated = DeonticProjection()._gate_universalizability(_sub("protect", "negated"))
    assert affirmed.passed is True   # protecting passes
    assert negated.passed is False   # not protecting fails (CIW)


# ----------------------------------------------------- virtue gate


def test_virtue_gate_refraining_from_vice_is_not_a_concern():
    proj = VirtueProjection()
    affirmed = proj._character_finding(_sub("deceive", "affirmed"))
    negated = proj._character_finding(_sub("deceive", "negated"))
    assert affirmed.passed is False  # deceiving = vice concern
    assert negated.passed is True    # not deceiving = expresses honesty


def test_virtue_gate_omitting_a_virtue_is_a_concern():
    proj = VirtueProjection()
    negated = proj._character_finding(_sub("protect", "negated"))
    assert negated.passed is False   # not protecting = callousness
