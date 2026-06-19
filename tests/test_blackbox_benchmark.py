"""Locks the ErisML-vs-black-box benchmark finding.

The claim is procedural, not "more moral": ErisML is negation-aware and
auditable where a scalar keyword classifier is neither. Skips if spaCy is
absent (the ErisML path needs the SRL extractor).
"""

from __future__ import annotations

import pytest

from erisml_compiler.annotation.maxim_extractor_srl import is_srl_available

pytestmark = pytest.mark.skipif(
    not is_srl_available(), reason="spaCy + en_core_web_sm not installed"
)

from benchmarks.baseline import BlackBoxScorer  # noqa: E402
from benchmarks.corpus import GOLD  # noqa: E402
from benchmarks.run_vs_blackbox import erisml_judge  # noqa: E402


def _accuracy(judge_verdict):
    correct = total = neg_correct = neg_total = 0
    for text, gold, _cat, negated in GOLD:
        ok = judge_verdict(text) == gold
        correct += ok
        total += 1
        if negated:
            neg_total += 1
            neg_correct += ok
    return correct / total, neg_correct / neg_total


def test_erisml_beats_blackbox_on_negation():
    bb = BlackBoxScorer()
    e_overall, e_neg = _accuracy(lambda t: erisml_judge(t).verdict)
    b_overall, b_neg = _accuracy(lambda t: bb.judge(t).verdict)
    # The discriminating signal is the negated subset.
    assert e_neg > b_neg
    assert e_overall > b_overall


def test_erisml_is_auditable_blackbox_is_not():
    bb = BlackBoxScorer()
    sample = GOLD[0][0]
    assert len(erisml_judge(sample).reasoning_fields) >= 3  # action_kind, polarity, ...
    assert len(bb.judge(sample).reasoning_fields) == 0  # scalar only


def test_blackbox_misreads_negated_prohibition():
    # "did not lie" should be permitted; the keyword scorer forbids it.
    bb = BlackBoxScorer()
    assert bb.judge("The doctor did not lie to the patient.").verdict == "forbid"
    assert erisml_judge("The doctor did not lie to the patient.").verdict == "permit"
