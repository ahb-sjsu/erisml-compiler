"""Benchmark: the ErisML compiler vs a black-box scalar classifier.

The honest claim under test is NOT "ErisML is more moral." It is the Churchill /
procedural claim: a structure-preserving, auditable compiler is *negation-aware*
and *shows its work* where a scalar/keyword black box is neither.

Two axes are measured:

  1. Structural correctness — does the verdict respond to negation? Reported
     overall and on the negated subset (where the gap lives).
  2. Auditability — how many inspectable reasoning fields each system exposes
     per decision (ErisML: action_kind, polarity, contradiction_type,
     justification; black box: none).

The ErisML path uses the shipped components: the spaCy SRL maxim extractor and
the polarity-aware universalizability gate. The black box is a keyword/scalar
classifier (see baseline.py) standing in for safety filters / scalar reward
models.

Run:
    python -m benchmarks.run_vs_blackbox
"""
from __future__ import annotations

from dataclasses import dataclass

from erisml_compiler.annotation.maxim_extractor_srl import extract_maxim_srl, is_srl_available
from erisml_compiler.delta.universalizability import test_universalizability

from .baseline import BlackBoxScorer
from .corpus import GOLD


@dataclass
class ErisMLResult:
    verdict: str
    trace: dict
    reasoning_fields: list[str]


def erisml_judge(text: str) -> ErisMLResult:
    """Deontic verdict via the shipped ErisML reasoning path, with an audit trace."""
    maxim, _ev = extract_maxim_srl(text)
    if maxim is None or maxim.action_kind is None:
        return ErisMLResult("permit", {"reason": "no maxim extracted"}, [])
    dep = test_universalizability(maxim.action_kind, polarity=maxim.polarity)
    verdict = "permit" if dep.passes else "forbid"
    trace = {
        "action_kind": maxim.action_kind,
        "polarity": maxim.polarity,
        "contradiction_type": dep.contradiction_type,
        "justification": dep.justification,
    }
    return ErisMLResult(verdict, trace, list(trace.keys()))


def run() -> int:
    if not is_srl_available():
        print("spaCy / en_core_web_sm not installed; cannot run the ErisML path.")
        return 1

    bb = BlackBoxScorer()
    rows = []
    e_correct = b_correct = 0
    e_neg_correct = b_neg_correct = e_neg_total = 0
    e_fields = b_fields = 0

    for text, gold, _cat, negated in GOLD:
        e = erisml_judge(text)
        b = bb.judge(text)
        e_ok = e.verdict == gold
        b_ok = b.verdict == gold
        e_correct += e_ok
        b_correct += b_ok
        e_fields += len(e.reasoning_fields)
        b_fields += len(b.reasoning_fields)
        if negated:
            e_neg_total += 1
            e_neg_correct += e_ok
            b_neg_correct += b_ok
        rows.append((text, gold, e.verdict, e_ok, b.verdict, b_ok, negated))

    n = len(GOLD)
    print("ErisML compiler  vs  black-box scalar classifier")
    print("=" * 78)
    print(f"{'scenario':<46}{'gold':>7}{'ErisML':>9}{'blackbox':>10}")
    print("-" * 78)
    for text, gold, ev, eok, bv, bok, neg in rows:
        tag = "neg" if neg else "   "
        em = "ok" if eok else "X"
        bm = "ok" if bok else "X"
        print(f"{tag} {text[:42]:<42}{gold:>7}{ev+'('+em+')':>11}{bv+'('+bm+')':>12}")
    print("-" * 78)
    print(f"{'overall accuracy':<46}{'':>7}{e_correct/n:>9.0%}{b_correct/n:>10.0%}")
    if e_neg_total:
        print(f"{'accuracy on negated items':<46}{'':>7}"
              f"{e_neg_correct/e_neg_total:>9.0%}{b_neg_correct/e_neg_total:>10.0%}")
    print(f"{'avg reasoning fields exposed':<46}{'':>7}"
          f"{e_fields/n:>9.1f}{b_fields/n:>10.1f}")
    print("=" * 78)
    print("Structural correctness AND auditability - the procedural claim, measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
