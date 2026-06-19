"""Black-box baseline for the comparison benchmark.

A keyword/scalar moral classifier: it maps text to a single harm score by
lexicon lookup and thresholds it into a verdict. This stands in for the
*actual* deployed alternatives to a structured moral-reasoning IR — keyword
safety filters and scalar (RLHF-style) reward/▒harm classifiers. Like them, it:

  * collapses everything to one number,
  * is syntactically blind (no notion of negation, agency, or maxim), and
  * exposes no inspectable reasoning — only the scalar.

It is deliberately NOT a strawman on coverage: the lexicon includes the same
violation vocabulary ErisML knows about. The point of the benchmark is not
"ErisML knows more words" — it is "ErisML reasons over structure (negation,
action-kind) and shows its work, where a scalar classifier cannot."

NOTE: a modern LLM judge would handle negation better than this lexicon. It
would still fail the *auditability* and *determinism* axes (it emits a verdict,
not an inspectable trace, and is nondeterministic). This baseline isolates the
structural axis; the auditability axis is reported separately and holds against
an LLM too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Violation lexicon (harm-bearing terms). Mirrors ErisML's prohibition kinds.
_VIOLATION_TERMS: dict[str, float] = {
    "lie": 0.9, "lied": 0.9, "lying": 0.9, "deceive": 0.9, "deceived": 0.9,
    "mislead": 0.85, "misled": 0.85,
    "coerce": 0.9, "coerced": 0.9, "threaten": 0.85, "threatened": 0.85,
    "harm": 0.9, "harmed": 0.9, "hurt": 0.8, "injure": 0.85, "injured": 0.85,
    "kill": 1.0, "killed": 1.0, "steal": 0.85, "stole": 0.85, "cheat": 0.85,
    "exploit": 0.8, "exploited": 0.8, "fail": 0.6, "failed": 0.6, "neglect": 0.7,
}

_THRESHOLD = 0.5  # harm score at/above which the action is "forbidden"


@dataclass
class BlackBoxResult:
    """What a scalar black box exposes: a number and a verdict. No trace."""

    verdict: str            # "forbid" | "permit"
    harm_score: float
    reasoning_fields: list[str] = field(default_factory=list)  # always empty


class BlackBoxScorer:
    """Keyword/scalar moral classifier (the black-box baseline)."""

    name = "black_box_lexicon"

    def judge(self, text: str) -> BlackBoxResult:
        tokens = "".join(c.lower() if c.isalpha() else " " for c in text).split()
        score = max((_VIOLATION_TERMS.get(t, 0.0) for t in tokens), default=0.0)
        verdict = "forbid" if score >= _THRESHOLD else "permit"
        # A scalar classifier exposes no inspectable reasoning — by construction.
        return BlackBoxResult(verdict=verdict, harm_score=score, reasoning_fields=[])
