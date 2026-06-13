"""Project Social Chem 101 MFT-tagged RoTs onto the EM-DAG modules.

The MFT (Moral Foundations Theory, Haidt 2007) five channels don't
map 1:1 onto our 10-module EM-DAG, so we use a hand-curated mapping
with explicit weights. The full mapping is recorded in every emitted
profile YAML so callers can audit how MFT signals were translated.

Modules with no MFT channel mapping (epistemic, autonomy, externality,
repair) receive zero contribution from this projection — they get the
floor weight at normalisation. This is *honest* — Dear Abby's audience
genuinely does not surface epistemic-virtue concerns the way they
surface care/loyalty/fairness ones, and the fitted profile should
reflect that asymmetry rather than make up signal where there is none.
"""

from __future__ import annotations

from collections import defaultdict

from erisml_compiler.social_chem.schema import (
    Situation,
    SituationAggregate,
    SituationRoT,
)


MFT_FOUNDATIONS: tuple[str, ...] = (
    "care-harm",
    "fairness-cheating",
    "loyalty-betrayal",
    "authority-subversion",
    "sanctity-degradation",
)


DEFAULT_MFT_TO_EM_DAG: dict[str, dict[str, float]] = {
    "care-harm": {"harm": 1.0, "care": 0.8},
    "fairness-cheating": {"fairness": 1.0},
    "loyalty-betrayal": {"fidelity": 1.0, "legitimacy": 0.4},
    "authority-subversion": {"legitimacy": 1.0},
    "sanctity-degradation": {"rights": 0.5},
}
"""MFT-foundation -> {EM-module-name: contribution_weight in [0, 1]}.

Keys are the canonical lowercase module names (`module.name`), which
is what the moral_vector projector dispatches on. Class names
('HarmEM') are NOT used here.

Multiple foundations can contribute to the same module (e.g. `harm`
gets care-harm). Multiple modules can receive from one foundation
(e.g. care-harm lifts both `harm` and `care`, with `care` at 0.8
because the link is slightly less direct).

These weights are *projection weights* — they shape how an MFT label
becomes per-module evidence. They are NOT the fitted ethos weights;
those are computed empirically from the corpus in fitting.py.
"""


def _foundation_judgment(
    rot: SituationRoT,
    *,
    foundation: str,
) -> tuple[float, float] | None:
    """For one RoT row, return (signed_value, confidence) if it tags
    `foundation`, else None.

    - signed_value comes from action-moral-judgment normalised to
      [-1, +1] (so -2 -> -1.0, 0 -> 0.0, +2 -> +1.0). If
      action-moral-judgment is missing, falls back to rot-judgment
      ("It is bad to" -> -1, "It is good to" -> +1, else 0).
    - confidence comes from rot-agree normalised to [0, 1]
      (1 -> 0.2, ..., 5 -> 1.0). Missing rot-agree -> 0.5.
    """
    if foundation not in rot.rot_moral_foundations:
        return None

    if rot.action_moral_judgment is not None:
        v = max(-1.0, min(1.0, rot.action_moral_judgment / 2.0))
    else:
        # rot.rot starts with phrases like "It is bad to" / "It is good to"
        lower = rot.rot.lower()
        if any(p in lower for p in ("it is bad", "it's bad", "it is wrong", "you shouldn't")):
            v = -0.5
        elif any(p in lower for p in ("it is good", "it's good", "you should ", "you ought")):
            v = 0.5
        else:
            v = 0.0

    if rot.rot_agree is not None:
        c = max(0.0, min(1.0, rot.rot_agree / 5.0))
    else:
        c = 0.5

    return v, c


def project_situation(
    situation: Situation,
    *,
    mapping: dict[str, dict[str, float]] | None = None,
) -> SituationAggregate:
    """Aggregate one situation's RoT rows into per-EM-module signals.

    For each foundation that appears in any RoT row, compute an
    agreement-weighted average value across rows tagging it. Then
    project that foundation's signal onto each EM module per the
    mapping, weighted by the mapping's contribution coefficient.

    Per-module value is a weighted mean across contributing
    foundations; per-module confidence is the max coverage across
    contributing foundations (we don't sum confidences because two
    weak signals to the same module shouldn't compound past 1.0).
    """
    m = DEFAULT_MFT_TO_EM_DAG if mapping is None else mapping

    sums_v: dict[str, float] = defaultdict(float)
    sums_c: dict[str, float] = defaultdict(float)
    cov: dict[str, float] = defaultdict(float)

    for foundation in MFT_FOUNDATIONS:
        readings: list[tuple[float, float]] = []
        for r in situation.rots:
            row = _foundation_judgment(r, foundation=foundation)
            if row is not None:
                readings.append(row)
        if not readings:
            continue
        c_total = sum(c for _, c in readings)
        if c_total <= 0:
            f_val = sum(v for v, _ in readings) / len(readings)
            f_conf = 0.0
        else:
            f_val = sum(v * c for v, c in readings) / c_total
            f_conf = min(1.0, c_total / max(1, len(situation.rots)))

        for em_module, w in m.get(foundation, {}).items():
            sums_v[em_module] += f_val * w
            sums_c[em_module] += w
            cov[em_module] = max(cov[em_module], f_conf)

    per_value: dict[str, float] = {}
    per_conf: dict[str, float] = {}
    for em_module, w_total in sums_c.items():
        if w_total <= 0:
            continue
        per_value[em_module] = max(-1.0, min(1.0, sums_v[em_module] / w_total))
        per_conf[em_module] = cov[em_module]

    return SituationAggregate(
        situation_short_id=situation.situation_short_id,
        per_module_value=per_value,
        per_module_confidence=per_conf,
        n_rots=len(situation.rots),
    )
