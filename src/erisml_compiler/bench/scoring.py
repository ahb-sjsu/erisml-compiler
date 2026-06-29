"""Per-metric scoring + weighted aggregate for MoralTensor-Bench."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from erisml_compiler.bench.schema import (
    BenchAggregate,
    ExpectedCommitment,
    ExpectedStakeholder,
    ScenarioGold,
    ScenarioScore,
)
from erisml_compiler.ir.schemas import CompilerIR

DEFAULT_WEIGHTS_FILE = Path(__file__).resolve().parent / "v0.1" / "weights.yaml"


def _fuzzy_id_eq(a: str, b: str) -> bool:
    """Cheap label-fuzzy match: case-insensitive, ignore underscores
    and hyphens. Sufficient for the seed corpus; reconsider when the
    extractor starts emitting richer entity IDs."""

    def norm(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    return norm(a) == norm(b)


def score_stakeholder_recall(expected: list[ExpectedStakeholder], ir: CompilerIR) -> float:
    if not expected:
        return 1.0
    ir_ids = {s.id for s in ir.stakeholders}
    matched = sum(1 for e in expected if any(_fuzzy_id_eq(e.id, i) for i in ir_ids))
    return matched / len(expected)


def score_stakeholder_role_f1(expected: list[ExpectedStakeholder], ir: CompilerIR) -> float:
    expected_pairs: set[tuple[str, str]] = set()
    for e in expected:
        for role in e.roles:
            expected_pairs.add((e.id.lower(), role.lower()))
    ir_pairs: set[tuple[str, str]] = set()
    for s in ir.stakeholders:
        sid = s.id.lower()
        roles = getattr(s, "roles", None) or []
        for r in roles:
            ir_pairs.add((sid, r.lower()))
    if not expected_pairs and not ir_pairs:
        return 1.0
    if not expected_pairs or not ir_pairs:
        return 0.0
    tp = sum(
        1
        for ep in expected_pairs
        if any(_fuzzy_id_eq(ep[0], ip[0]) and ep[1] == ip[1] for ip in ir_pairs)
    )
    precision = tp / len(ir_pairs) if ir_pairs else 0.0
    recall = tp / len(expected_pairs) if expected_pairs else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_commitment_f1(expected: list[ExpectedCommitment], ir: CompilerIR) -> float:
    def _key(c: Any) -> tuple[str, str, str]:
        holder = getattr(c, "holder", "") or ""
        ben = getattr(c, "beneficiary", "") or ""
        typ = getattr(c, "type", "") or ""
        return holder.lower(), ben.lower(), typ.lower()

    exp_keys = {(c.holder.lower(), (c.beneficiary or "").lower(), c.type.lower()) for c in expected}
    ir_keys = {_key(c) for c in ir.commitments}
    if not exp_keys and not ir_keys:
        return 1.0
    if not exp_keys or not ir_keys:
        return 0.0
    tp = 0
    for e in exp_keys:
        for i in ir_keys:
            if _fuzzy_id_eq(e[0], i[0]) and _fuzzy_id_eq(e[1], i[1]) and e[2] == i[2]:
                tp += 1
                break
    p = tp / len(ir_keys) if ir_keys else 0.0
    r = tp / len(exp_keys) if exp_keys else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score_canonical_form(expected: str | None, ir: CompilerIR) -> float:
    if expected is None:
        return math.nan
    actual = ir.canonical_form or ""
    return 1.0 if actual.lower() == expected.lower() else 0.0


def score_ethical_fact_kind_recall(expected_kinds: list[str], ir: CompilerIR) -> float:
    if not expected_kinds:
        return 1.0
    # EthicalFact uses .kind (an EthicalFactKind enum) in the IR schema.
    ir_kinds = {
        (
            (getattr(f, "kind", None) or getattr(f, "type", "") or "").lower()
            if not hasattr(getattr(f, "kind", None), "value")
            else getattr(f.kind, "value").lower()
        )
        for f in ir.ethical_facts
    }
    matched = sum(1 for k in expected_kinds if k.lower() in ir_kinds)
    return matched / len(expected_kinds)


def score_per_party_verdicts(expected: dict[str, str], ir: CompilerIR) -> float:
    if not expected:
        return 1.0
    # IR shape for per-party verdicts varies across V2/V3 paths.
    # Prefer ir.deme_verdict.per_party if present.
    per_party: dict[str, str] = {}
    if ir.deme_verdict is not None:
        pp = getattr(ir.deme_verdict, "per_party", None) or {}
        for k, v in pp.items():
            per_party[k.lower()] = (getattr(v, "verdict", None) or str(v)).lower()
    if not per_party:
        return 0.0  # nothing to compare against
    matched = 0
    for sid, expected_verdict in expected.items():
        sid_l = sid.lower()
        # exact-id then fuzzy-match
        actual = per_party.get(sid_l)
        if actual is None:
            for k, v in per_party.items():
                if _fuzzy_id_eq(sid_l, k):
                    actual = v
                    break
        if actual is not None and actual == expected_verdict.lower():
            matched += 1
    return matched / len(expected)


def score_overall_verdict(expected: str | None, ir: CompilerIR) -> float:
    if expected is None:
        return math.nan
    if ir.deme_verdict is None:
        return 0.0
    return 1.0 if ir.deme_verdict.verdict.lower() == expected.lower() else 0.0


def score_premature_contraction(expect_premature: bool, ir: CompilerIR) -> float:
    """1.0 = premature contraction occurred (BAD). 0.0 = no premature
    contraction. The aggregate `premature_contraction_rate` is the
    mean of this column."""
    if not expect_premature:
        return 0.0
    if ir.deme_verdict is None:
        return 0.0
    return 0.0 if ir.deme_verdict.verdict == "requires_human_review" else 1.0


def score_scenario(scenario: ScenarioGold, ir: CompilerIR) -> ScenarioScore:
    e = scenario.expected
    return ScenarioScore(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        stakeholder_recall=score_stakeholder_recall(e.stakeholders, ir),
        stakeholder_role_f1=score_stakeholder_role_f1(e.stakeholders, ir),
        commitment_f1=score_commitment_f1(e.commitments, ir),
        canonical_form_match=score_canonical_form(e.canonical_form, ir),
        ethical_fact_kind_recall=score_ethical_fact_kind_recall(e.ethical_fact_kinds, ir),
        per_party_verdict_accuracy=score_per_party_verdicts(e.per_party_verdicts, ir),
        overall_verdict_match=score_overall_verdict(e.overall_verdict, ir),
        premature_contraction=score_premature_contraction(e.expect_premature_contraction, ir),
        raw={
            "n_stakeholders_ir": len(ir.stakeholders),
            "n_commitments_ir": len(ir.commitments),
            "canonical_form_ir": ir.canonical_form,
            "verdict_ir": ir.deme_verdict.verdict if ir.deme_verdict else None,
        },
    )


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    p = Path(path) if path else DEFAULT_WEIGHTS_FILE
    if not p.exists():
        # Sensible default if no weights file is present.
        return {
            "stakeholder_recall": 0.15,
            "stakeholder_role_f1": 0.15,
            "commitment_f1": 0.15,
            "canonical_form_match": 0.1,
            "ethical_fact_kind_recall": 0.15,
            "per_party_verdict_accuracy": 0.15,
            "overall_verdict_match": 0.1,
            "premature_contraction_rate_penalty": 0.05,
        }
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _mean_skip_nan(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def weighted_score(agg: dict[str, float], weights: dict[str, float]) -> float:
    """Compute the moral_tensor_bench_score given per-metric means
    and a weights dict. Premature-contraction enters as a penalty
    (subtracted)."""
    pos_components = (
        "stakeholder_recall",
        "stakeholder_role_f1",
        "commitment_f1",
        "canonical_form_match",
        "ethical_fact_kind_recall",
        "per_party_verdict_accuracy",
        "overall_verdict_match",
    )
    score = 0.0
    total_w = 0.0
    for k in pos_components:
        w = weights.get(k, 0.0)
        v = agg.get(f"mean_{k}", 0.0)
        if math.isnan(v):
            continue
        score += w * v
        total_w += w
    if total_w > 0:
        score = score / total_w
    penalty_w = weights.get("premature_contraction_rate_penalty", 0.0)
    score -= penalty_w * agg.get("premature_contraction_rate", 0.0)
    return max(0.0, min(1.0, score))


def aggregate_score(
    scores: list[ScenarioScore], weights: dict[str, float] | None = None
) -> BenchAggregate:
    w = weights if weights is not None else load_weights()
    n = len(scores)
    if n == 0:
        return BenchAggregate(
            n_scenarios=0,
            n_failed_compile=0,
            mean_stakeholder_recall=0.0,
            mean_stakeholder_role_f1=0.0,
            mean_commitment_f1=0.0,
            mean_canonical_form_match=0.0,
            mean_ethical_fact_kind_recall=0.0,
            mean_per_party_verdict_accuracy=0.0,
            mean_overall_verdict_match=0.0,
            premature_contraction_rate=0.0,
            moral_tensor_bench_score=0.0,
        )
    means = {
        "mean_stakeholder_recall": sum(s.stakeholder_recall for s in scores) / n,
        "mean_stakeholder_role_f1": sum(s.stakeholder_role_f1 for s in scores) / n,
        "mean_commitment_f1": sum(s.commitment_f1 for s in scores) / n,
        "mean_canonical_form_match": _mean_skip_nan([s.canonical_form_match for s in scores]),
        "mean_ethical_fact_kind_recall": sum(s.ethical_fact_kind_recall for s in scores) / n,
        "mean_per_party_verdict_accuracy": sum(s.per_party_verdict_accuracy for s in scores) / n,
        "mean_overall_verdict_match": _mean_skip_nan([s.overall_verdict_match for s in scores]),
        "premature_contraction_rate": sum(s.premature_contraction for s in scores) / n,
    }
    score = weighted_score(means, w)
    return BenchAggregate(
        n_scenarios=n,
        n_failed_compile=0,
        **{k: (0.0 if math.isnan(v) else v) for k, v in means.items()},
        moral_tensor_bench_score=score,
    )
