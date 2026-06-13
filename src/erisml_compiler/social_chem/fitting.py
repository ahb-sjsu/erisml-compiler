"""Aggregate per-situation projections into a fitted ethos profile.

Fit method:

    weight_m = max(epsilon, mean_confidence_m * salience_m)

where:
    mean_confidence_m  = average across situations of per-module confidence
    salience_m         = fraction of situations where |value_m| >= 0.05

Then we normalise so sum(weights) == len(modules) — equal-weight
baseline corresponds to all weights = 1.0. The ten EM-DAG modules are
included by name even if they receive no MFT signal (epistemic, etc.)
— they take the floor weight, and that's recorded explicitly so the
profile is honest about its coverage gaps.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from erisml_compiler.social_chem.projection import DEFAULT_MFT_TO_EM_DAG
from erisml_compiler.social_chem.schema import (
    CorpusFingerprint,
    ProfileFitResult,
    SituationAggregate,
)


EM_DAG_MODULES_DEFAULT: tuple[str, ...] = (
    "harm",
    "rights",
    "fairness",
    "legitimacy",
    "epistemic",
    "autonomy",
    "fidelity",
    "externality",
    "care",
    "repair",
)
"""Canonical lowercase EM module names — matches `module.name` in
the loaded EM-DAG. Class names ('HarmEM', etc.) are NOT used as keys
in fitted ethos profiles."""

_SALIENCE_THRESHOLD = 0.05
_EPSILON = 1e-6


def aggregate_situations(
    aggregates: Iterable[SituationAggregate],
    *,
    modules: tuple[str, ...] = EM_DAG_MODULES_DEFAULT,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
    """Return (weights_raw, priors, coverage, n_situations).

    `modules` is the full set of EM modules to emit weights for —
    even those with no Social Chem signal get an entry (floor weight).
    """
    aggs = [a for a in aggregates if a.per_module_value]
    n = len(aggs)
    if n == 0:
        return {m: _EPSILON for m in modules}, {m: 0.0 for m in modules}, {m: 0.0 for m in modules}, 0

    weights_raw: dict[str, float] = {}
    priors: dict[str, float] = {}
    coverage: dict[str, float] = {}

    for em in modules:
        confs: list[float] = []
        vals: list[float] = []
        salient = 0
        fires = 0
        for a in aggs:
            v = a.per_module_value.get(em)
            c = a.per_module_confidence.get(em)
            if v is None or c is None:
                continue
            fires += 1
            confs.append(c)
            vals.append(v)
            if abs(v) >= _SALIENCE_THRESHOLD:
                salient += 1

        if not confs:
            weights_raw[em] = _EPSILON
            priors[em] = 0.0
            coverage[em] = 0.0
            continue

        mean_conf = sum(confs) / len(confs)
        salience = salient / n
        weights_raw[em] = max(_EPSILON, mean_conf * salience)
        priors[em] = sum(vals) / len(vals) if vals else 0.0
        coverage[em] = fires / n

    return weights_raw, priors, coverage, n


def normalise_weights(weights_raw: dict[str, float]) -> dict[str, float]:
    """Normalise so sum(weights) == len(modules).

    Modules with raw weight at the floor (epsilon) are dropped from
    the result: a missing module in the loader defaults to weight 1.0
    rather than ~0, which is the correct semantics for "no fitted
    signal — leave the default behaviour alone." `coverage:` still
    records 0.0 for these modules so the gap is auditable.
    """
    if not weights_raw:
        return {}
    fitted = {m: w for m, w in weights_raw.items() if w > _EPSILON * 10}
    if not fitted:
        return {}
    n = len(fitted)
    total = sum(fitted.values())
    if total < _EPSILON:
        return {m: 1.0 for m in fitted}
    scale = n / total
    return {m: w * scale for m, w in fitted.items()}


def fit_profile(
    aggregates: list[SituationAggregate],
    *,
    corpus: CorpusFingerprint,
    name: str,
    description: str,
    ethos_description: str,
    bias_notes: list[str],
    modules: tuple[str, ...] = EM_DAG_MODULES_DEFAULT,
    mapping: dict[str, dict[str, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProfileFitResult:
    """Run the fit and return a ProfileFitResult."""
    weights_raw, priors, coverage, n_used = aggregate_situations(
        aggregates, modules=modules
    )
    weights = normalise_weights(weights_raw)

    return ProfileFitResult(
        name=name,
        description=description,
        ethos_description=ethos_description,
        bias_notes=bias_notes,
        corpus=corpus,
        fit_method="mft_to_em_via_agreement_weighted_means",
        mft_to_em_mapping=mapping if mapping is not None else DEFAULT_MFT_TO_EM_DAG,
        weights=weights,
        priors=priors,
        coverage=coverage,
        fitted_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metadata={"n_situations_used": n_used, **(metadata or {})},
    )
