"""Equivariance check — BIP criterion for activation probes.

The BIP framework (sqnd-probe v10.16.9) tests whether a layer-wise probe
`h_ℓ` respects a group action `g` acting on inputs:

    h_ℓ(g · x) ≈ ρ_ℓ(g) · h_ℓ(x)

For the I-EIP Monitor we instantiate this for `ρ_ℓ(g) = identity` — that
is, we check whether the probe is *invariant* under semantically
meaning-preserving rewrites of the input (synonyms, voice changes,
trivial reorderings). Failure of invariance is a safety signal: the
probe is picking up surface form rather than moral content.

This module supplies:

  - `Rewrite`: a labelled callable `text -> text` that should preserve
    moral semantics.
  - `EquivarianceReport`: per-layer per-rewrite cosine + L2 distances
    between pooled hidden states for the original and rewritten texts.
  - `check_equivariance(source, probe_map, text, rewrites)`: run all
    rewrites through a shared ActivationSource, probe each layer, and
    return a structured report. Includes a `failed_layers` list of
    layer indices where the probe's output drifted beyond a tolerance.

We use cosine distance on the *pooled hidden state* (not the probe
output) because the pooled state is the probe's input — drift here
quantifies the equivariance failure of the underlying model, and drift
in the probe output then quantifies how much the probe amplifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from erisml_compiler.monitor.activation_probe import ActivationProbe
from erisml_compiler.monitor.base import ActivationSource


@dataclass(frozen=True)
class Rewrite:
    """A semantics-preserving input transformation.

    `name` identifies the rewrite for reports. `fn` is the actual
    transformation. The compiler ships a small set of default rewrites
    (case change, whitespace normalisation, parenthetical-aside removal);
    the caller can supply richer rewrites (paraphrase via LLM, etc.).
    """

    name: str
    fn: Callable[[str], str]


@dataclass(frozen=True)
class LayerEquivarianceResult:
    layer_index: int
    rewrite_name: str
    pooled_cosine_sim: float  # in [-1, 1]; 1 = identical
    pooled_l2: float  # >= 0; 0 = identical
    probe_logits_cosine_sim: float
    probe_logits_l2: float
    passed: bool  # both cosine sims above threshold


@dataclass(frozen=True)
class EquivarianceReport:
    text: str
    rewrites_applied: list[str]
    layer_indices: list[int]
    per_layer_per_rewrite: list[LayerEquivarianceResult]
    failed_layers: list[int]
    config: dict = field(default_factory=dict)

    def by_layer(self, layer_index: int) -> list[LayerEquivarianceResult]:
        return [r for r in self.per_layer_per_rewrite if r.layer_index == layer_index]

    def by_rewrite(self, rewrite_name: str) -> list[LayerEquivarianceResult]:
        return [r for r in self.per_layer_per_rewrite if r.rewrite_name == rewrite_name]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "rewrites_applied": self.rewrites_applied,
            "layer_indices": self.layer_indices,
            "per_layer_per_rewrite": [
                {
                    "layer_index": r.layer_index,
                    "rewrite_name": r.rewrite_name,
                    "pooled_cosine_sim": r.pooled_cosine_sim,
                    "pooled_l2": r.pooled_l2,
                    "probe_logits_cosine_sim": r.probe_logits_cosine_sim,
                    "probe_logits_l2": r.probe_logits_l2,
                    "passed": r.passed,
                }
                for r in self.per_layer_per_rewrite
            ],
            "failed_layers": self.failed_layers,
            "config": self.config,
        }


# ---------- default rewrites: meaning-preserving by construction ----------


def _norm_whitespace(s: str) -> str:
    return " ".join(s.split())


def _lowercase(s: str) -> str:
    return s.lower()


def _trim_punct(s: str) -> str:
    """Strip a single trailing period — does not change moral semantics."""
    return s.rstrip(".").strip()


DEFAULT_REWRITES: list[Rewrite] = [
    Rewrite("normalise_whitespace", _norm_whitespace),
    Rewrite("lowercase", _lowercase),
    Rewrite("trim_trailing_period", _trim_punct),
]


def _cosine_sim(a, b) -> float:
    import torch

    a = a.float().flatten()
    b = b.float().flatten()
    na = a.norm()
    nb = b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 1.0 if float((a - b).norm()) == 0.0 else 0.0
    return float(torch.dot(a, b) / (na * nb))


def _l2(a, b) -> float:
    return float((a.float().flatten() - b.float().flatten()).norm())


def check_equivariance(
    source: ActivationSource,
    probes: Mapping[int, ActivationProbe],
    text: str,
    *,
    rewrites: Sequence[Rewrite] = DEFAULT_REWRITES,
    layers: Sequence[int] | None = None,
    pooled_cosine_threshold: float = 0.95,
    probe_cosine_threshold: float = 0.9,
) -> EquivarianceReport:
    """Run the BIP equivariance check across rewrites and layers.

    Args:
        source: the ActivationSource (Mock, HF, or Remote).
        probes: layer_index -> ActivationProbe map. Layers not in the
            map are skipped (no probe to compare).
        text: the original input.
        rewrites: meaning-preserving transformations to test under.
        layers: optional subset of layer indices to capture. If None,
            uses every layer with a probe.
        pooled_cosine_threshold / probe_cosine_threshold: a layer passes
            when both cosine sims exceed their respective thresholds.
    """
    layer_targets = list(layers) if layers is not None else sorted(probes.keys())

    base_capture = source.capture(text, layers=layer_targets)
    base_probes = {
        la.layer_index: (la.pooled, probes[la.layer_index].probe_layer(la).logits)
        for la in base_capture.layers
        if la.layer_index in probes
    }

    results: list[LayerEquivarianceResult] = []
    failed_layers_set: set[int] = set()

    for rw in rewrites:
        rewritten = rw.fn(text)
        if rewritten == text:
            # No-op rewrite — skip; otherwise we'd report a trivial pass.
            continue
        cap = source.capture(rewritten, layers=layer_targets)
        for la in cap.layers:
            if la.layer_index not in probes:
                continue
            base_pooled, base_logits = base_probes[la.layer_index]
            probe_result = probes[la.layer_index].probe_layer(la)
            new_logits = probe_result.logits

            pc = _cosine_sim(base_pooled, la.pooled)
            pl2 = _l2(base_pooled, la.pooled)
            lc = _cosine_sim(base_logits, new_logits)
            ll2 = _l2(base_logits, new_logits)

            passed = pc >= pooled_cosine_threshold and lc >= probe_cosine_threshold
            if not passed:
                failed_layers_set.add(la.layer_index)

            results.append(
                LayerEquivarianceResult(
                    layer_index=la.layer_index,
                    rewrite_name=rw.name,
                    pooled_cosine_sim=pc,
                    pooled_l2=pl2,
                    probe_logits_cosine_sim=lc,
                    probe_logits_l2=ll2,
                    passed=passed,
                )
            )

    return EquivarianceReport(
        text=text,
        rewrites_applied=[rw.name for rw in rewrites],
        layer_indices=sorted(probes.keys()),
        per_layer_per_rewrite=results,
        failed_layers=sorted(failed_layers_set),
        config={
            "pooled_cosine_threshold": pooled_cosine_threshold,
            "probe_cosine_threshold": probe_cosine_threshold,
        },
    )
