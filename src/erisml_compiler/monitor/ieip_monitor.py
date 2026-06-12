"""IEIPMonitor — orchestrates the Activation lens.

Given an ActivationSource and a set of layer-keyed ActivationProbes,
produces a structured `MonitorTrace` for one input: per-layer moral
vectors, the activation-magnitude trajectory, and (when probes share
the same num_classes) a layerwise-aggregated MoralVector.

The aggregation is intentionally coarse — argmax over confidence per
dimension. The delta lens (Track B) does the actual comparison work.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Sequence

from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, DimensionScore, MoralVector
from erisml_compiler.monitor.activation_probe import (
    ActivationProbe,
    LayerProbeResult,
)
from erisml_compiler.monitor.base import ActivationCapture, ActivationSource


@dataclass
class MonitorTrace:
    text: str
    source_name: str
    model_id: str
    hidden_dim: int
    per_layer: list[LayerProbeResult]
    aggregated: MoralVector
    activation_norms: list[float]
    layer_indices: list[int]
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_name": self.source_name,
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
            "captured_at": self.captured_at,
            "layer_indices": self.layer_indices,
            "activation_norms": self.activation_norms,
            "per_layer": [
                {
                    "layer_index": r.layer_index,
                    "layer_name": r.layer_name,
                    "moral_vector": r.moral_vector.model_dump(),
                    "pooled_norm": r.pooled_norm,
                    "provenance": (r.provenance.model_dump() if r.provenance is not None else None),
                }
                for r in self.per_layer
            ],
            "aggregated": self.aggregated.model_dump(),
        }

    def trace_hash(self) -> str:
        """SHA-256 over the JSON-serialised trace. Used by the audit chain."""
        # Exclude captured_at so the hash is reproducible across runs of
        # the same source on the same input.
        d = self.to_dict()
        d.pop("captured_at", None)
        s = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _aggregate(per_layer: Sequence[LayerProbeResult]) -> MoralVector:
    """Per-dimension argmax-by-confidence across layers.

    This is the natural fold for an activation lens: a dimension is the
    layer's where the probe is *most certain*, not the layer's mean.
    The delta lens flags within-trace incoherence separately.
    """
    if not per_layer:
        raise ValueError("Cannot aggregate empty per-layer results.")

    fields: dict[str, DimensionScore] = {}
    for dim in MORAL_DIMENSIONS:
        best: DimensionScore | None = None
        for r in per_layer:
            ds = getattr(r.moral_vector, dim)
            if best is None or ds.confidence > best.confidence:
                best = ds
        assert best is not None
        fields[dim] = best
    return MoralVector(**fields)


class IEIPMonitor:
    """Activation-lens orchestrator.

    Args:
        source: an ActivationSource (Mock, HF, or Remote).
        probes: optional dict mapping layer_index -> ActivationProbe. If
            None, the monitor builds one fresh probe per layer using the
            source's `hidden_dim`. Reuse a pre-trained probe map by
            passing it in.
        seed: when constructing fresh probes, seed for reproducibility.
    """

    def __init__(
        self,
        source: ActivationSource,
        probes: dict[int, ActivationProbe] | None = None,
        seed: int | None = 0,
    ):
        self.source = source
        self._probes_in: dict[int, ActivationProbe] | None = probes
        self._seed = seed
        self._probes: dict[int, ActivationProbe] = {}

    def _ensure_probe(self, layer_index: int, hidden_dim: int) -> ActivationProbe:
        if layer_index in self._probes:
            return self._probes[layer_index]
        if self._probes_in is not None and layer_index in self._probes_in:
            p = self._probes_in[layer_index]
            if p.hidden_dim != hidden_dim:
                raise ValueError(
                    f"Probe at layer {layer_index} has hidden_dim={p.hidden_dim} "
                    f"but source produced hidden_dim={hidden_dim}"
                )
            self._probes[layer_index] = p
            return p
        # Fresh probe — useful for smoke tests / scenarios where we have
        # no calibrated checkpoints.
        seed = None if self._seed is None else self._seed + layer_index
        p = ActivationProbe(hidden_dim=hidden_dim, seed=seed)
        self._probes[layer_index] = p
        return p

    def monitor(self, text: str, *, layers: Sequence[int] | None = None) -> MonitorTrace:
        capture = self.source.capture(text, layers=layers)
        return self.monitor_capture(capture)

    def monitor_capture(self, capture: ActivationCapture) -> MonitorTrace:
        per_layer: list[LayerProbeResult] = []
        norms: list[float] = []
        for la in capture.layers:
            probe = self._ensure_probe(la.layer_index, capture.hidden_dim)
            result = probe.probe_layer(la)
            per_layer.append(result)
            norms.append(result.pooled_norm)

        aggregated = _aggregate(per_layer)
        return MonitorTrace(
            text=capture.text,
            source_name=capture.source_name,
            model_id=capture.model_id,
            hidden_dim=capture.hidden_dim,
            per_layer=per_layer,
            aggregated=aggregated,
            activation_norms=norms,
            layer_indices=[la.layer_index for la in capture.layers],
        )
