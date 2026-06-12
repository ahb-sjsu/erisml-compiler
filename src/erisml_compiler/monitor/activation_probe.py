"""Per-layer activation probe.

Reuses the Phase-3 `ProbeBackbone.head` shape (a small MLP from a fixed
embedding dim to 10 moral-dimension logits) but trains *per layer* of the
target model rather than per LaBSE embedding. Same probe architecture,
different feature source.

The activation probe is *not* the text probe: even when both target the
same 10 moral dimensions, they answer different questions:

  Text probe          → "What is the moral content of the model's output?"
  Activation probe    → "What is the moral state of the model's internals?"

The delta lens (Track B) compares the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, DimensionScore, MoralVector
from erisml_compiler.monitor.base import LayerActivation


@dataclass
class LayerProbeResult:
    layer_index: int
    layer_name: str
    logits: Any  # (num_classes,) torch tensor — the raw probe output
    moral_vector: MoralVector
    pooled_norm: float  # ||pooled||, a coarse activation-magnitude diagnostic
    provenance: Any = None  # CalibrationProvenance | None — see monitor/provenance.py


def _logits_to_moral_vector(logits, dimensions: Sequence[str] = MORAL_DIMENSIONS) -> MoralVector:
    """Map a (D,) logits vector to a MoralVector. Logits are assumed to be
    in [-inf, +inf]; we map through tanh -> [-1, 1] and treat |value| as
    confidence. Direction is sign-of-value with a small dead band."""
    import torch

    if logits.ndim != 1 or logits.numel() != len(dimensions):
        raise ValueError(
            f"Probe logits shape mismatch: got {tuple(logits.shape)}, "
            f"expected ({len(dimensions)},)"
        )

    values = torch.tanh(logits).tolist()
    fields: dict[str, DimensionScore] = {}
    for name, v in zip(dimensions, values):
        if v > 0.05:
            direction = "positive"
        elif v < -0.05:
            direction = "negative"
        else:
            direction = "neutral"
        fields[name] = DimensionScore(
            value=float(v),
            confidence=float(abs(v)),
            uncertainty=float(1.0 - abs(v)),
            direction=direction,
            source_spans=[],
            explanation=None,
        )
    return MoralVector(**fields)


class ActivationProbe:
    """A small per-layer head that maps pooled (D,) hidden states to a
    MoralVector. One probe per (model_id, layer_index).

    Probes can be initialised:
      - randomly (for unit tests / smoke runs),
      - by loading Phase-3 calibration checkpoints (.pt files), or
      - by tying weights to an existing `ProbeBackbone.head` instance
        (for ablations where activation lens and text lens share the
        same final classifier).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = len(MORAL_DIMENSIONS),
        hidden: int = 256,
        n_layers: int = 2,
        device: str = "cpu",
        seed: int | None = None,
        provenance: Any = None,
    ):
        from erisml_compiler.calibration.probe_head import ProbeHead

        import torch

        from erisml_compiler.monitor.provenance import CalibrationProvenance

        self._torch = torch
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        self.head = ProbeHead(
            in_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden,
            n_layers=n_layers,
        ).to(device)
        self.head.eval()

        # Default provenance: explicitly uncalibrated rather than None,
        # so downstream consumers can branch on is_calibrated.
        self.provenance: CalibrationProvenance = provenance or CalibrationProvenance.uncalibrated(
            source="random", seed=seed
        )

    def load_head_state(self, state_dict: dict, *, provenance: Any = None) -> None:
        """Load a head's state_dict (e.g. from a Phase-3 checkpoint).

        Accepts both the bare head state_dict and the wrapped form
        produced by `ProbeBackbone.state_dict_for_checkpoint()` (which
        nests under "head"). Raises ValueError on shape mismatch.

        If a `provenance` is supplied (CalibrationProvenance), it
        replaces the probe's default "uncalibrated" marker. Callers
        loading from a .pt produced by `calibration.train.save_checkpoint`
        should pass the provenance saved inside that file.
        """
        if "head" in state_dict and not any(k.startswith("net.") for k in state_dict):
            sd = state_dict["head"]
        else:
            sd = state_dict
        self.head.load_state_dict(sd)
        if provenance is not None:
            self.provenance = provenance

    def probe_layer(self, layer_act: LayerActivation) -> LayerProbeResult:
        torch = self._torch
        pooled = layer_act.pooled
        if pooled.ndim != 1:
            raise ValueError(f"Expected pooled (D,), got {tuple(pooled.shape)}")
        if pooled.numel() != self.hidden_dim:
            raise ValueError(f"Pooled dim {pooled.numel()} != probe hidden_dim {self.hidden_dim}")

        with torch.no_grad():
            x = pooled.to(self.device).unsqueeze(0).float()
            logits = self.head(x).squeeze(0)

        mv = _logits_to_moral_vector(logits)
        norm = float(pooled.norm().item())
        return LayerProbeResult(
            layer_index=layer_act.layer_index,
            layer_name=layer_act.layer_name,
            logits=logits.cpu(),
            moral_vector=mv,
            pooled_norm=norm,
            provenance=self.provenance,
        )
