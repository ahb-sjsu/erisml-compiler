"""Mock activation source for tests and CI.

Produces deterministic, structured hidden states without needing a real
LLM. The "structure" is designed so that downstream probes have something
to bite into: each token's hidden state is a sum of (a) a layer-dependent
positional drift, (b) a text-dependent fixed encoding, and (c) Gaussian
noise seeded by the (text, layer) pair. This lets tests assert that the
probe responds to text changes and to layer changes, not just to noise.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from erisml_compiler.monitor.base import (
    ActivationCapture,
    ActivationSource,
    LayerActivation,
)


def _seed_from(text: str, layer: int) -> int:
    """Deterministic 32-bit seed from (text, layer)."""
    h = hashlib.sha256(f"{layer}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


class MockActivationSource(ActivationSource):
    """Deterministic synthetic source. No GPU, no network.

    Args:
        hidden_dim: D in the (T, D) hidden states.
        n_layers: total layer count to expose.
        n_tokens: how many tokens to fake per input (kept small for tests).
        model_id: opaque identifier — useful for probe keying.
    """

    name = "mock"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 8,
        n_tokens: int = 16,
        model_id: str = "mock-llm-base",
    ):
        # Lazy torch import so the package itself stays import-light.
        import torch

        self._torch = torch
        self.hidden_dim = hidden_dim
        self.model_id = model_id
        self._n_layers = n_layers
        self._n_tokens = n_tokens

    def available_layers(self) -> list[int]:
        return list(range(self._n_layers))

    def capture(self, text: str, *, layers: Sequence[int] | None = None) -> ActivationCapture:
        torch = self._torch
        if layers is None:
            layers = self.available_layers()

        # Text-dependent fixed encoding (D,)
        text_hash = hashlib.sha256(text.encode("utf-8")).digest()
        text_enc = torch.tensor(
            [b / 255.0 - 0.5 for b in text_hash[: self.hidden_dim]],
            dtype=torch.float32,
        )
        if text_enc.numel() < self.hidden_dim:
            # Pad with zeros when hidden_dim > 32 (sha256 is 32 bytes).
            pad = torch.zeros(self.hidden_dim - text_enc.numel())
            text_enc = torch.cat([text_enc, pad])

        layer_acts: list[LayerActivation] = []
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= self._n_layers:
                raise ValueError(f"layer {layer_idx} outside [0, {self._n_layers})")

            g = torch.Generator()
            g.manual_seed(_seed_from(text, layer_idx))
            noise = torch.randn(self._n_tokens, self.hidden_dim, generator=g) * 0.05
            # Layer-dependent positional drift in the first dim — gives
            # probes a deterministic gradient across layers.
            drift = torch.zeros(self._n_tokens, self.hidden_dim)
            drift[:, 0] = float(layer_idx) / max(1, self._n_layers - 1)
            hidden = text_enc.unsqueeze(0).expand(self._n_tokens, -1) + drift + noise
            pooled = hidden.mean(dim=0)

            layer_acts.append(
                LayerActivation(
                    layer_index=layer_idx,
                    layer_name=f"mock.layer.{layer_idx}",
                    hidden=hidden,
                    pooled=pooled,
                )
            )

        return ActivationCapture(
            text=text,
            source_name=self.name,
            model_id=self.model_id,
            hidden_dim=self.hidden_dim,
            layers=layer_acts,
            metadata={"deterministic": True},
        )
