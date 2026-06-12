"""Activation-source contract.

An `ActivationSource` is anything that, given an input text, returns the
hidden-state representations at a chosen set of transformer layers. The
contract is deliberately narrow so that mock sources (synthetic tensors),
local sources (HuggingFace transformers on the host), and remote sources
(paramiko-driven Atlas inference) all interoperate.

Why a layered representation matters: the I-EIP Monitor's activation lens
is not a single probe — it is a *family* of probes, one per chosen layer,
and the per-layer disagreement is itself a safety signal. A model whose
final-layer "moral state" looks fine but whose middle layers encode
something the head was trained to suppress is exactly the kind of thing
the delta lens needs visibility into.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class LayerActivation:
    """A single layer's hidden state for one input.

    `hidden` has shape (T, D) where T is the token count and D is the
    hidden dimension of the underlying model. We do not enforce a tensor
    type here so that pure-Python mock sources (lists of floats) and
    torch-backed real sources both fit.
    """

    layer_index: int
    layer_name: str
    hidden: Any  # tensor-like (T, D) — torch.Tensor in practice
    pooled: Any  # tensor-like (D,) — mean over tokens, used by the probe head


@dataclass(frozen=True)
class ActivationCapture:
    """Result of one forward pass: per-layer activations for one input.

    `text` is the original input. `layers` is keyed by layer_index in
    ascending order (typically 0..N-1 of the chosen layer subset, not
    necessarily contiguous in the underlying model).

    `source_name` identifies the producing source so downstream traces
    can record provenance (audit chain extension; spec §31.7).
    """

    text: str
    source_name: str
    model_id: str
    hidden_dim: int
    layers: Sequence[LayerActivation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_layer(self, layer_index: int) -> LayerActivation:
        for la in self.layers:
            if la.layer_index == layer_index:
                return la
        raise KeyError(f"layer_index {layer_index} not in capture")

    def layer_indices(self) -> list[int]:
        return [la.layer_index for la in self.layers]


class ActivationSource(ABC):
    """Abstract interface for anything that produces ActivationCaptures.

    Subclasses must declare a stable `name` and `model_id`. The probe
    head is keyed by `(model_id, layer_index)` so that probes trained
    against one source can be reused at inference time by another source
    pointing at the same underlying model.
    """

    name: str = "abstract"
    model_id: str = "abstract"
    hidden_dim: int = 0

    @abstractmethod
    def capture(self, text: str, *, layers: Sequence[int] | None = None) -> ActivationCapture:
        """Run a forward pass and return per-layer activations.

        `layers` selects which transformer layers to capture. If None,
        the source picks a sensible default (typically every 4th layer
        plus the final one — see concrete subclasses).
        """

    @abstractmethod
    def available_layers(self) -> list[int]:
        """All layer indices this source can produce activations for."""

    def close(self) -> None:
        """Release any resources held by the source (GPU memory, SSH, etc.).

        Default no-op; subclasses override if needed.
        """
