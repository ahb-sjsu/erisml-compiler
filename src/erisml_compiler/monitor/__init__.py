"""I-EIP Monitor: Internal / Activation / Delta lenses.

Phase 4 of the ErisML Compiler closes the loop between text output and the
internal moral state of a deployed model. Three lenses:

    Text lens        — the IR extracted from model output (Phases 1-3)
    Activation lens  — per-layer probe over hidden states (this package)
    Delta lens       — comparison + equivariance test (sibling `delta/` package)

The safety property: when the three lenses disagree, raise
`requires_human_review` instead of collapsing to a verdict.
"""

from __future__ import annotations

from erisml_compiler.monitor.base import (
    ActivationCapture,
    ActivationSource,
    LayerActivation,
)
from erisml_compiler.monitor.mock_source import MockActivationSource

__all__ = [
    "ActivationCapture",
    "ActivationSource",
    "LayerActivation",
    "MockActivationSource",
]

# HuggingFace + remote sources are imported lazily because they pull in torch
# and (optionally) paramiko. Importing them at package load would break
# environments without those extras installed.


def get_huggingface_source(*args, **kwargs):
    """Lazy constructor for HuggingFaceActivationSource."""
    from erisml_compiler.monitor.huggingface_source import HuggingFaceActivationSource

    return HuggingFaceActivationSource(*args, **kwargs)


def get_remote_atlas_source(*args, **kwargs):
    """Lazy constructor for RemoteAtlasActivationSource."""
    from erisml_compiler.monitor.remote_source import RemoteAtlasActivationSource

    return RemoteAtlasActivationSource(*args, **kwargs)
