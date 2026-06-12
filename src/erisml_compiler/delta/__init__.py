"""Delta lens — comparison + equivariance test (Phase 4, Track B).

The text lens (Phases 1-3) tells us what moral content the model *says* it
is producing. The activation lens (Track A) tells us what moral state the
model *internally* exhibits. The delta lens is what makes them speak to
each other:

  - `compare_morals(text_mv, activation_mv)` returns a per-dimension delta
    and an overall divergence score in [0, 1].

  - `equivariance.check_equivariance` implements the BIP criterion
    h_ℓ(g·x) ≈ ρ_ℓ(g)·h_ℓ(x) for layer-wise probes under a group action
    g acting on inputs. For the I-EIP Monitor we instantiate g as a small
    set of input rewrites that should leave the moral semantics invariant
    (e.g. synonym swaps, paraphrases) and check that the per-layer probe
    output is invariant under them.

  - `failure_modes` enumerates the 5 named detectors that raise the
    `requires_human_review` flag.

Imports here are intentionally light. The torch-dependent paths (equivariance
test, internal-vector comparison) lazy-import inside their entry points.
"""

from __future__ import annotations

from erisml_compiler.delta.compare import (
    DeltaResult,
    DimensionDelta,
    compare_morals,
)
from erisml_compiler.delta.failure_modes import (
    FailureMode,
    FailureModeReport,
    detect_failure_modes,
)

__all__ = [
    "DeltaResult",
    "DimensionDelta",
    "FailureMode",
    "FailureModeReport",
    "compare_morals",
    "detect_failure_modes",
]
