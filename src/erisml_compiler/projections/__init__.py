"""Framework-pluralist two-layer IR.

The compiler now bifurcates explicitly:

  1. **`MoralSubstrate`** — descriptive layer (stakeholders, acts,
     commitments-already-standing, consent states, authorities and
     their procedural standing, maxims, repair states). Framework-
     neutral up to which categories of fact you choose to extract.

  2. **`Projection`** — framework-bound. Each projection reads the same
     substrate and emits a `ProjectionResult` whose `findings` are
     framework-specific (e.g. tensor + Gini for the consequentialist
     projection; categorical gates for the deontic projection).

The previous single-IR design pinned every output to a fixed dimensional
vocabulary and per-stakeholder distributional aggregates — a
pluralist-consequentialist-with-deontic-side-constraints stance, baked
in below the `--ethos-profile` layer. The two-layer split makes the
substrate's commitments smaller and labels what's framework-bound as
such. See `docs/plans/release-planning-06-framework-pluralist-architecture.md`
for the architectural argument.

Backward compat: the existing top-level CompilerIR fields (`moral_vectors`,
`moral_tensor_v3`, `deme_verdict`, `per_party_verdicts`, `fairness_metrics`)
remain populated from `projections["consequentialist"]` so older code
continues to work. New code should read from `ir.projections`.
"""
from erisml_compiler.projections.base import (
    GateFinding,
    Projection,
    ProjectionResult,
)
from erisml_compiler.projections.consequentialist import (
    ConsequentialistProjection,
)
from erisml_compiler.projections.deontic import DeonticProjection
from erisml_compiler.projections.substrate import (
    AuthorityLegitimacy,
    ConsentState,
    Maxim,
    MoralSubstrate,
    substrate_from_graph,
    substrate_from_ir,
)

__all__ = [
    "AuthorityLegitimacy",
    "ConsentState",
    "ConsequentialistProjection",
    "DeonticProjection",
    "GateFinding",
    "Maxim",
    "MoralSubstrate",
    "Projection",
    "ProjectionResult",
    "substrate_from_graph",
    "substrate_from_ir",
]
