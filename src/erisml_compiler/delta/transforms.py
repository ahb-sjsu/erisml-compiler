"""Labelled, parameterised semantics-preserving text transformations.

Per release-planning-02-rho-estimation.md, this is the registry the
equivariance check + ρ estimator both pull from. Each `Transform`
carries its family (surface vs paraphrase vs role-swap vs ...) and
an `expected_rho_class` declaring what shape the layer-wise map ρ_ℓ(g)
should take if the probe is well-calibrated.

Backward-compatible with the legacy `Rewrite` API in
`delta.equivariance` — a Transform exposes `.name` and `.fn` the same
way a Rewrite does.

This module ships only the core types + a small set of surface-family
transforms. Paraphrase and role-swap transforms require external
infrastructure (LLM adapters, IR-equivalence gates) and are documented
as future work in release-planning-02.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Literal


class TransformFamily(str, enum.Enum):
    SURFACE = "surface"
    """Whitespace, case, punctuation. ρ_ℓ(g) should be identity."""
    PARAPHRASE = "paraphrase"
    """Lexical/syntactic rewording, meaning preserved. ρ expected
    orthogonal: a rotation/reflection of activation space, no
    magnitude change."""
    ROLE_SWAP = "role_swap"
    """Stakeholder label swap (alice <-> bob). ρ expected to be a
    permutation of activation subspaces."""
    UNIT_CHANGE = "unit_change"
    """Quantitative re-expression (10 miles -> 16 km). ρ expected
    near-identity for the moral subspace."""
    ORDER_PERM = "order_perm"
    """Reordering of independent clauses. ρ expected near-identity."""


RhoClass = Literal["identity", "orthogonal", "linear", "permutation"]


@dataclass(frozen=True)
class ValidationResult:
    """Output of a Transform's semantic-preservation gate."""

    valid: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Transform:
    """A labelled, parameterised semantics-preserving rewrite.

    `fn(text, params)` returns the rewritten text. `params` is an
    arbitrary dict the registry may pass in (e.g. seed for paraphrase,
    swap-table for role-swap); `None` is fine for parameterless ones.

    `expected_rho_class` declares what family ρ_ℓ(g) should fall in if
    the probe is well-calibrated:

      - 'identity'    — surface rewrites; ρ should be exactly I
      - 'orthogonal'  — paraphrase; ρ should be a rotation/reflection
      - 'permutation' — role-swap; ρ should permute subspaces
      - 'linear'      — anything else; ρ is unconstrained linear

    `validation_hook` is the semantic-preservation gate: when set, it
    receives (orig, rewritten) and returns whether the transform truly
    preserved moral content. Surface-family transforms can leave this
    None (whitespace normalisation cannot change meaning by construction).
    Non-surface families REQUIRE a hook; see open question A in the
    release-planning-02 design note.
    """

    name: str
    family: TransformFamily
    fn: Callable[..., str]
    expected_rho_class: RhoClass = "identity"
    validation_hook: Callable[[str, str], ValidationResult] | None = None
    cost: float = 1.0
    """Relative wall-clock cost vs the cheapest surface transform.
    Used by the equivariance scheduler to prefer cheap transforms
    when the calibration budget is tight."""

    def apply(self, text: str, params: dict | None = None) -> str:
        if params is None:
            return self.fn(text)
        return self.fn(text, params)


class TransformRegistry:
    """Named-transform store. Use the module-level `default_registry()`
    in production; instantiate fresh in tests to avoid global state.
    """

    def __init__(self) -> None:
        self._transforms: dict[str, Transform] = {}

    def register(self, transform: Transform) -> None:
        if transform.name in self._transforms:
            raise ValueError(f"Transform {transform.name!r} already registered")
        self._transforms[transform.name] = transform

    def get(self, name: str) -> Transform:
        if name not in self._transforms:
            raise KeyError(f"No transform named {name!r}")
        return self._transforms[name]

    def filter(self, *, family: TransformFamily | None = None) -> list[Transform]:
        if family is None:
            return list(self._transforms.values())
        return [t for t in self._transforms.values() if t.family == family]

    def names(self) -> list[str]:
        return sorted(self._transforms)

    def __contains__(self, name: str) -> bool:
        return name in self._transforms

    def __len__(self) -> int:
        return len(self._transforms)


# ----- bundled surface-family transforms (ρ expected: identity) -----


def _normalise_whitespace(s: str) -> str:
    return " ".join(s.split())


def _lowercase(s: str) -> str:
    return s.lower()


def _trim_trailing_period(s: str) -> str:
    return s.rstrip(".").strip()


def _double_quotes(s: str) -> str:
    """Replace straight single quotes with double quotes (typographic
    rewrite). Meaning-preserving."""
    return s.replace("'", '"')


SURFACE_TRANSFORMS: tuple[Transform, ...] = (
    Transform(
        name="normalise_whitespace",
        family=TransformFamily.SURFACE,
        fn=_normalise_whitespace,
        expected_rho_class="identity",
    ),
    Transform(
        name="lowercase",
        family=TransformFamily.SURFACE,
        fn=_lowercase,
        expected_rho_class="identity",
    ),
    Transform(
        name="trim_trailing_period",
        family=TransformFamily.SURFACE,
        fn=_trim_trailing_period,
        expected_rho_class="identity",
    ),
    Transform(
        name="straight_to_double_quotes",
        family=TransformFamily.SURFACE,
        fn=_double_quotes,
        expected_rho_class="identity",
    ),
)


_DEFAULT_REGISTRY: TransformRegistry | None = None


def default_registry() -> TransformRegistry:
    """Return the module-level singleton registry, populated with the
    bundled surface transforms on first access."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        reg = TransformRegistry()
        for t in SURFACE_TRANSFORMS:
            reg.register(t)
        _DEFAULT_REGISTRY = reg
    return _DEFAULT_REGISTRY
