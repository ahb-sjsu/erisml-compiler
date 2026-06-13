"""SMT-based universalizability test using Z3.

The v1 implementation in `universalizability.py` is a hand-curated
lookup table — principled but not derived from a world model. This
module replaces the lookup with a real SMT formulation:

  - Universe of institutional facts encoded as Z3 Bool variables
  - Each action_kind's universalization is a Z3 constraint that
    destroys/preserves specific institutions
  - The action's presupposition is a Z3 constraint that the
    institution must hold for the act to be possible
  - We solve: is (universalized maxim) ∧ (presupposition) SAT?
    - UNSAT → Contradiction in Conception
    - SAT but agent's own ends are violated → Contradiction in Will
    - SAT and ends preserved → no contradiction

The output keeps the same `InstitutionDependency` shape as the
v1 KB lookup, but adds a `model_facts` field recording the actual
satisfying (or unsatisfiable) assignment so reviewers can audit
*which* constraint produced the verdict.

## Z3 fact universe

Each Z3 Bool variable corresponds to a global institutional fact:
  - `truth_telling_default`        — most agents tell the truth
  - `promises_create_trust`        — promise-keeping is the norm
  - `property_rules_followed`      — fair-dealing is the norm
  - `bodily_integrity_respected`   — agents don't routinely harm others
  - `autonomy_respected`           — agents aren't routinely coerced
  - `help_available_when_needed`   — mutual aid is the norm
  - `authority_legitimacy_grounded`— authority comes from procedures
  - `confidentiality_norms`        — confidants keep confidences
  - `non_consenting_party_standing`— non-consenting parties have
                                     standing not to bear imposed cost

## Agent ends (CIW model)

The agent is modeled as a rational being whose own existence
presupposes a baseline of:
  - `bodily_integrity_respected` (they want not to be harmed)
  - `autonomy_respected`         (they want self-determination)
  - `help_available_when_needed` (they want help when in need)

A maxim that universalizes-away any of these is CIW.

## Limitations

  - The fact universe is hand-curated (it has to be — moral
    semantics doesn't fall out of a solver). What the solver
    contributes is *consistency checking* over those facts:
    we can no longer accidentally claim "deceive presupposes
    truth-telling AND universal deception preserves truth-telling"
    — the solver would surface the contradiction in our model
    rather than letting it slip.
  - Z3 is overkill for v1 (each query reduces to a single
    SAT check on a tiny formula). The real win is that this
    framework scales to richer agent-models, multi-agent
    coordination, and conditional maxims later — exactly the
    things a lookup table can't.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from erisml_compiler.delta.universalizability import (
    DEFAULT_INSTITUTION_DEPENDENCIES,
    InstitutionDependency,
)

SmtAvailable = bool


_Z3_IMPORT_ERROR: str | None = None


def _try_import_z3():
    global _Z3_IMPORT_ERROR
    try:
        import z3

        return z3
    except ImportError as e:
        _Z3_IMPORT_ERROR = str(e)
        return None


def is_smt_available() -> SmtAvailable:
    return _try_import_z3() is not None


# ---------------------------------------------------------------------- model
#
# Each entry encodes:
#   destroys_under_universalization: which institutional Z3 vars are
#     forced to False when the maxim is universalized
#   presupposes: which institutional Z3 vars must be True for the
#     action to be possible
#   agent_end_violated: True iff universalizing the maxim destroys
#     a state the agent rationally needs
#
# The Z3 solver does the consistency check.


@dataclass(frozen=True)
class _ZModel:
    destroys: tuple[str, ...]
    presupposes: tuple[str, ...]
    agent_end_violated: bool
    contested_reading: str | None = None


_AGENT_ENDS: tuple[str, ...] = (
    "bodily_integrity_respected",
    "autonomy_respected",
    "help_available_when_needed",
    # The agent is themselves a potential non-consenting party in
    # future situations; their rationality presupposes the standing
    # not to bear cost they didn't agree to.
    "non_consenting_party_standing",
)


_Z3_ACTION_MODEL: dict[str, _ZModel] = {
    "deceive": _ZModel(
        destroys=("truth_telling_default",),
        presupposes=("truth_telling_default",),
        agent_end_violated=False,
    ),
    "break_commitment": _ZModel(
        destroys=("promises_create_trust",),
        presupposes=("promises_create_trust",),
        agent_end_violated=False,
    ),
    "make_or_keep_commitment": _ZModel(
        destroys=(),
        presupposes=("promises_create_trust",),
        agent_end_violated=False,
    ),
    "cheat": _ZModel(
        destroys=("property_rules_followed",),
        presupposes=("property_rules_followed",),
        agent_end_violated=False,
    ),
    "inflict_harm": _ZModel(
        destroys=("bodily_integrity_respected",),
        presupposes=(),
        agent_end_violated=True,  # destroys an agent end → CIW
    ),
    "coerce": _ZModel(
        destroys=("autonomy_respected",),
        presupposes=(),
        agent_end_violated=True,
    ),
    "coerce_or_be_coerced": _ZModel(
        destroys=("autonomy_respected",),
        presupposes=(),
        agent_end_violated=True,
    ),
    "impose_externality": _ZModel(
        destroys=("non_consenting_party_standing",),
        presupposes=(),
        agent_end_violated=True,  # agent doesn't want costs imposed on them
    ),
    "use_as_means": _ZModel(
        destroys=("autonomy_respected",),
        presupposes=(),
        agent_end_violated=True,
    ),
    "refuse": _ZModel(
        destroys=("help_available_when_needed",),
        presupposes=(),
        agent_end_violated=True,  # Kant's classic CIW
    ),
    "protect": _ZModel(
        destroys=(),
        presupposes=(),
        agent_end_violated=False,
    ),
    "help": _ZModel(
        destroys=(),
        presupposes=(),
        agent_end_violated=False,
    ),
    "disclose": _ZModel(
        destroys=(),
        presupposes=(),
        agent_end_violated=False,
        contested_reading=(
            "Contested: 'disclose' for confidentiality (medical/legal/religious) "
            "presupposes confidentiality_norms which universal disclosure would "
            "destroy → reads as CIC on that mapping. Default (whistleblower) "
            "reading: no contradiction."
        ),
    ),
    "act_under_norm": _ZModel(destroys=(), presupposes=(), agent_end_violated=False),
    "act_under_authority": _ZModel(
        destroys=(), presupposes=("authority_legitimacy_grounded",), agent_end_violated=False
    ),
}


@dataclass(frozen=True)
class SmtUniversalizabilityResult:
    """Adds an SMT proof trace to the v1 InstitutionDependency shape."""

    base: InstitutionDependency
    model_facts: dict[str, bool]
    """The Z3 model (which institutional facts the solver assigned).
    Empty dict when the problem was UNSAT."""
    used_z3: bool
    """True iff Z3 actually ran. False when Z3 isn't installed or
    when the action_kind wasn't in the Z3 model."""
    fallback_reason: str | None = None


def test_universalizability_smt(
    action_kind: str | None,
    *,
    fact_model: dict[str, _ZModel] | None = None,
) -> SmtUniversalizabilityResult:
    """SMT-based universalizability test. Returns a
    SmtUniversalizabilityResult containing the v1-compatible
    InstitutionDependency plus the SMT model trace.

    Falls back to the v1 KB lookup when Z3 isn't available or when
    the action_kind isn't in the Z3 model — the returned object's
    `used_z3=False` flags the fallback.
    """
    fact_model = fact_model if fact_model is not None else _Z3_ACTION_MODEL

    if action_kind is None or action_kind not in fact_model:
        base = DEFAULT_INSTITUTION_DEPENDENCIES.get(
            action_kind or "<unknown>"
        ) or InstitutionDependency(
            action_kind=action_kind or "<unknown>",
            presupposes=(),
            contradiction_type="undetermined",
            passes=True,
            justification="action_kind not in Z3 model; falling back to KB lookup.",
        )
        return SmtUniversalizabilityResult(
            base=base,
            model_facts={},
            used_z3=False,
            fallback_reason="action_kind not in Z3 model",
        )

    z3 = _try_import_z3()
    if z3 is None:
        # Fall back to the v1 KB lookup.
        from erisml_compiler.delta.universalizability import test_universalizability

        return SmtUniversalizabilityResult(
            base=test_universalizability(action_kind),
            model_facts={},
            used_z3=False,
            fallback_reason="z3-solver not installed",
        )

    spec = fact_model[action_kind]
    return _solve(action_kind, spec, z3)


def _solve(action_kind: str, spec: _ZModel, z3) -> SmtUniversalizabilityResult:
    """Build the Z3 problem and check satisfiability."""
    # All institutional facts that might appear in this query.
    fact_names = list(
        set(spec.destroys)
        | set(spec.presupposes)
        | set(_AGENT_ENDS)
        | {
            "truth_telling_default",
            "promises_create_trust",
            "property_rules_followed",
            "bodily_integrity_respected",
            "autonomy_respected",
            "help_available_when_needed",
            "authority_legitimacy_grounded",
            "confidentiality_norms",
            "non_consenting_party_standing",
        }
    )
    fact_vars = {name: z3.Bool(name) for name in fact_names}

    solver = z3.Solver()

    # Universalization: facts the maxim destroys become False.
    for f in spec.destroys:
        solver.add(z3.Not(fact_vars[f]))

    # Presupposition: facts the maxim presupposes must be True.
    for f in spec.presupposes:
        solver.add(fact_vars[f])

    # CIC check: is (destroys ∧ presupposes) satisfiable?
    cic_result = solver.check()
    cic_failed = cic_result == z3.unsat

    # CIW check: with the universalization in force, are agent ends
    # still satisfiable?
    ciw_failed = False
    agent_solver = None
    if not cic_failed:
        # Add agent ends as constraints; check again.
        agent_solver = z3.Solver()
        for f in spec.destroys:
            agent_solver.add(z3.Not(fact_vars[f]))
        for f in spec.presupposes:
            agent_solver.add(fact_vars[f])
        for end in _AGENT_ENDS:
            agent_solver.add(fact_vars[end])
        ciw_result = agent_solver.check()
        ciw_failed = ciw_result == z3.unsat

    # Extract the model (if SAT). Prefer the agent-ends-constrained
    # model so the audit trail shows the agent ends really are
    # satisfiable in the post-universalization world.
    model_facts: dict[str, bool] = {}
    if cic_failed:
        contradiction_type: Literal[
            "contradiction_in_conception", "contradiction_in_will", "no_contradiction"
        ] = "contradiction_in_conception"
        passes = False
    elif ciw_failed:
        contradiction_type = "contradiction_in_will"
        passes = False
    else:
        contradiction_type = "no_contradiction"
        passes = True
        m = (agent_solver if agent_solver is not None else solver).model()
        for name, var in fact_vars.items():
            v = m.eval(var, model_completion=True)
            model_facts[name] = bool(v)

    # Justification anchored in the model.
    if contradiction_type == "contradiction_in_conception":
        justification = (
            f"Universalising '{action_kind}' destroys "
            f"{{{', '.join(spec.destroys)}}}, which the act itself "
            f"presupposes. Z3 finds no model satisfying both — "
            f"strict Contradiction in Conception."
        )
    elif contradiction_type == "contradiction_in_will":
        justification = (
            f"Universalising '{action_kind}' is internally coherent "
            f"but destroys agent ends "
            f"({', '.join(end for end in _AGENT_ENDS if end in spec.destroys)})"
            f" — the agent cannot rationally will it. CIW."
        )
    else:
        justification = (
            f"Universalising '{action_kind}' destroys nothing the act "
            f"presupposes and nothing the agent rationally needs. "
            f"No contradiction."
        )

    base = InstitutionDependency(
        action_kind=action_kind,
        presupposes=spec.presupposes,
        contradiction_type=contradiction_type,
        passes=passes,
        justification=justification,
        contested_reading=spec.contested_reading,
    )
    return SmtUniversalizabilityResult(base=base, model_facts=model_facts, used_z3=True)
