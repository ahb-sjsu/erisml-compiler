"""DEME V3 bridge — compiler IR to V3 modules to MoralTensorV3.

This module is the compiler-side replacement for the V2 EM-DAG when
producing the V3 tensor. The V2 EM-DAG continues to run for the
legacy `moral_vectors` / `timeline` surface (retired in Phase 4); this
bridge runs alongside it to produce `ir.moral_tensor_v3`.

Pipeline shape::

    CompilerIR
      |
      | (heuristic aggregation of raw EthicalFact list)
      v
    erisml.ethics.facts.EthicalFacts          (V2 dimension-grouped)
      |
      | (EthicalFactsV3.from_v2 with parties=stakeholder_ids)
      v
    erisml.ethics.facts_v3.EthicalFactsV3     (per-party tracking)
      |
      | (judge_distributed across registered V3 modules)
      v
    list[EthicalJudgementV3]
      |
      | (weighted aggregation by default_weight)
      v
    erisml.ethics.moral_tensor.MoralTensor    (numpy-backed)
      |
      | (MoralTensorV3.from_deme_tensor)
      v
    MoralTensorV3                              (JSON-serialisable)

The bridge is *honest about being heuristic* at the IR → V2 facts
boundary. The compiler IR's `EthicalFact` is a list of typed claim
instances; the V2 `EthicalFacts` is an aggregate per-dimension
structure. The mapping is a documented many-to-one — see
`_FACT_KIND_TO_V2_FIELD`.

When erisml-lib is not installed, all entry points raise a clear
ImportError pointing at the migration doc. Phase 2's fallback path
remains active in the orchestrator for that case.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from erisml_compiler.ir.schemas import CompilerIR, EthicalFact, Stakeholder
from erisml_compiler.ir.v3 import MoralTensorV3

if TYPE_CHECKING:  # pragma: no cover
    from erisml.ethics.facts import EthicalFacts as V2EthicalFacts

log = logging.getLogger(__name__)


# ---------- severity & mapping tables --------------------------------------


_SEVERITY_MAGNITUDE: dict[str | None, float] = {
    "minor": 0.25,
    "moderate": 0.5,
    "grave": 0.75,
    "catastrophic": 1.0,
    None: 0.5,   # default when extractor didn't tag severity
}

# Maps the compiler's EthicalFactKind to the V2 EthicalFacts field(s) it
# should influence. Each entry is a list of (field_path, op) tuples
# where `op` is one of:
#   ("max", magnitude)    — set float field to max(current, magnitude)
#   ("flag", True)        — set boolean field to True
#   ("inc", magnitude)    — add magnitude to a float field
_FACT_KIND_TO_V2_FIELD: dict[str, list[tuple[str, str]]] = {
    "harm":              [("consequences.expected_harm", "max")],
    "non_maleficence":   [("consequences.expected_harm", "max")],
    "coercion":          [
        ("autonomy_and_agency.coercion_or_undue_influence", "flag"),
        ("autonomy_and_agency.has_meaningful_choice", "unflag"),
    ],
    "consent":           [
        ("rights_and_duties.has_valid_consent", "unflag_if_grave"),
        ("autonomy_and_agency.informed_consent", "flag_if_consent_present"),
    ],
    "legitimacy":        [
        ("procedural_and_legitimacy.legitimacy_score", "max"),
    ],
    "vulnerability":     [
        ("justice_and_fairness.exploits_vulnerable_population", "flag"),
        ("societal_and_environmental.burden_on_vulnerable_groups", "max"),
    ],
    "uncertainty":       [("epistemic_status.uncertainty_level", "max")],
    "externality":       [("societal_and_environmental.long_term_societal_risk", "max")],
    "justice":           [("justice_and_fairness.distributive_pattern", "set_str")],
    "care":              [("virtue_and_care.expresses_compassion", "flag")],
    "truth":             [("virtue_and_care.respects_person_as_end", "flag")],
    "deception":         [("virtue_and_care.betrays_trust", "flag")],
    "role_duty":         [("rights_and_duties.role_duty_conflict", "flag")],
    "reciprocity":       [],  # no clean V2 mapping; stored in metadata
}


# ---------- IR -> V2 EthicalFacts -----------------------------------------


def _ensure_erisml_lib() -> None:
    try:
        import erisml.ethics.facts  # noqa: F401, PLC0415
        import erisml.ethics.facts_v3  # noqa: F401, PLC0415
        import erisml.ethics.modules.base_v3  # noqa: F401, PLC0415
        import erisml.ethics.modules.tier0.geneva_em_v3  # noqa: F401, PLC0415
        import erisml.ethics.modules.triage_em_v3  # noqa: F401, PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "DEME V3 bridge requires erisml-lib. "
            "See docs/migration/deme_v3_alignment.md for installation."
        ) from e


def ir_to_v2_facts(ir: CompilerIR) -> "V2EthicalFacts":
    """Aggregate compiler IR's raw EthicalFact list into a V2
    `EthicalFacts` instance.

    This is the heuristic boundary: the compiler IR records each
    ethical claim as a separate `EthicalFact` row, while DEME V2 wants
    an aggregated per-dimension structure. We group by claim `kind`
    and route into V2 fields via `_FACT_KIND_TO_V2_FIELD`.
    """
    _ensure_erisml_lib()
    from erisml.ethics.facts import (  # noqa: PLC0415
        AutonomyAndAgency,
        Consequences,
        EpistemicStatus,
        EthicalFacts,
        JusticeAndFairness,
        ProceduralAndLegitimacy,
        RightsAndDuties,
        SocietalAndEnvironmental,
        VirtueAndCare,
    )

    consequences = Consequences(
        affected_count=max(1, len(ir.stakeholders)),
    )
    rights_and_duties = RightsAndDuties()
    justice_and_fairness = JusticeAndFairness()
    autonomy_and_agency = AutonomyAndAgency()
    societal_and_environmental = SocietalAndEnvironmental()
    virtue_and_care = VirtueAndCare()
    procedural_and_legitimacy = ProceduralAndLegitimacy()
    epistemic_status = EpistemicStatus()

    nodes: dict[str, Any] = {
        "consequences": consequences,
        "rights_and_duties": rights_and_duties,
        "justice_and_fairness": justice_and_fairness,
        "autonomy_and_agency": autonomy_and_agency,
        "societal_and_environmental": societal_and_environmental,
        "virtue_and_care": virtue_and_care,
        "procedural_and_legitimacy": procedural_and_legitimacy,
        "epistemic_status": epistemic_status,
    }

    for fact in ir.ethical_facts:
        magnitude = _SEVERITY_MAGNITUDE.get(fact.severity, 0.5) * fact.confidence
        for field_path, op in _FACT_KIND_TO_V2_FIELD.get(fact.kind, []):
            _apply_op(nodes, field_path, op, magnitude, fact)

    return EthicalFacts(
        option_id=ir.document.doc_id if ir.document else "scenario",
        consequences=consequences,
        rights_and_duties=rights_and_duties,
        justice_and_fairness=justice_and_fairness,
        autonomy_and_agency=autonomy_and_agency,
        privacy_and_data=None,
        societal_and_environmental=societal_and_environmental,
        virtue_and_care=virtue_and_care,
        procedural_and_legitimacy=procedural_and_legitimacy,
        epistemic_status=epistemic_status,
    )


def _apply_op(
    nodes: dict[str, Any],
    field_path: str,
    op: str,
    magnitude: float,
    fact: EthicalFact,
) -> None:
    """Apply one mapping op to the V2 facts nodes."""
    node_name, _, field_name = field_path.partition(".")
    node = nodes.get(node_name)
    if node is None or not hasattr(node, field_name):
        return

    if op == "max":
        current = float(getattr(node, field_name, 0.0) or 0.0)
        setattr(node, field_name, max(current, magnitude))
    elif op == "inc":
        current = float(getattr(node, field_name, 0.0) or 0.0)
        setattr(node, field_name, min(1.0, current + magnitude))
    elif op == "flag":
        setattr(node, field_name, True)
    elif op == "unflag":
        setattr(node, field_name, False)
    elif op == "unflag_if_grave":
        if fact.severity in ("grave", "catastrophic"):
            setattr(node, field_name, False)
    elif op == "flag_if_consent_present":
        # If a "consent" fact is being recorded as present (not absent),
        # mark informed_consent True; if absent (severity grave or worse),
        # leave it false. Heuristic: rely on description.
        if "obtained" in fact.description.lower() or "given" in fact.description.lower():
            setattr(node, field_name, True)
    elif op == "set_str":
        # For distributive_pattern: take the fact's severity as the
        # pattern label (heuristic).
        if fact.severity:
            setattr(node, field_name, fact.severity)


# ---------- V3 module invocation ------------------------------------------


def _instantiate_v3_modules() -> list:
    """Instantiate the default V3 module set.

    Currently: GenevaEMV3 (tier 0 constitutional), TriageEMV3 (tier 2
    rights/fairness). Phase 4 will widen this to the full tier
    discovered via EMRegistry.list_by_tier(...).
    """
    _ensure_erisml_lib()
    from erisml.ethics.modules.tier0.geneva_em_v3 import GenevaEMV3  # noqa: PLC0415
    from erisml.ethics.modules.triage_em_v3 import TriageEMV3  # noqa: PLC0415

    return [
        GenevaEMV3(em_name="geneva_v3", stakeholder="universal", em_tier=0),
        TriageEMV3(em_name="triage_v3", stakeholder="distributional", em_tier=2),
    ]


# ---------- Top-level entry point -----------------------------------------


def compile_to_v3_tensor(
    ir: CompilerIR,
    *,
    rank: int = 2,
    aggregation: str = "weighted_mean",
) -> MoralTensorV3:
    """Run the V3 bridge and return a serialisable MoralTensorV3.

    Args:
        ir: the compiler IR (after the V2 pipeline has populated facts).
        rank: 1 (collapse to global vector) or 2 (per-stakeholder).
              Rank-2 is the natural output; rank-1 is a `mean` collapse
              over the n axis. Higher ranks land in Phase 5.
        aggregation: how to combine module judgements. Currently
              "weighted_mean" — sum of `weight * module_tensor.values`
              divided by sum of weights, applied element-wise.

    Raises:
        ImportError: if erisml-lib is not installed.
        NotImplementedError: for ranks > 2.
    """
    if rank not in (1, 2):
        raise NotImplementedError(
            f"V3 bridge supports rank 1 and 2; rank {rank} arrives in Phase 5."
        )

    _ensure_erisml_lib()
    import numpy as np  # noqa: PLC0415
    from erisml.ethics.facts_v3 import EthicalFactsV3  # noqa: PLC0415

    party_ids = [s.id for s in ir.stakeholders] if ir.stakeholders else ["aggregate"]

    v2_facts = ir_to_v2_facts(ir)
    v3_facts = EthicalFactsV3.from_v2(v2_facts, parties=party_ids)

    modules = _instantiate_v3_modules()
    weighted_values = np.zeros((9, len(party_ids)), dtype=float)
    weight_total = 0.0
    veto_flags: list[str] = []
    veto_locations: list[tuple[int, ...]] = []
    reason_codes: list[str] = []

    for module in modules:
        try:
            judgement = module.judge_distributed(v3_facts)
        except Exception as e:
            log.warning(
                "V3 bridge: module %s raised %s; skipping",
                getattr(module, "em_name", module.__class__.__name__), e,
            )
            continue
        # The module's moral_tensor is rank-2 (9, n). Pull values.
        mt = judgement.moral_tensor
        try:
            arr = np.asarray(mt._data, dtype=float)
        except Exception:
            continue
        if arr.shape != (9, len(party_ids)):
            log.debug(
                "V3 bridge: module %s returned shape %s; reshaping skipped",
                module.em_name, arr.shape,
            )
            continue
        weight = float(getattr(module, "default_weight", 1.0))
        weighted_values += weight * arr
        weight_total += weight
        veto_flags.extend(getattr(mt, "veto_flags", []))
        veto_locations.extend(
            tuple(int(x) for x in v) for v in getattr(mt, "veto_locations", [])
        )
        reason_codes.extend(getattr(mt, "reason_codes", []))

    if weight_total > 0:
        weighted_values /= weight_total

    rank2 = MoralTensorV3(
        rank=2,
        shape=(9, len(party_ids)),
        axis_names=("k", "n"),
        axis_labels={
            "k": list(_canonical_k_labels()),
            "n": list(party_ids),
        },
        values=weighted_values.tolist(),
        veto_flags=veto_flags,
        veto_locations=veto_locations,
        reason_codes=reason_codes,
        metadata={
            "build_strategy": "phase3_v3_bridge",
            "modules_invoked": [m.em_name for m in modules],
            "aggregation": aggregation,
            "n_parties": len(party_ids),
        },
    )

    if rank == 1:
        # Mean collapse over the n axis -> rank-1 (k,) tensor.
        mean_vals = weighted_values.mean(axis=1).tolist()
        return MoralTensorV3(
            rank=1,
            shape=(9,),
            axis_names=("k",),
            axis_labels={"k": list(_canonical_k_labels())},
            values=mean_vals,
            veto_flags=veto_flags,
            metadata={**rank2.metadata, "collapsed_from_rank2": True},
        )
    return rank2


def _canonical_k_labels() -> tuple[str, ...]:
    """Look up the canonical 9-dimension order. Defer to ir.v3 to avoid drift."""
    from erisml_compiler.ir.v3 import MORAL_DIMENSIONS_V3  # noqa: PLC0415

    return MORAL_DIMENSIONS_V3
