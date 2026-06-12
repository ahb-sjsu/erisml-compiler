"""Phase 4: build EthicalFactsV3 *directly* from compiler IR.

Phase 3's bridge took the path::

    IR -> V2 EthicalFacts (aggregated) -> EthicalFactsV3.from_v2(...)

The V2 aggregator collapses per-party information by construction, and
`from_v2` then "distributes aggregate values uniformly across all parties"
(per its own docstring). That uniform distribution is why Phase 3's
rank-2 columns were identical — every stakeholder saw the same input.

Phase 4 builds `EthicalFactsV3` directly, using `EthicalFact.subjects`
on each raw compiler fact to attribute its effect to specific parties.
The result is a `EthicalFactsV3` instance whose per-party records
`PartyConsequences`, `PartyRights`, `PartyAutonomy`, etc. genuinely
diverge across stakeholders when the underlying facts mention them
specifically.

Aggregate fields on each V3 dimension class (e.g., `expected_harm`)
are computed from the per-party records (mean, max, or sum depending
on the field's semantics).

Subjects missing entirely from `fact.subjects`:
  - if `fact.subjects` is empty, the fact applies *uniformly* to all
    parties (Phase 3 behaviour preserved as a fallback);
  - this keeps the bridge correct on extractors that don't yet tag
    subjects (mock_extractor for whistleblower, for instance).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from erisml_compiler.ir.schemas import CompilerIR, EthicalFact

if TYPE_CHECKING:  # pragma: no cover
    pass

# Severity → magnitude mapping (mirrors v3_bridge.py).
_SEVERITY_MAGNITUDE: dict[str | None, float] = {
    "minor": 0.25,
    "moderate": 0.5,
    "grave": 0.75,
    "catastrophic": 1.0,
    None: 0.5,
}


def ir_to_v3_facts(ir: CompilerIR) -> Any:
    """Build EthicalFactsV3 directly from compiler IR with per-party
    attribution from `EthicalFact.subjects`.

    Returns an `erisml.ethics.facts_v3.EthicalFactsV3` instance. Raises
    `ImportError` if erisml-lib is not available — the caller is
    expected to have checked.
    """
    from erisml.ethics.facts_v3 import (  # noqa: PLC0415
        ConsequencesV3,
        EpistemicStatusV3,
        EthicalFactsV3,
        JusticeAndFairnessV3,
        PartyAutonomy,
        PartyConsequences,
        PartyJustice,
        PartyPrivacy,
        PartyProcedural,
        PartyRights,
        PartySocietal,
        PartyVirtue,
        PrivacyAndDataGovernanceV3,
        ProceduralAndLegitimacyV3,
        RightsAndDutiesV3,
        SocietalAndEnvironmentalV3,
        VirtueAndCareV3,
        AutonomyAndAgencyV3,
    )

    parties = [s.id for s in ir.stakeholders] if ir.stakeholders else ["aggregate"]

    # Per-party scratch state — accumulate fact effects here.
    state: dict[str, dict[str, Any]] = {
        pid: {
            # consequences
            "expected_harm": 0.0,
            "expected_benefit": 0.0,
            "vulnerability_weight": 1.0,
            # rights
            "rights_violated": False,
            "consent_given": True,
            "duty_owed": False,
            # justice
            "relative_burden": 0.0,
            "relative_benefit": 0.0,
            "is_disadvantaged": False,
            # autonomy
            "has_meaningful_choice": True,
            "is_coerced": False,
            "can_withdraw": True,
            # privacy (rarely populated by compiler IR yet)
            "privacy_invasion_level": 0.0,
            "consent_for_data_use": True,
            "reidentification_risk": 0.0,
            # societal
            "environmental_burden": 0.0,
            "long_term_risk": 0.0,
            "benefit_to_future": 0.0,
            # virtue
            "receives_compassion": True,
            "trust_preserved": True,
            "treated_as_end": True,
            # procedural
            "was_consulted": False,
            "can_contest": True,
            "decision_explained": True,
        }
        for pid in parties
    }
    # Aggregate-only procedural state (not per-party).
    aggregate_legitimacy = 0.0

    # Decision-level (epistemic) accumulators.
    epistemic_uncertainty = 0.0

    for fact in ir.ethical_facts:
        # Decide which parties this fact applies to.
        subjects = list(fact.subjects) or list(parties)
        # Filter to known parties only (skip unknown subjects gracefully).
        subjects = [s for s in subjects if s in state] or list(parties)
        magnitude = _SEVERITY_MAGNITUDE.get(fact.severity, 0.5) * fact.confidence

        for pid in subjects:
            _apply_fact_to_party_state(state[pid], fact, magnitude)

        if fact.kind == "uncertainty":
            epistemic_uncertainty = max(epistemic_uncertainty, magnitude)
        if fact.kind == "legitimacy":
            aggregate_legitimacy = max(aggregate_legitimacy, magnitude)

    # Build the per-party tuples in stakeholder order.
    per_consequences = tuple(
        PartyConsequences(
            party_id=pid,
            expected_benefit=_clamp01(state[pid]["expected_benefit"]),
            expected_harm=_clamp01(state[pid]["expected_harm"]),
            vulnerability_weight=state[pid]["vulnerability_weight"],
        )
        for pid in parties
    )
    per_rights = tuple(
        PartyRights(
            party_id=pid,
            rights_violated=state[pid]["rights_violated"],
            consent_given=state[pid]["consent_given"],
            duty_owed=state[pid]["duty_owed"],
        )
        for pid in parties
    )
    per_justice = tuple(
        PartyJustice(
            party_id=pid,
            relative_burden=_clamp01(state[pid]["relative_burden"]),
            relative_benefit=_clamp01(state[pid]["relative_benefit"]),
            is_disadvantaged=state[pid]["is_disadvantaged"],
        )
        for pid in parties
    )
    per_autonomy = tuple(
        PartyAutonomy(
            party_id=pid,
            has_meaningful_choice=state[pid]["has_meaningful_choice"],
            is_coerced=state[pid]["is_coerced"],
            can_withdraw=state[pid]["can_withdraw"],
        )
        for pid in parties
    )
    per_privacy = tuple(
        PartyPrivacy(
            party_id=pid,
            privacy_invasion_level=_clamp01(state[pid]["privacy_invasion_level"]),
            consent_for_data_use=state[pid]["consent_for_data_use"],
            reidentification_risk=_clamp01(state[pid]["reidentification_risk"]),
        )
        for pid in parties
    )
    per_societal = tuple(
        PartySocietal(
            party_id=pid,
            environmental_burden=_clamp01(state[pid]["environmental_burden"]),
            long_term_risk=_clamp01(state[pid]["long_term_risk"]),
            benefit_to_future=_clamp01(state[pid]["benefit_to_future"]),
        )
        for pid in parties
    )
    per_virtue = tuple(
        PartyVirtue(
            party_id=pid,
            receives_compassion=state[pid]["receives_compassion"],
            trust_preserved=state[pid]["trust_preserved"],
            treated_as_end=state[pid]["treated_as_end"],
        )
        for pid in parties
    )
    per_procedural = tuple(
        PartyProcedural(
            party_id=pid,
            was_consulted=state[pid]["was_consulted"],
            can_contest=state[pid]["can_contest"],
            decision_explained=state[pid]["decision_explained"],
        )
        for pid in parties
    )

    n_parties = len(parties)
    consequences = ConsequencesV3(
        expected_benefit=_mean([p.expected_benefit for p in per_consequences]),
        expected_harm=_mean([p.expected_harm for p in per_consequences]),
        urgency=_max([state[pid]["expected_harm"] for pid in parties]) if parties else 0.0,
        affected_count=n_parties,
        per_party=per_consequences,
    )
    rights_and_duties = RightsAndDutiesV3(
        violates_rights=any(p.rights_violated for p in per_rights),
        has_valid_consent=all(p.consent_given for p in per_rights),
        violates_explicit_rule=False,
        role_duty_conflict=any(p.duty_owed for p in per_rights),
        per_party=per_rights,
    )
    justice_and_fairness = JusticeAndFairnessV3(
        discriminates_on_protected_attr=False,
        prioritizes_most_disadvantaged=any(p.is_disadvantaged for p in per_justice),
        exploits_vulnerable_population=any(p.is_disadvantaged for p in per_justice),
        exacerbates_power_imbalance=False,
        per_party=per_justice,
    )
    autonomy_and_agency = AutonomyAndAgencyV3(
        has_meaningful_choice=all(p.has_meaningful_choice for p in per_autonomy),
        coercion_or_undue_influence=any(p.is_coerced for p in per_autonomy),
        can_withdraw_without_penalty=all(p.can_withdraw for p in per_autonomy),
        manipulative_design_present=False,
        per_party=per_autonomy,
    )
    privacy_and_data = PrivacyAndDataGovernanceV3(
        privacy_invasion_level=_max([p.privacy_invasion_level for p in per_privacy]),
        data_minimization_respected=True,
        secondary_use_without_consent=any(not p.consent_for_data_use for p in per_privacy),
        data_retention_excessive=False,
        reidentification_risk=_max([p.reidentification_risk for p in per_privacy]),
        per_party=per_privacy,
    )
    societal_and_environmental = SocietalAndEnvironmentalV3(
        environmental_harm=_max([p.environmental_burden for p in per_societal]),
        long_term_societal_risk=_max([p.long_term_risk for p in per_societal]),
        benefits_to_future_generations=_mean([p.benefit_to_future for p in per_societal]),
        burden_on_vulnerable_groups=_max([p.environmental_burden for p in per_societal]),
        per_party=per_societal,
    )
    virtue_and_care = VirtueAndCareV3(
        expresses_compassion=all(p.receives_compassion for p in per_virtue),
        betrays_trust=any(not p.trust_preserved for p in per_virtue),
        respects_person_as_end=all(p.treated_as_end for p in per_virtue),
        per_party=per_virtue,
    )
    procedural_and_legitimacy = ProceduralAndLegitimacyV3(
        followed_approved_procedure=aggregate_legitimacy > 0.5,
        stakeholders_consulted=any(p.was_consulted for p in per_procedural),
        decision_explainable_to_public=all(p.decision_explained for p in per_procedural),
        contestation_available=all(p.can_contest for p in per_procedural),
        per_party=per_procedural,
    )
    epistemic_status = EpistemicStatusV3(
        uncertainty_level=_clamp01(epistemic_uncertainty),
        evidence_quality=(
            "low" if epistemic_uncertainty > 0.6
            else "medium" if epistemic_uncertainty > 0.3
            else "high"
        ),
        novel_situation_flag=False,
    )

    return EthicalFactsV3(
        option_id=ir.document.doc_id if ir.document else "scenario",
        consequences=consequences,
        rights_and_duties=rights_and_duties,
        justice_and_fairness=justice_and_fairness,
        autonomy_and_agency=autonomy_and_agency,
        privacy_and_data=privacy_and_data,
        societal_and_environmental=societal_and_environmental,
        virtue_and_care=virtue_and_care,
        procedural_and_legitimacy=procedural_and_legitimacy,
        epistemic_status=epistemic_status,
    )


# ---------- per-party fact application ------------------------------------


def _apply_fact_to_party_state(s: dict, fact: EthicalFact, magnitude: float) -> None:
    """Mutate per-party state dict in place based on a single compiler fact."""
    kind = fact.kind

    # Cross-dimensional propagation policy:
    # most ethical-fact kinds have a primary V3 dimension they affect,
    # plus a *secondary* harm/burden signal that the multi-dim moral
    # tensor needs to register. e.g., "coercion" is primarily an
    # autonomy violation but also a harm signal for the coerced party.
    if kind == "harm" or kind == "non_maleficence":
        s["expected_harm"] = max(s["expected_harm"], magnitude)
        s["relative_burden"] = max(s["relative_burden"], magnitude)
    elif kind == "coercion":
        s["is_coerced"] = True
        s["has_meaningful_choice"] = False
        s["can_withdraw"] = False
        # Coercion at grave+ severity is also a harm and rights signal.
        if fact.severity in ("grave", "catastrophic"):
            s["expected_harm"] = max(s["expected_harm"], magnitude * 0.8)
            s["rights_violated"] = True
    elif kind == "consent":
        if fact.severity in ("grave", "catastrophic"):
            s["consent_given"] = False
            s["rights_violated"] = True
            s["expected_harm"] = max(s["expected_harm"], magnitude * 0.5)
        elif "obtained" in fact.description.lower() or "given" in fact.description.lower():
            s["consent_given"] = True
    elif kind == "legitimacy":
        # Aggregate-only field; per-party only marks `was_consulted` when
        # severity is mild (procedural inclusion implied).
        if fact.severity in (None, "minor", "moderate"):
            s["was_consulted"] = True
        else:
            # Severe legitimacy facts (an authority that ought to be
            # illegitimate) impose burden on the parties named.
            s["relative_burden"] = max(s["relative_burden"], magnitude * 0.4)
    elif kind == "vulnerability":
        s["vulnerability_weight"] = max(s["vulnerability_weight"], 1.0 + magnitude)
        s["is_disadvantaged"] = True
    elif kind == "externality":
        s["long_term_risk"] = max(s["long_term_risk"], magnitude)
        s["environmental_burden"] = max(s["environmental_burden"], magnitude * 0.5)
        # Externalities at grave+ are direct harm to bearing parties.
        if fact.severity in ("grave", "catastrophic"):
            s["expected_harm"] = max(s["expected_harm"], magnitude * 0.9)
            s["relative_burden"] = max(s["relative_burden"], magnitude)
    elif kind == "justice":
        if fact.severity in ("grave", "catastrophic"):
            s["relative_burden"] = max(s["relative_burden"], magnitude)
            s["is_disadvantaged"] = True
    elif kind == "care":
        s["receives_compassion"] = True
        s["treated_as_end"] = True
        # Care facts for vulnerable subjects raise their benefit and
        # vulnerability weight.
        s["expected_benefit"] = max(s["expected_benefit"], magnitude * 0.5)
        s["vulnerability_weight"] = max(s["vulnerability_weight"], 1.0 + magnitude * 0.3)
    elif kind == "truth":
        s["trust_preserved"] = True
    elif kind == "deception":
        s["trust_preserved"] = False
        s["treated_as_end"] = False
        # Deception at moderate+ is also a small rights/harm signal.
        if fact.severity in ("moderate", "grave", "catastrophic"):
            s["expected_harm"] = max(s["expected_harm"], magnitude * 0.4)
    elif kind == "role_duty":
        s["duty_owed"] = True
    # uncertainty is decision-level — handled outside this fn.
    # reciprocity has no V3 mapping yet.


# ---------- helpers -------------------------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _max(xs: list[float]) -> float:
    return max(xs) if xs else 0.0
