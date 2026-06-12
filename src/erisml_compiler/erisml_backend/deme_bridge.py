"""DEME bridge.

The Tier-1 geometric DEME evaluator. Given a fully-populated IR with EM-DAG
outputs and a MoralVector, produce a structured DEMEVerdict.

This is the Phase-1 working implementation. The interface is the
structural-fuzzing target shape so Phase 2 can swap in a real DEME
implementation without changing the IR contract.

Decision logic (deterministic, silicon-castable):

    if any catastrophic externality + any active commitment defeasibility:
        -> tragic_conflict_escalate
    elif harm.value < -0.85 and care.value < 0.5:
        -> prohibited
    elif fidelity.value > 0.5 and externality.value > -0.5 and harm.value > -0.5:
        -> permitted
    elif repair.value < -0.3:
        -> permitted_with_residue
    elif any required_review_facts:
        -> requires_human_review
    else:
        -> indeterminate
"""

from __future__ import annotations

from erisml_compiler.ir.schemas import (
    CompilerIR,
    DEMEVerdict,
    MoralVector,
)


class DEMEBridge:
    """The DEME bridge. Phase-1 implementation."""

    def __init__(self, profile_name: str = "default_em_dag_v0.1"):
        self.profile_name = profile_name

    def evaluate(self, ir: CompilerIR, vector: MoralVector) -> DEMEVerdict:
        # Surface common signals.
        externality_v = vector.third_party_externality.value
        harm_v = vector.physical_harm.value
        care_v = vector.care_protection.value
        fidelity_v = vector.vow_fidelity.value
        repair_v = vector.repair_residue.value
        legitimacy_v = vector.legitimacy_trust.value

        # Defeasibility check.
        any_defeasible = any(c.status == "active_but_defeasible" for c in ir.commitments)
        catastrophic_externality = externality_v <= -0.85

        # Allowed / prohibited action regions (text descriptions).
        allowed: list[str] = []
        prohibited: list[str] = []
        residue: list[str] = []

        # Tragic conflict: catastrophic externality + defeasible commitment.
        if catastrophic_externality and any_defeasible:
            allowed.append("Escalate to higher human decision authority.")
            prohibited.append("Unilateral action that imposes catastrophic non-consensual risk.")
            residue.append(
                "Even after escalation, non-consenting affected parties bear residual risk."
            )
            return DEMEVerdict(
                verdict="tragic_conflict_escalate",
                confidence=0.85,
                rationale=(
                    "Catastrophic non-consensual externality is in tension with "
                    "an active-but-defeasible commitment. No clean resolution; "
                    "human escalation required."
                ),
                allowed_actions=allowed,
                prohibited_actions=prohibited,
                moral_residue=residue,
                escalation_required=True,
            )

        # Outright prohibition: severe harm with no countervailing care.
        if harm_v < -0.85 and care_v < 0.5 and legitimacy_v < 0:
            prohibited.append("Any action whose proximate effect is the documented harm.")
            return DEMEVerdict(
                verdict="prohibited",
                confidence=0.9,
                rationale=(
                    "Catastrophic harm with no protective counterweight and "
                    "with legitimacy compromised. Action is prohibited."
                ),
                allowed_actions=allowed,
                prohibited_actions=prohibited,
                moral_residue=residue,
                escalation_required=False,
            )

        # Permitted with residue: legitimate action with some repair debt.
        if fidelity_v > 0.3 and externality_v > -0.3 and harm_v > -0.5 and repair_v < -0.1:
            allowed.append("Proceed with the proposed action.")
            residue.append(
                "Minor repair debt remains; consider mitigation, restitution, or apology."
            )
            return DEMEVerdict(
                verdict="permitted_with_residue",
                confidence=0.8,
                rationale="Net positive evaluation with manageable repair debt.",
                allowed_actions=allowed,
                prohibited_actions=prohibited,
                moral_residue=residue,
                escalation_required=False,
            )

        # Clean permit.
        if fidelity_v > 0.5 and externality_v > -0.3 and harm_v > -0.3:
            allowed.append("Proceed with the proposed action.")
            return DEMEVerdict(
                verdict="permitted",
                confidence=0.85,
                rationale="No active conflict; commitments honoured; no significant harm or externality.",
                allowed_actions=allowed,
                prohibited_actions=prohibited,
                moral_residue=residue,
                escalation_required=False,
            )

        # Require human review when uncertainty or extractor flags are high.
        flagged = any(s.requires_review for s in ir.stakeholders)
        if flagged or vector.epistemic_quality.uncertainty > 0.5:
            return DEMEVerdict(
                verdict="requires_human_review",
                confidence=0.6,
                rationale="Uncertainty above threshold or stakeholders flagged for review.",
                allowed_actions=allowed,
                prohibited_actions=prohibited,
                moral_residue=residue,
                escalation_required=False,
            )

        return DEMEVerdict(
            verdict="indeterminate",
            confidence=0.5,
            rationale="Insufficient signal to resolve.",
            allowed_actions=allowed,
            prohibited_actions=prohibited,
            moral_residue=residue,
            escalation_required=False,
        )
