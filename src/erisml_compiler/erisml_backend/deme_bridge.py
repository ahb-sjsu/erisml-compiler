"""DEME bridge: the compiler's verdict, judged by erisml-lib's DEME.

The compiler extracts; DEME judges. The IR's facts go through the V3 bridge
(``ir_to_v2_facts``) to one ``EthicalFacts`` for the situation the document
describes, and erisml-lib's ``DEMEPipeline`` judges it under a DEME profile: the
design guide's hard-shell pattern, Geneva as the base module at tier 0,
CorePhysicalSafety at tier 1, AutonomyConsent at tier 2, and the advisory
tragic-conflict module at tier 3.

Until 0.10 this module was a stand-in, a few thresholds on the compiler's own
moral vector ("harm below -0.85 and care below 0.5 is prohibited"), shaped so
that "Phase 2 can swap in a real DEME implementation without changing the IR
contract". This is that swap. The contract (``DEMEVerdict``) is unchanged.

DEME judges options; the compiler's verdict labels a situation. The mapping:

- ``prohibited``: DEME forbids the option (a reflex rule, the deontic maxim gate,
  or a binding module's veto); the vetoes are the prohibited actions;
- ``tragic_conflict_escalate``: not forbidden, and DEME escalates it (a prohibited
  act whose purpose is an imperfect duty, as "deceive to protect those in my
  care"), or the tragic-conflict module finds a high conflict index; escalation
  is required;
- ``requires_human_review``: not forbidden, and a stakeholder is flagged for
  review or the facts' uncertainty exceeds 0.5;
- ``permitted``: DEME's aggregate for the option prefers it (scalar at least 0.6,
  the governance aggregation's threshold for "prefer");
- ``permitted_with_residue``: permitted but not preferred, the moral costs its
  modules recorded being the residue;
- ``indeterminate``: the IR has no ethical facts to judge.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from erisml.ethics.governance.config_v2 import DimensionWeights
from erisml.ethics.interop.profile_adapters import pipeline_from_profile
from erisml.ethics.moral_vector import MoralVector
from erisml.ethics.modules.base import V1ToV2Adapter
from erisml.ethics.modules.greek_tragedy_tragic_conflict_em import TragicConflictEM
from erisml.ethics.profile_v04 import DEMEProfileV04

from erisml_compiler.ir.schemas import CompilerIR, DEMEVerdict

DEFAULT_PROFILE = DEMEProfileV04(
    name="erisml_compiler_default",
    description="The compiler's default DEME lens: the design guide's hard shell.",
    stakeholder_label="the document's parties",
    composition="hard_shell",
    active_em_names=["core_physical_safety", "autonomy_consent"],
)
PREFER = 0.6  # aggregation_v2.aggregate_judgements_v2: a scalar at or above this "prefers"


class DEMEBridge:
    """Judge an IR's situation with erisml-lib's DEME under a profile."""

    def __init__(
        self,
        profile_name: str | None = None,
        profile: DEMEProfileV04 | None = None,
        dimension_weights: DimensionWeights | None = None,
    ):
        """``dimension_weights`` (a fitted ethos profile's, evaluation.ethos) replace the
        profile's own."""
        self.profile = profile or DEFAULT_PROFILE
        self.ethos_applied = dimension_weights is not None
        if dimension_weights is not None:
            self.profile = dataclasses.replace(
                self.profile, moral_dimension_weights=dimension_weights
            )
        self.profile_name = profile_name or self.profile.name
        self.pipeline = pipeline_from_profile(self.profile)
        self.tragic = V1ToV2Adapter(TragicConflictEM(), em_tier=3)
        self.pipeline.add_em(self.tragic)

    def evaluate(self, ir: CompilerIR, vector: Any = None) -> DEMEVerdict:
        """``vector`` is accepted for the earlier signature and not read: DEME
        judges the facts, not a vector."""
        return self.judge(ir)[0]

    def judge(self, ir: CompilerIR) -> tuple[DEMEVerdict, MoralVector | None]:
        """DEME's verdict on the IR's situation and its aggregated moral vector (erisml-lib's
        nine dimensions), or no vector when there are no facts to judge."""
        from erisml_compiler.erisml_backend.v3_bridge import ir_to_v2_facts

        if not ir.ethical_facts:
            return (
                DEMEVerdict(
                    verdict="indeterminate",
                    confidence=0.5,
                    rationale="The IR has no ethical facts for DEME to judge.",
                ),
                None,
            )
        facts = ir_to_v2_facts(ir)
        result = self.pipeline.decide([facts])
        # an option DEME vetoed before judging it (a reflex rule, the deontic gate) has no
        # aggregated vector; its facts' own vector (the library's standard mapping) stands in
        vector = result.moral_landscape.get(facts.option_id) or MoralVector.from_ethical_facts(
            facts
        )
        return self._verdict(ir, facts, result), vector

    def _verdict(self, ir: CompilerIR, facts, result) -> DEMEVerdict:
        oid = facts.option_id
        vetoes, costs = self._reasons(result)
        es = facts.epistemic_status
        uncertainty = es.uncertainty_level if es is not None else 0.0
        common: dict[str, Any] = dict(confidence=max(0.0, min(1.0, 1.0 - uncertainty)))

        if oid in result.forbidden_options:
            return DEMEVerdict(
                verdict="prohibited",
                rationale=f"DEME ({self.profile_name}) forbids it. {result.rationale}",
                prohibited_actions=vetoes or ["The described action, which DEME forbids."],
                **common,
            )
        if oid in result.escalations:
            return DEMEVerdict(
                verdict="tragic_conflict_escalate",
                rationale=f"DEME ({self.profile_name}) escalates it: {result.escalations[oid]}",
                allowed_actions=["Escalate to a higher human decision authority."],
                moral_residue=costs,
                escalation_required=True,
                **common,
            )
        if self.tragic.judge(facts).metadata.get("tragic_conflict_high"):
            return DEMEVerdict(
                verdict="tragic_conflict_escalate",
                rationale=f"DEME ({self.profile_name}) finds a tragic conflict. {result.rationale}",
                allowed_actions=["Escalate to a higher human decision authority."],
                moral_residue=costs,
                escalation_required=True,
                **common,
            )
        if any(s.requires_review for s in ir.stakeholders) or uncertainty > 0.5:
            return DEMEVerdict(
                verdict="requires_human_review",
                rationale=(
                    "A stakeholder is flagged for review or the facts' uncertainty is above "
                    f"0.5. {result.rationale}"
                ),
                **common,
            )
        vec = result.moral_landscape.get(oid)
        weights = self.pipeline.config.governance.dimension_weights.to_dict()
        score = vec.to_scalar(weights=weights) if vec is not None else 0.0
        if score >= PREFER:
            return DEMEVerdict(
                verdict="permitted",
                rationale=f"DEME ({self.profile_name}) prefers it (score {score:.3f}).",
                allowed_actions=["Proceed with the described action."],
                **common,
            )
        return DEMEVerdict(
            verdict="permitted_with_residue",
            rationale=f"DEME ({self.profile_name}) permits it with costs (score {score:.3f}).",
            allowed_actions=["Proceed with the described action."],
            moral_residue=costs or [f"DEME's aggregate score is {score:.3f}."],
            **common,
        )

    @staticmethod
    def _reasons(result) -> tuple[list[str], list[str]]:
        """The vetoes and the costs DEME's modules recorded, read from the decision proof."""
        vetoes: list[str] = []
        costs: list[str] = []
        proof = result.proof
        if proof is None:
            return vetoes, costs
        for j in proof.em_judgements:
            if j.veto_triggered and j.reason_summary:
                vetoes.append(f"{j.em_name}: {j.reason_summary}")
            elif j.verdict in ("avoid", "neutral") and j.reason_summary:
                costs.append(f"{j.em_name}: {j.reason_summary}")
        for layer in proof.layer_outputs:
            for reason in (layer.output_data or {}).get("deontic_vetoes", {}).values():
                vetoes.append(f"deontic gate: {reason}")
        return vetoes, costs
