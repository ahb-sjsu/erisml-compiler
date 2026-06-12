"""AutonomyEM: evaluates autonomy-consent dimension.

Depends on `legitimacy`: consent obtained under illegitimate authority is
not real consent. If the upstream legitimacy score is strongly negative, the
autonomy score is dragged down even when explicit consent is recorded.
"""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, DimensionScore, EMOutput


class AutonomyEM(EthicalModule):
    name = "autonomy"
    dimension = "autonomy_consent"
    dependencies: tuple[str, ...] = ("legitimacy",)

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        consent_facts = facts_of_kind(ir, "consent")
        coercion_facts = facts_of_kind(ir, "coercion")
        all_relevant = consent_facts + coercion_facts
        bad = [
            f for f in all_relevant
            if any(kw in (f.description or "").lower() for kw in ("coerc", "non-consensual", "imposed", "forced", "without consent"))
        ]
        score = aggregate_negative(bad, explanation_prefix="Autonomy assessment: ")

        # Compose with legitimacy: if legitimacy is severely negative,
        # autonomy cannot be positive (no real consent under tyranny).
        legit_score = upstream["legitimacy"].score
        if legit_score.value < -0.5 and score.value > legit_score.value:
            score = DimensionScore(
                value=legit_score.value,
                confidence=min(score.confidence, legit_score.confidence),
                uncertainty=max(score.uncertainty, legit_score.uncertainty),
                direction="negative",
                source_spans=score.source_spans + legit_score.source_spans,
                explanation=(
                    f"{score.explanation} Composed with legitimacy assessment: "
                    f"consent under illegitimate authority is void."
                ),
            )
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in bad],
            upstream_dependencies=["legitimacy"],
            notes=None,
        )
