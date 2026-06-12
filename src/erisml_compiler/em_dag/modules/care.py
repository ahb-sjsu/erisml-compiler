"""CareEM: evaluates care-protection dimension.

Depends on `harm`. Care is the positive counterpart to harm: actively
protecting vulnerable parties. We compute care strength from `care` facts
and from the presence of vulnerable stakeholders with protectors.
"""

from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_positive, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, DimensionScore, EMOutput


class CareEM(EthicalModule):
    name = "care"
    dimension = "care_protection"
    dependencies: tuple[str, ...] = ("harm",)

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        care_facts = facts_of_kind(ir, "care")
        # If there are vulnerable parties with explicit protectors, that
        # strengthens the care signal.
        vulnerable = [s for s in ir.stakeholders if s.vulnerability in ("high", "extreme")]
        protectors = [s for s in ir.stakeholders if "protector" in s.roles]
        if care_facts:
            score = aggregate_positive(care_facts, explanation_prefix="Care assessment: ")
        elif vulnerable and protectors:
            score = DimensionScore(
                value=0.7,
                confidence=0.75,
                uncertainty=0.25,
                direction="positive",
                source_spans=[],
                explanation=(
                    f"{len(vulnerable)} vulnerable stakeholder(s) with "
                    f"{len(protectors)} protector(s) identified."
                ),
            )
        else:
            score = DimensionScore(
                value=0.0,
                confidence=1.0,
                uncertainty=0.0,
                direction="neutral",
                source_spans=[],
                explanation="No care-related facts detected.",
            )
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in care_facts],
            upstream_dependencies=["harm"],
            notes=f"vulnerable={len(vulnerable)}, protectors={len(protectors)}",
        )
