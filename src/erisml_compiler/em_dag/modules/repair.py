"""RepairEM: evaluates repair-residue dimension.

Depends on multiple upstream modules. Repair-residue measures the moral
debt remaining after an action: when harm has occurred or commitments have
been violated, repair-residue is high (negative score = repair needed).
"""

from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import (
    violated_commitments,
)
from erisml_compiler.ir.schemas import CompilerIR, DimensionScore, EMOutput


class RepairEM(EthicalModule):
    name = "repair"
    dimension = "repair_residue"
    dependencies: tuple[str, ...] = ("harm", "externality", "fidelity")

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        # Aggregate negative contributions from harm, externality, fidelity.
        contributions = []
        for dep in ("harm", "externality", "fidelity"):
            up = upstream[dep]
            if up.score.value < 0:
                contributions.append((dep, up.score))

        if not contributions:
            return EMOutput(
                module_name=self.name,
                score=DimensionScore(
                    value=0.0,
                    confidence=1.0,
                    uncertainty=0.0,
                    direction="neutral",
                    source_spans=[],
                    explanation="No repair debt detected.",
                ),
                contributing_facts=[],
                upstream_dependencies=list(self.dependencies),
                notes=None,
            )

        # Repair-residue magnitude = average of negative upstream magnitudes.
        magnitudes = [abs(s.value) for _, s in contributions]
        avg = sum(magnitudes) / len(magnitudes)
        spans: list[str] = []
        for _, s in contributions:
            spans.extend(s.source_spans)
        violated_ids = [c.id for c in violated_commitments(ir)]
        explanation = (
            f"Repair debt from: {', '.join(dep for dep, _ in contributions)}. "
            f"Magnitude={avg:.2f}."
        )
        if violated_ids:
            explanation += f" Violated commitments: {violated_ids}."

        return EMOutput(
            module_name=self.name,
            score=DimensionScore(
                value=-avg,
                confidence=0.8,
                uncertainty=0.2,
                direction="negative",
                source_spans=spans,
                explanation=explanation,
            ),
            contributing_facts=[],
            upstream_dependencies=list(self.dependencies),
            notes=None,
        )
