"""FidelityEM: evaluates vow-fidelity dimension.

Depends on `legitimacy`: a vow extracted by an illegitimate authority is
void; a vow to commit a wrong is defeasible. Per spec section 8.6:
'Vows are binding but defeasible.'
"""

from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import (
    active_commitments,
    violated_commitments,
)
from erisml_compiler.ir.schemas import CompilerIR, DimensionScore, EMOutput


class FidelityEM(EthicalModule):
    name = "fidelity"
    dimension = "vow_fidelity"
    dependencies: tuple[str, ...] = ("legitimacy",)

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        active = active_commitments(ir)
        violated = violated_commitments(ir)
        # Positive: active commitments being honored. Negative: violated.
        if violated:
            spans: list[str] = []
            for c in violated:
                spans.extend(c.source_spans)
            score = DimensionScore(
                value=-0.8,
                confidence=0.9,
                uncertainty=0.1,
                direction="negative",
                source_spans=spans,
                explanation=f"{len(violated)} commitment(s) violated.",
            )
            return EMOutput(
                module_name=self.name,
                score=score,
                contributing_facts=[],
                upstream_dependencies=["legitimacy"],
                notes=f"Violated commitment ids: {[c.id for c in violated]}",
            )
        if active:
            spans = []
            for c in active:
                spans.extend(c.source_spans)
            defeasible = any(c.status == "active_but_defeasible" for c in active)
            # If legitimacy is void, vows to that authority are defeated;
            # vows to protected parties remain.
            value = 0.6 if defeasible else 0.85
            score = DimensionScore(
                value=value,
                confidence=0.85,
                uncertainty=0.15,
                direction="positive",
                source_spans=spans,
                explanation=(
                    f"{len(active)} active commitment(s); "
                    f"{'some defeasible' if defeasible else 'all binding'}."
                ),
            )
            return EMOutput(
                module_name=self.name,
                score=score,
                contributing_facts=[],
                upstream_dependencies=["legitimacy"],
                notes=f"Active commitment ids: {[c.id for c in active]}",
            )

        return EMOutput(
            module_name=self.name,
            score=DimensionScore(
                value=0.0,
                confidence=1.0,
                uncertainty=0.0,
                direction="neutral",
                source_spans=[],
                explanation="No commitments detected.",
            ),
            contributing_facts=[],
            upstream_dependencies=["legitimacy"],
            notes=None,
        )
