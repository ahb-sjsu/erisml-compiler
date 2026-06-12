"""FairnessEM: evaluates fairness-equity dimension. No upstream deps."""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class FairnessEM(EthicalModule):
    name = "fairness"
    dimension = "fairness_equity"
    dependencies: tuple[str, ...] = ()

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        facts = facts_of_kind(ir, "justice")
        # Treat as negative-signal if descriptions mention unfair, biased, discriminatory.
        bad = [f for f in facts if any(
            kw in (f.description or "").lower() for kw in ("unfair", "biased", "discriminat")
        )]
        score = aggregate_negative(bad, explanation_prefix="Fairness assessment: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in bad],
            upstream_dependencies=[],
            notes=None,
        )
