"""HarmEM: evaluates physical-harm dimension. No upstream deps."""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class HarmEM(EthicalModule):
    name = "harm"
    dimension = "physical_harm"
    dependencies: tuple[str, ...] = ()

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        facts = facts_of_kind(ir, "harm") + facts_of_kind(ir, "non_maleficence")
        score = aggregate_negative(facts, explanation_prefix="Harm assessment: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in facts],
            upstream_dependencies=[],
            notes=None,
        )
