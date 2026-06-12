"""RightsEM: evaluates rights-respect dimension. No upstream deps."""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class RightsEM(EthicalModule):
    name = "rights"
    dimension = "rights_respect"
    dependencies: tuple[str, ...] = ()

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        # Rights violations are encoded as "harm" facts whose subjects are
        # protected-class stakeholders, or as explicit rights violations.
        facts = facts_of_kind(ir, "harm")
        # Tier-2/3 extractors may attach `rights_violation` to the description.
        rights_facts = [f for f in facts if "right" in (f.description or "").lower()]
        score = aggregate_negative(rights_facts, explanation_prefix="Rights assessment: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in rights_facts],
            upstream_dependencies=[],
            notes=None,
        )
