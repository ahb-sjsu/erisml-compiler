"""LegitimacyEM: evaluates legitimacy-trust dimension. No upstream deps."""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class LegitimacyEM(EthicalModule):
    name = "legitimacy"
    dimension = "legitimacy_trust"
    dependencies: tuple[str, ...] = ()

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        # Legitimacy is violated when authority is coercive, fraudulent, or
        # tyrannical. We pull legitimacy facts and coercion facts directed at
        # an authority figure.
        legit_facts = facts_of_kind(ir, "legitimacy")
        coercion_facts = facts_of_kind(ir, "coercion")
        bad_legitimacy = [
            f for f in legit_facts
            if any(kw in (f.description or "").lower() for kw in ("void", "tyrann", "coerc", "fraud"))
        ]
        all_bad = bad_legitimacy + coercion_facts
        score = aggregate_negative(all_bad, explanation_prefix="Legitimacy assessment: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in all_bad],
            upstream_dependencies=[],
            notes=None,
        )
