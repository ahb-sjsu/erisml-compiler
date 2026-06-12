"""Ethical Module abstract base."""
from __future__ import annotations

from abc import ABC, abstractmethod

from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class EthicalModule(ABC):
    """A single Ethical Module in the DAG.

    Each module evaluates a focused ethical concern (harm, autonomy, legitimacy,
    fidelity, etc.) and emits an `EMOutput` carrying a DimensionScore plus
    pointers to the ethical facts that contributed.

    Subclasses declare:
        - `name`: identifier used in the DAG and in audit records.
        - `dimension`: which MoralVector dimension this module projects onto.
        - `dependencies`: names of other modules this one needs the output of.

    Subclasses implement `evaluate`. The evaluator may inspect any field of the
    IR and may consult prior module outputs from upstream dependencies.
    """

    name: str
    dimension: str
    dependencies: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name") or not cls.name:
            raise TypeError(f"{cls.__name__} must define `name`")
        if not hasattr(cls, "dimension") or not cls.dimension:
            raise TypeError(f"{cls.__name__} must define `dimension`")

    @abstractmethod
    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        """Evaluate this module against the IR.

        Args:
            ir: the compiler IR, fully populated up through the EM-DAG pass
                (segments, stakeholders, events, commitments, ethical facts).
            upstream: outputs from modules listed in `self.dependencies`,
                keyed by module name.

        Returns:
            an EMOutput whose `dimension_score` reflects this module's
            assessment.
        """
        raise NotImplementedError
