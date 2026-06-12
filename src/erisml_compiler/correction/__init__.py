"""Human-correction loop (Track A of Phase 3).

The correction loop closes a critical feedback channel:

    raw extraction -> human correction -> RLEF dataset -> calibration ->
    better extraction -> fewer corrections needed

This package provides:
    - `diff_irs` and `IRDiff`: structural diff between two CompilerIR objects.
    - `Corrector`: apply a corrections file to an IR and re-validate.
    - `CorrectionRecord`: append-only audit-trail entry for one correction.
"""
from erisml_compiler.correction.corrector import (
    CorrectionRecord,
    Corrector,
    apply_corrections,
)
from erisml_compiler.correction.diff import IRDiff, diff_irs

__all__ = [
    "CorrectionRecord",
    "Corrector",
    "IRDiff",
    "apply_corrections",
    "diff_irs",
]
