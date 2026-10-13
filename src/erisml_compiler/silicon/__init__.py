"""Silicon-target scaffolding for the Tier-1 evaluator core.

Compiles the Python-defined FSMs into synthesizable C++ that Vitis HLS can target. (The EM-DAG
pipeline, the evaluator kernel and its Makefile are retired with the EM-DAG in 0.10; a DEME
evaluator target is future work.) The reference deployment target is the
**Xilinx Alveo U55C** PCIe accelerator, available via NRP Coder
(see docs/nrp_coder_deployment.md).

This package ships:
    - `hls_emit.py`         Python -> Vitis HLS C++ stub generator for
                            CommitmentFSM, LegitimacyFSM and ConsentFSM.
    - `fixed_point.py`      fixed-point arithmetic helpers for porting the
                            Mahalanobis evaluator off floating-point.
    - `examples/erisml_fsm.cpp`  example emitted output for the three FSMs.

What is NOT in this package:
    - A working bitstream. The actual `v++` synthesis run is a Phase 5
      task on real NRP hardware; see docs/silicon_target.md for the
      workflow. This package is the bridge code that turns Python
      definitions into HLS C++ ready for synthesis.
"""

from erisml_compiler.silicon.fixed_point import (
    FixedPointConfig,
    quantize_array,
    quantize_scalar,
)
from erisml_compiler.silicon.hls_emit import emit_fsm_cpp

__all__ = [
    "FixedPointConfig",
    "emit_fsm_cpp",
    "quantize_array",
    "quantize_scalar",
]
