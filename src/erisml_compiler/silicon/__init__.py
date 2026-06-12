"""Silicon-target scaffolding for the Tier-1 evaluator core.

Compiles the Python-defined FSMs and EM-DAG into synthesizable C++ that
Vitis HLS can target. The reference deployment target is the
**Xilinx Alveo U55C** PCIe accelerator, available via NRP Coder
(see docs/nrp_coder_deployment.md).

This package ships:
    - `hls_emit.py`         Python -> Vitis HLS C++ stub generator for
                            CommitmentFSM, LegitimacyFSM, ConsentFSM, and
                            the EM-DAG topology.
    - `fixed_point.py`      fixed-point arithmetic helpers for porting the
                            Mahalanobis evaluator off floating-point.
    - `examples/*.cpp`      example emitted output for the three FSMs.
    - `u55c_constraints.tcl`  Vivado constraint template for the U55C.
    - `Makefile.template`   build recipe targeting `v++` (Vitis compiler).

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
from erisml_compiler.silicon.hls_emit import (
    emit_em_dag_pipeline,
    emit_fsm_cpp,
    emit_makefile,
    emit_top_module,
)

__all__ = [
    "FixedPointConfig",
    "emit_em_dag_pipeline",
    "emit_fsm_cpp",
    "emit_makefile",
    "emit_top_module",
    "quantize_array",
    "quantize_scalar",
]
