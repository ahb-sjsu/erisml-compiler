// ===============================================================
// erisml_top.cpp
//
// Top-level Vitis HLS entry point for the Tier-1 evaluator
// targeting Xilinx Alveo U55C via NRP Coder.
//
// The host program (NRP Coder workspace) feeds a stream of
// (IR feature vector, FSM event tags) tuples; the FPGA returns
// the DEME verdict + 10-dim EMOutput array.
// ===============================================================

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Forward declarations from erisml_fsm.cpp and erisml_em_dag.cpp.
struct IRFeatures;
struct EMOutput;
void em_dag_pipeline(const IRFeatures &features, EMOutput out[10]);

extern "C" {

// Top-level kernel: one IRFeatures in, 10 EMOutput out, plus a
// packed verdict word.
void erisml_evaluate(
    const IRFeatures &features,
    EMOutput em_outputs[10],
    ap_uint<32> &packed_verdict
) {
#pragma HLS INTERFACE m_axi port=em_outputs offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=features
#pragma HLS INTERFACE s_axilite port=packed_verdict
#pragma HLS INTERFACE s_axilite port=return

    em_dag_pipeline(features, em_outputs);

    // Phase-5: pack DEME verdict bits.
    // bit  0-2 : verdict_kind (8 verdict types)
    // bit  3-10: confidence_q8 (0..255)
    // bit 11    : escalation_required
    // bit 12-31: reserved
    packed_verdict = 0;
}

}  // extern "C"
