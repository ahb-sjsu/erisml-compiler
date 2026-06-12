"""Tests for the silicon-target scaffolding (Track C)."""

from erisml_compiler.silicon.fixed_point import (
    FixedPointConfig,
    cpp_typedef,
    dequantize_scalar,
    quantize_array,
    quantize_scalar,
)
from erisml_compiler.silicon.hls_emit import (
    emit_em_dag_pipeline,
    emit_fsm_cpp,
    emit_makefile,
    emit_top_module,
)

# ---------- fixed point ----------


def test_fixed_point_default_range():
    cfg = FixedPointConfig(total_bits=16, int_bits=4, signed=True)
    assert cfg.frac_bits == 12
    lo, hi = cfg.value_range()
    assert lo < 0 and hi > 0
    assert cfg.cpp_type == "ap_fixed<16, 4>"


def test_quantize_round_trip():
    cfg = FixedPointConfig(total_bits=16, int_bits=4, signed=True)
    for v in (0.0, 0.5, -0.5, 1.0, -1.0, 7.99, -7.99):
        q = quantize_scalar(v, cfg)
        recovered = dequantize_scalar(q, cfg)
        # Round-trip error bounded by the fixed-point resolution.
        assert abs(recovered - v) < 1.0 / (1 << cfg.frac_bits) * 2


def test_quantize_saturates_at_extremes():
    cfg = FixedPointConfig(total_bits=8, int_bits=2, signed=True)  # range ~[-2, +1.98]
    # Way above max:
    q_hi = quantize_scalar(1000.0, cfg)
    q_lo = quantize_scalar(-1000.0, cfg)
    assert q_hi == (1 << (cfg.total_bits - 1)) - 1
    assert q_lo == -(1 << (cfg.total_bits - 1))


def test_quantize_array_preserves_length():
    cfg = FixedPointConfig()
    out = quantize_array([0.1, 0.2, 0.3, -0.4], cfg)
    assert len(out) == 4


def test_cpp_typedef_format():
    cfg = FixedPointConfig(total_bits=24, int_bits=8, signed=False)
    td = cpp_typedef("my_t", cfg)
    assert td == "typedef ap_ufixed<24, 8> my_t;"


# ---------- HLS emission ----------


def test_emit_fsm_cpp_contains_three_fsms():
    src = emit_fsm_cpp()
    assert "commitment_step" in src
    assert "legitimacy_step" in src
    assert "consent_step" in src
    assert "#include <ap_int.h>" in src
    # Terminal-state guards present:
    assert "COMMITMENT_VIOLATED" in src
    assert "LEGITIMACY_VOID" in src


def test_emit_fsm_cpp_well_formed_braces():
    """Sanity: balanced braces in the emitted code."""
    src = emit_fsm_cpp()
    assert src.count("{") == src.count("}")
    assert src.count("(") == src.count(")")


def test_emit_fsm_cpp_includes_hls_pragmas():
    src = emit_fsm_cpp()
    assert "#pragma HLS INLINE" in src
    assert "#pragma HLS PIPELINE" in src


def test_emit_em_dag_pipeline_topologically_ordered():
    src = emit_em_dag_pipeline()
    # legitimacy must appear before autonomy (autonomy depends on legitimacy).
    legitimacy_pos = src.find("em_legitimacy")
    autonomy_pos = src.find("em_autonomy")
    assert legitimacy_pos < autonomy_pos
    # harm must appear before externality.
    harm_pos = src.find("em_harm")
    externality_pos = src.find("em_externality")
    assert harm_pos < externality_pos


def test_emit_em_dag_pipeline_has_dataflow_pragma():
    src = emit_em_dag_pipeline()
    assert "#pragma HLS DATAFLOW" in src
    assert "em_dag_pipeline" in src


def test_emit_em_dag_pipeline_balanced_braces():
    src = emit_em_dag_pipeline()
    assert src.count("{") == src.count("}")


def test_emit_top_module_has_axi_pragmas():
    src = emit_top_module()
    assert "INTERFACE m_axi" in src
    assert "INTERFACE s_axilite" in src
    assert "erisml_evaluate" in src


def test_emit_makefile_targets_u55c():
    mk = emit_makefile()
    assert "xilinx_u55c" in mk
    assert "v++" in mk or "VPP" in mk
    assert "TARGET" in mk


# ---------- end-to-end smoke ----------


def test_silicon_emit_writes_all_files(tmp_path):
    """Mimic the `silicon-emit` CLI subcommand without invoking click."""
    out = tmp_path / "silicon"
    out.mkdir()
    (out / "erisml_fsm.cpp").write_text(emit_fsm_cpp(), encoding="utf-8")
    (out / "erisml_em_dag.cpp").write_text(emit_em_dag_pipeline(), encoding="utf-8")
    (out / "erisml_top.cpp").write_text(emit_top_module(), encoding="utf-8")
    (out / "Makefile").write_text(emit_makefile(), encoding="utf-8")
    files = sorted(p.name for p in out.iterdir())
    assert files == ["Makefile", "erisml_em_dag.cpp", "erisml_fsm.cpp", "erisml_top.cpp"]
