"""Tests for the silicon-target scaffolding (Track C)."""

from erisml_compiler.silicon.fixed_point import (
    FixedPointConfig,
    cpp_typedef,
    dequantize_scalar,
    quantize_array,
    quantize_scalar,
)
from erisml_compiler.silicon.hls_emit import emit_fsm_cpp

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


# ---------- end-to-end smoke ----------


def test_silicon_emit_writes_the_fsm_source(tmp_path):
    """Mimic the `silicon-emit` CLI subcommand without invoking click. Since 0.10 it emits the
    FSMs only; the EM-DAG pipeline, the evaluator kernel and its Makefile are retired."""
    out = tmp_path / "silicon"
    out.mkdir()
    (out / "erisml_fsm.cpp").write_text(emit_fsm_cpp(), encoding="utf-8")
    assert sorted(p.name for p in out.iterdir()) == ["erisml_fsm.cpp"]
