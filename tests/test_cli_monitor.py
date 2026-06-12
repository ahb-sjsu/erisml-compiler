"""CLI smoke tests for `eris-compile monitor` and `eris-compile delta`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from click.testing import CliRunner

from erisml_compiler.cli import cli


def _run(args: list[str], **kw):
    runner = CliRunner()
    return runner.invoke(cli, args, **kw)


def test_cli_monitor_mock_emits_trace(tmp_path: Path):
    out = tmp_path / "trace.json"
    result = _run(
        [
            "monitor",
            "hello world",
            "--source",
            "mock",
            "--hidden-dim",
            "16",
            "--n-layers",
            "3",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    d = json.loads(out.read_text())
    assert d["source_name"] == "mock"
    assert d["hidden_dim"] == 16
    assert len(d["per_layer"]) == 3
    assert "aggregated" in d
    assert "trace_hash" in result.output


def test_cli_delta_against_synthetic_ir_and_trace(tmp_path: Path):
    # Create a trace.
    trace_path = tmp_path / "trace.json"
    res1 = _run(
        [
            "monitor",
            "subject text",
            "--source",
            "mock",
            "--hidden-dim",
            "12",
            "--n-layers",
            "4",
            "--out",
            str(trace_path),
        ]
    )
    assert res1.exit_code == 0, res1.output

    # Synthesise a minimal IR-shaped JSON whose only required slot for
    # `delta` is global_moral_vector.
    from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, DimensionScore, MoralVector

    text_mv = MoralVector(
        **{
            dim: DimensionScore(value=0.0, confidence=0.5, uncertainty=0.3, direction="neutral")
            for dim in MORAL_DIMENSIONS
        }
    )
    ir_path = tmp_path / "ir.json"
    ir_path.write_text(
        json.dumps({"global_moral_vector": text_mv.model_dump()}),
        encoding="utf-8",
    )

    out = tmp_path / "delta.json"
    res2 = _run(
        [
            "delta",
            str(ir_path),
            str(trace_path),
            "--out",
            str(out),
        ]
    )
    assert res2.exit_code == 0, res2.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "delta" in payload
    assert "failure_modes" in payload
    assert "divergence" in payload["delta"]
    # Sanity: the per-dimension list has all 10 moral dims.
    assert len(payload["delta"]["per_dimension"]) == 10


def test_cli_delta_falls_back_to_timeline_when_no_global_vector(tmp_path: Path):
    trace_path = tmp_path / "trace.json"
    res1 = _run(
        ["monitor", "subject", "--hidden-dim", "8", "--n-layers", "2", "--out", str(trace_path)]
    )
    assert res1.exit_code == 0

    from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, DimensionScore, MoralVector

    mv = MoralVector(
        **{
            dim: DimensionScore(value=0.1, confidence=0.6, uncertainty=0.2, direction="positive")
            for dim in MORAL_DIMENSIONS
        }
    )
    ir_path = tmp_path / "ir.json"
    ir_path.write_text(
        json.dumps(
            {"moral_timeline": [{"time_index": 0, "event_label": "e", "vector": mv.model_dump()}]}
        ),
        encoding="utf-8",
    )

    out = tmp_path / "d.json"
    res2 = _run(["delta", str(ir_path), str(trace_path), "--out", str(out)])
    assert res2.exit_code == 0, res2.output
    assert out.exists()


def test_cli_delta_errors_on_ir_with_no_moral_vector(tmp_path: Path):
    trace_path = tmp_path / "t.json"
    _run(["monitor", "x", "--hidden-dim", "8", "--n-layers", "2", "--out", str(trace_path)])

    ir_path = tmp_path / "ir.json"
    ir_path.write_text("{}", encoding="utf-8")

    out = tmp_path / "d.json"
    res = _run(["delta", str(ir_path), str(trace_path), "--out", str(out)])
    assert res.exit_code != 0
    assert "global_moral_vector" in res.output or "moral_timeline" in res.output
