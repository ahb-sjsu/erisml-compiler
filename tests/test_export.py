"""Export tests: JSON round-trip, RLEF schema."""

import json
from pathlib import Path

import pytest

from erisml_compiler.export.json_export import export_json, load_json
from erisml_compiler.export.rlef import export_rlef, to_rlef_record
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def compiled_ir():
    return compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
    )


def test_json_round_trip(tmp_path, compiled_ir):
    out = tmp_path / "ir.json"
    export_json(compiled_ir, out)
    loaded = load_json(out)
    assert loaded.audit.ir_hash == compiled_ir.audit.ir_hash
    assert loaded.canonical_form == compiled_ir.canonical_form
    assert len(loaded.stakeholders) == len(compiled_ir.stakeholders)


def test_rlef_export(tmp_path, compiled_ir):
    out = tmp_path / "rlef.json"
    export_rlef(compiled_ir, out)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema"] == "rlef_v0.1"
    assert record["source_text"]
    assert record["canonical_form"] == "coercive_murderous_interrogation_with_collective_reprisal"
    assert record["deme_verdict"]["verdict"] == "tragic_conflict_escalate"


def test_rlef_with_corrections(compiled_ir):
    record = to_rlef_record(compiled_ir, human_corrections={"reviewer": "alice", "notes": "ok"})
    assert record["human_corrections"]["reviewer"] == "alice"
