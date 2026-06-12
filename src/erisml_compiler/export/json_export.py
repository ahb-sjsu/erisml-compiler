"""JSON export and round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from erisml_compiler.ir.schemas import CompilerIR


def export_json(ir: CompilerIR, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = ir.model_dump(mode="json")
    p.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    return p


def load_json(path: str | Path) -> CompilerIR:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return CompilerIR.model_validate(data)
