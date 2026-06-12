"""Export the IR as an ErisML-like YAML source file."""

from __future__ import annotations

from pathlib import Path

from erisml_compiler.erisml_backend.codegen import render_erisml
from erisml_compiler.ir.schemas import CompilerIR


def export_erisml(ir: CompilerIR, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_erisml(ir), encoding="utf-8")
    return p
