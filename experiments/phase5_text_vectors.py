"""Phase-5 step 1/3: precompute text-lens MoralVectors for the 40 scenarios.

Runs the deterministic RULES-tier text lens ONLY (no torch) — it must be a
separate process from the activation probe, which loads torch: co-loading the
text lens and torch in one interpreter segfaults (native/OpenMP conflict).

  python experiments/phase5_text_vectors.py \
      --scenarios experiments/scenarios_phase5.json --out experiments/phase5/text_vectors.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def text_moral_vector_dict(text: str) -> dict:
    from erisml_compiler.ir.schemas import MoralVector
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        p = Path(f.name)
    try:
        ir = compile_document(p, CompileOptions(tier=CompilerTier.RULES))
        mv = getattr(ir, "global_moral_vector", None)
        if mv is None and getattr(ir, "moral_vectors", None):
            mv = ir.moral_vectors[0]
        if mv is None:
            d = ir.model_dump()
            raw = d.get("global_moral_vector") or (d.get("moral_vectors") or [None])[0]
            mv = MoralVector.model_validate(raw) if raw else None
        return mv.model_dump() if mv is not None else None
    finally:
        p.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="experiments/scenarios_phase5.json")
    ap.add_argument("--out", default="experiments/phase5/text_vectors.json")
    a = ap.parse_args()

    scenarios = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))["scenarios"]
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Resume: the text lens loads a torch model per compile and can crash after
    # many iterations; keep partial results and skip already-done ids so an outer
    # loop can relaunch until complete. Write after EACH compile.
    out = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    for s in scenarios:
        if s["id"] in out:
            continue
        out[s["id"]] = text_moral_vector_dict(s["text"])
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        nz = sum(1 for k, v in (out[s["id"]] or {}).items()
                 if isinstance(v, dict) and abs(v.get("value", 0)) > 0.01)
        print(f"  {s['id']} ({s['class']}): {nz} non-neutral dims", flush=True)
    print(f"wrote {outp} ({len(out)}/{len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
