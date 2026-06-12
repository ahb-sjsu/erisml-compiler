"""Reproduce the headline nazi_attic result end-to-end.

Emits all the artifacts a researcher / reviewer would want to verify
the compiler against the bundled example:

  out/nazi_attic.ir.json        — compiled IR with V3 tensor
  out/nazi_attic.rlef.json      — RLEF training record
  out/nazi_attic.trace.json     — I-EIP Monitor activation trace
  out/nazi_attic.delta.json     — Delta lens report (text vs. activations)
  out/nazi_attic.report.html    — HTML viz of the IR
  out/nazi_attic.bundle/        — full audit bundle directory
  out/nazi_attic.summary.txt    — plain-text summary of all hashes

Idempotent; safe to re-run. Exits non-zero on any failure so this can
land in CI as a smoke for the full pipeline.

Per release-planning-01:
> Add a make reproduce-nazi-attic command that emits IR, V3 tensor,
> DEME verdict, audit hash, monitor trace, delta report, RLEF record,
> and HTML report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out" / "reproduce_nazi_attic"
EXAMPLE = REPO_ROOT / "examples" / "nazi_attic.txt"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if res.returncode != 0:
        print(f"!! command failed with exit {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ir_path = OUT_DIR / "nazi_attic.ir.json"
    rlef_path = OUT_DIR / "nazi_attic.rlef.json"
    trace_path = OUT_DIR / "nazi_attic.trace.json"
    delta_path = OUT_DIR / "nazi_attic.delta.json"
    report_path = OUT_DIR / "nazi_attic.report.html"
    bundle_dir = OUT_DIR / "nazi_attic.bundle"
    summary_path = OUT_DIR / "nazi_attic.summary.txt"

    # 1. Compile with V3 rank-2. Strict mode so a silent V2 fallback fails CI.
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "compile",
        str(EXAMPLE),
        "--extractor", "mock",
        "--canonicalizer", "registry",
        "--rank", "2",
        "--strict-v3",
        "--out", str(ir_path),
    ])

    # 2. RLEF training record.
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "rlef",
        str(ir_path),
        "--out", str(rlef_path),
    ])

    # 3. I-EIP Monitor activation trace (mock source — deterministic).
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "monitor",
        "Soldiers are at the door asking about the Jews you are hiding.",
        "--source", "mock",
        "--hidden-dim", "64",
        "--n-layers", "8",
        "--out", str(trace_path),
    ])

    # 4. Delta lens.
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "delta",
        str(ir_path),
        str(trace_path),
        "--out", str(delta_path),
    ])

    # 5. HTML report.
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "report",
        str(ir_path),
        "--out", str(report_path),
    ])

    # 6. Audit bundle.
    _run([
        sys.executable, "-m", "erisml_compiler.cli", "bundle",
        str(ir_path),
        "--out", str(bundle_dir),
    ])

    # 7. Plain-text summary covering hashes + headline numbers.
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    delta = json.loads(delta_path.read_text(encoding="utf-8"))

    audit_hash = (ir.get("audit") or {}).get("ir_hash", "(missing)")
    tensor = ir.get("moral_tensor_v3") or {}
    verdict = (ir.get("deme_verdict") or {}).get("verdict", "(missing)")
    canonical = ir.get("canonical_form", "(missing)")
    fairness = ir.get("fairness_metrics") or {}
    ppv = ir.get("per_party_verdicts") or {}
    proof = ir.get("decision_proof") or {}
    sa = ir.get("strategic_analysis") or {}
    fm = delta.get("failure_modes") or {}
    div = (delta.get("delta") or {}).get("divergence", float("nan"))

    summary_lines = [
        "ErisML Compiler — nazi_attic reproduction summary",
        "=" * 60,
        f"  canonical form     : {canonical}",
        f"  DEME verdict       : {verdict}",
        f"  IR audit hash      : {audit_hash}",
        "",
        "V3 tensor:",
        f"  rank               : {tensor.get('rank')}",
        f"  shape              : {tensor.get('shape')}",
        f"  build_strategy     : {tensor.get('metadata', {}).get('build_strategy')}",
        "",
        "Per-party verdicts:",
        *[f"  {pid:20s} : {v}" for pid, v in ppv.items()],
        "",
        f"Gini (harm)          : {fairness.get('gini_harm')}",
        f"Worst-off (harm)     : {fairness.get('worst_off_harm_value')}",
        "",
        "Strategic (Shapley):",
        *[f"  {k:20s} : {v}" for k, v in (sa.get("shapley_values") or {}).items()],
        "",
        "DecisionProof:",
        f"  proof_hash         : {proof.get('proof_hash', '(missing)')}",
        f"  prev (=audit hash) : {proof.get('previous_proof_hash', '(missing)')}",
        f"  forbidden          : {proof.get('forbidden_options')}",
        "",
        "I-EIP Monitor trace:",
        f"  trace_hash         : {(trace.get('per_layer') or [{}])[0].get('layer_name', '?')}",
        f"  n layers probed    : {len(trace.get('per_layer', []))}",
        f"  delta divergence   : {div:.4f}" if isinstance(div, (int, float)) else f"  delta divergence   : {div}",
        f"  failure modes fired: {fm.get('fired')}",
        f"  requires_human_review: {fm.get('requires_human_review')}",
        "",
        "Artifacts emitted:",
        f"  IR     : {ir_path}",
        f"  RLEF   : {rlef_path}",
        f"  trace  : {trace_path}",
        f"  delta  : {delta_path}",
        f"  report : {report_path}",
        f"  bundle : {bundle_dir}",
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print(f"[+] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
