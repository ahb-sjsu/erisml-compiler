"""Phase-5 discrimination run — calibrated activation lens over the 40 scenarios.

Wires the calibrated layer-8 probe into the I-EIP monitor and applies the prereg
instrument fixes C1-C4:

  C1  readout/equivariance drop the final layer (27); the activation MoralVector
      is read at the calibrated layer 8 (early-mid layers calibrate best).
  C2  gating rewrites: identity, lowercase, collapse_whitespace,
      strip_trailing_punctuation.
  C3  the delta lens scores only the 7 CALIBRATED dimensions
      (compare_morals(dimensions=...)).
  C4  text-lens neutrality: a dimension the text lens did not score
      (confidence == 0) is excluded from the delta, not treated as neutral.

Text lens: deterministic RULES tier (matches Phase 4, reproducible). Activation
lens: pooled Qwen2.5-7B activations (Atlas), read by the calibrated probe.

STATUS: WIP. The C1-C4 host analysis (analyze_scenario) is correct — C3 is
unit-tested, the calibrated probe reads valence standalone, the text lens runs
standalone. Two INTEGRATION fixes remain before the full run, both "separate
processes + files":
  1. TRANSPORT: Phase-4 run_atlas streams the payload as one base64 stdout line,
     which truncates for 40x4 captures. Swap to the file+sftp path used by
     experiments/calibration/atlas_label_and_extract.py (write npz on Atlas,
     sftp back).
  2. PROCESS ISOLATION: compile_document (text lens, loads its own models)
     SEGFAULTS when co-loaded with the torch probe in one process (native/OpenMP
     conflict). Precompute all 40 text MoralVectors in a text-lens-only
     subprocess -> json, then run the probe analysis in a separate process that
     loads those. Do NOT import torch and the text lens in the same interpreter.

  python experiments/phase5_run.py --scenarios experiments/scenarios_phase5.json \
      --checkpoint experiments/calibration/out/checkpoints/layer_08.pt --out experiments/phase5/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CALIBRATED = ["fairness_equity", "autonomy_consent", "legitimacy_trust",
              "care_protection", "vow_fidelity", "third_party_externality", "repair_residue"]
READOUT_LAYER = 8
EQUIV_COSINE = 0.95  # pooled-cosine threshold for equivariance (C1: excludes layer 27)

REWRITES = {  # C2 gating set (identity-rho asserted)
    "identity": lambda s: s,
    "lowercase": lambda s: s.lower(),
    "collapse_whitespace": lambda s: " ".join(s.split()),
    "strip_trailing_punctuation": lambda s: s.rstrip(" .!?,;:"),
}


def text_moral_vector(text: str):
    """Deterministic RULES-tier text lens -> MoralVector (Phase-4 comparable)."""
    import tempfile

    from erisml_compiler.ir.schemas import MoralVector
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        p = Path(f.name)
    try:
        ir = compile_document(p, CompileOptions(tier=CompilerTier.RULES))
        d = ir.model_dump() if hasattr(ir, "model_dump") else None
        mv = getattr(ir, "global_moral_vector", None)
        if mv is None and getattr(ir, "moral_vectors", None):
            mv = ir.moral_vectors[0]
        if mv is None and d:
            raw = d.get("global_moral_vector") or (d.get("moral_vectors") or [None])[0]
            mv = MoralVector.model_validate(raw) if raw else None
        return mv
    finally:
        p.unlink(missing_ok=True)


def scored_dims(text_mv) -> list[str]:
    """C3 x C4: calibrated dims that the text lens actually scored (confidence>0)."""
    out = []
    for d in CALIBRATED:
        ds = getattr(text_mv, d)
        if ds.confidence > 0.0:  # C4: skip dims the text lens did not score
            out.append(d)
    return out


def analyze_scenario(sid, text, captures, probe, layers):
    """captures: {rewrite_name: {layer_index: pooled_tensor}}. Returns a report dict
    in the Phase-4 schema that analyze_phase5.py consumes."""
    import torch

    from erisml_compiler.delta.compare import compare_morals
    from erisml_compiler.delta.equivariance import _cosine_sim
    from erisml_compiler.monitor.base import LayerActivation

    # --- activation MoralVector at the calibrated readout layer (C1) ---
    ident = captures["identity"]
    la8 = LayerActivation(layer_index=READOUT_LAYER, layer_name=f"q.{READOUT_LAYER}",
                          hidden=ident[READOUT_LAYER].unsqueeze(0), pooled=ident[READOUT_LAYER])
    act_mv = probe.probe_layer(la8).moral_vector

    # --- text lens + C3/C4 scored dims ---
    text_mv = text_moral_vector(text)
    scored = scored_dims(text_mv)
    delta = compare_morals(text_mv, act_mv, dimensions=scored) if scored else None

    # --- equivariance (C1: all layers except the readout-artifact layer 27) ---
    equiv_layers = [L for L in layers if L != 27]
    failed = []
    for name, cap in captures.items():
        if name == "identity":
            continue
        for L in equiv_layers:
            cos = _cosine_sim(ident[L], cap[L])
            if cos < EQUIV_COSINE:
                failed.append(L)
    failed = sorted(set(failed))

    text_mismatch = bool(delta.flag_for_review) if delta else False
    symmetry_break = len(failed) > 0
    modes = []
    if text_mismatch:
        modes.append("text_internal_mismatch")
    if symmetry_break:
        modes.append("group_symmetry_break")
    requires = text_mismatch or symmetry_break

    return {
        "summary": {
            "scenario": sid,
            "requires_human_review": requires,
            "delta_divergence": float(delta.divergence) if delta else 0.0,
            "delta_direction_breaks": int(delta.direction_break_count) if delta else 0,
            "equivariance_failed_layers": failed,
            "failure_modes_fired": modes,
            "scored_dimensions": scored,
            "readout_layer": READOUT_LAYER,
        },
        "delta": delta.to_dict() if delta else {"note": "no calibrated dim scored by text lens"},
        "activation_moral_vector": act_mv.model_dump(),
        "text_moral_vector": text_mv.model_dump(),
    }


def main():
    import base64
    import io

    import torch

    from scripts.experiments.atlas_phase4_experiment import _ATLAS_HARNESS, run_atlas  # noqa
    from erisml_compiler.calibration.activation_calibration import load_calibrated_probe

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="experiments/scenarios_phase5.json")
    ap.add_argument("--checkpoint", default="experiments/calibration/out/checkpoints/layer_08.pt")
    ap.add_argument("--out", default="experiments/phase5")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N scenarios (0=all)")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    scenarios = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))["scenarios"]
    if a.limit:
        # keep class balance in the smoke: take some of each class
        by = {}
        for s in scenarios:
            by.setdefault(s["class"], []).append(s)
        k = max(1, a.limit // 3)
        scenarios = (by.get("B", [])[:k] + by.get("D", [])[:k] + by.get("E", [])[:k])[:a.limit]

    items = [{"scenario": s["id"],
              "rewrites": [{"name": n, "text": fn(s["text"])} for n, fn in REWRITES.items()]}
             for s in scenarios]
    req = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "device": a.device,
           "max_tokens": a.max_tokens, "items": items}
    print(f"[phase5] extracting {len(items)} scenarios x {len(REWRITES)} rewrites on Atlas...")
    payload = run_atlas(req)
    layers = payload["selected_layers"]

    def decode(b64):
        return torch.load(io.BytesIO(base64.b64decode(b64)), weights_only=True)

    probe = load_calibrated_probe(a.checkpoint, hidden_dim=payload["hidden_dim"])
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2text = {s["id"]: s["text"] for s in scenarios}
    summaries = []
    for sc in payload["results"]:
        sid = sc["scenario"]
        captures = {rw["name"]: {int(L): decode(rw["layer_vecs_b64"][str(L)]) for L in layers}
                    for rw in sc["rewrites"]}
        rep = analyze_scenario(sid, id2text[sid], captures, probe, layers)
        (out_dir / f"{sid}_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        summaries.append(rep["summary"])
        print(f"  {sid}: flag={rep['summary']['requires_human_review']} "
              f"div={rep['summary']['delta_divergence']:.2f} "
              f"eqfail={rep['summary']['equivariance_failed_layers']} "
              f"scored={len(rep['summary']['scored_dimensions'])}")
    (out_dir / "phase5_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[phase5] wrote {len(summaries)} reports to {out_dir}")


if __name__ == "__main__":
    main()
