"""Train calibrated activation probes from per-layer feature files (host, CPU).

Consumes the `layer_*.npz` files produced by `atlas_label_and_extract.py`,
fits one `ProbeHead(D->10)` per layer, and writes:
  - checkpoints/layer_XX.pt        : calibrated probe + CalibrationProvenance
  - phase5_calibration_table.md    : the §6 table (best held-out sign-acc per dim)
  - phase5_calibration_results.json: full per-layer/per-dim metrics

Only dimensions whose held-out sign-agreement >= 0.70 (prereg C3) are marked
included; the rest are reported as uncalibrated so the Phase-5 delta lens
excludes them. Pick the readout layer for the monitor from the summary (C1 says
avoid the final layer).

  python experiments/calibration/train_all_layers.py --features calib_features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from erisml_compiler.calibration.activation_calibration import (
    ActivationCalibConfig,
    calibration_table_rows,
    render_calibration_table_md,
    save_layer_checkpoint,
    train_layer_probe,
)


def run(features_dir: str, out_dir: str, teacher: str, epochs: int, seed: int,
        corpus_fp: dict | None):
    feat = sorted(Path(features_dir).glob("layer_*.npz"))
    if not feat:
        raise SystemExit(f"no layer_*.npz in {features_dir}")
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = ActivationCalibConfig(epochs=epochs, seed=seed, log_every=0)
    results = []
    for f in feat:
        d = np.load(f, allow_pickle=True)
        X, Y = d["X"], d["Y"]
        layer_index = int(d["layer_index"])
        model_id = str(d["model_id"]) if "model_id" in d else None
        head, res = train_layer_probe(X, Y, cfg, layer_index=layer_index)
        save_layer_checkpoint(
            head, out / "checkpoints" / f"layer_{layer_index:02d}.pt", res,
            corpus_fingerprint=corpus_fp, model_id=model_id, teacher=teacher)
        results.append(res)
        print(res.summary(), flush=True)

    rows = calibration_table_rows(results, teacher=teacher)
    (out / "phase5_calibration_table.md").write_text(
        "# Phase-5 §6 probe calibration (activation lens)\n\n"
        f"Teacher: {teacher}. Metric: held-out sign-agreement vs teacher, on dims "
        "the teacher took a stance on (|v|>0.05). Included iff >= 0.70 (prereg C3).\n\n"
        + render_calibration_table_md(rows) + "\n", encoding="utf-8")

    payload = {
        "teacher": teacher, "epochs": epochs, "seed": seed,
        "layers": [{
            "layer_index": r.layer_index, "n_train": r.n_train, "n_val": r.n_val,
            "final_train_loss": r.final_train_loss,
            "per_dim_signacc": r.per_dim_signacc, "per_dim_mae": r.per_dim_mae,
            "per_dim_n_eval": r.per_dim_n_eval, "included": r.included,
        } for r in results],
        "table": rows,
    }
    (out / "phase5_calibration_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_dims_any = sum(any(r.included[d] for r in results) for d in results[0].included)
    print(f"\n{n_dims_any}/{len(results[0].included)} dims calibrated on at least one layer")
    print(f"wrote {out/'phase5_calibration_table.md'} and results json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="calib_features")
    ap.add_argument("--out-dir", default="experiments/calibration/out")
    ap.add_argument("--teacher", default="qwen3+glm-5 signed consensus")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--corpus-fingerprint", default=None,
                    help="path to calib_corpus.fingerprint.json (folded into provenance)")
    a = ap.parse_args()
    corpus_fp = json.loads(Path(a.corpus_fingerprint).read_text()) if a.corpus_fingerprint else None
    run(a.features, a.out_dir, a.teacher, a.epochs, a.seed, corpus_fp)


if __name__ == "__main__":
    main()
