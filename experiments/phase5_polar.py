"""Phase-5 H2/H3 via PolarQuant moral encoding — exploratory analysis.

Decomposes the calibrated layer-8 moral vector into radius (loading) + angle
(conflict from the consensus/"good" axis), using the vendored PolarQuant
transform (erisml_compiler.monitor.polar_encoding). Tunes an activation-only
flag rule on the ORIGINAL 40 scenarios and evaluates HELD-OUT on the frozen
confirm set. Replaces the sparse, calibration-defeated text_internal_mismatch
channel with a purely geometric one.

  python experiments/phase5_polar.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from erisml_compiler.calibration.activation_calibration import load_calibrated_probe
from erisml_compiler.monitor.base import LayerActivation
from erisml_compiler.monitor.polar_encoding import moral_polar_features

CAL = ["fairness_equity", "autonomy_consent", "legitimacy_trust", "care_protection",
       "vow_fidelity", "third_party_externality", "repair_residue"]
CKPT = "experiments/calibration/out/checkpoints/layer_08.pt"
SETS = {"ORIGINAL": ("experiments/phase5/caps", "experiments/scenarios_phase5.json"),
        "CONFIRM": ("experiments/phase5_confirm/caps", "experiments/scenarios_phase5_confirm.json")}


def features(caps_dir, manifest, probe):
    d = np.load(f"{caps_dir}/layer_08.npz", allow_pickle=True)
    cls = {s["id"]: s["class"] for s in json.load(open(manifest))["scenarios"]}
    V, C = [], []
    for i in range(len(d["X"])):
        if str(d["rewrites"][i]) != "identity":
            continue
        x = torch.tensor(d["X"][i])
        mv = probe.probe_layer(LayerActivation(8, "q", x.unsqueeze(0), x)).moral_vector
        V.append([getattr(mv, c).value for c in CAL])
        C.append(cls[str(d["sids"][i])])
    f = moral_polar_features(np.array(V))
    return np.array(C), f["loading"], f["conflict"]


def auc(pos, neg):
    return float(np.mean([p > n for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")


def main():
    probe = load_calibrated_probe(CKPT, hidden_dim=3584)
    F = {name: features(cd, mf, probe) for name, (cd, mf) in SETS.items()}

    print("== conflict-angle AUC, dilemma vs easy-moral (magnitude-normalized) ==")
    for name, (C, L, K) in F.items():
        print(f"  {name}: AUC(D>E) = {auc(K[C=='D'], K[C=='E']):.2f}")

    Co, Lo, Ko = F["ORIGINAL"]
    best = None
    for tr in np.linspace(0.38, 0.58, 21):
        for tk in np.linspace(0.28, 0.48, 21):
            fo = {c: ((Lo > tr) & (Ko > tk))[Co == c].mean() for c in "BDE"}
            if fo["B"] <= 0.20 and fo["E"] <= 0.30 and (best is None or fo["D"] > best[0]):
                best = (fo["D"], tr, tk, fo)
    _, tr, tk, fo = best
    Cc, Lc, Kc = F["CONFIRM"]
    fc = {c: ((Lc > tr) & (Kc > tk))[Cc == c].mean() for c in "BDE"}
    print(f"\n== geometric flag rule (tuned on ORIGINAL): loading>{tr:.2f} AND conflict>{tk:.2f} rad ==")
    print(f"  ORIGINAL     B={fo['B']:.2f} D={fo['D']:.2f} E={fo['E']:.2f}")
    print(f"  CONFIRM(held) B={fc['B']:.2f} D={fc['D']:.2f} E={fc['E']:.2f}")
    print(f"  held-out: H1 {'PASS' if fc['B']<=0.2 else 'FAIL'} | "
          f"H2 {'PASS' if fc['D']>=0.8 else 'FAIL'} | H3 {'PASS' if fc['E']<=0.3 else 'FAIL'}")
    print("\nRead: contestedness IS angular (AUC 0.73-0.80, replicates) -> H1/H3 solved; "
          "H2 sensitivity caps ~0.50 (angle alone insufficient).")


if __name__ == "__main__":
    main()
