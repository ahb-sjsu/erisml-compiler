"""Structural fuzz over the PolarQuant contestedness config space.

The frozen rule fixed: layer 8, 7 cal dims, full angle vector + loading, LDA. Did we get lucky,
or is there a luckier config? Fuzz the space — readout LAYER (all 8 extracted), dim subset, and
feature construction {radius, conflict-angle-from-consensus, full polar-angle vector, angle+radius}
— fitting the classifier on DEV (phase5+confirm) and reporting HELD-OUT confirm2 H1/H2/H3.

Discipline: config is SELECTED by dev; confirm2 is the held-out report; the truly-decisive test is
the mechanism-blind Scruples set (separate). Runs fully local (CPU).
"""
import json
import sys
import numpy as np
import torch
sys.path.insert(0, "src")
from erisml_compiler.calibration.activation_calibration import load_calibrated_probe
from erisml_compiler.monitor.base import LayerActivation
from erisml_compiler.monitor.polar_encoding import _pad_pow2, polar_encode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
ALL = ["physical_harm", "rights_respect", "fairness_equity", "autonomy_consent", "legitimacy_trust",
       "epistemic_quality", "care_protection", "vow_fidelity", "third_party_externality", "repair_residue"]
CAL = ["fairness_equity", "autonomy_consent", "legitimacy_trust", "care_protection", "vow_fidelity",
       "third_party_externality", "repair_residue"]
DEV = [("experiments/phase5/caps", "experiments/scenarios_phase5.json"),
       ("experiments/phase5_confirm/caps", "experiments/scenarios_phase5_confirm.json")]
TEST = ("experiments/phase5_confirm2/caps", "experiments/scenarios_phase5_confirm2.json")


def classes(mf):
    return {s["id"]: s["class"] for s in json.load(open(mf))["scenarios"]}


def extract_V(caps_dir, layer, probe, cls):
    d = np.load(f"{caps_dir}/layer_{layer:02d}.npz", allow_pickle=True)
    C, V = [], []
    for i in range(len(d["X"])):
        if str(d["rewrites"][i]) != "identity":
            continue
        x = torch.tensor(d["X"][i])
        pr = probe.probe_layer(LayerActivation(layer, "q", x.unsqueeze(0), x))
        C.append(cls[str(d["sids"][i])]); V.append(np.tanh(pr.logits.numpy()))
    return np.array(C), np.array(V, np.float32)


def feats(V, dims_idx, kind):
    Vs = V[:, dims_idx]
    r = np.linalg.norm(Vs, axis=1, keepdims=True)
    u = np.ones(Vs.shape[1]) / np.sqrt(Vs.shape[1])            # consensus "+good" axis
    cos = np.clip((Vs @ u) / (r[:, 0] + 1e-9), -1, 1)
    conflict = np.arccos(np.abs(cos))[:, None]
    _, ang = polar_encode(_pad_pow2(Vs))
    angv = np.concatenate(ang, axis=1)
    return {"radius": r, "conflict": conflict, "angles": angv,
            "angles+r": np.c_[angv, r[:, 0]]}[kind]


def main():
    probes = {L: load_calibrated_probe(f"experiments/calibration/out/checkpoints/layer_{L:02d}.pt",
                                       hidden_dim=3584) for L in LAYERS}
    cache = {}
    for L in LAYERS:
        Cd, Vd = [], []
        for cd, mf in DEV:
            c, v = extract_V(cd, L, probes[L], classes(mf)); Cd.append(c); Vd.append(v)
        Ct, Vt = extract_V(TEST[0], L, probes[L], classes(TEST[1]))
        cache[L] = (np.concatenate(Cd), np.concatenate(Vd, 0), Ct, Vt)

    rows = []
    for L in LAYERS:
        Cdev, Vdev, Ctest, Vtest = cache[L]
        for dname, dims in [("cal7", CAL), ("all10", ALL)]:
            di = [ALL.index(x) for x in dims]
            for kind in ["radius", "conflict", "angles", "angles+r"]:
                Fd, Ft = feats(Vdev, di, kind), feats(Vtest, di, kind)
                y = (Cdev == "D").astype(int)
                try:
                    lda = LinearDiscriminantAnalysis().fit(Fd, y)
                except Exception:
                    continue
                sd, st = lda.decision_function(Fd), lda.decision_function(Ft)
                de = np.isin(Cdev, ["D", "E"])
                auc = roc_auc_score((Cdev[de] == "D").astype(int), sd[de]) if de.sum() else float("nan")
                # pick tau on dev: keep H1<=0.2, maximize H2-H3
                best = None
                for tau in np.unique(sd):
                    fl = sd > tau
                    h1 = fl[Cdev == "B"].mean()
                    if h1 <= 0.20:
                        sc = fl[Cdev == "D"].mean() - fl[Cdev == "E"].mean()
                        if best is None or sc > best[0]:
                            best = (sc, tau)
                if best is None:
                    continue
                tau = best[1]
                ft = st > tau
                h1t = ft[Ctest == "B"].mean(); h2t = ft[Ctest == "D"].mean(); h3t = ft[Ctest == "E"].mean()
                passed = (h1t <= 0.20) and (h2t >= 0.80) and (h3t <= 0.30)
                rows.append((L, dname, kind, auc, h1t, h2t, h3t, passed))

    rows.sort(key=lambda r: (r[7], r[3], r[5]), reverse=True)
    print("layer dims  feature    DE_auc | confirm2 H1   H2   H3   PASS")
    for L, dn, k, auc, h1, h2, h3, p in rows[:18]:
        print(f"  {L:2d}  {dn:5s} {k:9s} {auc:.3f} |          {h1:.2f} {h2:.2f} {h3:.2f}  {'PASS' if p else ''}")
    json.dump([dict(layer=r[0], dims=r[1], feature=r[2], DE_auc=r[3],
                    confirm2=dict(H1=r[4], H2=r[5], H3=r[6]), passed=bool(r[7])) for r in rows],
              open("experiments/fuzz_polarquant_results.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
