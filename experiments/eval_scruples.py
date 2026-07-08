"""Mechanism-blind test: apply the top PolarQuant configs (classifier FROZEN on phase5+confirm dev)
to the Scruples set (real AITA, authored with zero knowledge of the layer-8 conflict mechanism).

D = contested (balanced AITA votes), E = clear (lopsided votes), B = benign. The decisive question:
does contestedness-is-angular survive stimuli nobody tuned to?
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

ALL = ["physical_harm", "rights_respect", "fairness_equity", "autonomy_consent", "legitimacy_trust",
       "epistemic_quality", "care_protection", "vow_fidelity", "third_party_externality", "repair_residue"]
CAL = ["fairness_equity", "autonomy_consent", "legitimacy_trust", "care_protection", "vow_fidelity",
       "third_party_externality", "repair_residue"]
DEV = [("experiments/phase5/caps", "experiments/scenarios_phase5.json"),
       ("experiments/phase5_confirm/caps", "experiments/scenarios_phase5_confirm.json")]
SCR = ("experiments/scruples_caps", "experiments/scruples_manifest.json")


def cls(mf):
    return {s["id"]: s["class"] for s in json.load(open(mf))["scenarios"]}


def extract(caps_dir, layer, probe, classmap):
    d = np.load(f"{caps_dir}/layer_{layer:02d}.npz", allow_pickle=True)
    C, V = [], []
    for i in range(len(d["X"])):
        if str(d["rewrites"][i]) != "identity":
            continue
        x = torch.tensor(d["X"][i])
        pr = probe.probe_layer(LayerActivation(layer, "q", x.unsqueeze(0), x))
        C.append(classmap[str(d["sids"][i])]); V.append(np.tanh(pr.logits.numpy()))
    return np.array(C), np.array(V, np.float32)


def feats(V, di, kind):
    Vs = V[:, di]
    r = np.linalg.norm(Vs, axis=1, keepdims=True)
    _, ang = polar_encode(_pad_pow2(Vs))
    angv = np.concatenate(ang, axis=1)
    return {"angles": angv, "angles+r": np.c_[angv, r[:, 0]]}[kind]


def run(layer, dims, kind):
    probe = load_calibrated_probe(f"experiments/calibration/out/checkpoints/layer_{layer:02d}.pt", hidden_dim=3584)
    Cd, Vd = [], []
    for cd, mf in DEV:
        c, v = extract(cd, layer, probe, cls(mf)); Cd.append(c); Vd.append(v)
    Cdev, Vdev = np.concatenate(Cd), np.concatenate(Vd, 0)
    Cs, Vs = extract(SCR[0], layer, probe, cls(SCR[1]))
    di = [ALL.index(x) for x in dims]
    Fd, Fs = feats(Vdev, di, kind), feats(Vs, di, kind)
    lda = LinearDiscriminantAnalysis().fit(Fd, (Cdev == "D").astype(int))
    sd, ss = lda.decision_function(Fd), lda.decision_function(Fs)
    # freeze tau on dev: keep H1<=0.2, maximize H2-H3
    best = None
    for tau in np.unique(sd):
        fl = sd > tau
        if fl[Cdev == "B"].mean() <= 0.20:
            sc = fl[Cdev == "D"].mean() - fl[Cdev == "E"].mean()
            if best is None or sc > best[0]:
                best = (sc, tau)
    tau = best[1]
    fs = ss > tau
    return {c: float(fs[Cs == c].mean()) for c in "BDE"}


for layer, dims, kind, name in [(8, CAL, "angles+r", "FROZEN L8/cal7/angles+r"),
                                (8, CAL, "angles", "L8/cal7/angles"),
                                (4, ALL, "angles", "L4/all10/angles (strongest signal)"),
                                (4, ALL, "angles+r", "L4/all10/angles+r (H2=1.0 on confirm2)")]:
    r = run(layer, dims, kind)
    ok = "PASS" if (r["B"] <= 0.20 and r["D"] >= 0.80 and r["E"] <= 0.30) else ""
    print(f"{name:36s}  B={r['B']:.2f} D={r['D']:.2f} E={r['E']:.2f}  {ok}")
