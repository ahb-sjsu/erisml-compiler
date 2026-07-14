"""Close the MM-null caveat: is ρ≈0 because the conflict-angle has no VARIANCE on Moral Machine
(uninformative test), or because it varies but doesn't track contestedness (real null)?

Reports, for layers 4 and 8 (cal7): angle spread on DEV vs MM, the contestedness spread, the
DEV D-vs-E AUC (where the effect 'worked'), and the MM top-vs-bottom-tercile AUC (mechanism-blind).
"""
import json
import sys
import numpy as np
import torch
sys.path.insert(0, "src")
from erisml_compiler.calibration.activation_calibration import load_calibrated_probe
from erisml_compiler.monitor.base import LayerActivation
from sklearn.metrics import roc_auc_score

CAL = ["fairness_equity", "autonomy_consent", "legitimacy_trust", "care_protection", "vow_fidelity",
       "third_party_externality", "repair_residue"]
ALL = ["physical_harm", "rights_respect", "fairness_equity", "autonomy_consent", "legitimacy_trust",
       "epistemic_quality", "care_protection", "vow_fidelity", "third_party_externality", "repair_residue"]
DEV = [("experiments/phase5/caps", "experiments/scenarios_phase5.json"),
       ("experiments/phase5_confirm/caps", "experiments/scenarios_phase5_confirm.json")]
CONTEST = json.load(open("experiments/mm_contest.json"))
di = [ALL.index(x) for x in CAL]


def extract(caps_dir, layer, probe):
    d = np.load(f"{caps_dir}/layer_{layer:02d}.npz", allow_pickle=True)
    ids, V = [], []
    for i in range(len(d["X"])):
        if str(d["rewrites"][i]) != "identity":
            continue
        x = torch.tensor(d["X"][i])
        pr = probe.probe_layer(LayerActivation(layer, "q", x.unsqueeze(0), x))
        ids.append(str(d["sids"][i])); V.append(np.tanh(pr.logits.numpy()))
    return ids, np.array(V, np.float32)


def cangle(V):
    Vs = V[:, di]; r = np.linalg.norm(Vs, axis=1)
    u = np.ones(len(di)) / np.sqrt(len(di))
    return np.arccos(np.clip(np.abs((Vs @ u) / (r + 1e-9)), 0, 1))


for L in [4, 8]:
    probe = load_calibrated_probe(f"experiments/calibration/out/checkpoints/layer_{L:02d}.pt", hidden_dim=3584)
    # dev
    Cd, Vd = [], []
    for cd, mf in DEV:
        cm = {s["id"]: s["class"] for s in json.load(open(mf))["scenarios"]}
        i2, v2 = extract(cd, L, probe); Cd += [cm[i] for i in i2]; Vd.append(v2)
    Cd = np.array(Cd); ad = cangle(np.concatenate(Vd, 0))
    de = np.isin(Cd, ["D", "E"])
    auc_dev = roc_auc_score((Cd[de] == "D").astype(int), ad[de])   # dev D>E on angle (where it "worked")
    # mm
    ids, Vmm = extract("experiments/mm_caps", L, probe)
    amm = cangle(Vmm); y = np.array([CONTEST[i] for i in ids])
    lo, hi = np.quantile(y, [1/3, 2/3])
    mask = (y <= lo) | (y >= hi)
    auc_mm = roc_auc_score((y[mask] >= hi).astype(int), amm[mask])  # high-contest vs low on angle
    print(f"layer {L}:")
    print(f"  angle spread  DEV std={ad.std():.3f} [{ad.min():.2f},{ad.max():.2f}] | "
          f"MM std={amm.std():.3f} [{amm.min():.2f},{amm.max():.2f}]   (MM has variance? {'YES' if amm.std()>0.05 else 'NO'})")
    print(f"  contestedness spread  std={y.std():.3f} [{y.min():.2f},{y.max():.2f}]")
    print(f"  DEV  angle AUC(D>E)            = {auc_dev:.3f}   (the effect that 'worked')")
    print(f"  MM   angle AUC(high>low contest)= {auc_mm:.3f}   (mechanism-blind; 0.50 = null)")
