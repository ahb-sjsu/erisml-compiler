"""Moral Machine adjudication: does the PolarQuant conflict-angle CORRELATE with contestedness
(cross-agent choice entropy) across ~628 mechanism-blind trolley scenarios spanning the range?

Reports Spearman(feature, entropy) per layer for: conflict-angle (the thesis), loading/radius
(the control — should NOT track contestedness), and the dev-frozen full-angle LDA score.
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
from scipy.stats import spearmanr

LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
ALL = ["physical_harm", "rights_respect", "fairness_equity", "autonomy_consent", "legitimacy_trust",
       "epistemic_quality", "care_protection", "vow_fidelity", "third_party_externality", "repair_residue"]
CAL = ["fairness_equity", "autonomy_consent", "legitimacy_trust", "care_protection", "vow_fidelity",
       "third_party_externality", "repair_residue"]
DEV = [("experiments/phase5/caps", "experiments/scenarios_phase5.json"),
       ("experiments/phase5_confirm/caps", "experiments/scenarios_phase5_confirm.json")]
CONTEST = json.load(open("experiments/mm_contest.json"))


def extract(caps_dir, layer, probe, want_class=None):
    d = np.load(f"{caps_dir}/layer_{layer:02d}.npz", allow_pickle=True)
    ids, V = [], []
    for i in range(len(d["X"])):
        if str(d["rewrites"][i]) != "identity":
            continue
        x = torch.tensor(d["X"][i])
        pr = probe.probe_layer(LayerActivation(layer, "q", x.unsqueeze(0), x))
        ids.append(str(d["sids"][i])); V.append(np.tanh(pr.logits.numpy()))
    return ids, np.array(V, np.float32)


def conflict_angle(V, di):
    Vs = V[:, di]
    r = np.linalg.norm(Vs, axis=1)
    u = np.ones(len(di)) / np.sqrt(len(di))
    cos = np.clip((Vs @ u) / (r + 1e-9), -1, 1)
    return np.arccos(np.abs(cos)), r


def angles(V, di):
    _, ang = polar_encode(_pad_pow2(V[:, di]))
    return np.concatenate(ang, axis=1)


print(f"{'layer':5s} {'dims':5s} | rho(conflict-angle) rho(loading,ctrl) rho(dev-LDA-angles)")
for L in LAYERS:
    probe = load_calibrated_probe(f"experiments/calibration/out/checkpoints/layer_{L:02d}.pt", hidden_dim=3584)
    ids, Vmm = extract("experiments/mm_caps", L, probe)
    y = np.array([CONTEST[i] for i in ids])                      # contestedness = choice entropy
    # dev for the LDA-on-angles rule
    Cd, Vd = [], []
    for cd, mf in DEV:
        classmap = {s["id"]: s["class"] for s in json.load(open(mf))["scenarios"]}
        i2, v2 = extract(cd, L, probe)
        Cd += [classmap[i] for i in i2]; Vd.append(v2)
    Cd = np.array(Cd); Vd = np.concatenate(Vd, 0)
    for dname, dims in [("cal7", CAL), ("all10", ALL)]:
        di = [ALL.index(x) for x in dims]
        ca, r = conflict_angle(Vmm, di)
        rho_c = spearmanr(ca, y).correlation
        rho_r = spearmanr(r, y).correlation
        lda = LinearDiscriminantAnalysis().fit(angles(Vd, di), (Cd == "D").astype(int))
        s = lda.decision_function(angles(Vmm, di))
        rho_l = spearmanr(s, y).correlation
        print(f"  {L:3d}  {dname:5s} |      {rho_c:+.3f}            {rho_r:+.3f}          {rho_l:+.3f}")
