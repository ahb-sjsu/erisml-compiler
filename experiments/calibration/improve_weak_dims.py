"""Does denoising the teacher lift the 3 weak dims (rights/physical/epistemic)?

Diagnosis showed those dims have low qwen3<->glm-5 sign agreement, so their
CONSENSUS label is noisy. This retrains layer 8 on the CLEAN per-dimension
subset (both judges present and agreeing in sign, incl. both-neutral) with a
per-dim masked loss, and evaluates held-out sign-agreement on clean items only.

If a weak dim clears 0.70 under clean labels -> it was disagreement noise (use
clean calibration). If it stays flat -> the activations don't encode it and
Phase 5 honestly runs on the dims that do.

  python experiments/calibration/improve_weak_dims.py --layer 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_rubric import DIMENSIONS, to_vector  # noqa: E402  (sibling module)

DB = 0.05
THRESH = 0.70


def build_clean(labels_path: str, ids: np.ndarray):
    """Return (target (N,10), mask (N,10)) aligned to ids. mask=1 where both
    judges took the SAME sign (or both neutral) on that dim; target=mean."""
    raw = {r["id"]: r["raw"] for r in
           (json.loads(l) for l in Path(labels_path).read_text(encoding="utf-8").splitlines() if l.strip())}
    N, D = len(ids), len(DIMENSIONS)
    target = np.zeros((N, D), np.float32)
    mask = np.zeros((N, D), np.float32)
    for i, tid in enumerate(ids):
        judges = raw.get(str(tid), {})
        if len(judges) < 2:
            continue
        vecs = [to_vector(v) for v in judges.values()]
        for k in range(D):
            vals = [v[k] for v in vecs]
            signs = [1 if x > DB else (-1 if x < -DB else 0) for x in vals]
            if signs[0] == signs[1]:                 # agree (incl. both neutral)
                mask[i, k] = 1.0
                target[i, k] = float(np.mean(vals))
    return target, mask


def masked_train(X, target, mask, epochs, seed, device="cpu"):
    import torch
    import torch.nn.functional as F

    from erisml_compiler.calibration.activation_calibration import _split
    from erisml_compiler.calibration.probe_head import ProbeHead

    X = torch.as_tensor(X, dtype=torch.float32)
    T = torch.as_tensor(target, dtype=torch.float32)
    M = torch.as_tensor(mask, dtype=torch.float32)
    n, d = X.shape
    tr, va = _split(n, 0.25, seed)
    torch.manual_seed(seed)
    head = ProbeHead(in_dim=d, num_classes=len(DIMENSIONS), hidden_dim=256, n_layers=2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=g)]
        for s in range(0, len(perm), 64):
            bi = perm[s:s + 64]
            pred = torch.tanh(head(X[bi]))
            m = M[bi]
            loss = ((pred - T[bi]) ** 2 * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pv = torch.tanh(head(X[va]))
    Tv, Mv = T[va], M[va]
    out = {}
    for k, dim in enumerate(DIMENSIONS):
        stance = (Tv[:, k].abs() > DB) & (Mv[:, k] > 0)
        n_ev = int(stance.sum())
        acc = float(((torch.sign(pv[:, k]) == torch.sign(Tv[:, k])) & stance).sum()) / max(1, n_ev)
        out[dim] = (acc, n_ev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--features", default="experiments/calibration/calib_features")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260707)
    a = ap.parse_args()

    d = np.load(f"{a.features}/layer_{a.layer:02d}.npz", allow_pickle=True)
    X, ids = d["X"], d["ids"]
    target, mask = build_clean(f"{a.features}/labels.jsonl", ids)
    print(f"clean-label coverage per dim (frac items with both-judge agreement):")
    cov = mask.mean(0)
    base = json.load(open("experiments/calibration/out/phase5_calibration_results.json"))
    base = {x["layer_index"]: x for x in base["layers"]}[a.layer]

    res = masked_train(X, target, mask, a.epochs, a.seed)
    print(f"\n{'dimension':24} {'baseline':>8} {'clean':>7} {'n_clean':>8} {'verdict':>10}")
    lifted = []
    for dim in DIMENSIONS:
        b = base["per_dim_signacc"][dim]
        c, n = res[dim]
        was = b >= THRESH
        now = c >= THRESH
        verdict = "lifted!" if (now and not was) else ("kept" if now else "still<0.70")
        if now and not was:
            lifted.append(dim)
        print(f"{dim:24} {b:>8.3f} {c:>7.3f} {n:>8} {verdict:>10}  cov={cov[list(DIMENSIONS).index(dim)]:.2f}")
    n_incl = sum(1 for dim in DIMENSIONS if res[dim][0] >= THRESH)
    print(f"\nclean-label @L{a.layer}: {n_incl}/10 dims >= {THRESH}"
          + (f"  (newly lifted: {lifted})" if lifted else "  (no new dims lifted)"))


if __name__ == "__main__":
    main()
