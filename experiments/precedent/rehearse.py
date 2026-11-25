"""Rehearsal of the precedent experiment on planted truths.

Known truths are planted on REAL encoder embeddings with the labels overwritten, as in the
prereg-learned-v1 rehearsals. The chain (coordinates -> metric -> arms -> grader) must read
back which truth was planted. Every output is written to results_sim_*.json and is marked
simulated; the grader gives a simulated input a rehearsal reading, never a grade.

Worlds (what generates the judgement y):
    null          noise only
    smooth_only   one smooth trade-off, gates irrelevant
    gates_only    a constant per stratum
    gates_pooled  constant per stratum + one trade-off shared by all strata
    gates_local   constant per stratum + a different trade-off in each stratum (3 coordinates
                  each, drawn from the same 16, so one pooled metric of rank 8 nearly covers them)
    gates_local_strong   as gates_local, with 5 coordinates per stratum, disjoint across strata
                  (20 coordinates in all, more than the pooled metric's rank)

Held-out families:
    tradeoff      ordinary held-out cases
    flips         a held-out case with one gate bit flipped and its embedding barely moved
    crossdomain   a whole region of embedding space absent from training (reported, not graded)

Self-tests (the suite must show it can fail):
    S1 split guard     a sampling unit on both sides of the split must be refused
    S2 provenance      the grader must refuse an undeclared source
    S3 relabelling     renaming the strata must leave every arm's errors unchanged
    S4 negation        negating y must leave every arm's squared errors unchanged
    S5 nonvacuity      on the null world no hypothesis may pass more often than chance allows

Usage:  python rehearse.py --emb /path/labse.npz --out results_sim --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from grade_precedent import HYPOTHESES, grade
from precedent_arms import ARMS, K_NEIGHBOURS, Cases, Fitted

WORLDS = ("null", "smooth_only", "gates_only", "gates_pooled", "gates_local", "gates_local_strong")
EXPECTED = {  # (H1_gates, H2_metric, H3_local)
    "null": (False, False, False),
    "smooth_only": (False, True, False),
    "gates_only": (True, False, False),
    "gates_pooled": (True, True, False),
    "gates_local": (True, True, True),
    "gates_local_strong": (True, True, True),
}
N_SAMPLE, N_TRAIN, N_TRADE, N_FLIP = 6000, 3000, 800, 600
NOISE_SD, SMOOTH_SD = 0.5, 0.7
FLIP_SHIFT = 0.2  # how far a flip moves the embedding along the gate direction, in sd units


class World:
    """A planted truth. Coordinates of the truth belong to the world, not to any split."""

    def __init__(self, e: np.ndarray, kind: str, seed: int):
        rng = np.random.default_rng(seed)
        self.kind, self.rng = kind, rng
        self.mu = e.mean(0)
        _, sv, vt = np.linalg.svd(e - self.mu, full_matrices=False)
        self.vt, self.sd = vt[:48], sv[:48] / np.sqrt(len(e) - 1)
        self.gate_dirs = []
        for _ in range(2):
            v = np.zeros(48)
            v[:16] = rng.normal(size=16)
            self.gate_dirs.append(v / np.linalg.norm(v))
        self.level = rng.permutation(np.array([-1.5, -0.5, 0.5, 1.5]))
        self.slopes = []
        for _ in range(4):
            b = np.zeros(48)
            b[rng.choice(16, size=3, replace=False)] = rng.normal(size=3)
            self.slopes.append(SMOOTH_SD * b / np.linalg.norm(b))
        self.slopes_strong = []
        disjoint = rng.permutation(24)[:20].reshape(4, 5)
        for dims in disjoint:
            b = np.zeros(48)
            b[dims] = rng.normal(size=5)
            self.slopes_strong.append(SMOOTH_SD * b / np.linalg.norm(b))

    def z(self, e: np.ndarray) -> np.ndarray:
        return ((e - self.mu) @ self.vt.T) / self.sd

    def gates(self, e: np.ndarray) -> np.ndarray:
        z = self.z(e)
        bits = [(z @ v + 0.3 * self.rng.normal(size=len(e)) > 0).astype(int) for v in self.gate_dirs]
        return bits[0] + 2 * bits[1]

    def judge(self, e: np.ndarray, s: np.ndarray) -> np.ndarray:
        z = self.z(e)
        y = NOISE_SD * self.rng.normal(size=len(e))
        if self.kind == "null":
            return y
        if self.kind != "smooth_only":
            y = y + self.level[s]
        if self.kind in ("smooth_only", "gates_pooled"):
            y = y + z @ self.slopes[0]
        if self.kind == "gates_local":
            y = y + np.einsum("ij,ij->i", z, np.stack(self.slopes)[s])
        if self.kind == "gates_local_strong":
            y = y + np.einsum("ij,ij->i", z, np.stack(self.slopes_strong)[s])
        return y

    def flip(self, e: np.ndarray, s: np.ndarray):
        """One gate bit flipped per case; the embedding moves FLIP_SHIFT sd along that gate."""
        k = self.rng.integers(0, 2, size=len(e))
        bit = (s >> k) & 1
        s2 = s ^ (1 << k)
        step = np.stack(self.gate_dirs)[k] * self.sd * FLIP_SHIFT * np.where(bit == 1, -1.0, 1.0)[:, None]
        e2 = e + step @ self.vt
        return e2 / np.linalg.norm(e2, axis=1, keepdims=True), s2


def check_split(train_units: np.ndarray, held_units: np.ndarray) -> None:
    both = np.intersect1d(train_units, held_units)
    if both.size:
        raise ValueError(f"VOID V1: {both.size} sampling unit(s) on both sides of the split")


def build(e_all: np.ndarray, kind: str, seed: int):
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(1000 + seed)
    pick = rng.choice(len(e_all), size=N_SAMPLE, replace=False)
    e = e_all[pick]
    world = World(e, kind, seed)
    unit = pick.copy()  # each sampled text is its own sampling unit in the rehearsal
    region = KMeans(n_clusters=8, n_init=4, random_state=seed).fit_predict(world.z(e)[:, :16])
    away = region == 0
    home = np.flatnonzero(~away)
    rng.shuffle(home)
    i_tr, i_td, i_fl = np.split(home[: N_TRAIN + N_TRADE + N_FLIP], [N_TRAIN, N_TRAIN + N_TRADE])
    s = world.gates(e)
    y = world.judge(e, s)
    train = Cases(e[i_tr], s[i_tr], y[i_tr], unit[i_tr])
    fams = {
        "tradeoff": Cases(e[i_td], s[i_td], y[i_td], unit[i_td]),
        "crossdomain": Cases(e[away], s[away], y[away], unit[away]),
    }
    e2, s2 = world.flip(e[i_fl], s[i_fl])
    fams["flips"] = Cases(e2, s2, world.judge(e2, s2), unit[i_fl])
    for f in fams.values():
        check_split(train.cluster, f.cluster)
    return train, fams, world


def run(train: Cases, fams: dict, seed: int):
    fit = Fitted(train, seed=seed)
    errors = {f: {a: ((p - c.y) ** 2) for a, p in fit.predict(c).items()} for f, c in fams.items()}
    clusters = {f: c.cluster for f, c in fams.items()}
    return fit, errors, clusters


def paraphrase_stability(fit: Fitted, q: Cases, world: World, seed: int) -> dict:
    """Jaccard@k of the retrieved set before and after a wording change.

    The simulated wording change moves the embedding along directions the truth does not use
    (the trailing world coordinates). Reported for cosine and for the learned pooled metric.
    """
    import torch

    rng = np.random.default_rng(7000 + seed)
    step = np.zeros((len(q.e), 48))
    step[:, 24:] = rng.normal(size=(len(q.e), 24))
    e2 = q.e + (step * world.sd) @ world.vt
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)

    def cos_ids(x):
        d = torch.cdist(torch.tensor(x, dtype=torch.float32), torch.tensor(fit.train.e, dtype=torch.float32))
        return d.topk(K_NEIGHBOURS, largest=False).indices.numpy()

    def jac(a, b):
        return float(np.mean([len(set(r) & set(t)) / len(set(r) | set(t)) for r, t in zip(a, b)]))

    return {
        "jaccard_cosine": jac(cos_ids(q.e), cos_ids(e2)),
        "jaccard_learned_metric": jac(fit.retrieved_ids(q.e), fit.retrieved_ids(e2)),
    }


def selftests(e_all: np.ndarray) -> dict:
    out = {}
    # S1: the split guard must speak, and for the right reason
    try:
        check_split(np.array([1, 2, 3]), np.array([3, 4]))
        out["S1_split_guard_fires"] = False
    except ValueError as err:
        out["S1_split_guard_fires"] = "VOID V1" in str(err)
    # S2: the grader must refuse an undeclared source, and name the source
    try:
        grade({}, {}, source="pilot")
        out["S2_provenance_refusal"] = False
    except ValueError as err:
        out["S2_provenance_refusal"] = "REFUSED: source" in str(err)
    # S3, S4: metamorphic invariance on one world
    train, fams, _ = build(e_all, "gates_local", seed=0)
    _, base, _ = run(train, fams, seed=0)
    perm = np.array([2, 0, 3, 1])
    tr2 = Cases(train.e, perm[train.s], train.y, train.cluster)
    fm2 = {f: Cases(c.e, perm[c.s], c.y, c.cluster) for f, c in fams.items()}
    _, rel, _ = run(tr2, fm2, seed=0)
    tr3 = Cases(train.e, train.s, -train.y, train.cluster)
    fm3 = {f: Cases(c.e, c.s, -c.y, c.cluster) for f, c in fams.items()}
    _, neg, _ = run(tr3, fm3, seed=0)

    def worst(other):
        return max(
            abs(float(np.mean(base[f][a])) - float(np.mean(other[f][a]))) / (float(np.mean(base[f][a])) + 1e-12)
            for f in base
            for a in ARMS
        )

    out["S3_relabel_max_rel_change"] = worst(rel)
    out["S4_negation_max_rel_change"] = worst(neg)
    out["S3_relabel_invariant"] = out["S3_relabel_max_rel_change"] < 0.02
    out["S4_negation_invariant"] = out["S4_negation_max_rel_change"] < 0.02
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", default="results_sim")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--null-seeds", type=int, default=20)
    ap.add_argument("--worlds", default=",".join(WORLDS))
    ap.add_argument("--skip-selftests", action="store_true")
    a = ap.parse_args()
    worlds = tuple(w for w in a.worlds.split(",") if w in WORLDS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e_all = np.load(a.emb)["vecs"].astype(np.float64)
    t0 = time.time()
    summary = {"simulation": True, "embedding_file": a.emb, "n_texts": int(len(e_all)), "worlds": {}}
    summary["selftests"] = {} if a.skip_selftests else selftests(e_all)
    for kind in worlds:
        rows = []
        for seed in range(a.null_seeds if kind == "null" else a.seeds):
            train, fams, world = build(e_all, kind, seed)
            fit, errors, clusters = run(train, fams, seed)
            g = grade(errors, clusters, source="simulated")
            g["world"], g["seed"] = kind, seed
            g["strata_sizes_train"] = np.bincount(train.s, minlength=4).tolist()
            g["paraphrase"] = paraphrase_stability(fit, fams["tradeoff"], world, seed)
            got = tuple(g["verdicts"][h]["passes"] for h in ("H1_gates", "H2_metric", "H3_local"))
            g["expected"], g["recovered"] = list(EXPECTED[kind]), bool(got == EXPECTED[kind])
            g["V5_expected_signal"] = kind != "null"
            (out / f"results_sim_{kind}_s{seed}.json").write_text(json.dumps(g, indent=1))
            rows.append(g)
            print(kind, seed, g["rehearsal_reading"], "recovered" if g["recovered"] else "MISSED",
                  {h: round(g["verdicts"][h]["z"], 1) for h, *_ in HYPOTHESES}, flush=True)
        summary["worlds"][kind] = {
            "runs": len(rows),
            "recovered": sum(r["recovered"] for r in rows),
            "passes": {h: sum(r["verdicts"][h]["passes"] for r in rows) for h, *_ in HYPOTHESES},
            "V5_signal_runs": sum(bool(r.get("V5_signal")) for r in rows),
            "z": {h: [round(r["verdicts"][h]["z"], 2) for r in rows] for h, *_ in HYPOTHESES},
            "mean_sq_error": {
                f: {arm: round(float(np.mean([r["mean_sq_error"][f][arm] for r in rows])), 4) for arm in ARMS}
                for f in ("tradeoff", "flips", "crossdomain")
            },
            "paraphrase_jaccard": {
                k: round(float(np.mean([r["paraphrase"][k] for r in rows])), 3)
                for k in ("jaccard_cosine", "jaccard_learned_metric")
            },
        }
    if "null" in summary["worlds"]:
        n0 = summary["worlds"]["null"]
        summary["selftests"]["S5_null_false_passes"] = n0["passes"]
        summary["selftests"]["S5_nonvacuous"] = all(v <= 2 for v in n0["passes"].values())
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "rehearsal_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "worlds"}, indent=1))
    for kind, w in summary["worlds"].items():
        print(kind, f"recovered {w['recovered']}/{w['runs']}", w["passes"])


if __name__ == "__main__":
    main()
