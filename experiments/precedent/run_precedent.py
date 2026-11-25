"""The sealed run of prereg-precedent-v1. Held-out judgements are read here and nowhere else.

Refuses to run unless the seal is intact. Order of work:
    1. seal check
    2. cases, embeddings (V4 snapshot), gate readings from two readers
    3. gate agreement on the full data (V2), strata from the primary reader, rare strata merged (V3)
    4. judgements: training split first, then the held-out test split
    5. arms fitted on training cases only, predictions on the held-out family
    6. grade_precedent.grade(..., source="held_out")

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_precedent.py --cases cases.jsonl --emb emb_labse \
        --gates-primary gates_gpt-oss.jsonl --gates-second gates_gemma.jsonl \
        --data <dir with *.scruples-anecdotes.jsonl> --out results_sealed.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import seal_precedent
from build_cases import MIN_VOTES, judgement
from grade_precedent import grade
from precedent_arms import MIN_STRATUM, Cases, Fitted

# Fixed at the seal from the second Stage 0 pilot (300 full training posts, two readers):
# a gate is kept when the readers agreed at kappa >= KAPPA_BAR and the primary reader found it
# in >= 2% of texts. See STAGE0.md.
SEALED_GATES = ("commitment", "abuse")
KAPPA_BAR = 0.6
MAX_UNREAD_SHARE = 0.05  # V8: more unread cases than this in any role voids the run


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    po = float((x == y).mean())
    pa, pb = float(x.mean()), float(y.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_gates(path: str) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("gates") is not None:
            out[d["id"]] = d["gates"]
    return out


def judgements(path: str, wanted: set) -> tuple[dict, int]:
    """Share of votes finding the author in the wrong, for anecdotes with at least MIN_VOTES votes."""
    out, thin = {}, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["id"] not in wanted:
                continue
            j = judgement(d)
            if j is None or j[1] < MIN_VOTES:
                thin += 1
                continue
            out[d["id"]] = j[0]
    return out, thin


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("cases", "emb", "gates-primary", "gates-second", "data", "out"):
        ap.add_argument("--" + name, required=True)
    # --smoke exercises the whole path on TRAINING posts only (both roles take their judgements
    # from the training split). It skips the seal, is graded as "simulated", and must write to a
    # file with "smoke" in its name. It is a check that the code runs, never a result.
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke and "smoke" not in os.path.basename(a.out):
        sys.exit("REFUSED: a smoke run must write to a file with 'smoke' in its name")

    # 1. the seal
    if a.smoke:
        rec = {"combined_sha256": "SMOKE-NOT-SEALED", "encoder_snapshot": None}
    else:
        rec = json.load(open(seal_precedent.SEAL, encoding="utf-8"))
        cur = seal_precedent.compute()
        if rec["combined_sha256"] != cur["combined_sha256"]:
            sys.exit("REFUSED: the seal is broken; the bundle differs from prereg-precedent-v1.sha256")
        if hashlib.sha256(open(a.cases, "rb").read()).hexdigest() != rec["cases_sha256"]:
            sys.exit("REFUSED: cases file differs from the sealed one")
    report = {"prereg": "prereg-precedent-v1", "seal": rec["combined_sha256"], "voids": []}

    # 2. cases, embeddings, gates
    cases = [json.loads(x) for x in open(a.cases, encoding="utf-8")]
    emb = np.load(a.emb + ".npz")
    meta = json.load(open(a.emb + ".meta.json"))
    if not a.smoke and meta["snapshot"] != rec["encoder_snapshot"]:
        sys.exit(f"VOID V4: encoder snapshot {meta['snapshot']} is not the sealed one")
    if list(emb["ids"]) != [c["id"] for c in cases]:
        sys.exit("REFUSED: embedding rows do not match the cases file")
    vec = {c["id"]: emb["vecs"][i].astype(np.float64) for i, c in enumerate(cases)}
    g1, g2 = load_gates(a.gates_primary), load_gates(a.gates_second)
    role_ids = defaultdict(list)
    for c in cases:
        role_ids[c["role"]].append(c["id"])
    if set(role_ids["train"]) & set(role_ids["ordinary"]):
        sys.exit("VOID V1: a unit on both sides of the split")
    unread = {r: 1 - sum(i in g1 for i in ids) / len(ids) for r, ids in role_ids.items()}
    report["unread_share_primary"] = {r: round(u, 4) for r, u in unread.items()}
    if max(unread.values()) > MAX_UNREAD_SHARE:
        report["voids"].append("V8: too many cases with no gate reading")

    # 3. agreement on the full data, strata
    both = [i for c in cases if (i := c["id"]) in g1 and i in g2]
    names = list(next(iter(g1.values())))
    report["gate_kappa_full_all_gates"] = {
        g: round(kappa(np.array([g1[i][g] for i in both]), np.array([g2[i][g] for i in both])), 3) for g in names
    }
    report["gate_prevalence_primary"] = {g: round(float(np.mean([g1[i][g] for i in g1])), 4) for g in names}
    kept = [g for g in SEALED_GATES if report["gate_kappa_full_all_gates"][g] >= KAPPA_BAR]
    report["gates_sealed"], report["gates_kept"] = list(SEALED_GATES), kept
    if len(kept) < 2:
        report["voids"].append("V2: fewer than two gates read reliably on the full data")

    def pattern(i):
        return "".join(str(g1[i][g]) for g in kept)

    train_ids = [i for i in role_ids["train"] if i in g1]
    counts = Counter(pattern(i) for i in train_ids)
    big = sorted(p for p, n in counts.items() if n >= MIN_STRATUM)
    index = {p: k + 1 for k, p in enumerate(big)}  # 0 is "other": rare or unseen patterns (V3)
    report["strata"] = {
        "patterns": {p: counts[p] for p in big},
        "merged_into_other": sum(n for p, n in counts.items() if p not in index),
    }

    # 4. judgements: training split, then held-out
    def build(ids, y):
        ids = [i for i in ids if i in y and i in g1]
        return Cases(
            np.stack([vec[i] for i in ids]),
            np.array([index.get(pattern(i), 0) for i in ids]),
            np.array([y[i] for i in ids]),
            np.array(ids),
        )

    y_train, thin_train = judgements(os.path.join(a.data, "train.scruples-anecdotes.jsonl"), set(train_ids))
    train = build(train_ids, y_train)
    report["n_train"], report["train_below_min_votes"] = int(len(train.y)), thin_train
    report["train_judgement_distinct_values"] = int(len(np.unique(np.round(train.y, 3))))
    if report["train_judgement_distinct_values"] < 5:
        report["voids"].append("V6: judgement scale too coarse")
    held_file = "train.scruples-anecdotes.jsonl" if a.smoke else "test.scruples-anecdotes.jsonl"
    y_held, thin_held = judgements(os.path.join(a.data, held_file), set(role_ids["ordinary"]))
    fams = {"tradeoff": build(role_ids["ordinary"], y_held)}
    report["n_held_out"], report["held_out_below_min_votes"] = int(len(fams["tradeoff"].y)), thin_held

    # 5, 6. fit, predict, grade
    fit = Fitted(train, seed=0)
    errors = {f: {arm: (p - c.y) ** 2 for arm, p in fit.predict(c).items()} for f, c in fams.items()}
    clusters = {f: c.cluster for f, c in fams.items()}
    g = grade(errors, clusters, source="simulated" if a.smoke else "held_out")
    if a.smoke:
        g["diagnosis"] = "SMOKE RUN ON TRAINING POSTS, NOT A RESULT (" + g.pop("rehearsal_reading") + ")"
    elif report["voids"]:
        g["diagnosis"] = "VOID: " + "; ".join(report["voids"])
    report["pending"] = (
        "V2 human spot-check of 200 cases (agreement >= 0.8) is the owner's to do; "
        "until then the result is provisional"
    )
    report["grade"] = g
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "grade"}, indent=1))
    print("diagnosis:", g["diagnosis"])
    for h, v in g["verdicts"].items():
        print(" ", h, v.get("compares"), "not run" if v.get("not_run") else f"z={v['z']:.2f} passes={v['passes']}")
    print(json.dumps(g["mean_sq_error"], indent=1))


if __name__ == "__main__":
    main()
