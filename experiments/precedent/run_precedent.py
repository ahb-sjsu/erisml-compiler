"""The sealed run of prereg-precedent-v1. Held-out judgements are read here and nowhere else.

Refuses to run unless the seal is intact. Order of work:
    1. seal check
    2. cases, embeddings (V4 snapshot), gate readings from two readers
    3. gate agreement on the full data (V2), strata from the primary reader, rare strata merged (V3)
    4. judgements: training side first, then the held-out side
    5. arms fitted on training cases only, predictions on held-out families
    6. grade_precedent.grade(..., source="held_out")

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_precedent.py --cases cases.jsonl --emb emb_labse \
        --gates-primary gates_gpt-oss.jsonl --gates-second gates_gemma.jsonl \
        --aita <tsv> --dearabby <tsv> --out results_sealed.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import seal_precedent
from grade_precedent import grade
from precedent_arms import MIN_STRATUM, Cases, Fitted

# Fixed at the seal from the Stage 0 pilot (300 case-base texts, two readers):
# a gate is kept when the readers agreed at kappa >= KAPPA_BAR and it was present in >= 2% of texts.
SEALED_GATES = ("no_consent", "commitment", "coercion")  # PLACEHOLDER: not sealed; the first pilot passed only "commitment" (STAGE0.md)
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


def judgements(tsv: str, wanted: set) -> dict:
    csv.field_size_limit(sys.maxsize)
    acc = defaultdict(list)
    with open(tsv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            sid = r["situation-short-id"]
            if sid in wanted and r.get("action-moral-judgment", "") not in ("", None):
                try:
                    acc[sid].append(float(r["action-moral-judgment"]))
                except ValueError:
                    pass
    return {s: sum(v) / len(v) for s, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("cases", "emb", "gates-primary", "gates-second", "aita", "dearabby", "out"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()

    # 1. the seal
    rec = json.load(open(seal_precedent.SEAL, encoding="utf-8"))
    cur = seal_precedent.compute()
    if rec["combined_sha256"] != cur["combined_sha256"]:
        sys.exit("REFUSED: the seal is broken; the bundle differs from prereg-precedent-v1.sha256")
    cases_digest = hashlib.sha256(open(a.cases, "rb").read()).hexdigest()
    if cases_digest != rec["cases_sha256"]:
        sys.exit("REFUSED: cases file differs from the sealed one")
    report = {"prereg": "prereg-precedent-v1", "seal": rec["combined_sha256"], "voids": [], "notes": []}

    # 2. cases, embeddings, gates
    cases = [json.loads(x) for x in open(a.cases, encoding="utf-8")]
    emb = np.load(a.emb + ".npz")
    meta = json.load(open(a.emb + ".meta.json"))
    if meta["snapshot"] != rec["encoder_snapshot"]:
        sys.exit(f"VOID V4: encoder snapshot {meta['snapshot']} is not the sealed one")
    if list(emb["ids"]) != [c["id"] for c in cases]:
        sys.exit("REFUSED: embedding rows do not match the cases file")
    vec = {c["id"]: emb["vecs"][i].astype(np.float64) for i, c in enumerate(cases)}
    g1, g2 = load_gates(a.gates_primary), load_gates(a.gates_second)
    role_ids = defaultdict(list)
    for c in cases:
        role_ids[c["role"]].append(c["id"])
    unread = {r: 1 - sum(i in g1 for i in ids) / len(ids) for r, ids in role_ids.items()}
    report["unread_share_primary"] = {r: round(u, 4) for r, u in unread.items()}
    if max(unread.values()) > MAX_UNREAD_SHARE:
        report["voids"].append("V8: too many cases with no gate reading")

    # 3. agreement, strata
    both = [i for c in cases if (i := c["id"]) in g1 and i in g2]
    kap = {g: kappa(np.array([g1[i][g] for i in both]), np.array([g2[i][g] for i in both])) for g in SEALED_GATES}
    report["gate_kappa_full"] = {g: round(k, 3) for g, k in kap.items()}
    report["gate_prevalence_primary"] = {
        g: round(float(np.mean([g1[i][g] for i in g1])), 4) for g in SEALED_GATES
    }
    kept = [g for g in SEALED_GATES if kap[g] >= KAPPA_BAR]
    report["gates_kept"] = kept
    if len(kept) < 2:
        report["voids"].append("V2: fewer than two gates read reliably")

    def pattern(i):
        return "".join(str(g1[i][g]) for g in kept)

    train_ids = [i for i in role_ids["train"] if i in g1]
    counts = Counter(pattern(i) for i in train_ids)
    big = sorted(p for p, n in counts.items() if n >= MIN_STRATUM)
    index = {p: k + 1 for k, p in enumerate(big)}  # 0 is "other": rare or unseen patterns (V3)
    report["strata"] = {"patterns": {p: counts[p] for p in big}, "merged_into_other": sum(n for p, n in counts.items() if p not in index)}

    # 4. judgements: training side, then held-out
    y_train = judgements(a.aita, set(train_ids))
    train_ids = [i for i in train_ids if i in y_train]
    held = {"tradeoff": (a.aita, "ordinary"), "crossdomain": (a.dearabby, "crossdomain")}

    def build(ids, y):
        ids = [i for i in ids if i in y and i in g1]
        return Cases(
            np.stack([vec[i] for i in ids]),
            np.array([index.get(pattern(i), 0) for i in ids]),
            np.array([y[i] for i in ids]),
            np.array(ids),
        )

    train = build(train_ids, y_train)
    report["n_train"] = int(len(train.y))
    report["train_judgement_distinct_values"] = int(len(np.unique(np.round(train.y, 3))))
    if report["train_judgement_distinct_values"] < 5:
        report["voids"].append("V6: judgement scale too coarse")

    fams = {}
    for fam, (tsv, role) in held.items():
        ids = role_ids[role]
        if set(ids) & set(train_ids):
            sys.exit("VOID V1: a unit on both sides of the split")
        fams[fam] = build(ids, judgements(tsv, set(ids)))
    report["n_held_out"] = {f: int(len(c.y)) for f, c in fams.items()}

    # 5, 6. fit, predict, grade
    fit = Fitted(train, seed=0)
    errors = {f: {arm: (p - c.y) ** 2 for arm, p in fit.predict(c).items()} for f, c in fams.items()}
    clusters = {f: c.cluster for f, c in fams.items()}
    g = grade(errors, clusters, source="held_out")
    if report["voids"]:
        g["diagnosis"] = "VOID: " + "; ".join(report["voids"])
    report["pending"] = "V2 human spot-check of 200 cases (agreement >= 0.8) is the owner's to do; until then the result is provisional"
    report["grade"] = g
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "grade"}, indent=1))
    print("diagnosis:", g["diagnosis"])
    for h, v in g["verdicts"].items():
        print(" ", h, v.get("compares"), "not run" if v.get("not_run") else f"z={v['z']:.2f} passes={v['passes']}")
    print(json.dumps(g["mean_sq_error"], indent=1))


if __name__ == "__main__":
    main()
