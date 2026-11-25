"""Stage 0: choose the cases from the Scruples Anecdotes. Writes texts only.

Reads `id`, `title` and `text` of each anecdote and nothing else. No vote count is read here.

Roles, fixed by the dataset's own splits and by hashes of the anecdote id, so nobody chooses a case:
    train     training split, the N_TRAIN lowest by sha256("train|" + id)
    ordinary  test split, the N_ORDINARY lowest by sha256("ordinary|" + id)

The dev split is not used.

With --count-train it also reports the judgement scale on the chosen TRAINING cases only
(the judgement is the share of votes finding the author in the wrong).

Usage: python build_cases.py --data <dir with *.scruples-anecdotes.jsonl> --out cases.jsonl [--count-train]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

N_TRAIN, N_ORDINARY = 4000, 1500
MIN_VOTES = 5  # applied when judgements are read (run_precedent.py), never here


def texts(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            t = (d["title"] + "\n" + d["text"]).strip()
            if t:
                out[d["id"]] = t
    return out


def lowest(ids, salt: str, n: int) -> list[str]:
    return sorted(ids, key=lambda s: hashlib.sha256((salt + "|" + s).encode()).hexdigest())[:n]


def judgement(d: dict) -> tuple[float, int] | None:
    s = d["binarized_label_scores"]
    n = s["RIGHT"] + s["WRONG"]
    return (s["WRONG"] / n, n) if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count-train", action="store_true")
    a = ap.parse_args()
    train = texts(os.path.join(a.data, "train.scruples-anecdotes.jsonl"))
    test = texts(os.path.join(a.data, "test.scruples-anecdotes.jsonl"))
    if set(train) & set(test):
        raise SystemExit("VOID V1: an anecdote id in both splits")
    roles = {"train": (train, lowest(train, "train", N_TRAIN)), "ordinary": (test, lowest(test, "ordinary", N_ORDINARY))}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for role, (src, ids) in roles.items():
            for s in ids:
                fh.write(json.dumps({"id": s, "role": role, "text": src[s]}) + "\n")
    report = {
        "corpus": "Scruples Anecdotes v1.0",
        "available": {"train": len(train), "test": len(test)},
        "cases": {r: len(ids) for r, (_, ids) in roles.items()},
        "cases_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
    }
    if a.count_train:
        chosen = set(roles["train"][1])
        ys, ns = [], []
        with open(os.path.join(a.data, "train.scruples-anecdotes.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d["id"] in chosen and (j := judgement(d)):
                    ys.append(j[0])
                    ns.append(j[1])
        kept = [y for y, n in zip(ys, ns) if n >= MIN_VOTES]
        mean = sum(kept) / len(kept)
        report["train_side_only"] = {
            "with_votes": len(ys),
            "with_at_least_5_votes": len(kept),
            "median_votes": sorted(ns)[len(ns) // 2],
            "distinct_judgement_values": len({round(y, 3) for y in kept}),
            "judgement_mean": round(mean, 4),
            "judgement_variance": round(sum((y - mean) ** 2 for y in kept) / len(kept), 4),
            "share_majority_wrong": round(sum(y > 0.5 for y in kept) / len(kept), 4),
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
