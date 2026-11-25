"""Stage 0, step 2: choose the cases. Writes texts only. No judgement is written or read here.

Roles (all fixed by hashes of the situation id, so nobody chooses a case):
    train        AITA, case-base side (bucket < 80), the N_TRAIN lowest by sha256("train|" + id)
    ordinary     AITA, held-out side (bucket >= 80), the N_ORDINARY lowest by sha256("ordinary|" + id)
    crossdomain  Dear Abby, held-out side, the N_CROSS lowest by sha256("cross|" + id)

A situation needs at least one judged row to be eligible. That is a fact about presence, read
from whether the judgement field is empty, never from its value.

Usage: python build_cases.py --aita <tsv> --dearabby <tsv> --out cases.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

from stage0_counts import HELD_OUT_FROM, bucket

N_TRAIN, N_ORDINARY, N_CROSS = 6000, 1500, 1500


def eligible(tsv: str) -> dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    text, judged = {}, set()
    with open(tsv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            sid = r["situation-short-id"]
            text[sid] = r["situation"]
            if r.get("action-moral-judgment", "") not in ("", None):
                judged.add(sid)
    return {s: t for s, t in text.items() if s in judged and t.strip()}


def lowest(ids, salt: str, n: int) -> list[str]:
    return sorted(ids, key=lambda s: hashlib.sha256((salt + "|" + s).encode()).hexdigest())[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aita", required=True)
    ap.add_argument("--dearabby", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    aita, abby = eligible(a.aita), eligible(a.dearabby)
    roles = {
        "train": (aita, lowest([s for s in aita if bucket(s) < HELD_OUT_FROM], "train", N_TRAIN)),
        "ordinary": (aita, lowest([s for s in aita if bucket(s) >= HELD_OUT_FROM], "ordinary", N_ORDINARY)),
        "crossdomain": (abby, lowest([s for s in abby if bucket(s) >= HELD_OUT_FROM], "cross", N_CROSS)),
    }
    n = {}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for role, (src, ids) in roles.items():
            n[role] = len(ids)
            for s in ids:
                fh.write(json.dumps({"id": s, "role": role, "text": src[s]}) + "\n")
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"cases": n, "cases_sha256": digest}, indent=1))


if __name__ == "__main__":
    main()
