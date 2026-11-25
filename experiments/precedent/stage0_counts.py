"""Stage 0, step 1: count units and check the judgement scale. No held-out judgement is read.

The split is fixed by a hash of the situation id, so nobody chooses it:
    bucket = int(sha256(situation-short-id)[:8], 16) % 100
    bucket < 80   case base (training and development)
    bucket >= 80  held out
Judgement statistics are computed on the case-base side only. For the held-out side this
script reports the number of situations and nothing about their judgements.

Usage: python stage0_counts.py --tsv social-chem-101.amitheasshole.tsv --area amitheasshole
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict

HELD_OUT_FROM = 80


def bucket(situation_id: str) -> int:
    return int(hashlib.sha256(situation_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--area", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    csv.field_size_limit(sys.maxsize)
    rows = 0
    judged = defaultdict(list)  # situation id -> judgements, case-base side only
    seen_base, seen_held = set(), set()
    held_with_judgement = set()
    with open(a.tsv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows += 1
            sid = r["situation-short-id"]
            has = r.get("action-moral-judgment", "") not in ("", None)
            if bucket(sid) >= HELD_OUT_FROM:
                seen_held.add(sid)
                if has:
                    held_with_judgement.add(sid)  # presence only; the value is not read
                continue
            seen_base.add(sid)
            if has:
                try:
                    judged[sid].append(float(r["action-moral-judgment"]))
                except ValueError:
                    pass
    means = {s: sum(v) / len(v) for s, v in judged.items()}
    n_rows = sorted(len(v) for v in judged.values())
    distinct = sorted({round(m, 3) for m in means.values()})
    hist = defaultdict(int)
    for m in means.values():
        hist[round(m)] += 1
    out = {
        "area": a.area,
        "rows": rows,
        "split_rule": "sha256(situation-short-id)[:8] % 100 >= 80 is held out",
        "case_base_situations": len(seen_base),
        "case_base_situations_with_judgement": len(means),
        "held_out_situations": len(seen_held),
        "held_out_situations_with_judgement": len(held_with_judgement),
        "judgement_rows_per_situation": {
            "min": n_rows[0] if n_rows else 0,
            "median": n_rows[len(n_rows) // 2] if n_rows else 0,
            "max": n_rows[-1] if n_rows else 0,
            "share_with_one_row": round(sum(1 for n in n_rows if n == 1) / max(1, len(n_rows)), 3),
        },
        "case_base_distinct_judgement_values": len(distinct),
        "case_base_judgement_histogram_rounded": dict(sorted(hist.items())),
        "case_base_judgement_variance": (
            round(sum((m - sum(means.values()) / len(means)) ** 2 for m in means.values()) / len(means), 4)
            if means
            else None
        ),
        "V6_resolves_at_least_5_values": len(distinct) >= 5,
    }
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
