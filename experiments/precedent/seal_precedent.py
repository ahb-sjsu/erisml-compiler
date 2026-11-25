#!/usr/bin/env python3
"""Hash the prereg-precedent-v1 bundle, in the format of geometric-economics seal_learned.py.

A file is in the bundle if changing it would change what the study concludes.

    python seal_precedent.py           # verify against the recorded seal
    python seal_precedent.py --write   # write it
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEAL = os.path.join(HERE, "prereg-precedent-v1.sha256")

BUNDLE = [
    "prereg-precedent-v1.md",  # the registration
    "gates.json",  # the gate registry the readers are asked about
    "read_gates.py",  # the gate reader and its prompt
    "stage0_counts.py",  # the split rule
    "build_cases.py",  # which cases, by hash
    "embed_cases.py",  # the encoder and its snapshot
    "precedent_arms.py",  # coordinates, metric, arms, fixed constants
    "grade_precedent.py",  # every bar
    "run_precedent.py",  # the sealed run, the gate list, the void conditions
    "rehearse.py",  # the planted-truth rehearsal
    "stage0/counts_aita.json",  # units counted before the seal
    "stage0/counts_dearabby.json",
    "stage0/pilot_agreement.json",  # the pilot the gate list was fixed from
    "stage0/cases.sha256.json",  # digest of the cases file
    "rehearsal/round3/results_sim/rehearsal_summary.json",  # rehearsal of the code as sealed
]

META = {
    "prereg": "prereg-precedent-v1",
    "title": "Gates, precedents and a learned metric as predictors of moral judgement",
    "author": "Andrew H. Bond, San Jose State University",
    "frozen_predictions": "H2 and H4 in prereg-precedent-v1.md Section 4; void conditions Section 5. "
    "H1 is not run in v1 (no flip family); H3 is reported only.",
    "data": "Social-Chem-101, AITA and Dear Abby slices. Held-out judgements not read before freezing. "
    "Texts, embeddings and gate readings may be computed before freezing; none reads a judgement.",
    "encoder_snapshot": "836121a0533e5664b21c7aacc5d22951f2b8b25b",
    "hash_note": "sha256 of each file with CRLF normalized to LF; combined = sha256 over the "
    "concatenated component hex digests in bundle_files order",
}


def digest(path: str) -> str:
    data = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def compute() -> dict:
    files = {f: digest(os.path.join(HERE, f)) for f in BUNDLE}
    combined = hashlib.sha256("".join(files[f] for f in BUNDLE).encode()).hexdigest()
    cases = json.load(open(os.path.join(HERE, "stage0", "cases.sha256.json"), encoding="utf-8"))
    return {
        **META,
        "cases_sha256": cases["cases_sha256"],
        "bundle_files": BUNDLE,
        "files": files,
        "combined_sha256": combined,
    }


def main() -> None:
    cur = compute()
    if "--write" in sys.argv:
        with open(SEAL, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cur, f, indent=2)
            f.write("\n")
        print("sealed", cur["combined_sha256"])
        return
    rec = json.load(open(SEAL, encoding="utf-8"))
    ok = True
    for f in BUNDLE:
        good = rec["files"].get(f) == cur["files"][f]
        ok &= good
        print(f"    {f:<58} {'ok' if good else 'CHANGED'}")
    print("  SEAL INTACT" if ok and rec["combined_sha256"] == cur["combined_sha256"] else "  SEAL BROKEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
