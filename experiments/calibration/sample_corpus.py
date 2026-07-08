"""Sample a calibration corpus from Social Chemistry 101 (host-side).

Reads the in-repo Social Chem 101 TSVs (CC-BY 4.0), takes a deterministic
length-filtered sample of distinct situations, and writes:
  - calib_texts.jsonl        : {"id", "area", "text"} per line (input to labeling)
  - calib_corpus.fingerprint.json : reproducible corpus provenance (hash + counts)

The situations are the source texts the Atlas job will (a) send to the signed
judges and (b) feed through Qwen to capture activations. Kept host-side because
the TSVs live here; only the small jsonl needs to travel to Atlas.

  python experiments/calibration/sample_corpus.py --n 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from erisml_compiler.social_chem.loader import fingerprint_corpus, load_situations

DEFAULT_TSVS = [
    "data/social-chem-101/social-chem-101.dearabby.tsv",
    "data/social-chem-101/social-chem-101.amitheasshole.tsv",
]


def _stable_key(text: str, seed: int) -> str:
    """Deterministic per-text hash for seeded sampling (no Random() state)."""
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def sample(tsvs: list[str], n: int, seed: int, min_chars: int, max_chars: int):
    seen: set[str] = set()
    pool: list[dict] = []
    used_situations = []
    for tsv in tsvs:
        p = Path(tsv)
        if not p.exists():
            print(f"[warn] missing {tsv}, skipping")
            continue
        sits = load_situations(tsv)
        used_situations.extend(sits)
        for s in sits:
            t = s.situation.strip()
            if not (min_chars <= len(t) <= max_chars):
                continue
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            pool.append({"id": s.situation_short_id, "area": s.area, "text": t})
    # Deterministic sample: sort by a seeded content hash, take the first n.
    pool.sort(key=lambda r: _stable_key(r["text"], seed))
    chosen = pool[:n]
    fp = fingerprint_corpus(used_situations, source="social-chem-101")
    return chosen, pool, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="append", default=None, help="repeatable; default = both slices")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--out-dir", default="experiments/calibration")
    a = ap.parse_args()

    tsvs = a.tsv or DEFAULT_TSVS
    chosen, pool, fp = sample(tsvs, a.n, a.seed, a.min_chars, a.max_chars)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts_path = out_dir / "calib_texts.jsonl"
    with open(texts_path, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "source": "social-chem-101",
        "license": "CC-BY 4.0",
        "citation": fp.citation,
        "tsvs": tsvs,
        "seed": a.seed,
        "min_chars": a.min_chars,
        "max_chars": a.max_chars,
        "n_requested": a.n,
        "n_available_after_filter": len(pool),
        "n_written": len(chosen),
        "corpus_canonical_sha256": fp.canonical_sha256,
        "sample_sha256": hashlib.sha256(
            "\n".join(r["text"] for r in chosen).encode("utf-8")
        ).hexdigest(),
    }
    (out_dir / "calib_corpus.fingerprint.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {texts_path}  ({len(chosen)} texts of {len(pool)} available)")
    print(f"  sample_sha256 = {meta['sample_sha256'][:16]}...")
    if chosen:
        print(f"  sample text: {chosen[0]['text'][:90]!r}")


if __name__ == "__main__":
    main()
