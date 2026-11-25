"""Stage 0, step 3: embed the case texts with one frozen encoder. Reads no judgement.

Writes <out>.npz (`vecs` float32 unit norm, `ids`) and <out>.meta.json with the encoder's
resolved snapshot. A run whose snapshot differs from the sealed one is void (V4).

Usage: CUDA_VISIBLE_DEVICES=1 python embed_cases.py --cases cases.jsonl --model labse --out emb_labse
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODELS = {"labse": "sentence-transformers/LaBSE", "bgem3": "BAAI/bge-m3"}
SEALED_SNAPSHOT = {"labse": "836121a0533e5664b21c7aacc5d22951f2b8b25b"}


def snapshot(name: str) -> str:
    root = os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--"))
    ref = os.path.join(root, "refs", "main")
    return open(ref).read().strip() if os.path.exists(ref) else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    cases = [json.loads(x) for x in open(a.cases, encoding="utf-8")]
    name = MODELS[a.model]
    snap = snapshot(name)
    want = SEALED_SNAPSHOT.get(a.model)
    if want and snap != want:
        raise SystemExit(f"VOID V4: {name} snapshot {snap} is not the sealed {want}")
    enc = SentenceTransformer(name, device="cuda")
    vecs = enc.encode([c["text"] for c in cases], batch_size=256, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float32)
    np.savez(a.out + ".npz", vecs=vecs, ids=np.array([c["id"] for c in cases]))
    meta = {"model": name, "snapshot": snap, "n_texts": len(cases), "dim": int(vecs.shape[1])}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
