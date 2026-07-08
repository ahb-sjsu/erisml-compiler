"""Atlas-side: label the calibration corpus with signed judges + extract Qwen
activations. Produces the (X, Y) per-layer training data for
`calibration.activation_calibration`.

Runs ON Atlas (needs the NRP token at ~/.llmtoken and a GPU). Uses GPU-1 by
default — GPU-0 hosts artemis-avatar and must not be disturbed. Never reboots
or kills anything; it only loads a model for inference and drops it.

Pipeline:
  1. LABEL  (network, threaded): for each text, query qwen3 + glm-5 via the NRP
     ellm endpoint for a SIGNED 10-dim valence vector; consensus = mean, with an
     inter-judge sign-agreement diagnostic. Written incrementally to labels.jsonl
     so the step is resumable.
  2. EXTRACT (GPU-1): load Qwen2.5-7B-Instruct (bf16), hook every-4th + final
     layer, mean-pool hidden states over tokens -> (D,) per layer, per text.
  3. DUMP: one compressed .npz per layer with X (N,D), Y (N,10 consensus signed),
     ids, n_judges, sign_agreement. These are the inputs to the CPU trainer.

  python experiments/calibration/atlas_label_and_extract.py \
      --texts calib_texts.jsonl --out-dir calib_features --device cuda:1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path

import numpy as np

import signed_rubric as sr  # same directory; kept import-light on purpose

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGES = ["qwen3", "glm-5"]


def load_texts(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def label_corpus(texts: list[dict], token: str, url: str, out_path: Path,
                 max_workers: int = 16) -> dict[str, dict]:
    """Label each text with every judge; resumable via out_path (labels.jsonl).
    Returns {id: {"text","raw":{judge:vec},"Y":[...],"n_judges","sign_agreement"}}.

    All (text, judge) calls share one pool so 5000x2 requests run max_workers-wide,
    not serialised per text. A record is finalised + written as soon as every
    judge for that id has returned (success or failure)."""
    from collections import defaultdict

    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"[label] resuming: {len(done)} already labeled", flush=True)

    todo = [t for t in texts if t["id"] not in done]
    print(f"[label] {len(todo)} to label x {len(JUDGES)} judges "
          f"({len(todo) * len(JUDGES)} calls, {max_workers}-wide)", flush=True)
    if not todo:
        return done

    by_id = {t["id"]: t for t in todo}
    raw: dict[str, dict] = defaultdict(dict)
    remaining: dict[str, int] = {t["id"]: len(JUDGES) for t in todo}
    n_final = 0
    with open(out_path, "a", encoding="utf-8") as fout, \
            cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for t in todo:
            for m in JUDGES:
                futs[ex.submit(sr.score_text, t["text"], m, token, url=url)] = (t["id"], m)
        for fut in cf.as_completed(futs):
            tid, m = futs[fut]
            v = fut.result()
            if v is not None:
                raw[tid][m] = v
            remaining[tid] -= 1
            if remaining[tid] == 0:
                t = by_id[tid]
                Y, agg = sr.consensus(list(raw[tid].values()))
                rec = {"id": tid, "text": t["text"], "raw": raw[tid],
                       "Y": Y, "n_judges": agg["n_judges"], "sign_agreement": agg["sign_agreement"]}
                done[tid] = rec
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_final += 1
                if n_final % 100 == 0:
                    print(f"[label] {n_final}/{len(todo)} finalized", flush=True)
    return done


def extract_activations(items: list[dict], device: str, max_tokens: int):
    """Load Qwen once, return (selected_layers, hidden_dim, {layer: X (N,D)}) with
    row order matching `items`."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      trust_remote_code=False).to(device).eval()
    print(f"[extract] model loaded on {device}", flush=True)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_mods = model.model.layers
    elif hasattr(model, "layers"):
        layer_mods = model.layers
    else:
        raise SystemExit("could not resolve transformer layers")
    total = len(layer_mods)
    selected = list(range(0, total, 4))
    if (total - 1) not in selected:
        selected.append(total - 1)
    hidden_dim = model.config.hidden_size

    captured: dict[int, "torch.Tensor"] = {}
    handles = []

    def mk(idx):
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[idx] = hs.detach().squeeze(0).mean(dim=0).to("cpu", torch.float32)
        return h

    for idx in selected:
        handles.append(layer_mods[idx].register_forward_hook(mk(idx)))

    per_layer = {idx: np.zeros((len(items), hidden_dim), dtype=np.float32) for idx in selected}
    for i, it in enumerate(items):
        captured.clear()
        enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
        with torch.no_grad():
            model(**enc, use_cache=False)
        for idx in selected:
            per_layer[idx][i] = captured[idx].numpy()
        if (i + 1) % 100 == 0:
            print(f"[extract] {i + 1}/{len(items)}", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return selected, hidden_dim, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="calib_texts.jsonl")
    ap.add_argument("--out-dir", default="calib_features")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--token-path", default=os.path.expanduser("~/.llmtoken"))
    ap.add_argument("--url", default=sr.DEFAULT_URL)
    ap.add_argument("--min-judges", type=int, default=1, help="drop items with fewer valid judges")
    ap.add_argument("--workers", type=int, default=16, help="concurrent judge calls")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = Path(a.token_path).read_text().strip()
    texts = load_texts(a.texts)

    labeled = label_corpus(texts, token, a.url, out_dir / "labels.jsonl", max_workers=a.workers)

    # Keep row order deterministic (input order), drop under-labeled items.
    items = [labeled[t["id"]] for t in texts
             if t["id"] in labeled and labeled[t["id"]]["n_judges"] >= a.min_judges]
    print(f"[main] {len(items)}/{len(texts)} items kept (>= {a.min_judges} judges)", flush=True)

    selected, hidden_dim, per_layer = extract_activations(items, a.device, a.max_tokens)

    Y = np.array([it["Y"] for it in items], dtype=np.float32)
    ids = np.array([it["id"] for it in items])
    n_judges = np.array([it["n_judges"] for it in items], dtype=np.int32)
    sign_agreement = np.array(
        [it["sign_agreement"] if it["sign_agreement"] is not None else np.nan for it in items],
        dtype=np.float32)

    for idx in selected:
        np.savez_compressed(
            out_dir / f"layer_{idx:02d}.npz",
            X=per_layer[idx], Y=Y, ids=ids, n_judges=n_judges,
            sign_agreement=sign_agreement, layer_index=idx,
            model_id=MODEL_ID, dimensions=np.array(sr.DIMENSIONS))
    meta = {"model_id": MODEL_ID, "hidden_dim": int(hidden_dim),
            "selected_layers": selected, "n_items": len(items),
            "judges": JUDGES, "label_semantics": "signed_valence_[-1,1]"}
    (out_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[main] wrote {len(selected)} layer files + extract_meta.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
