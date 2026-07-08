"""Phase-5 step 2/3 (Atlas side): extract pooled Qwen activations for every
(scenario, rewrite) capture, save per-layer npz. File+sftp transport — the
Phase-4 stdout-base64 path truncates at this scale.

Input JSON: {"items": [{"scenario": id, "rewrites": [{"name":..,"text":..}, ...]}]}
Output: calib_features-style npz per layer with X (n_caps, D), sids, rewrites.

Runs ON Atlas (GPU-1). Never disturbs GPU-0 (artemis-avatar).

  python atlas_phase5_extract.py --items phase5_items.json --out-dir phase5_caps --device cuda:1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="phase5_items.json")
    ap.add_argument("--out-dir", default="phase5_caps")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    items = json.loads(Path(a.items).read_text(encoding="utf-8"))["items"]
    # Flatten to (sid, rewrite, text) captures, preserving order.
    caps = [(it["scenario"], rw["name"], rw["text"]) for it in items for rw in it["rewrites"]]
    print(f"[extract] {len(items)} scenarios -> {len(caps)} captures", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      trust_remote_code=False).to(a.device).eval()
    print(f"[extract] model on {a.device}", flush=True)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_mods = model.model.layers
    else:
        layer_mods = model.layers
    total = len(layer_mods)
    selected = list(range(0, total, 4))
    if (total - 1) not in selected:
        selected.append(total - 1)
    hidden = model.config.hidden_size

    captured: dict[int, "torch.Tensor"] = {}
    handles = []

    def mk(idx):
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[idx] = hs.detach().squeeze(0).mean(dim=0).to("cpu", torch.float32)
        return h

    for idx in selected:
        handles.append(layer_mods[idx].register_forward_hook(mk(idx)))

    per_layer = {idx: np.zeros((len(caps), hidden), dtype=np.float32) for idx in selected}
    for i, (sid, rw, text) in enumerate(caps):
        captured.clear()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=a.max_tokens).to(a.device)
        with torch.no_grad():
            model(**enc, use_cache=False)
        for idx in selected:
            per_layer[idx][i] = captured[idx].numpy()
        if (i + 1) % 40 == 0:
            print(f"[extract] {i + 1}/{len(caps)}", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sids = np.array([c[0] for c in caps])
    rewrites = np.array([c[1] for c in caps])
    for idx in selected:
        np.savez_compressed(out / f"layer_{idx:02d}.npz", X=per_layer[idx],
                            sids=sids, rewrites=rewrites, layer_index=idx, model_id=MODEL_ID)
    (out / "phase5_extract_meta.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "hidden_dim": int(hidden), "selected_layers": selected,
         "n_captures": len(caps), "n_scenarios": len(items)}, indent=2), encoding="utf-8")
    print(f"[extract] wrote {len(selected)} layer files to {out}", flush=True)


if __name__ == "__main__":
    main()
