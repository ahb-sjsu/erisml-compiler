"""Embed the TRAINING posts of the sealed cases file with BGE-M3 (dense CLS vector), for the signal check."""
import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
run, out = sys.argv[1], sys.argv[2]
cases = [json.loads(x) for x in open(run + "/cases.jsonl", encoding="utf-8") ]
cases = [c for c in cases if c["role"] == "train"]
texts = [((c.get("title") or "") + "\n" + (c.get("text") or "")).strip() for c in cases]
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
order = np.argsort([len(t) for t in texts]); vecs = np.zeros((len(texts), 1024), dtype=np.float32)
with torch.no_grad():
    for s in range(0, len(texts), 16):
        idx = order[s:s + 16]
        b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        v = model(**b).last_hidden_state[:, 0]
        vecs[idx] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
np.savez(out, ids=np.array([c["id"] for c in cases]), vecs=vecs)
print("embedded", len(texts), "max tokens 1024")
