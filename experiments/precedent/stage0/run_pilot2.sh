#!/bin/bash
# Pilot of gate registry v2 on 300 full Scruples anecdotes (training split). Reads texts only.
cd /archive/experiments/precedent-2026-11-25/stage0b
set -a; . ~/.primer.env; set +a
export GATE_LLM_BASE_URL=https://ellm.nrp-nautilus.io/v1 GATE_LLM_TOKEN="$NRP_LLM_TOKEN"
for pass in 1 2; do
  python3 read_gates.py --in pilot_cases.jsonl --out pilot_gpt-oss.jsonl --model gpt-oss --workers 4 --max-tokens 4000 > pilot_gpt-oss.log 2>&1 &
  python3 read_gates.py --in pilot_cases.jsonl --out pilot_gemma.jsonl --model gemma --workers 4 --max-tokens 4000 > pilot_gemma.log 2>&1 &
  wait
done
python3 - <<"PY" > pilot_agreement.txt 2>&1
import json
def load(p): return {d["id"]: d["gates"] for d in map(json.loads, open(p)) if d["gates"] is not None}
a, b = load("pilot_gpt-oss.jsonl"), load("pilot_gemma.jsonl")
ids = sorted(set(a) & set(b)); n = len(ids); names = list(next(iter(a.values()))); out = {}
for g in names:
    x = [a[i][g] for i in ids]; y = [b[i][g] for i in ids]
    pa, pb = sum(x)/n, sum(y)/n; po = sum(int(u == v) for u, v in zip(x, y))/n; pe = pa*pb + (1-pa)*(1-pb)
    out[g] = {"prev_gpt_oss": round(pa, 3), "prev_gemma": round(pb, 3), "agree": round(po, 3), "kappa": round((po-pe)/(1-pe), 3) if pe < 1 else None, "both_yes": sum(u & v for u, v in zip(x, y))}
json.dump({"registry": "gates-v2", "corpus": "scruples anecdotes, training split", "n": n, "read_gpt_oss": len(a), "read_gemma": len(b), "gates": out}, open("pilot_agreement.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
echo DONE > pilot2.done
