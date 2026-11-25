"""Gate reader: one structured call per case to an OpenAI-compatible endpoint.

Reads the gate registry (gates.json) and, for each text, asks a model which gates the text
states or clearly implies. It never sees a judgement and is not asked for one.

Input:  a JSONL of {"id": ..., "text": ...}
Output: a JSONL of {"id", "model", "served_model", "gates": {name: 0|1}, "raw", "error"}
        appended as answers arrive, so an interrupted run resumes where it stopped.

Endpoint and token come from the environment (never from this file):
    GATE_LLM_BASE_URL   e.g. https://.../v1
    GATE_LLM_TOKEN
Usage: python read_gates.py --in cases.jsonl --out gates_<model>.jsonl --model <alias> --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REGISTRY = json.loads((HERE / "gates.json").read_text(encoding="utf-8"))
NAMES = [g["name"] for g in REGISTRY["gates"]]

SYSTEM = (
    "You read a short description of a situation and report which of the listed conditions the "
    "text states or clearly implies. Do not judge whether anyone acted well or badly. Do not "
    "guess beyond the text: if the text does not state or clearly imply a condition, answer no. "
    "Answer with one JSON object whose keys are exactly the condition names and whose values are "
    '"yes" or "no". No other text.'
)


MAX_TEXT_CHARS = 4000  # longer texts are cut here, and the cut is the same for every reader


def prompt(text: str) -> str:
    text = text[:MAX_TEXT_CHARS]
    lines = [f"- {g['name']}: {g['question']}" for g in REGISTRY["gates"]]
    return "Conditions:\n" + "\n".join(lines) + f"\n\nSituation: {text}\n\nJSON:"


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for n in NAMES:
        v = str(d.get(n, "")).strip().lower()
        if v not in ("yes", "no"):
            return None  # a missing or malformed gate is not read as "no"
        out[n] = int(v == "yes")
    return out


def ask(base: str, token: str, model: str, text: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(text)}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last = ""
    for attempt in range(3):
        try:
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
                timeout=timeout,
            )
            if r.status_code != 200:
                last = f"http {r.status_code}"
                time.sleep(3 * (attempt + 1))
                continue
            j = r.json()
            raw = j["choices"][0]["message"].get("content") or ""
            gates = parse(raw)
            if gates is None:
                last = "unparseable"
                continue
            return {"served_model": j.get("model"), "gates": gates, "raw": raw.strip()[:400], "error": None}
        except (requests.RequestException, KeyError, ValueError) as err:
            last = type(err).__name__
            time.sleep(3 * (attempt + 1))
    return {"served_model": None, "gates": None, "raw": "", "error": last}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    base, token = os.environ.get("GATE_LLM_BASE_URL"), os.environ.get("GATE_LLM_TOKEN")
    if not base or not token:
        sys.exit("REFUSED: set GATE_LLM_BASE_URL and GATE_LLM_TOKEN in the environment")
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("gates") is not None:
                done.add(rec["id"])
    todo = [json.loads(x) for x in open(a.inp, encoding="utf-8")]
    todo = [c for c in todo if c["id"] not in done]
    lock = threading.Lock()
    n = {"ok": 0, "err": 0}
    with open(a.out, "a", encoding="utf-8") as fh:

        def work(case):
            res = ask(base.rstrip("/"), token, a.model, case["text"], a.max_tokens, a.timeout)
            rec = {"id": case["id"], "model": a.model, "registry": REGISTRY["version"], **res}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n["ok" if res["gates"] is not None else "err"] += 1
                if (n["ok"] + n["err"]) % 50 == 0:
                    print(f"{a.model}: {n['ok']} read, {n['err']} failed, of {len(todo)}", flush=True)

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(work, todo))
    print(f"{a.model}: done. {n['ok']} read, {n['err']} failed, {len(done)} already present")


if __name__ == "__main__":
    main()
