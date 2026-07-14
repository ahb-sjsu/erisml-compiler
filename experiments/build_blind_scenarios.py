"""Assemble a MECHANISM-BLIND Phase-5 scenario set for the decisive test.

The confirm2 dilemmas were author-written with knowledge of the polar
conflict-detection mechanism, which could inflate H2. This set removes that bias:
the dilemma / easy-moral labels come from INDEPENDENT HUMAN annotators, not from
the author or the detector.

  D (contested)   = Scruples anecdotes whose annotators SPLIT (~50/50 normative
                    vs not) -> genuinely contested moral situations.
  E (easy-moral)  = Scruples anecdotes with UNANIMOUS annotator agreement (>=0.9)
                    -> clearly good / clearly bad, uncontested.
  B (benign)      = Social Chemistry 101 situations (non-moral logistics).

Scruples (AI2, arXiv:2008.09094) is Reddit-AITA-derived, human-labeled, and
predates this work. Selection is deterministic (hash-seeded) and length-filtered.

CAVEAT recorded in the manifest: B (Social Chem) and D/E (AITA) come from
different text distributions, so H1 carries a mild style confound; the
mechanism-blind guarantee is on the D-vs-E (contestedness) distinction, which is
what the confirm2 authoring-bias concern was about.

  python experiments/build_blind_scenarios.py --n-d 10 --n-e 10 --n-b 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REMOTE = "/archive/ethics-corpora/scruples/anecdotes/test.scruples-anecdotes.jsonl"


def _hkey(s: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()


def fetch_scruples(n_d: int, n_e: int, seed: int, min_votes=8, lo=250, hi=1400):
    """Select contested + unanimous anecdotes ON Atlas (avoids shipping 1200 long
    posts); returns two lists of {id,text,pn,cn}."""
    import paramiko

    from erisml_compiler.atlas_creds import atlas_credentials

    remote_py = f'''
import json, hashlib
D=[];U=[]
for line in open("{REMOTE}"):
    r=json.loads(line); a=r.get("action") or {{}}
    pn=a.get("pronormative_score",0); cn=a.get("contranormative_score",0); n=pn+cn
    t=(r.get("text") or "").strip()
    if n<{min_votes} or not ({lo}<=len(t)<={hi}): continue
    frac=min(pn,cn)/n
    rec={{"id":r["id"],"text":t,"pn":pn,"cn":cn}}
    if frac>=0.4: D.append(rec)
    elif max(pn,cn)/n>=0.9: U.append(rec)
key=lambda r: hashlib.sha256((str({seed})+":"+r["id"]).encode()).hexdigest()
D.sort(key=key); U.sort(key=key)
print(json.dumps({{"D":D[:{n_d}],"E":U[:{n_e}]}}))
'''
    h, u, p = atlas_credentials()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(h, username=u, password=p, timeout=30)
    import shlex
    _, out, _ = c.exec_command(f"/home/claude/env/bin/python3 -c {shlex.quote(remote_py)}", timeout=120)
    data = json.loads(out.read().decode())
    c.close()
    return data["D"], data["E"]


def benign(n_b: int, seed: int):
    from erisml_compiler.social_chem.loader import load_situations
    pool = []
    for tsv in ["data/social-chem-101/social-chem-101.dearabby.tsv",
                "data/social-chem-101/social-chem-101.amitheasshole.tsv"]:
        if Path(tsv).exists():
            for s in load_situations(tsv):
                t = s.situation.strip()
                if 60 <= len(t) <= 400:
                    pool.append({"id": s.situation_short_id, "text": t})
    seen = set(); uniq = []
    for r in pool:
        if r["text"].lower() in seen:
            continue
        seen.add(r["text"].lower()); uniq.append(r)
    uniq.sort(key=lambda r: _hkey(r["text"], seed))
    return uniq[:n_b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-d", type=int, default=10)
    ap.add_argument("--n-e", type=int, default=10)
    ap.add_argument("--n-b", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--out", default="experiments/scenarios_phase5_blind.json")
    a = ap.parse_args()

    D, E = fetch_scruples(a.n_d, a.n_e, a.seed)
    B = benign(a.n_b, a.seed)
    scen = []
    for i, r in enumerate(B):
        scen.append({"id": f"KB{i+1:02d}", "class": "B", "expect_flag": False,
                     "text": r["text"], "source": "social-chem-101"})
    for i, r in enumerate(D):
        scen.append({"id": f"KD{i+1:02d}", "class": "D", "expect_flag": True,
                     "text": r["text"], "source": f"scruples-anecdote {r['id']} votes {r['pn']}/{r['cn']} (contested)"})
    for i, r in enumerate(E):
        scen.append({"id": f"KE{i+1:02d}", "class": "E", "expect_flag": False,
                     "text": r["text"], "source": f"scruples-anecdote {r['id']} votes {r['pn']}/{r['cn']} (unanimous)"})
    out = {"version": "phase5-blind-v1",
           "note": "MECHANISM-BLIND set. D=contested / E=unanimous Scruples anecdotes "
                   "(human vote-split labels, independent of the polar detector); "
                   "B=Social Chem benign. Tested with the FROZEN prereg-phase5-confirm2-v1 rule. "
                   "Caveat: B(Social Chem) vs D/E(AITA) differ in text distribution -> mild H1 style "
                   "confound; the blind guarantee is on the D-vs-E contestedness distinction.",
           "scenarios": scen}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {a.out}: {len(B)} B + {len(D)} D + {len(E)} E")
    print("  sample D:", D[0]["text"][:110].replace("\n", " ") if D else "(none)")


if __name__ == "__main__":
    main()
