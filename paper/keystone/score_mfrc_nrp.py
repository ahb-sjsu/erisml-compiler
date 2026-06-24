import json, requests, re, sys, concurrent.futures as cf
TOKEN=open('/home/claude/.llmtoken').read().strip()
URL="https://ellm.nrp-nautilus.io/v1/chat/completions"
DIMS=["physical_harm","rights_respect","fairness_equity","autonomy_consent","legitimacy_trust",
      "epistemic_quality","care_protection","vow_fidelity","third_party_externality","repair_residue"]
RUBRIC=("Rate from 0.0 to 1.0 how strongly each moral dimension is AT STAKE / ENGAGED in the text "
 "(0=not engaged, 1=central). Dimensions: physical_harm (bodily or material harm); rights_respect "
 "(rights and duties); fairness_equity (fairness, equal or proportional treatment); autonomy_consent "
 "(autonomy, consent, coercion); legitimacy_trust (authority, legitimacy, institutional trust); "
 "epistemic_quality (honesty, truth, evidence); care_protection (care, compassion, protecting the "
 "vulnerable); vow_fidelity (loyalty, promises, fidelity to a relationship or group); "
 "third_party_externality (effects on uninvolved third parties); repair_residue (apology, repair, "
 "making amends). Respond with ONLY a JSON object mapping each dimension name to its score.")
def parse(c):
    c=re.sub(r"<think>.*?</think>","",c,flags=re.S)
    objs=re.findall(r"\{[^{}]*\}", c.replace("\n"," "))
    for o in reversed(objs):
        try:
            d=json.loads(o)
            if any(k in d for k in DIMS): return {k:float(d.get(k,0.0)) for k in DIMS}
        except Exception: pass
    return None
def score(item):
    body={"model":"qwen3","messages":[
        {"role":"system","content":"You are a careful moral-dimension annotator. Output only JSON."},
        {"role":"user","content":RUBRIC+"\n\nTEXT:\n"+item['text']}],
        "temperature":0,"max_tokens":1500,"chat_template_kwargs":{"enable_thinking":False}}
    try:
        r=requests.post(URL,headers={"Authorization":f"Bearer {TOKEN}"},json=body,timeout=90)
        vec=parse(r.json()["choices"][0]["message"]["content"])
        if vec is None: return {"text":item['text'],"error":"parse"}
        return {"text":item['text'],"frac":item['frac'],"nonmoral":item['nonmoral'],
                "dominant":item['dominant'],"vec":vec}
    except Exception as e: return {"text":item['text'],"error":str(e)[:80]}
sub=json.load(open('mfrc_subset.json'))
lim=int(sys.argv[1]) if len(sys.argv)>1 else len(sub); sub=sub[:lim]
out=[]
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    out=list(ex.map(score,sub))
ok=[r for r in out if 'vec' in r]
json.dump(out,open('/tmp/deme_mfrc_vectors.json','w'))
print(f"scored {len(ok)}/{len(out)} ok")
if ok: print("sample vec:",json.dumps(ok[0]['vec']))
