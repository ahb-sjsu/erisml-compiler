import json, requests, re, concurrent.futures as cf
TOKEN=open('/home/claude/.llmtoken').read().strip()
URL="https://ellm.nrp-nautilus.io/v1/chat/completions"
INSTR={
 "m10":["physical_harm: bodily or material harm","rights_respect: rights and duties",
   "fairness_equity: fairness, equal or proportional treatment","autonomy_consent: autonomy, consent, coercion",
   "legitimacy_trust: authority, legitimacy, institutional trust","epistemic_quality: honesty, truth, evidence",
   "care_protection: care, compassion, protecting the vulnerable","vow_fidelity: loyalty, promises, fidelity to a relationship or group",
   "third_party_externality: effects on uninvolved third parties","repair_residue: apology, repair, making amends"],
 "m9":["physical_harm: bodily or material harm","rights_respect: rights and duties",
   "fairness_equity: fairness, equal or proportional treatment","autonomy_respect: autonomy, consent, freedom from coercion",
   "privacy_protection: privacy, personal data and information","societal_environmental: effects on society and environment",
   "virtue_care: care, compassion, virtue","legitimacy_trust: authority, legitimacy, institutional trust",
   "epistemic_quality: honesty, truth, evidence"],
 "m7":["physical: physical or bodily harm","emotional: emotional or psychological harm","financial: financial or material harm",
   "autonomy: loss of autonomy or coercion","trust: breach of trust or legitimacy","social: social or reputational harm",
   "identity: harm to identity or dignity"]}
def dims(k): return [d.split(':')[0] for d in INSTR[k]]
def parse(c,ds):
    c=re.sub(r"<think>.*?</think>","",c,flags=re.S)
    for o in reversed(re.findall(r"\{[^{}]*\}", c.replace("\n"," "))):
        try:
            d=json.loads(o)
            if any(k in d for k in ds): return {k:float(d.get(k,0.0)) for k in ds}
        except Exception: pass
    return None
def score(text,key,model):
    ds=dims(key)
    rub=("Rate from 0.0 to 1.0 how strongly each dimension is AT STAKE / ENGAGED in the text "
         "(0=not engaged, 1=central). Dimensions:\n"+"\n".join(INSTR[key])+
         "\nRespond with ONLY a JSON object mapping each dimension name to its score.")
    body={"model":model,"messages":[{"role":"system","content":"Careful moral annotator. Output only JSON."},
          {"role":"user","content":rub+"\n\nTEXT:\n"+text}],
          "temperature":0,"max_tokens":1500,"chat_template_kwargs":{"enable_thinking":False}}
    try:
        r=requests.post(URL,headers={"Authorization":f"Bearer {TOKEN}"},json=body,timeout=90)
        return parse(r.json()["choices"][0]["message"]["content"],ds)
    except Exception: return None
sub=json.load(open('mfrc_subset.json'))
JOBS=[("m10_qwen","m10","qwen3"),("m9_qwen","m9","qwen3"),("m7_qwen","m7","qwen3"),("m10_glm","m10","glm-5")]
res=[{"frac":a['frac'],"dominant":a['dominant'],"nonmoral":a['nonmoral']} for a in sub]
for col,key,model in JOBS:
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        vecs=list(ex.map(lambda a: score(a['text'],key,model), sub))
    ok=sum(v is not None for v in vecs)
    for r,v in zip(res,vecs): r[col]=v
    print(f"{col}: {ok}/{len(sub)} ok", flush=True)
json.dump(res,open('/tmp/mfrc_multi.json','w'))
print("done")
