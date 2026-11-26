"""Does ANY simple predictor read the judgement from the embedding? TRAINING POSTS ONLY, 5-fold CV.
No dev or test judgement is read. Exploratory, after the sealed run was void."""
import json, sys, numpy as np
run, data = sys.argv[1], sys.argv[2]
cases = [json.loads(x) for x in open(run + "/cases.jsonl", encoding="utf-8")]
emb = np.load(sys.argv[3])
row = {i: k for k, i in enumerate(emb["ids"])}
y, n = {}, {}
for line in open(data + "/train.scruples-anecdotes.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d["id"] in row:
        s = d.get("binarized_label_scores") or {}
        r, w = s.get("RIGHT", 0), s.get("WRONG", 0)
        if r + w >= 5:
            y[d["id"]] = w / (r + w); n[d["id"]] = r + w
ids = [i for i in row if i in y]
X = emb["vecs"][[row[i] for i in ids]].astype(np.float64)
Y = np.array([y[i] for i in ids]); N = np.array([n[i] for i in ids])
X /= np.linalg.norm(X, axis=1, keepdims=True)
print("training posts with >= 5 votes:", len(ids), "variance of judgement:", round(Y.var(), 4))
samp = float(np.mean(Y * (1 - Y) / (N - 1)))
print("sampling variance (noise floor of MSE):", round(samp, 4), "ceiling R2:", round(1 - samp / Y.var(), 3))
rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(ids))
def cv(pred):
    err = np.empty(len(ids)); base = np.empty(len(ids))
    for k in range(5):
        tr, te = fold != k, fold == k
        err[te] = (pred(X[tr], Y[tr], X[te]) - Y[te]) ** 2
        base[te] = (Y[tr].mean() - Y[te]) ** 2
    d = err - base
    return err.mean(), base.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
def ridge(a):
    def f(Xtr, Ytr, Xte):
        mu = Xtr.mean(0); m = Ytr.mean(); A = Xtr - mu
        w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (Ytr - m))
        return (Xte - mu) @ w + m
    return f
def knn(k):
    def f(Xtr, Ytr, Xte):
        idx = np.argpartition(-(Xte @ Xtr.T), k, axis=1)[:, :k]
        return Ytr[idx].mean(1)
    return f
print("predictor            MSE      mean-MSE  z against the mean (negative is better)  R2")
for name, p in [("ridge a=1", ridge(1.0)), ("ridge a=10", ridge(10.0)), ("ridge a=100", ridge(100.0)),
                ("kNN k=10", knn(10)), ("kNN k=30", knn(30)), ("kNN k=100", knn(100)), ("kNN k=300", knn(300))]:
    e, b, z = cv(p)
    print(f"{name:<20} {e:.5f}  {b:.5f}  {z:+.2f}   {1 - e / b:+.4f}")
