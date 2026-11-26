"""Noise ceiling of the Scruples Anecdotes judgement against a minimum vote count.
TRAINING SPLIT ONLY. No dev or test judgement is read."""
import json, sys
path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    d = json.loads(line)
    s = d.get("binarized_label_scores") or {}
    r, w = s.get("RIGHT", 0), s.get("WRONG", 0)
    n = r + w
    if n > 0:
        rows.append((n, w / n, len((d.get("title") or "") + (d.get("text") or ""))))
print("posts with at least one vote:", len(rows))
ns = sorted(n for n, _, _ in rows)
print("votes per post: median", ns[len(ns)//2], "p75", ns[3*len(ns)//4], "p90", ns[9*len(ns)//10], "p99", ns[99*len(ns)//100], "max", ns[-1])
print("min_votes posts mean_share total_var sampling_var true_var ceiling_R2 median_votes")
for m in (5, 10, 20, 30, 50, 100, 200):
    sel = [(n, p) for n, p, _ in rows if n >= m]
    if len(sel) < 30:
        print(m, len(sel), "too few"); continue
    k = len(sel); mean = sum(p for _, p in sel) / k
    tot = sum((p - mean) ** 2 for _, p in sel) / (k - 1)
    samp = sum(p * (1 - p) / (n - 1) for n, p in sel) / k
    med = sorted(n for n, _ in sel)[k // 2]
    print(m, k, round(mean, 3), round(tot, 4), round(samp, 4), round(tot - samp, 4), round(1 - samp / tot, 3), med)
