import os, json, glob, statistics as stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _to_bool(v):
    if isinstance(v, bool): return v
    if v is None: return None
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}

def summarize(records):
    recs = list(records)
    n = len(recs)
    if n == 0:
        return {"n": 0, "hit_rate": 0.0, "calls_avoided": 0, "p50_latency_ms": 0,
                "p95_latency_ms": 0, "false_reuse_rate": 0.0,
                "precision": None, "recall": None, "f1": None}

    hits = sum(1 for r in recs if _to_bool(r.get("cache_hit")))
    latencies = [int(r.get("latency_ms", 0)) for r in recs]
    p50 = int(stats.median(latencies)) if latencies else 0
    p95 = int(sorted(latencies)[max(0, int(0.95*len(latencies))-1)]) if latencies else 0

    TP = FP = TN = FN = 0
    labeled = 0
    for r in recs:
        label = _to_bool(r.get("semduplicatelabel"))
        if label is None:
            continue
        labeled += 1
        hit = _to_bool(r.get("cache_hit"))
        if label and hit: TP += 1
        elif label and not hit: FN += 1
        elif (not label) and hit: FP += 1
        else: TN += 1

    precision = TP / (TP + FP) if (TP + FP) else None
    recall    = TP / (TP + FN) if (TP + FN) else None
    f1        = (2*precision*recall/(precision+recall)) if (precision and recall) else None
    false_reuse_rate = (FP / hits) if hits else 0.0

    return {"n": n, "hit_rate": round(hits / n, 3), "calls_avoided": hits, "p50_latency_ms": p50,
            "p95_latency_ms": p95, "false_reuse_rate": round(false_reuse_rate, 3), "precision": None if precision is None else round(precision, 3),
            "recall": None if recall is None else round(recall, 3), "f1": None if f1 is None else round(f1, 3),
    }

if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")))
    if not paths:
        print("No logs found in results/. Run run_demo.py first.")
        raise SystemExit(0)

    headers = ["setup","n","hit_rate","calls_avoided","p50_latency_ms","p95_latency_ms","false_reuse_rate","precision","recall","f1"]
    print(",".join(headers))

    for p in paths:
        setup = os.path.splitext(os.path.basename(p))[0]
        s = summarize(load_jsonl(p))
        row = [setup] + [s[h] for h in headers if h != "setup"]
        print(",".join("" if v is None else str(v) for v in row))