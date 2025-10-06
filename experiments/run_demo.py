import os, sys, time, json, csv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# import package with class names
from semanticcache import Embedder, ANNindex, CachePolicy, SemanticCacheItem, Seshmanager

# setup gemini
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# models being used
EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

# path for data and results
DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "convos.json")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "labels.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    _raw_convos = json.load(f)

# debug prints in order to verify which files or keys are beign used
print("Using DATA_PATH:", DATA_PATH)
print("First item keys:", list(_raw_convos[0].keys()))

CONVOS = []
for i, c in enumerate(_raw_convos):
    sid = c.get("sessionid")
    c["sessionid"] = sid # ensure the key exists
    CONVOS.append(c)

# load labels for evaluation
LABELS = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for raw in reader:
            # normalize keys/values (strip spaces, lowercase keys)
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}

            sid  = row.get("sessionid")
            tidx = row.get("trnindx")
            lab  = row.get("semduplicatelabel")
            if not (sid and tidx and lab):
                print("Skipping row (missing keys):", row)
                continue

            LABELS[(sid, int(tidx))] = lab.lower() in {"true", "1", "yes", "y", "t"}

print("Loaded labels:", len(LABELS))


def run_demo(setup_name: str, use_cache: bool, tau: float, log_path: str):
    # Runs one pass over the dataset:
    # - If use_cache=False: baseline will always call LLM)
    # - If use_cache=True : semantic cache (embed -> search -> threshold -> reuse/insert)
    # Writes one JSON record per user turn to log_path and prints a summary.
    embedder = Embedder(EMBED_MODEL)
    index = ANNindex()
    policy = CachePolicy(threshold=tau)
    sessions = Seshmanager()
    cache = SemanticCacheItem(embedder, index, policy)
    llm = genai.GenerativeModel(LLM_MODEL)
    hits = 0
    falsereuse = 0
    latencies = []
    totalusrtrns = 0

    llm = genai.GenerativeModel(LLM_MODEL)      
    print(f"Running {setup_name} over {len(CONVOS)} sessions...")

    with open(log_path, "w", encoding="utf-8") as logf:
        for convo in CONVOS:
            sid = convo["sessionid"]
            for trnindx, turn in enumerate(convo["turns"]):
                role, text = turn["role"], turn["text"]
                if role == "ai":
                    # append to history so the next user turn includes recent context in its embedding
                    sessions.append(sid, "ai", text)
                    continue
                totalusrtrns += 1
                context = sessions.contextstr(sid, text, k=2)

                # Metadata gate for cache compatibility
                meta = {"model_id": LLM_MODEL, "system_hash": "default_v1"}

                # Measure latency from call to response
                t0 = time.time()

                if use_cache:
                    hit = cache.lookup(meta, context)
                else:
                    hit = {"hit": False, "vec": None}
                
                if hit["hit"]:
                    # reuse stored response 
                    response = hit["response"]
                    sim = hit["sim"]
                    llm_called = False
                    hits += 1
                else:
                    # call LLM with context string, then insert into cache 
                    resp = llm.generate_content(context) 
                    response = resp.text
                    sim = None
                    cache.insert(meta, context, response, vec=hit.get("vec"))
                    llm_called = True
                
                # compute the elapsed time 
                latency_ms = int((time.time() - t0) * 1000)
                latencies.append(latency_ms)

                # add response to session history 
                sessions.append(sid, "ai", response)
                label_dup = LABELS.get((sid, trnindx))
                if hit["hit"] and (label_dup is False):
                    falsereuse += 1

                rec = {"setup": setup_name, "sessionid": sid, "trnindx": trnindx, "threshold": tau,
                       "cache_hit": hit["hit"], "similarity": sim, "latency_ms": latency_ms, 
                       "llm_called": llm_called, "semduplicatelabel": label_dup, "embedding_model": EMBED_MODEL,
                       "generation_model": LLM_MODEL}
                logf.write(json.dumps(rec) + "\n")
                print(f"{setup_name}: {sid} turn {trnindx} -> "
                      f"hit={hit['hit']} llm_called={llm_called} latency={latency_ms}ms")
    
    # summary statistics for the run
    def p50(vals):
        s = sorted(vals)
        return s[len(s)//2] if s else 0
    def p95(vals):
        if not vals: return 0
        s = sorted(vals); idx = max(0, int(0.95*len(s)) - 1)
        return s[idx]
    
    summary = {"setup": setup_name, 
               "hit_rate": round(hits / totalusrtrns, 3) if totalusrtrns else 0.0, 
               "calls_avoided": hits, "p50_latency_ms": p50(latencies), "p95_latency_ms": p95(latencies),
               "false_reuse_rate": round(falsereuse / hits, 3) if hits else 0.0, "log_path": log_path
    } 
    print(summary)

if __name__ == "__main__":
     # Baseline (no cache): always call the LLM
    run_demo(setup_name="no_cache", use_cache=False, tau=0.0, log_path=os.path.join(RESULTS_DIR, "no_cache_log.jsonl"),
    )
     # Semantic cache run (tau=0.82): use embeddings + FAISS + threshold to reuse answers
    run_demo(setup_name="semantic_cache_tau_0.82", use_cache=True, tau=0.82, log_path=os.path.join(RESULTS_DIR, "semantic_cache_tau_0.82.jsonl"),
    )





