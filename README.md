# semantic-cache-llm
This project is a proof-of-concept semantic cache designed to reduce redundant API calls to Large Language Models (LLMs), specifically Google’s Gemini API. It intercepts multi-turn conversational queries, detects when a new query is semantically similar to a past one, and serves cached results to improve latency and cost efficiency.

# Project Overview
Language: Python
\
Embeddings: Google Gemini Embeddings API
\
Vector Store: FAISS
\
Objective: Demonstrate that a context-aware semantic cache would be able to reduce redundant LLM calls while simultaneously preserving the answer quality. We will also perform an in-depth analysis of more advanced semantic caching strategies for other AI agents, beyond simple Q&A.

# Operation
- Each session request prompted by the user will be tracked by session_id and stored in an in-memory dictionary per run
- Every new message makes the system build a context-aware embedding, which combines the current query with recent conversation turns
- The current embedding will then be compared to previous embeddings within the FAISS index via cosine similarity
- If the similarity is greater than or equal to the threshold, a cached response will then be returned. If not, the query would be sent to the Gemini LLM, and the result would then be cached.

# Design Choices
- Embeddings: We will be using a Gemini embedding model for the semantic cache. 'text-embedding-004' is the latest embedding model available
- Generation model: We will be using 'gemini-2.5-flash', as it provide fast responses for cache misses and the quality of answers is sufficient for keeping the cost low
- Vector Store: We will be using FAISS as it is simple, fast, and doesn't require a server to run
- Threshold: We chose to run three different thresholds, 0.78, 0.82, 0.86 to see which one is best to use.
- Context Window: We made k=2 in order to build the embedding input.  This lets us join the last two turns and the current user message with role tags, then embed that string.
- Safety Gate: We are to only resuse if the same model produced the cache answer

# Setup Instructions
1. Clone repo:
   \
   git clone https://github.com/scred0405/semantic-cache-llm.git
   \
   cd semantic-cache-llm
   
2. Create virtual environment:
   \
   python -m venv .venv
   \
   source .venv\Scripts\activate
   
3. Install dependencies:
   \
   pip install -r requirements.txt
   
4. Add Gemini API key(environment variable):
   \
   Open the .env file and add Gemini API key:
   \
   GEMINI_API_KEY=KEYHERE

# Evaluation 
We will be evaluating the cache using various metrics such as:
- Cache hit rate
- Average latency improvement
- LLM calls avoided

# Results

| setup                   |  n | hit_rate | calls_avoided | p50_latency_ms | p95_latency_ms | false_reuse_rate | precision | recall | f1
|-------------------------|---:|---------:|--------------:|---------------:|---------------:|-----------------:|----------:|-------:|
| no_cache_log            | 20 |   0.00   |             0 |          13128 |          19670 |            0.000 |      0.00 |   0.00 | 0.000
| semantic_cache_tau_0.78 | 20 |   0.45   |             9 |          12520 |          17837 |            0.000 |      1.00 |   1.00 | 1.000
| semantic_cache_tau_0.82 | 20 |   0.20   |             4 |          12246 |          17707 |            0.000 |      0.75 |   0.25 | 0.222
| semantic_cache_tau_0.86 | 20 |   0.00   |             0 |          13161 |          19564 |            0.000 |      0.00 |   0.00 | 0.000

# Threshold
We swept tau = 0.78, 0.82, 0.86. On this dataset, tau = 0.78 avoided 9/20 calls with 0 false reuse, while higher thresholds became too strict. 
We therefore use tau = 0.78 by default as the best balance between reuse and safety.

# Scalability
At million-query scale, there would be a heavy burden on memory, index latency, embedding throughput, staleness, and model drift. I would shard the vector index by time or domain where hot shards are in memory and cold ones on disk, use compressed ANN for speed recall balance, write new items to a small buffer index and periodically merge into main shards offline, batch embeddings and cache repeated prompts, version embeddings and cached entries, and apply short TTLs for time sensitive content.

# Eviction
I would use a simple policy that combines a time-to-live (TTL) and a size cap with LRU. Then, I'd expire anything past its TTL, and when we hit the cap, evict the least-recently-used items first. If you want a bit more smarts, weight by recent hit count in order for frequently reused answers to stay. I'd also add per-namespace caps so one topic can’t block out the rest, and keep the logic light so eviction never blocks lookups.
