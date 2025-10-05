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
   
4. Add Gemini API key:
   \
   Open the .env file and add Gemini API key:
   \
   GEMINI_API_KEY=KEYHERE

# Operation
- Each session request prompted by the user will be tracked by session_id
- Every new message makes the system build a context-aware embedding, which combines the current query with recent conversation turns
- The current embedding will then be compared to previous embeddings within the FAISS index via cosine similarity
- If the similarity is greater than or equal to the threshold, a cached response will then be returned. If not, the query would be sent to the Gemini LLM, and the result would then be cached.

# Evaluation 
We will be evaluating the cache using various metrics such as:
- Cache hit rate
- Average latency improvement
- LLM calls avoided
