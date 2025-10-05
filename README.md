# semantic-cache-llm
This project is a proof-of-concept semantic cache designed to reduce redundant API calls to Large Language Models (LLMs), specifically Google’s Gemini API. It intercepts multi-turn conversational queries, detects when a new query is semantically similar to a past one, and serves cached results to improve latency and cost efficiency.

# Project Overview
Language: Python
\
Embeddings: Google Gemini Embeddings API
\
Vector Store: FAISS
\
Objective: Demonstrate that a context-aware semantic cache would be able to reduce redundant LLM calls while simultaneously preserving the answer quality. We will also perform an in-depth analysis of more advanced semantic caching strategies for other AI agents for work other than just simple Q&A.

# Setup Instructions
1. Clone repo:
   git clone https://github.com/scred0405/semantic-cache-llm.git
   cd semantic-cache-llm
2. 

