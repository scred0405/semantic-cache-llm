from dotenv import load_dotenv
import os, google.generativeai as genai

load_dotenv() 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

emb = "text-embedding-004" #embedding model
llm = "gemini-2.5-flash" #llm model

#embedding test
em = genai.embed_content(model=emb, content="Why do stocks go up?", task_type="retrieval_query")
print("Embedding model:", emb, "| dim:", len(em["embedding"]))

#llm test
answer = genai.GenerativeModel(llm).generate_content("Say 'understood' if you can read this.")
print("LLM sample:", answer.text.strip()[:40])
