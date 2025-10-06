import google.generativeai as genai

class Embedder:
    def __init__(self, model: str = "text-embedding-004"):
        self.model = model 

    def embed(self, text: str): # returning a vector for text via Gemini embedding model
        response = genai.embed_content(model=self.model, content=text, task_type="retrieval_query")
        return response["embedding"]
    