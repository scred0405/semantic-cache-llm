import faiss
import numpy as np
from typing import Any, Dict, List, Tuple

def l2norm(v: np.ndarray) -> np.ndarray: 
    # normalze the vector to unit length so inner product is equal to cosine simlilarity
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

class ANNindex:
    # FAISS wrapper which stores a vector as well as random metadata and returns top-k tuples
    def __init__(self):
        # Becomes IndexFlatIP when the first vector is added
        self.index = None
        self.payloads: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
    
    def add(self, vec: List[float], payload: Dict[str, Any]) -> int:
        # stores a vector and random metadata
        v = l2norm(np.asarray(vec)).astype("float32").reshape(1, -1)
        if self.index is None:
            dim = v.shape[1] # vector dimension
            self.index = faiss.IndexFlatIP(dim)  # inner product via cosine similarity
        self.index.add(v) # add a vector
        pid = self.next_id
        self.payloads[pid] = payload
        self.next_id += 1
        return pid
    
    def search(self, vec: List[float], top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        # returns top-k tuples
        if self.index is None or self.index.ntotal == 0:
            return [] # none found
        v = l2norm(np.asarray(vec)).astype("float32").reshape(1, -1)
        sims, ids = self.index.search(v, min(top_k, self.index.ntotal))  
        results = []
        for sim, score in zip(sims[0], ids[0]):
            if score == -1:  # if fewer than top_k items
                continue
            results.append(((int(score)), float(sim), self.payloads[int(score)]))
        return results
    


    

