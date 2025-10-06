from typing import Dict, Optional

class SemanticCacheItem:
    def __init__(self, embedder, index, policy):
        self.embedder = embedder 
        self.index = index   
        self.policy = policy   
    
    def lookup(self, seshmeta: Dict, cntxtxt: str) -> Dict:
        vec = self.embedder.embed(cntxtxt)
        nn = self.index.search(vec, top_k=5)
        for i, sim, payload in nn:
            if self.policy.passthreshold(sim) and self.policy.cachecomp(seshmeta, payload.get("meta", {})):
                # reuse previous answer
                return {"hit": True, "response": payload["response"], "sim": sim, "vec": vec}
        # no match
        return {"hit": False, "vec": vec} 
    
    def insert(self, seshmeta: Dict, cntxtxt: str, resptxt: str, vec: Optional[list] = None) -> None:
        v = vec if vec is not None else self.embedder.embed(cntxtxt)
        self.index.add(v, {"response": resptxt, "meta": seshmeta})

