from typing import Dict

class CachePolicy:
    # Simple policy which performs threshold and metadata compatability checks
    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold # establish similarity threshold
    
    def passthreshold(self, similarity: float) -> bool:
        # checks if similarity is above threshold
        return similarity >= self.threshold
    
    def cachecomp(self, a: Dict, b: Dict) -> int:
        # checks to see if cache item is compatible with current request
        keys = ["model_id", "system_hash"]
        return all(a.get(k) == b.get(k) for k in keys)