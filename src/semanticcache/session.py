from typing import Dict, List, Tuple

class Seshmanager:
    def __init__(self):
        self.hist: Dict[str, List[Tuple[str, str]]] = {}


    def append(self, session_id: str, role: str, text: str):
        # add turn to conversation history
        self.hist.setdefault(session_id, []).append((role, text))

    def contextstr(self, session_id: str, currusrtxt: str, k: int = 2) -> str:
        # get last k turns and current user text as a string
        turns = self.hist.get(session_id, [])[-2*k:]
        parts = [f"{r.upper()}: {t}" for r, t in turns]
        parts.append(f"USER: {currusrtxt}")
        return "\n".join(parts)