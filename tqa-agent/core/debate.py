from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .trace import ReasoningPath
from .schemas import DebateAgent, DecisionAgent, EvalResult

@dataclass
class DebateRunner:
    """
    Phase 2: Given multiple candidate reasoning paths, select the best one and
    produce the final answer (based on that path).
    """
    decision_agent: Optional[DecisionAgent] = None
    debate_agents: Optional[List[DebateAgent]] = None

    def run(self, question: str, candidates: List[ReasoningPath]) -> ReasoningPath:
        if not candidates:
            raise ValueError("No candidate paths provided to DebateRunner.")

        # 1) Optional: multi-agent judging to refine scores / add critiques
        if self.debate_agents:
            for p in candidates:
                # aggregate judges
                scores = []
                critiques = []
                for j in self.debate_agents:
                    er: EvalResult = j.judge(question, p)
                    scores.append(er.score)
                    if er.critique:
                        critiques.append(er.critique)
                if scores:
                    # average as refined score
                    p.total_score = float(sum(scores) / len(scores))
                if critiques:
                    p.meta["debate_critiques"] = critiques

        # 2) Decision: choose best index
        if self.decision_agent:
            idx = self.decision_agent.decide(question, candidates)
            idx = max(0, min(idx, len(candidates) - 1))
            return candidates[idx]

        # default: highest score
        candidates.sort(key=lambda p: p.total_score, reverse=True)
        return candidates[0]
