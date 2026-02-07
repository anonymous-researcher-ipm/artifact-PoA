# agents/selection/decision_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from core.schemas import EvalResult
from core.trace import ReasoningPath
from llm.json_utils import json_chat_with_retry

from agents.utils.prompt_loader import PromptLoader
from .adapter import build_candidates
from .verifier import SimpleVerifier


@dataclass
class PathDecisionAgent:
    """
    Implements core.schemas.DecisionAgent:
      decide(question, candidates)->int

    Fully protocolized: all info needed must be carried by candidates (ReasoningPath steps/obs/meta),
    plus injected llm_client + prompts.
    """
    llm_client: Any
    prompt_loader: PromptLoader = PromptLoader("prompts")

    debate_prompt: str = "selection_debate_many"
    decide_prompt: str = "selection_decide"

    temperature: float = 0.2
    max_tokens: int = 1200

    use_verifier: bool = True
    verifier: SimpleVerifier = SimpleVerifier()

    def decide(self, question: str, candidates: List[Any]) -> int:
        # candidates are expected to be List[ReasoningPath]
        paths: List[ReasoningPath] = candidates  # type: ignore
        if not paths:
            return 0
        if len(paths) == 1:
            return 0

        compact = build_candidates(paths)
        ids = [c["id"] for c in compact]

        # 1) Multi-path debate
        debate_pack = self.prompt_loader.load(self.debate_prompt)
        debate_vars = {"question": question, "candidates": str(compact), "candidate_ids": str(ids)}
        debate_system = debate_pack.system.format(**debate_vars)
        debate_user = debate_pack.user.format(**debate_vars)

        debate = json_chat_with_retry(
            self.llm_client.chat, debate_system, debate_user,
            max_retries=2,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 2) Decide best path
        decide_pack = self.prompt_loader.load(self.decide_prompt)
        decide_vars = {
            "question": question,
            "candidates": str(compact),
            "candidate_ids": str(ids),
            "debate": str(debate),
        }
        decide_system = decide_pack.system.format(**decide_vars)
        decide_user = decide_pack.user.format(**decide_vars)

        out = json_chat_with_retry(
            self.llm_client.chat, decide_system, decide_user,
            max_retries=2,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        selected_id = out.get("selected_id") if isinstance(out, dict) else None

        chosen_idx = 0
        if selected_id is not None:
            for i, c in enumerate(compact):
                if c["id"] == selected_id:
                    chosen_idx = i
                    break

        # 3) Optional deterministic sanity
        if self.use_verifier:
            ans = paths[chosen_idx].final_answer
            v = self.verifier.verify(question, ans)
            # If question wants number but selected answer is non-numeric -> fallback to max total_score
            if v.get("wants_number") and (not v.get("numeric")):
                chosen_idx = self._fallback_by_total_score(paths)

        return chosen_idx

    @staticmethod
    def _fallback_by_total_score(paths: List[ReasoningPath]) -> int:
        best_i, best_s = 0, -1e18
        for i, p in enumerate(paths):
            s = float(getattr(p, "total_score", 0.0) or 0.0)
            if s > best_s:
                best_s = s
                best_i = i
        return best_i
