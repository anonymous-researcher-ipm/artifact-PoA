# agents/selection/debate_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from core.schemas import EvalResult
from core.trace import ReasoningPath
from llm.json_utils import json_chat_with_retry

from agents.utils.prompt_loader import PromptLoader
from .adapter import path_to_candidate


@dataclass
class PathDebateAgent:
    """
    Implements core.schemas.DebateAgent:
      judge(question, path)->EvalResult

    Note: This is per-path judging (protocol). Multi-path debate is done in DecisionAgent.decide().
    """
    llm_client: Any
    prompt_loader: PromptLoader = PromptLoader("prompts")
    prompt_name: str = "selection_judge_path"

    temperature: float = 0.2
    max_tokens: int = 900

    def judge(self, question: str, path: ReasoningPath) -> EvalResult:
        cand = path_to_candidate(path, idx=0)

        pack = self.prompt_loader.load(self.prompt_name)
        variables = {
            "question": question,
            "candidate": str(cand),
        }
        system = pack.system.format(**variables)
        user = pack.user.format(**variables)

        out = json_chat_with_retry(
            self.llm_client.chat, system, user,
            max_retries=2,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # expected: {"score":..., "critique":"...", "extra":{...}}
        score = float(out.get("score", 0.0)) if isinstance(out, dict) else 0.0
        critique = str(out.get("critique", "")) if isinstance(out, dict) else ""
        extra = out.get("extra", {}) if isinstance(out, dict) else {}
        if extra is None or not isinstance(extra, dict):
            extra = {}

        return EvalResult(score=score, critique=critique, extra=extra)
