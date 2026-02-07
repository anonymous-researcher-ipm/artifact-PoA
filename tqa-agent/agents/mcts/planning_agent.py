# agents/mcts/planning_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.reasoning_context import ReasoningContext
from core.schemas import ActionSpec
from llm.json_utils import json_chat_with_retry
from llm.errors import LLMParseError

from agents.utils.prompt_loader import PromptLoader
from .context_sensing_agent import SimpleContextSensingAgent


@dataclass
class TQAPlanningAgent:
    """
    Implements core.schemas.PlanningAgent
      propose(ctx)-> ranked List[ActionSpec]
    """
    llm_client: Any
    prompt_loader: PromptLoader = PromptLoader("prompts")
    prompt_name: str = "mcts_planning"

    topk: int = 5
    temperature: float = 0.2
    max_tokens: int = 900

    use_context_report: bool = True
    context_sensor: Optional[SimpleContextSensingAgent] = None

    def propose(self, ctx: ReasoningContext) -> List[ActionSpec]:
        report = None
        if self.use_context_report:
            sensor = self.context_sensor or SimpleContextSensingAgent()
            report = sensor.report(ctx)

        pack = self.prompt_loader.load(self.prompt_name)
        variables = {
            "question": ctx.question,
            "headers": str(list(ctx.view.headers)),
            "context_report": str(report),
            "topk": str(self.topk),
        }
        system = pack.system.format(**variables)
        user = pack.user.format(**variables)

        out = json_chat_with_retry(
            self.llm_client.chat, system, user,
            max_retries=2,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not isinstance(out, dict) or "action_specs" not in out or not isinstance(out["action_specs"], list):
            raise LLMParseError("Planner must output JSON: { action_specs: [ {...}, ... ] }")

        specs: List[ActionSpec] = []
        for x in out["action_specs"]:
            if isinstance(x, dict) and "type" in x:
                specs.append(x)

        # Safe fallback
        if not specs:
            if "header_info" not in (ctx.memory or {}):
                return [{"type": "HeaderParsing"}]
            return [{"type": "Computing", "mode": "auto", "out_var": "result"}]

        return specs[: self.topk]
