# actions/termination/finish.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError
from actions.registry import register_action


@register_action
@dataclass
class Finish(Action):
    """
    13) Finish (deterministic)
    Mark terminal and set final answer from ctx.memory or literal.
    """
    TYPE: str = "Finish"

    answer_from: Optional[str] = None
    literal: Any = None

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        if self.answer_from is not None:
            if self.answer_from not in ctx.memory:
                raise ActionError(f"answer_from var not found in ctx.memory: {self.answer_from}")
            ctx.answer = ctx.memory[self.answer_from]
        else:
            ctx.answer = self.literal

        ctx.done = True
        ctx.path.terminal = True
        ctx.path.final_answer = ctx.answer

        obs = {"answer": ctx.answer, "answer_from": self.answer_from, "literal_used": self.answer_from is None}
        return ctx, obs
