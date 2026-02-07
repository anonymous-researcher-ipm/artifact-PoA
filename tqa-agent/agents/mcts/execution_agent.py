# agents/mcts/execution_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from core.reasoning_context import ReasoningContext
from core.schemas import ActionSpec, Observation

from actions.registry import build_action


@dataclass
class TQAExecutionAgent:
    """
    Implements core.schemas.ExecutionAgent
      execute(ctx, action_spec)-> (new_ctx, observation)
    """
    penalize_on_error: float = -1e6

    def execute(self, ctx: ReasoningContext, action_spec: ActionSpec) -> Tuple[ReasoningContext, Observation]:
        try:
            action = build_action(action_spec)
            new_ctx, obs = action.apply(ctx)
            return new_ctx, (obs or {})
        except Exception as e:
            return ctx, {
                "ok": False,
                "error_type": e.__class__.__name__,
                "error": str(e),
                "penalty": self.penalize_on_error,
            }
