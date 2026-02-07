# actions/decomposition/parallel_decomposing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


@register_action
@dataclass
class ParallelDecomposing(Action):
    """
    11) Parallel Decomposing
    - If sub_questions provided: store directly.
    - Else if llm enabled: LLM outputs {"sub_questions":[...]} and code stores.
    """
    TYPE: str = "ParallelDecomposing"

    sub_questions: Optional[List[str]] = None
    use_llm: bool = True
    out_key: str = "sub_questions_parallel"

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        subs = None
        llm_used = False

        if self.sub_questions is not None:
            subs = [q.strip() for q in self.sub_questions if isinstance(q, str) and q.strip()]
        elif self.use_llm:
            try:
                system = "You decompose questions. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "task": "Decompose into independent sub-questions that can be answered separately and then combined.",
                    "output_schema": {"sub_questions": ["<q1>", "<q2>"]},
                    "constraints": ["Sub-questions should be independent (parallel).", "Return 1-5 items."],
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict) and isinstance(out.get("sub_questions"), list):
                    subs = [q.strip() for q in out["sub_questions"] if isinstance(q, str) and q.strip()]
                    llm_used = True
            except Exception:
                llm_used = False

        if subs is None:
            subs = []  # placeholder, but explicit
            note = "No sub_questions; stored empty list."
        else:
            note = "Sub-questions stored."

        ctx.memory[self.out_key] = subs
        obs = {"out_key": self.out_key, "count": len(subs), "llm_used": llm_used, "note": note}
        return ctx, obs
