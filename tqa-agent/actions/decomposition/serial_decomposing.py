# actions/decomposition/serial_decomposing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


@register_action
@dataclass
class SerialDecomposing(Action):
    """
    12) Serial Decomposing
    - If chain provided: lightly validate and store.
    - Else if llm enabled: LLM outputs {"chain":[{"q":..., "depends_on":[...], "var":"x0"}, ...]}
    """
    TYPE: str = "SerialDecomposing"

    chain: Optional[List[Dict[str, Any]]] = None
    use_llm: bool = True
    out_key: str = "sub_questions_serial"

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        llm_used = False

        if self.chain is not None:
            for i, item in enumerate(self.chain):
                if not isinstance(item, dict) or "q" not in item:
                    continue
                plan.append({
                    "q": str(item.get("q", "")).strip(),
                    "depends_on": list(item.get("depends_on", [])) if item.get("depends_on") is not None else [],
                    "var": str(item.get("var", f"x{i}")),
                })
        elif self.use_llm:
            try:
                system = "You decompose questions into dependent steps. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "task": "Decompose into dependent (serial) sub-questions with explicit dependencies.",
                    "output_schema": {
                        "chain": [
                            {"q": "<subq>", "depends_on": [0], "var": "x0"}
                        ]
                    },
                    "constraints": ["Use 0-based indices in depends_on.", "Return 1-6 steps."],
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict) and isinstance(out.get("chain"), list):
                    for i, item in enumerate(out["chain"]):
                        if not isinstance(item, dict) or "q" not in item:
                            continue
                        plan.append({
                            "q": str(item.get("q", "")).strip(),
                            "depends_on": list(item.get("depends_on", [])) if item.get("depends_on") is not None else [],
                            "var": str(item.get("var", f"x{i}")),
                        })
                    llm_used = True
            except Exception:
                llm_used = False

        ctx.memory[self.out_key] = plan
        obs = {"out_key": self.out_key, "count": len(plan), "llm_used": llm_used}
        return ctx, obs
