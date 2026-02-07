# agents/mcts/evaluation_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.reasoning_context import ReasoningContext
from core.schemas import EvalResult
from llm.json_utils import json_chat_with_retry

from agents.utils.prompt_loader import PromptLoader


def _is_terminal(ctx: ReasoningContext) -> bool:
    if bool(getattr(ctx, "done", False)):
        return True
    if getattr(ctx, "path", None) is not None and bool(getattr(ctx.path, "terminal", False)):
        return True
    if (ctx.memory or {}).get("final_answer", None) is not None:
        return True
    return False


def _progress_features(ctx: ReasoningContext) -> Dict[str, float]:
    mem = ctx.memory or {}
    feats: Dict[str, float] = {}
    feats["has_header_info"] = 1.0 if "header_info" in mem else 0.0
    feats["has_located_columns"] = 1.0 if "located_columns" in mem else 0.0
    feats["has_located_rows"] = 1.0 if "located_rows" in mem else 0.0

    num_vars = 0
    for v in mem.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            num_vars += 1
    feats["numeric_vars"] = float(min(num_vars, 10)) / 10.0

    steps = getattr(getattr(ctx, "path", None), "steps", [])
    feats["steps_norm"] = float(min(len(steps), 30)) / 30.0 if isinstance(steps, list) else 0.0
    feats["has_candidate_answer"] = 1.0 if ("answer" in mem or "final_answer" in mem) else 0.0
    return feats


def _heuristic_process_value(ctx: ReasoningContext) -> float:
    f = _progress_features(ctx)
    v = (
        0.20 * f["has_header_info"]
        + 0.25 * f["has_located_columns"]
        + 0.20 * f["has_located_rows"]
        + 0.25 * f["numeric_vars"]
        + 0.20 * f["has_candidate_answer"]
        - 0.15 * f["steps_norm"]
    )
    return float(max(0.0, min(1.0, v)))


def _terminal_value(ctx: ReasoningContext) -> float:
    pred = getattr(ctx, "answer", None)
    gold = (ctx.memory or {}).get("gold_answer", None)

    if gold is not None and pred is not None:
        return 1.0 if str(pred).strip() == str(gold).strip() else -1.0

    # weak plausibility
    q = (ctx.question or "").lower()
    wants_number = any(x in q for x in ["how many", "total", "sum", "average", "amount", "cost", "number"])
    if wants_number:
        try:
            float(str(pred).replace(",", "").strip())
            return 0.6
        except Exception:
            return -0.2
    return 0.2


@dataclass
class TQAEvaluationAgent:
    """
    Implements core.schemas.EvaluationAgent
      evaluate(ctx)-> EvalResult
    Mixed reward (C): process + terminal.

    If use_llm=True, refine the score using prompts/mcts_evaluation.*.txt
    """
    llm_client: Optional[Any] = None
    prompt_loader: PromptLoader = PromptLoader("prompts")
    prompt_name: str = "mcts_evaluation"

    use_llm: bool = False
    temperature: float = 0.0
    max_tokens: int = 500

    def evaluate(self, ctx: ReasoningContext) -> EvalResult:
        terminal = _is_terminal(ctx)

        if terminal:
            base = _terminal_value(ctx)
            score = float(max(-1.0, min(1.0, base)))
            critique = "terminal evaluation (deterministic base)"
            extra: Dict[str, Any] = {"terminal": True}
        else:
            base = _heuristic_process_value(ctx)
            score = float(max(0.0, min(1.0, base)))
            critique = "process evaluation (deterministic base)"
            extra = {"terminal": False, "features": _progress_features(ctx)}

        if self.use_llm and self.llm_client is not None:
            pack = self.prompt_loader.load(self.prompt_name)
            variables = {
                "question": ctx.question,
                "headers": str(list(ctx.view.headers)),
                "terminal": str(terminal),
                "hint_base_score": str(score),
                "memory_keys": str(sorted([k for k in (ctx.memory or {}).keys() if k != "llm_client"])),
            }
            system = pack.system.format(**variables)
            user = pack.user.format(**variables)

            try:
                j = json_chat_with_retry(
                    self.llm_client.chat, system, user,
                    max_retries=1,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if isinstance(j, dict) and "score" in j:
                    s = float(j["score"])
                    s = max(-1.0, min(1.0, s)) if terminal else max(0.0, min(1.0, s))
                    score = s
                    critique = str(j.get("critique", critique))
                    ex = j.get("extra", None)
                    if isinstance(ex, dict):
                        extra = {**extra, **ex}
                    extra["llm_used"] = True
            except Exception:
                pass

        if extra is None:
            extra = {}
        return EvalResult(score=score, critique=critique, extra=extra)
