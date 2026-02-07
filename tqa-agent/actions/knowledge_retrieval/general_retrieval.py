# actions/knowledge_retrieval/general_retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


class GeneralKnowledgeProvider:
    def search(self, query: str, topk: int = 3) -> Dict[str, Any]:
        raise NotImplementedError


@register_action
@dataclass
class GeneralRetrieval(Action):
    """
    9) General Retrieval (Wiki-like)
    - If query missing and llm enabled, LLM generates query JSON {"query": "..."}.
    - Retrieval itself is executed by a provider if configured; otherwise stub.
    """
    TYPE: str = "GeneralRetrieval"

    query: str = ""
    topk: int = 3
    use_llm: bool = True
    out_key: str = "general_knowledge"

    def validate(self) -> None:
        if self.topk <= 0:
            raise ActionError("topk must be positive.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        q = (self.query or "").strip()
        llm_used = False

        if (not q) and self.use_llm:
            try:
                system = "You generate a concise search query. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "headers": ctx.view.headers,
                    "task": "Generate a short general-knowledge query to clarify ambiguous terms/abbreviations if needed.",
                    "output_schema": {"query": "<string>"},
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict) and isinstance(out.get("query"), str):
                    q = out["query"].strip()
                    llm_used = True
            except Exception:
                llm_used = False

        if not q:
            raise ActionError("query missing (LLM unavailable/failed).")

        provider = ctx.memory.get("general_kb_provider")
        if provider is None:
            result = {"query": q, "items": [], "note": "No general_kb_provider configured; stub result returned."}
        else:
            if not hasattr(provider, "search"):
                raise ActionError("general_kb_provider must implement search(query, topk).")
            result = provider.search(q, self.topk)  # type: ignore

        ctx.memory[self.out_key] = result
        obs = {"out_key": self.out_key, "query": q, "num_items": len(result.get("items", [])), "llm_used": llm_used}
        return ctx, obs
