# actions/knowledge_retrieval/domain_specific_retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


class DomainKnowledgeProvider:
    def lookup(self, term: str, topk: int = 3) -> Dict[str, Any]:
        raise NotImplementedError


@register_action
@dataclass
class DomainSpecificRetrieval(Action):
    """
    10) Domain-Specific Retrieval (FIBO-like)
    - If term missing and llm enabled, LLM generates term JSON {"term": "..."}.
    - Retrieval is executed by a domain provider if configured; otherwise stub.
    """
    TYPE: str = "DomainSpecificRetrieval"

    term: str = ""
    topk: int = 3
    use_llm: bool = True
    out_key: str = "domain_knowledge"

    def validate(self) -> None:
        if self.topk <= 0:
            raise ActionError("topk must be positive.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        term = (self.term or "").strip()
        llm_used = False

        if (not term) and self.use_llm:
            try:
                system = "You generate a domain glossary lookup term. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "headers": ctx.view.headers,
                    "task": "Generate a domain-specific term for lookup (e.g., finance) if the question contains specialized jargon.",
                    "output_schema": {"term": "<string>"},
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict) and isinstance(out.get("term"), str):
                    term = out["term"].strip()
                    llm_used = True
            except Exception:
                llm_used = False

        if not term:
            raise ActionError("term missing (LLM unavailable/failed).")

        provider = ctx.memory.get("domain_kb_provider")
        if provider is None:
            result = {"term": term, "items": [], "note": "No domain_kb_provider configured; stub result returned."}
        else:
            if not hasattr(provider, "lookup"):
                raise ActionError("domain_kb_provider must implement lookup(term, topk).")
            result = provider.lookup(term, self.topk)  # type: ignore

        ctx.memory[self.out_key] = result
        obs = {"out_key": self.out_key, "term": term, "num_items": len(result.get("items", [])), "llm_used": llm_used}
        return ctx, obs
