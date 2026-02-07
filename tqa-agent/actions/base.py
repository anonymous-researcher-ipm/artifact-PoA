# actions/base.py
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple, Optional, ClassVar, Protocol, List
import json
import re

from core.reasoning_context import ReasoningContext


class ActionError(Exception):
    """Raised when an action cannot be applied under current context."""


class LLMClient(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        ...


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ActionError(msg)


def _as_dict_dataclass(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields(obj):
        out[f.name] = getattr(obj, f.name)
    out["type"] = getattr(obj, "TYPE", obj.__class__.__name__)
    return out


def get_llm(ctx: ReasoningContext) -> Optional[LLMClient]:
    client = ctx.memory.get("llm_client")
    if client is None:
        return None
    # duck-typing
    if not hasattr(client, "chat"):
        return None
    return client  # type: ignore


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object or array from text.
    Robust to LLM wrapping with explanations.
    """
    if not text:
        raise ActionError("Empty LLM output.")
    # find first { ... } or [ ... ]
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        raise ActionError("No JSON object/array found in LLM output.")
    return m.group(1)


def llm_json(
    ctx: ReasoningContext,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    llm = get_llm(ctx)
    if llm is None:
        raise ActionError("No llm_client configured in ctx.memory['llm_client'].")
    raw = llm.chat(system_prompt, user_prompt)
    jtxt = _extract_json(raw)
    try:
        return json.loads(jtxt)
    except Exception as e:
        raise ActionError(f"Failed to parse JSON from LLM output: {e}")


def ensure_keys(obj: Any, required: List[str]) -> None:
    if not isinstance(obj, dict):
        raise ActionError(f"Expected dict JSON, got {type(obj)}")
    for k in required:
        if k not in obj:
            raise ActionError(f"LLM JSON missing required key: {k}")


@dataclass
class Action:
    """
    Base class for all actions.
    Each action aligns to spec dict:
      {"type": "<ActionType>", ...fields...}
    """
    TYPE: ClassVar[str] = "Action"

    # Optional cost estimate
    approx_cost: float = 0.0

    def validate(self) -> None:
        return

    def describe(self) -> Dict[str, Any]:
        return _as_dict_dataclass(self)

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        raise NotImplementedError
