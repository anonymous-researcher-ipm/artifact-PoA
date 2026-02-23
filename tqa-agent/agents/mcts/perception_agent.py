# agents/mcts/context_sensing_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from core.reasoning_context import ReasoningContext
from agents.utils.json_compact import compact


@dataclass
class SimpleContextSensingAgent:
    max_recent_steps: int = 6
    max_memory_keys: int = 50

    def report(self, ctx: ReasoningContext) -> Dict[str, Any]:
        mem = ctx.memory or {}
        keys = [k for k in sorted(mem.keys()) if k not in {"llm_client", "general_kb_provider", "domain_kb_provider"}]
        keys = keys[: self.max_memory_keys]

        recent = []
        if getattr(ctx, "path", None) is not None and isinstance(getattr(ctx.path, "steps", None), list):
            for st in ctx.path.steps[-self.max_recent_steps:]:
                try:
                    a = getattr(st, "action_spec", None) or {}
                    recent.append({"type": a.get("type"), "error": getattr(st, "error", None)})
                except Exception:
                    continue

        return {
            "depth": int(getattr(ctx, "depth", 0)),
            "done": bool(getattr(ctx, "done", False)),
            "headers": list(getattr(ctx.view, "headers", [])),
            "known_memory_keys": keys,
            "recent_steps": recent,
            "memory_compact": compact({k: mem.get(k) for k in keys}, depth=2),
        }
