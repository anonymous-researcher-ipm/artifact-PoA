# actions/table_retrieval/header_parsing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import re

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json, ensure_keys
from actions.registry import register_action


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


@register_action
@dataclass
class HeaderParsing(Action):
    """
    1) Header Parsing
    - Deterministic: normalize headers + heuristic compound split.
    - LLM-assisted (optional): generate alias_map and header_groups (multi-level semantics),
      output is structured JSON and then stored into ctx.memory.
    """
    TYPE: str = "HeaderParsing"

    # If provided, used directly; else can be produced by LLM when use_llm=True
    aliases: Optional[Dict[str, str]] = None  # alias -> canonical header
    split_compound: bool = True
    use_llm: bool = True  # if llm_client exists, allow LLM to enhance parsing
    out_key: str = "header_info"

    def validate(self) -> None:
        if self.aliases is not None and not isinstance(self.aliases, dict):
            raise ActionError("aliases must be a dict if provided.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        headers = list(ctx.view.headers)
        norm_headers = [_norm(h) for h in headers]

        # deterministic compound split
        compounds: Dict[str, List[str]] = {}
        if self.split_compound:
            for h in headers:
                parts = [p.strip() for p in re.split(r"[/\n\-]+", h) if p.strip()]
                if len(parts) >= 2:
                    compounds[h] = parts

        alias_map: Dict[str, str] = {}
        if self.aliases:
            for k, v in self.aliases.items():
                alias_map[_norm(k)] = v

        llm_used = False
        header_groups: List[Dict[str, Any]] = []

        # LLM enhancement: only if enabled and no explicit aliases provided
        if self.use_llm and (self.aliases is None):
            try:
                system = (
                    "You are a table understanding component. "
                    "Return STRICT JSON only, no extra text."
                )
                user = {
                    "question": ctx.question,
                    "headers": headers,
                    "task": (
                        "Propose (1) alias_map for abbreviations/synonyms and "
                        "(2) optional header_groups capturing multi-level/semantic grouping if any."
                    ),
                    "output_schema": {
                        "alias_map": {"<alias>": "<canonical_header>"},
                        "header_groups": [
                            {"group": "<name>", "members": ["<header1>", "<header2>"]}
                        ],
                    },
                    "constraints": [
                        "alias_map keys should be short forms or synonyms; values must be from headers if possible.",
                        "header_groups members must be from headers.",
                    ],
                }
                out = llm_json(ctx, system, str(user))
                ensure_keys(out, ["alias_map", "header_groups"])
                if isinstance(out["alias_map"], dict):
                    for k, v in out["alias_map"].items():
                        if isinstance(k, str) and isinstance(v, str) and v:
                            alias_map[_norm(k)] = v
                if isinstance(out["header_groups"], list):
                    for g in out["header_groups"]:
                        if not isinstance(g, dict):
                            continue
                        members = g.get("members", [])
                        if isinstance(members, list):
                            members2 = [m for m in members if isinstance(m, str) and m in headers]
                            header_groups.append({"group": str(g.get("group", "")), "members": members2})
                llm_used = True
            except Exception:
                # safe fallback: just keep deterministic outputs
                llm_used = False

        info = {
            "headers": headers,
            "normalized_headers": norm_headers,
            "compound_splits": compounds,
            "alias_map": alias_map,
            "header_groups": header_groups,
            "llm_used": llm_used,
        }
        ctx.memory[self.out_key] = info

        obs = {
            "out_key": self.out_key,
            "num_headers": len(headers),
            "alias_count": len(alias_map),
            "compound_count": len(compounds),
            "group_count": len(header_groups),
            "llm_used": llm_used,
        }
        return ctx, obs
