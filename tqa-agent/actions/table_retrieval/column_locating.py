# actions/table_retrieval/column_locating.py
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
class ColumnLocating(Action):
    """
    2) Column Locating
    Deterministic: soft matching against headers (+ alias_map if present).
    LLM-assisted: if targets missing OR ambiguous, ask LLM to output:
      [{"target": "...", "matched_header": "..."}]
    Then code validates matched_header belongs to headers and stores indices.
    """
    TYPE: str = "ColumnLocating"

    targets: Optional[List[str]] = None
    mode: str = "soft"          # exact|soft
    use_llm: bool = True
    out_key: str = "located_columns"

    def validate(self) -> None:
        if self.mode not in {"exact", "soft"}:
            raise ActionError("mode must be 'exact' or 'soft'.")
        if self.targets is not None and not isinstance(self.targets, list):
            raise ActionError("targets must be list[str] if provided.")

    def _deterministic_match(self, headers: List[str], targets: List[str], alias_map: Dict[str, str]) -> List[Dict[str, Any]]:
        norm_headers = [_norm(h) for h in headers]
        matches: List[Dict[str, Any]] = []
        for t in targets:
            t_norm = _norm(t)
            if t_norm in alias_map:
                t_norm = _norm(alias_map[t_norm])

            found_idx: Optional[int] = None
            found_name: Optional[str] = None

            if self.mode == "exact":
                for i, nh in enumerate(norm_headers):
                    if nh == t_norm:
                        found_idx, found_name = i, headers[i]
                        break
            else:
                best = None
                for i, nh in enumerate(norm_headers):
                    score = 0
                    if t_norm == nh:
                        score = 100
                    elif t_norm in nh or nh in t_norm:
                        score = 60
                    else:
                        a = set(t_norm.split())
                        b = set(nh.split())
                        inter = len(a & b)
                        if inter:
                            score = 10 * inter
                    if best is None or score > best[0]:
                        best = (score, i)
                if best and best[0] > 0:
                    found_idx = best[1]
                    found_name = headers[found_idx]

            matches.append({"target": t, "matched": found_name, "col_index": found_idx})
        return matches

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        headers = list(ctx.view.headers)

        header_info = ctx.memory.get("header_info", {})
        alias_map = header_info.get("alias_map", {}) if isinstance(header_info, dict) else {}
        if not isinstance(alias_map, dict):
            alias_map = {}

        llm_used = False
        matches: List[Dict[str, Any]] = []

        # If targets absent, LLM should propose them + mapping
        if (not self.targets) and self.use_llm:
            try:
                system = "You are a table column locator. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "headers": headers,
                    "alias_map": alias_map,
                    "task": "Identify which column headers are needed to answer the question.",
                    "output_schema": [
                        {"target": "<concept from question>", "matched_header": "<one header from headers>"}
                    ],
                    "constraints": [
                        "matched_header MUST be exactly one of headers.",
                        "Return 1-5 items depending on need.",
                    ],
                }
                out = llm_json(ctx, system, str(user))
                if not isinstance(out, list):
                    raise ActionError("LLM must output a JSON list.")
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    mh = item.get("matched_header")
                    if isinstance(mh, str) and mh in headers:
                        matches.append({
                            "target": str(item.get("target", "")),
                            "matched": mh,
                            "col_index": headers.index(mh),
                        })
                llm_used = True
            except Exception:
                llm_used = False

        # Otherwise deterministic (or fallback)
        if not matches:
            if not self.targets:
                raise ActionError("targets missing and LLM locate failed/unavailable.")
            matches = self._deterministic_match(headers, self.targets, alias_map)

        ctx.memory[self.out_key] = matches
        obs = {"out_key": self.out_key, "matches": matches, "llm_used": llm_used}
        return ctx, obs
