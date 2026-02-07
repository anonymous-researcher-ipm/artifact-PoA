# actions/reasoning/grouping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import re

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError
from actions.registry import register_action


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


@register_action
@dataclass
class Grouping(Action):
    """
    7) Grouping (deterministic)
    Group rows by a column; optionally compute per-group aggregates.
    """
    TYPE: str = "Grouping"

    group_by: str = ""
    agg_col: Optional[str] = None
    agg: str = "sum"  # sum/avg/count
    out_key: str = "groups"

    def validate(self) -> None:
        if not self.group_by:
            raise ActionError("group_by is required.")
        if self.agg not in {"sum", "avg", "count"}:
            raise ActionError("agg must be sum/avg/count.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view
        g_idx = table.resolve_col(self.group_by)

        a_idx = None
        if self.agg_col:
            a_idx = table.resolve_col(self.agg_col)

        groups: Dict[str, List[int]] = {}
        for i, row in enumerate(table.rows):
            key = row[g_idx] if g_idx < len(row) else ""
            groups.setdefault(_norm(key), []).append(i)

        aggregates: Dict[str, Any] = {}
        if a_idx is not None:
            def to_num(x: str) -> Optional[float]:
                x = (x or "").strip().replace(",", "")
                try:
                    return float(x)
                except Exception:
                    return None

            for k, idxs in groups.items():
                if self.agg == "count":
                    aggregates[k] = len(idxs)
                else:
                    nums = []
                    for ii in idxs:
                        cell = table.rows[ii][a_idx] if a_idx < len(table.rows[ii]) else ""
                        n = to_num(cell)
                        if n is not None:
                            nums.append(n)
                    if not nums:
                        aggregates[k] = None
                    elif self.agg == "sum":
                        aggregates[k] = sum(nums)
                    else:
                        aggregates[k] = sum(nums) / len(nums)

        ctx.memory[self.out_key] = {"group_by": self.group_by, "agg_col": self.agg_col, "agg": self.agg, "groups": groups, "aggregates": aggregates}
        obs = {"out_key": self.out_key, "num_groups": len(groups), "has_aggregates": bool(aggregates)}
        return ctx, obs
