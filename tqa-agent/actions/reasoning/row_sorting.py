# actions/reasoning/row_sorting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError
from actions.registry import register_action


def _to_num(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@register_action
@dataclass
class RowSorting(Action):
    """
    6) Row Sorting (deterministic)
    Sort rows by a column; store the sorted indices into memory.
    """
    TYPE: str = "RowSorting"

    by: str = ""
    order: str = "desc"
    numeric: bool = True
    row_key: Optional[str] = None
    out_key: str = "sorted_rows"

    def validate(self) -> None:
        if not self.by:
            raise ActionError("by must be provided.")
        if self.order not in {"asc", "desc"}:
            raise ActionError("order must be asc/desc.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view
        idx = table.resolve_col(self.by)

        subset = None
        if self.row_key:
            v = ctx.memory.get(self.row_key)
            if isinstance(v, list) and v:
                subset = [int(x) for x in v]

        indices = list(range(len(table.rows))) if subset is None else [i for i in subset if 0 <= i < len(table.rows)]

        def keyfun(i: int):
            cell = table.rows[i][idx] if idx < len(table.rows[i]) else ""
            if self.numeric:
                n = _to_num(cell)
                return -1e30 if n is None else n
            return cell or ""

        reverse = (self.order == "desc")
        sorted_idx = sorted(indices, key=keyfun, reverse=reverse)

        ctx.memory[self.out_key] = sorted_idx
        obs = {"by": self.by, "order": self.order, "numeric": self.numeric, "out_key": self.out_key, "row_indices": sorted_idx}
        return ctx, obs
