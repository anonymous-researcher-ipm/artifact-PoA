# actions/reasoning/row_constructing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Union

from core.reasoning_context import ReasoningContext
from core.table_context import TableContext
from actions.base import Action, ActionError
from actions.registry import register_action


@register_action
@dataclass
class RowConstructing(Action):
    """
    5) Row Constructing (deterministic)
    Construct a new aggregate row from selected rows.
    """
    TYPE: str = "RowConstructing"

    new_row_name: str = "DerivedRow"
    agg: str = "sum"  # sum/avg/min/max
    row_key: Optional[str] = "located_rows"
    rows: Optional[List[int]] = None
    insert_at: Union[str, int] = "end"
    name_column: Optional[str] = None

    def validate(self) -> None:
        if self.agg not in {"sum", "avg", "min", "max"}:
            raise ActionError("agg must be one of sum/avg/min/max.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view
        src_rows = self.rows
        if src_rows is None and self.row_key:
            v = ctx.memory.get(self.row_key, [])
            if isinstance(v, list):
                src_rows = [int(x) for x in v]
        if not src_rows:
            raise ActionError("No source rows to construct from.")

        def to_num(s: str) -> Optional[float]:
            try:
                return float((s or "").replace(",", "").strip())
            except Exception:
                return None

        cols = len(table.headers)
        agg_vals: List[str] = [""] * cols

        for c in range(cols):
            nums: List[float] = []
            for r_i in src_rows:
                if r_i < 0 or r_i >= len(table.rows):
                    continue
                cell = table.rows[r_i][c] if c < len(table.rows[r_i]) else ""
                n = to_num(cell)
                if n is not None:
                    nums.append(n)

            if nums:
                if self.agg == "sum":
                    val = sum(nums)
                elif self.agg == "avg":
                    val = sum(nums) / len(nums)
                elif self.agg == "min":
                    val = min(nums)
                else:
                    val = max(nums)
                agg_vals[c] = str(val)

        # set row name
        if self.name_column:
            try:
                name_idx = table.resolve_col(self.name_column)
            except Exception:
                name_idx = 0
        else:
            name_idx = 0
        if name_idx < len(agg_vals):
            agg_vals[name_idx] = self.new_row_name

        # insert
        new_rows = list(table.rows)
        if self.insert_at == "end":
            ins = len(new_rows)
        elif isinstance(self.insert_at, int):
            ins = max(0, min(self.insert_at, len(new_rows)))
        else:
            ins = len(new_rows)

        new_rows.insert(ins, agg_vals)
        ctx.view = TableContext(list(table.headers), new_rows)

        obs = {"new_row_name": self.new_row_name, "agg": self.agg, "insert_at": ins, "source_rows": src_rows}
        return ctx, obs
