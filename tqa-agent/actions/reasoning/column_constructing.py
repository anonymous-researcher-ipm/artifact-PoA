# actions/reasoning/column_constructing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Union
import re

from core.reasoning_context import ReasoningContext
from core.table_context import TableContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


def _to_number(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "" or s.lower() in {"na", "n/a", "null", "none", "-"}:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].replace(",", "")) / 100.0
        except ValueError:
            return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


@register_action
@dataclass
class ColumnConstructing(Action):
    """
    4) Column Constructing
    Deterministic execution: build derived column by an arithmetic expr over existing columns.
    LLM-assisted: if expr missing, ask LLM to output {"expr":"..."} referencing headers as variables.
    """
    TYPE: str = "ColumnConstructing"

    new_column: str = "derived"
    expr: str = ""  # e.g., "planned + actual"
    insert_at: Union[str, int] = "end"
    missing_as_zero: bool = False
    use_llm: bool = True

    def validate(self) -> None:
        if not self.new_column:
            raise ActionError("new_column required.")

    def _safe_check_expr(self, expr: str) -> None:
        allowed = set("+-*/(). _")
        for ch in expr:
            if ch.isalnum() or ch in allowed:
                continue
            raise ActionError(f"Unsafe character in expr: {ch!r}")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view
        headers = list(table.headers)

        expr = (self.expr or "").strip()
        llm_used = False

        if (not expr) and self.use_llm:
            try:
                system = "You are a table reasoning helper. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "headers": headers,
                    "task": "Propose an arithmetic expression to compute a derived column needed for answering the question.",
                    "output_schema": {"expr": "<expression using header names as identifiers>"},
                    "constraints": [
                        "Use only + - * / ( ) and identifiers.",
                        "If a header has spaces, replace spaces with underscore in identifier (e.g., Planned Unit Cost -> Planned_Unit_Cost).",
                    ],
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict) and isinstance(out.get("expr"), str):
                    expr = out["expr"].strip()
                    llm_used = True
            except Exception:
                llm_used = False

        if not expr:
            raise ActionError("expr missing (LLM unavailable/failed).")

        # map identifiers to column indices (spaces -> underscore)
        header_id_map: Dict[str, str] = {}
        for h in headers:
            hid = re.sub(r"\s+", "_", h.strip())
            header_id_map[hid] = h

        self._safe_check_expr(expr)
        tokens = re.findall(r"[A-Za-z_]\w*", expr)
        col_vars: Dict[str, int] = {}
        for t in set(tokens):
            if t in header_id_map:
                col_vars[t] = table.resolve_col(header_id_map[t])
            else:
                # try direct resolve
                try:
                    col_vars[t] = table.resolve_col(t)
                except Exception:
                    pass

        derived_values: List[Optional[float]] = []
        for r in table.rows:
            local: Dict[str, Optional[float]] = {}
            for var, idx in col_vars.items():
                cell = r[idx] if idx < len(r) else ""
                num = _to_number(cell)
                if num is None and self.missing_as_zero:
                    num = 0.0
                local[var] = num

            if (not self.missing_as_zero) and any(local[v] is None for v in col_vars):
                val = None
            else:
                try:
                    val = eval(expr, {"__builtins__": {}}, {k: (v if v is not None else 0.0) for k, v in local.items()})  # type: ignore
                    if not isinstance(val, (int, float)):
                        val = None
                except Exception:
                    val = None
            derived_values.append(float(val) if val is not None else None)

        # insert new column
        if self.insert_at == "end":
            ins = len(headers)
        elif isinstance(self.insert_at, int):
            ins = max(0, min(self.insert_at, len(headers)))
        else:
            ins = len(headers)

        new_headers = list(headers)
        new_headers.insert(ins, self.new_column)

        new_rows: List[List[str]] = []
        for i, r in enumerate(table.rows):
            rr = list(r)
            if len(rr) < len(headers):
                rr += [""] * (len(headers) - len(rr))
            rr.insert(ins, "" if derived_values[i] is None else str(derived_values[i]))
            new_rows.append(rr)

        ctx.view = TableContext(new_headers, new_rows)

        obs = {"new_column": self.new_column, "expr": expr, "insert_at": ins, "llm_used": llm_used}
        return ctx, obs
