# actions/table_retrieval/row_locating.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import re

from core.reasoning_context import ReasoningContext
from actions.base import Action, ActionError, llm_json
from actions.registry import register_action


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


@register_action
@dataclass
class RowLocating(Action):
    """
    3) Row Locating
    Deterministic: filter rows via constraints and/or row_contains.
    LLM-assisted: if neither constraints nor row_contains provided, ask LLM to output:
      {"constraints":[...], "combine":"and|or"}  OR  {"row_contains":"..."}
    Then code executes deterministically.
    """
    TYPE: str = "RowLocating"

    constraints: Optional[List[Dict[str, Any]]] = None
    row_contains: Optional[str] = None
    combine: str = "and"
    use_llm: bool = True
    out_key: str = "located_rows"

    def validate(self) -> None:
        if self.combine not in {"and", "or"}:
            raise ActionError("combine must be 'and' or 'or'.")
        if self.constraints is not None and not isinstance(self.constraints, list):
            raise ActionError("constraints must be list[dict] if provided.")

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view

        constraints = self.constraints
        row_contains = self.row_contains
        combine = self.combine
        llm_used = False

        # LLM synthesize constraints if missing
        if (not constraints) and (not row_contains) and self.use_llm:
            try:
                system = "You are a table row locator. Return STRICT JSON only."
                user = {
                    "question": ctx.question,
                    "headers": table.headers,
                    "task": "Propose row selection condition(s) to locate relevant rows for answering the question.",
                    "output_schema": {
                        "row_contains": "<optional phrase>",
                        "combine": "and|or",
                        "constraints": [
                            {"column": "<header>", "op": "==|!=|contains|>|>=|<|<=", "value": "<string or number>"}
                        ],
                    },
                    "constraints_hint": "If question references a year/category/name, prefer constraints; else use row_contains.",
                }
                out = llm_json(ctx, system, str(user))
                if isinstance(out, dict):
                    rc = out.get("row_contains")
                    if isinstance(rc, str) and rc.strip():
                        row_contains = rc.strip()
                    cc = out.get("constraints")
                    if isinstance(cc, list) and cc:
                        constraints = [c for c in cc if isinstance(c, dict)]
                    cmb = out.get("combine")
                    if cmb in {"and", "or"}:
                        combine = cmb
                llm_used = True
            except Exception:
                llm_used = False

        if (not constraints) and (not row_contains):
            raise ActionError("Provide constraints or row_contains (LLM unavailable/failed).")

        def to_num(x: str) -> Optional[float]:
            x = (x or "").strip()
            if x == "":
                return None
            x2 = x.replace(",", "")
            try:
                return float(x2)
            except ValueError:
                return None

        def check_constraint(row: List[str], c: Dict[str, Any]) -> bool:
            col = c.get("column")
            op = (c.get("op") or "").strip()
            val = str(c.get("value", "")).strip()
            if not col or not op:
                return False
            try:
                idx = table.resolve_col(col)
            except Exception:
                return False

            cell = row[idx] if idx < len(row) else ""
            cell_s = (cell or "").strip()

            if op in {"==", "="}:
                return cell_s == val
            if op == "!=":
                return cell_s != val
            if op.lower() == "contains":
                return _norm(val) in _norm(cell_s)

            if op in {">", ">=", "<", "<="}:
                a = to_num(cell_s)
                b = to_num(val)
                if a is None or b is None:
                    return False
                if op == ">":
                    return a > b
                if op == ">=":
                    return a >= b
                if op == "<":
                    return a < b
                if op == "<=":
                    return a <= b
            return False

        selected: List[int] = []
        for i, row in enumerate(table.rows):
            ok = True
            if row_contains:
                ok = _norm(row_contains) in _norm(" | ".join(row))
            if constraints:
                checks = [check_constraint(row, c) for c in constraints]
                ok = ok and (all(checks) if combine == "and" else any(checks))
            if ok:
                selected.append(i)

        ctx.memory[self.out_key] = selected
        obs = {
            "out_key": self.out_key,
            "row_indices": selected,
            "count": len(selected),
            "llm_used": llm_used,
            "combine": combine,
            "constraints_used": constraints or [],
            "row_contains_used": row_contains,
        }
        return ctx, obs
