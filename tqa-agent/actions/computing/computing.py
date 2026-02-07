# actions/computing/computing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List, Union
import re
import math

from core.reasoning_context import ReasoningContext
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


def _safe_expr_check(expr: str) -> None:
    allowed = set("+-*/(). _")
    for ch in expr:
        if ch.isalnum() or ch in allowed:
            continue
        raise ActionError(f"Unsafe character in expr: {ch!r}")


@register_action
@dataclass
class Computing(Action):
    """
    8) Computing (LLM-plan + deterministic execution)

    Two-phase inside this action:
      (1) If needed, LLM generates a structured computation plan from context.
      (2) Deterministic code executes the plan and stores numeric result in ctx.memory[out_var].

    Plan schema (STRICT JSON):
      {
        "mode": "agg" | "expr",
        "out_var": "<string>",

        # agg mode
        "agg": "sum"|"avg"|"min"|"max"|"count",
        "column": "<header or alias>",
        "row_key": "<optional memory key of row indices>",

        # expr mode
        "expr": "<arithmetic expression using ctx.memory numeric vars>",
        "missing_as_zero": true|false
      }

    Notes:
    - LLM is allowed to choose between agg and expr based on ctx (question, located columns/rows, memory vars).
    - Deterministic execution is ALWAYS used for actual math.
    """

    TYPE: str = "Computing"

    # If mode is None/empty, we can let LLM decide.
    mode: str = "auto"  # "auto" | "expr" | "agg"
    # Fields for explicit (non-auto) usage
    agg: str = "sum"
    column: Optional[str] = None
    row_key: Optional[str] = None
    expr: Optional[str] = None

    out_var: str = "result"
    missing_as_zero: bool = False

    # enable LLM plan generation
    use_llm: bool = True

    def validate(self) -> None:
        if self.mode not in {"auto", "expr", "agg"}:
            raise ActionError("mode must be auto/expr/agg.")
        if not self.out_var:
            raise ActionError("out_var required.")

        if self.mode == "agg":
            if not self.column:
                raise ActionError("column required for agg mode.")
            if self.agg not in {"sum", "avg", "min", "max", "count"}:
                raise ActionError("invalid agg type.")
        if self.mode == "expr":
            if not self.expr:
                raise ActionError("expr required for expr mode.")

    def _build_context_brief(self, ctx: ReasoningContext) -> Dict[str, Any]:
        """
        Provide LLM with a compact, structured snapshot of the current context.
        Avoid dumping the entire table (could be huge).
        """
        table = ctx.view
        brief: Dict[str, Any] = {
            "question": ctx.question,
            "headers": list(table.headers),
            "memory_keys": sorted(list(ctx.memory.keys())),
        }

        # include located columns / rows if present
        lc = ctx.memory.get("located_columns")
        if isinstance(lc, list):
            # keep minimal form
            brief["located_columns"] = [
                {
                    "target": x.get("target"),
                    "matched": x.get("matched"),
                    "col_index": x.get("col_index"),
                }
                for x in lc
                if isinstance(x, dict)
            ]

        lr = ctx.memory.get("located_rows")
        if isinstance(lr, list):
            brief["located_rows_count"] = len(lr)
            brief["located_rows_sample"] = [int(x) for x in lr[:10] if isinstance(x, (int, float, str))]

        # include numeric vars summary
        numeric_vars: Dict[str, float] = {}
        for k, v in ctx.memory.items():
            if isinstance(v, (int, float)) and not (isinstance(v, bool)):
                numeric_vars[k] = float(v)
        if numeric_vars:
            brief["numeric_vars"] = numeric_vars

        # small row sample (first 3 rows) can help LLM choose correct column
        sample_rows = table.rows[:3]
        brief["row_sample"] = sample_rows

        return brief

    def _llm_generate_plan(self, ctx: ReasoningContext) -> Dict[str, Any]:
        system = (
            "You are a TableQA computation planner. "
            "Return STRICT JSON only, no extra text. "
            "Your job is to generate a computation plan that can be executed deterministically."
        )

        user = {
            "context": self._build_context_brief(ctx),
            "task": (
                "Generate a computation plan to advance toward answering the question. "
                "Choose mode='agg' when the result is an aggregation over a table column (optionally restricted to located rows). "
                "Choose mode='expr' when the result should be computed from existing numeric variables in memory."
            ),
            "output_schema": {
                "mode": "agg|expr",
                "out_var": "<string>",
                # agg
                "agg": "sum|avg|min|max|count",
                "column": "<header from headers>",
                "row_key": "<optional memory key for row indices>",
                # expr
                "expr": "<arithmetic expression over memory vars>",
                "missing_as_zero": True,
            },
            "constraints": [
                "If mode='agg', column MUST be exactly one of headers.",
                "If mode='expr', expr can only use identifiers (memory variable names), numbers, + - * / ( ).",
                "Prefer using existing located_rows (row_key='located_rows') when question suggests filtering.",
                "Set out_var to a concise variable name like 'x0' or 'result'.",
            ],
        }

        out = llm_json(ctx, system, str(user))
        if not isinstance(out, dict):
            raise ActionError("LLM plan must be a JSON dict.")
        return out

    def _normalize_plan(self, ctx: ReasoningContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize plan fields to safe deterministic execution.
        """
        mode = plan.get("mode")
        if mode not in {"agg", "expr"}:
            raise ActionError("LLM plan.mode must be 'agg' or 'expr'.")

        out_var = plan.get("out_var") or self.out_var
        if not isinstance(out_var, str) or not out_var.strip():
            out_var = self.out_var

        missing_as_zero = plan.get("missing_as_zero")
        if isinstance(missing_as_zero, bool):
            maz = missing_as_zero
        else:
            maz = self.missing_as_zero

        if mode == "agg":
            agg = plan.get("agg") or self.agg
            if agg not in {"sum", "avg", "min", "max", "count"}:
                raise ActionError("LLM plan.agg invalid.")

            column = plan.get("column") or self.column
            if not isinstance(column, str) or not column:
                raise ActionError("LLM plan.column missing for agg mode.")

            # Must be exact header
            headers = list(ctx.view.headers)
            if column not in headers:
                raise ActionError("LLM plan.column must be exactly one of headers.")

            row_key = plan.get("row_key")
            if row_key is not None and not isinstance(row_key, str):
                row_key = None

            return {
                "mode": "agg",
                "agg": agg,
                "column": column,
                "row_key": row_key,
                "out_var": out_var,
                "missing_as_zero": maz,
            }

        # expr
        expr = plan.get("expr") or self.expr
        if not isinstance(expr, str) or not expr.strip():
            raise ActionError("LLM plan.expr missing for expr mode.")
        expr = expr.strip()
        _safe_expr_check(expr)

        return {
            "mode": "expr",
            "expr": expr,
            "out_var": out_var,
            "missing_as_zero": maz,
        }

    def _exec_agg(self, ctx: ReasoningContext, agg: str, column: str, row_key: Optional[str], out_var: str, missing_as_zero: bool) -> Tuple[ReasoningContext, Dict[str, Any]]:
        table = ctx.view
        col_idx = table.resolve_col(column)

        rows = None
        if row_key:
            v = ctx.memory.get(row_key)
            if isinstance(v, list):
                rows = [int(x) for x in v if isinstance(x, (int, float, str)) and str(x).strip() != ""]
        if rows is None:
            rows = list(range(len(table.rows)))

        nums: List[float] = []
        for i in rows:
            if i < 0 or i >= len(table.rows):
                continue
            cell = table.rows[i][col_idx] if col_idx < len(table.rows[i]) else ""
            n = _to_number(cell)
            if n is None and missing_as_zero:
                n = 0.0
            if n is not None:
                nums.append(n)

        if agg == "count":
            val = float(len(nums))
        elif not nums:
            val = 0.0
        elif agg == "sum":
            val = float(sum(nums))
        elif agg == "avg":
            val = float(sum(nums) / len(nums))
        elif agg == "min":
            val = float(min(nums))
        else:
            val = float(max(nums))

        ctx.memory[out_var] = val
        obs = {
            "mode": "agg",
            "agg": agg,
            "column": column,
            "row_key": row_key,
            "rows_used": len(rows),
            "out_var": out_var,
            "value": val,
        }
        return ctx, obs

    def _exec_expr(self, ctx: ReasoningContext, expr: str, out_var: str, missing_as_zero: bool) -> Tuple[ReasoningContext, Dict[str, Any]]:
        tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[+\-*/().]", expr)
        if not tokens:
            raise ActionError(f"Bad expr: {expr}")

        local: Dict[str, float] = {}
        for t in tokens:
            if re.fullmatch(r"[A-Za-z_]\w*", t):
                if t in local:
                    continue
                v = ctx.memory.get(t)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    local[t] = float(v)
                else:
                    if missing_as_zero:
                        local[t] = 0.0
                    else:
                        raise ActionError(f"Variable {t} not found/numeric in ctx.memory for expr.")

        try:
            val = eval(expr, {"__builtins__": {}}, local)  # type: ignore
        except Exception as e:
            raise ActionError(f"Expr eval failed: {e}")

        if not isinstance(val, (int, float)) or math.isnan(float(val)):
            raise ActionError("Expr output not numeric.")

        valf = float(val)
        ctx.memory[out_var] = valf
        obs = {
            "mode": "expr",
            "expr": expr,
            "out_var": out_var,
            "value": valf,
            "vars_used": list(local.keys()),
        }
        return ctx, obs

    def apply(self, ctx: ReasoningContext) -> Tuple[ReasoningContext, Dict[str, Any]]:
        """
        Execution strategy:
          - If mode is explicit ('agg'/'expr') and required fields exist -> execute deterministically.
          - Otherwise (mode='auto' OR missing expr/column) and use_llm=True -> ask LLM for plan -> execute deterministically.
        """
        llm_used = False

        # 1) explicit deterministic path
        if self.mode == "agg":
            return self._exec_agg(ctx, self.agg, self.column or "", self.row_key, self.out_var, self.missing_as_zero)
        if self.mode == "expr":
            return self._exec_expr(ctx, self.expr or "", self.out_var, self.missing_as_zero)

        # 2) auto / missing fields -> LLM plan
        if not self.use_llm:
            raise ActionError("Computing in auto mode requires use_llm=True to generate a plan.")

        plan_raw = self._llm_generate_plan(ctx)
        plan = self._normalize_plan(ctx, plan_raw)
        llm_used = True

        if plan["mode"] == "agg":
            ctx, obs = self._exec_agg(
                ctx,
                plan["agg"],
                plan["column"],
                plan.get("row_key"),
                plan["out_var"],
                plan["missing_as_zero"],
            )
        else:
            ctx, obs = self._exec_expr(
                ctx,
                plan["expr"],
                plan["out_var"],
                plan["missing_as_zero"],
            )

        # attach plan summary for trace/debug
        obs["llm_used"] = llm_used
        obs["plan"] = plan
        return ctx, obs
