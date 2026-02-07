# agents/selection/adapter.py
from __future__ import annotations

from typing import Any, Dict, List
from core.trace import ReasoningPath


def _summarize(obs: Dict[str, Any], max_chars: int = 600) -> str:
    try:
        s = str(obs)
    except Exception:
        s = "<unprintable>"
    return s if len(s) <= max_chars else s[:max_chars] + " ...[truncated]"


def path_to_candidate(path: ReasoningPath, idx: int) -> Dict[str, Any]:
    steps_compact: List[Dict[str, Any]] = []
    for t, st in enumerate(path.steps):
        a = st.action_spec or {}
        steps_compact.append({
            "t": t,
            "action_type": a.get("type"),
            "action_spec": a,
            "observation_brief": _summarize(st.observation or {}),
            "error": st.error,
            "step_score": st.score,
        })

    return {
        "id": f"path_{idx}",
        "total_score": float(path.total_score),
        "terminal": bool(path.terminal),
        "final_answer": path.final_answer,
        "meta": path.meta or {},
        "steps": steps_compact,
    }


def build_candidates(paths: List[ReasoningPath]) -> List[Dict[str, Any]]:
    return [path_to_candidate(p, i) for i, p in enumerate(paths)]
