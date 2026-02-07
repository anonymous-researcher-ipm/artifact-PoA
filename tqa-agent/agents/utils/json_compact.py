# agents/utils/json_compact.py
from __future__ import annotations

from typing import Any, Dict, List


def compact(obj: Any, max_str: int = 800, max_list: int = 20, max_dict: int = 30, depth: int = 3) -> Any:
    """
    Make an object JSON-friendly and compact for prompts.
    """
    if depth <= 0:
        return "<depth_limit>"

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        return obj if len(obj) <= max_str else (obj[:max_str] + " ...[truncated]")

    if isinstance(obj, list):
        return [compact(x, max_str, max_list, max_dict, depth - 1) for x in obj[:max_list]]

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in list(obj.keys())[:max_dict]:
            out[str(k)] = compact(obj[k], max_str, max_list, max_dict, depth - 1)
        return out

    # fallback
    s = str(obj)
    return s if len(s) <= max_str else (s[:max_str] + " ...[truncated]")
