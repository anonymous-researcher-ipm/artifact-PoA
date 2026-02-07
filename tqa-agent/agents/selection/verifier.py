# agents/selection/verifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SimpleVerifier:
    """
    Deterministic sanity checks used inside DecisionAgent.
    """
    def verify(self, question: str, answer: Any) -> Dict[str, Any]:
        ans = str(answer).strip()

        numeric = True
        try:
            float(ans.replace(",", ""))
        except Exception:
            numeric = False

        wants_number = any(x in (question or "").lower() for x in [
            "how many", "total", "sum", "average", "amount", "cost", "number"
        ])

        return {"numeric": numeric, "wants_number": wants_number}
