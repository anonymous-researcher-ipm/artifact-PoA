from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class TraceStep:
    """
    One step on a reasoning path: action spec + observation + optional error.
    """
    action_spec: Dict[str, Any]
    observation: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    score: Optional[float] = None   # per-step evaluation if needed

@dataclass
class ReasoningPath:
    """
    A candidate reasoning path (root->leaf) produced in Phase 1.
    """
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Any = None
    terminal: bool = False
    total_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [asdict(s) for s in self.steps],
            "final_answer": self.final_answer,
            "terminal": self.terminal,
            "total_score": self.total_score,
            "meta": self.meta,
        }
