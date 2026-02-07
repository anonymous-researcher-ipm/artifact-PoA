from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from .table_context import TableContext
from .trace import ReasoningPath

@dataclass
class ReasoningContext:
    """
    The node context in the reasoning tree (Phase 1).
    This is the "context-centric" state that actions/agents operate on.
    """
    table: TableContext                      # immutable original table
    view: Optional[TableContext] = None      # derived current view
    question: str = ""                       # current question (can be rewritten)
    memory: Dict[str, Any] = field(default_factory=dict)  # intermediate variables / evidence
    path: ReasoningPath = field(default_factory=ReasoningPath)
    done: bool = False
    answer: Any = None
    depth: int = 0

    def __post_init__(self) -> None:
        if self.view is None:
            self.view = self.table

    def fork(self) -> "ReasoningContext":
        # shallow-copy table/view; deep-copy memory/path steps
        new = ReasoningContext(
            table=self.table,
            view=self.view,
            question=self.question,
            memory=dict(self.memory),
            path=ReasoningPath(
                steps=list(self.path.steps),
                final_answer=self.path.final_answer,
                terminal=self.path.terminal,
                total_score=self.path.total_score,
                meta=dict(self.path.meta),
            ),
            done=self.done,
            answer=self.answer,
            depth=self.depth,
        )
        return new
