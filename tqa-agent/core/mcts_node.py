from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from .reasoning_context import ReasoningContext

@dataclass
class MCTSNode:
    """
    MCTS node for context-centric reasoning search.
    """
    ctx: ReasoningContext
    parent: Optional["MCTSNode"] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    # MCTS stats
    N: int = 0
    Q: float = 0.0  # mean value estimate
    prior: float = 1.0  # optional

    # bookkeeping
    last_action_index: Optional[int] = None
