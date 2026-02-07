from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class SearchHeuristics:
    """
    Pluggable heuristics for stopping and collecting paths.
    Keep it simple first; later you can align with your thesis-specific rules.
    """
    max_depth: int = 12
    min_score_to_expand: float = -1e9  # expand regardless by default

    def should_stop(self, depth: int) -> bool:
        return depth >= self.max_depth

    def should_expand(self, score: float) -> bool:
        return score >= self.min_score_to_expand
