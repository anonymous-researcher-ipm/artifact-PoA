from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

ActionSpec = Dict[str, Any]   # e.g., {"type":"SelectColumns", ...}
Observation = Dict[str, Any]  # arbitrary structured output from action execution

@dataclass
class EvalResult:
    score: float
    critique: str = ""
    extra: Dict[str, Any] = None

# ---- Agent contracts (implemented in agents/) ----

class ContextSensingAgent(Protocol):
    """
    Optional: produce a structured report of current node context
    to condition the planner (context-aware planning).
    """
    def report(self, ctx) -> Dict[str, Any]:
        ...

class PlanningAgent(Protocol):
    """
    Given current context, propose candidate next actions.
    Return a ranked list of action specs (can include priors if you want).
    """
    def propose(self, ctx) -> List[ActionSpec]:
        ...

class ExecutionAgent(Protocol):
    """
    Execute one action spec on a context, returning updated context + observation.
    """
    def execute(self, ctx, action_spec: ActionSpec) -> Tuple[Any, Observation]:
        ...

class EvaluationAgent(Protocol):
    """
    Evaluate a context/path (for reward / pruning / backprop).
    """
    def evaluate(self, ctx) -> EvalResult:
        ...

class DebateAgent(Protocol):
    """
    Optional: assess a candidate path and return critique/score.
    """
    def judge(self, question: str, path) -> EvalResult:
        ...

class DecisionAgent(Protocol):
    """
    Choose the best path from candidates and generate final answer if needed.
    """
    def decide(self, question: str, candidates: List[Any]) -> int:
        ...
