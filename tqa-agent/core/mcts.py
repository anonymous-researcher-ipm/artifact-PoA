from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import math
import random

from .mcts_node import MCTSNode
from .reasoning_context import ReasoningContext
from .trace import TraceStep, ReasoningPath
from .schemas import PlanningAgent, ExecutionAgent, EvaluationAgent, EvalResult, ActionSpec
from .heuristics import SearchHeuristics

def ucb_score(parent_N: int, child_Q: float, child_N: int, c: float = 1.4) -> float:
    if child_N == 0:
        return float("inf")
    return child_Q + c * math.sqrt(math.log(max(1, parent_N)) / child_N)

@dataclass
class MCTSRunner:
    """
    Phase 1: Use MCTS to generate K candidate reasoning paths.
    This runner is 'agent-driven': planners/executors/evaluators supply domain behavior.
    """
    planner: PlanningAgent
    executor: ExecutionAgent
    evaluator: EvaluationAgent
    heuristics: SearchHeuristics = SearchHeuristics()
    exploration_c: float = 1.4
    seed: int = 7

    def run(
        self,
        root_ctx: ReasoningContext,
        num_iters: int = 64,
        max_candidates: int = 8,
    ) -> List[ReasoningPath]:
        random.seed(self.seed)
        root = MCTSNode(ctx=root_ctx)

        candidates: List[ReasoningPath] = []

        for _ in range(num_iters):
            node = self._select(root)
            if node.ctx.done or self.heuristics.should_stop(node.ctx.depth):
                # already terminal -> collect
                self._maybe_collect(node, candidates, max_candidates)
                continue

            # Expansion: planner proposes candidate next actions (ordered)
            action_specs = self.planner.propose(node.ctx)

            # If planner returns nothing, treat as dead-end
            if not action_specs:
                dead = node.ctx.fork()
                dead.done = True
                dead.path.terminal = True
                dead.path.meta["dead_end"] = True
                leaf_eval = self.evaluator.evaluate(dead)
                self._backprop(node, leaf_eval.score)
                self._maybe_collect_ctx(dead, candidates, max_candidates, leaf_eval)
                continue

            # Choose one action to simulate this iteration (classic MCTS):
            # - If you want progressive widening later, you can.
            a_idx, a_spec = self._pick_untried_or_best(node, action_specs)

            # Execute -> new context + observation
            child_ctx, obs = self.executor.execute(node.ctx.fork(), a_spec)

            # Append trace step
            child_ctx.path.steps.append(TraceStep(action_spec=a_spec, observation=obs))
            child_ctx.depth = node.ctx.depth + 1

            # Evaluate (reward)
            er = self.evaluator.evaluate(child_ctx)

            # Optional heuristic: stop expanding low-quality branches
            if not self.heuristics.should_expand(er.score):
                child_ctx.done = True
                child_ctx.path.meta["pruned"] = True

            # Mark terminal if done
            if child_ctx.done:
                child_ctx.path.terminal = True
                child_ctx.path.final_answer = child_ctx.answer

            # Create child node
            child = MCTSNode(ctx=child_ctx, parent=node, last_action_index=a_idx)
            node.children[a_idx] = child

            # Backprop
            self._backprop(child, er.score)

            # Collect candidates
            self._maybe_collect(child, candidates, max_candidates, er)

            if len(candidates) >= max_candidates:
                break

        # sort by score desc
        candidates.sort(key=lambda p: p.total_score, reverse=True)
        return candidates[:max_candidates]

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and (not node.ctx.done):
            # choose child with best UCB
            best_k = None
            best_s = -float("inf")
            for k, ch in node.children.items():
                s = ucb_score(node.N + 1, ch.Q, ch.N, self.exploration_c)
                if s > best_s:
                    best_s = s
                    best_k = k
            node = node.children[best_k]  # type: ignore
        return node

    def _pick_untried_or_best(self, node: MCTSNode, action_specs: List[ActionSpec]) -> Tuple[int, ActionSpec]:
        # untried indices are those not in node.children
        for i, spec in enumerate(action_specs):
            if i not in node.children:
                return i, spec
        # if all tried, just pick the first (planner is assumed ranked)
        return 0, action_specs[0]

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.N += 1
            # incremental mean update
            cur.Q += (reward - cur.Q) / cur.N
            cur = cur.parent

    def _maybe_collect(
        self,
        node: MCTSNode,
        candidates: List[ReasoningPath],
        max_candidates: int,
        er: Optional[EvalResult] = None,
    ) -> None:
        self._maybe_collect_ctx(node.ctx, candidates, max_candidates, er)

    def _maybe_collect_ctx(
        self,
        ctx: ReasoningContext,
        candidates: List[ReasoningPath],
        max_candidates: int,
        er: Optional[EvalResult] = None,
    ) -> None:
        if len(candidates) >= max_candidates:
            return
        if ctx.done or ctx.path.terminal:
            p = ctx.path
            if er is not None:
                p.total_score = float(er.score)
                if er.extra:
                    p.meta = {**p.meta, **er.extra}
            else:
                # still provide something
                p.total_score = float(p.total_score or 0.0)
            p.final_answer = ctx.answer
            candidates.append(p)
