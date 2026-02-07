# main.py
from __future__ import annotations

import argparse
import sys
from typing import Any, Optional, Tuple

from config import AppConfig

import actions

from core.reasoning_context import ReasoningContext
from core.mcts import MCTSRunner
from core.trace import ReasoningPath

from agents.utils.prompt_loader import PromptLoader
from agents.mcts.planning_agent import TQAPlanningAgent
from agents.mcts.execution_agent import TQAExecutionAgent
from agents.mcts.evaluation_agent import TQAEvaluationAgent
from agents.selection.decision_agent import PathDecisionAgent


# LLM client factory

def build_llm_client(cfg: AppConfig) -> Any:
    """
    We try to discover a client builder inside your llm/ package.
    This keeps main.py decoupled from your exact llm implementation details.

    Expected: the returned object has method:
      chat(system: str, user: str, **kwargs) -> dict/str (your json_utils handles parsing)
    """
    from llm.errors import LLMConfigError

    api_key = cfg.llm.api_key()
    # Some local deployments might not require api_key; keep it optional but warn.
    if cfg.verbose and not api_key:
        print(f"[warn] No API key found in env {cfg.llm.api_key_env}.", file=sys.stderr)

    # Try a few common factory entrypoints
    tried = []

    # 1) llm.factory.create_llm_client(config: dict | LLMConfig)
    try:
        from llm.factory import create_llm_client  # type: ignore
        return create_llm_client(cfg.llm)
    except Exception as e:
        tried.append(("llm.factory.create_llm_client", repr(e)))

    # 2) llm.client_factory.create_client(provider, model, api_key, base_url, **extra)
    try:
        from llm.client_factory import create_client  # type: ignore
        return create_client(
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            api_key=api_key,
            base_url=cfg.llm.base_url,
            timeout_s=cfg.llm.timeout_s,
            **(cfg.llm.extra or {}),
        )
    except Exception as e:
        tried.append(("llm.client_factory.create_client", repr(e)))

    # 3) llm.providers.get_client(LLMConfig)
    try:
        from llm.providers import get_client  # type: ignore
        return get_client(cfg.llm)
    except Exception as e:
        tried.append(("llm.providers.get_client", repr(e)))

    # 4) llm.client.LLMClient(provider/model...)
    try:
        from llm.client import LLMClient  # type: ignore
        return LLMClient(
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            api_key=api_key,
            base_url=cfg.llm.base_url,
            timeout_s=cfg.llm.timeout_s,
            **(cfg.llm.extra or {}),
        )
    except Exception as e:
        tried.append(("llm.client.LLMClient", repr(e)))

    msg_lines = [
        "Could not find a usable LLM client factory in llm/.",
        "Tried the following entrypoints:",
    ]
    for name, err in tried:
        msg_lines.append(f"  - {name}: {err}")
    msg_lines.append(
        "Fix: expose one of these functions/classes in llm/ or adjust build_llm_client() accordingly."
    )
    raise LLMConfigError("\n".join(msg_lines))


# Table parsing helpers

def load_table_text(path: Optional[str]) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_table_view(table_text: str) -> Any:
    """
    Your core likely already has a TableView/TableParser.
    We try a few likely imports; if your actual module name differs,
    just change the import here (only this function).
    """
    # 1) core.table_view.TableView.from_text
    try:
        from core.table_view import TableView  # type: ignore
        if hasattr(TableView, "from_text"):
            return TableView.from_text(table_text)
        return TableView(table_text)  # type: ignore
    except Exception:
        pass

    # 2) core.table_parser.parse_table_text
    try:
        from core.table_parser import parse_table_text  # type: ignore
        return parse_table_text(table_text)
    except Exception:
        pass

    # 3) fallback: a minimal TSV-like parser (safe default)
    lines = [ln.strip("\n") for ln in table_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty table text.")

    # detect delimiter
    delim = "\t" if ("\t" in lines[0]) else "|"
    raw = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(delim)]
        raw.append(parts)

    headers = raw[0]
    rows = raw[1:]

    class _FallbackTableView:
        def __init__(self, headers, rows):
            self.headers = headers
            self.rows = rows

        def resolve_col(self, name: str) -> int:
            if name in self.headers:
                return self.headers.index(name)
            raise ValueError(f"Unknown column: {name}")

    return _FallbackTableView(headers, rows)


def build_root_context(question: str, table_view: Any, llm_client: Any) -> ReasoningContext:
    ctx = ReasoningContext(
        question=question,
        view=table_view,
        memory={},
    )
    ctx.memory["llm_client"] = llm_client
    return ctx


# Pipeline

def run_pipeline(cfg: AppConfig, question: str, table_text: str) -> Tuple[Any, ReasoningPath]:
    llm_client = build_llm_client(cfg)
    table_view = build_table_view(table_text)
    root_ctx = build_root_context(question, table_view, llm_client)

    # Prompt loader for agents
    prompt_loader = PromptLoader(cfg.prompts_dir)

    # Phase 1: MCTS generate candidate paths
    planner = TQAPlanningAgent(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        prompt_name="mcts_planning",
        topk=cfg.planning.topk,
        temperature=cfg.planning.temperature,
        max_tokens=cfg.planning.max_tokens,
        use_context_report=cfg.planning.use_context_report,
    )
    executor = TQAExecutionAgent()

    evaluator = TQAEvaluationAgent(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        prompt_name="mcts_evaluation",
        use_llm=cfg.evaluation.use_llm,
        temperature=cfg.evaluation.temperature,
        max_tokens=cfg.evaluation.max_tokens,
    )

    runner = MCTSRunner(
        planner=planner,
        executor=executor,
        evaluator=evaluator,
        exploration_c=cfg.mcts.exploration_c,
        seed=cfg.mcts.seed,
    )

    candidates = runner.run(
        root_ctx=root_ctx,
        num_iters=cfg.mcts.num_iters,
        max_candidates=cfg.mcts.max_candidates,
    )

    if cfg.verbose:
        print(f"[info] Phase1 produced {len(candidates)} candidate paths.")
        for i, p in enumerate(candidates[: min(len(candidates), 5)]):
            print(f"  - cand#{i}: score={p.total_score:.4f} terminal={p.terminal} answer={p.final_answer}")

    if not candidates:
        raise RuntimeError("No candidate reasoning paths produced by MCTS.")

    # Phase 2: Selection (protocolized, no ctx needed)
    decider = PathDecisionAgent(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        temperature=cfg.selection.temperature,
        max_tokens=cfg.selection.max_tokens,
        use_verifier=cfg.selection.use_verifier,
    )

    best_idx = decider.decide(question, candidates)
    best_idx = max(0, min(best_idx, len(candidates) - 1))
    best_path = candidates[best_idx]

    final_answer = best_path.final_answer
    return final_answer, best_path


# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TQA-Agent runner (MCTS + Selection)")
    p.add_argument("--question", "-q", required=True, help="Question string.")
    p.add_argument("--table", "-t", required=True, help="Path to table text file. Use '-' for stdin.")
    p.add_argument("--config", "-c", default=None, help="Optional config.json path.")
    p.add_argument("--provider", default=None, help="Override LLM provider (e.g., openai, deepseek).")
    p.add_argument("--model", default=None, help="Override LLM model (e.g., gpt-4o, deepseek-v3).")
    p.add_argument("--api-key-env", default=None, help="Override API key env var name.")
    p.add_argument("--base-url", default=None, help="Override base URL.")
    p.add_argument("--iters", type=int, default=None, help="Override MCTS num_iters.")
    p.add_argument("--cands", type=int, default=None, help="Override MCTS max_candidates.")
    p.add_argument("--topk", type=int, default=None, help="Override planning topk.")
    p.add_argument("--eval-use-llm", action="store_true", help="Enable LLM refine for evaluation.")
    p.add_argument("--quiet", action="store_true", help="Disable verbose logs.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    cfg = AppConfig.from_env()
    if args.config:
        cfg = AppConfig.from_json(args.config)

    # CLI overrides
    if args.provider:
        cfg.llm.provider = args.provider
    if args.model:
        cfg.llm.model = args.model
    if args.api_key_env:
        cfg.llm.api_key_env = args.api_key_env
    if args.base_url:
        cfg.llm.base_url = args.base_url

    if args.iters is not None:
        cfg.mcts.num_iters = args.iters
    if args.cands is not None:
        cfg.mcts.max_candidates = args.cands
    if args.topk is not None:
        cfg.planning.topk = args.topk

    if args.eval_use_llm:
        cfg.evaluation.use_llm = True

    if args.quiet:
        cfg.verbose = False

    table_text = load_table_text(args.table)

    try:
        final_answer, best_path = run_pipeline(cfg, args.question, table_text)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # Output
    print("\n=== FINAL ANSWER ===")
    print(final_answer)

    if cfg.verbose:
        print("\n=== BEST PATH SUMMARY ===")
        print(f"score={best_path.total_score:.4f} terminal={best_path.terminal}")
        print(f"steps={len(best_path.steps)}")
        for i, st in enumerate(best_path.steps):
            a = st.action_spec or {}
            print(f"  [{i}] {a.get('type')}  err={st.error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
