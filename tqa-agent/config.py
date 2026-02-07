# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import json


@dataclass
class LLMConfig:
    """
    LLM backend config.

    provider: e.g. "openai", "deepseek"
    model: e.g. "gpt-4o", "deepseek-v3"
    api_key_env: which env var holds the API key
    base_url: optional custom base url (some providers use it)
    timeout_s: request timeout seconds (if supported by your llm client)
    extra: provider-specific parameters
    """
    provider: str = "openai"
    model: str = "gpt-4o"

    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    timeout_s: int = 60

    # optional: organization / project / api_version / proxy / etc.
    extra: Dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


@dataclass
class MCTSConfig:
    num_iters: int = 64
    max_candidates: int = 5
    exploration_c: float = 1.4
    seed: int = 7


@dataclass
class EvalConfig:
    # Whether to use LLM to refine MCTS evaluation (core score already works deterministically)
    use_llm: bool = False
    temperature: float = 0.0
    max_tokens: int = 500


@dataclass
class PlanningConfig:
    topk: int = 5
    temperature: float = 0.0
    max_tokens: int = 900
    use_context_report: bool = True


@dataclass
class SelectionConfig:
    temperature: float = 0.0
    max_tokens: int = 1200
    use_verifier: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    # runtime
    prompts_dir: str = "prompts"
    # whether to print debug info
    verbose: bool = True

    @staticmethod
    def from_env() -> "AppConfig":
        """
        Minimal env-based configuration.
        You can override:
          TQA_PROVIDER, TQA_MODEL, TQA_API_KEY_ENV, TQA_BASE_URL
          TQA_NUM_ITERS, TQA_MAX_CANDIDATES, TQA_TOPK
          TQA_VERBOSE
        """
        cfg = AppConfig()

        cfg.llm.provider = os.getenv("TQA_PROVIDER", cfg.llm.provider)
        cfg.llm.model = os.getenv("TQA_MODEL", cfg.llm.model)
        cfg.llm.api_key_env = os.getenv("TQA_API_KEY_ENV", cfg.llm.api_key_env)
        cfg.llm.base_url = os.getenv("TQA_BASE_URL", cfg.llm.base_url) or cfg.llm.base_url

        if os.getenv("TQA_TIMEOUT_S"):
            cfg.llm.timeout_s = int(os.getenv("TQA_TIMEOUT_S", str(cfg.llm.timeout_s)))

        if os.getenv("TQA_NUM_ITERS"):
            cfg.mcts.num_iters = int(os.getenv("TQA_NUM_ITERS", str(cfg.mcts.num_iters)))
        if os.getenv("TQA_MAX_CANDIDATES"):
            cfg.mcts.max_candidates = int(os.getenv("TQA_MAX_CANDIDATES", str(cfg.mcts.max_candidates)))
        if os.getenv("TQA_EXPLORATION_C"):
            cfg.mcts.exploration_c = float(os.getenv("TQA_EXPLORATION_C", str(cfg.mcts.exploration_c)))
        if os.getenv("TQA_SEED"):
            cfg.mcts.seed = int(os.getenv("TQA_SEED", str(cfg.mcts.seed)))

        if os.getenv("TQA_TOPK"):
            cfg.planning.topk = int(os.getenv("TQA_TOPK", str(cfg.planning.topk)))

        if os.getenv("TQA_EVAL_USE_LLM"):
            cfg.evaluation.use_llm = os.getenv("TQA_EVAL_USE_LLM", "0") in {"1", "true", "True", "YES", "yes"}

        if os.getenv("TQA_VERBOSE"):
            cfg.verbose = os.getenv("TQA_VERBOSE", "1") in {"1", "true", "True", "YES", "yes"}

        return cfg

    @staticmethod
    def from_json(path: str) -> "AppConfig":
        """
        Optional: load all config from a json file.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        def _merge(dc_obj, dd: Dict[str, Any]) -> None:
            for k, v in dd.items():
                if hasattr(dc_obj, k):
                    cur = getattr(dc_obj, k)
                    if isinstance(cur, (LLMConfig, MCTSConfig, EvalConfig, PlanningConfig, SelectionConfig, AppConfig)) and isinstance(v, dict):
                        _merge(cur, v)
                    else:
                        setattr(dc_obj, k, v)

        cfg = AppConfig()
        if isinstance(d, dict):
            _merge(cfg, d)
        return cfg
