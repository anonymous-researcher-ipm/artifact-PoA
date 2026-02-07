# llm/factory.py
from __future__ import annotations

from typing import Any, Dict

from .errors import LLMConfigError
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .deepseek_client import DeepSeekClient


def build_llm_client(cfg: Dict[str, Any]) -> BaseLLMClient:
    """
    cfg example:
      {
        "provider": "openai" | "deepseek",
        "model": "gpt-4o" | "deepseek-chat" | ...,
        "client": <sdk client instance>,
        "temperature": 0.2,
        "max_tokens": 1024
      }
    """
    provider = (cfg.get("provider") or "").strip().lower()
    model = cfg.get("model")
    client = cfg.get("client")

    if not provider:
        raise LLMConfigError("llm.provider is required.")
    if not model or not isinstance(model, str):
        raise LLMConfigError("llm.model is required and must be a string.")
    if client is None:
        raise LLMConfigError("llm.client (SDK instance) is required.")

    temperature = float(cfg.get("temperature", 0.2))
    max_tokens = cfg.get("max_tokens", None)
    if max_tokens is not None:
        max_tokens = int(max_tokens)

    if provider == "openai":
        return OpenAIClient(
            client=client,
            model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )
    if provider == "deepseek":
        return DeepSeekClient(
            client=client,
            model=model,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    raise LLMConfigError(f"Unknown llm.provider: {provider}")
