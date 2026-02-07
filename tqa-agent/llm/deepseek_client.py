# llm/deepseek_client.py
from __future__ import annotations

from typing import Any, Optional

from .errors import LLMRequestError, LLMConfigError
from .base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """
    Adapter for DeepSeek client that supports OpenAI-compatible:
      client.chat.completions.create(...)
    """

    def __init__(
        self,
        client: Any,
        model: str,
        provider: str = "deepseek",
        default_temperature: float = 0.2,
        default_max_tokens: Optional[int] = None,
    ) -> None:
        if client is None:
            raise LLMConfigError("DeepSeek client is None.")
        self._client = client
        self.model = model
        self.provider = provider
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def chat(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            raise LLMRequestError(f"DeepSeek request failed: {e}")

        try:
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise LLMRequestError(f"DeepSeek response parse failed: {e}")
