# llm/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol

from .errors import LLMRequestError


@dataclass
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: Optional[LLMUsage] = None
    raw: Optional[Dict[str, Any]] = None


class BaseLLMClient(Protocol):
    """
    Your whole codebase should depend on this minimal interface.
    """
    def chat(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Return raw text content (string).
        - agents/actions can parse JSON from it if needed
        - kwargs allows passing temperature, max_tokens, etc.
        """
        ...
