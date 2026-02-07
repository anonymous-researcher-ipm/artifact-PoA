# llm/errors.py
from __future__ import annotations


class LLMError(Exception):
    pass


class LLMConfigError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMRateLimitError(LLMRequestError):
    pass


class LLMTimeoutError(LLMRequestError):
    pass


class LLMParseError(LLMError):
    pass
