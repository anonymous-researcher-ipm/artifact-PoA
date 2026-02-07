# llm/json_utils.py
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional, Callable

from .errors import LLMParseError, LLMRequestError


_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json_block(text: str) -> str:
    if not text:
        raise LLMParseError("Empty LLM output.")
    m = _JSON_RE.search(text)
    if not m:
        raise LLMParseError("No JSON object/array found in LLM output.")
    return m.group(1)


def parse_json(text: str) -> Any:
    block = extract_json_block(text)
    try:
        return json.loads(block)
    except Exception as e:
        raise LLMParseError(f"JSON parse failed: {e}")


def json_chat_with_retry(
    chat_fn: Callable[..., str],
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int = 2,
    retry_backoff_s: float = 0.6,
    **kwargs,
) -> Any:
    """
    Call chat_fn(...) expecting STRICT JSON.
    If parsing fails, retry with a stronger system reminder.
    """
    last_err: Optional[Exception] = None
    sys = system_prompt

    for i in range(max_retries + 1):
        try:
            out = chat_fn(sys, user_prompt, **kwargs)
            return parse_json(out)
        except (LLMParseError, LLMRequestError) as e:
            last_err = e
            if i == max_retries:
                break
            # strengthen instruction
            sys = system_prompt + "\n\nIMPORTANT: Output MUST be STRICT JSON only. No explanations."
            time.sleep(retry_backoff_s * (i + 1))

    raise LLMParseError(f"json_chat_with_retry failed after retries: {last_err}")
