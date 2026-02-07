# actions/registry.py
from __future__ import annotations

from typing import Any, Dict, Type

from .base import Action, ActionError

ACTION_REGISTRY: Dict[str, Type[Action]] = {}


def register_action(cls: Type[Action]) -> Type[Action]:
    typ = getattr(cls, "TYPE", None) or cls.__name__
    cls.TYPE = typ
    ACTION_REGISTRY[typ] = cls
    return cls


def build_action(spec: Dict[str, Any]) -> Action:
    if not isinstance(spec, dict):
        raise ActionError(f"Action spec must be dict, got {type(spec)}")
    typ = spec.get("type")
    if not typ or not isinstance(typ, str):
        raise ActionError("Action spec missing string field 'type'.")

    cls = ACTION_REGISTRY.get(typ)
    if cls is None:
        raise ActionError(f"Unknown action type: {typ}. Known: {list(ACTION_REGISTRY.keys())}")

    kwargs = dict(spec)
    kwargs.pop("type", None)

    try:
        act = cls(**kwargs)  # type: ignore
    except TypeError as e:
        raise ActionError(f"Failed to construct action {typ}: {e}")

    act.validate()
    return act
