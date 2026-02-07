# agents/utils/prompt_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PromptPack:
    system: str
    user: str


class PromptLoader:
    """
    Load prompts from prompts/ directory (sibling of agents/).
    Convention:
      prompts/<name>.system.txt
      prompts/<name>.user.txt
    """

    def __init__(self, prompts_dir: str = "prompts") -> None:
        self.root = Path(prompts_dir)
        self._cache: Dict[str, PromptPack] = {}

    def load(self, name: str) -> PromptPack:
        if name in self._cache:
            return self._cache[name]

        sys_path = self.root / f"{name}.system.txt"
        usr_path = self.root / f"{name}.user.txt"
        if not sys_path.exists() or not usr_path.exists():
            raise FileNotFoundError(f"Missing prompt files for '{name}': {sys_path}, {usr_path}")

        pack = PromptPack(
            system=sys_path.read_text(encoding="utf-8"),
            user=usr_path.read_text(encoding="utf-8"),
        )
        self._cache[name] = pack
        return pack

    def render(self, name: str, variables: Optional[Dict[str, str]] = None) -> PromptPack:
        pack = self.load(name)
        if not variables:
            return pack
        return PromptPack(
            system=pack.system.format(**variables),
            user=pack.user.format(**variables),
        )
