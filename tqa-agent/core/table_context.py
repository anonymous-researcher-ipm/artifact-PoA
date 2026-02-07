from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence
import re

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

@dataclass
class TableContext:
    """
    Unified structured table representation for text tables.
    This is the substrate for all table-related actions.
    """
    headers: List[str]
    rows: List[List[str]]
    header_index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.header_index = {_norm(h): i for i, h in enumerate(self.headers)}

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)

    def resolve_col(self, name: str) -> int:
        key = _norm(name)
        if key in self.header_index:
            return self.header_index[key]

        # soft matching: substring containment
        cands = [(k, i) for k, i in self.header_index.items() if key in k or k in key]
        if not cands:
            raise KeyError(f"Column not found: {name!r}. Available: {self.headers}")
        cands.sort(key=lambda x: len(x[0]))  # prefer shortest match
        return cands[0][1]

    def col(self, name: str) -> List[str]:
        idx = self.resolve_col(name)
        return [r[idx] if idx < len(r) else "" for r in self.rows]

    def row(self, i: int) -> List[str]:
        return self.rows[i]

    def get_cell(self, row_i: int, col_name: str) -> str:
        idx = self.resolve_col(col_name)
        row = self.rows[row_i]
        return row[idx] if idx < len(row) else ""

    def select_cols(self, col_names: Sequence[str]) -> "TableContext":
        idxs = [self.resolve_col(c) for c in col_names]
        new_headers = [self.headers[i] for i in idxs]
        new_rows = [[(r[i] if i < len(r) else "") for i in idxs] for r in self.rows]
        return TableContext(new_headers, new_rows)

    def filter_rows(self, predicate: Callable[[List[str]], bool]) -> "TableContext":
        return TableContext(self.headers[:], [r for r in self.rows if predicate(r)])
