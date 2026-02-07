from __future__ import annotations
from typing import List, Optional
import csv, io, re
from .table_context import TableContext

def _infer_delim(lines: List[str]) -> Optional[str]:
    for ln in lines[:3]:
        if "\t" in ln:
            return "\t"
    for ln in lines[:3]:
        if "," in ln:
            return ","
    return None

def _looks_like_md_pipe(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    return ("|" in lines[0]) and ("|" in lines[1]) and bool(re.search(r"-{3,}", lines[1]))

def _parse_md_pipe(lines: List[str]) -> TableContext:
    def split_row(ln: str) -> List[str]:
        ln = ln.strip()
        if ln.startswith("|"): ln = ln[1:]
        if ln.endswith("|"): ln = ln[:-1]
        return [p.strip() for p in ln.split("|")]

    header = split_row(lines[0])
    body = lines[2:] if len(lines) >= 3 else []
    rows = [split_row(ln) for ln in body if ln.strip()]
    rows = [r[:len(header)] + [""] * max(0, len(header) - len(r)) for r in rows]
    return TableContext(header, rows)

def _parse_delimited(lines: List[str], delim: str) -> TableContext:
    buf = io.StringIO("\n".join(lines))
    reader = csv.reader(buf, delimiter=delim)
    table = [row for row in reader if any(c.strip() for c in row)]
    if len(table) < 2:
        raise ValueError("Delimited table must have header + at least one row.")
    header = [c.strip() for c in table[0]]
    rows = [[c.strip() for c in r] for r in table[1:]]
    rows = [r[:len(header)] + [""] * max(0, len(header) - len(r)) for r in rows]
    return TableContext(header, rows)

def _parse_ws_aligned(lines: List[str]) -> TableContext:
    def split_ws(ln: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"\s{2,}", ln.strip())]
        if len(parts) <= 1:
            parts = [p.strip() for p in re.split(r"\s+", ln.strip())]
        return parts

    header = split_ws(lines[0])
    rows = [split_ws(ln) for ln in lines[1:]]
    rows = [r[:len(header)] + [""] * max(0, len(header) - len(r)) for r in rows]
    return TableContext(header, rows)

def parse_table_text(text: str) -> TableContext:
    """
    Parse text-form tables into TableContext.
    Supported:
      - Markdown pipe tables
      - CSV/TSV
      - whitespace-aligned tables
    """
    raw = (text or "").strip("\n")
    if not raw.strip():
        raise ValueError("Empty table text.")
    lines = [ln.rstrip("\n") for ln in raw.splitlines() if ln.strip()]
    if any("|" in ln for ln in lines) and _looks_like_md_pipe(lines):
        return _parse_md_pipe(lines)
    delim = _infer_delim(lines)
    if delim in {",", "\t"} and lines[0].count(delim) >= 1:
        return _parse_delimited(lines, delim)
    return _parse_ws_aligned(lines)
