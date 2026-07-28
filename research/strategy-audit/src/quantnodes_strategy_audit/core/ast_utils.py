"""AST utilities shared across detectors."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator


def parse_file(file: Path) -> ast.Module | None:
    """Parse a Python file into an AST, returning None on error.

    Catches SyntaxError and ValueError (e.g. invalid encoding).
    """
    try:
        source = file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    try:
        return ast.parse(source, filename=str(file))
    except SyntaxError:
        return None


def iter_lines(file: Path) -> list[str]:
    """Read file lines, returning empty list on error."""
    try:
        return file.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []


def get_snippet(file: Path, line: int, context: int = 1) -> str:
    """Get a code snippet around a given line (1-indexed)."""
    lines = iter_lines(file)
    if not lines or line < 1 or line > len(lines):
        return ""
    start = max(1, line - context)
    end = min(len(lines), line + context)
    snippet_lines = lines[start - 1:end]
    return "\n".join(snippet_lines)


def get_call_chain(node: ast.Call) -> str:
    """Extract the call chain (e.g. 'df.groupby.mean') from a Call node.

    Walks back through .func/.value to build the full method chain.
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Call):
        if isinstance(current.func, ast.Attribute):
            parts.append(current.func.attr)
            current = current.func.value
        elif isinstance(current.func, ast.Name):
            parts.append(current.func.id)
            break
        else:
            break
    parts.reverse()
    return ".".join(parts)


def has_decorator(func_node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Check if a function has a specific decorator (e.g. '@nan_safe_pct_change')."""
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
    return False


def iter_call_names(tree: ast.AST) -> Iterator[tuple[str, ast.Call]]:
    """Yield (call_chain, node) for every Call in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = get_call_chain(node)
            yield chain, node


def is_method_call(node: ast.Call, method: str) -> bool:
    """Check if a Call node ends with a specific method name."""
    return get_call_chain(node).endswith(method)


def is_chained_call(node: ast.Call, chain: str) -> bool:
    """Check if a Call node matches a full method chain (e.g. 'nav.pct_change')."""
    return get_call_chain(node) == chain
