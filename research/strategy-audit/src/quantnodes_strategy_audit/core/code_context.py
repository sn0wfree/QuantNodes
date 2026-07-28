"""AST-based code context extraction.

Provides structured context around focus lines for Agent consumption.
Does NOT do semantic judgment — only extracts facts.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeContext:
    """Structured code context around focus lines."""

    file: str  # absolute path string
    focus_lines: list[int]
    imports: list[str] = field(default_factory=list)
    enclosing_function: dict | None = None
    called_by: list[str] = field(default_factory=list)
    data_flow: list[dict] = field(default_factory=list)
    variables: list[dict] = field(default_factory=list)
    control_flow: str = ""
    surrounding_lines: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "file": self.file,
            "focus_lines": self.focus_lines,
            "imports": self.imports,
            "enclosing_function": self.enclosing_function,
            "called_by": self.called_by,
            "data_flow": self.data_flow,
            "variables": self.variables,
            "control_flow": self.control_flow,
            "surrounding_lines": self.surrounding_lines,
        }


class CodeContextExtractor:
    """Extract structured context around focus lines in a Python file."""

    def extract(
        self,
        file: Path,
        focus_lines: list[int],
        depth: int = 2,
    ) -> CodeContext:
        """Extract context around focus_lines.

        Args:
            file: Python source file
            focus_lines: 1-indexed line numbers of interest
            depth: Context depth (lines of surrounding context)
        """
        file = Path(file).resolve()
        if not file.exists():
            return CodeContext(file=str(file), focus_lines=focus_lines)

        try:
            source = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return CodeContext(file=str(file), focus_lines=focus_lines)

        lines = source.splitlines()

        try:
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            return CodeContext(
                file=str(file),
                focus_lines=focus_lines,
                surrounding_lines=self._extract_surrounding_lines(lines, focus_lines, depth),
            )

        imports = self._extract_imports(tree)
        enclosing_func = self._find_enclosing_function(tree, focus_lines)
        called_by = self._find_callers(tree, file)
        variables = self._find_variables_in_focus(tree, focus_lines, lines)
        control_flow = self._describe_control_flow(tree, focus_lines)

        return CodeContext(
            file=str(file),
            focus_lines=focus_lines,
            imports=imports,
            enclosing_function=enclosing_func,
            called_by=called_by,
            variables=variables,
            control_flow=control_flow,
            surrounding_lines=self._extract_surrounding_lines(lines, focus_lines, depth),
        )

    def _extract_imports(self, tree: ast.Module) -> list[str]:
        """Extract top-level imports."""
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(alias.name for alias in node.names)
                imports.append(f"from {module} import {names}")
        return imports

    def _find_enclosing_function(
        self, tree: ast.Module, focus_lines: list[int]
    ) -> dict | None:
        """Find the function containing focus_lines."""
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    start = node.lineno
                    end = node.end_lineno or start
                    if any(start <= line <= end for line in focus_lines):
                        candidates.append((start, end, node))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1] - x[0])
        _, _, func = candidates[0]

        decorators = []
        for dec in func.decorator_list:
            decorators.append(ast.unparse(dec))

        return {
            "name": func.name,
            "lines": f"{func.lineno}-{func.end_lineno}",
            "docstring": ast.get_docstring(func),
            "decorators": decorators,
            "args": [arg.arg for arg in func.args.args],
        }

    def _find_callers(self, tree: ast.Module, file: Path) -> list[str]:
        """Find top-level callers (functions that call this function).

        Note: This is a simplified version — for production use, cross-file
        call graph analysis is needed.
        """
        # Find the enclosing function name
        enclosing = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosing = node.name
                break
        if not enclosing:
            return []
        # Without cross-file analysis, return empty
        return []

    def _find_variables_in_focus(
        self, tree: ast.Module, focus_lines: list[int], lines: list[str]
    ) -> list[dict]:
        """Find variables assigned or referenced near focus lines."""
        variables = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not hasattr(node, "lineno"):
                continue
            # Get focus window
            in_focus = any(
                abs(node.lineno - line) <= 5 for line in focus_lines
            )
            if not in_focus:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    variables.append({
                        "name": target.id,
                        "line": node.lineno,
                        "snippet": line_text.strip()[:200],
                    })
        return variables[:20]  # limit

    def _describe_control_flow(
        self, tree: ast.Module, focus_lines: list[int]
    ) -> str:
        """Describe control flow at focus lines."""
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While, ast.If)):
                continue
            if not hasattr(node, "lineno"):
                continue
            if any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in focus_lines):
                if isinstance(node, ast.For):
                    return f"for loop (line {node.lineno})"
                if isinstance(node, ast.While):
                    return f"while loop (line {node.lineno})"
                if isinstance(node, ast.If):
                    return f"if branch (line {node.lineno})"
        return "no loop/branch at focus"

    def _extract_surrounding_lines(
        self, lines: list[str], focus_lines: list[int], depth: int
    ) -> list[dict]:
        """Extract lines around focus_lines."""
        if not focus_lines:
            return []
        start = max(1, min(focus_lines) - depth)
        end = min(len(lines), max(focus_lines) + depth)
        return [
            {"line": i, "content": lines[i - 1]}
            for i in range(start, end + 1)
        ]
