# coding=utf-8
"""
代码安全沙箱

提供代码安全校验和执行环境。
"""
from __future__ import annotations

import ast
import re
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class CodeValidationResult:
    """代码验证结果"""
    is_safe: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    warnings_only: bool = False


class DangerousCodeError(Exception):
    """危险代码异常"""
    pass


class CodeSandbox:
    """
    代码安全沙箱

    提供代码安全校验，防止执行危险操作。

    Examples:
        >>> sandbox = CodeSandbox()
        >>> result = sandbox.validate("import os\\nos.system('ls')")
        >>> print(result.is_safe)  # False
    """

    DANGEROUS_IMPORTS: Set[str] = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'httplib', 'ftplib', 'telnetlib', 'telnet', 'poplib', 'imaplib',
        'smtplib', 'nntplib', 'anydbm', 'dbhash', 'gdbm', 'dbm',
        'marshal', 'pickle', 'cPickle', 'shelve', 'anydbm',
        'threading', 'multiprocessing', 'concurrent',
        'ctypes', 'cffi', 'mmap', 'resource', 'signal',
        'pty', 'tty', 'termios', 'fcntl', 'grp', 'pwd',
        'platform', 'syslog', 'crypt', 'spwd',
        'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma',
        'tempfile', 'glob', 'fnmatch', 'linecache', 'macpath',
        'macurl2path', 'mailcap', 'mimetypes', 'MimeWriter',
        'mimify', 'multifile', 'mutex', 'newdir', 'rexec',
        'robotparser', 'user', 'whichdb', 'xdrlib',
    }

    DANGEROUS_PATTERNS: List[str] = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'__import__\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'vars\s*\(',
        r'locals\s*\(',
        r'globals\s*\(',
        r'mro\s*\(',
        r'__subclasses__\s*\(',
        r'__bases__\s*\(',
        r'__init__\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'print\s*\(',
        r'execfile\s*\(',
        r'runpy\s*\(',
        r'os\.system\s*\(',
        r'os\.popen\s*\(',
        r'subprocess\.',
        r'socket\.',
        r'shelve\.open',
        r'pickle\.load',
        r'pickle\.loads',
        r'marshal\.load',
        r'yaml\.load',
        r'yaml\.unsafe_load',
    ]

    ALLOWED_PATTERNS: List[str] = [
        r'^import\s+quantnodes',
        r'^from\s+quantnodes',
        r'^import\s+pandas',
        r'^from\s+pandas',
        r'^import\s+numpy',
        r'^from\s+numpy',
    ]

    def __init__(
        self,
        allow_warnings: bool = False,
        max_code_length: int = 10000,
        # ===== PR-QN-1 (2026-06-21): 实例级可配置白/黑名单 =====
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        # ===== PR-QN-4 (2026-06-22): 默认引擎 =====
        default_engine: str = "polars",
        **kwargs
    ):
        """
        初始化代码沙箱

        Args:
            allow_warnings: 是否允许警告（不阻断执行）
            max_code_length: 最大代码长度
            allowed_imports: 追加到白名单的 import pattern (regex 列表), 默认 None.
                实例级配置. 已默认允许的 (quantnodes/pandas/numpy) 不受影响.
                PR-QN-1 新增, 之前需 monkey-patch 类属性.
            blocked_imports: 追加到黑名单的 import pattern (regex/字面量列表), 默认 None.
                实例级配置. 增强默认黑名单 (60+ 危险模块).
            default_engine: 默认引擎 ("polars" | "pandas" | "auto").
                PR-QN-4 新增. "polars" (默认) 保持向后兼容;
                "auto" 启用 import 扫描自动检测.

        Note:
            默认参数 (allowed_imports/blocked_imports 均为 None) 时, 行为与 PR-QN-1
            之前**完全一致** — 现有 4608+ tests 无需任何修改.
        """
        self.allow_warnings = allow_warnings
        self.max_code_length = max_code_length
        # PR-QN-1: 实例级白/黑名单 (拷贝类级别作为基础, 再追加用户配置)
        self._allowed_patterns: List[str] = list(self.ALLOWED_PATTERNS) + (
            allowed_imports or []
        )
        self._blocked_imports: Set[str] = set(self.DANGEROUS_IMPORTS) | set(
            blocked_imports or []
        )
        # PR-QN-4: 默认引擎
        self.default_engine: str = default_engine
        self.logger = logging.getLogger(f"sandbox.{self.__class__.__name__}")

    def _detect_engine(self, code: str) -> str:
        """Detect engine from code (PR-QN-4).

        Scans import statements to determine polars or pandas.
        Returns self.default_engine when no imports detected or when
        default_engine is not "auto".
        """
        if self.default_engine != "auto":
            return self.default_engine
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "polars"
        has_pl = has_pd = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "polars" or alias.name.startswith("polars."):
                        has_pl = True
                    elif alias.name == "pandas" or alias.name.startswith("pandas."):
                        has_pd = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "polars" or node.module.startswith("polars.")):
                    has_pl = True
                elif node.module and (node.module == "pandas" or node.module.startswith("pandas.")):
                    has_pd = True
        if has_pd and not has_pl:
            return "pandas"
        return "polars"

    def validate(self, code: str) -> CodeValidationResult:
        """
        验证代码安全性

        Args:
            code: 待验证的代码

        Returns:
            CodeValidationResult 验证结果
        """
        result = CodeValidationResult(is_safe=True)

        if not code or not code.strip():
            result.is_safe = False
            result.errors.append("Empty code")
            return result

        if len(code) > self.max_code_length:
            result.is_safe = False
            result.errors.append(f"Code exceeds max length ({self.max_code_length})")
            return result

        result.warnings.extend(self._check_dangerous_imports(code))
        result.errors.extend(self._check_dangerous_patterns(code))

        if result.errors:
            result.is_safe = False
        elif result.warnings and not self.allow_warnings:
            result.is_safe = False
            result.warnings_only = True

        return result

    def _check_dangerous_imports(self, code: str) -> List[str]:
        """检查危险导入 (PR-QN-1: 读 self._blocked_imports 实例属性)"""
        warnings = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in self._blocked_imports:
                            warnings.append(f"Dangerous import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in self._blocked_imports:
                        warnings.append(f"Dangerous import: from {node.module}")

        except SyntaxError:
            pass

        return warnings

    def _check_dangerous_patterns(self, code: str) -> List[str]:
        """检查危险模式"""
        errors = []

        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected: {pattern}")

        return errors

    def validate_and_execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
    ) -> Any:
        """
        验证并执行代码

        Args:
            code: 待执行的代码
            context: 执行上下文
            engine: 引擎覆盖 ("polars"|"pandas"|None). None = use default_engine.

        Returns:
            执行结果

        Raises:
            DangerousCodeError: 代码不安全
            SyntaxError: 代码语法错误
        """
        result = self.validate(code)

        # PR-QN-4: detect engine (if not explicitly overridden)
        detected_engine = engine or self._detect_engine(code)

        if not result.is_safe:
            if result.warnings_only:
                self.logger.warning(f"Code has warnings: {result.warnings}")
            else:
                raise DangerousCodeError(f"Code validation failed: {result.errors}")

        context = context or {}
        safe_builtins = {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'bytes': bytes,
            'chr': chr,
            'dict': dict,
            'dir': dir,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hash': hash,
            'hex': hex,
            'id': id,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'object': object,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }

        try:
            compiled = compile(code, '<string>', 'exec')
            exec_globals = {**safe_builtins, **context}
            # PR-QN-4: inject detected engine info
            exec_globals["__engine__"] = detected_engine
            exec(compiled, exec_globals)
            return exec_globals
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error: {e}")
        except Exception as e:
            raise DangerousCodeError(f"Execution error: {e}")

    def extract_imports(self, code: str) -> Dict[str, List[str]]:
        """提取代码中的导入语句 (PR-QN-1: 读 self._blocked_imports 实例属性)"""
        imports = {'standard': [], 'third_party': [], 'local': []}

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name.split('.')[0] in self._blocked_imports:
                            continue
                        imports['standard'].append(name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports['third_party'].append(node.module)

        except SyntaxError:
            pass

        return imports

    def extract_quantnodes_usage(self, code: str) -> List[str]:
        """提取代码中 QuantNodes 的使用情况"""
        usage = []

        patterns = [
            (r'from\s+QuantNodes\.(\w+)', 'module'),
            (r'import\s+QuantNodes\.(\w+)', 'module'),
            (r'(\w+Node)\s*\(', 'node_class'),
            (r'(\w+Node)\s*\[', 'node_class'),
        ]

        for pattern, usage_type in patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                usage.append(f"{usage_type}: {match}")

        return list(set(usage))
