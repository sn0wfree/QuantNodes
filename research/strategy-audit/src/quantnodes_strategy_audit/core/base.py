"""Base classes for detectors and validators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from quantnodes_strategy_audit.core.warning import Severity, Warning


class BaseDetector(ABC):
    """Abstract base class for all static detectors (AST analysis).

    Subclasses must define:
        - name: unique identifier, e.g. "lookahead.same_period"
        - category: classification, e.g. "lookahead"
        - severity: default Severity
        - scan_file(): yields Warning objects for a single file

    Subclasses may override:
        - scan_directory(): custom directory traversal
        - description, related_lessons, rule_url: metadata

    Example:
        @DetectorRegistry.register
        class MyDetector(BaseDetector):
            name = "custom.my_rule"
            category = "custom"
            severity = Severity.HIGH

            def scan_file(self, file: Path) -> Iterator[Warning]:
                # detection logic
                yield Warning(...)
    """

    name: str = ""
    category: str = ""
    severity: Severity = Severity.MEDIUM
    description: str = ""
    related_lessons: tuple[str, ...] = ()
    rule_url: str = ""

    @abstractmethod
    def scan_file(self, file: Path) -> Iterator[Warning]:
        """Scan a single Python file and yield Warnings."""

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = "*.py",
    ) -> Iterator[Warning]:
        """Scan a directory for matching files.

        Default implementation uses pathlib.rglob/glob.
        Subclasses may override for better performance (e.g. parallel scan).
        """
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        for file in files:
            if file.is_file():
                yield from self.scan_file(file)


class BaseValidator(ABC):
    """Abstract base class for runtime validators.

    Validators execute code (e.g. backtest runs) to verify properties.
    They differ from detectors (which are pure AST analysis).

    Subclasses must define:
        - name: unique identifier
        - category: classification
        - validate(): returns ValidationResult
    """

    name: str = ""
    category: str = ""
    description: str = ""

    @abstractmethod
    def validate(self, **kwargs) -> object:
        """Execute validation and return a result object."""
