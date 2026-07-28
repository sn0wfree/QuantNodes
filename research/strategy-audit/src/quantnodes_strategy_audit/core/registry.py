"""Detector registry for plugin-based extension."""
from __future__ import annotations

from typing import Type

from quantnodes_strategy_audit.core.base import BaseDetector


class DetectorRegistry:
    """Global registry for detectors with plugin-style extension.

    Example:
        @DetectorRegistry.register
        class MyDetector(BaseDetector):
            name = "custom.my_rule"
            ...
    """

    _registry: dict[str, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, detector_cls: Type[BaseDetector]) -> Type[BaseDetector]:
        """Register a detector class."""
        if not detector_cls.name:
            raise ValueError(f"{detector_cls.__name__} has empty 'name' attribute")
        if detector_cls.name in cls._registry:
            existing = cls._registry[detector_cls.name]
            if existing is not detector_cls:
                raise ValueError(
                    f"Detector name '{detector_cls.name}' already registered by {existing.__name__}"
                )
        cls._registry[detector_cls.name] = detector_cls
        return detector_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseDetector]:
        """Get a detector class by name."""
        if name not in cls._registry:
            raise KeyError(f"Detector '{name}' not registered. Available: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def all(cls) -> list[Type[BaseDetector]]:
        """Get all registered detector classes."""
        return list(cls._registry.values())

    @classmethod
    def by_category(cls, category: str) -> list[Type[BaseDetector]]:
        """Get all detectors in a given category."""
        return [d for d in cls._registry.values() if d.category == category]

    @classmethod
    def by_severity(cls, severity: object) -> list[Type[BaseDetector]]:
        """Get all detectors with a given severity."""
        from quantnodes_strategy_audit.core.warning import Severity
        if isinstance(severity, str):
            severity = Severity(severity)
        return [d for d in cls._registry.values() if d.severity == severity]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered detectors (mainly for testing)."""
        cls._registry.clear()
