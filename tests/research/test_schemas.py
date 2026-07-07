"""Tests for schemas: 2 个 dataclass (BacktestResult + FactorBacktestResult).

WikiFactor V2 (PR6.5): WikiFactor 和 WikiStrategy 已被合并到
`QuantNodes.research.wiki.WikiFactor` (23 字段)。相关 V2 测试见
`tests/research/test_wiki.py::TestWikiFactorV2`。
"""

from __future__ import annotations

import pytest

from QuantNodes.research.paper_understanding import schemas as s



class TestBacktestResult:
    """Test BacktestResult dataclass (3 测试)."""

    def test_default_construction(self) -> None:
        """BacktestResult() 默认构造."""
        r = s.BacktestResult()
        assert r is not None

    def test_to_dict(self) -> None:
        """to_dict() 序列化."""
        r = s.BacktestResult()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_with_equity_curve(self) -> None:
        """BacktestResult 接受 equity_curve."""
        r = s.BacktestResult(equity_curve=[{"date": "2024-01-01", "value": 100000.0}])
        assert len(r.equity_curve) == 1


class TestFactorBacktestResult:
    """Test FactorBacktestResult (2 测试)."""

    def test_construction(self) -> None:
        """FactorBacktestResult 默认构造 (字段很多, 全部 default)."""
        r = s.FactorBacktestResult()
        assert r is not None
        assert r.ic_mean == 0.0 or r.ic_mean is None  # 看 default

    def test_to_dict(self) -> None:
        """to_dict() 序列化."""
        r = s.FactorBacktestResult()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "ic_mean" in d


class TestModuleStructure:
    """Test 模块结构 (1 测试)."""

    def test_all_classes_have_to_dict(self) -> None:
        """所有 2 个剩余类都有 to_dict()."""
        for cls in [s.BacktestResult, s.FactorBacktestResult]:
            r = cls()
            assert hasattr(r, "to_dict"), f"{cls.__name__} missing to_dict"
