# coding=utf-8
"""v7.10 工厂函数测试 (Stage 32/33).

测试 strategy_versions.py 中的 v7_10_std_newλ 函数.
"""
from __future__ import annotations

import pytest

from QuantNodes.strategy.momentum_etf_rotation.strategy_versions import (
    v7_10_std_newλ,
    get_version,
    VERSIONS,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config


class TestV710Factory:
    """v7_10_std_newλ 工厂函数测试."""

    def test_returns_config(self):
        """工厂函数返回 V7_6Config 实例."""
        cfg = v7_10_std_newλ()
        assert isinstance(cfg, V7_6Config)

    def test_lambda_values(self):
        """默认 λ_tv=0.06, λ_l1=0.105."""
        cfg = v7_10_std_newλ()
        assert cfg.lambda_tv == 0.06
        assert cfg.lambda_l1 == 0.105

    def test_stop_loss_enabled(self):
        """默认 stop_loss = -0.15."""
        cfg = v7_10_std_newλ()
        assert cfg.stop_loss_threshold == -0.15
        assert cfg.stop_loss_cooldown == 5

    def test_name(self):
        """名称为 v7_10_std_newλ."""
        cfg = v7_10_std_newλ()
        assert cfg.name == "v7_10_std_newλ"

    def test_override_lambda(self):
        """工厂函数固定 λ 值, overrides 不影响."""
        cfg = v7_10_std_newλ(lambda_tv=0.10)
        # 工厂固定 λ_tv=0.06, overrides 不影响
        assert cfg.lambda_tv == 0.06

    def test_override_stop_loss(self):
        """工厂函数固定 stop_loss, overrides 不影响."""
        cfg = v7_10_std_newλ(stop_loss_threshold=None)
        # 工厂固定 stop_loss=-0.15, overrides 不影响
        assert cfg.stop_loss_threshold == -0.15

    def test_not_in_versions_dict(self):
        """v7.10 不在 VERSIONS dict 中 (类型不同)."""
        assert "7.10" not in VERSIONS

    def test_data_loads(self):
        """v7.10 数据可以加载 (auto-generate if needed)."""
        from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_10_data
        X, Y, codes = load_v7_10_data()
        assert X.shape[0] > 0
        assert Y.shape[0] > 0
        assert len(codes) > 0
