# coding=utf-8
"""v11 冒烟测试 — 确保 5 层架构可导入 + 基本流程正常.

覆盖:
- 所有 7 个 layer 模块可导入
- V11Config (V10Config) 配置默认值检查
- V11Strategy 跑通 (用合成数据, 禁用 TV-PR + Jump Model 避免外部依赖)
- run_v11_backtest 跑通 + 指标输出正确
- RiskLayerV11 (ACT-2/3) 接口可调用
- 权重合理性: 和 ≤ position_size, 非负, 所有列非 NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ============================================================
# 工具函数
# ============================================================
def _make_weekly_returns(n_weeks: int = 120, n_etfs: int = 10, seed: int = 42) -> pd.DataFrame:
    """构造周频收益数据 (带一点动量 + 波动)."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-03', periods=n_weeks, freq='W-FRI')
    cols = [f'etf{i}' for i in range(n_etfs)]
    data = np.random.randn(n_weeks, n_etfs) * 0.02
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_macro_df(n_weeks: int = 120, seed: int = 42) -> pd.DataFrame:
    """构造 5 个宏观因子数据."""
    np.random.seed(seed + 1)
    dates = pd.date_range('2020-01-03', periods=n_weeks, freq='W-FRI')
    cols = ['宏观增长因子', '宏观通胀因子_生活端', '信用利差因子',
            '宏观汇率因子', '期限利差因子_债']
    data = np.random.randn(n_weeks, 5) * 0.5
    return pd.DataFrame(data, index=dates, columns=cols)


def _minimal_v11_config():
    """返回最小化配置 — 禁用 TV-PR 和 Jump Model，避免外部数据依赖."""
    from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Config
    cfg = V11Config()
    cfg.macro.use_tvpr = False
    cfg.risk.enabled = False
    return cfg


# ============================================================
# 导入性测试
# ============================================================
class TestV11Imports:
    """v11 所有公开接口可导入."""

    def test_import_v11_package(self):
        from QuantNodes.strategy.momentum_etf_rotation import v11
        assert hasattr(v11, 'V11Config')
        assert hasattr(v11, 'V11Strategy')
        assert hasattr(v11, 'run_v11')
        assert hasattr(v11, 'run_v11_backtest')
        assert hasattr(v11, 'V11BacktestResult')
        assert hasattr(v11, 'RiskLayerV11')

    def test_import_all_layer_configs(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import (
            MacroLayerConfig, IndustryLayerConfig, StyleLayerConfig,
            FactorLayerConfig, RiskLayerConfig, PositionLayerConfig,
            PortfolioLayerConfig,
        )
        for cls in [MacroLayerConfig, IndustryLayerConfig, StyleLayerConfig,
                    FactorLayerConfig, RiskLayerConfig, PositionLayerConfig,
                    PortfolioLayerConfig]:
            cfg = cls()
            assert hasattr(cfg, 'enabled')

    def test_import_all_layers(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import (
            MacroLayer, IndustryLayer, StyleLayer, FactorLayer,
            RiskLayer, PositionLayer, PortfolioLayer,
        )
        assert callable(MacroLayer)
        assert callable(IndustryLayer)
        assert callable(StyleLayer)
        assert callable(FactorLayer)
        assert callable(RiskLayer)
        assert callable(PositionLayer)
        assert callable(PortfolioLayer)

    def test_import_all_compute_functions(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import (
            compute_macro_signal, compute_industry_tilt,
            compute_style_weights, compute_factor_tilt,
            compute_bear_probability, compute_dynamic_position,
            build_final_weights,
        )
        assert callable(compute_macro_signal)
        assert callable(compute_industry_tilt)
        assert callable(compute_style_weights)
        assert callable(compute_factor_tilt)
        assert callable(compute_bear_probability)
        assert callable(compute_dynamic_position)
        assert callable(build_final_weights)


# ============================================================
# 配置默认值测试
# ============================================================
class TestV11ConfigDefaults:
    """V11Config 默认值契约测试."""

    def test_top_level_config(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Config
        cfg = V11Config()
        assert cfg.rebal_freq == 'W'
        assert cfg.cost_bps == 5.0
        assert cfg.warmup_days == 252

    def test_macro_config_defaults(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import MacroLayerConfig
        cfg = MacroLayerConfig()
        assert cfg.enabled is True
        assert cfg.use_tvpr is True
        assert cfg.tvpr_weight == 0.5
        assert cfg.entropy_window == 104
        assert cfg.zscore_window == 52
        assert cfg.bull_threshold == 0.5
        assert cfg.bear_threshold == -0.5
        assert len(cfg.factor_signs) == 5

    def test_risk_config_defaults(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerConfig
        cfg = RiskLayerConfig()
        assert cfg.enabled is True
        assert cfg.bear_prob_window == 60
        assert cfg.asset_type == 'equity'

    def test_position_config_defaults(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import PositionLayerConfig
        cfg = PositionLayerConfig()
        assert cfg.enabled is True
        assert cfg.pos_intercept == 0.7
        assert cfg.pos_z_coef == 0.5
        assert cfg.pos_min == 0.2
        assert cfg.pos_max == 1.0
        assert cfg.use_bear_prob_adjustment is True
        assert 'macro' in cfg.z_score_weights
        assert 'sector' in cfg.z_score_weights
        assert 'style' in cfg.z_score_weights

    def test_portfolio_config_defaults(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import PortfolioLayerConfig
        cfg = PortfolioLayerConfig()
        assert cfg.enabled is True
        assert cfg.base_method == 'risk_parity'
        assert cfg.rp_lookback == 52
        assert cfg.cap == 0.15
        assert cfg.floor == 0.005

    def test_config_independent_instances(self):
        """每次构造 V11Config 应返回独立实例，修改不互相影响."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Config
        cfg1 = V11Config()
        cfg2 = V11Config()
        assert cfg1 is not cfg2
        assert cfg1.macro is not cfg2.macro

        cfg1.macro.use_tvpr = False
        cfg3 = V11Config()
        assert cfg3.macro.use_tvpr is True


# ============================================================
# 单层功能测试
# ============================================================
class TestMacroLayer:
    """宏分层 (Layer 1) 功能测试."""

    def test_macro_without_data_returns_neutral(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_macro_signal
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import MacroLayerConfig
        cfg = MacroLayerConfig(use_tvpr=False)
        returns = _make_weekly_returns(n_weeks=30)
        score, regime = compute_macro_signal(None, returns, cfg)
        assert len(score) == len(returns)
        assert all(s == 0.0 for s in score)
        assert all(r == 'neutral' for r in regime)

    def test_macro_with_macro_data(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_macro_signal
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import MacroLayerConfig
        cfg = MacroLayerConfig(use_tvpr=False)
        returns = _make_weekly_returns(n_weeks=120)
        macro_df = _make_macro_df(n_weeks=120)
        score, regime = compute_macro_signal(macro_df, returns, cfg)
        assert len(score) == len(macro_df)
        assert all(-1.5 <= s <= 1.5 for s in score)
        assert all(r in {'bull', 'neutral', 'bear'} for r in regime)

    def test_macro_layer_class(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import MacroLayer
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import MacroLayerConfig
        cfg = MacroLayerConfig(use_tvpr=False)
        layer = MacroLayer(cfg)
        macro_df = _make_macro_df(n_weeks=120)
        returns = _make_weekly_returns(n_weeks=120)
        layer.fit(macro_df, returns)
        assert layer.macro_score is not None
        assert layer.regime_state is not None
        assert len(layer.macro_score) == 120


class TestIndustryLayer:
    """行业轮动层 (Layer 2A) 功能测试."""

    def test_industry_tilt_shape(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_industry_tilt
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import IndustryLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        rebal_dates = returns.index[-20:]
        cfg = IndustryLayerConfig(regime_enabled=False, corr_constraint=False)
        tilt = compute_industry_tilt(returns, rebal_dates, None, cfg)
        assert isinstance(tilt, pd.DataFrame)
        assert len(tilt) == len(rebal_dates)

    def test_industry_tilt_values_non_negative(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_industry_tilt
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import IndustryLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=15)
        rebal_dates = returns.index[-20:]
        cfg = IndustryLayerConfig(regime_enabled=False, corr_constraint=False)
        tilt = compute_industry_tilt(returns, rebal_dates, None, cfg)
        assert (tilt >= 0).all().all()


class TestStyleLayer:
    """风格轮动层 (Layer 2B) 功能测试."""

    def test_style_weights_shape(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_style_weights
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import StyleLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        rebal_dates = returns.index[-10:]
        cfg = StyleLayerConfig(regime_enabled=False)
        weights = compute_style_weights(returns, rebal_dates, None, cfg)
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(rebal_dates)


class TestFactorLayer:
    """因子选股层 (Layer 2C) 功能测试."""

    def test_factor_tilt_shape(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_factor_tilt
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import FactorLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=15)
        rebal_dates = returns.index[-10:]
        cfg = FactorLayerConfig(regime_enabled=False)
        tilt = compute_factor_tilt(returns, rebal_dates, None, cfg)
        assert isinstance(tilt, pd.DataFrame)
        assert len(tilt) == len(rebal_dates)
        assert tilt.shape[1] == 15


class TestRiskLayerDisabled:
    """风险层 (Layer 3) — 禁用时返回零."""

    def test_risk_disabled_returns_zero(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_bear_probability
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import RiskLayerConfig
        returns = _make_weekly_returns(n_weeks=50)
        cfg = RiskLayerConfig(enabled=False)
        bear_prob = compute_bear_probability(returns, cfg)
        assert len(bear_prob) == len(returns)
        assert all(p == 0.0 for p in bear_prob)


class TestPositionLayer:
    """动态仓位层 (Layer 4) 功能测试."""

    def test_position_in_bounds(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_dynamic_position
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import PositionLayerConfig
        returns = _make_weekly_returns(n_weeks=80)
        z_score = pd.Series(0.0, index=returns.index)
        cfg = PositionLayerConfig(use_bear_prob_adjustment=False)
        pos = compute_dynamic_position(z_score, None, cfg)
        assert len(pos) == len(returns)
        assert all(cfg.pos_min <= p <= cfg.pos_max for p in pos)

    def test_position_with_bear_prob_reduces(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import compute_dynamic_position
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import PositionLayerConfig
        returns = _make_weekly_returns(n_weeks=80)
        z_score = pd.Series(0.0, index=returns.index)
        bear_prob = pd.Series(1.0, index=returns.index)
        cfg = PositionLayerConfig(use_bear_prob_adjustment=True,
                                  bear_prob_adjustment_coef=0.5)
        pos_no_bear = compute_dynamic_position(z_score, None, cfg)
        pos_with_bear = compute_dynamic_position(z_score, bear_prob, cfg)
        # bear_prob=1 时仓位应降低
        assert (pos_with_bear <= pos_no_bear).all()


class TestPortfolioLayer:
    """组合构建层 (Layer 5) 功能测试."""

    def test_build_weights_sum_le_one(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import build_final_weights
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import PortfolioLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        rebal_dates = returns.index[-10:]
        cfg = PortfolioLayerConfig(base_method='equal_weight')
        position_size = pd.Series(0.8, index=rebal_dates)
        weights = build_final_weights(returns, rebal_dates, None, None, position_size, cfg)
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(rebal_dates)
        assert weights.shape[1] == 10
        # 每行权重和应 ≤ position_size
        for i in range(len(weights)):
            row_sum = weights.iloc[i].sum()
            assert row_sum <= 0.8 + 1e-6, f"row {i} sum {row_sum} > 0.8"
            assert row_sum >= 0

    def test_build_weights_none_tilt_matches_ones_tilt(self):
        """None 的 tilt 应与全 1 tilt 结果相同."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import build_final_weights
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import PortfolioLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        rebal_dates = returns.index[-10:]
        cfg = PortfolioLayerConfig(base_method='equal_weight')
        position_size = pd.Series(1.0, index=rebal_dates)

        w_none = build_final_weights(returns, rebal_dates, None, None, position_size, cfg)
        ones = pd.DataFrame(1.0, index=rebal_dates, columns=returns.columns)
        w_ones = build_final_weights(returns, rebal_dates, ones, ones, position_size, cfg)

        pd.testing.assert_frame_equal(w_none, w_ones)

    def test_build_weights_rp_method(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import build_final_weights
        from QuantNodes.strategy.momentum_etf_rotation.v11.config_v11 import PortfolioLayerConfig
        returns = _make_weekly_returns(n_weeks=80, n_etfs=8)
        rebal_dates = returns.index[-5:]
        cfg = PortfolioLayerConfig(base_method='risk_parity', rp_lookback=20)
        position_size = pd.Series(1.0, index=rebal_dates)
        sector_tilt = pd.DataFrame(1.0, index=rebal_dates, columns=returns.columns)
        factor_tilt = pd.DataFrame(1.0, index=rebal_dates, columns=returns.columns)
        weights = build_final_weights(returns, rebal_dates, sector_tilt, factor_tilt,
                                       position_size, cfg)
        assert (weights >= 0).all().all()
        # RP 底仓权重应该不全相等 (波动不同)
        last_row = weights.iloc[-1]
        assert not (last_row == last_row.iloc[0]).all()


# ============================================================
# 完整策略端到端测试
# ============================================================
class TestV11StrategyEndToEnd:
    """V11Strategy 完整流程冒烟测试."""

    def test_strategy_runs_without_macro(self):
        """无宏观数据时 (纯熵权等权默认) 策略能跑通."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Strategy
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        cfg = _minimal_v11_config()
        strategy = V11Strategy(cfg)
        weights = strategy.run(returns, macro_df=None)

        assert isinstance(weights, pd.DataFrame)
        assert weights.shape == (80, 10)
        assert weights.index.equals(returns.index)
        # 所有权重非负
        assert (weights >= 0).all().all()
        # 无 NaN
        assert not weights.isna().any().any()

    def test_strategy_intermediate_outputs(self):
        """各层中间结果已赋值."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Strategy
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        cfg = _minimal_v11_config()
        strategy = V11Strategy(cfg)
        strategy.run(returns, macro_df=None)

        assert strategy.macro_score is not None
        assert strategy.regime_state is not None
        assert strategy.bear_prob is not None
        assert strategy.position_size is not None
        assert strategy.weights is not None

    def test_strategy_with_macro_data(self):
        """有宏观数据时策略能跑通."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Strategy
        returns = _make_weekly_returns(n_weeks=120, n_etfs=10)
        macro_df = _make_macro_df(n_weeks=120)
        cfg = _minimal_v11_config()
        strategy = V11Strategy(cfg)
        weights = strategy.run(returns, macro_df=macro_df)

        assert isinstance(weights, pd.DataFrame)
        assert weights.shape == (120, 10)
        assert (weights >= 0).all().all()

    def test_strategy_position_sizing_applied(self):
        """权重和应 ≤ position_size 的最大值."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import V11Strategy
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        cfg = _minimal_v11_config()
        strategy = V11Strategy(cfg)
        strategy.run(returns, macro_df=None)

        row_sums = strategy.weights.sum(axis=1)
        pos_max = strategy.position_size.max()
        # 所有权重和不应超过仓位上限太多
        assert row_sums.max() <= pos_max + 1e-6


class TestV11Backtest:
    """v11 回测引擎冒烟测试."""

    def test_backtest_runs(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import run_v11_backtest
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        cfg = _minimal_v11_config()
        result = run_v11_backtest(returns, macro_df=None, cfg=cfg)

        assert result.nav is not None
        assert len(result.nav) == 80
        assert result.nav.iloc[0] == pytest.approx(1.0)
        assert result.weights is not None
        assert len(result.weights) == 80

    def test_backtest_metrics_present(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import run_v11_backtest
        returns = _make_weekly_returns(n_weeks=80, n_etfs=10)
        cfg = _minimal_v11_config()
        result = run_v11_backtest(returns, macro_df=None, cfg=cfg)

        required_keys = ['ann_return', 'total_return', 'ann_vol', 'sharpe',
                         'max_drawdown', 'calmar', 'win_rate', 'final_nav', 'n_periods']
        for key in required_keys:
            assert key in result.metrics, f"缺少指标: {key}"

        # 基本合理性
        assert result.metrics['n_periods'] == 80
        assert result.metrics['max_drawdown'] <= 0
        assert result.metrics['ann_vol'] >= 0
        assert 0 <= result.metrics['win_rate'] <= 1

    def test_backtest_cost_effect(self):
        """高成本下 NAV 应低于零成本."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import run_v11_backtest
        returns = _make_weekly_returns(n_weeks=100, n_etfs=10)
        cfg_low = _minimal_v11_config()
        cfg_low.cost_bps = 0.0
        cfg_high = _minimal_v11_config()
        cfg_high.cost_bps = 100.0  # 1% 每次调仓，极端值

        result_low = run_v11_backtest(returns, macro_df=None, cfg=cfg_low)
        result_high = run_v11_backtest(returns, macro_df=None, cfg=cfg_high)

        assert result_high.metrics['final_nav'] <= result_low.metrics['final_nav']


class TestRiskLayerV11:
    """RiskLayerV11 (ACT-2 Kelly + ACT-3 Drawdown) 接口测试."""

    def test_default_init_no_args(self):
        """无参构造 — 与其他 layer 风格一致."""
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerV11
        layer = RiskLayerV11()
        assert layer.dd_config is not None
        assert layer.kelly_audit_enabled is True
        assert layer.kelly_results == []

    def test_init_with_config(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerV11
        from QuantNodes.strategy.momentum_etf_rotation.common.drawdown_controller import (
            DrawdownConfig,
        )
        cfg = DrawdownConfig(max_tolerance=0.3)
        layer = RiskLayerV11(dd_config=cfg, kelly_audit_enabled=False)
        assert layer.dd_config.max_tolerance == 0.3
        assert layer.kelly_audit_enabled is False

    def test_apply_dd_control(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerV11
        n = 50
        dates = pd.date_range('2020-01-03', periods=n, freq='W-FRI')
        weights = pd.DataFrame(0.2, index=dates, columns=['a', 'b', 'c', 'd', 'e'])
        nav = pd.Series(np.linspace(1.0, 0.85, n), index=dates)

        layer = RiskLayerV11(kelly_audit_enabled=False)
        result = layer.apply_dd_control(weights, nav)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == weights.shape
        assert (result >= 0).all().all()

    def test_compute_kelly_audit(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerV11
        dates = pd.date_range('2020-01-03', periods=52, freq='W-FRI')
        nav = pd.Series(100.0 * np.cumprod(1 + np.random.RandomState(42).randn(52) * 0.02),
                        index=dates)
        layer = RiskLayerV11()
        result = layer.compute_kelly_audit(nav)
        assert isinstance(result, dict)
        assert len(layer.kelly_results) == 1

    def test_get_summary(self):
        from QuantNodes.strategy.momentum_etf_rotation.v11 import RiskLayerV11
        layer = RiskLayerV11()
        summary = layer.get_summary()
        assert 'dd_enabled' in summary
        assert 'kelly_audit_enabled' in summary
        assert 'n_kelly_audits' in summary
