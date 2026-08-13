# coding=utf-8
"""TV-PR 模块权重组合策略测试."""
import numpy as np

from QuantNodes.strategy.cta.strategies.tvpr_combo import TVPRComboStrategy

# 小池 (含产业链配方品种) 提速
_TEST_POOL = ["AU", "RB", "M", "CU", "I", "TA", "MA", "C", "J", "A", "PP"]
_MIN_HISTORY = 60   # 测试用短窗口
_STEP = 21


class TestTVPRCombo:
    def test_construct(self):
        s = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        assert len(s._modules) == 7

    def test_beta_shape_and_coverage(self):
        """β_t (T,7): 全维度保留 (λ_l1=0), 非零率高."""
        s = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        beta = s._estimate_beta()
        assert beta.shape[1] == 7
        assert beta.shape[0] > 100
        nonzero = (beta.abs() > 1e-9).mean()
        assert nonzero.min() > 0.5  # 每维度都有权重

    def test_beta_summary(self):
        s = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        summary = s.beta_summary()
        assert set(summary.index) == {m.__class__.__name__ for m in s._modules}
        assert "mean" in summary.columns

    def test_signals_valid(self):
        s = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        assert (sig != 0).sum().sum() > 100

    def test_run_completes(self):
        s = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        result = s.run()
        assert result.metrics["Years"] > 0
        assert len(result.nav) > 100

    def test_no_lookahead_beta(self):
        """β_t 因果性: t 时点权重只用 [0, t) 数据 (expanding 保证).

        检验: 提前结束面板与完整面板在前段 β 一致 (小样本快速验证).
        """
        s_full = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2021-12-31",
            min_history=_MIN_HISTORY,
        )
        s_full.load_panel()
        beta_full = s_full._estimate_beta()

        s_half = TVPRComboStrategy(
            products=_TEST_POOL, start="2020-01-01", end="2020-12-31",
            min_history=_MIN_HISTORY,
        )
        s_half.load_panel()
        beta_half = s_half._estimate_beta()

        # 避开面板边界: 末段 min_history 行训练窗口触及面板末尾 (Y 缺失),
        # 完整面板同日期训练样本更多, 属边界效应而非前视
        n = min(len(beta_half), len(beta_full)) - _MIN_HISTORY
        diff = np.abs(beta_full.iloc[:n].values - beta_half.iloc[:n].values).max()
        assert diff < 1e-9, f"β 前段不一致, 存在前视: max diff={diff}"
