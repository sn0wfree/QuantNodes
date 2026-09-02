# coding=utf-8
"""AllWeather — backtest 引擎单元测试 (Step 5, PLAN §十二)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.backtest import (
    AllWeatherConfig,
    run_backtest,
    run_equal_weight_vol_target,
)


def _panel(t: int = 400, n_fut: int = 6, n_etf: int = 3, seed: int = 0) -> pd.DataFrame:
    """合成价格面板: 期货簇结构 + 一只迟上市 ETF (渐入池)."""
    r = np.random.default_rng(seed)
    cols = {}
    for j in range(n_fut):
        common = r.normal(0, 0.008, size=(t, 1))
        idio = r.normal(0, 0.010, size=(t, 1))
        cols[f"fut_f{j}"] = np.cumprod(1 + common + idio, axis=0).ravel()
    for j in range(n_etf):
        cols[f"etf_e{j}"] = np.cumprod(
            1 + r.normal(0.0003, 0.011, size=t), axis=0)
    panel = pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=t))
    # etf_e2 迟 200 日上市
    panel.loc[:panel.index[199], "etf_e2"] = np.nan
    return panel


def _metrics_stub(nav: pd.Series, freq=None, oos_start=None) -> dict:
    rets = nav.pct_change().dropna()
    return {"sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0}


class TestRunBacktest:
    def test_smoke(self):
        panel = _panel()
        cfg = AllWeatherConfig(window=126)
        res = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        assert len(res.nav) == len(panel)
        assert not res.nav.isna().any()
        assert res.nav.iloc[0] == 1.0
        assert res.weights.shape == panel.shape

    def test_equal_weight_ratio_at_rebalance(self):
        panel = _panel()
        cfg = AllWeatherConfig(window=126, use_hrp=False, trend_mode="none",
                               target_vol=0.50, max_leverage=10.0)
        res = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        row = res.weights.iloc[-1]
        avail = row[row > 0]
        assert len(avail) == panel.shape[1]
        # 等权基础 + 分 sleeve 统一杠杆: 期货组内相等, ETF 组内相等
        fut_vals = avail[[c for c in avail.index if c.startswith("fut_")]]
        etf_vals = avail[[c for c in avail.index if c.startswith("etf_")]]
        np.testing.assert_allclose(fut_vals.values, fut_vals.iloc[0], rtol=1e-8)
        np.testing.assert_allclose(etf_vals.values, etf_vals.iloc[0], rtol=1e-8)

    def test_late_listing_asset_zero_before(self):
        panel = _panel()
        cfg = AllWeatherConfig(window=126)
        res = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        first_valid = panel["etf_e2"].first_valid_index()
        before = res.weights.loc[:first_valid, "etf_e2"]
        assert (before == 0.0).all()

    def test_costs_reduce_nav(self):
        panel = _panel()
        lo = run_backtest(panel, AllWeatherConfig(window=126, futures_commission=0.0,
                                                  etf_commission=0.0),
                          compute_metrics_fn=_metrics_stub)
        hi = run_backtest(panel, AllWeatherConfig(window=126, futures_commission=0.005,
                                                  etf_commission=0.005),
                          compute_metrics_fn=_metrics_stub)
        assert hi.nav.iloc[-1] < lo.nav.iloc[-1]
        assert lo.turnover.sum() > 0

    def test_no_lookahead_truncation_consistent(self):
        panel = _panel(500, 6, 3, seed=5)
        cfg = AllWeatherConfig(window=126)
        full = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        cut = panel.index[350]
        trunc = run_backtest(panel.loc[:cut], cfg, compute_metrics_fn=_metrics_stub)
        np.testing.assert_allclose(
            trunc.nav.values, full.nav.loc[:cut].values, rtol=1e-9, atol=1e-12
        )

    def test_trend_v1_reduces_gross_in_downtrend(self):
        # 构造后半段持续下跌 → V1 将动量<0 资产权重砍到 0 → 总权重下降
        r = np.random.default_rng(9)
        t = 500
        up = np.cumprod(1 + r.normal(0.001, 0.008, size=(t // 2, 1)), axis=0)
        down = np.cumprod(1 + r.normal(-0.003, 0.008, size=(t - t // 2, 1)), axis=0)
        px = pd.DataFrame(
            np.vstack([up, down]) * np.ones((1, 5)),
            index=pd.bdate_range("2020-01-01", periods=t),
            columns=[f"fut_f{j}" for j in range(5)],
        )
        cfg = AllWeatherConfig(window=126, trend_mode="v1", trend_lookback=60)
        res = run_backtest(px, cfg, compute_metrics_fn=_metrics_stub)
        gross = res.weights.sum(axis=1)
        early = gross.iloc[200:250].mean()
        late = gross.iloc[-30:].mean()
        assert late < early

    def test_monthly_rebalance_runs(self):
        panel = _panel()
        cfg = AllWeatherConfig(window=126, rebalance="ME")
        res = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        assert len(res.nav) == len(panel)

    def test_lw_cov_method_runs(self):
        panel = _panel()
        cfg = AllWeatherConfig(window=126, denoise="lw")
        res = run_backtest(panel, cfg, compute_metrics_fn=_metrics_stub)
        assert not res.nav.isna().any()

    def test_convenience_baseline(self):
        panel = _panel()
        res = run_equal_weight_vol_target(panel, window=126)
        assert not res.nav.isna().any()

    def test_ann_turnover_recorded(self):
        panel = _panel()
        res = run_backtest(panel, AllWeatherConfig(window=126),
                           compute_metrics_fn=_metrics_stub)
        assert "ann_turnover" in res.metrics
        assert res.metrics["ann_turnover"] >= 0
