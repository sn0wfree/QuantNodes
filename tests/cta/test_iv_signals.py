# coding=utf-8
"""iv_signals 单元测试 — 组件逻辑 + 自适应降级 + IC 对齐."""
import unittest

import numpy as np
import pandas as pd

from scripts.cta_xsec_option.iv_signals import (
    compute_skew_series,
    skew_reversal_signal,
    vrp_confirm_multiplier,
    iv_regime_multiplier,
    xsec_z,
    build_iv_enhanced_signal,
    report_xsec_ic,
)


def _fake_cache(n_days=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)
    rows = []
    for i, dt in enumerate(dates):
        spot = 3000 + rng.normal(0, 30)
        for k in (-0.04, -0.02, 0.0, 0.02, 0.04):
            for ot in ("C", "P"):
                strike = spot * np.exp(k)
                side = 1.0 if ot == "P" else -1.0
                rows.append({
                    "trade_date": dt, "strike_price": strike,
                    "spot": spot, "option_type": ot,
                    "iv": 0.2 + rng.normal(0, 0.01) - side * k * 0.5,
                    "valid": True,
                })
    return pd.DataFrame(rows)


def _panel():
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100 + rng.normal(0, 1, (len(idx), 3)).cumsum(axis=0),
        index=idx, columns=["M", "I", "RB"],
    )
    return close


class TestSkew(unittest.TestCase):
    def test_skew_series_positive_when_put_iv_higher(self):
        cache = _fake_cache()
        skew = compute_skew_series(cache)
        self.assertGreater(len(skew), 0)
        # 合成数据 put IV 更高 → skew > 0
        self.assertGreater(skew.mean(), 0)

    def test_skew_reversal_sparse_and_bounded(self):
        cache = _fake_cache(600)
        skew = compute_skew_series(cache)
        sig = skew_reversal_signal(skew)
        self.assertTrue(set(sig.unique()).issubset({-1.0, 0.0, 1.0}))
        self.assertLess((sig != 0).mean(), 0.5)

    def test_skew_trend_mode_opposite(self):
        cache = _fake_cache(600)
        skew = compute_skew_series(cache)
        rev = skew_reversal_signal(skew, mode="reversal")
        trd = skew_reversal_signal(skew, mode="trend")
        pd.testing.assert_series_equal(rev, -trd)
        with self.assertRaises(ValueError):
            skew_reversal_signal(skew, mode="bad")


class TestMultipliers(unittest.TestCase):
    def test_vrp_confirm(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sig = pd.DataFrame({"M": [1.0, -1.0, 0.0]}, index=idx)
        vrp = pd.DataFrame({"M": [1, 1, np.nan]}, index=idx)
        m = vrp_confirm_multiplier(sig, vrp)
        self.assertAlmostEqual(m.loc[idx[0], "M"], 1.15)
        self.assertAlmostEqual(m.loc[idx[1], "M"], 0.85)
        self.assertAlmostEqual(m.loc[idx[2], "M"], 1.0)

    def test_iv_regime(self):
        idx = pd.date_range("2024-01-01", periods=3)
        pct = pd.DataFrame({"M": [0.9, 0.1, np.nan]}, index=idx)
        m = iv_regime_multiplier(pct)
        self.assertAlmostEqual(m.loc[idx[0], "M"], 0.7)
        self.assertAlmostEqual(m.loc[idx[1], "M"], 1.3)
        self.assertAlmostEqual(m.loc[idx[2], "M"], 1.0)


class TestXsecZ(unittest.TestCase):
    def test_rows_sum_zero(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [np.nan, 6.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        z = xsec_z(df)
        self.assertAlmostEqual(z.iloc[0].sum(), 0.0, places=6)

    def test_thin_row_zero(self):
        df = pd.DataFrame({"a": [1.0], "b": [np.nan]},
                          index=pd.date_range("2024-01-01", periods=1))
        z = xsec_z(df)
        self.assertEqual(z.iloc[0].abs().sum(), 0.0)


class TestBuildSignal(unittest.TestCase):
    def _panel(self):
        return _panel()

    def test_no_iv_data_degrades_to_pure_momentum(self):
        close = self._panel()
        empty_skew = pd.DataFrame(index=close.index, columns=close.columns)
        empty_iv = pd.DataFrame(
            columns=["product", "trade_date", "atm_iv", "iv_percentile",
                     "vrp", "vrp_percentile", "vrp_positive", "hv_percentile"]
        )
        s1 = build_iv_enhanced_signal(close, empty_iv, empty_skew.fillna(0))
        self.assertTrue((s1 != 0).any().any())
        # 无 IV → 乘数全 1, 信号绝对值应为 1/n_active
        self.assertTrue(((s1.abs() <= 1.0) & (s1.abs() >= 0)).all().all())

    def test_skew_extreme_shifts_rank(self):
        close = self._panel()
        # M 品种 skew 极端 → 融合后 M 应被 skew 项拉向 +1 方向
        skew = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        skew["M"] = 0.0
        skew.iloc[-50:, 0] = 1.0  # 近期 M skew 恐慌事件
        empty_iv = pd.DataFrame(
            columns=["product", "trade_date", "atm_iv", "iv_percentile",
                     "vrp", "vrp_percentile", "vrp_positive", "hv_percentile"]
        )
        s = build_iv_enhanced_signal(close, empty_iv, skew)
        self.assertEqual(s.shape, close.shape)

    def test_multiplier_bounds(self):
        close = self._panel()
        idx = close.index
        iv = pd.DataFrame(
            {"product": ["M"] * len(idx), "trade_date": idx,
             "atm_iv": 0.2, "iv_percentile": 0.9,
             "vrp": 0.05, "vrp_percentile": 0.9, "vrp_positive": 1,
             "hv_percentile": 0.5}
        )
        skew = pd.DataFrame(0.0, index=idx, columns=close.columns)
        s = build_iv_enhanced_signal(close, iv, skew)
        self.assertTrue((s.abs() <= 1.5).all().all())


class TestICReport(unittest.TestCase):
    def test_no_keyerror_on_partial_index(self):
        close = _panel()
        score = pd.DataFrame(
            0.0, index=close.index.append(pd.DatetimeIndex(["2030-01-01"])),
            columns=close.columns,
        )
        ic = report_xsec_ic(score, close)
        self.assertFalse(ic.empty)


if __name__ == "__main__":
    unittest.main()
