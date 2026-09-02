# coding=utf-8
"""AllWeather — hrp 单元测试 (Step 2, PLAN §十二)."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.allweather.hrp import (
    hrp_weights,
    hrp_weights_from_returns,
)


def _two_cluster_cov(n_per_cluster: int = 5) -> np.ndarray:
    """两簇构造: 簇内相关 0.9, 簇间相关 0."""
    n = 2 * n_per_cluster
    corr = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j and i // n_per_cluster == j // n_per_cluster:
                corr[i, j] = 0.9
    return corr


class TestHrpWeights:
    def test_sum_to_one_nonnegative(self):
        w = hrp_weights(_two_cluster_cov(), linkage_method="single")
        assert w.shape == (10,)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)
        assert (w >= 0).all()

    def test_two_clusters_balanced(self):
        # 簇内等方差等相关 → 两簇应各得约一半风险
        w = hrp_weights(_two_cluster_cov(5), linkage_method="single")
        left = w[:5].sum()
        assert left == pytest.approx(0.5, abs=0.05)

    def test_all_linkages_run_and_differ(self):
        # 非对称相关结构 (链式衰减): linkage 选择会产生不同的树
        r = np.random.default_rng(11)
        n = 8
        a = r.normal(size=(200, n))
        corr = np.corrcoef(a.T)
        cov = corr  # HRP 对缩放不敏感, 直接用相关矩阵
        results = {}
        for lm in ("single", "average", "ward"):
            w = hrp_weights(cov, linkage_method=lm)
            assert w.sum() == pytest.approx(1.0, abs=1e-10)
            assert (w >= 0).all()
            results[lm] = w
        # linkage 是自由参数: 至少两种结果不同 (D1/Phase 3 对照的前提)
        diffs = [
            np.abs(results[a] - results[b]).max()
            for a, b in (("single", "average"), ("single", "ward"), ("average", "ward"))
        ]
        assert max(diffs) > 1e-6

    def test_single_asset(self):
        w = hrp_weights(np.array([[0.04]]))
        np.testing.assert_allclose(w, [1.0])

    def test_invalid_linkage_raises(self):
        with pytest.raises(ValueError):
            hrp_weights(_two_cluster_cov(2), linkage_method="complete")


class TestHrpFromReturns:
    def _panel(self, t: int = 400, n: int = 8, seed: int = 0) -> pd.DataFrame:
        r = np.random.default_rng(seed)
        return pd.DataFrame(r.normal(0, 0.01, size=(t, n)),
                            columns=[f"a{i}" for i in range(n)])

    def test_basic_shape_and_sum(self):
        rets = self._panel()
        w = hrp_weights_from_returns(rets, window=252, denoise=None)
        assert list(w.index) == list(rets.columns)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)
        assert (w >= 0).all()

    def test_nan_assets_excluded(self):
        rets = self._panel()
        rets["a6"] = np.nan  # 完全无数据
        rets.iloc[:300, 7] = np.nan  # a7 前 300 日无数据, 窗口 252 内有 100 日 < min_obs
        w = hrp_weights_from_returns(rets, window=252, denoise=None,
                                     min_obs_per_asset=200)
        assert w["a6"] == 0.0
        assert w["a7"] == 0.0
        assert w[:6].sum() == pytest.approx(1.0, abs=1e-10)
        assert (w[:6] > 0).all()

    def test_few_available_degrades_to_equal_weight(self):
        rets = self._panel(n=8)
        rets.iloc[:, 2:] = np.nan  # 只剩 2 只可用 (< MIN_ASSETS_FOR_HRP=5)
        w = hrp_weights_from_returns(rets, window=252)
        avail = w[w > 0]
        assert len(avail) == 2
        np.testing.assert_allclose(avail.values, [0.5, 0.5], rtol=1e-10)

    def test_no_available_all_equal(self):
        rets = self._panel(n=4)
        rets.iloc[:, :] = np.nan
        w = hrp_weights_from_returns(rets, window=252)
        np.testing.assert_allclose(w.values, [0.25] * 4, rtol=1e-10)

    def test_denoise_on_off_differ_or_equal(self):
        # denoise 开关可跑通; 对近 i.i.d. 数据结果差异应很小 (全噪声收缩近似等权化)
        rets = self._panel(600, 10, seed=3)
        w0 = hrp_weights_from_returns(rets, window=504, denoise=None)
        w1 = hrp_weights_from_returns(rets, window=504, denoise="mp")
        assert w0.sum() == pytest.approx(1.0, abs=1e-10)
        assert w1.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.abs(w0 - w1).max() < 0.2  # 量级合理性
