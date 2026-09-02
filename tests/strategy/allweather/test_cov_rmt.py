# coding=utf-8
"""AllWeather — cov_rmt 单元测试 (Step 1, PLAN §十二)."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.allweather.cov_rmt import (
    ledoit_wolf_shrink,
    mp_denoise,
    mp_lambda_plus,
    sample_cov,
)

rng = np.random.default_rng(42)


def _returns_frame(t: int, n: int, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(0, 0.01, size=(t, n)))


class TestSampleCov:
    def test_shape_and_symmetry(self):
        rets = _returns_frame(300, 10)
        cov = sample_cov(rets, window=252)
        assert cov.shape == (10, 10)
        assert np.allclose(cov, cov.T)

    def test_window_slicing(self):
        rets = _returns_frame(500, 5, seed=1)
        c1 = sample_cov(rets, window=252)  # 取最后 252 行
        c2 = sample_cov(rets.iloc[-252:], window=252)
        np.testing.assert_allclose(c1, c2, rtol=1e-12)

    def test_nan_rows_dropped(self):
        rets = _returns_frame(300, 6, seed=2)
        rets.iloc[100:180, 2] = np.nan  # 某资产一段时间无数据 (渐入池场景)
        cov = sample_cov(rets, window=252)
        assert cov.shape == (6, 6)
        assert np.isfinite(cov).all()

    def test_insufficient_obs_returns_identity(self):
        rets = _returns_frame(10, 4)
        cov = sample_cov(rets, window=252)
        np.testing.assert_allclose(cov, np.eye(4))


class TestMpDenoise:
    def test_lambda_plus_value(self):
        # T=252, N=38: λ+ = (1+√(38/252))² ≈ 1.928
        assert mp_lambda_plus(252, 38) == pytest.approx(1.928, abs=1e-2)

    def test_iid_data_all_noise_collapsed(self):
        # i.i.d. 数据无信号 → 去噪后所有特征值收缩到噪声均值
        rets = _returns_frame(504, 38, seed=3)
        cov = sample_cov(rets, window=504)
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        corr_dn = mp_denoise(corr, n_obs=504)

        eig = np.linalg.eigvalsh(corr_dn)
        lam_plus = mp_lambda_plus(504, 38)
        # 全噪声 → 去噪后最大特征值不超过 λ+ (收缩后)
        assert eig.max() <= lam_plus + 1e-8
        # 对称 + 半正定
        assert np.allclose(corr_dn, corr_dn.T)
        assert eig.min() >= -1e-10

    def test_signal_eigenvalues_preserved(self):
        # 3 个独立簇 (簇内相关 ~0.5) → 3 个真实因子特征值 > λ+, 原样保留
        t, n, k = 756, 39, 3
        r = np.random.default_rng(7)
        cols = []
        for g in range(k):
            common = r.normal(0, 0.01, size=(t, 1))
            idio = r.normal(0, 0.01, size=(t, n // k))
            cols.append(0.7 * common + idio)  # 簇内相关 ≈ 0.7²/(0.7²+1) ≈ 0.33
        rets = pd.DataFrame(np.hstack(cols))

        cov = sample_cov(rets, window=t)
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        corr_dn = mp_denoise(corr, n_obs=t)

        eig_before = np.linalg.eigvalsh(corr)[::-1]
        eig_after = np.linalg.eigvalsh(corr_dn)[::-1]
        lam_plus = mp_lambda_plus(t, n)
        # 前 3 大 (簇因子) 在 λ+ 之上, 且被完全保留
        assert (eig_before[:k] > lam_plus).all()
        np.testing.assert_allclose(eig_after[:k], eig_before[:k], rtol=1e-8)
        # 第 4 大被收缩到 ≤ λ+
        assert eig_after[k] <= lam_plus + 1e-8

    def test_cov_input_rescales_back(self):
        # 输入协方差 (非相关矩阵) → 输出仍是协方差量纲, 对角线≈原对角线
        rets = _returns_frame(504, 12, seed=4)
        # 不同量纲
        rets = rets * np.linspace(0.5, 2.0, 12)
        cov = sample_cov(rets, window=504)
        cov_dn = mp_denoise(cov, n_obs=504)
        # 去噪保持对角线量级 (常数残差法保 trace, 单资产方差近似不变)
        assert cov_dn.shape == cov.shape
        assert np.allclose(np.trace(cov_dn), np.trace(cov), rtol=1e-6)


class TestLedoitWolfShrink:
    def test_psd_and_trace_preserved(self):
        rets = _returns_frame(300, 15, seed=5)
        cov_lw = ledoit_wolf_shrink(rets, window=252)
        eig = np.linalg.eigvalsh(cov_lw)
        assert eig.min() >= -1e-12
        # 收缩目标 trace(S)/N·I 与 S trace 相同 → 收缩后 trace 不变
        cov_s = sample_cov(rets, window=252)
        assert np.isclose(np.trace(cov_lw), np.trace(cov_s), rtol=1e-6)

    def test_shrunk_toward_identity(self):
        # 极少样本 → 收缩强度大 → 结果接近 scaled identity
        rets = _returns_frame(25, 20, seed=6)
        cov_lw = ledoit_wolf_shrink(rets, window=252)
        eig = np.linalg.eigvalsh(cov_lw)
        # 特征值离散度应小于样本协方差
        cov_s = sample_cov(rets, window=252)
        eig_s = np.linalg.eigvalsh(cov_s)
        assert (eig.max() - eig.min()) <= (eig_s.max() - eig_s.min()) + 1e-12

    def test_insufficient_obs_returns_identity(self):
        rets = _returns_frame(10, 4)
        cov = ledoit_wolf_shrink(rets, window=252)
        np.testing.assert_allclose(cov, np.eye(4))
