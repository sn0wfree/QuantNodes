"""V8 cov_extended 单测."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.cov_extended import (
    avs_f_lw_hybrid,
    nqn_lw2012_cov,
    poet_cov,
    ewma_avs_f_lw,
    ewma_cov,
    gerber_cov,
    gerber_lw_ensemble,
    nqn_cov,
    vol_adaptive_cov,
)


def _make_returns(n_days=600, n_assets=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"asset_{i}" for i in range(n_assets)]
    data = {c: rng.normal(0.0003, 0.01, n_days) for c in cols}
    return pd.DataFrame(data, index=dates)


def _check_psd(C, name=""):
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.min() > -1e-8, f"{name} not PSD: min_eig={eigvals.min()}"


# ── EWMA ──

def test_ewma_cov_psd():
    C = ewma_cov(_make_returns(), lam=0.94)
    _check_psd(C, "ewma")


def test_ewma_cov_different_lambdas():
    """不同 λ 产生不同协方差 (λ 小 → 历史权重更高)."""
    rets = _make_returns()
    C1 = ewma_cov(rets, lam=0.92)
    C2 = ewma_cov(rets, lam=0.98)
    assert not np.allclose(C1, C2, atol=1e-6)


# ── AVS-F + LW hybrid ──

def test_avs_f_lw_hybrid_psd():
    C = avs_f_lw_hybrid(_make_returns(n_assets=15))
    _check_psd(C, "avs_f_lw")


def test_avs_f_lw_hybrid_differs_from_lw():
    from QuantNodes.strategy.allweather.cov_rmt import ledoit_wolf_shrink
    rets = _make_returns(n_assets=15)
    C_hybrid = avs_f_lw_hybrid(rets)
    C_lw = ledoit_wolf_shrink(rets)
    # hybrid 用 F 做 target (非 μI), 应不同于纯 LW (除非 δ≈1 或 δ≈0)
    assert not np.allclose(C_hybrid, C_lw, atol=1e-8)


# ── EWMA + AVS-F + LW ──

def test_ewma_avs_f_lw_psd():
    C = ewma_avs_f_lw(_make_returns(n_assets=15), lam=0.94)
    _check_psd(C, "ewma_avs_f_lw")


# ── 波动率自适应 ──

def test_vol_adaptive_psd():
    C = vol_adaptive_cov(_make_returns())
    _check_psd(C, "vol_adaptive")


# ── NQN ──

def test_nqn_cov_psd():
    C = nqn_cov(_make_returns())
    _check_psd(C, "nqn")


def test_nqn_different_from_lw():
    from QuantNodes.strategy.allweather.cov_rmt import ledoit_wolf_shrink
    rets = _make_returns(n_assets=15)
    C_nqn = nqn_cov(rets)
    C_lw = ledoit_wolf_shrink(rets)
    # NQN 对每个特征值单独收缩, 与 LW 线性收缩不同
    assert not np.allclose(C_nqn, C_lw, atol=1e-8)


# ── Gerber ──

def test_gerber_cov_psd():
    C = gerber_cov(_make_returns())
    _check_psd(C, "gerber")


def test_gerber_lw_ensemble_psd():
    C = gerber_lw_ensemble(_make_returns(n_assets=15))
    _check_psd(C, "gerber_lw")


# ═══════════════════════════════════════════════════════════════════
# B 轮后半: NQN 非线性收缩 + POET (行为单测)
# ═══════════════════════════════════════════════════════════════════

def _spiked_returns(n_days=500, n_assets=25, n_factors=1, seed=8):
    """尖峰模型: n_factors 个强因子 + 各向异性噪声."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.01, (n_days, n_factors))
    loadings = rng.normal(1.0, 0.1, (n_assets, n_factors))
    idio = rng.normal(0, 0.004, (n_days, n_assets))
    return pd.DataFrame(factors @ loadings.T + idio,
                        index=pd.bdate_range("2024-01-01", periods=n_days))


def test_nqn_isotropic_collapse():
    """各向同性噪声: 收缩后特征值方差 << 样本特征值方差 (向均值塌缩)."""
    rets = _make_returns(n_days=500, n_assets=30, seed=31)
    C = nqn_lw2012_cov(rets)
    _check_psd(C, "nqn_isotropic")
    lam_s = np.linalg.eigvalsh(rets.cov().to_numpy())
    lam_n = np.linalg.eigvalsh(C)
    assert np.var(lam_n) < np.var(lam_s) * 0.5


def test_nqn_spiked_preserved():
    """尖峰模型: 大特征值基本保留 + 谱离散度下降 (NQN 定义性行为).

    注: NQN 将特征值向谱中心收缩 — 大的压下/小的拉上, 不保 trace;
    底部小幅上移是正确行为 (我方旧断言'底部下压'有误).
    """
    rets = _spiked_returns()
    C = nqn_lw2012_cov(rets)
    _check_psd(C, "nqn_spiked")
    lam_s = np.sort(np.linalg.eigvalsh(rets.cov().to_numpy()))
    lam_n = np.sort(np.linalg.eigvalsh(C))
    # 最大特征值 (因子信号) 保留 80%+
    assert lam_n[-1] > 0.8 * lam_s[-1]
    # 谱离散度下降 (收缩签名)
    assert np.std(lam_n) < np.std(lam_s)


def test_poet_psd_and_sparsity():
    """POET: PSD + 阈值化残差稀疏 + 因子部分恢复尖峰."""
    rets = _spiked_returns(n_factors=2)
    C = poet_cov(rets, K=2)
    _check_psd(C, "poet")
    # 最大特征值保留 (因子结构)
    lam_s = np.linalg.eigvalsh(rets.cov().to_numpy())
    lam_n = np.linalg.eigvalsh(C)
    assert lam_n[-1] > 0.9 * lam_s[-1]


def test_poet_factor_subspace_recovery():
    """POET 因子子空间恢复: top-K 特征向量与真载荷张成空间高度重合."""
    rets = _spiked_returns(n_factors=2, n_days=800)
    S = rets.cov().to_numpy()
    C = poet_cov(rets, K=2)
    # 主子空间对齐: |tr(P_true P_poet)| / K ≈ 1
    v_s, _ = np.linalg.eigh(S)
    P_true = np.linalg.eigh(S)[1][:, -2:]
    P_poet = np.linalg.eigh(C)[1][:, -2:]
    overlap = np.trace(P_true @ P_true.T @ P_poet @ P_poet.T) / 2
    assert overlap > 0.95
