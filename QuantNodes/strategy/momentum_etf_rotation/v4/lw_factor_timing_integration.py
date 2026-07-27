# coding=utf-8
"""Nagel 风格 Ledoit-Wolf + λ 收缩因子择时 — 集成到 v4.

参考:
- 《Optimal Factor Timing in a High-Dimensional Setting》 (Nagel 团队)
- QuantML 知识星球《论文复现 | 最优因子择时框架》

核心改进 (vs Stage 18 v4 当前):
1. Ledoit-Wolf 协方差收缩 (instead of 单变量 IC 驱动)
2. MVO 权重 (考虑协方差, 不只是均值)
3. λ 权重收缩 (向等权收缩, 论文 OOS 显示 λ=30-100 稳健)
4. 滚动验证选 λ (扩展窗口 + 12m 验证期)
5. 总敞口约束 |w| = 1 (L1 范数)

预期效果:
- 降低 overfit 风险 (vs 当前 λ=0 完全 IC 驱动)
- OOS Sharpe 应该提升 (论文 A 股复现: 静态 3.60 > 择时 2.66)
- Calmar 可能略降 (vs 当前 0.613), 但稳健性大幅提升
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lw_factor_timing import (
    compute_lambda_weights,
    ledoit_wolf_shrinkage,
    mvo_weights,
)
from .factor_timing_v4 import (
    FactorTimingConfig,
    get_active_factors,
)


@dataclass
class LWFactorTimingConfig:
    """Nagel 风格因子择时配置 (Stage 18 v4 LW 增强)."""
    # 基础 (与 v4 一致)
    ic_step: int = 5
    lookback: int = 60
    factor_lookback: int = 60

    # 因子特异 FW (同 v4)
    factor_fw: dict[str, int] = field(default_factory=lambda: {
        "momentum": 120, "reversal": 60, "value": 40,
        "dividend": 180, "quality": 252,
    })
    use_low_vol: bool = False

    # 滚动验证 λ
    train_window: int = 60          # 训练期 (月, 论文 120)
    val_window: int = 12            # 验证期 (月, 论文 12)
    candidate_lambdas: tuple[float, ...] = (
        0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0
    )

    # MVO 约束
    long_only: bool = True
    l1_norm: float = 1.0

    # 最小 IC 阈值 (Stage 18 #5 同)
    factor_ic_threshold: float = 0.05

    # 总敞口 (含 cash)
    cash_target: float = 0.0       # 0 = 全仓, 0.3 = 70% 仓位

    # 因子→ETF (同 v4)
    factor_to_etf: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "momentum": ("510300", "510500", "159915", "588000", "510880"),
        "reversal": ("510300", "510500", "159915", "588000", "510880"),
        "value":    ("512040",),
        "dividend": ("510880", "512890", "515080", "515100"),
        "quality":  ("515900",),
    })

    # Regime 因子选择 (同 v4)
    regime_factors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bull":     ("momentum", "value"),
        "bear":     ("value", "dividend", "quality"),
        "sideways": ("value",),
    })

    # 调仓
    rebalance_freq: str = "M"
    min_history: int = 312         # 论文要求 ~120m 训练 + 12m 验证

    # 调仓日 lambda 是否更新 (论文: 每月更新)
    update_lambda_each_rebal: bool = True


def compute_lw_weights(
    ic_window: pd.DataFrame,
    cfg: LWFactorTimingConfig,
    lam: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Ledoit-Wolf MVO + λ 收缩计算权重.

    Args:
        ic_window: IC DataFrame (T × N), T = train_window 个月
        cfg: 配置
        lam: λ 收缩参数 (论文: 1-100)

    Returns:
        factor_weights: 因子权重 dict
        meta: 诊断信息 (delta, mu, cov)
    """
    factors = list(ic_window.columns)
    X = ic_window[factors].fillna(0.0).values
    T, N = X.shape
    if T < 12 or N < 2:
        equal = 1.0 / max(N, 1)
        return ({f: equal for f in factors}, {"delta": 0.0, "lam": lam})

    mu = X.mean(axis=0)
    cov, delta = ledoit_wolf_shrinkage(X)
    w_mvo = mvo_weights(mu, cov, long_only=cfg.long_only, l1_norm=cfg.l1_norm)

    equal_w = np.ones(N) / N
    w_final = compute_lambda_weights(w_mvo, equal_w, lam)

    if cfg.factor_ic_threshold > 0:
        for i, fac in enumerate(factors):
            if abs(mu[i]) < cfg.factor_ic_threshold:
                w_final[i] = 0.0
        l1 = np.abs(w_final).sum()
        if l1 > 1e-12:
            w_final = w_final * (cfg.l1_norm / l1)
        else:
            w_final = equal_w

    f_w = {fac: float(w) for fac, w in zip(factors, w_final)}
    meta = {"delta": delta, "lam": lam, "mu": mu.tolist(), "cov_diag": np.diag(cov).tolist()}
    return f_w, meta


def select_lambda_for_date(
    ic_history: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: LWFactorTimingConfig,
) -> float:
    """在 as_of 用滚动验证选最优 λ.

    训练期: as_of 前 train_window 月
    验证期: as_of 前 val_window 月 (在训练期后)
    """
    if as_of not in ic_history.index:
        idx = ic_history.index.get_indexer([as_of], method="ffill")[0]
        if idx < 0:
            return 1.0
        as_of = ic_history.index[idx]

    pos = ic_history.index.get_loc(as_of)
    val_end = pos
    val_start = max(0, val_end - cfg.val_window)
    train_end = val_start
    train_start = max(0, train_end - cfg.train_window)

    if val_end - val_start < 3 or train_end - train_start < 12:
        return 1.0

    train_ic = ic_history.iloc[train_start:train_end]
    val_ic = ic_history.iloc[val_start:val_end]

    factors = list(train_ic.columns)
    X_train = train_ic[factors].fillna(0.0).values
    mu = X_train.mean(axis=0)
    cov, _ = ledoit_wolf_shrinkage(X_train)
    w_mvo = mvo_weights(mu, cov, long_only=cfg.long_only, l1_norm=cfg.l1_norm)
    equal_w = np.ones(len(factors)) / len(factors)

    if val_ic.empty or len(val_ic) < 3:
        return 1.0

    val_signals = val_ic[factors].fillna(0.0).values
    mean_val_signals = val_signals.mean(axis=0)

    best_lam = cfg.candidate_lambdas[len(cfg.candidate_lambdas) // 2]
    best_score = -np.inf

    for lam in cfg.candidate_lambdas:
        w = compute_lambda_weights(w_mvo, equal_w, lam)
        score = float(np.dot(mean_val_signals, w))
        if score > best_score:
            best_score = score
            best_lam = lam

    return best_lam


def compute_lw_factor_weights(
    ic_history: pd.DataFrame,
    cfg: LWFactorTimingConfig,
    regime: str = "sideways",
    as_of: pd.Timestamp | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """完整 LW 因子择时: regime → λ → MVO → 收缩 → ETF 映射.

    Returns:
        etf_weights: ETF → weight
        factor_weights: factor → weight
        lam: 使用的 λ
    """
    active = get_active_factors(_to_factor_timing_config(cfg))
    available = cfg.regime_factors.get(regime, active)
    available = [f for f in available if f in ic_history.columns]
    if not available:
        return {}, {}, 1.0

    if as_of is not None:
        lam = select_lambda_for_date(ic_history, as_of, cfg)
    else:
        lam = 1.0

    f_w, _ = compute_lw_weights(ic_history[available], cfg, lam=lam)

    f_w_active = {f: f_w.get(f, 0.0) for f in available}
    l1 = sum(abs(w) for w in f_w_active.values())
    if l1 > 1e-12:
        f_w_active = {f: w / l1 * cfg.l1_norm for f, w in f_w_active.items()}

    etf_w = _aggregate_factor_to_etf_lw(f_w_active, cfg)

    if cfg.cash_target > 0:
        total = sum(etf_w.values())
        if total > 1.0 - cfg.cash_target:
            scale = (1.0 - cfg.cash_target) / total
            etf_w = {k: v * scale for k, v in etf_w.items()}

    return etf_w, f_w_active, lam


def _aggregate_factor_to_etf_lw(
    factor_weights: dict[str, float],
    cfg: LWFactorTimingConfig,
) -> dict[str, float]:
    """因子权重 → ETF 权重."""
    out: dict[str, float] = {}
    for fac, w in factor_weights.items():
        if w <= 0:
            continue
        codes = cfg.factor_to_etf.get(fac, ())
        if not codes:
            continue
        per_etf = w / len(codes)
        for c in codes:
            out[c] = out.get(c, 0.0) + per_etf
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _to_factor_timing_config(cfg: LWFactorTimingConfig) -> FactorTimingConfig:
    """LWFactorTimingConfig → FactorTimingConfig (用于 get_active_factors)."""
    return FactorTimingConfig(
        use_low_vol=cfg.use_low_vol,
        factor_fw=cfg.factor_fw,
        lookback=cfg.factor_lookback,
    )


__all__ = [
    "LWFactorTimingConfig",
    "compute_lw_weights",
    "select_lambda_for_date",
    "compute_lw_factor_weights",
]
