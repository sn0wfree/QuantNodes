# coding=utf-8
"""v4 因子择时 (Stage 18 合并版 + Stage 27 v4E/v4F) — IC 驱动 + HMM 融合.

核心思想 (金融街证券《风格轮动与因子择时》):
- 把因子作为"可交易品种"
- 用滚动 IC 监控哪个因子最近有效
- 给子策略动态分配权重: IC 高的因子权重大

Stage 18 升级 (基于 v4 诊断 SUB_STRATEGY_DIAGNOSTIC.md §2):
1. **因子特异性 forward_window** (诊断: momentum 120d, value 40d, reversal 60d)
2. **因子特异性 lag 平滑** (诊断: 5 因子 lag1=0.48-0.69 高持续, reversal 不用)
3. **Regime-conditioned 因子选择** (诊断: bull 仅 m+v, bear v+d+q, sideways 仅 v)
4. **删除 low_vol 因子** (诊断: IC vs forward 相关 -0.454, 反指因子)
5. **IC 质量过滤** (诊断: |IC|<0.05 视为噪声, 84-94% 频率)

Stage 27 新增 (v4E/v4F HMM 因子择时):
6. **v4E: 纯 HMM regime → 因子权重** (HMM 检测牛/熊/转换, 调整因子权重)
7. **v4F: IC + HMM 融合** (IC 70% + HMM 30%, 默认推荐)

向后兼容:
- `forward_window` 字段保留 (默认 None = 走因子特异 FW)
- `use_low_vol: bool = True` 字段保留 (默认 False, 走 v5 优化)
- `factor_to_strategy` 字段保留 (默认 None = 走因子→ETF 直接映射)

权重公式:
    v4D (IC): raw_weight[name] = max(0, IC[name] + base) ** power
    v4E (HMM): weight[name] = equal_weight * regime_adjustment[name]
    v4F (融合): weight = alpha * ic_weight + (1-alpha) * hmm_weight

参数:
- factor_fw: 因子→前向窗口 (默认 5 因子特异)
- factor_smooth_window: 因子→lag 平滑窗口
- factor_ic_threshold: |IC| 阈值
- regime_factors: regime→可用因子
- hmm_enabled: 启用 HMM 因子择时
- hmm_mode: "v4E" (纯 HMM) 或 "v4F" (IC+HMM 融合)
- hmm_fusion_alpha: v4F 融合权重 (IC 占比)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .factor_ic import (
    FACTOR_NAMES,
    factor_ic_at,
    factor_ic_rolling_mean,
    rolling_factor_ic,
)
from .universe_v4 import ALL_V4_CODES, SMART_BETA_CODES, STYLE_GROUP_CODES


@dataclass
class FactorTimingConfig:
    """v4 因子择时配置 (Stage 18 升级版).

    Stage 18 新增字段 (defaults = v5 优化值):
    - factor_fw: 因子→前向窗口
    - factor_smooth_window: 因子→lag 平滑
    - factor_ic_threshold: IC 质量阈值
    - regime_factors: regime→可用因子
    - factor_to_etf: 因子→具体 ETF (替代 factor_to_strategy)
    - use_low_vol: 是否启用 low_vol (默认 False)

    向后兼容字段:
    - forward_window: 统一前向窗口 (默认 None = 走 factor_fw)
    - factor_to_strategy: 因子→子策略名 (默认 None = 走 factor_to_etf)
    """
    # 通用
    ic_step: int = 5
    lookback: int = 60
    base: float = 0.05
    power: float = 2.0
    min_weight: float = 0.10
    max_weight: float = 0.50
    warmup_period: int = 120

    # Stage 18 #1: 因子特异性 forward_window
    forward_window: int | None = None
    factor_fw: dict[str, int] = field(default_factory=lambda: {
        "momentum": 120,   # 诊断: 最优 120d
        "reversal": 60,    # 诊断: 最优 60d
        "value":    40,    # 诊断: 最优 40d (峰值)
        "dividend": 180,   # 诊断: 最优 180d
        "quality":  252,   # 诊断: 最优 252d
    })

    # Stage 18 #2: 因子特异性 lag 平滑
    smooth_window: int = 12
    factor_smooth_window: dict[str, int] = field(default_factory=lambda: {
        "momentum": 4,     # 诊断: lag1=+0.59, 用 4w 平滑
        "value":    4,     # 诊断: lag1=+0.58
        "dividend": 4,     # 诊断: lag1=+0.48
        "quality":  4,     # 诊断: lag1=+0.53
        "reversal": 1,     # 诊断: lag1=-0.01, 不平滑
    })

    # Stage 18 #5: IC 质量阈值
    factor_ic_threshold: float = 0.05

    # Stage 18 #4: low_vol 开关 (默认 False = 删除反指因子)
    use_low_vol: bool = False

    # Stage 18 #3: Regime-conditioned 因子选择
    regime_factors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bull":     ("momentum", "value", "value_proxy"),
        "bear":     ("value", "dividend", "quality", "quality_proxy"),
        "sideways": ("value", "value_proxy"),
    })

    # 因子 → ETF 直接映射 (Stage 18 新增, 替代 factor_to_strategy)
    factor_to_etf: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "momentum": ("510300", "510500", "159915", "588000", "510880"),
        "reversal": ("510300", "510500", "159915", "588000", "510880"),
        "value":    ("512040",),
        "dividend": ("510880", "512890", "515080", "515100"),
        "quality":  ("515900",),
        "value_proxy": ("510300", "510500", "159915", "588000", "510880"),  # Stage 27: 估值代理
        "quality_proxy": ("515900", "512040"),  # Stage 27: 基本面代理
    })

    # 向后兼容: 因子→子策略名映射
    factor_to_strategy: dict[str, str] = field(default_factory=lambda: {
        "momentum": "style_rotation",
        "reversal": "style_rotation",
        "value":    "smart_beta",
        "low_vol":  "smart_beta",
        "dividend": "smart_beta",
        "quality":  "smart_beta",
    })

    # Stage 19: Nagel 风格 Ledoit-Wolf + λ 收缩 (可选, 默认关闭)
    # 参考: 《Optimal Factor Timing in a High-Dimensional Setting》 (Nagel 团队)
    # A 股复现: QuantML《论文复现 | 最优因子择时框架》
    # 启用: lw_enabled=True, lw_lambda 选择 "fixed"/"rolling"
    # 效果: 更稳健 (Sharpe 高), 但 Calmar 略低于 IC^2 (v4 默认)
    lw_enabled: bool = False
    lw_lambda_mode: str = "fixed"  # "fixed" | "rolling"
    lw_lambda_fixed: float = 10.0
    lw_candidate_lambdas: tuple[float, ...] = (
        0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0
    )
    lw_train_window: int = 60
    lw_val_window: int = 12
    lw_long_only: bool = True
    lw_l1_norm: float = 1.0

    # Stage 27: v4E/v4F HMM 因子择时 (可选, 默认关闭)
    # v4E: 纯 HMM regime → 因子权重
    # v4F: IC + HMM 融合 → 因子权重
    hmm_enabled: bool = False
    hmm_mode: str = "v4F"  # "v4E" | "v4F"
    hmm_fusion_alpha: float = 0.7  # v4F: IC 权重占比 (0.7 = IC 70% + HMM 30%)
    hmm_regime_factor_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "bull": {
                "momentum": 1.5,   # 牛市加动量
                "reversal": 0.8,   # 牛市减反转
                "value": 0.7,      # 牛市减价值
                "dividend": 0.8,   # 牛市减红利
                "quality": 1.0,    # 牛市质量中性
            },
            "bear": {
                "momentum": 0.5,   # 熊市减动量
                "reversal": 1.2,   # 熊市加反转
                "value": 1.5,      # 熊市加价值
                "dividend": 1.5,   # 熊市加红利
                "quality": 1.3,    # 熊市加质量
            },
            "transition": {
                "momentum": 0.8,   # 转换期减动量
                "reversal": 1.0,   # 转换期反转中性
                "value": 1.2,      # 转换期加价值
                "dividend": 1.2,   # 转换期加红利
                "quality": 1.1,    # 转换期加质量
            },
        }
    )


def get_active_factors(cfg: FactorTimingConfig) -> tuple[str, ...]:
    """当前活跃因子 (考虑 use_low_vol)."""
    if cfg.use_low_vol:
        return FACTOR_NAMES
    return tuple(f for f in FACTOR_NAMES if f != "low_vol")


def compute_factor_weights(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig | None = None,
    regime: str = "sideways",
) -> dict[str, float]:
    """根据最新一期 IC 计算因子权重 (Stage 18 升级).

    算法:
        1. 选可用因子 (regime 过滤 + use_low_vol)
        2. |IC| < threshold → weight = 0 (Stage 18 #5)
        3. raw_w[name] = max(0, IC[name] + cfg.base) ** cfg.power
        4. 应用 min_weight / max_weight
        5. 归一化

    Args:
        ic_history: 滚动 IC DataFrame (index=date, columns=factor)
        cfg: 配置
        regime: "bull" | "bear" | "sideways"

    Returns:
        dict, factor name → weight (sum=1, 或 sum<1 表示有 cash)
    """
    cfg = cfg or FactorTimingConfig()
    active = get_active_factors(cfg)
    available = cfg.regime_factors.get(regime, active)

    if ic_history.empty:
        return {n: 1.0 / len(available) for n in available}

    latest = ic_history.iloc[-1].fillna(0.0)

    raw: dict[str, float] = {}
    for name in available:
        if name not in latest.index:
            continue
        ic = float(latest.get(name, 0.0))
        if abs(ic) < cfg.factor_ic_threshold:
            continue
        raw[name] = max(0.0, ic + cfg.base) ** cfg.power

    if not raw:
        return {n: 0.0 for n in available}

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    for name in raw:
        if raw[name] < cfg.min_weight:
            raw[name] = cfg.min_weight
        if raw[name] > cfg.max_weight:
            raw[name] = cfg.max_weight

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    return raw


def compute_factor_weights_hmm(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig | None = None,
    regime: str = "sideways",
) -> dict[str, float]:
    """v4E: 纯 HMM regime → 因子权重.

    算法:
        1. 选可用因子 (regime 过滤 + use_low_vol)
        2. 根据 regime 查表获取因子权重调整系数
        3. 等权 × 调整系数 → 归一化

    Args:
        ic_history: 滚动 IC DataFrame (用于获取因子列表)
        cfg: 配置
        regime: "bull" | "bear" | "transition" | "sideways"

    Returns:
        dict, factor name → weight (sum=1)
    """
    cfg = cfg or FactorTimingConfig()
    active = get_active_factors(cfg)
    available = cfg.regime_factors.get(regime, active)

    if ic_history.empty:
        return {n: 1.0 / len(available) for n in available}

    latest = ic_history.iloc[-1].fillna(0.0)

    # 获取 regime 对应的因子权重调整系数
    regime_weights = cfg.hmm_regime_factor_weights.get(regime, {})
    if not regime_weights:
        # 无 regime 信息, 返回等权
        return {n: 1.0 / len(available) for n in available}

    raw: dict[str, float] = {}
    for name in available:
        if name not in latest.index:
            continue
        # 基础权重: 等权
        base_weight = 1.0
        # 应用 regime 调整系数
        adjustment = regime_weights.get(name, 1.0)
        raw[name] = base_weight * adjustment

    if not raw:
        return {n: 0.0 for n in available}

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    for name in raw:
        if raw[name] < cfg.min_weight:
            raw[name] = cfg.min_weight
        if raw[name] > cfg.max_weight:
            raw[name] = cfg.max_weight

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    return raw


def compute_factor_weights_fusion(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig | None = None,
    regime: str = "sideways",
) -> dict[str, float]:
    """v4F: IC + HMM 融合 → 因子权重.

    算法:
        1. 计算 IC 权重 (同 v4D)
        2. 计算 HMM 权重 (同 v4E)
        3. 融合: final = alpha * ic_weight + (1-alpha) * hmm_weight
        4. 归一化

    Args:
        ic_history: 滚动 IC DataFrame
        cfg: 配置
        regime: "bull" | "bear" | "transition" | "sideways"

    Returns:
        dict, factor name → weight (sum=1)
    """
    cfg = cfg or FactorTimingConfig()

    # 1. 计算 IC 权重
    ic_weights = compute_factor_weights(ic_history, cfg, regime)

    # 2. 计算 HMM 权重
    hmm_weights = compute_factor_weights_hmm(ic_history, cfg, regime)

    # 3. 融合
    all_factors = set(list(ic_weights.keys()) + list(hmm_weights.keys()))
    fused: dict[str, float] = {}

    for name in all_factors:
        ic_w = ic_weights.get(name, 0.0)
        hmm_w = hmm_weights.get(name, 0.0)
        fused[name] = cfg.hmm_fusion_alpha * ic_w + (1 - cfg.hmm_fusion_alpha) * hmm_w

    # 4. 归一化
    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}

    return fused


def compute_strategy_weights(
    factor_weights: dict[str, float],
    factor_to_strategy: dict[str, str] | None = None,
) -> dict[str, float]:
    """聚合因子权重到子策略权重 (向后兼容).

    多个因子映射到同一子策略 → 权重累加.

    Args:
        factor_weights: factor → weight
        factor_to_strategy: factor → strategy name (默认 FactorTimingConfig 默认值)

    Returns:
        strategy name → weight (sum=1)
    """
    f2s = factor_to_strategy or FactorTimingConfig().factor_to_strategy
    out: dict[str, float] = {}
    for f, w in factor_weights.items():
        s = f2s.get(f, "unknown")
        out[s] = out.get(s, 0.0) + w

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def aggregate_factor_to_etf(
    factor_weights: dict[str, float],
    cfg: FactorTimingConfig,
) -> dict[str, float]:
    """因子权重 → ETF 权重 (因子内等权, Stage 18 #1 主用法).

    多个因子同时选中同一只 ETF → 权重累加.
    """
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


def backtest_factor_timing(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """回测因子择时的 IC 时序 (Stage 18 升级).

    Stage 18 #1: 因子特异性 forward_window
    Stage 18 #2: 因子特异性 lag 平滑

    Args:
        nav_df: 价格面板
        all_codes: 候选 ETF codes
        cfg: 因子择时配置
        start/end: 范围

    Returns:
        pd.DataFrame, index=date, columns=factor (平滑后 IC)
    """
    cfg = cfg or FactorTimingConfig()
    active = get_active_factors(cfg)

    if cfg.forward_window is not None:
        fw_dict = {n: cfg.forward_window for n in active}
    else:
        fw_dict = {n: cfg.factor_fw.get(n, 60) for n in active}

    panel = nav_df.loc[start:end]
    dates = panel.index
    sample_dates = dates[::cfg.ic_step]
    if len(dates) > 0 and dates[-1] not in sample_dates:
        sample_dates = sample_dates.append(pd.Index([dates[-1]]))

    rows = []
    for ts in sample_dates:
        if panel.index.get_loc(ts) < cfg.lookback + max(fw_dict.values()):
            continue
        row = {"date": ts}
        for fac in active:
            fw = fw_dict[fac]
            ic_val = factor_ic_at(
                panel, ts, list(all_codes),
                forward_window=fw, lookback=cfg.lookback,
            )
            row[fac] = ic_val.get(fac, 0.0)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")

    smooth = pd.DataFrame(index=df.index)
    for fac in df.columns:
        w = cfg.factor_smooth_window.get(fac, cfg.smooth_window)
        if w <= 1:
            smooth[fac] = df[fac]
        else:
            lookback_steps = max(1, w)
            smooth[fac] = (df[fac] * 2 + df[fac].shift(lookback_steps).fillna(df[fac])) / 3

    return smooth


def backtest_factor_weights_history(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:
    """回测因子权重时序 (Stage 18 + Stage 19 LW 模式).

    Args:
        nav_df: 价格面板
        all_codes: 候选 ETF
        cfg: 配置
        start/end: 范围
        regime_series: (可选) 预计算的 regime 时序 (用于 LW 模式需要更多历史时)

    Returns:
        pd.DataFrame, index=date, columns=factor weights
    """
    cfg = cfg or FactorTimingConfig()
    ic_history = backtest_factor_timing(
        nav_df, all_codes, cfg, start, end,
    )
    if ic_history.empty:
        return pd.DataFrame()

    weights_list = []
    dates = []
    for i, (ts, row) in enumerate(ic_history.iterrows()):
        ic_window = ic_history.iloc[:i + 1]
        regime = "sideways"
        if regime_series is not None and ts in regime_series.index:
            regime = str(regime_series.loc[ts])

        if cfg.hmm_enabled:
            # v4E/v4F: HMM 因子择时
            if cfg.hmm_mode == "v4E":
                f_w = compute_factor_weights_hmm(ic_window, cfg, regime=regime)
            else:  # v4F
                f_w = compute_factor_weights_fusion(ic_window, cfg, regime=regime)
        elif cfg.lw_enabled:
            f_w = compute_factor_weights_lw(ic_window, cfg, regime=regime)
        else:
            f_w = compute_factor_weights(
                pd.DataFrame([row.to_dict()], index=[ts]), cfg, regime=regime,
            )
        weights_list.append(f_w)
        dates.append(ts)

    return pd.DataFrame(weights_list, index=dates)


__all__ = [
    "FactorTimingConfig",
    "compute_factor_weights",
    "compute_strategy_weights",
    "aggregate_factor_to_etf",
    "backtest_factor_timing",
    "backtest_factor_weights_history",
    "get_active_factors",
    "compute_factor_weights_lw",
    "compute_factor_weights_hmm",
    "compute_factor_weights_fusion",
]


def compute_factor_weights_lw(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig,
    regime: str = "sideways",
) -> dict[str, float]:
    """Nagel 风格 Ledoit-Wolf + λ 收缩 因子权重 (Stage 19).

    算法:
        1. regime-conditioned 因子选择 (同 IC^2)
        2. 用 ic_history 计算 mean (μ) + Ledoit-Wolf 协方差
        3. MVO: w ∝ cov_lw⁻¹ · μ (long-only + L1 norm)
        4. λ 收缩: w = (1-shrink)·w_mvo + shrink·w_equal
           shrink = λ / (1+λ)
        5. 归一化 (L1 = lw_l1_norm)

    Args:
        ic_history: IC DataFrame (index=date, columns=factor)
        cfg: 配置 (需 lw_enabled=True)
        regime: "bull" | "bear" | "sideways"

    Returns:
        dict, factor name → weight (sum=|w|=lw_l1_norm)
    """
    from .lw_factor_timing import (
        compute_lambda_weights,
        ledoit_wolf_shrinkage,
        mvo_weights,
    )

    active = get_active_factors(cfg)
    available = cfg.regime_factors.get(regime, active)
    available = [f for f in available if f in ic_history.columns]

    if not available or ic_history.empty:
        return {n: 0.0 for n in available}

    X = ic_history[available].fillna(0.0).values
    T, N = X.shape
    if T < 12 or N < 2:
        equal = cfg.lw_l1_norm / max(N, 1)
        return {f: equal for f in available}

    mu = X.mean(axis=0)
    cov, _ = ledoit_wolf_shrinkage(X)
    w_mvo = mvo_weights(mu, cov, long_only=cfg.lw_long_only, l1_norm=cfg.lw_l1_norm)

    if cfg.lw_lambda_mode == "rolling":
        lam = _select_lambda_rolling_lw(
            X, cfg, ic_history, available,
        )
    else:
        lam = cfg.lw_lambda_fixed

    equal_w = np.ones(N) * (cfg.lw_l1_norm / N)
    w_final = compute_lambda_weights(w_mvo, equal_w, lam)

    if cfg.factor_ic_threshold > 0:
        for i, fac in enumerate(available):
            if abs(mu[i]) < cfg.factor_ic_threshold:
                w_final[i] = 0.0
        l1 = np.abs(w_final).sum()
        if l1 > 1e-12:
            w_final = w_final * (cfg.lw_l1_norm / l1)
        else:
            w_final = equal_w

    return {f: float(w) for f, w in zip(available, w_final)}


def _select_lambda_rolling_lw(
    X_full: np.ndarray,
    cfg: FactorTimingConfig,
    ic_history: pd.DataFrame,
    available: list[str],
) -> float:
    """在 X_full 上滚动选 λ: train 后段 vs val 后段 选最大 dot product."""
    from .lw_factor_timing import (
        compute_lambda_weights,
        ledoit_wolf_shrinkage,
        mvo_weights,
    )

    T, N = X_full.shape
    val_end = T
    val_start = max(0, val_end - cfg.lw_val_window)
    train_end = val_start
    train_start = max(0, train_end - cfg.lw_train_window)

    if val_end - val_start < 3 or train_end - train_start < 12:
        return cfg.lw_lambda_fixed

    X_train = X_full[train_start:train_end]
    X_val = X_full[val_start:val_end]

    mu = X_train.mean(axis=0)
    cov, _ = ledoit_wolf_shrinkage(X_train)
    w_mvo = mvo_weights(mu, cov, long_only=cfg.lw_long_only, l1_norm=cfg.lw_l1_norm)
    equal_w = np.ones(N) * (cfg.lw_l1_norm / N)

    mean_val = X_val.mean(axis=0)
    best_lam = cfg.lw_candidate_lambdas[len(cfg.lw_candidate_lambdas) // 2]
    best_score = -np.inf
    for lam in cfg.lw_candidate_lambdas:
        w = compute_lambda_weights(w_mvo, equal_w, lam)
        score = float(np.dot(mean_val, w))
        if score > best_score:
            best_score = score
            best_lam = lam
    return best_lam
