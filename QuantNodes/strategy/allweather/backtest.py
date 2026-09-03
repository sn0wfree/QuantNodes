# coding=utf-8
"""AllWeather — 混合频率回测引擎 (PLAN Step 5).

频率语义 (用户确认):
    - 周频 (W-FRI): 权重求解 — 窗口收益 → 协方差 (mp/lw) → HRP/等权 →
      趋势 overlay (V1/V4) → 波动率目标杠杆 → 目标权重
    - 日频: NAV 计价 + 期货 sleeve 的 vol-targeting 微调
      (ETF 两个调仓日之间持仓不动)

时序约定 (无前视):
    - t 日收盘决策 (权重/杠杆), 应用于 t→t+1 的收益
    - 调仓日 ex-ante 波动: w'Σw (与权重求解同一协方差)
    - 日频微调: 已实现组合波动 (lookback=20, 截至当日收盘), 次日生效
    - 换手成本: 按权重变化 × 分资产费率 (fut/etf), 在换手次日 NAV 扣减
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cov_rmt import ledoit_wolf_shrink, sample_cov
from .hrp import hrp_weights_from_returns
from .trend_overlay import combined

ANN = 252.0
MIN_OBS = 60  # 与 hrp.min_obs_per_asset 一致
DAILY_VOL_LOOKBACK = 20


@dataclass
class AllWeatherConfig:
    """消融/敏感性全部通过本配置开关组合, 不写多份代码."""

    window: int = 252              # 协方差估计窗口
    rebalance: str = "W-FRI"       # W-FRI / ME
    use_hrp: bool = True           # HRP vs 等权 (消融 3a/0)
    weight_scheme: str = "equal"   # equal / inv_vol (等权变体)
    denoise: str | None = "mp"     # 'mp' / 'lw' / None (消融 1/2, §5.3)
    linkage: str = "single"        # single / average / ward
    trend_mode: str = "v1"         # v1 / v3 / v4 / tilt / none (消融 3a/3b/3)
    trend_lookback: int = 60
    trend_coef: float = 0.0
    trend_risk_adj: bool = False   # tilt: 风险调整动量 (mom/vol)
    trend_floor: float = 0.25      # tilt: 负动量资产软保底
    target_vol: float = 0.10
    max_leverage: float = 3.0
    max_weight: float = 1.0        # 结构权重上限 (1.0=不约束); 修复 HRP 低波资产集中
    bond_class_cap: float = 1.0    # 债券簇(T/TF/TS)结构权重合计上限 (1.0=不约束)
    use_vol_target: bool = True    # False → 杠杆恒 1 (无杠杆实验, gross ≤ 1)
    daily_vol_adjust: bool = True  # 混合频率: 期货 sleeve 日频微调
    futures_commission: float = 0.0001
    etf_commission: float = 0.0005
    min_obs_per_asset: int = MIN_OBS

    # V6 子因子 (PLAN §13)
    subfactor_mode: str = "none"   # none / a / b / c / abc / ab / ac / bc
    subfactor_alpha: float = 0.3   # 子因子叠加强度 (调整范围 ±60%)
    subfactor_score: object = None  # 外部传入 W × N 综合因子分数 (z-score 加权和)
    subfactor_alpha_pos: float | None = None  # 非对称 alpha: 正向分数
    subfactor_alpha_neg: float | None = None  # 非对称 alpha: 负向分数


@dataclass
class BacktestResult:
    """回测输出."""

    nav: pd.Series
    weights: pd.DataFrame          # 每日收盘后执行权重 w_exec
    turnover: pd.Series            # 每日换手 Σ|Δw|
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def _rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """调仓日掩码: W-FRI → 周五; ME → 月内最后交易日."""
    if freq == "W-FRI":
        return dates.dayofweek == 4
    if freq == "ME":
        idx = pd.Series(dates, index=dates)
        month_end = idx.groupby(dates.to_period("M")).max().values
        return np.isin(dates, month_end)
    raise ValueError(f"不支持调仓频率: {freq}, 可选 W-FRI / ME")


def _available(returns_hist: pd.DataFrame, window: int,
               min_obs: int) -> list[str]:
    """窗口内观测充足的资产 (渐入池)."""
    w = returns_hist.iloc[-window:]
    return [c for c in returns_hist.columns if w[c].notna().sum() >= min_obs]


def _equal_weights(cols_all: list[str], avail: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=cols_all)
    if avail:
        w[avail] = 1.0 / len(avail)
    return w


def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """迭代封顶再分配: 超限权重按比例回填给未超限资产 (HRP 集中度修复)."""
    if cap >= 1.0 or w.abs().sum() == 0:
        return w
    w = w.copy()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        base = float(w[under].sum())
        if base > 0:
            w[under] += excess * (w[under] / base)
        else:
            break
    return w


def _cap_group(w: pd.Series, members: list[str], cap: float) -> pd.Series:
    """类别合计封顶: 超出部分按比例回填给类别外资产."""
    members = [m for m in members if m in w.index]
    if not members or cap >= 1.0:
        return w
    w = w.copy()
    for _ in range(50):
        cur = float(w[members].sum())
        if cur <= cap:
            break
        scale = cap / cur
        excess = cur - cap
        w[members] *= scale
        outside = w.index.difference(members)
        base = float(w[outside].sum())
        if base > 0:
            w[outside] += excess * (w[outside] / base)
    return w


def subfactor_weights(
    base_weights: pd.Series,
    factor_score: pd.Series,
    alpha: float = 0.3,
    alpha_pos: float | None = None,
    alpha_neg: float | None = None,
) -> pd.Series:
    """子因子加权叠加 sleeve 基础权重 (PLAN §13.3.3).

    公式: w_final = w_base × (1 + α × factor_score)
    其中 factor_score ∈ [-2, +2] (z-score 范围), α = 0.3 → 调整范围 [-60%, +60%].
    输出: clip 到 [0, 1] (不允许负权重, 不允许单资产 > 100%).

    Args:
        base_weights: 基础权重 (Panel cols, 来自 _base_weights 或 trend overlay 后).
        factor_score: 子因子分数 (Panel cols, 与 base_weights 同 index).
        alpha: 默认叠加强度 (对称).
        alpha_pos: 正向分数 (score > 0) 的单独强度 (Sub-A 修复点 #3, 非对称叠加).
        alpha_neg: 负向分数 (score < 0) 的单独强度.

    Returns:
        调整后权重 Series (Panel cols).
    """
    if alpha == 0 or factor_score is None:
        return base_weights
    common = base_weights.index.intersection(factor_score.index)
    adjusted = base_weights.copy()
    score = factor_score.loc[common].fillna(0.0)
    base = base_weights.loc[common]
    if alpha_pos is None and alpha_neg is None:
        adjusted.loc[common] = base * (1.0 + alpha * score)
    else:
        ap = alpha_pos if alpha_pos is not None else alpha
        an = alpha_neg if alpha_neg is not None else alpha
        eff_alpha = np.where(score >= 0, ap, an)
        adjusted.loc[common] = base * (1.0 + eff_alpha * score)
    return adjusted.clip(lower=0.0, upper=1.0)


def _apply_subfactor(
    base: pd.Series,
    date,
    cfg: AllWeatherConfig,
) -> pd.Series:
    """从 cfg.subfactor_score (W × N) 中取 date 那一行, 应用 subfactor_weights.

    cfg.subfactor_score 是 None 或 mode='none' 时直接返回 base (零开销).
    """
    if cfg.subfactor_mode == "none" or cfg.subfactor_score is None:
        return base
    score_panel = cfg.subfactor_score
    if date not in score_panel.index:
        return base
    factor_score = score_panel.loc[date]
    return subfactor_weights(
        base, factor_score, cfg.subfactor_alpha,
        cfg.subfactor_alpha_pos, cfg.subfactor_alpha_neg,
    )


def _base_weights(returns_hist: pd.DataFrame, panel_cols: list[str],
                  cfg: AllWeatherConfig) -> pd.Series:
    """HRP / 等权 / inv-vol 基础权重 (未乘趋势/杠杆)."""
    avail = _available(returns_hist, cfg.window, cfg.min_obs_per_asset)
    if cfg.weight_scheme == "inv_vol":
        w = pd.Series(0.0, index=panel_cols)
        if avail:
            sub = returns_hist[avail].iloc[-cfg.window:]
            vols = sub.std()
            vols = vols.replace(0, np.nan).dropna()
            inv = 1.0 / vols
            w[inv.index] = inv / inv.sum()
        return w
    if not cfg.use_hrp:
        return _equal_weights(panel_cols, avail)
    if len(avail) < 5:  # MIN_ASSETS_FOR_HRP
        return _equal_weights(panel_cols, avail)
    return hrp_weights_from_returns(
        returns_hist, window=cfg.window, denoise=cfg.denoise,
        linkage_method=cfg.linkage, min_obs_per_asset=cfg.min_obs_per_asset,
    )


def _window_cov(returns_hist: pd.DataFrame, cfg: AllWeatherConfig) -> np.ndarray:
    """与权重求解同源的窗口协方差 (用于 ex-ante 波动).

    注意: 迟上市资产会把 dropna 交集窗口砍短 — 从可用资产的最晚首日起截取,
    保证窗口尽量接近 cfg.window.
    """
    avail = _available(returns_hist, cfg.window, cfg.min_obs_per_asset)
    sub = returns_hist[avail].iloc[-cfg.window:].dropna(axis=0, how="any")
    firsts = [sub[c].first_valid_index() for c in avail]
    firsts = [f for f in firsts if f is not None]
    if firsts:
        start = max(firsts)
        sub = sub.loc[start:]
    if cfg.denoise == "lw":
        return ledoit_wolf_shrink(sub, window=cfg.window)
    cov = sample_cov(sub, window=cfg.window)
    if cfg.denoise == "mp":
        from .cov_rmt import mp_denoise
        cov = mp_denoise(cov, n_obs=len(sub))
    return cov


def run_backtest(panel: pd.DataFrame, config: AllWeatherConfig | None = None,
                 compute_metrics_fn=None) -> BacktestResult:
    """运行混合频率回测.

    Args:
        panel: (T x 38) 价格面板 (fut_/etf_ 前缀), 见 data_loader
        config: 策略配置
        compute_metrics_fn: 指标函数 (默认复用 momentum_etf_rotation.common.metrics)

    Returns:
        BacktestResult (nav / weights / turnover / metrics)
    """
    cfg = config or AllWeatherConfig()
    if compute_metrics_fn is None:
        from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
            compute_metrics,
        )
        compute_metrics_fn = compute_metrics

    panel = panel.sort_index()
    dates = panel.index
    returns = panel.pct_change(fill_method=None)
    fut_cols = [c for c in panel.columns if c.startswith("fut_")]
    rate = {c: (cfg.futures_commission if c.startswith("fut_")
                else cfg.etf_commission) for c in panel.columns}

    rb_mask = _rebalance_mask(dates, cfg.rebalance)

    n = len(dates)
    w_exec = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    base_rb = pd.Series(0.0, index=panel.columns)   # 最近一次调仓的基础权重 (未含杠杆)
    leverage_rb = 1.0                                # 调仓日全池杠杆 (ETF 维持此值)
    lev_fut_cur = 1.0                                # 当前期货 sleeve 杠杆 (乘性修正的载体)
    port_ret_hist: list[float] = []                  # 已实现组合收益 (日频微调用)

    # ---- 决策循环: t 日收盘产生 w_exec[t] ----
    for i, date in enumerate(dates):
        is_rb = bool(rb_mask[i]) and i >= cfg.window + 1
        if is_rb:
            rets_hist = returns.loc[:date]
            base = _base_weights(rets_hist, list(panel.columns), cfg)

            # 趋势 overlay: 用截至 date 的价格 (按模式传参)
            ov_kwargs: dict = {}
            if cfg.trend_mode in ("v1", "v3"):
                ov_kwargs = {"lookback": cfg.trend_lookback,
                             "coef": cfg.trend_coef}
            elif cfg.trend_mode == "v4":
                ov_kwargs = {"lookback": cfg.trend_lookback}
            elif cfg.trend_mode == "tilt":
                ov_kwargs = {"lookback": cfg.trend_lookback,
                             "risk_adj": cfg.trend_risk_adj,
                             "floor": cfg.trend_floor}
            # trigA/trigB: 使用各自默认参数 (episode 锚定值), 不透传引擎参数
            ov = combined(panel.loc[:date], mode=cfg.trend_mode,
                          **ov_kwargs).iloc[-1]
            base = base * ov
            # V6 子因子 overlay (PLAN §13): 在 trend overlay 后, cap 前
            base = _apply_subfactor(base, date, cfg)
            if cfg.max_weight < 1.0:
                base = _cap_weights(base, cfg.max_weight)
            if cfg.bond_class_cap < 1.0:
                base = _cap_group(base, ["fut_t", "fut_tf", "fut_ts"],
                                  cfg.bond_class_cap)

            # ex-ante 波动 → 调仓杠杆 (cov 只含可用资产, base 对齐到同子集)
            cov = _window_cov(rets_hist, cfg)
            avail = _available(rets_hist, cfg.window, cfg.min_obs_per_asset)
            base_avail = base[avail].values
            var_ex = float(base_avail @ cov @ base_avail)
            # cov 为日频协方差 → 年化后再除目标波动 (否则杠杆虚大 √252 倍)
            sigma_ex = np.sqrt(max(var_ex, 0.0)) * np.sqrt(ANN)
            if not cfg.use_vol_target:
                leverage_rb = 1.0
            else:
                leverage_rb = (cfg.target_vol / sigma_ex) if sigma_ex > 0 else 1.0
                leverage_rb = float(np.clip(leverage_rb, 0.0, cfg.max_leverage))

            base_rb = base
            lev_fut_cur = leverage_rb
            w_exec.loc[date] = base * leverage_rb
        else:
            row = base_rb * leverage_rb  # 默认持有
            if (cfg.use_vol_target and cfg.daily_vol_adjust
                    and port_ret_hist and base_rb.abs().sum() > 0 and lev_fut_cur > 0):
                hist = np.array(port_ret_hist[-DAILY_VOL_LOOKBACK:])
                if len(hist) >= DAILY_VOL_LOOKBACK and hist.std() > 0:
                    sigma_d = hist.std() * np.sqrt(ANN)
                    # 乘性修正当前期货持仓 (稳定反馈, 有界); ETF 维持调仓日杠杆
                    c = float(np.clip(cfg.target_vol / sigma_d, 0.5, 2.0))
                    c = min(c, cfg.max_leverage / lev_fut_cur)  # 期货总杠杆约束
                    prev_row = w_exec.iloc[i - 1]
                    row = prev_row.copy()
                    row[fut_cols] = prev_row[fut_cols] * c
                    lev_fut_cur *= c
            w_exec.loc[date] = row

        # 记录当日 (以 w_exec[前一交易日] 持仓实现的) 组合收益, 供次日微调估计
        if i > 0:
            prev_w = w_exec.iloc[i - 1]
            r = float((prev_w * returns.iloc[i].fillna(0.0)).sum())
            port_ret_hist.append(r)

    # ---- NAV 与换手成本 ----
    nav_vals = [1.0]
    turnover_vals = [0.0]
    for i in range(1, n):
        w_applied = w_exec.iloc[i - 1]
        r = float((w_applied * returns.iloc[i].fillna(0.0)).sum())
        dw = (w_exec.iloc[i - 1] - w_exec.iloc[i - 2]).abs() if i >= 2 else w_exec.iloc[i - 1]
        cost = float(sum(dw[c] * rate[c] for c in panel.columns))
        turnover_vals.append(float(dw.sum()))
        nav_vals.append(nav_vals[-1] * (1.0 + r - cost))

    nav = pd.Series(nav_vals, index=dates, name="nav")
    turnover = pd.Series(turnover_vals, index=dates, name="turnover")

    # 年化换手 (双边, NAV 归一)
    active = nav > 0.01
    ann_turnover = float(turnover[active].mean() * ANN) if active.any() else 0.0

    metrics = compute_metrics_fn(nav, freq="D")
    metrics["ann_turnover"] = ann_turnover

    return BacktestResult(
        nav=nav,
        weights=w_exec,
        turnover=turnover,
        config=vars(cfg).copy(),
        metrics=metrics,
    )


def run_equal_weight_vol_target(panel: pd.DataFrame, **kwargs) -> BacktestResult:
    """消融 0 (baseline) 便捷入口: 等权 + vol target, 无 HRP/趋势."""
    cfg = AllWeatherConfig(use_hrp=False, trend_mode="none", **kwargs)
    return run_backtest(panel, cfg)
