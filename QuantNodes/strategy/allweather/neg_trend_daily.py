"""自适应负向趋势因子 · 日频风控层 (D.1/D.2+, PLAN §17.12/§17.14).

子信号 (设计文稿 §3.3 + §5.29 急跌检测器扩展):
- sub_trend:  多周期 (5/10/20日) t-stat 风险调整动量, tanh(0.5·t-stat)
- sub_vol:    当前波动率 / 自身 60 日高分位
- sub_flow:   价跌 + 量增/OI 增 → 负向
- rapid:      资产通用急跌检测 (5日回撤 + vol-of-vol 突变, §5.29)

异质权重 (设计文稿 §3.3): equity 重 trend / futures 重 flow / gold 重 vol
rapid 维度: 所有资产统一权重 (急跌是速度问题, 非类别问题)
自适应阈值: 高波动 regime 阈值下移 (更易触发)
双阈值滞回: penalty 上行过 θ_high 才计满, 回落过 θ_low 才释放
消融开关: mode='3d' / 'trend' / 'rapid' / 'none'
include_rapid: 为 True 时在 trend/3d 上叠加 rapid (10% 权重)

输出: penalty ∈ [0,1] (N×T 日频), 乘性作用于权重 w×(1−penalty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── 异质权重 (按 asset_class) ──
CLASS_HETERO_WEIGHTS: dict[str, dict[str, float]] = {
    # (trend, vol, flow)
    "bond_rate": {"trend": 0.5, "vol": 0.3, "flow": 0.2},
    "equity_fut": {"trend": 0.5, "vol": 0.2, "flow": 0.3},
    "equity_etf_dom": {"trend": 0.5, "vol": 0.3, "flow": 0.2},
    "equity_etf_off": {"trend": 0.5, "vol": 0.3, "flow": 0.2},
    "precious": {"trend": 0.3, "vol": 0.5, "flow": 0.2},
    "nonferrous": {"trend": 0.3, "vol": 0.2, "flow": 0.5},
    "ferrous": {"trend": 0.3, "vol": 0.2, "flow": 0.5},
    "energy_chem": {"trend": 0.3, "vol": 0.2, "flow": 0.5},
    "agri_oilseed": {"trend": 0.3, "vol": 0.2, "flow": 0.5},
    "agri_soft": {"trend": 0.3, "vol": 0.2, "flow": 0.5},
}

DEFAULT_WEIGHTS = {"trend": 1 / 3, "vol": 1 / 3, "flow": 1 / 3}

# ── D.1 三档参数 (§5.27 稳定性扫描, 三档全超基线, 用户确认全部保留) ──
# 实测 (真实量价, D v2 lw 基线 1.036/DD -9.84%):
#   aggressive: Sharpe 1.142 / MaxDD -6.74% / 换手 14.6x (激活最多, 风控最强)
#   balanced:   Sharpe 1.111 / MaxDD -7.28% / 换手 12.5x (默认)
#   conservative: Sharpe 1.080 / MaxDD -7.64% / 换手 10.7x (激活最少)
NEG_TREND_PRESETS: dict[str, dict[str, float]] = {
    "aggressive": {"base_threshold": 0.4, "threshold_low": 0.2},
    "balanced": {"base_threshold": 0.5, "threshold_low": 0.3},
    "conservative": {"base_threshold": 0.6, "threshold_low": 0.4},
}

DEFAULT_PRESET = "balanced"


def get_preset_params(preset: str) -> dict[str, float]:
    """按档位名取阈值参数 (未知名 → balanced)."""
    return dict(NEG_TREND_PRESETS.get(preset, NEG_TREND_PRESETS[DEFAULT_PRESET]))


def _asset_class(col: str, asset_classes: dict[str, list[str]]) -> str | None:
    for cls, members in asset_classes.items():
        if col in members:
            return cls
    return None


def _hetero_weight(col: str, asset_classes: dict[str, list[str]]) -> dict[str, float]:
    cls = _asset_class(col, asset_classes)
    if cls is None:
        return DEFAULT_WEIGHTS
    return CLASS_HETERO_WEIGHTS.get(cls, DEFAULT_WEIGHTS)


def _slope_vol_adj(close: pd.Series, win: int) -> pd.Series:
    """win 日风险调整动量 (t-stat 尺度): 收益率单位斜率 / vol·√win.

    缺陷修复 (§5.27): 原实现斜率用价格单位 (P_t − P_{t−win})/win, 除以
    收益率 vol 后单位错配 — 高价资产 (铜/金) 的 slope/vol 是天文数字,
    tanh 必然重度饱和 (实测 70% 格子 |x|≥1) → 危险分长期高位.
    修正: 动量在收益率单位计算 (win 日累计收益), 除以 vol·√win
    → 经典 t-stat 风险调整动量, 量纲统一.
    """
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(win, min_periods=max(win // 2, 5)).std()
    mom = ret.rolling(win).sum()          # win 日累计收益 (收益率单位)
    return mom / (vol.replace(0, np.nan) * np.sqrt(win))


def compute_sub_signals(
    panel: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    oi: pd.DataFrame | None = None,
    trend_windows: tuple[int, ...] = (5, 10, 20),
    vol_lookback: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """三维子信号, 各标准化到 [−1, 1].

    Returns:
        (sub_trend, sub_vol, sub_flow): 各 (T×N)
    """
    # ── trend: 多周期斜率均值 → tanh 压缩到 [−1,1] ──
    slopes = []
    for win in trend_windows:
        slopes.append(_slope_vol_adj(panel, win))
    trend_raw = pd.concat(slopes, axis=0).groupby(level=0).mean() \
        if len(slopes) == 1 else sum(s.dropna(how="all") for s in slopes) \
        / max(len([s for s in slopes if not s.empty]), 1)
    # 缺陷修复 (§5.27): 缩放 0.5 防饱和 — 原 tanh(x) 在 |x|~2-3 时重度饱和
    # (25% 分位 = ±1.0) → trend-only 危险分长期高位 → 常态砍仓
    sub_trend = np.tanh(0.5 * trend_raw)

    # ── vol: 当前 vol / 自身 vol_lookback 高分位 (0.8) ──
    ret = panel.pct_change(fill_method=None)
    vol_now = ret.rolling(20, min_periods=10).std()
    vol_high = vol_now.rolling(vol_lookback, min_periods=vol_lookback // 2).quantile(0.8)
    vol_ratio = (vol_now / vol_high.replace(0, np.nan)).clip(0, 3)
    # 上行低波突变/下行放量才计分: 波动比 > 1 → 贡献正 (危险), 映射到 [0,1]
    sub_vol = ((vol_ratio - 1.0)).clip(0, 1)

    # ── flow: 价跌 + 量增 (期货加 OI 增) ──
    ret_signed = ret.clip(upper=0)  # 只取下跌日
    if volume is not None and not volume.empty:
        vol_ma = volume.rolling(20, min_periods=10).mean()
        vol_growth = (volume / vol_ma.replace(0, np.nan) - 1.0).clip(-1, 2)
        flow = (-ret_signed) * vol_growth.clip(lower=0)
    else:
        flow = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    if oi is not None and not oi.empty:
        # 期货: OI 增 + 价跌 = 增仓下行 (确认)
        oi_ma = oi.rolling(20, min_periods=10).mean()
        oi_growth = (oi / oi_ma.replace(0, np.nan) - 1.0).clip(-1, 2)
        fut_cols = [c for c in panel.columns if c in oi.columns]
        for col in fut_cols:
            flow[col] = flow[col] + (-ret_signed[col]) * oi_growth[col].clip(lower=0)
    sub_flow = (flow / (flow.rolling(60, min_periods=20).max().replace(0, np.nan))
                ).clip(0, 1).fillna(0.0)

    # 对齐 panel 全索引 (滚动窗口损失的前几行补 0)
    sub_trend = sub_trend.reindex(panel.index).fillna(0.0)
    sub_vol = sub_vol.reindex(panel.index).fillna(0.0)
    sub_flow = sub_flow.reindex(panel.index).fillna(0.0)

    return sub_trend, sub_vol, sub_flow


def _rapid_move_score(returns: pd.DataFrame, short: int = 5,
                      long: int = 60) -> pd.DataFrame:
    """资产通用急跌检测 (§5.29): 5日回撤代理 + vol-of-vol 扩张 → [0, 2].

    A. 短期回撤代理: |5日累计收益| / (60日最大5日累计收益) → [0, 1]
    B. 波动率扩张: (5日vol / 60日vol − 1).clip(0, 1) — 只响应 vol 增加, 基线 0
    combined = max(A, B): 任一触发即有效, 覆盖"已发生急跌"和"正在发生急跌".

    注: B 用 (ratio−1) 而非 raw ratio, 因为 vol5/vol60 ≈ 1.0 是随机数据基线,
    raw ratio 会让 rapid 在平静期也常活跃 (§5.29 发现的设计修正).
    """
    ret = returns.pct_change(fill_method=None)
    # A. 短期回撤代理 (5日累计收益的相对幅度)
    cum5 = ret.rolling(short).sum()
    cum5_abs = cum5.abs()
    roll_max_abs = cum5_abs.rolling(long, min_periods=long // 2).max()
    dd_proxy = (cum5_abs / roll_max_abs.replace(0, np.nan)).clip(0, 1).fillna(0.0)

    # B. 波动率扩张 (基线 0, 只响应 vol 增加)
    vol5 = ret.rolling(short).std()
    vol60 = ret.rolling(long, min_periods=long // 2).std()
    vol_spike = ((vol5 / vol60.replace(0, np.nan)) - 1.0).clip(0, 1).fillna(0.0)

    # 合成: 取两者的 max (任一触发即有效)
    return pd.DataFrame(
        np.maximum(dd_proxy.values, vol_spike.values),
        index=returns.index, columns=returns.columns,
    )


def compute_neg_trend_penalty(
    panel: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    oi: pd.DataFrame | None = None,
    asset_classes: dict[str, list[str]] | None = None,
    mode: str = "3d",
    include_rapid: bool = False,
    base_threshold: float = 0.5,
    threshold_low: float = 0.3,
    vol_regime_window: int = 60,
) -> pd.DataFrame:
    """负趋势因子 → penalty ∈ [0,1] (T×N 日频).

    Args:
        panel: 价格面板
        volume/oi: 量价面板 (None → flow 退化为 0)
        asset_classes: 类别映射 (异质权重)
        mode: '3d' / 'trend' / 'rapid' / 'none'
        include_rapid: True 时在 trend/3d 上叠加 rapid 维度 (10%权重, 原维度缩至90%)
        base_threshold: 基础阈值 (高波 regime 下移); 0.5/0.3 为 §5.27 稳定性扫描均衡点
        threshold_low: 释放阈值 (滞回下沿)
        vol_regime_window: regime 检测窗口

    Returns:
        penalty DataFrame ∈ [0,1], 索引对齐 panel
    """
    if mode == "none" and not include_rapid:
        return pd.DataFrame(0.0, index=panel.index, columns=panel.columns)

    asset_classes = asset_classes or {}
    sub_trend, sub_vol, sub_flow = compute_sub_signals(panel, volume, oi)

    # 计算 rapid 信号 (需要时)
    rapid_score = None
    if mode == "rapid" or include_rapid:
        rapid_score = _rapid_move_score(panel)

    if mode == "rapid":
        # standalone rapid: full weight on rapid dimension
        neg = rapid_score
    elif mode == "trend":
        raw = sub_trend.clip(upper=0)
        neg = (-raw).clip(lower=0)
        if include_rapid and rapid_score is not None:
            neg = neg * 0.9 + rapid_score * 0.1
    else:  # "3d" (or 3d+rapid)
        neg = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        for col in panel.columns:
            hw = _hetero_weight(col, asset_classes)
            neg[col] = (-sub_trend[col]).clip(lower=0) * hw["trend"] \
                + sub_vol[col] * hw["vol"] \
                + sub_flow[col] * hw["flow"]
        if include_rapid and rapid_score is not None:
            neg = neg * 0.9 + rapid_score * 0.1

    # 自适应阈值: 组合波动 regime (全池均值 vol 分位) → 阈值下移
    ret = panel.pct_change(fill_method=None)
    pool_vol = ret.mean(axis=1).rolling(20, min_periods=10).std()
    vol_q = pool_vol.rolling(vol_regime_window, min_periods=20).rank(pct=True)
    threshold = base_threshold * (1.0 - 0.3 * vol_q.fillna(0.5))  # 高波 → 阈值↓30%

    # 滞回 (作用在 neg 分数上): 未激活 → neg > threshold(t) 进入;
    # 已激活 → neg > threshold_low 保持 (退出需深回落)
    neg_vals = neg.to_numpy()
    thr = threshold.to_numpy()[:, None]
    out = np.zeros_like(neg_vals)
    prev_active = np.zeros(neg_vals.shape[1], dtype=bool)
    for t in range(len(neg_vals)):
        cur = np.where(prev_active,
                       neg_vals[t] > threshold_low,    # 已激活: 回落到 low 之下才释放
                       neg_vals[t] > thr[t, 0])        # 未激活: 越过自适应阈值才进入
        # penalty 幅度: (neg − threshold)/(1 − threshold), 激活期下限 0.1
        mag = (neg_vals[t] - thr[t, 0]) / max(1.0 - base_threshold, 1e-6)
        out[t] = np.where(cur, np.clip(mag, 0.1, 1.0), 0.0)
        prev_active = cur > 0

    penalty = pd.DataFrame(out, index=panel.index, columns=panel.columns)
    return penalty
