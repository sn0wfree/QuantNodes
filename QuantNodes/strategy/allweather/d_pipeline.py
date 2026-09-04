"""D sleeve: 资产级配置管线 (设计文稿核心版, PLAN §17).

管线: AVS分层去噪 → HRP(Ward) → 债券硬上限 + RC重分配 → vol_target
频率: 可配置 (默认 W-FRI; 支持 20D 月频)

与 A sleeve (class ERC) 是两种独立配置引擎的头对头测试.
D.0 不含负趋势因子 (延后 D.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .cov_avs import hierarchical_avs
from .hrp import hrp_weights
from .backtest import _rebalance_mask, ANN

# ── 常量 ──
BOND_COLS = ["fut_t", "fut_tf", "fut_ts"]
MIN_ASSETS = 5
DEFAULT_BOND_CAP = 0.40


# ── 输出结构 ──
@dataclass
class DSleeveResult:
    nav: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series | None = None


# ── 核心权重计算 ──
def _d_base_weights(
    returns_hist: pd.DataFrame,
    panel_cols: list[str],
    window: int = 252,
    n_clusters: int | None = None,
    bond_cap: float = DEFAULT_BOND_CAP,
    min_obs_per_asset: int = 60,
) -> pd.Series:
    """单次调仓: AVS → HRP(ward) → 债券cap → RC重分配 → 归一."""
    avail = [c for c in panel_cols
             if returns_hist[c].notna().sum() >= min_obs_per_asset]
    if len(avail) < MIN_ASSETS:
        out = pd.Series(0.0, index=panel_cols)
        if avail:
            out[avail] = 1.0 / len(avail)
        return out

    # ① AVS 去噪
    C_avs, delta, _, _ = hierarchical_avs(
        returns_hist[panel_cols], window=window, n_clusters=n_clusters,
    )
    _ = delta  # 诊断用, 不影响权重

    # ② HRP(Ward)
    w_raw = hrp_weights(C_avs, linkage_method="ward")
    w = pd.Series(w_raw, index=[c for c in panel_cols
                                 if c in returns_hist[avail].columns
                                 and returns_hist[c].notna().sum() >= min_obs_per_asset])

    # 对齐到 panel_cols (缺失资产权重=0)
    full_w = pd.Series(0.0, index=panel_cols)
    full_w[w.index] = w.values

    # ③ 债券硬上限 + RC 重分配
    bond_total = full_w[BOND_COLS].sum()
    if bond_cap < 1.0 and bond_total > bond_cap:
        overflow = bond_total - bond_cap
        # 缩放债券到 cap
        scale = bond_cap / bond_total if bond_total > 0 else 0
        for b in BOND_COLS:
            full_w[b] *= scale
        # 非债资产按 pre-cap RC share 重分配
        non_bond = [c for c in avail if c not in BOND_COLS]
        if non_bond:
            # 用 AVS 协方差计算 pre-cap RC
            non_bond_idx = [list(avail).index(c) for c in non_bond]
            w_avail = w.values
            mrc = C_avs @ w_avail
            rc = w_avail * mrc
            rc_non_bond = rc[non_bond_idx]
            rc_sum = rc_non_bond.sum()
            if rc_sum > 0:
                rc_share = rc_non_bond / rc_sum
            else:
                rc_share = np.ones(len(non_bond)) / len(non_bond)
            for i, c in enumerate(non_bond):
                full_w[c] += overflow * rc_share[i]

    # ④ 归一
    total = full_w.sum()
    if total > 0:
        full_w = full_w / total
    return full_w


def run_d_sleeve(
    panel: pd.DataFrame,
    bond_cap: float = DEFAULT_BOND_CAP,
    window: int = 252,
    n_clusters: int | None = None,
    rebalance_freq: str = "W-FRI",
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    min_obs_per_asset: int = 60,
    compute_metrics_fn=None,
) -> DSleeveResult:
    """运行 D sleeve 回测.

    Args:
        panel: (T x N) 价格面板
        bond_cap: 债券硬上限 (默认0.40)
        window: 协方差估计窗口 (HRP+AVS共用)
        n_clusters: Ward簇数 (None=N//6)
        rebalance_freq: 调仓频率 ("W-FRI" / "20D" / "ME")
        target_vol: 目标波动率
        max_leverage: 杠杆上限
        min_obs_per_asset: 资产渐入池门槛
        compute_metrics_fn: 指标函数 (None = 不计算)

    Returns:
        DSleeveResult (nav, weights)
    """
    panel = panel.sort_index()
    dates = panel.index
    returns = panel.pct_change(fill_method=None)

    # 调仓掩码
    rb_mask = _rebalance_mask(dates, rebalance_freq)

    w_exec = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    leverage_rb = 1.0

    for i, date in enumerate(dates):
        is_rb = bool(rb_mask[i]) and i >= window + 1
        if is_rb:
            rets_hist = returns.loc[:date]
            base = _d_base_weights(
                rets_hist, list(panel.columns), window=window,
                n_clusters=n_clusters, bond_cap=bond_cap,
                min_obs_per_asset=min_obs_per_asset,
            )

            # vol_target
            avail = [c for c in panel.columns
                     if rets_hist[c].notna().sum() >= min_obs_per_asset]
            if avail and max_leverage > 0:
                cov = np.cov(rets_hist[avail].iloc[-window:].dropna(
                    axis=0, how="any").to_numpy(), rowvar=False)
                base_avail = base[avail].values
                var_ex = float(base_avail @ cov @ base_avail)
                sigma_ex = np.sqrt(max(var_ex, 0.0)) * np.sqrt(ANN)
                if sigma_ex > 0:
                    leverage_rb = min(target_vol / sigma_ex, max_leverage)
                else:
                    leverage_rb = 1.0
            else:
                leverage_rb = 1.0

            w_exec.loc[date] = base * leverage_rb

        elif i > 0:
            # 非调仓日: 延用上次权重 (无杠杆微调)
            w_exec.iloc[i] = w_exec.iloc[i - 1]

    # 组合收益
    port_ret = (w_exec.shift(1) * returns).sum(axis=1)
    nav = (1 + port_ret).cumprod()
    nav.name = "D_sleeve"

    return DSleeveResult(nav=nav, weights=w_exec)
