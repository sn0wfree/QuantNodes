"""
v7.0 行业轮动: v6.2 + 5 状态 HMM 动态 vol_target (Stage 30.3).

[设计] v7.0 = v6.2 (11 量价因子 + IC 加权 + Gram-Schmidt) + 5 状态 vol_target

[与 v6.2 关键差异]
1. 5 状态 HMM (复苏/过热/滞胀/衰退/中性) → 5 个 vol_target (20%/12%/6%/10%/14%)
2. 每个调仓日根据 HMM 输出状态, 计算 vol scale = clip(target_vol / realized_vol)
3. weights *= scale (剩余仓位等价于"现金", 通过把组合波动率压到目标)

[PIT 关键]
- HMM 输入是 PIT 调整后的 5 维宏观特征 (PMI/CPI/M2/CN10Y/US10Y)
- 调仓日 d 用 release_date <= d 的最新宏观值 → 防 look-ahead
- vol_target 在 d 日确定后, d+1 月度生效 (T+1 lag)

[不沿用 v6.2 之处]
- 不将 5 宏观因子纳入 11 因子 IC 加权 (宏观是时间序列, 截面 IC 不可定义)
- 宏观因子只用于状态检测 → vol_target 调整, 不直接参与选股
- 这是 v7.0 的核心简化: 选股用 v6.2 11 因子, 仓位用 HMM 5 状态

[与 v6.2 兼容]
- 关闭 use_regime=True 退化为 v6.2 行为 (vol_target = 1.0, 不缩放)
- 因子加权 / 正交化 / 反向波动加权全部沿用 v6.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..v6_2.industry_rotation_v6_2 import (
    V6_2Config,
    V6_2SubStrategy,
    _orthogonalize_panel,
    _qr_composite_factor,
)
from ..v5.industry_rotation_v5 import compute_composite_factor
from ..v6_1.factor_weighting import (
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
)
from .regime_macro import (
    REGIME_NAMES,
    REGIME_VOL_TARGETS,
    _build_pit_features,
)
from .factor_macro import CACHE_DIR


@dataclass
class V7Config(V6_2Config):
    """v7.0 配置: 继承 v6.2, 加 5 状态 vol_target.

    [Stage 30.3 默认]
    - sort_method: 沿用 v6.2 默认 "ir_expanding" (Stage 29 PROMISING)
    - use_regime: True (5 状态 HMM 启用)
    - vol_lookback: 60 (realized_vol 窗口, 与 v2 一致)
    - vol_min_scale/max_scale: 0.3/2.0 (防止过度缩放)
    - 5 状态 vol_target 来自 REGIME_VOL_TARGETS
    """
    name: str = "industry_rotation_v7"

    # [v7.0 新增] 5 状态 vol_target 开关
    use_regime: bool = True
    vol_lookback: int = 60
    vol_min_scale: float = 0.3
    vol_max_scale: float = 2.0

    # 5 状态 → vol_target 映射 (可被用户覆盖)
    regime_vol_targets: dict = field(default_factory=lambda: dict(REGIME_VOL_TARGETS))

    # 宏观因子缓存目录
    macro_cache_dir: str = str(CACHE_DIR)


class V7SubStrategy(V6_2SubStrategy):
    """v7.0 子策略: v6.2 + 5 状态 vol_target 缩放.

    选股/加权/正交化完全沿用 v6.2; 仅在 weight() 后对 weights 应用 vol scaling.
    """
    config: V7Config

    def __init__(self, config: V7Config):
        super().__init__(config)
        self.config: V7Config = config
        # 当前调仓日的状态/缩放, 由 run_v7_0_backtest 注入
        self.current_regime_: str | None = None
        self.current_vol_scale_: float = 1.0

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """v7.0: 沿用 v6.2 inverse_vol_weights_v5_1, 再按 current_vol_scale_ 缩放."""
        # 1. 沿用 v5.1 inverse_vol 加权
        from ..v5_1.industry_rotation_v5_1 import inverse_vol_weights_v5_1
        weights = inverse_vol_weights_v5_1(
            nav_df, list(codes), as_of,
            vol_window=self.config.vol_window,
            vol_floor=self.config.vol_floor,
        )
        if not weights:
            return {}
        # 2. 应用 max_weight 约束 (v6.2 通过 _apply_max_weight, 这里手动)
        weights = self._apply_max_weight(weights, self.config.max_weight)
        # 3. [v7.0 新增] vol_target 缩放
        if self.config.use_regime and self.current_vol_scale_ != 1.0:
            weights = {k: v * self.current_vol_scale_ for k, v in weights.items()}
        return weights


# ============================================================
# 状态→vol_scale 工具函数
# ============================================================
def _compute_vol_scale(
    nav: pd.Series,
    target_vol: float,
    as_of: pd.Timestamp,
    lookback: int = 60,
    min_scale: float = 0.3,
    max_scale: float = 2.0,
) -> float:
    """计算 vol_target 缩放系数.

    scale = clip(target_vol / realized_vol, min_scale, max_scale)
    realized_vol 为 lookback 日年化波动率 (× √252).
    """
    rets = nav.loc[:as_of].pct_change().dropna()
    if len(rets) < lookback:
        return 1.0
    realized_vol = rets.iloc[-lookback:].std() * np.sqrt(252)
    if realized_vol <= 0:
        return 1.0
    scale = target_vol / realized_vol
    return float(np.clip(scale, min_scale, max_scale))


# ============================================================
# Backtest 主入口
# ============================================================
def run_v7_0_backtest(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    cfg: V7Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
    regime_timeline: pd.DataFrame | None = None,
) -> pd.Series:
    """v7.0 回测: v6.2 选股 + 5 状态 vol_target 缩放.

    Args:
        panel_close: 收盘价面板
        panel_ohlcv: OHLCV 面板
        cfg: v7.0 config (None = 默认)
        rebalance_dates: 调仓日期 (None = 月末)
        regime_timeline: 预计算的 5 状态时间线 (None = 在函数内自算)

    Returns:
        NAV Series
    """
    if cfg is None:
        cfg = V7Config()

    # [Stage 29 兼容] 沿用 v6.2 的 ir_full DEPRECATED 校验
    if cfg.sort_method == "ir_full":
        from ..v6_2.industry_rotation_v6_2 import run_v6_2_backtest
        # 直接复用 v6.2 的错误信息
        run_v6_2_backtest(panel_close, panel_ohlcv, cfg, rebalance_dates)

    dates = panel_close.index

    # 1. 调仓日 (沿用 v6.2 逻辑)
    if rebalance_dates is None:
        period = dates.to_period("M")
        rebal_series = dates.to_series().groupby(period).tail(1)
        rebal_dates_idx = rebal_series.index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)
    rebal_set = set(d for d in rebal_dates_idx if d in dates)

    # 2. 因子 panel + 正交化 (沿用 v6.2)
    from ..v5.industry_factors import compute_all_factors_panel
    factor_panel_raw = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)
    factors = list(cfg.factor_cfg.name_map.keys())

    if cfg.use_orthogonal:
        cfg.factor_cfg._v6_2_warmup_months_override = cfg.warmup_months
        factor_panel_used, factors_used = _orthogonalize_panel(
            factor_panel_raw, cfg.factor_cfg, panel_close, rebal_dates_idx,
            sort_method=cfg.sort_method,
        )
    else:
        factor_panel_used = factor_panel_raw
        factors_used = factors

    use_qr_panel = cfg.sort_method == "qr"

    # 3. IC 加权 (沿用 v6.2)
    factor_weights_ts = None
    if cfg.use_ic_weighting:
        ic_ts = compute_ic_timeseries(
            factor_panel_used, panel_close, rebal_dates_idx, factors_used,
            horizon=cfg.ic_horizon_days,
        )
        factor_weights_df = compute_factor_weights(
            ic_ts,
            min_months=cfg.ic_min_months,
            smooth_window=cfg.ic_smooth_window,
        )
        factor_weights_ts = align_weights_with_rebal_dates(
            factor_weights_df, rebal_dates_idx, dates,
        )

    # 4. [v7.0 新增] 5 状态时间线 (PIT 调整)
    if cfg.use_regime:
        if regime_timeline is None:
            # 内部自算: 用 _build_pit_features + zscore + HMM
            # 沿用 regime_macro.build_regime_timeline (POC 时已验证)
            from .regime_macro import build_regime_timeline
            # 用 dates 范围
            start = dates.min().strftime("%Y-%m-%d")
            end = dates.max().strftime("%Y-%m-%d")
            regime_timeline = build_regime_timeline(start=start, end=end)
        # 构造 regime 查表: date -> regime_name
        regime_lookup: dict[pd.Timestamp, str] = {}
        if "date" in regime_timeline.columns and "regime" in regime_timeline.columns:
            for _, row in regime_timeline.iterrows():
                regime_lookup[pd.Timestamp(row["date"])] = row["regime"]
        elif regime_timeline.index.name == "date" or "regime" in regime_timeline.columns:
            for d, r in zip(regime_timeline.index, regime_timeline["regime"] if "regime" in regime_timeline.columns else regime_timeline.iloc[:, 0]):
                regime_lookup[pd.Timestamp(d)] = r
    else:
        regime_lookup = {}

    # 5. 模拟回测
    sub = V7SubStrategy(cfg)
    sub._factor_panel = factor_panel_used
    if len(dates) > 0:
        sub._last_init_date = dates[0]

    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue

        is_rebal = date in rebal_set and i > 252

        if is_rebal:
            # 5.1 [v7.0 新增] 调仓日确定状态 + vol_scale
            if cfg.use_regime:
                # 找最近一个 <= date 的 regime
                regime = None
                for d_regime in sorted(regime_lookup.keys(), reverse=True):
                    if d_regime <= date:
                        regime = regime_lookup[d_regime]
                        break
                if regime is None:
                    regime = "neutral"  # fallback
                target_vol = cfg.regime_vol_targets.get(regime, 0.14)
                sub.current_regime_ = regime
                sub.current_vol_scale_ = _compute_vol_scale(
                    nav, target_vol, date,
                    lookback=cfg.vol_lookback,
                    min_scale=cfg.vol_min_scale,
                    max_scale=cfg.vol_max_scale,
                )
            else:
                sub.current_regime_ = "neutral"
                sub.current_vol_scale_ = 1.0

            # 5.2 因子加权 (沿用 v6.2)
            if factor_weights_ts is not None and date in factor_weights_ts.index:
                curr_fw = factor_weights_ts.loc[date]
            else:
                curr_fw = None

            try:
                if use_qr_panel:
                    composite = _qr_composite_factor(
                        factor_panel_used, factors_used, date, curr_fw,
                    )
                else:
                    composite = compute_composite_factor(
                        factor_panel_used, cfg.factor_cfg, date, curr_fw,
                    )
                if composite.empty or len(composite) < cfg.top_n:
                    chosen = []
                else:
                    chosen = composite.sort_values(ascending=False).head(cfg.top_n).index.tolist()
            except Exception:
                chosen = []

            # 5.3 加权 (V7SubStrategy.weight 内部已应用 vol_scale)
            weights = {}
            if chosen:
                try:
                    weights = sub.weight(panel_close, chosen, date)
                except Exception:
                    weights = {}
            prev_weights = weights

        if prev_weights:
            daily_ret = 0.0
            for code, w in prev_weights.items():
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)

    return nav


__all__ = [
    "V7Config",
    "V7SubStrategy",
    "run_v7_0_backtest",
    "_compute_vol_scale",
]
