# coding=utf-8
"""v6.2 量价族 + IC 加权 + 因子正交化 (Stage 27 v6.2 + Phase 1/3 look-ahead fix).

v6.2 = v5 选股 + IC 加权 + 因子正交化 (残差化) + v5.1.1 加权.

与 v6.1 区别:
- 在因子计算之后, 选股之前, 加一层因子正交化
- 正交化方法: Gram-Schmidt 残差法 或 QR 分解对称正交

无风控层 (与 v6.1 一致).

[Stage 30] 交易成本: 默认启用 (5bp 佣金 + 10bp 滑点), 与 v1.0 对齐.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.sub_strategy_v4 import SubStrategyConfig
from ..v5.industry_factors import FactorEngineConfig
from ..v5.industry_rotation_v5 import (
    IndustryRotationV5SubStrategy,
    compute_composite_factor,
)
from ..v5.industry_rotation_v5_1 import inverse_vol_weights_v5_1
from .factor_orthogonal import (
    get_factor_ir_order,
    get_factor_ir_order_expanding,
    get_factor_ir_order_warmup,
    orthogonalize_factor_panel,
    orthogonalize_factor_panel_qr,
    PREDEFINED_FACTOR_ORDER,
)
from .factor_weighting import (
    compute_ic_timeseries,
    compute_factor_weights,
    compute_softmax_weights,
    align_weights_with_rebal_dates,
    DEFAULT_HORIZON_DAYS,
    MIN_MONTHS_FOR_IC,
    DEFAULT_SMOOTH_WINDOW,
)
from .industry_rotation_v6_1 import V6_1Config, V6_1SubStrategy


# ============================================================
# Config
# ============================================================
@dataclass
class V6_2Config(SubStrategyConfig):
    """v6.2 配置: v6.1 + 因子正交化 (Phase 1/3 look-ahead fix).

    字段继承 V6_1Config 全部, 加 (Phase 1):
    - use_orthogonal: 是否启用正交化
    - sort_method: 正交化顺序策略
                    可选: "ir_expanding" | "predefined" | "ir_full" (DEPRECATED) | "qr"

    [Stage 29 决策]
    Phase 1 (expanding IR Gram-Schmidt) 5-fold 验证 4/5 胜 v6.1 IC12:
      - v6.2_ir_expanding mean OOS Calmar 1.512 vs v6.1 0.867
      - 每 fold 都 ≥ v6.1 (除 fold 4)
      - 升为 v6.2 默认 (主推), 标 PROMISING

    其他 sort_method 状态:
    - "warmup_ir" 12m: 早期主推但 OOS 0.629 < v6.1 0.748, 5-fold 不如 expanding, 降为备选
    - "predefined": 金融预定义, OOS 0.674, 备选
    - "qr": 0.056, Phase 3 失败, 不推荐
    - "ir_full": DEPRECATED 含 look-ahead, 见 tests/_helpers/deprecated_order.py
    """
    name: str = "industry_rotation_v6_2"

    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)
    top_n: int = 5
    factor_weights: dict[str, float] | None = None

    use_ic_weighting: bool = True
    ic_horizon_days: int = DEFAULT_HORIZON_DAYS
    ic_min_months: int = MIN_MONTHS_FOR_IC
    ic_smooth_window: int = DEFAULT_SMOOTH_WINDOW

    use_orthogonal: bool = True
    ir_order_min_periods: int = 12

    # [Stage 29] 5-fold 验证 4/5 胜 v6.1, 升为默认
    sort_method: str = "ir_expanding"
    ir_order_min_periods_lookback: int = 36
    warmup_months: int = 0  # ir_expanding 不需要 warmup (默认 sort_method 已切, 留 0)

    # 向后兼容字段 (新默认值已废弃, 仅兼容测试)
    use_predefined_factor_order: bool = True
    weight_method: str = "clip"
    sharpness: float = 3.0
    min_ir_threshold: float = 0.5

    rebalance_freq: str = "M"
    min_history: int = 252

    universe: tuple[str, ...] | None = None

    max_weight: float = 0.25
    vol_window: int = 60
    vol_floor: float = 0.01
    rebal_lag: int = 1

    # [Stage 30] 交易成本 (与 v1.0 对齐)
    cost_enabled: bool = True
    commission_bp: float = 5.0       # 佣金 (基点, 万 5)
    slippage_bp: float = 10.0       # 滑点 (基点)


class V6_2SubStrategy(V6_1SubStrategy):
    """v6.2 子策略, 继承 v6.1 (含 IC 加权), 加正交化."""
    config: V6_2Config

    def __init__(self, config: V6_2Config):
        super().__init__(config)
        self.config: V6_2Config = config


# ============================================================
# 正交化 routing
# ============================================================
def _orthogonalize_panel(factor_panel, factor_cfg, panel_close, rebal_dates, sort_method):
    """[Phase 1/3] 对原始 panel 做正交化, 返回 (panel_orth, factors_used).

    Args:
        factor_panel: dict[code] -> DataFrame
        factor_cfg: FactorEngineConfig
        panel_close: 收盘价
        rebal_dates: 调仓日
        sort_method: "qr" | "ir_expanding" | "predefined" | "ir_full"
                       "qr" 是 Phase 3 主推 (顺序无关, 无 look-ahead)
                       "ir_full" DEPRECATED (含 look-ahead, 仅 ablation)
    """
    factors = list(factor_cfg.name_map.keys())
    factors_in_panel = set()
    for code, df in factor_panel.items():
        if not df.empty:
            factors_in_panel.update(df.columns)

    # ---- qr (Phase 3 主推) ----
    if sort_method == "qr":
        panel_orth = orthogonalize_factor_panel_qr(factor_panel, rebal_dates)
        # QR 输出列名是 f_qr_0, f_qr_1, ... (K 列)
        # 用第一个 code 的列决定 K
        first_df = next(iter(panel_orth.values())) if panel_orth else None
        if first_df is not None and not first_df.empty:
            qr_factors = list(first_df.columns)
        else:
            qr_factors = [f"f_qr_{k}" for k in range(len(factors_in_panel))]
        return panel_orth, qr_factors

    # ---- predefined (Stage 28 已测 OOS 0.473, 仅 ablation 对照) ----
    if sort_method == "predefined":
        order = [f for f in PREDEFINED_FACTOR_ORDER if f in factors_in_panel]
        if len(order) < 2:
            return factor_panel, list(factors_in_panel)
        panel_orth = orthogonalize_factor_panel(factor_panel, order, rebal_dates)
        return panel_orth, order

    # ---- ir_full (Stage 28 DEPRECATED, 含 look-ahead, 仅 ablation 对照) ----
    if sort_method == "ir_full":
        # [Stage 29 决策] ir_full 全样本 IR 排序含严重 look-ahead, 已从生产路径移出.
        # 历史 ablation 脚本 (scripts/v6_2_*_ablation.py) 仍可从
        # tests/strategy/momentum_etf_rotation/_helpers/deprecated_order.py 直接调用
        # get_factor_ir_order_deprecated 作对照, 但 production path 不再支持.
        raise NotImplementedError(
            "sort_method='ir_full' is DEPRECATED (Stage 28) — 严重 look-ahead. "
            "生产路径已禁用. "
            "ablation 对照请从 tests/strategy/momentum_etf_rotation/_helpers/"
            "deprecated_order.py 导入 get_factor_ir_order_deprecated 替代."
        )

    # ---- ir_expanding (Phase 1 已测 OOS 0.430, 失败, 仅 ablation 对照) ----
    if sort_method == "ir_expanding":
        order_per_date = get_factor_ir_order_expanding(
            factor_panel, panel_close, rebal_dates, factors,
            horizon=21, min_periods=12, lookback_months=36,
        )
        seq_counter: dict[tuple, int] = {}
        for d, o in order_per_date.items():
            key = tuple(o)
            seq_counter[key] = seq_counter.get(key, 0) + 1
        default_order = max(seq_counter.items(), key=lambda x: x[1])[0]
        panel_orth = orthogonalize_factor_panel(
            factor_panel, default_order, rebal_dates, order_per_date,
        )
        return panel_orth, list(default_order)

    # ---- warmup_ir (Phase 4 主推: 早期 24 月数据定序, 整个回测期共用) ----
    if sort_method == "warmup_ir":
        warmup_months = getattr(factor_cfg, "_v6_2_warmup_months_override", None) or 24
        order = get_factor_ir_order_warmup(
            factor_panel, panel_close, rebal_dates, factors,
            horizon=21, warmup_months=warmup_months,
        )
        if len(order) < 2:
            return factor_panel, factors
        panel_orth = orthogonalize_factor_panel(factor_panel, order, rebal_dates)
        return panel_orth, order

    raise ValueError(f"Unknown sort_method: {sort_method}")


# ============================================================
# QR 路径 composite helper (不依赖 cfg.factor_cfg.name_map, 用 factors_used)
# ============================================================
def _qr_composite_factor(
    factor_panel: dict[str, pd.DataFrame],
    factors_used: list[str],
    as_of: pd.Timestamp,
    weights: pd.Series | dict[str, float] | None,
) -> pd.Series:
    """[Phase 3] 在 QR panel 上做 composite: 截面 z-score 加权求和.

    与 v5.compute_composite_factor 接口相同, 但 panel 用 QR 输出 (列名 f_qr_k),
    不用 cfg.factor_cfg.name_map.
    """
    if isinstance(weights, pd.Series):
        w_dict = {k: float(weights.loc[k]) if k in weights.index else 0.0
                  for k in factors_used}
    elif isinstance(weights, dict):
        w_dict = weights
    elif weights is None:
        w_dict = {f: 1.0 / len(factors_used) for f in factors_used}
    else:
        w_dict = {f: 1.0 / len(factors_used) for f in factors_used}

    composite = pd.Series(dtype=float)
    codes_for_date = [c for c, df in factor_panel.items()
                      if as_of in df.index]
    if not codes_for_date:
        return composite

    for fac in factors_used:
        w = w_dict.get(fac, 0.0)
        if w == 0:
            continue
        # 截面 z-score
        vals = {}
        for c in codes_for_date:
            if fac in factor_panel[c].columns:
                v = factor_panel[c][fac].loc[as_of]
                if pd.notna(v):
                    vals[c] = float(v)
        if not vals:
            continue
        v_series = pd.Series(vals)
        mu, sd = v_series.mean(), v_series.std()
        if sd == 0 or pd.isna(sd):
            z = v_series - mu  # all zeros
        else:
            z = (v_series - mu) / sd
        composite = composite.add(z * w, fill_value=0.0)
    return composite


# ============================================================
# Backtest 主入口
# ============================================================
def run_v6_2_backtest(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    cfg: V6_2Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.Series:
    """v6.2 回测: 与 v6.1 流程一致 + 因子正交化.

    Args:
        panel_close: 收盘价面板
        panel_ohlcv: OHLCV 面板
        cfg: v6.2 config (None = 默认 sort_method="qr")
        rebalance_dates: 调仓日期 (None = 月末)

    Returns:
        NAV Series
    """
    if cfg is None:
        cfg = V6_2Config()

    # [Stage 29] sort_method 提前校验, 在主流程前抛 NotImplementedError
    if cfg.sort_method == "ir_full":
        raise NotImplementedError(
            "sort_method='ir_full' is DEPRECATED (Stage 28) — 严重 look-ahead. "
            "生产路径已禁用. "
            "ablation 对照请从 tests/strategy/momentum_etf_rotation/_helpers/"
            "deprecated_order.py 导入 get_factor_ir_order_deprecated 替代."
        )

    dates = panel_close.index

    if rebalance_dates is None:
        # [Phase 4 修复] 取每月最后一个**实际 trading day**
        # pandas 的 resample("M").last() 在 trading day index 上
        # 输出 calendar month end (例如 03-31, 04-30), 不是 trading day,
        # 导致 orthogonalize / compute_cross_section_ic 找不到 as_of 而退化.
        # 改为: groupby by month, 取每月最后一个 trading day (落在原 index 中).
        period = dates.to_period("M")
        rebal_series = dates.to_series().groupby(period).tail(1)
        rebal_dates_idx = rebal_series.index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)
    rebal_set = set(d for d in rebal_dates_idx if d in dates)

    from ..v5.industry_factors import compute_all_factors_panel
    factor_panel_raw = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)
    factors = list(cfg.factor_cfg.name_map.keys())

    if cfg.use_orthogonal:
        # [Phase 4] warmup_months 暴露到 cfg; 在 factor_cfg 上挂钩透传给 _orthogonalize_panel
        # 用 cfg 自身属性作为传递通道, 避免改 factor_orthogonal 接口
        cfg.factor_cfg._v6_2_warmup_months_override = cfg.warmup_months
        factor_panel_used, factors_used = _orthogonalize_panel(
            factor_panel_raw, cfg.factor_cfg, panel_close, rebal_dates_idx,
            sort_method=cfg.sort_method,
        )
    else:
        factor_panel_used = factor_panel_raw
        factors_used = factors

    # [Phase 3] QR 路径特殊处理
    use_qr_panel = cfg.sort_method == "qr"

    sub = V6_2SubStrategy(cfg)
    sub._factor_panel = factor_panel_used
    if len(dates) > 0:
        sub._last_init_date = dates[0]

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

    # 模拟回测
    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue

        is_rebal = date in rebal_set and i > 252

        if is_rebal:
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

            new_weights = {}
            if chosen:
                try:
                    new_weights = sub.weight(panel_close, chosen, date)
                except Exception:
                    new_weights = {}

            # [Stage 30] 交易成本扣减 (仅调仓日, 用新旧权重差)
            if cfg.cost_enabled:
                turnover = 0.0
                for code in set(list(prev_weights.keys()) + list(new_weights.keys())):
                    w_old = prev_weights.get(code, 0.0)
                    w_new = new_weights.get(code, 0.0)
                    turnover += abs(w_new - w_old)
                cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000
                # 成本在调仓日扣减
                nav.iloc[i] = nav.iloc[i - 1] * (1 - turnover * cost_rate)
            else:
                nav.iloc[i] = nav.iloc[i - 1]

            prev_weights = new_weights
        elif prev_weights:
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
    "V6_2Config",
    "V6_2SubStrategy",
    "run_v6_2_backtest",
]
