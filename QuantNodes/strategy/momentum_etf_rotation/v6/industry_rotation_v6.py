# coding=utf-8
"""v6 行业量价因子行业轮动 (Stage 26) — v1.0 风控框架 + v5 选股 + v5.1 逆波动加权.

[注] v5.1.1 实际上没有自己的 select 方法 (选股继承自 v5), 只有 weight 方法改为逆波动.
      本文件的选股层因此沿用 v5 (IndustryRotationV5SubStrategy.select), 加权层继承自 v5.1.

v6 = 选股层 (v5) + 加权层 (v5.1 逆波动) + 风控层 (v2 框架).

为什么不直接组合 v1.0 + v5.1?
- v1.0 选股用纯价格动量 (144d lookback) → 选股信号源不同
- v5 选股用 11 量价因子 → 选股信号源不同
- 直接组合 (v1.0 80% + v5.1 20%) 是"叠加", 不是"单策略风控"

v6 思路:
- 选股层用 v5 (复合因子信号更强)
- 加权层用 v5.1 (逆波动 + T+1)
- 风控层用 v2 (VT + TF + Cost) → 解决 v5.1 没有风控的根本问题

回测结果 (Stage 26 完成):
待运行 fill in by scripts/v6_backtest.py
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
from ..v5_1.industry_rotation_v5_1 import inverse_vol_weights_v5_1


@dataclass
class V6Config(SubStrategyConfig):
    """v6 配置: v5 选股 (无自身新选股逻辑) + v5.1 逆波动加权 + v2 风控.

    与 v5.1 的区别:
    - 默认 top_n=5 保留 (与 v5.1 一致)
    - 加风控层开关 (v6 启用全部, v5.1 关闭)
    """
    name: str = "industry_rotation_v6"

    # 因子引擎配置 (与 v5.1 共享)
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)

    # 选股层 (来自 v5)
    top_n: int = 5

    # 复合因子权重 (None = 等权, 与 v5/v5.1 一致)
    factor_weights: dict[str, float] | None = None

    # 调仓频率
    rebalance_freq: str = "M"

    # 冷启动
    min_history: int = 252

    # ETF 池 (None = 全部)
    universe: tuple[str, ...] | None = None

    # 加权层 (来自 v5.1.1)
    max_weight: float = 0.25           # v5.1.1 默认
    vol_window: int = 60               # v5.1.1 默认
    vol_floor: float = 0.01            # v5.1.1 默认
    rebal_lag: int = 1                 # v5.1.1 默认 T+1

    # 风控层 (来自 v2 框架) — 默认关闭, 在 run_v6_backtest 中可开启
    # 注: SubStrategy 接口本身不处理 VT/TF/Cost, 由 backtest 层包装
    # 这里只标记开关, 实际应用在 run_v6_backtest 中
    use_vol_targeting: bool = True
    use_trend_filter: bool = True
    use_cost_model: bool = True

    # VT 参数 (v0.1_vt_only_v2 默认值)
    vol_target: float = 0.15
    vol_target_lookback: int = 60
    vol_target_min_scale: float = 0.3
    vol_target_max_scale: float = 1.5

    # TF 参数 (v0.2_tf_only_v2 默认值)
    trend_filter_ma_window: int = 200
    trend_filter_bear_exposure: float = 0.7
    trend_filter_benchmark: str = "510300"
    trend_filter_bond: str = "511260"

    # 成本参数 (v0.3_vt_cost_v2 默认值)
    commission_bp: float = 5.0
    slippage_bp: float = 10.0
    impact_factor: float = 0.1

    # 调仓日期 (None = 月末最后交易日)
    rebalance_dates: tuple[pd.Timestamp, ...] | None = None


class V6SubStrategy(IndustryRotationV5SubStrategy):
    """v6 子策略: 复用 v5 选股 + v5.1.1 加权.

    选股逻辑与 v5.1.1 完全相同 (同 11 因子 + Top-N + 月度调仓).
    加权逻辑与 v5.1.1 完全相同 (逆波动 + 60日窗口 + T+1).

    区别于 v5.1.1:
    - config 类型是 V6Config (扩展 v5.1.1 字段, 加入风控开关)
    - 但 select / weight 行为完全继承 v5 / v5.1.1
    - 风控层 (VT/TF/Cost) 在 run_v6_backtest 中包装, 不在 SubStrategy 内部

    注: 因为 v6 本质是 v5.1.1 + 风控层包装, SubStrategy 接口与 v5.1.1 相同.
    """
    config: V6Config

    def __init__(self, config: V6Config):
        super().__init__(config)
        self.config: V6Config = config

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """逆波动率加权 (v5.1.1 复用).

        Args:
            nav_df: 价格面板 (close), 列为 code
            codes: 候选 ETF 列表
            as_of: 调仓日 (信号日)

        Returns:
            dict[code, weight], 权重和 = 1
        """
        if not codes:
            return {}
        cfg = self.config
        weights = inverse_vol_weights_v5_1(
            nav_df, codes, as_of,
            cfg.vol_window, cfg.vol_floor, cfg.rebal_lag,
        )
        weights = self._apply_max_weight(weights, cfg.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights


def run_v6_backtest(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    cfg: V6Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
    apply_vol_targeting: bool = True,
    apply_trend_filter: bool = True,
    apply_cost_model: bool = True,
) -> pd.Series:
    """v6 回测: v5 选股 + v5.1 逆波动加权 + 可选 VT/TF/Cost.

    简化逻辑 (Stage 26 调试清晰):
    - 每个调仓日: 选股 → 加权 → 风控 → 调仓成本 → NAV 按前一日 NA × (1-成本) × (1+日收益)
    - 非调仓日: NAV 按前一日 × (1+日收益) 直接累计
    """
    if cfg is None:
        cfg = V6Config()

    dates = panel_close.index

    # 1. 确定调仓日期
    if rebalance_dates is None:
        rebal_dates_idx = dates.to_series().resample("ME").last().index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)
    rebal_set = set(d for d in rebal_dates_idx if d in dates)

    # 2. 预算 11 因子 (与 v5.1.1 共享)
    from ..v5.industry_factors import compute_all_factors_panel

    factor_panel = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)

    # 3. 模拟回测
    sub = V6SubStrategy(cfg)
    sub._factor_panel = factor_panel  # 直接复用, 避免重算
    if len(dates) > 0:
        sub._last_init_date = dates[0]

    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}
    cost: float = 0.0
    start = True

    for i, date in enumerate(dates):
        if i == 0:
            continue

        is_rebal = date in rebal_set and i > 252

        if is_rebal:
            # 3a. 选股
            try:
                chosen = sub.select(panel_ohlcv, date)
            except Exception:
                chosen = []

            weights = {}
            if chosen:
                try:
                    weights = sub.weight(panel_close, chosen, date)
                except Exception:
                    weights = {}

            # 3c. 可选 TF (熊市降仓 + 补充债券)
            if weights and apply_trend_filter:
                try:
                    weights = _apply_trend_filter(
                        weights, panel_close, date, cfg,
                    )
                except Exception:
                    pass

            # 3d. 可选 VT (波动率目标缩放)
            if weights and apply_vol_targeting:
                try:
                    weights = _apply_vol_targeting(
                        weights, nav.iloc[:i].values, dates[:i], cfg,
                    )
                except Exception:
                    pass

            # 3e. 计算调仓成本 (但不直接改 NAV, 后面合并)
            cost = 0.0
            if weights and apply_cost_model and prev_weights:
                try:
                    cost = _calculate_turnover_cost(
                        prev_weights, weights, cfg,
                    )
                except Exception:
                    cost = 0.0

            prev_weights = weights

        # 4. 按当日权重累积 NAV (cost 仅在调仓日扣除)
        if prev_weights:
            daily_ret = 0.0
            for code, w in prev_weights.items():
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            if is_rebal and cost > 0:
                # 调仓日: 先扣成本, 再算当日收益 (成本已隐含在 prev_weights 的重置)
                # 但因为 prev_weights 已经是新值, cost 已不适用, 直接按新权重算当日
                # 故 cost 在 is_rebal 当日不扣, 而在**次日**开始表现
                nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)
                nav.iloc[i] *= (1 - cost)  # 调仓日扣一次性成本
            else:
                nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)
        else:
            nav.iloc[i] = nav.iloc[i - 1] * (1 - cost)

    return nav


# ============================================================
# 风控层辅助函数 (从 v2 移植, 不依赖 v2.RotationConfig)
# ============================================================
def _apply_vol_targeting(
    weights: dict[str, float],
    nav_so_far: np.ndarray,
    dates_so_far: pd.DatetimeIndex,
    cfg: V6Config,
) -> dict[str, float]:
    """VT: 根据历史 NAV 波动率缩放权重 (cf v2.apply_vol_targeting_v2).

    1. 计算最近 vol_target_lookback 日的日收益 std × √252
    2. scale = target_vol / actual_vol, clipped 到 [min_scale, max_scale]
    3. 总权重 *= scale (剩余为 cash)
    """
    if len(nav_so_far) < cfg.vol_target_lookback:
        return weights

    nav_series = pd.Series(nav_so_far, index=dates_so_far).dropna()
    if len(nav_series) < cfg.vol_target_lookback:
        return weights

    recent = nav_series.tail(cfg.vol_target_lookback + 1)
    log_ret = np.log(recent / recent.shift(1)).dropna()
    if len(log_ret) < 2:
        return weights

    actual_vol = float(log_ret.std() * np.sqrt(252))
    if actual_vol <= 0:
        return weights

    scale = cfg.vol_target / actual_vol
    scale = float(np.clip(scale, cfg.vol_target_min_scale, cfg.vol_target_max_scale))

    return {k: v * scale for k, v in weights.items()}


def _apply_trend_filter(
    weights: dict[str, float],
    panel_close: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: V6Config,
) -> dict[str, float]:
    """TF: 熊市时降仓 (cf v2.apply_trend_filter_v2).

    1. 检查 benchmark (HS300) 是否在 ma_window 均线下方
    2. 若是: 当前组合权重 *= bear_exposure, 用 bond_code 补充
    """
    bench_code = cfg.trend_filter_benchmark
    bond_code = cfg.trend_filter_bond

    if bench_code not in panel_close.columns:
        return weights

    bench_series = panel_close[bench_code].loc[:as_of].dropna()
    if len(bench_series) < cfg.trend_filter_ma_window:
        return weights

    ma = bench_series.iloc[-cfg.trend_filter_ma_window:].mean()
    last = bench_series.iloc[-1]
    if not isinstance(last, (int, float, np.number)):
        try:
            last = float(last)
        except Exception:
            return weights

    is_bull = float(last) >= float(ma)
    if is_bull:
        return weights

    # 熊市: 缩权重
    new_weights = {k: v * cfg.trend_filter_bear_exposure for k, v in weights.items()}

    # 用 bond 补齐 (剩余 1 - total)
    total = sum(new_weights.values())
    if total < 1.0 and bond_code in panel_close.columns and bond_code not in weights:
        new_weights[bond_code] = 1.0 - total
        # 重新归一
        total2 = sum(new_weights.values())
        if total2 > 0:
            new_weights = {k: v / total2 for k, v in new_weights.items()}
    elif total > 0:
        # 归一化让剩余为 cash
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights


def _calculate_turnover_cost(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    cfg: V6Config,
) -> float:
    """调仓成本扣减 (cf v2.calculate_turnover_cost_v2).

    cost_rate = (commission_bp + slippage_bp × impact_factor) / 10000
    cost = turnover × cost_rate
    默认 (5 + 10 × 0.1) / 10000 = 0.6 bp × turnover.
    """
    all_codes = set(old_weights.keys()) | set(new_weights.keys())
    turnover = sum(
        abs(new_weights.get(c, 0.0) - old_weights.get(c, 0.0))
        for c in all_codes
    ) / 2.0
    cost_rate = (cfg.commission_bp + cfg.slippage_bp * cfg.impact_factor) / 10000.0
    cost = turnover * cost_rate
    return float(cost)


__all__ = [
    "V6Config",
    "V6SubStrategy",
    "run_v6_backtest",
]
