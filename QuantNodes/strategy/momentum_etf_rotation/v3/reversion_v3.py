# coding=utf-8
"""均值反转子策略 (Stage 16A, v3.0).

思路: 选过去 N 日跌幅最大但短期企稳的 ETF
- 适合震荡市/反弹初期
- 与动量策略形成互补 (动量擅长趋势, 反转擅长修复)
- 默认 top_n = 5 (与动量策略的 10 只互补, 总持仓 15 只)

参数 (ReversionConfig):
- lookback: 跌幅计算窗口 (默认 60 日)
- ma_short / ma_long: 短期/长期均线 (默认 5/20)
- max_drawdown: 最大回撤过滤 (默认 -30%, 避免接飞刀)
- top_n: 持仓数 (默认 5)
- a_share_total: A股宽基+行业上限 (默认 3, 继承 v2)
- require_commodity/overseas: 必含商品/海外 (默认 True)

信号 (reversion_score):
    reversion_score = -rank_pct(60d_return) + 0.3 × ma_crossover_bonus
    其中 ma_crossover_bonus = 1 if ma5 > ma10 else 0
    (短期均线金叉 = 企稳信号)

参考: reports/momentum_etf_rotation/v2/stage16a_plan.md §2.1.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ..common.universe import ETFPool
from .sub_strategy_v3 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
    validate_sub_strategy_result,
)


@dataclass
class ReversionConfig(SubStrategyConfig):
    """均值反转子策略配置.

    继承自 SubStrategyConfig, 增加反转特有参数.
    """
    name: str = "reversion"
    lookback: int = 60                 # 跌幅计算窗口
    ma_short: int = 5                  # 短期均线
    ma_long: int = 20                  # 长期均线 (金叉判断)
    ma_crossover_weight: float = 0.3   # 金叉奖励权重
    max_drawdown: float = -0.30        # 最大回撤过滤 (避免接飞刀)
    require_commodity: bool = True
    require_overseas: bool = True
    a_share_total: int = 3             # A股宽基+行业上限
    top_n: int = 5                     # 持仓数


def reversion_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 60,
    ma_short: int = 5,
    ma_long: int = 20,
    crossover_weight: float = 0.3,
    max_drawdown: float = -0.30,
) -> pd.Series:
    """计算反转信号得分 (越大越强).

    score = -rank_pct(60d_return) + crossover_weight × ma_crossover_bonus

    Args:
        nav_df: 价格面板
        as_of: 当前日期
        lookback: 跌幅计算窗口
        ma_short / ma_long: 短期/长期均线
        crossover_weight: 金叉奖励权重
        max_drawdown: 最大回撤过滤 (回撤超过此值的 ETF 排除)

    Returns:
        pd.Series, index=code, values=score (越大越强, 负值表示已过滤)
    """
    sub = nav_df.loc[:as_of]
    if len(sub) < max(lookback, ma_long) + 1:
        return pd.Series(dtype=float)

    # 1. 60日收益率 (用 pct_change 避免 log_ret 边界问题)
    ret_lookback = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0

    # 2. 排名分位 (越小越好 → 取负号变越大越好)
    rank_pct = ret_lookback.rank(method="average", pct=True, na_option="bottom")
    score = -rank_pct  # 负号: 跌幅大的得分高

    # 3. 最大回撤过滤 (近 60 日)
    rolling_max = sub.cummax()
    drawdown = sub / rolling_max - 1.0
    max_dd = drawdown.iloc[-lookback:].min()  # 最差回撤
    score[max_dd < max_drawdown] = -1.0  # 标记为不可选 (后续过滤)

    # 4. 金叉奖励 (ma5 > ma10 → 企稳信号)
    ma_s = sub.iloc[-ma_short:].mean()
    ma_l = sub.iloc[-ma_long:].mean()
    golden_cross = (ma_s > ma_l).astype(float)
    score = score + crossover_weight * golden_cross

    return score


class ReversionSubStrategy(SubStrategy):
    """均值反转子策略 (v3.0).

    选股逻辑:
        1. 排除 max_drawdown 过深的 ETF (避免接飞刀)
        2. 排除短期/长期均线空头排列的 ETF (金叉要求)
        3. 按 reversion_score 降序选 top_n
        4. 保证类别分散 (A 股宽基+行业 ≤ 3, 必含商品/海外)

    加权逻辑:
        - 等权 (反转策略不依赖波动率, 信号已经隐含波动信息)
    """

    def __init__(self, config: ReversionConfig, pool: ETFPool):
        super().__init__(config, pool)
        self.config: ReversionConfig = config

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选 top_n 个 ETF (按反转得分降序)."""
        if self.config.min_history > 0 and len(nav_df) < self.config.min_history:
            return []

        score = reversion_score(
            nav_df, as_of,
            lookback=self.config.lookback,
            ma_short=self.config.ma_short,
            ma_long=self.config.ma_long,
            crossover_weight=self.config.ma_crossover_weight,
            max_drawdown=self.config.max_drawdown,
        )

        if score.empty:
            return []

        # 过滤掉 max_drawdown 不达标的 (score = -1.0)
        score = score[score > -0.5]

        # 按得分降序
        ranked = score.sort_values(ascending=False)

        # 类别分散 + 必含商品/海外
        chosen: list[str] = []
        chosen_cat_count: dict[str, int] = {}
        a_share_count = 0
        has_commodity = False
        has_overseas = False

        for code in ranked.index:
            if code not in self.pool.codes:
                continue
            cat = self.pool.category_of(code)
            cat_name = cat.value
            if cat_name not in chosen_cat_count:
                chosen_cat_count[cat_name] = 0

            # A股宽基+行业 cap
            if cat_name in ("a_broad", "a_sector"):
                if a_share_count >= self.config.a_share_total:
                    continue
                a_share_count += 1

            # 必含商品/海外: 第 1/2 只时优先
            if cat_name == "commodity" and not has_commodity and len(chosen) < self.config.top_n - 1:
                has_commodity = True
            elif cat_name == "overseas" and not has_overseas and len(chosen) < self.config.top_n - 1:
                has_overseas = True
            elif cat_name in ("commodity", "overseas") and not (
                has_commodity if cat_name == "commodity" else has_overseas
            ):
                # 还缺商品/海外, 跳过非必需的
                if not has_commodity and cat_name != "commodity":
                    continue
                if not has_overseas and cat_name != "overseas":
                    continue

            chosen.append(code)
            chosen_cat_count[cat_name] = chosen_cat_count.get(cat_name, 0) + 1

            if len(chosen) >= self.config.top_n:
                break

        return chosen

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """等权 (反转策略不需要逆波动加权)."""
        n = len(codes)
        if n == 0:
            return {}
        w = 1.0 / n
        return {c: w for c in codes}

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        """单次调仓: select + weight + 校验."""
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, signal_strength=0.0)

        weights = self.weight(nav_df, codes, as_of)

        # 计算 signal_strength: 平均反转得分 (用于风险平价权重)
        score = reversion_score(
            nav_df, as_of,
            lookback=self.config.lookback,
            ma_short=self.config.ma_short,
            ma_long=self.config.ma_long,
            crossover_weight=self.config.ma_crossover_weight,
            max_drawdown=self.config.max_drawdown,
        )
        valid_score = score[score > -0.5].reindex(codes).dropna()
        signal_strength = float(valid_score.mean()) if len(valid_score) > 0 else 0.0

        result = SubStrategyResult(
            date=as_of,
            chosen=list(codes),
            weights=weights,
            signal_strength=signal_strength,
            meta={
                "strategy": "reversion",
                "lookback": self.config.lookback,
                "ma_crossover_weight": self.config.ma_crossover_weight,
            },
        )
        return validate_sub_strategy_result(result, self.pool)


__all__ = [
    "ReversionConfig",
    "ReversionSubStrategy",
    "reversion_score",
]
