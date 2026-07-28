# coding=utf-8
"""V7_3 v2 完整还原 — 忠实于 source notebook (NowCasting之Factor Mimicking组合-高频宏观因子.ipynb).

[重大错误修正]
v7.3 v1 失败: OOS Calmar 0.010 vs source 1.626.
根因: 5 个差异 (见 docs/38):
  1. 数据池: 5 ETF → 13 INDICES (level-1, 沪深300/500/1000/恒生 + 4 中债 + 4 商品)
  2. 调仓窗口: 8 quarter = 2 年
  3. Bootstrap times: 500
  4. Bootstrap resample: 78-104 周
  5. Symmetry 时机: 窗口全样本 Symmetry (Klein 2013)

[忠实复刻 source cell 102 + 104]
- main_idx.resample('W').last().pct_change(1)    [w=13 indices, weekly]
- pd.concat([idx, factor]).dropna()             [dropna 严格]
- quarter_window = 8                            [2 years]
- bootstrap_lasso_mapping(times=500, resample=78-104)
- Symmetry(rolling_window)                      [窗口全样本白化]
- FactorRiskParity(beta.T[factor_cols], factor_cov, sum=[0.9, 1.0])
- max_weight=0.5
- 简单回测: np.dot(opt_weight, NV.T), cumprod()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .bootstrap_lasso import BootstrapLassoMapping
from .data_loader import INDEX_COLS, EXPANDED_COLS, EQUITY_ETF_COLS, COMMODITY_ETF_COLS, EXPANDED_BOND_INDICES
from .factor_risk_parity import FactorRiskParityOptimizer

# [Stage 6 TF 权益专属化] 债券指数常量 (与 INDEX_COLS 同级)
# 熊市时 freed weight 按比例分配到这些债券指数
BOND_INDICES = [
    '中债10年期国债指数', '中债3-5年期国债指数', '中债1-3年国债财富指数',
    '中债国开行债券总指数', '中债企业债总指数',
]


@dataclass
class V7_3Config:
    """v7.3 v2 配置 (忠实于 source)."""
    rebalance_freq: str = "Q"

    # 回看窗口 (source cell 102 length=8)
    quarter_window: int = 8

    # Bootstrap-Lasso (source cell 102)
    bootstrap_times: int = 500
    bootstrap_resample_min: int = 52 * 1 + 26   # 78 周 = 1.5 年
    bootstrap_resample_max: int = 52 * 2       # 104 周 = 2 年
    bootstrap_random_state: int = 42
    bootstrap_cache_alpha: bool = True

    # Symmetry (窗口内全样本, source cell 102)
    symmetry_min_periods: int = 12

    # FactorRiskParity (source cell 94)
    max_weight: float = 0.5
    sum_lower: float = 0.9
    sum_upper: float = 1.0
    rp_max_iter: int = 200
    rp_tol: float = 1e-8

    # 池 (源 cell 102)
    index_pool: tuple[str, ...] = tuple(INDEX_COLS)  # 13 indices
    factor_cols: tuple[str, ...] = (
        "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
        "无风险收益率", "信用利差因子", "期限利差因子_债",
        "期限利差因子_股", "宏观汇率因子",  # 8 factors (source 不含加权)
    )

    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0

    # [Stage 4 v2 新增 2026-07-13] 趋势过滤 (Trend Filter)
    # 当 benchmark 跌破 ma_window 日均线时, 缩放权益仓位到 exposure_bear
    # 释放权重按比例分配给债券 (flight to safety)
    # v7_macro_baseline 默认 False (不启用); v7_macro_baseline_v2_tf 启用
    trend_filter_enabled: bool = False
    trend_filter_benchmark: str = "沪深300指数"
    trend_filter_ma: int = 200
    trend_filter_bear: float = 0.5

    # [Stage 6 TF 权益专属化] 权益指数列表 (可自由添加)
    equity_indices: list = field(default_factory=lambda: [
        '沪深300指数', '中证500指数', '中证1000', '恒生指数',
    ])

    # [Stage 7 v5 硬止损 2026-07-13] 净值回撤止损
    # 当组合 NAV 相对历史峰值回撤 > stop_loss_threshold 时, 强制权益清零, 全仓债券
    # 这是最基础的风控, 不依赖任何宏观/市场信号, 防止系统性风险
    # v5 stop_loss_enabled 默认 False; v5_stop_loss 启用
    stop_loss_enabled: bool = False
    stop_loss_threshold: float = -0.10   # 10% DD 触发 (用户决策: 8%太紧, 10%为佳)
    stop_loss_bond_alloc: float = 1.0    # 止损后 100% 债券


@dataclass
class V7_4Config(V7_3Config):
    """v7.4 扩大资产池配置 (51 ETFs + 5 bond indices = 56 assets)."""
    asset_pool: str = "expanded"  # "index" | "expanded"

    # 资产分类 (expanded pool)
    index_pool: tuple[str, ...] = tuple(EXPANDED_COLS)  # 56 assets
    equity_cols: tuple[str, ...] = tuple(EQUITY_ETF_COLS)      # 45 equity ETFs
    commodity_cols: tuple[str, ...] = tuple(COMMODITY_ETF_COLS) # 6 commodity ETFs
    bond_cols: tuple[str, ...] = tuple(EXPANDED_BOND_INDICES)   # 5 bond indices

    # v7.3.2: β 预筛选 + 分散度约束 (借鉴 CICC)
    n_assets: int = 15                    # 最大持仓数
    div_a_share_max: int = 8              # A股(宽基+行业+SmartBeta) ≤ 8
    div_hk_max: int = 2                   # 港股 ≤ 2
    div_commodity_min: int = 1            # 商品 ≥ 1
    div_commodity_max: int = 3            # 商品 ≤ 3
    div_overseas_min: int = 1             # 海外 ≥ 1
    div_overseas_max: int = 3             # 海外 ≤ 3
    div_bond_min: int = 1                 # 债券 ≥ 1
    div_bond_max: int = 5                 # 债券 ≤ 5


@dataclass
class V7_5Config(V7_4Config):
    """v7.5 连续 TF Score + 时变 LASSO 配置 (2026-07-13).

    相对 v7.4 改动:
    - 连续 TF Score: 替代二值 MA200 (trend_filter_*), 用 trend_score_* 系列参数
    - 时变 LASSO: 滚动窗口替代 expanding 窗口 (lasso_rolling_window)

    [TF Score 设计] (用户深度讨论 2026-07-13)
    二值 MA200 缺陷:
    - 信息损失: 距 MA200 5% 和 20% 触发同样减仓
    - 滞后: 200 日均线反应慢
    - 忽视非宏观信号: 没有动量/vol 维度

    连续 Score 公式:
        score = w_ma × MA200距离_score
              + w_mom × 60日动量_score
              + w_vol × 波动率比率_score

    仓位调整 (线性插值):
        if score < bear_threshold: equity_scale = bear_equity_alloc (0.3)
        elif score > bull_threshold: equity_scale = bull_equity_alloc (1.2)
        else: 线性插值 bear_equity_alloc → bull_equity_alloc
    """
    # TF Score 开关 (与 trend_filter_enabled 互斥)
    tf_score_enabled: bool = False

    # TF Score 权重 (用户决策: ma200:0.5, momentum:0.3, vol:0.2)
    tf_score_weights: dict = field(default_factory=lambda: {
        "ma200": 0.5,
        "momentum_60d": 0.3,
        "vol_ratio": 0.2,
    })

    # TF Score 阈值
    tf_score_bear_threshold: float = -0.3   # 低于此 → 强熊市
    tf_score_bull_threshold: float = 0.3    # 高于此 → 强牛市

    # 仓位缩放因子
    tf_score_bear_equity_alloc: float = 0.3  # 强熊市: 30% 权益
    tf_score_bull_equity_alloc: float = 1.2  # 强牛市: 120% 权益 (杠杆效果, 受 max_weight 约束)

    # [Stage 7 v5 Step 3] 时变 LASSO: 滚动窗口 (None=expanding 兼容, 156=3年滚动)
    lasso_rolling_window: int | None = None


def symmetry_full_window(
    sample: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    """对 sample 整个窗口做 Symmetry (仿 source cell 102, Klein 2013)."""
    if not isinstance(factor_cols, list):
        factor_cols = list(factor_cols)
    factor_cols = [c for c in factor_cols if c in sample.columns]
    if not factor_cols:
        return None
    F = sample[factor_cols].dropna()
    if len(F) < 12:
        return None
    cov = np.cov(F.values, rowvar=False)
    D, U = np.linalg.eigh(cov)
    D = np.maximum(D, 1e-8)
    S = U @ np.diag(D ** -0.5) @ U.T
    out = F.values @ S
    return pd.DataFrame(out, index=F.index, columns=F.columns)


def apply_trend_filter(
    w: pd.Series,
    benchmark_price: pd.Series,
    as_of: pd.Timestamp,
    cfg: V7_3Config,
) -> pd.Series:
    """[Stage 6 TF 权益专属化] 应用趋势过滤 (equity→bonds).

    熊市 (benchmark < ma_window 日均线): 只减权益仓位 × bear,
                                         释放权重按比例分配给债券 (flight to safety).
    多头: 返回 w 不变.

    Args:
        w: 当前 FRP 算出的权重 (index = INDEX_COLS)
        benchmark_price: benchmark 日价格 (pd.Series)
        as_of: 当前调仓日
        cfg: V7_3Config 配置 (含 trend_filter_* 字段)

    Returns:
        应用 TF 后的新权重 Series.
    """
    if not cfg.trend_filter_enabled:
        return w
    s = benchmark_price.loc[:as_of].dropna()
    if len(s) < cfg.trend_filter_ma:
        return w  # 数据不足, 默认多头
    ma = s.iloc[-cfg.trend_filter_ma:].mean()
    if s.iloc[-1] >= ma:
        return w  # 多头

    # 熊市: 只减权益, freed weight → 债券按比例分配
    bear = cfg.trend_filter_bear

    # 支持 expanded pool (V7_4Config) 和 index pool (V7_3Config)
    if isinstance(cfg, V7_4Config):
        eq_cols = list(cfg.equity_cols)
        bd_cols = list(cfg.bond_cols)
    else:
        eq_cols = list(cfg.equity_indices)
        bd_cols = BOND_INDICES

    equity_mask = w.index.isin(eq_cols)
    bond_mask = w.index.isin(bd_cols)

    # 计算释放的权益权重
    freed_weight = float((w[equity_mask] * (1.0 - bear)).sum())

    w_new = w.copy()
    w_new[equity_mask] = w_new[equity_mask] * bear  # 权益减仓

    # 释放权重按比例分配给债券
    bond_sum = float(w_new[bond_mask].sum())
    if bond_sum > 0:
        w_new[bond_mask] = w_new[bond_mask] + freed_weight * (w_new[bond_mask] / bond_sum)

    return w_new


def compute_trend_score(
    benchmark_price: pd.Series,
    as_of: pd.Timestamp,
    cfg: V7_5Config,
) -> float:
    """[Stage 7 v5 连续 TF] 计算连续 trend score ∈ [-1, +1].

    组成:
    1. MA200 距离: (price - MA200) / MA200, × 5, clip [-1, 1]
    2. 60日动量: price/price[60] - 1, × 5, clip [-1, 1]
    3. 波动率比率: -1 × (vol_20d / vol_60d - 1), × 2, clip [-1, 1] (高 vol = 恐慌 → -1)

    加权合成:
        score = w_ma × ma200_score + w_mom × mom_score + w_vol × vol_score
        score = clip(score, -1, 1)

    Returns:
        float in [-1, +1], 正=牛市, 负=熊市, 0=中性.
    """
    s = benchmark_price.loc[:as_of].dropna()
    if len(s) < 200:
        return 0.0  # 数据不足, 中性

    # 1. MA200 距离
    ma200 = s.iloc[-200:].mean()
    ma200_dist = (s.iloc[-1] - ma200) / ma200
    ma200_score = float(np.clip(ma200_dist * 5, -1, 1))

    # 2. 60日动量
    if len(s) >= 60:
        mom_60 = (s.iloc[-1] / s.iloc[-60]) - 1
        mom_score = float(np.clip(mom_60 * 5, -1, 1))
    else:
        mom_score = 0.0

    # 3. 波动率比率
    rets = np.log(s / s.shift(1)).dropna()
    if len(rets) >= 60:
        vol_current = rets.iloc[-20:].std()
        vol_history = rets.iloc[-60:].std()
        vol_ratio = float(vol_current / vol_history) if vol_history > 0 else 1.0
        vol_score = float(np.clip(-(vol_ratio - 1) * 2, -1, 1))
    else:
        vol_score = 0.0

    # 加权合成
    w = cfg.tf_score_weights
    score = (
        w.get("ma200", 0.5) * ma200_score
        + w.get("momentum_60d", 0.3) * mom_score
        + w.get("vol_ratio", 0.2) * vol_score
    )
    return float(np.clip(score, -1, 1))


def apply_trend_score_filter(
    w: pd.Series,
    benchmark_price: pd.Series,
    as_of: pd.Timestamp,
    cfg: V7_5Config,
) -> pd.Series:
    """[Stage 7 v5] 应用连续 trend score 调整权重.

    仓位调整逻辑 (线性插值):
        if score < bear_threshold: equity_scale = bear_equity_alloc
        elif score > bull_threshold: equity_scale = bull_equity_alloc
        else: 线性插值 [bear_threshold, bear_alloc] → [bull_threshold, bull_alloc]

    equity 减仓释放的权重按比例分配给债券 (flight to safety).
    equity 加仓超过 1.0 的部分 (杠杆) 需要 max_weight 约束 (FRP 已经处理).
    """
    if not cfg.tf_score_enabled:
        return w

    score = compute_trend_score(benchmark_price, as_of, cfg)

    # 确定权益缩放因子
    if score < cfg.tf_score_bear_threshold:
        equity_scale = cfg.tf_score_bear_equity_alloc
    elif score > cfg.tf_score_bull_threshold:
        equity_scale = cfg.tf_score_bull_equity_alloc
    else:
        # 线性插值
        t = (score - cfg.tf_score_bear_threshold) / (
            cfg.tf_score_bull_threshold - cfg.tf_score_bear_threshold
        )
        equity_scale = (
            cfg.tf_score_bear_equity_alloc
            + t * (cfg.tf_score_bull_equity_alloc - cfg.tf_score_bear_equity_alloc)
        )

    # 支持 expanded pool (V7_5Config/V7_4Config) 和 index pool (V7_3Config)
    if isinstance(cfg, V7_4Config):
        eq_cols = list(cfg.equity_cols)
        bd_cols = list(cfg.bond_cols)
    else:
        eq_cols = list(cfg.equity_indices)
        bd_cols = BOND_INDICES

    equity_mask = w.index.isin(eq_cols)
    bond_mask = w.index.isin(bd_cols)

    w_new = w.copy()

    if equity_scale <= 1.0:
        # 减仓: 释放的权益权重 → 债券按比例
        freed = float((w[equity_mask] * (1.0 - equity_scale)).sum())
        w_new[equity_mask] = w_new[equity_mask] * equity_scale
        bond_sum = float(w_new[bond_mask].sum())
        if bond_sum > 0:
            w_new[bond_mask] = w_new[bond_mask] + freed * (w_new[bond_mask] / bond_sum)
    else:
        # 加仓: 从债券/商品调权重到权益 (按比例)
        added = float((w[equity_mask] * (equity_scale - 1.0)).sum())
        w_new[equity_mask] = w_new[equity_mask] * equity_scale
        # 释放其他资产 (优先债券)
        non_eq_mask = ~equity_mask
        non_eq_sum = float(w_new[non_eq_mask].sum())
        if non_eq_sum > 0:
            w_new[non_eq_mask] = w_new[non_eq_mask] - added * (w_new[non_eq_mask] / non_eq_sum)

    return w_new


# ============================================================
# v7.3.2: β 预筛选 + 分散度约束 (借鉴 CICC)
# ============================================================
_HK_PREFIXES = ("510900", "159920", "513010", "513050", "159740")
_OVERSEAS_PREFIXES = ("513100", "513300", "513500", "513520", "513880", "159941")


def _classify_asset(code: str, cfg: V7_4Config) -> str:
    """将资产代码分类为: a_share / hk / commodity / overseas / bond."""
    if code in cfg.bond_cols:
        return "bond"
    if code in cfg.commodity_cols:
        return "commodity"
    if any(code.startswith(p) for p in _HK_PREFIXES):
        return "hk"
    if any(code.startswith(p) for p in _OVERSEAS_PREFIXES):
        return "overseas"
    return "a_share"


def _filter_beta_with_diversification(
    β: pd.DataFrame,
    cfg: V7_4Config,
) -> pd.DataFrame:
    """按 |β|_1 排序 + 分散度约束, 保留 ≤ n_assets 个资产.

    流程:
        1. 计算每个资产的 |β|_1 (对所有因子暴露的绝对值之和)
        2. 按类别分组, 组内按 |β|_1 降序
        3. 按类别 cap 选取 (a_share≤8, hk≤2, commodity 1-3, overseas 1-3, bond 1-5)
        4. 合并后检查总数 ≤ n_assets
        5. 确保 commodity/overseas/bond 至少各 1 只
    """
    # 1. 计算 |β|_1
    beta_abs = β.abs().sum(axis=1)

    # 2. 按类别分组
    classified: dict[str, list[tuple[str, float]]] = {}
    for code in β.index:
        cat = _classify_asset(code, cfg)
        classified.setdefault(cat, []).append((code, beta_abs[code]))

    # 3. 组内按 |β|_1 降序
    for cat in classified:
        classified[cat].sort(key=lambda x: x[1], reverse=True)

    # 4. 按 cap 选取
    selected: list[str] = []
    cat_caps = {
        "a_share": cfg.div_a_share_max,
        "hk": cfg.div_hk_max,
        "commodity": cfg.div_commodity_max,
        "overseas": cfg.div_overseas_max,
        "bond": cfg.div_bond_max,
    }
    cat_mins = {
        "commodity": cfg.div_commodity_min,
        "overseas": cfg.div_overseas_min,
        "bond": cfg.div_bond_min,
    }

    for cat, cap in cat_caps.items():
        items = classified.get(cat, [])
        for code, _ in items[:cap]:
            selected.append(code)

    # 5. 确保 min 约束
    for cat, min_n in cat_mins.items():
        current = sum(1 for c in selected if _classify_asset(c, cfg) == cat)
        if current < min_n:
            items = classified.get(cat, [])
            for code, _ in items:
                if code not in selected:
                    selected.append(code)
                    current += 1
                    if current >= min_n:
                        break

    # 6. 如果总数超过 n_assets, 从 |β|_1 最小的开始裁剪
    if len(selected) > cfg.n_assets:
        selected_beta = [(c, beta_abs[c]) for c in selected]
        selected_beta.sort(key=lambda x: x[1])
        selected = [c for c, _ in selected_beta[-cfg.n_assets:]]

    return β.loc[selected]


class V7_3SubStrategy:
    """v7.3 v2 完整还原 — 季度调仓 (与 source 一致)."""

    def __init__(self, cfg: V7_3Config) -> None:
        self.cfg = cfg
        self.lasso = BootstrapLassoMapping(
            times=cfg.bootstrap_times,
            resample_min_weeks=cfg.bootstrap_resample_min,
            resample_max_weeks=cfg.bootstrap_resample_max,
            random_state=cfg.bootstrap_random_state,
            cache_alpha=cfg.bootstrap_cache_alpha,
        )
        self.rp = FactorRiskParityOptimizer(
            max_weight=cfg.max_weight,
            sum_lower=cfg.sum_lower,
            sum_upper=cfg.sum_upper,
        )

    def select(
        self,
        sample: pd.DataFrame,
        end_dt: pd.Timestamp,
    ) -> Mapping[str, float] | None:
        """计算 end_dt 时刻的 13 INDICES 权重 (faithful to source cell 102)."""
        cfg = self.cfg

        # 截至 end_dt 的所有数据
        s_so_far = sample.loc[:end_dt].dropna(how="all")
        if len(s_so_far) < 52 * 2 - 26:  # 至少 ~1.5 年数据
            return None

        # Quarter 滚动窗口 (source cell 102)
        end_loc = s_so_far.index.searchsorted(end_dt, side="left")
        if end_loc > len(s_so_far):
            return None
        if end_loc == len(s_so_far):
            end_loc -= 1

        # 取 end_dt 之前的 8 quarter
        quarter_last = pd.DataFrame(
            index=s_so_far.index[: end_loc + 1]
        ).resample(cfg.rebalance_freq).last().index

        if len(quarter_last) < cfg.quarter_window + 1:
            return None

        start_dt = quarter_last[-(cfg.quarter_window + 1)]
        rolling = s_so_far.loc[start_dt:end_dt].dropna(how="all")
        if len(rolling) < 52 * 2 - 26:
            return None

        # [Stage 7 v5 Step 3] 时变 LASSO: 滚动窗口 vs expanding
        # 默认 (None) 用 expanding, 与原版 bit-exact
        # 设置 lasso_rolling_window (e.g. 156=3 年周) 时, 取最近 N 周
        rolling_window = getattr(cfg, "lasso_rolling_window", None)
        if rolling_window is not None and len(rolling) > rolling_window:
            rolling = rolling.iloc[-rolling_window:]

        # Symmetry 应用到 rolling 窗口 (source cell 102)
        sym_factors = symmetry_full_window(rolling, cfg.factor_cols)
        if sym_factors is None or len(sym_factors) < 26:
            return None

        # Concat 索引 + symmetried factor
        symmetried_sample = pd.concat(
            [rolling[list(cfg.index_pool)], sym_factors],
            axis=1,
        ).dropna(how="all")
        if len(symmetried_sample) < cfg.bootstrap_resample_min:
            return None

        # Bootstrap-Lasso
        β = self.lasso.estimate_exposure(
            asset_returns=symmetried_sample[list(cfg.index_pool)],
            factor_returns=symmetried_sample[list(cfg.factor_cols)],
            as_of_idx=len(symmetried_sample) - 1,
        )

        # 如果 β 全零, 等权兜底
        if np.abs(β.values).sum() < 1e-8:
            n = len(cfg.index_pool)
            return {col: 1.0 / n for col in cfg.index_pool}

        # v7.3.2: β 预筛选 + 分散度约束 (expanded pool)
        if isinstance(cfg, V7_4Config) and cfg.n_assets < len(β):
            β = _filter_beta_with_diversification(β, cfg)

        # 因子协方差
        factor_cov = symmetried_sample[list(cfg.factor_cols)].cov()

        # FactorRiskParity (source cell 104 factor_expo.T[factor_cols].fillna(0))
        w = self.rp.optimize(β, factor_cov)
        return {col: float(w.get(col, 0.0)) for col in cfg.index_pool}


def run_v7_3_backtest(
    asset_prices: pd.DataFrame,
    factor_nav: pd.DataFrame,
    cfg: V7_3Config | None = None,
    benchmark_price: pd.Series | None = None,
    return_weights: bool = False,
):
    """v7.3 v2 端到端回测.

    Args:
        asset_prices: N 个资产日价格/NAV (DataFrame, 日频)
        factor_nav: 8 宏观因子周频净值 (DataFrame, 周频)
        cfg: V7_3Config 配置 (含 trend_filter_* 字段)
        benchmark_price: benchmark 日价格 (用于 TF).
        return_weights: 是否同时返回调仓权重 DataFrame.

    Returns:
        if return_weights=False: pd.Series 索引=业务日, 值=NAV (起点=1).
        if return_weights=True:  (nav: pd.Series, weights: pd.DataFrame)
    """
    if cfg is None:
        cfg = V7_3Config()

    if cfg.trend_filter_enabled and benchmark_price is None:
        from .data_loader import load_benchmark_price
        benchmark_price = load_benchmark_price(cfg.trend_filter_benchmark)

    # 信号: 周频 simple return (从价格计算)
    asset_weekly_ret = asset_prices[list(cfg.index_pool)].resample("W").last().pct_change()
    factor_weekly_ret = factor_nav[list(cfg.factor_cols)].pct_change()
    how = "all" if isinstance(cfg, V7_4Config) else "any"
    sample = pd.concat([asset_weekly_ret, factor_weekly_ret], axis=1).dropna(how=how)

    # Quarter 边界
    quarter_idx = pd.DataFrame(index=sample.index).resample(cfg.rebalance_freq).last().index
    quarter_idx = quarter_idx[quarter_idx <= sample.index.max()]

    if len(quarter_idx) <= cfg.quarter_window:
        raise ValueError(
            f"Insufficient data: need > {cfg.quarter_window} quarters, got {len(quarter_idx)}"
        )

    # [Stage 7 v5 硬止损] 预先构建 stop loss 需要的辅助函数
    def _check_stop_loss_and_override(w_series: pd.Series) -> pd.Series | None:
        """若当前 NAV 回撤超阈值, 返回 100% 债券权重; 否则返回 None 表示不修改."""
        if not cfg.stop_loss_enabled or not _stop_loss_initialized:
            return None
        dd = nav_so_far / peak_nav - 1
        if dd < cfg.stop_loss_threshold:
            if isinstance(cfg, V7_4Config):
                bd_cols = list(cfg.bond_cols)
            else:
                bd_cols = BOND_INDICES
            w_stop = pd.Series(0.0, index=w_series.index)
            bond_sum = float(w_series[bd_cols].sum())
            if bond_sum > 0:
                w_stop[bd_cols] = w_series[bd_cols] * (cfg.stop_loss_bond_alloc / bond_sum)
            else:
                w_stop[bd_cols] = cfg.stop_loss_bond_alloc / len(bd_cols)
            return w_stop
        return None

    # 回测: 日频 simple return (从价格计算)
    sub = V7_3SubStrategy(cfg)
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    rebal_dates = list(quarter_idx[cfg.quarter_window:])

    nav_so_far: float = 1.0
    peak_nav: float = 1.0
    _stop_loss_initialized: bool = False
    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000.0
    all_ret: list[pd.Series] = []

    # 日频 simple return
    daily_returns = asset_prices[list(cfg.index_pool)].pct_change()

    for i, curr_date in enumerate(rebal_dates):
        w = sub.select(sample, curr_date)
        if w is None:
            continue
        w_series = pd.Series(w)
        if cfg.trend_filter_enabled and benchmark_price is not None:
            w_series = apply_trend_filter(w_series, benchmark_price, curr_date, cfg)
        elif getattr(cfg, "tf_score_enabled", False) and benchmark_price is not None:
            w_series = apply_trend_score_filter(w_series, benchmark_price, curr_date, cfg)

        w_override = _check_stop_loss_and_override(w_series)
        if w_override is not None:
            w_series = w_override
        weights_history[curr_date] = w_series
        _stop_loss_initialized = True

        # 计算当期收益 [curr_date, next_date) 期间
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else asset_prices.index[-1] + pd.Timedelta(days=1)
        mask = (daily_returns.index > curr_date) & (daily_returns.index < next_date)
        if not mask.any():
            continue

        idx_ret_window = daily_returns.loc[mask]
        ret_data = idx_ret_window.fillna(0).values @ w_series.reindex(cfg.index_pool).fillna(0).values
        ret_series = pd.Series(ret_data, index=idx_ret_window.index)

        # 调仓日成本: 首次 (i=0) cost=0; 否则 cost = |w[curr]-w[prev]|/2
        if i == 0:
            ret_series.iloc[0] -= 0
        else:
            prev_date = rebal_dates[i - 1]
            turnover = np.abs(
                w_series.reindex(cfg.index_pool).fillna(0).values
                - weights_history[prev_date].reindex(cfg.index_pool).fillna(0).values
            ).sum() / 2.0
            ret_series.iloc[0] -= turnover * cost_rate

        all_ret.append(ret_series)
        nav_after = (1 + ret_series).cumprod() * nav_so_far
        peak_nav = max(peak_nav, float(nav_after.max()))
        nav_so_far = float(nav_after.iloc[-1])

    if not all_ret:
        raise ValueError("No valid weights generated")

    all_ret_series = pd.concat(all_ret)
    nav = (1 + all_ret_series).cumprod()
    nav = nav / nav.iloc[0]

    if return_weights:
        weights_df = pd.DataFrame(
            {d: w.reindex(cfg.index_pool) for d, w in weights_history.items()},
        ).T
        weights_df.index.name = "rebalance_date"
        return nav, weights_df

    return nav


__all__ = [
    "V7_3Config",
    "V7_4Config",
    "V7_5Config",
    "V7_3SubStrategy",
    "run_v7_3_backtest",
    "apply_trend_filter",
    "apply_trend_score_filter",
    "compute_trend_score",
    "symmetry_full_window",
]
