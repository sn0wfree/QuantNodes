# coding=utf-8
"""组合管理: 4 条规则 + 权重 + 止损补位.

对齐 CICC 2026-07-03 报告 (图表 2 伪代码 + 主文规则 1-4):

    1. 去重 + 剔高相关 (同指数去重 + 入选项相关 > 0.9 则跳)
    2. 强制分散 (A 股宽基+行业 ≤ a_share_total, HK ≤ 1, 必含商品+海外)
    3. 逆波动加权 (权重 ∝ 1/σ, 21 日窗口)
    4. 止损 + 补位 (跌破 55 日均线 + 排名跌出后 30% 分位)

CICC 伪代码关键约束:
    - 池先预去重: ``keep_most_liquid_by_tracking_index`` (best liquidity per index)
    - 主循环检查顺序: caps 先, corr 后
    - 末尾: ``fill_by_rank_if_needed`` 补回到 TARGET_N
    - 逆波动窗口: lookback=21 (非 60)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from .momentum import (
    below_ma,
    compute_momentum_score,
    distance_to_52w_high,
    fused_signal,
    pairwise_corr,
    rank_pctl,
    realized_vol,
    yang_zhang_vol,
)
from ..common.universe import ETFPool
from ..common.backtest_config import CostConfig as CostModel, VolTargeting, TrendFilter


# ----------------------------------------------------------------------------
# 分散规则
# ----------------------------------------------------------------------------
@dataclass
class DiversificationCaps:
    """强制分散上限.

    CICC 实验: 10 个标的池里 A 股宽基+行业 ≤ 3 (a_share_total), HK ≤ 1, 必含商品与海外.
    """
    a_share_broad: int = 2              # A 股宽基 (per-category cap)
    a_share_sector: int = 2             # A 股行业/策略 (per-category cap)
    a_share_total: int = 3              # A 股宽基+行业 合并总 cap (CICC: ≤ 3)
    hk: int = 1
    require_commodity: bool = True
    require_overseas: bool = True
    # 旧字段, 保留向后兼容但实际不使用 (见 a_share_total)
    a_share: int = 3

    def cap_for(self, category_name: str) -> int:
        """Per-category cap. 99 = 无限制 (用于 commodity/overseas)."""
        return {
            "a_broad": self.a_share_broad,
            "a_sector": self.a_share_sector,
            "hk": self.hk,
        }.get(category_name, 99)


# ----------------------------------------------------------------------------
# 策略参数 (TrendFilter, VolTargeting, CostModel 统一来自 common/backtest_config)
# ----------------------------------------------------------------------------

@dataclass
class ConcentrationCaps:
    """集中度约束 (Stage 10): 限制单 ETF / Top N / 类别集中度.

    启用时, 加权完成后对权重进行缩放, 满足:
        - 单 ETF 权重 <= single_etf_max
        - Top N ETF 合计 <= top_n_total_max
        - 单类别合计 <= category_max
    """
    enabled: bool = False
    single_etf_max: float = 0.15     # 单 ETF 权重上限 (默认 15%)
    top_n_total_max: float = 0.45   # Top 3 ETF 合计上限 (默认 45%)
    top_n_count: int = 3
    category_max: float = 0.40      # 单类别合计上限 (默认 40%)


def calculate_turnover_cost(turnover: float, cost: CostModel) -> float:
    """计算单次换手成本.

    turnover: 单边换手率 (如 0.5 表示 50% 换手)
    Returns: 成本率 (如 0.001 表示 0.1% 成本)
    """
    if not cost.enabled:
        return 0.0
    # 单边成本 = 佣金 + 滑点 + 冲击成本 (简化)
    cost_rate = (cost.commission_bp + cost.slippage_bp * cost.impact_factor) / 10000
    return turnover * cost_rate


# Stage 9-D: Regime detector 占位 (实际类在 regime_detector.py)
# 为避免循环 import, 这里只做类型提示
if TYPE_CHECKING:
    from .regime_detector import RegimeDetector


@dataclass
class RotationConfig:
    """动量轮动策略的所有可调参数."""
    lookback: int = 144                 # 动量回看 (CICC 144)
    top_n: int = 10
    corr_threshold: float = 0.9
    corr_window: int = 60

    # 规则 4
    ma_window: int = 55
    rank_cutoff: float = 0.30           # 排名跌出后 30% 分位 → 剔出

    # 规则 2
    diversification: DiversificationCaps = field(default_factory=DiversificationCaps)

    # 规则 3
    weight_method: str = "inv_vol"      # "inv_vol" | "equal"
    vol_window: int = 21                # CICC 伪代码: lookback=21 (非 60)
    weight_floor: float = 1e-4

    # 信号类型 (Stage 9-A)
    signal_type: str = "momentum"       # "momentum" | "dist_52w" | "fused"
    signal_fused_weight: float = 0.4    # 52周新高在 fused 中的权重
    signal_52w_window: int = 252        # 52 周高点窗口

    # 动量打分方式 (Stage 12A)
    momentum_type: str = "price"        # "price" | "slope_r2" | "hybrid"
    momentum_fused_weight: float = 0.5  # hybrid 中 slope_r2 权重
    momentum_scale: float = 10000.0     # slope_r2 缩放系数

    # 趋势过滤器 (Stage 9-B)
    trend_filter: TrendFilter = field(default_factory=TrendFilter)

    # 波动率目标 (Stage 9-C)
    vol_targeting: VolTargeting = field(default_factory=VolTargeting)

    # 集中度约束 (Stage 10)
    concentration: ConcentrationCaps = field(default_factory=ConcentrationCaps)

    # 交易成本 (Stage 13)
    cost_model: CostModel = field(default_factory=CostModel)

    # Regime 检测 (Stage 9-D): 延迟 import 避免循环依赖
    regime_detector: "RegimeDetector | None" = None

    # 通用
    min_history: int = 144


# ----------------------------------------------------------------------------
# 状态
# ----------------------------------------------------------------------------
@dataclass
class PortfolioState:
    """一次调仓的快照 (便于日志/回测)."""
    date: pd.Timestamp
    ranked: list[str]                          # 动量降序的全榜
    chosen: list[str]                          # 入选的 (按入选顺序)
    weights: dict[str, float]                  # 入选 → 权重
    skipped_dedup: list[str] = field(default_factory=list)
    skipped_corr: list[str] = field(default_factory=list)
    skipped_div: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)        # 本轮被止损
    replaced: dict[str, str] = field(default_factory=dict)  # stop_code → new_code


# ----------------------------------------------------------------------------
# 权重
# ----------------------------------------------------------------------------
def inverse_vol_weights(
    nav_df: pd.DataFrame,
    codes: Sequence[str],
    as_of: pd.Timestamp,
    vol_window: int = 21,
    floor: float = 1e-4,
    ohlcv_df: pd.DataFrame | None = None,
    vol_method: str = "yang_zhang",
) -> dict[str, float]:
    """权重 ∝ 1/σ_i, σ 为年化已实现波动率 (CICC 伪代码窗口=21).

    Args:
        nav_df: close 价格面板 (fallback 用)
        codes: 标的代码列表
        as_of: 截止日期
        vol_window: 波动率窗口 (默认 21)
        floor: 最小权重阈值
        ohlcv_df: OHLCV 面板 (可选, 优先使用 yang_zhang_vol)
        vol_method: 波动率方法 ("yang_zhang" | "close_only")
    """
    if not codes:
        return {}

    # 优先使用 OHLC 数据计算 YZ 波动率
    if ohlcv_df is not None and vol_method == "yang_zhang":
        vols = yang_zhang_vol(ohlcv_df, as_of=as_of, window=vol_window).reindex(list(codes))
    else:
        vols = realized_vol(nav_df, as_of=as_of, window=vol_window).reindex(list(codes))

    vols = vols.fillna(vols.median() if not vols.empty and vols.median() > 0 else 1.0)
    inv = 1.0 / vols
    inv[inv < floor] = 0.0
    total = inv.sum()
    if total <= 0:
        return equal_weights(codes)
    return (inv / total).to_dict()


def equal_weights(codes: Sequence[str]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


# ----------------------------------------------------------------------------
# 预去重: best liquidity per index (CICC keep_most_liquid_by_tracking_index)
# ----------------------------------------------------------------------------
def _compute_best_per_index(
    pool: ETFPool,
    blacklist: set[str],
) -> dict[str, str]:
    """对每个 index_code, 返回最佳流动性且未在 blacklist 的 code."""
    result: dict[str, str] = {}
    for m in pool.members:
        if m.code in blacklist:
            continue
        cur = result.get(m.index_code)
        if cur is None or m.liquidity_rank < pool.liquidity_rank_of(cur):
            result[m.index_code] = m.code
    return result


# ----------------------------------------------------------------------------
# 核心: select_and_weight (调仓日的选择+权重)
# ----------------------------------------------------------------------------
def _count_categories(chosen: list[str], pool: ETFPool) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for c in chosen:
        cat = pool.category_of(c).value
        cnt[cat] = cnt.get(cat, 0) + 1
    return cnt


def select_and_weight(
    nav_df: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    as_of: pd.Timestamp,
    blacklist: Sequence[str] = (),
    base_categories: dict[str, int] | None = None,
) -> PortfolioState:
    """单次调仓 (严格按 CICC 伪代码流程).

    流程:
        1. 预去重: best liquidity per index, 排除 blacklist + NaN momentum
        2. 主循环: caps (含 a_share_total) → corr → append
        3. require_* 注入: 替换一个非商品/海外
        4. fill_by_rank_if_needed: 补回到 top_n
        5. 加权: 1/σ (21 日)

    Parameters
    ----------
    blacklist : 永远不入选的代码 (规则 4 止损后用)

    Stage 9-A: signal_type 支持
        "momentum":  纯动量 (默认, CICC 原版)
        "dist_52w":  距离52周新高 (CICC 报告图表 4 备选)
        "fused":     (1-w) × 动量 + w × 距离52周新高
    """
    # 计算信号 (Stage 9-A: 支持 3 种信号)
    # Stage 12A: momentum_type 优先 (slope_r2 / hybrid)
    if cfg.momentum_type in ("slope_r2", "hybrid"):
        # 斜率信号直接用 compute_momentum_score
        score = compute_momentum_score(
            nav_df, cfg.lookback, as_of,
            momentum_type=cfg.momentum_type,
            fused_weight=cfg.momentum_fused_weight,
        )
        pctl = score.rank(method="average", pct=True)
    elif cfg.signal_type == "dist_52w":
        score = distance_to_52w_high(
            nav_df, as_of, window=cfg.signal_52w_window,
        )
        pctl = score.rank(method="average", pct=True)
    elif cfg.signal_type == "fused":
        score = fused_signal(
            nav_df, cfg.lookback, as_of,
            fused_weight=cfg.signal_fused_weight,
            window_52w=cfg.signal_52w_window,
        )
        pctl = score.rank(method="average", pct=True)
    else:  # "momentum" (默认)
        pctl = rank_pctl(nav_df, cfg.lookback, as_of)

    ranked = pctl.sort_values(ascending=False).index.tolist()
    blacklist_set = set(blacklist)

    state = PortfolioState(date=as_of, ranked=ranked, chosen=[], weights={})

    # ── 1. 预去重: best liquidity per index (CICC keep_most_liquid) ──
    best_per_index = _compute_best_per_index(pool, blacklist_set)
    deduped_ranked: list[str] = []
    for code in ranked:
        if code not in pool.codes:
            continue
        if code in blacklist_set:
            state.skipped_dedup.append(code)
            continue
        # pctl.get 可能因索引重复返回 Series, 用 .iloc 安全获取
        try:
            pctl_val = pctl.loc[code]
        except KeyError:
            state.skipped_dedup.append(code)
            continue
        if isinstance(pctl_val, pd.Series):
            pctl_val = pctl_val.iloc[0]
        if pd.isna(pctl_val):
            state.skipped_dedup.append(code)
            continue
        idx = pool.index_of(code)
        if best_per_index.get(idx) != code:
            state.skipped_dedup.append(code)
            continue
        deduped_ranked.append(code)

    # ── 2. 主循环: caps FIRST, corr SECOND (per CICC pseudocode) ──
    chosen: list[str] = []
    chosen_cat_count: dict[str, int] = dict(base_categories or {})
    for code in deduped_ranked:
        if len(chosen) >= cfg.top_n:
            break

        cat_name = pool.category_of(code).value

        # 规则 2a: A 股宽基+行业 总 cap (CICC: ≤ 3)
        if cat_name in ("a_broad", "a_sector"):
            current_a_share = (chosen_cat_count.get("a_broad", 0) +
                                chosen_cat_count.get("a_sector", 0))
            if current_a_share >= cfg.diversification.a_share_total:
                state.skipped_div.append(code)
                continue

        # 规则 2b: per-category cap
        cap = cfg.diversification.cap_for(cat_name)
        if chosen_cat_count.get(cat_name, 0) >= cap:
            state.skipped_div.append(code)
            continue

        # 规则 1b: 与已选高相关则跳
        if chosen:
            corr = pairwise_corr(nav_df, [code] + chosen, as_of, cfg.corr_window)
            cc = corr.loc[code, chosen]
            # 防御: cc 可能是 DataFrame (列重复) 或 Series
            if isinstance(cc, pd.DataFrame):
                cc = cc.iloc[:, 0]
            if isinstance(cc, pd.Series):
                if (cc > cfg.corr_threshold).any():
                    state.skipped_corr.append(code)
                    continue
            else:
                # cc 是 scalar
                if cc > cfg.corr_threshold:
                    state.skipped_corr.append(code)
                    continue

        chosen.append(code)
        chosen_cat_count[cat_name] = chosen_cat_count.get(cat_name, 0) + 1

    state.chosen = chosen

    # ── 3. require_* 注入: 替换一个非商品/海外 ──
    _maybe_inject_required(state, ranked, pctl, pool, cfg)

    # ── 4. fill_by_rank_if_needed: 补回到 top_n (同样检查 caps) ──
    if len(state.chosen) < cfg.top_n:
        # 重建计数值 (包含 base_categories + 已选)
        base = dict(base_categories or {})
        chosen_cat_count = _count_categories(state.chosen, pool)
        for k, v in base.items():
            chosen_cat_count[k] = chosen_cat_count.get(k, 0) + v
        for code in deduped_ranked:
            if code in state.chosen:
                continue
            cat_name = pool.category_of(code).value
            # 检查 per-category cap
            cap = cfg.diversification.cap_for(cat_name)
            if chosen_cat_count.get(cat_name, 0) >= cap:
                state.skipped_div.append(code)
                continue
            # 检查 A 股合计 cap
            if cat_name in ("a_broad", "a_sector"):
                current = (chosen_cat_count.get("a_broad", 0) +
                           chosen_cat_count.get("a_sector", 0))
                if current >= cfg.diversification.a_share_total:
                    state.skipped_div.append(code)
                    continue
            state.chosen.append(code)
            chosen_cat_count[cat_name] = chosen_cat_count.get(cat_name, 0) + 1
            if len(state.chosen) >= cfg.top_n:
                break

    # ── 5. 加权 ──
    if cfg.weight_method == "inv_vol":
        state.weights = inverse_vol_weights(
            nav_df, state.chosen, as_of,
            vol_window=cfg.vol_window, floor=cfg.weight_floor,
        )
    else:
        state.weights = equal_weights(state.chosen)

    # Stage 10: 集中度约束 (缩放单 ETF / Top N / 类别集中度)
    if cfg.concentration.enabled:
        state.weights = _apply_concentration_caps(
            state.weights, cfg.concentration, pool,
        )

    # Stage 9-B: 趋势过滤
    apply_trend_filter(nav_df, cfg, as_of, state)

    return state


def _maybe_inject_required(
    state: PortfolioState,
    ranked: list[str],
    pctl: pd.Series,
    pool: ETFPool,
    cfg: RotationConfig,
) -> None:
    """如果 chosen 缺少商品/海外, 从 ranked 找一个并替换一个非商品/海外.

    来自 CICC 主文"必须有商品和海外配置" (图表 1 说明).
    替换而非 append, 是为了保持 top_n 不变.
    """
    div = cfg.diversification
    chosen = state.chosen

    for category in ("commodity", "overseas"):
        required = (div.require_commodity and category == "commodity") or \
                    (div.require_overseas and category == "overseas")
        if not required:
            continue
        if any(pool.category_of(c).value == category for c in chosen):
            continue
        # 找一个未入选的合格候选
        for code in ranked:
            if (code in pool.codes
                and pool.category_of(code).value == category
                and code not in chosen
                and code not in state.skipped_dedup):
                # 单独检查 pctl 值 (避免重复索引时返回 Series)
                try:
                    pctl_val = pctl.loc[code]
                except KeyError:
                    continue
                if isinstance(pctl_val, pd.Series):
                    pctl_val = pctl_val.iloc[0]
                if pd.isna(pctl_val):
                    continue
                replaced = False
                # 替换最后一个非商品/海外
                for i in range(len(chosen) - 1, -1, -1):
                    cn = chosen[i]
                    if pool.category_of(cn).value not in ("commodity", "overseas"):
                        chosen[i] = code
                        state.skipped_div.append(cn)
                        replaced = True
                        break
                if not replaced and len(chosen) < cfg.top_n:
                    chosen.append(code)
                break  # 已找到一个, 不再继续


# ----------------------------------------------------------------------------
# 趋势过滤器 (Stage 9-B)
# ----------------------------------------------------------------------------
def check_trend_filter(
    nav_df: pd.DataFrame,
    benchmark_code: str,
    ma_window: int,
    as_of: pd.Timestamp,
) -> bool:
    """判断当前是否处于多头趋势.

    返回 True 表示多头 (价格 >= ma_window 日均线), False 表示空头.
    数据不足时默认多头.
    """
    if benchmark_code not in nav_df.columns:
        return True
    benchmark = nav_df[benchmark_code].loc[:as_of]
    if len(benchmark) < ma_window:
        return True
    ma = benchmark.iloc[-ma_window:].mean()
    return bool(benchmark.iloc[-1] >= ma)


def apply_trend_filter(
    nav_df: pd.DataFrame,
    cfg: RotationConfig,
    as_of: pd.Timestamp,
    state: PortfolioState,
) -> PortfolioState:
    """对 PortfolioState 应用趋势过滤 (Stage 9-B).

    熊市时 (基准跌破 ma):
        - 缩放现有权重到 exposure_bear
        - 剩余仓位配到债券 ETF (bond_code)

    返回新的 state.weights.
    """
    if not cfg.trend_filter.enabled:
        return state
    tf = cfg.trend_filter
    is_bull = check_trend_filter(nav_df, tf.benchmark_code, tf.ma_window, as_of)
    if is_bull:
        return state
    scale = tf.exposure_bear
    new_weights = {k: v * scale for k, v in state.weights.items()}
    bond_weight = 1.0 - scale
    if tf.bond_code in nav_df.columns:
        new_weights[tf.bond_code] = new_weights.get(tf.bond_code, 0.0) + bond_weight
    state.weights = new_weights
    return state


# ----------------------------------------------------------------------------
# 波动率目标 (Stage 9-C)
# ----------------------------------------------------------------------------
def vol_targeting_scale(
    nav: pd.Series,
    target_vol: float,
    lookback: int,
    min_scale: float,
    max_scale: float,
    ohlcv_df: pd.DataFrame | None = None,
    code: str | None = None,
    vol_method: str = "yang_zhang",
) -> float:
    """计算当前应缩放系数.

    scale = clip(target_vol / realized_vol, min_scale, max_scale)
    realized_vol 为 lookback 日年化波动率 (× √252).

    Args:
        nav: close 价格序列 (单个标的)
        target_vol: 目标波动率
        lookback: 回看窗口
        min_scale: 最小缩放系数
        max_scale: 最大缩放系数
        ohlcv_df: OHLCV 面板 (可选, 优先使用 yang_zhang_vol)
        code: 标的代码 (使用 ohlcv_df 时必需)
        vol_method: 波动率方法 ("yang_zhang" | "close_only")
    """
    # 优先使用 OHLC 数据计算 YZ 波动率
    if ohlcv_df is not None and code is not None and vol_method == "yang_zhang":
        try:
            sub = ohlcv_df[code]
            if len(sub) >= lookback:
                vols = yang_zhang_vol(ohlcv_df, as_of=nav.index[-1], window=lookback)
                if code in vols.index:
                    realized_vol = vols[code]
                    if not np.isnan(realized_vol) and realized_vol > 0:
                        scale = target_vol / realized_vol
                        return float(np.clip(scale, min_scale, max_scale))
        except Exception:
            pass

    # Fallback 到 close-only (原逻辑)
    rets = nav.pct_change().dropna()
    if len(rets) < lookback:
        return 1.0
    realized_vol = rets.iloc[-lookback:].std() * np.sqrt(252)
    if realized_vol <= 0:
        return 1.0
    scale = target_vol / realized_vol
    return float(np.clip(scale, min_scale, max_scale))


def apply_vol_targeting(
    cfg: RotationConfig,
    nav: pd.Series,
    as_of: pd.Timestamp,
    state: PortfolioState,
) -> PortfolioState:
    """对 PortfolioState 应用波动率目标 (Stage 9-C)."""
    if not cfg.vol_targeting.enabled:
        return state
    vt = cfg.vol_targeting
    scale = vol_targeting_scale(
        nav.loc[:as_of],
        vt.target_vol,
        vt.lookback,
        vt.min_scale,
        vt.max_scale,
    )
    state.weights = {k: v * scale for k, v in state.weights.items()}
    # 剩余仓位补现金 (这里不实现, 用缩放代替)
    return state


# ----------------------------------------------------------------------------
# 集中度约束 (Stage 10)
# ----------------------------------------------------------------------------
def _apply_concentration_caps(
    weights: dict[str, float],
    caps: ConcentrationCaps,
    pool: ETFPool | None = None,
) -> dict[str, float]:
    """缩放权重以满足集中度约束 (Stage 10).

    三步:
        1. 单 ETF 权重 <= single_etf_max
        2. Top N ETF 合计 <= top_n_total_max
        3. 单类别合计 <= category_max (需 pool)

    Returns 新的权重 dict. 若约束导致总权重下降, 差额视为现金.

    注: 此实现不重新分配超出部分 (避免振荡), 而是允许总权重下降.
    """
    if not caps.enabled or not weights:
        return weights
    w = dict(weights)

    # 1. 单 ETF 约束
    for code in list(w.keys()):
        if w[code] > caps.single_etf_max:
            w[code] = caps.single_etf_max

    # 2. Top N 约束: 等比缩放
    sorted_items = sorted(w.items(), key=lambda x: -x[1])
    top_n_items = sorted_items[:caps.top_n_count]
    top_n_codes = {code for code, _ in top_n_items}
    top_n_total = sum(v for _, v in top_n_items)
    if top_n_total > caps.top_n_total_max and top_n_total > 0:
        scale = caps.top_n_total_max / top_n_total
        for code in top_n_codes:
            w[code] *= scale

    # 3. 类别约束
    if pool is not None:
        cat_weights: dict[str, float] = {}
        for code, weight in w.items():
            try:
                cat = pool.category_of(code).value
                cat_weights[cat] = cat_weights.get(cat, 0.0) + weight
            except KeyError:
                continue
        for cat, cat_total in cat_weights.items():
            if cat_total > caps.category_max and cat_total > 0:
                cat_scale = caps.category_max / cat_total
                cat_codes = [c for c in w if pool.category_of(c).value == cat]
                for c in cat_codes:
                    w[c] *= cat_scale

    return w


def apply_concentration_caps(
    cfg: RotationConfig,
    pool: ETFPool,
    state: PortfolioState,
) -> PortfolioState:
    """对 PortfolioState 应用集中度约束 (Stage 10)."""
    if not cfg.concentration.enabled:
        return state
    state.weights = _apply_concentration_caps(
        state.weights, cfg.concentration, pool,
    )
    return state


# ----------------------------------------------------------------------------
# 规则 4: 止损 + 补位
# ----------------------------------------------------------------------------
def apply_stops(
    nav_df: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    prev_weights: Mapping[str, float],
    as_of: pd.Timestamp,
) -> PortfolioState:
    """对已有持仓: 跌破 ma 且 排名跌出后 cutoff 分位 → 剔出, 按相同规则补入.

    返回: PortfolioState, 其中 chosen/weights 为"剔出并补位后"的新组合.
    """
    # 1) 找到要止损的 (同时满足: 跌破 ma + 排名跌出后 cutoff 分位)
    pctl_series = rank_pctl(nav_df, cfg.lookback, as_of)
    to_stop: list[str] = []
    for code, w in prev_weights.items():
        if w <= 0:
            continue
        if code not in pool.codes:
            # 池外代码 (如趋势过滤加入的债券) → 不止损
            continue
        if not below_ma(nav_df, code, cfg.ma_window, as_of):
            continue
        if code not in pctl_series.index:
            continue
        # 防御: pctl_series[code] 可能因索引重复返回 Series
        pctl_val = pctl_series[code]
        if isinstance(pctl_val, pd.Series):
            pctl_val = pctl_val.iloc[0]
        if pctl_val < cfg.rank_cutoff:
            to_stop.append(code)

    # 2) 在原 chosen 中去掉被止损的 (只考虑池内代码)
    prev_chosen = [c for c, w in prev_weights.items()
                   if w > 0 and c in pool.codes and c not in to_stop]
    stopped = list(to_stop)
    replaced: dict[str, str] = {}

    if not stopped:
        state = PortfolioState(
            date=as_of, ranked=pctl_series.sort_values(ascending=False).index.tolist(),
            chosen=prev_chosen, weights=dict(prev_weights),
        )
        # Stage 9-B: 趋势过滤 (无止损时也要应用)
        apply_trend_filter(nav_df, cfg, as_of, state)
        return state

    # 已有持仓的品类计数 (传给 select_and_weight 作为 base, 避免累积超限)
    base_cats: dict[str, int] = {}
    for c in prev_chosen:
        cat = pool.category_of(c).value
        base_cats[cat] = base_cats.get(cat, 0) + 1

    # 3) 重新选 (用与 select_and_weight 同样的规则), 但**排除**已止损的
    state = select_and_weight(nav_df, pool, cfg, as_of, blacklist=stopped,
                              base_categories=base_cats)

    # 4) 候选 = state.chosen 中 (非 prev_chosen) 的部分
    available = [c for c in state.chosen if c not in prev_chosen]

    # 5) 补位 (按动量降序, 因为 state.chosen 已按动量排好)
    for s in stopped:
        if not available:
            break
        new_code = available.pop(0)
        prev_chosen.append(new_code)
        replaced[s] = new_code

    state.chosen = prev_chosen
    if cfg.weight_method == "inv_vol":
        state.weights = inverse_vol_weights(
            nav_df, prev_chosen, as_of, vol_window=cfg.vol_window, floor=cfg.weight_floor
        )
    else:
        state.weights = equal_weights(prev_chosen)
    state.stopped = stopped
    state.replaced = replaced

    # Stage 10: 集中度约束 (在重新加权后再次应用)
    if cfg.concentration.enabled:
        state.weights = _apply_concentration_caps(
            state.weights, cfg.concentration, pool,
        )

    # Stage 9-B: 趋势过滤
    apply_trend_filter(nav_df, cfg, as_of, state)

    return state
