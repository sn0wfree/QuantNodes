# coding=utf-8
"""v4 ETF 池 + 大类/行业分组 (Stage 27 重构: 适配 43 ETF).

v4 重构后 = 大类轮动 + Smart β 代理 + 行业轮动 + 因子择时.

大类分组 (4 类, 43 ETF):
- broad: 6 个宽基 (510300, 510500, 510050, 159915, 588000, 159901)
- sector: 23 个行业 ETF
- overseas: 11 个海外 ETF
- gold: 3 个商品/黄金 ETF

Smart β 代理: 从行业 ETF 中用价值/质量因子筛选 (Stage 27 新增)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


# ============================================================
# 43 ETF 分类 (Stage 27)
# ============================================================

# 宽基 ETF (6 个)
BROAD_CODES: tuple[str, ...] = (
    "510300",   # 沪深 300
    "510500",   # 中证 500
    "510050",   # 上证 50
    "159915",   # 创业板
    "588000",   # 科创 50
    "159901",   # 深证 100
)

# 行业 ETF (23 个)
SECTOR_CODES: tuple[str, ...] = (
    "512760",   # 半导体
    "512480",   # 半导体 (国联安)
    "515030",   # 新能源车
    "515790",   # 光伏
    "512690",   # 酒
    "512170",   # 医疗
    "512010",   # 医药
    "515050",   # 通信
    "159928",   # 消费
    "512880",   # 证券
    "512000",   # 券商
    "512800",   # 银行
    "515220",   # 煤炭
    "512200",   # 房地产
    "512400",   # 有色金属
    "512660",   # 军工
    "512980",   # 传媒
    "515880",   # 通信 ETF
    "159996",   # 家电
    "512120",   # 化工
    "161226",   # 纳指
    "159981",   # 能源化工
    "159766",   # 旅游
)

# 海外 ETF (11 个)
OVERSEAS_CODES: tuple[str, ...] = (
    "510900",   # 恒生
    "159920",   # 恒生 ETF
    "513010",   # 港股科技
    "513050",   # 中概互联
    "159740",   # 恒生科技
    "513100",   # 纳指 ETF
    "513300",   # 纳斯达克
    "513500",   # 标普 500
    "513520",   # 日经 ETF
    "513880",   # 日经 225
    "159941",   # 深 F120
)

# 黄金/商品 ETF (3 个)
GOLD_CODES: tuple[str, ...] = (
    "518880",   # 黄金 ETF
    "518800",   # 黄金 ETF 9999
    "159985",   # 豆粕
)

# 防御型行业 (熊市超配)
DEFENSIVE_SECTOR_CODES: tuple[str, ...] = (
    "512800",   # 银行
    "512170",   # 医疗
    "512010",   # 医药
    "159928",   # 消费
    "159996",   # 家电
    "512120",   # 化工
)

# 进攻型行业 (牛市超配)
GROWTH_SECTOR_CODES: tuple[str, ...] = (
    "512760",   # 半导体
    "512480",   # 半导体 (国联安)
    "515030",   # 新能源车
    "515790",   # 光伏
    "512660",   # 军工
)

# ============================================================
# 大类分组 (新枚举)
# ============================================================

class AssetClass(Enum):
    """大类分类 (4 类)."""
    BROAD = "broad"          # 宽基
    SECTOR = "sector"        # 行业
    OVERSEAS = "overseas"    # 海外
    GOLD = "gold"            # 黄金/商品


# 大类 → ETF 列表
ASSET_CLASS_CODES: dict[AssetClass, tuple[str, ...]] = {
    AssetClass.BROAD: BROAD_CODES,
    AssetClass.SECTOR: SECTOR_CODES,
    AssetClass.OVERSEAS: OVERSEAS_CODES,
    AssetClass.GOLD: GOLD_CODES,
}


# ============================================================
# 兼容旧版: 风格组 + Smart β (保留接口)
# ============================================================

class StyleGroup(Enum):
    """风格组分类 (5 个, 兼容旧版)."""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    GROWTH = "growth"
    TECH = "tech"
    DIVIDEND = "dividend"


class SmartBetaFactor(Enum):
    """Smart β 因子分类 (保留枚举, 实际用代理筛选)."""
    DIV_LOW_VOL = "div_low_vol"
    LOW_VOL = "low_vol"
    QUALITY = "quality"
    VALUE = "value"
    CASHFLOW = "cashflow"
    DIV_100 = "div_100"
    DIV_LOW_VOL_100 = "div_low_vol_100"


# 兼容旧版: 风格组 (只保留宽基中有的)
STYLE_GROUP_CODES: dict[StyleGroup, tuple[str, ...]] = {
    StyleGroup.LARGE_CAP: ("510300",),
    StyleGroup.MID_CAP: ("510500",),
    StyleGroup.GROWTH: ("159915",),
    StyleGroup.TECH: ("588000",),
    StyleGroup.DIVIDEND: ("510880",),  # 注意: 510880 不在 43 ETF 中
}


# 兼容旧版: Smart β (12 ETF 数据集中的代码, 不在 43 ETF 中)
SMART_BETA_CODES: dict[SmartBetaFactor, str] = {
    SmartBetaFactor.DIV_LOW_VOL: "512890",
    SmartBetaFactor.LOW_VOL: "512260",
    SmartBetaFactor.QUALITY: "515900",
    SmartBetaFactor.VALUE: "512040",
    SmartBetaFactor.CASHFLOW: "159786",
    SmartBetaFactor.DIV_100: "515080",
    SmartBetaFactor.DIV_LOW_VOL_100: "515100",
}

SMART_BETA_FACTOR_TYPE: dict[SmartBetaFactor, str] = {
    SmartBetaFactor.DIV_LOW_VOL: "defensive",
    SmartBetaFactor.LOW_VOL: "defensive",
    SmartBetaFactor.QUALITY: "defensive",
    SmartBetaFactor.VALUE: "value",
    SmartBetaFactor.CASHFLOW: "value",
    SmartBetaFactor.DIV_100: "defensive",
    SmartBetaFactor.DIV_LOW_VOL_100: "defensive",
}


@dataclass(frozen=True)
class StyleGroupMeta:
    """风格组元数据 (兼容旧版)."""
    group: StyleGroup
    codes: tuple[str, ...]
    name_cn: str


@dataclass(frozen=True)
class SmartBetaMeta:
    """Smart β 元数据 (兼容旧版)."""
    factor: SmartBetaFactor
    code: str
    name_cn: str
    factor_type: str


# 风格组元数据
STYLE_GROUP_METAS: dict[StyleGroup, StyleGroupMeta] = {
    StyleGroup.LARGE_CAP: StyleGroupMeta(StyleGroup.LARGE_CAP, ("510300",), "大盘 (HS300)"),
    StyleGroup.MID_CAP: StyleGroupMeta(StyleGroup.MID_CAP, ("510500",), "中盘 (CSI500)"),
    StyleGroup.GROWTH: StyleGroupMeta(StyleGroup.GROWTH, ("159915",), "成长 (创业板)"),
    StyleGroup.TECH: StyleGroupMeta(StyleGroup.TECH, ("588000",), "科创 (科创50)"),
    StyleGroup.DIVIDEND: StyleGroupMeta(StyleGroup.DIVIDEND, ("510880",), "红利"),
}

SMART_BETA_METAS: dict[SmartBetaFactor, SmartBetaMeta] = {
    SmartBetaFactor.DIV_LOW_VOL: SmartBetaMeta(SmartBetaFactor.DIV_LOW_VOL, "512890", "红利低波", "defensive"),
    SmartBetaFactor.LOW_VOL: SmartBetaMeta(SmartBetaFactor.LOW_VOL, "512260", "300 低波", "defensive"),
    SmartBetaFactor.QUALITY: SmartBetaMeta(SmartBetaFactor.QUALITY, "515900", "中证质量", "defensive"),
    SmartBetaFactor.VALUE: SmartBetaMeta(SmartBetaFactor.VALUE, "512040", "国泰价值", "value"),
    SmartBetaFactor.CASHFLOW: SmartBetaMeta(SmartBetaFactor.CASHFLOW, "159786", "现金流", "value"),
    SmartBetaFactor.DIV_100: SmartBetaMeta(SmartBetaFactor.DIV_100, "515080", "中信红利", "defensive"),
    SmartBetaFactor.DIV_LOW_VOL_100: SmartBetaMeta(SmartBetaFactor.DIV_LOW_VOL_100, "515100", "红利低波 100", "defensive"),
}


# 兼容旧版: 12 ETF (在 smartbeta 数据集中)
ALL_V4_CODES: tuple[str, ...] = (
    "510300", "510500", "159915", "588000", "510880",
    "512890", "512260", "515900", "512040", "159786", "515080", "515100",
)


# ============================================================
# 43 ETF 大类轮动 (Stage 27 新增)
# ============================================================

def get_all_43_codes() -> tuple[str, ...]:
    """获取所有 43 ETF 代码."""
    return BROAD_CODES + SECTOR_CODES + OVERSEAS_CODES + GOLD_CODES


def classify_43_etf(codes: list[str] | tuple[str, ...]) -> dict[str, AssetClass]:
    """对 43 ETF 进行分类."""
    out: dict[str, AssetClass] = {}
    for code in codes:
        if code in BROAD_CODES:
            out[code] = AssetClass.BROAD
        elif code in SECTOR_CODES:
            out[code] = AssetClass.SECTOR
        elif code in OVERSEAS_CODES:
            out[code] = AssetClass.OVERSEAS
        elif code in GOLD_CODES:
            out[code] = AssetClass.GOLD
    return out


# ============================================================
# Smart β 代理筛选 (Stage 27 新增)
# ============================================================

def _zscore(s: pd.Series) -> pd.Series:
    """z-score 标准化 (横截面)."""
    mean = s.mean()
    std = s.std() + 1e-10
    return (s - mean) / std


def _winsorize(s: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """去极值化: 限制在 ±n_sigma 范围内."""
    mean = s.mean()
    std = s.std() + 1e-10
    return s.clip(lower=mean - n_sigma * std, upper=mean + n_sigma * std)


def _composite_score(
    sub: pd.DataFrame,
    weights: dict[str, float],
    zscore_norm: bool = True,
    winsorize_sigma: float = 3.0,
    momentum_windows: tuple[int, ...] | None = None,
    momentum_window_weights: tuple[float, ...] | None = None,
) -> pd.Series:
    """综合得分 (Stage 29 增强: 多窗口动量 + 相关性约束支持).

    Args:
        sub: 收益 DataFrame (rows=time, cols=code)
        weights: 4 因子权重 (value/quality/low_vol/momentum)
        zscore_norm: z-score 标准化
        winsorize_sigma: 去极值阈值 (3σ)
        momentum_windows: 多窗口动量 (如 (5, 20, 60))
        momentum_window_weights: 多窗口权重 (如 (0.3, 0.4, 0.3))
    """
    # 1. 价值得分: 累计收益反向 (跌得多的 = 便宜)
    cum_ret = (1 + sub).cumprod().iloc[-1] - 1
    value_score = -_zscore(cum_ret) if zscore_norm else -cum_ret.rank(pct=True)

    # 2. 质量得分: Sharpe ratio
    mean_ret = sub.mean()
    std_ret = sub.std() + 1e-10
    sharpe = mean_ret / std_ret
    quality_score = _zscore(sharpe) if zscore_norm else sharpe.rank(pct=True)

    # 3. 低波得分: 波动率反向
    vol = sub.std()
    low_vol_score = -_zscore(vol) if zscore_norm else -vol.rank(pct=True)

    # 4. 动量得分: Stage 29 多窗口动量综合
    if momentum_windows is not None and momentum_window_weights is not None:
        if len(momentum_windows) != len(momentum_window_weights):
            raise ValueError("momentum_windows 和 momentum_window_weights 长度必须一致")

        momentum_score = pd.Series(0.0, index=sub.columns)
        for window, w in zip(momentum_windows, momentum_window_weights):
            # 短窗口 skip (避免反转)
            skip = max(1, window // 4)
            if len(sub) < window + skip + 1:
                continue
            cum = (1 + sub).rolling(window).apply(np.prod, raw=True) - 1
            if skip > 0:
                cum = cum.shift(skip)
            window_mom = cum.iloc[-1]  # 最后一期动量
            if zscore_norm:
                window_score = _zscore(window_mom)
            else:
                window_score = window_mom.rank(pct=True)
            momentum_score = momentum_score.add(window_score * w, fill_value=0)
    else:
        # 默认: 单一窗口 (短期动量)
        momentum_score = _zscore(mean_ret) if zscore_norm else mean_ret.rank(pct=True)

    # 去极值化
    if winsorize_sigma > 0:
        value_score = _winsorize(value_score, winsorize_sigma)
        quality_score = _winsorize(quality_score, winsorize_sigma)
        low_vol_score = _winsorize(low_vol_score, winsorize_sigma)
        momentum_score = _winsorize(momentum_score, winsorize_sigma)

    # 加权合成
    composite = (
        weights.get("value", 0.33) * value_score +
        weights.get("quality", 0.33) * quality_score +
        weights.get("low_vol", 0.34) * low_vol_score +
        weights.get("momentum", 0.0) * momentum_score
    )

    return composite


def _apply_corr_constraint(
    selected: list[str],
    returns: pd.DataFrame,
    corr_threshold: float = 0.7,
    corr_window: int = 60,
) -> list[str]:
    """应用相关性约束: 剔除相关系数 > 阈值的冗余 ETF.

    Args:
        selected: 候选 ETF 列表 (按得分排序)
        returns: ETF 收益 DataFrame
        corr_threshold: 相关系数阈值 (默认 0.7)
        corr_window: 计算相关性的窗口 (默认 60)

    Returns:
        过滤后的 ETF 列表 (保持原有顺序)
    """
    if len(selected) <= 1:
        return selected

    valid = [c for c in selected if c in returns.columns]
    if len(valid) <= 1:
        return valid

    sub = returns[valid].iloc[-corr_window:]
    if sub.empty or len(sub) < 10:
        return valid

    try:
        corr = sub.corr()
    except Exception:
        return valid

    filtered = [valid[0]]
    for code in valid[1:]:
        # 检查与已选 ETF 的相关性
        max_corr = 0.0
        for s in filtered:
            try:
                c = corr.loc[code, s]
                if not np.isnan(c):
                    max_corr = max(max_corr, abs(c))
            except Exception:
                pass

        # 相关性低于阈值才保留
        if max_corr < corr_threshold:
            filtered.append(code)

    return filtered


def select_smart_beta_proxy(
    returns: pd.DataFrame,
    lookback: int = 60,
    top_k: int = 5,
    codes: tuple[str, ...] | None = None,
    weights: dict[str, float] | None = None,
    zscore_norm: bool = True,
    winsorize_sigma: float = 3.0,
    momentum_windows: tuple[int, ...] | None = None,
    momentum_window_weights: tuple[float, ...] | None = None,
    corr_constraint: bool = False,
    corr_threshold: float = 0.7,
    corr_window: int = 60,
) -> list[str]:
    """从行业 ETF 中筛选 Smart β 代理 (Stage 29 增强版).

    Stage 29 新增:
        - 多窗口动量加权 (短/中/长综合)
        - 相关性约束 (避免高度相关的 ETF)

    Args:
        returns: ETF 收益 DataFrame
        lookback: 回看窗口
        top_k: 选 top-k
        codes: 候选 ETF 池 (默认行业 ETF)
        weights: 4 因子权重 dict
        zscore_norm: 是否 z-score 标准化
        winsorize_sigma: 去极值阈值
        momentum_windows: 多窗口动量 (如 (5, 20, 60))
        momentum_window_weights: 多窗口权重
        corr_constraint: 是否启用相关性约束
        corr_threshold: 相关性阈值 (默认 0.7)
        corr_window: 相关性窗口

    Returns:
        选中的 ETF code 列表
    """
    if codes is None:
        codes = SECTOR_CODES

    valid = [c for c in codes if c in returns.columns]
    if len(valid) < top_k:
        return valid

    sub = returns[valid].iloc[-lookback:]

    if weights is None:
        weights = {"value": 0.33, "quality": 0.33, "low_vol": 0.34}

    composite = _composite_score(
        sub, weights, zscore_norm, winsorize_sigma,
        momentum_windows, momentum_window_weights,
    )

    # 选 top_k (多选一些用于相关性过滤)
    n_select = min(top_k * 2 if corr_constraint else top_k, len(valid))
    selected = composite.nlargest(n_select).index.tolist()

    # 应用相关性约束
    if corr_constraint:
        selected = _apply_corr_constraint(
            selected, returns, corr_threshold, corr_window
        )

    # 截断到 top_k
    return selected[:top_k]


def select_defensive_smart_beta(
    returns: pd.DataFrame,
    lookback: int = 60,
    top_k: int = 3,
    codes: tuple[str, ...] | None = None,
    weights: dict[str, float] | None = None,
    zscore_norm: bool = True,
    winsorize_sigma: float = 3.0,
    momentum_windows: tuple[int, ...] | None = None,
    momentum_window_weights: tuple[float, ...] | None = None,
    corr_constraint: bool = False,
    corr_threshold: float = 0.7,
    corr_window: int = 60,
) -> list[str]:
    """筛选防御型 Smart β 代理 (Stage 29 增强版)."""
    return select_smart_beta_proxy(
        returns=returns,
        lookback=lookback,
        top_k=top_k,
        codes=codes or DEFENSIVE_SECTOR_CODES,
        weights=weights or {"value": 0.10, "quality": 0.45, "low_vol": 0.45},
        zscore_norm=zscore_norm,
        winsorize_sigma=winsorize_sigma,
        momentum_windows=momentum_windows,
        momentum_window_weights=momentum_window_weights,
        corr_constraint=corr_constraint,
        corr_threshold=corr_threshold,
        corr_window=corr_window,
    )


def select_smart_beta_aggressive(
    returns: pd.DataFrame,
    lookback: int = 60,
    top_k: int = 3,
    codes: tuple[str, ...] | None = None,
    weights: dict[str, float] | None = None,
    zscore_norm: bool = True,
    winsorize_sigma: float = 3.0,
    momentum_windows: tuple[int, ...] | None = None,
    momentum_window_weights: tuple[float, ...] | None = None,
    corr_constraint: bool = False,
    corr_threshold: float = 0.7,
    corr_window: int = 60,
) -> list[str]:
    """筛选进攻型 Smart β 代理 (Stage 29 增强版)."""
    return select_smart_beta_proxy(
        returns=returns,
        lookback=lookback,
        top_k=top_k,
        codes=codes or GROWTH_SECTOR_CODES,
        weights=weights or {"value": 0.0, "quality": 0.30, "low_vol": 0.20, "momentum": 0.50},
        zscore_norm=zscore_norm,
        winsorize_sigma=winsorize_sigma,
        momentum_windows=momentum_windows,
        momentum_window_weights=momentum_window_weights,
        corr_constraint=corr_constraint,
        corr_threshold=corr_threshold,
        corr_window=corr_window,
    )


# ============================================================
# 网格搜索优化器 (Stage 27 新增)
# ============================================================

def grid_search_smart_beta_weights(
    returns: pd.DataFrame,
    codes: tuple[str, ...],
    out_sample_returns: pd.DataFrame | None = None,
    lookback: int = 60,
    top_k: int = 5,
    weight_grid: list[dict[str, float]] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """网格搜索最优 Smart β 筛选权重.

    Args:
        returns: ETF 收益 DataFrame (训练用)
        codes: 候选 ETF 池
        out_sample_returns: 测试集收益 (可选, 用于评估过拟合)
        lookback: 回看窗口
        top_k: 选 top-k
        weight_grid: 权重候选列表 (默认 6 个组合)

    Returns:
        (最优权重, 评估结果 DataFrame)
    """
    if weight_grid is None:
        weight_grid = [
            # 默认平衡
            {"value": 0.33, "quality": 0.33, "low_vol": 0.34},
            # 偏价值
            {"value": 0.50, "quality": 0.25, "low_vol": 0.25},
            # 偏质量
            {"value": 0.25, "quality": 0.50, "low_vol": 0.25},
            # 偏低波
            {"value": 0.25, "quality": 0.25, "low_vol": 0.50},
            # 加动量
            {"value": 0.20, "quality": 0.30, "low_vol": 0.20, "momentum": 0.30},
            # 偏动量
            {"value": 0.10, "quality": 0.20, "low_vol": 0.20, "momentum": 0.50},
        ]

    results = []
    valid = [c for c in codes if c in returns.columns]

    for weights in weight_grid:
        # 评估该权重下选股的回测表现
        nav = _evaluate_weights(returns, valid, weights, lookback, top_k)
        ret = nav.pct_change().fillna(0)

        sharpe = (ret.mean() / (ret.std() + 1e-10)) * np.sqrt(52) if ret.std() > 0 else 0
        annual_ret = (1 + ret).prod() ** (52 / len(ret)) - 1 if len(ret) > 0 else 0
        max_dd = ((nav / nav.cummax() - 1).min()) if len(nav) > 0 else 0
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0

        results.append({
            "weights": weights,
            "sharpe": sharpe,
            "annual_ret": annual_ret,
            "max_dd": max_dd,
            "calmar": calmar,
            "total_return": float(nav.iloc[-1] - 1) if len(nav) > 0 else 0,
        })

    df_results = pd.DataFrame(results)
    best_idx = df_results["sharpe"].idxmax()
    best_weights = df_results.loc[best_idx, "weights"]

    return best_weights, df_results


def _evaluate_weights(
    returns: pd.DataFrame,
    codes: list[str],
    weights: dict[str, float],
    lookback: int = 60,
    top_k: int = 5,
) -> pd.Series:
    """评估特定权重下的 Smart β 筛选表现."""
    if len(returns) < lookback + 10:
        return pd.Series(dtype=float)

    # 滚动选股 + 等权
    nav = pd.Series(1.0, index=returns.index)
    rebal_freq = 4  # 每月调仓
    current_selection = []

    for i in range(lookback, len(returns), rebal_freq):
        # 调仓日
        sub = returns[codes].iloc[i - lookback:i]
        if sub.empty:
            continue

        try:
            composite = _composite_score(sub, weights)
            current_selection = composite.nlargest(top_k).index.tolist()
        except Exception:
            continue

        # 持有到下次调仓
        next_i = min(i + rebal_freq, len(returns))
        for j in range(i, next_i):
            if j == 0 or not current_selection:
                continue
            # 等权收益
            valid_selection = [c for c in current_selection if c in returns.columns]
            if not valid_selection:
                continue
            period_ret = returns[valid_selection].iloc[j].mean()
            nav.iloc[j] = nav.iloc[j - 1] * (1 + period_ret)

    return nav


# ============================================================
# 辅助函数 (兼容旧版)
# ============================================================

def all_style_codes() -> tuple[str, ...]:
    """所有风格组 ETF code (兼容旧版)."""
    out: list[str] = []
    for meta in STYLE_GROUP_METAS.values():
        out.extend(meta.codes)
    return tuple(out)


def all_smart_beta_codes() -> tuple[str, ...]:
    """所有 Smart β 工具 ETF code (兼容旧版)."""
    return tuple(m.code for m in SMART_BETA_METAS.values())


def style_group_of(code: str) -> StyleGroup | None:
    """反查: ETF code 属于哪个风格组."""
    for group, meta in STYLE_GROUP_METAS.items():
        if code in meta.codes:
            return group
    return None


def smart_beta_of(code: str) -> SmartBetaFactor | None:
    """反查: ETF code 属于哪个 Smart β 因子."""
    for factor, meta in SMART_BETA_METAS.items():
        if meta.code == code:
            return factor
    return None


def load_smartbeta_panel(path: str | Path | None = None) -> pd.DataFrame:
    """加载 Smart β ETF 净值面板 (兼容旧版)."""
    if path is None:
        path = Path("data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Smart β 面板未找到: {path}\n"
            f"请先运行: python3.11 scripts/fetch_smartbeta_panel.py"
        )
    return pd.read_parquet(path)


def export_style_groups(path: str | Path) -> None:
    """导出大类/分组定义为 JSON."""
    out: dict = {
        "asset_class": {
            ac.value: list(codes) for ac, codes in ASSET_CLASS_CODES.items()
        },
        "broad": list(BROAD_CODES),
        "sector": list(SECTOR_CODES),
        "overseas": list(OVERSEAS_CODES),
        "gold": list(GOLD_CODES),
        "defensive_sector": list(DEFENSIVE_SECTOR_CODES),
        "growth_sector": list(GROWTH_SECTOR_CODES),
        "all_43": list(get_all_43_codes()),
        "style_groups": {
            g.value: {"name_cn": m.name_cn, "codes": list(m.codes)}
            for g, m in STYLE_GROUP_METAS.items()
        },
        "smart_beta": {
            f.value: {
                "name_cn": m.name_cn,
                "code": m.code,
                "factor_type": m.factor_type,
            }
            for f, m in SMART_BETA_METAS.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


__all__ = [
    # 43 ETF 分类
    "BROAD_CODES",
    "SECTOR_CODES",
    "OVERSEAS_CODES",
    "GOLD_CODES",
    "DEFENSIVE_SECTOR_CODES",
    "GROWTH_SECTOR_CODES",
    "AssetClass",
    "ASSET_CLASS_CODES",
    "get_all_43_codes",
    "classify_43_etf",
    "select_smart_beta_proxy",
    "select_defensive_smart_beta",
    "select_smart_beta_aggressive",
    "_apply_corr_constraint",
    "grid_search_smart_beta_weights",
    # 兼容旧版
    "StyleGroup",
    "SmartBetaFactor",
    "STYLE_GROUP_CODES",
    "SMART_BETA_CODES",
    "SMART_BETA_FACTOR_TYPE",
    "STYLE_GROUP_METAS",
    "SMART_BETA_METAS",
    "ALL_V4_CODES",
    "all_style_codes",
    "all_smart_beta_codes",
    "style_group_of",
    "smart_beta_of",
    "load_smartbeta_panel",
    "export_style_groups",
]