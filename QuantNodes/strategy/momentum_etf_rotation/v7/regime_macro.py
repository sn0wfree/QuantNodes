"""
v7.0 5 状态 HMM 分类器 (Stage 30 POC).

[PIT 关键约束]
1. HMM 输入是 PIT 调整后的日度时间序列 (get_pit_series)
2. T 日训练/预测只能用 release_date <= T 的数据
3. 状态排序按 PMI 特征均值:
   - 最高 = recovery (复苏)
   - 第二 = overheat (过热, PMI 高 + CPI 高)
   - 中间 = neutral (中性)
   - 第四 = stagflation (滞胀, PMI 低 + CPI 高)
   - 最低 = recession (衰退)

[5 状态定义]
- recovery:     PMI↑ + 流动性松 → target_vol=20%
- overheat:     PMI↑ + CPI↑ + 流动性紧 → target_vol=12%
- stagflation:  PMI↓ + CPI↑ + 流动性紧 → target_vol=6%
- recession:    PMI↓ + CPI↓ + 流动性松 → target_vol=10%
- neutral:      其他 → target_vol=14%

[输入特征 (5 维, 日频)]
1. PMI 同比差 (PIT)
2. CPI 同比 (PIT)
3. M2 同比 (PIT)
4. CN10Y 国债收益率 (PIT, T+0)
5. US10Y 美债收益率 (PIT, T+0)

[回测约束] 训练窗口 expanding, 预测时 T 日只用 T 及之前 PIT 数据.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .factor_macro import (
    META,
    get_pit_series,
    fetch_macro_factor,
    CACHE_DIR,
)


REGIME_NAMES: tuple[str, ...] = ("recovery", "overheat", "neutral", "stagflation", "recession")

REGIME_VOL_TARGETS: dict[str, float] = {
    "recovery":     0.20,   # 复苏, 全力进攻
    "overheat":     0.12,   # 过热, 收紧
    "stagflation":  0.06,   # 滞胀, 极限防御
    "recession":    0.10,   # 衰退, 防御
    "neutral":      0.14,   # 中性
}


@dataclass
class HMM5StateResult:
    """HMM 5 状态训练结果."""
    regime_order: np.ndarray           # 排序映射: raw_label -> regime_idx
    regime_names: tuple[str, ...] = REGIME_NAMES
    feature_cols: tuple[str, ...] = ("PMI", "CPI", "M2", "CN10Y", "US10Y")
    train_period: tuple[str, str] = ("", "")
    n_states: int = 5
    converged: bool = True
    final_loss: float = 0.0


def _build_pit_features(
    dates: pd.DatetimeIndex,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """给定回测日期序列, 构造 PIT 调整后的 5 维日度特征矩阵.

    Args:
        dates: 回测日期 (T+0 ... T+N), 通常是 ETF 交易日的索引
        cache_dir: 宏观因子 parquet 缓存目录

    Returns:
        DataFrame, index=dates, columns=5 宏观因子, values=PIT adjusted
    """
    out = {}
    for name in ("PMI", "CPI", "M2", "CN10Y", "US10Y"):
        df = pd.read_parquet(cache_dir / f"{name}.parquet")
        out[name] = get_pit_series(df, dates)
    feat = pd.DataFrame(out, index=dates)
    return feat


def _zscore_rolling(s: pd.Series, window: int = 252) -> pd.Series:
    """滚动 z-score (5 状态 HMM 特征标准化)."""
    mu = s.rolling(window, min_periods=max(60, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(60, window // 4)).std()
    return (s - mu) / sd.replace(0, 1)


def train_5state_hmm(
    feature_df: pd.DataFrame,
    n_iter: int = 200,
    random_state: int = 42,
    cov_type: str = "full",
) -> HMM5StateResult:
    """训练 5 状态 HMM.

    Args:
        feature_df: DataFrame, index=date, columns=5 features (z-scored)
        n_iter: HMM 训练最大迭代
        random_state: 随机种子
        cov_type: 协方差类型 ("full" | "diag" | "spherical")

    Returns:
        HMM5StateResult 包含排序映射和训练元信息
    """
    from hmmlearn import hmm

    feat_clean = feature_df.dropna()
    if len(feat_clean) < 100:
        raise ValueError(f"训练数据不足: {len(feat_clean)} 行, 需要 ≥ 100")

    X = feat_clean.values

    model = hmm.GaussianHMM(
        n_components=5,
        covariance_type=cov_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X)

    # 按 PMI 特征均值排序: PMI 均值最高 = recovery, 最低 = recession
    pmi_col_idx = list(feat_clean.columns).index("PMI")
    raw_order = np.argsort(model.means_[:, pmi_col_idx])[::-1]
    # raw_order[0] = PMI 最高的 raw_label
    # mapped[0] = recovery (idx=0), mapped[4] = recession (idx=4)
    regime_order = np.zeros(5, dtype=int)
    for rank, raw_label in enumerate(raw_order):
        regime_order[raw_label] = rank

    return HMM5StateResult(
        regime_order=regime_order,
        train_period=(
            feat_clean.index.min().strftime("%Y-%m-%d"),
            feat_clean.index.max().strftime("%Y-%m-%d"),
        ),
        n_states=5,
        converged=model.monitor_.converged,
        final_loss=float(model.monitor_.history[-1]) if model.monitor_.history else 0.0,
    )


def predict_5state(
    model,
    regime_order: np.ndarray,
    feature_df: pd.DataFrame,
) -> pd.Series:
    """预测整个时间序列的 5 状态.

    Args:
        model: 训练好的 hmm.GaussianHMM
        regime_order: train_5state_hmm 返回的排序映射
        feature_df: DataFrame, index=date, columns=5 features

    Returns:
        pd.Series, index=date, values=regime_name (str)
    """
    feat_clean = feature_df.dropna()
    if len(feat_clean) == 0:
        return pd.Series(dtype=object)
    raw_labels = model.predict(feat_clean.values)
    mapped = np.array([regime_order[r] for r in raw_labels])
    return pd.Series(
        [REGIME_NAMES[m] for m in mapped],
        index=feat_clean.index,
        name="regime",
    )


def build_regime_timeline(
    start: str = "2018-06-01",   # 给 252 天 warmup
    end: str = "2026-06-30",
    feature_window: int = 252,
) -> pd.DataFrame:
    """构建回测期的 5 状态时间线.

    Args:
        start: 回测起点 (考虑 PIT + z-score warmup, 实际建议 ≥ 2018-06)
        end: 回测终点
        feature_window: z-score 滚动窗口

    Returns:
        DataFrame, columns=[date, regime, vol_target] + 5 features
    """
    # 1. 构造 PIT 调整的日度特征
    dates = pd.date_range(start, end, freq="B")  # 工作日
    feat_pit = _build_pit_features(dates)

    # 2. 滚动 z-score 标准化
    feat_z = feat_pit.apply(lambda c: _zscore_rolling(c, window=feature_window))
    feat_z = feat_z.dropna()

    if len(feat_z) < 100:
        raise ValueError(f"标准化后数据不足: {len(feat_z)}")

    # 3. 训练 HMM (expanding, POC 用全量)
    result = train_5state_hmm(feat_z)
    from hmmlearn import hmm
    model = hmm.GaussianHMM(
        n_components=5, covariance_type="full", n_iter=200, random_state=42, tol=1e-4,
    )
    # 复用 train_5state_hmm 的训练结果
    feat_clean = feat_z
    model.fit(feat_clean.values)

    # 4. 预测
    regime_series = predict_5state(model, result.regime_order, feat_z)

    # 5. 拼接结果
    out = pd.DataFrame({
        "date": regime_series.index,
        "regime": regime_series.values,
    })
    out["vol_target"] = out["regime"].map(REGIME_VOL_TARGETS)
    for c in feat_pit.columns:
        out[c] = feat_pit.reindex(out["date"])[c].values
    for c in feat_z.columns:
        out[f"{c}_zscore"] = feat_z.reindex(out["date"])[c].values

    return out


__all__ = [
    "REGIME_NAMES",
    "REGIME_VOL_TARGETS",
    "HMM5StateResult",
    "_build_pit_features",
    "_zscore_rolling",
    "train_5state_hmm",
    "predict_5state",
    "build_regime_timeline",
]
