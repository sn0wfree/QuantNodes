# coding=utf-8
"""HMM 市场状态检测 (Stage 17, v4.0).

3 个状态 (基于风格组 ETF 收益的滚动特征):
- 0: 牛市 (高收益, 低波动)
- 1: 熊市 (负收益, 高波动)
- 2: 转换期 (震荡, 趋势不明确)

特征 (4 维):
1. 5 只风格组 ETF 的 20d 平均收益
2. 5 只风格组 ETF 的 20d 波动率
3. 5 只风格组 ETF 的 60d 收益趋势 (slope)
4. 5 只风格组 ETF 的横截面离散度 (max - min)

HMM 模型: 3 状态 GaussianHMM (对角协方差)
训练: 用历史数据 (默认 504 天 = 2 年)
预测: 在线预测当前状态
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .universe_v4 import (
    ALL_V4_CODES,
    STYLE_GROUP_CODES,
    StyleGroup,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """HMM 配置."""
    n_regimes: int = 3              # 状态数 (牛/熊/转换)
    lookback_train: int = 504       # 训练窗口 (2 年)
    feature_window: int = 20        # 特征计算窗口
    trend_window: int = 60          # 趋势窗口
    n_iter: int = 100               # EM 迭代次数
    random_state: int = 42


# 状态标签 (后验估计)
REGIME_LABELS: dict[int, str] = {
    0: "bull",       # 牛
    1: "bear",       # 熊
    2: "transition", # 转换
}


def _build_features(
    nav_df: pd.DataFrame,
    style_codes: Sequence[str],
    feature_window: int = 20,
    trend_window: int = 60,
) -> pd.DataFrame:
    """构建 4 维特征.

    Returns:
        DataFrame, index=date, columns=[mean_ret, vol, trend, dispersion]
    """
    valid = [c for c in style_codes if c in nav_df.columns]
    if not valid:
        return pd.DataFrame()

    sub = nav_df[valid].dropna(how="all")
    if sub.empty:
        return pd.DataFrame()

    # 对每只 ETF 计算 log return
    log_ret = np.log(sub / sub.shift(1))

    # 1. 5 只 ETF 的 20d 平均收益
    mean_ret = log_ret.rolling(feature_window).mean().mean(axis=1)
    mean_ret.name = "mean_ret"

    # 2. 5 只 ETF 的 20d 波动率 (平均)
    vol = log_ret.rolling(feature_window).std().mean(axis=1) * np.sqrt(252)
    vol.name = "vol"

    # 3. 60d 收益趋势 (对每只 ETF 算 slope, 然后取平均)
    # 用 simple linear regression slope / mean 归一化
    def _norm_slope(series: pd.Series) -> float:
        y = series.dropna().values
        if len(y) < 10:
            return 0.0
        x = np.arange(len(y))
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope / max(abs(y.mean()), 1e-6) * trend_window  # 归一化
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    trend_per_etf = log_ret.rolling(trend_window).apply(_norm_slope, raw=False)
    trend = trend_per_etf.mean(axis=1)
    trend.name = "trend"

    # 4. 5 只 ETF 20d 收益的横截面离散度
    cum_ret_20 = sub / sub.shift(feature_window) - 1.0
    dispersion = (cum_ret_20.max(axis=1) - cum_ret_20.min(axis=1))
    dispersion.name = "dispersion"

    features = pd.concat([mean_ret, vol, trend, dispersion], axis=1).dropna()
    return features


def _label_regimes(features: pd.DataFrame, labels: np.ndarray) -> dict[int, int]:
    """根据特征均值给状态打标 (0=bull, 1=bear, 2=transition).

    规则: 收益高+波动低=bull, 收益低+波动高=bear, 中间=transition
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return {int(unique_labels[0]): 2}  # 唯一状态 → 转换期
    if len(unique_labels) == 2:
        # 2 个状态: 高的→bull (0), 低的→bear (1)
        ret_by_label = {l: features.loc[labels == l, "mean_ret"].mean() for l in unique_labels}
        sorted_by_ret = sorted(unique_labels, key=lambda l: -ret_by_label[l])
        return {int(sorted_by_ret[0]): 0, int(sorted_by_ret[1]): 1}

    # 3+ 状态
    ret_by_label = {l: features.loc[labels == l, "mean_ret"].mean() for l in unique_labels}
    sorted_by_ret = sorted(unique_labels, key=lambda l: -ret_by_label[l])
    bull_label = int(sorted_by_ret[0])
    bear_label = int(sorted_by_ret[-1])

    out: dict[int, int] = {bull_label: 0, bear_label: 1}
    for l in unique_labels:
        if int(l) not in out:
            out[int(l)] = 2  # 其余 → transition
    return out


class RegimeDetector:
    """HMM 市场状态检测器 (Stage 17, v4.0)."""

    def __init__(
        self,
        config: RegimeConfig | None = None,
        style_codes: Sequence[str] | None = None,
    ):
        self.config = config or RegimeConfig()
        self.style_codes = list(style_codes or [])
        # 收集所有风格组 code
        for codes in STYLE_GROUP_CODES.values():
            for c in codes:
                if c not in self.style_codes:
                    self.style_codes.append(c)
        self.model = None
        self.label_map: dict[int, int] = {}

    def fit(self, nav_df: pd.DataFrame, end_date: pd.Timestamp | None = None) -> "RegimeDetector":
        """用历史数据训练 HMM (使用距离先验 + 软约束)."""
        from hmmlearn import hmm
        from .regime_transitions import (
            build_distance_transmat,
            soft_constrain,
            DistanceTransitionConfig,
        )

        if end_date is None:
            end_date = nav_df.index[-1]

        # 取训练数据
        train_data = nav_df.loc[:end_date].iloc[-self.config.lookback_train:]
        if len(train_data) < 252:
            logger.warning("训练数据不足 1 年 (%d 行), HMM 训练可能不准", len(train_data))

        features = _build_features(
            train_data, self.style_codes,
            feature_window=self.config.feature_window,
            trend_window=self.config.trend_window,
        )
        if len(features) < 60:
            logger.warning("特征数据不足 60 行, 跳过训练")
            return self

        X = features.values

        # 标准化 (用训练集 stats)
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ < 1e-8] = 1.0
        X_norm = (X - self.feature_mean_) / self.feature_std_

        # 距离先验矩阵 (作为 HMM 转移矩阵的先验)
        dist_cfg = DistanceTransitionConfig(
            alpha=1.5, gamma=0.3, sticky_bonus=0.0,
            n_states=self.config.n_regimes,
        )
        self.distance_prior_ = build_distance_transmat(
            dist_cfg.alpha, dist_cfg.gamma, dist_cfg.sticky_bonus,
            dist_cfg.n_states,
        )
        soft_lam = 0.3  # 软约束强度

        # 训练 HMM (多次初始化, 用距离先验)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = None
            best_score = -np.inf
            for init_idx in range(2):  # 减到 2 次
                try:
                    m = hmm.GaussianHMM(
                        n_components=self.config.n_regimes,
                        covariance_type="diag",
                        n_iter=min(self.config.n_iter, 30),  # 减到 30
                        random_state=self.config.random_state + init_idx,
                        transmat_prior=self.distance_prior_,
                    )
                    m.fit(X_norm)
                    if soft_lam > 0:
                        m.transmat_ = soft_constrain(
                            m.transmat_, self.distance_prior_, soft_lam,
                        )
                    score = m.score(X_norm)
                    if score > best_score:
                        best_score = score
                        best_model = m
                except Exception as e:
                    logger.warning("HMM init %d 失败: %s", init_idx, e)
                    continue

            if best_model is not None:
                # 最终软约束一次
                if soft_lam > 0:
                    best_model.transmat_ = soft_constrain(
                        best_model.transmat_, self.distance_prior_, soft_lam,
                    )
                self.model = best_model
                labels = self.model.predict(X_norm)
                self._force_full_label_map(features, labels)
            else:
                logger.warning("HMM 全部初始化失败, 退化为简单 regime 检测")
                self.model = None

        return self

    def _force_full_label_map(self, features: pd.DataFrame, labels: np.ndarray) -> None:
        """强制 3 个状态都映射 (避免某状态没出现).

        如果 HMM 只用 2 个状态, 缺失的状态用"中间"特征 fallback.
        """
        unique = np.unique(labels)
        if len(unique) == self.config.n_regimes:
            self.label_map = _label_regimes(features, labels)
            return

        # 找缺失状态
        present = set(unique.tolist())
        missing = [s for s in range(self.config.n_regimes) if s not in present]
        # 用 _label_regimes 算现有状态的语义标签
        partial_map = _label_regimes(features, labels)
        # 缺失状态分配中间标签
        used_sem = set(partial_map.values())
        for s in missing:
            for sem in (0, 1, 2):
                if sem not in used_sem:
                    partial_map[s] = sem
                    used_sem.add(sem)
                    break
        self.label_map = partial_map

    def predict(self, nav_df: pd.DataFrame, as_of: pd.Timestamp) -> int:
        """预测 as_of 当天的市场状态.

        Returns:
            0 (bull) | 1 (bear) | 2 (transition) | -1 (未训练)
        """
        if self.model is None:
            return -1

        features = _build_features(
            nav_df.loc[:as_of], self.style_codes,
            feature_window=self.config.feature_window,
            trend_window=self.config.trend_window,
        )
        if features.empty:
            return -1

        X = features.values
        if len(X) == 0:
            return -1
        X_norm = (X - self.feature_mean_) / self.feature_std_

        try:
            raw_label = int(self.model.predict(X_norm[-1:])[0])
        except Exception:
            return -1
        return self.label_map.get(raw_label, raw_label)

    def predict_series(
        self,
        nav_df: pd.DataFrame,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        step: int = 5,
        min_duration: int = 30,
        apply_min_duration: bool = True,
    ) -> pd.Series:
        """滚动预测 (用于因子择时), 含 min_duration 后处理. 批量优化版.

        Args:
            min_duration: 最小持续期 (天数)
            apply_min_duration: 是否应用 min_duration 后处理
        """
        if self.model is None:
            return pd.Series(dtype=int)

        # 用扩展窗口 (rolling) 模拟"as_of 之前的特征"
        # 实际: 用 _build_features on full panel, 但 HMM 是 time-aware 的
        full_features = _build_features(
            nav_df, self.style_codes,
            feature_window=self.config.feature_window,
            trend_window=self.config.trend_window,
        )
        if full_features.empty:
            return pd.Series(dtype=int)

        # 标准化
        X = full_features.values
        X_norm = (X - self.feature_mean_) / self.feature_std_

        # 批量预测 (1 次 HMM 调用, O(n))
        try:
            raw_labels = self.model.predict(X_norm)
        except Exception:
            return pd.Series(dtype=int)

        # 映射 HMM 状态 → 语义标签
        semantic = np.array([self.label_map.get(int(l), 2) for l in raw_labels])
        s_full = pd.Series(semantic, index=full_features.index, name="regime")

        # 取 start~end 范围, 步长 step
        s_window = s_full.loc[start:end]
        if step > 1:
            s_window = s_window.iloc[::step]

        if apply_min_duration and len(s_window) > 0:
            from .regime_transitions import enforce_minimum_duration
            s_arr = enforce_minimum_duration(s_window.values, min_duration=min_duration)
            s_window = pd.Series(s_arr, index=s_window.index, name="regime")

        return s_window


def get_regime_factor_weight(
    regime: int,
    factor_name: str,
) -> float:
    """根据市场状态返回因子权重偏好.

    regime: 0 (bull) | 1 (bear) | 2 (transition)
    factor_name: "momentum" | "reversal" | "value" | "low_vol" | "dividend" | "quality"

    Returns:
        0.5 ~ 2.0 范围的权重偏好 (1.0 = 中性)
    """
    # 牛市: 动量↑↑ 价值↓ 反转↓
    # 熊市: 动量↓ 价值↑↑ 低波↑↑ 红利↑↑ 质量↑
    # 转换期: 反转↑ 低波↑ 价值↑
    if regime == 0:  # bull
        return {
            "momentum": 2.0,
            "reversal": 0.5,
            "value":    0.5,
            "low_vol":  0.5,
            "dividend": 0.5,
            "quality":  1.0,
        }.get(factor_name, 1.0)
    elif regime == 1:  # bear
        return {
            "momentum": 0.3,
            "reversal": 0.5,
            "value":    2.0,
            "low_vol":  2.0,
            "dividend": 2.0,
            "quality":  1.5,
        }.get(factor_name, 1.0)
    else:  # transition
        return {
            "momentum": 0.7,
            "reversal": 1.5,
            "value":    1.2,
            "low_vol":  1.5,
            "dividend": 1.0,
            "quality":  1.0,
        }.get(factor_name, 1.0)


__all__ = [
    "RegimeConfig",
    "RegimeDetector",
    "REGIME_LABELS",
    "get_regime_factor_weight",
    "_build_features",
]
