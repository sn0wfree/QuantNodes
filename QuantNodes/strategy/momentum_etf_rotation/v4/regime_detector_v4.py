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
    """HMM 配置 (Stage 27 增强)."""
    n_regimes: int = 3              # 状态数 (牛/熊/转换)
    lookback_train: int = 504       # 训练窗口 (2 年)
    feature_window: int = 20        # 特征计算窗口
    trend_window: int = 60          # 趋势窗口
    n_iter: int = 200               # EM 迭代次数 (Stage 27: 100→200)
    random_state: int = 42
    n_restarts: int = 5             # Stage 27: 多次重启取最优
    covariance_type: str = "full"   # Stage 27: full 协方差 (更灵活)


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
    """构建增强特征 (Stage 27: 8 维特征).

    原始 4 维:
        1. mean_ret: 20d 平均收益
        2. vol: 20d 波动率
        3. trend: 60d 趋势斜率
        4. dispersion: 20d 横截面离散度

    新增 4 维 (Stage 27):
        5. momentum: 60d 动量 (中期趋势)
        6. mean_vol_change: 波动率变化 (VIX 类信号)
        7. correlation: 20d 横截面相关性 (市场一致性)
        8. drawdown: 最大回撤 (风险信号)

    Returns:
        DataFrame, index=date, columns=[8 features]
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
    def _norm_slope(series: pd.Series) -> float:
        y = series.dropna().values
        if len(y) < 10:
            return 0.0
        x = np.arange(len(y))
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope / max(abs(y.mean()), 1e-6) * trend_window
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    trend_per_etf = log_ret.rolling(trend_window).apply(_norm_slope, raw=False)
    trend = trend_per_etf.mean(axis=1)
    trend.name = "trend"

    # 4. 5 只 ETF 20d 收益的横截面离散度
    cum_ret_20 = sub / sub.shift(feature_window) - 1.0
    dispersion = (cum_ret_20.max(axis=1) - cum_ret_20.min(axis=1))
    dispersion.name = "dispersion"

    # Stage 27: 新增 4 维特征
    # 5. 60d 动量 (中期趋势, 比 trend 更稳定)
    momentum = sub / sub.shift(trend_window) - 1.0
    momentum_mean = momentum.mean(axis=1)
    momentum_mean.name = "momentum"

    # 6. 波动率变化 (VIX 类信号, 波动率上升 = 恐慌)
    vol_20 = log_ret.rolling(feature_window).std().mean(axis=1)
    vol_60 = log_ret.rolling(trend_window).std().mean(axis=1)
    vol_change = (vol_20 - vol_60) / (vol_60 + 1e-10)
    vol_change.name = "vol_change"

    # 7. 20d 横截面相关性 (市场一致性, 高相关 = 趋势明确)
    # 简化: 用 ETF 收益的平均自相关性代替
    correlation = log_ret.rolling(feature_window).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x.values)[0, 1] 
        if len(x) > 10 else 0.0, raw=False
    ).mean(axis=1)
    correlation.name = "correlation"

    # 8. 最大回撤 (风险信号, 大回撤 = 熊市)
    cummax = sub.cummax()
    drawdown = ((sub - cummax) / cummax).mean(axis=1)
    drawdown.name = "drawdown"

    features = pd.concat([
        mean_ret, vol, trend, dispersion,
        momentum_mean, vol_change, correlation, drawdown,
    ], axis=1).dropna()

    return features


def detect_regime_simple(
    nav_df: pd.DataFrame,
    style_codes: Sequence[str],
    short_window: int = 20,
    long_window: int = 60,
    vol_window: int = 20,
) -> pd.Series:
    """简单规则化 regime 检测 (Stage 27 替代 HMM).

    基于技术指标:
        - 短期动量 > 0 且 长期动量 > 0 → bull
        - 短期动量 < 0 且 长期动量 < 0 → bear
        - 其他 → transition

    Args:
        nav_df: 价格面板
        style_codes: ETF 代码列表
        short_window: 短期窗口
        long_window: 长期窗口
        vol_window: 波动率窗口

    Returns:
        Series: 0=bull, 1=bear, 2=transition
    """
    valid = [c for c in style_codes if c in nav_df.columns]
    if not valid:
        return pd.Series(dtype=int)

    sub = nav_df[valid].copy()
    # 填充 0 值 (避免除零)
    sub = sub.replace(0, np.nan).ffill().fillna(0)
    
    if sub.empty or len(sub) < long_window + 10:
        return pd.Series(dtype=int)

    # 计算收益 (安全版本)
    shifted = sub.shift(1)
    shifted = shifted.replace(0, np.nan)
    ret = (sub - shifted) / shifted
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 短期动量 (20d) - 用均值代替单点
    short_mom = sub.rolling(short_window).mean() / sub.rolling(short_window).mean().shift(short_window) - 1.0
    short_mom_avg = short_mom.mean(axis=1)

    # 长期动量 (60d)
    long_mom = sub.rolling(long_window).mean() / sub.rolling(long_window).mean().shift(long_window) - 1.0
    long_mom_avg = long_mom.mean(axis=1)

    # 波动率 (20d)
    vol = ret.rolling(vol_window).std().mean(axis=1) * np.sqrt(52)

    # 规则化 regime 检测
    regime = pd.Series(2, index=sub.index)  # 默认 transition

    # bull: 短期和长期动量都为正, 波动率低
    vol_median = vol.median()
    bull_mask = (short_mom_avg > 0) & (long_mom_avg > 0) & (vol < vol_median)
    regime[bull_mask] = 0

    # bear: 短期和长期动量都为负, 波动率高
    bear_mask = (short_mom_avg < 0) & (long_mom_avg < 0) & (vol > vol_median)
    regime[bear_mask] = 1

    return regime


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

        # 训练 HMM (多次重启, 用距离先验)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = None
            best_score = -np.inf
            for init_idx in range(self.config.n_restarts):  # Stage 27: 多次重启
                try:
                    m = hmm.GaussianHMM(
                        n_components=self.config.n_regimes,
                        covariance_type=self.config.covariance_type,  # Stage 27: full
                        n_iter=self.config.n_iter,
                        random_state=self.config.random_state + init_idx,
                        transmat_prior=self.distance_prior_,
                        tol=1e-4,  # Stage 27: 收敛阈值
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
                self.best_score_ = best_score
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
    """根据市场状态返回因子权重偏好 (Stage 27 更新: 适配 8 因子).

    regime: 0 (bull) | 1 (bear) | 2 (transition)
    factor_name: 8 因子之一

    Returns:
        0.3 ~ 2.0 范围的权重偏好 (1.0 = 中性)
    """
    # 牛市: 动量↑↑ 中期动量↑↑ 价值↓ 反转↓ 低波↓ 波动变化↓
    # 熊市: 动量↓ 价值↑↑ 低波↑↑ 估值代理↑↑ 基本面代理↑ 波动变化↑
    # 转换期: 反转↑ 低波↑ 价值↑
    if regime == 0:  # bull
        return {
            "momentum": 2.0,
            "momentum_12_1": 2.0,
            "reversal": 0.5,
            "value": 0.5,
            "low_vol": 0.5,
            "volatility_change": 0.3,
            "value_proxy": 0.5,
            "quality_proxy": 1.0,
        }.get(factor_name, 1.0)
    elif regime == 1:  # bear
        return {
            "momentum": 0.3,
            "momentum_12_1": 0.3,
            "reversal": 0.5,
            "value": 2.0,
            "low_vol": 2.0,
            "volatility_change": 2.0,
            "value_proxy": 2.0,
            "quality_proxy": 1.5,
        }.get(factor_name, 1.0)
    else:  # transition
        return {
            "momentum": 0.7,
            "momentum_12_1": 0.7,
            "reversal": 1.5,
            "value": 1.2,
            "low_vol": 1.5,
            "volatility_change": 1.0,
            "value_proxy": 1.0,
            "quality_proxy": 1.0,
        }.get(factor_name, 1.0)


# ============================================================
# 宏观因子融合 Regime 检测 (Stage 27 新增)
# ============================================================

def detect_regime_with_macro(
    nav_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    short_window: int = 20,
    long_window: int = 60,
    vol_window: int = 20,
    macro_threshold: float = 0.5,
) -> pd.Series:
    """宏观因子融合的 regime 检测 (Stage 27 新增).

    融合两个信号:
        1. 收益类信号: 短期动量 + 长期动量 + 波动率
        2. 宏观类信号 (可选): 增长 + 通胀 + 流动性

    Args:
        nav_df: ETF 收益面板
        macro_df: 宏观因子面板 (可选, 如果提供则融合)
        short_window, long_window, vol_window: 收益类窗口
        macro_threshold: 宏观信号阈值

    Returns:
        Series: 0=bull, 1=bear, 2=transition
    """
    # 基础 regime 检测 (从收益类)
    base_regime = detect_regime_simple(
        nav_df, list(nav_df.columns),
        short_window=short_window,
        long_window=long_window,
        vol_window=vol_window,
    )

    if macro_df is None or len(macro_df) == 0:
        return base_regime

    # 宏观信号
    macro_signal = pd.Series(0.0, index=base_regime.index)

    # 增长信号
    if "宏观增长因子" in macro_df.columns:
        growth = macro_df["宏观增长因子"]
        growth_z = (growth - growth.rolling(60).mean()) / (growth.rolling(60).std() + 1e-10)
        macro_signal = macro_signal.add(growth_z.fillna(0), fill_value=0)

    # 通胀信号 (反向, 高通胀 = 收紧)
    if "宏观通胀因子_生活端" in macro_df.columns:
        inflation = macro_df["宏观通胀因子_生活端"]
        inflation_z = (inflation - inflation.rolling(60).mean()) / (inflation.rolling(60).std() + 1e-10)
        macro_signal = macro_signal.add(-inflation_z.fillna(0), fill_value=0)

    # 流动性信号 (信用利差反向, 利差扩 = 紧)
    if "信用利差因子" in macro_df.columns:
        credit = macro_df["信用利差因子"]
        credit_z = (credit - credit.rolling(60).mean()) / (credit.rolling(60).std() + 1e-10)
        macro_signal = macro_signal.add(-credit_z.fillna(0), fill_value=0)

    # 对齐到 regime 索引
    macro_aligned = macro_signal.reindex(base_regime.index, method="ffill").fillna(0)

    # 融合: 收益 regime + 宏观调整
    final_regime = base_regime.copy()

    # 宏观信号强: 调整 regime
    bull_mask = (macro_aligned > macro_threshold)
    bear_mask = (macro_aligned < -macro_threshold)

    # 宏观 bullish + 当前是 bear → 调整为 transition
    final_regime[bull_mask & (final_regime == 1)] = 2

    # 宏观 bearish + 当前是 bull → 调整为 transition
    final_regime[bear_mask & (final_regime == 0)] = 2

    return final_regime


__all__ = [
    "RegimeConfig",
    "RegimeDetector",
    "REGIME_LABELS",
    "get_regime_factor_weight",
    "detect_regime_with_macro",
    "_build_features",
]
