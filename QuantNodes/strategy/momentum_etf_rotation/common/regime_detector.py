# coding=utf-8
"""HMM Regime 检测器 (Stage 9-D).

使用隐马尔可夫模型 (HMM) 识别 3 种市场状态:
    - 牛市 (bull): 高收益, 低波动
    - 震荡市 (neutral): 中性
    - 熊市 (bear): 低/负收益, 高波动

根据识别出的状态, 动态调整 RotationConfig 参数
(lookback / rank_cutoff / exposure_bear / vol_target).

依赖: hmmlearn (pip install hmmlearn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class RegimeParams:
    """每个 regime 对应的参数覆盖 (Stage 9-D).

    regime_label -> dict of overrides for RotationConfig.
    """
    bull: Mapping[str, float] = field(default_factory=lambda: {
        "lookback": 60, "rank_cutoff": 0.50,
    })
    neutral: Mapping[str, float] = field(default_factory=lambda: {
        "lookback": 90, "rank_cutoff": 0.30,
    })
    bear: Mapping[str, float] = field(default_factory=lambda: {
        "lookback": 90, "rank_cutoff": 0.10,  # 缩短到 90 以避免数据不足
    })


@dataclass
class RegimeDetector:
    """HMM regime 检测器配置 (Stage 9-D)."""
    enabled: bool = False
    n_regimes: int = 3
    lookback_train: int = 504       # 训练窗口 (2 年)
    retrain_freq: int = 60         # 重训练频率 (调仓次数)
    regime_params: RegimeParams = field(default_factory=RegimeParams)
    benchmark_code: str = "510300"  # 用于训练的特征 ETF


class HMMRegimeDetector:
    """3-regime HMM 检测器.

    训练数据: 基准 ETF 的日收益率 + 21 日波动率
    输出: 当前 regime (0=熊, 1=震荡, 2=牛)
    """

    def __init__(self, n_regimes: int = 3, lookback_train: int = 504):
        self.n_regimes = n_regimes
        self.lookback_train = lookback_train
        self.model = None  # 延迟初始化 (避免 hmmlearn 未装时 import 失败)
        self.regime_order_: np.ndarray | None = None
        self._fitted = False

    def _build_features(self, nav: pd.Series) -> np.ndarray:
        """构建 HMM 特征: 日收益率 + 21 日波动率."""
        rets = nav.pct_change().dropna()
        vol_21 = rets.rolling(21).std() * np.sqrt(252)
        # 拼接 [returns, vol], 删除 NaN
        features = pd.concat([rets.rename("ret"), vol_21.rename("vol")], axis=1).dropna()
        return features.values

    def fit(self, nav: pd.Series):
        """训练 HMM, 按均值排序 regime 标签."""
        from hmmlearn import hmm
        if len(nav) < self.lookback_train:
            raise ValueError(
                f"训练数据不足: 需 {self.lookback_train} 天, 实际 {len(nav)} 天"
            )
        # 仅用最近 lookback_train 天训练
        train_nav = nav.iloc[-self.lookback_train:]
        features = self._build_features(train_nav)
        if len(features) < 50:
            raise ValueError(f"训练特征不足: {len(features)} 行, 需要 ≥ 50")
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self.model.fit(features)
        # 按均值排序: 0=熊 (最低), 1=震荡, 2=牛 (最高)
        self.regime_order_ = np.argsort(self.model.means_[:, 0])
        self._fitted = True
        return self

    def predict(self, nav: pd.Series) -> int:
        """预测当前 regime (0=熊, 1=震荡, 2=牛)."""
        if not self._fitted:
            return 1  # 未训练, 默认震荡
        features = self._build_features(nav)
        if len(features) < 50:
            return 1  # 数据不足, 默认震荡
        labels = self.model.predict(features)
        current_label = labels[-1]
        # 映射到排序后的 regime
        return int(np.where(self.regime_order_ == current_label)[0][0])

    def predict_series(self, nav: pd.Series) -> pd.Series:
        """预测整个时间序列的 regime."""
        if not self._fitted:
            raise ValueError("必须先调用 fit()")
        features = self._build_features(nav)
        labels = self.model.predict(features)
        # 映射到排序后的 regime
        mapped = np.zeros(len(labels), dtype=int)
        for raw_label in range(self.n_regimes):
            mask = labels == raw_label
            regime_idx = int(np.where(self.regime_order_ == raw_label)[0][0])
            mapped[mask] = regime_idx
        return pd.Series(mapped, index=nav.index[-len(labels):])


def get_regime_params(cfg, regime: int, regime_params: RegimeParams) -> dict:
    """根据 regime 获取参数覆盖."""
    if regime == 0:  # 熊
        return dict(regime_params.bear)
    elif regime == 2:  # 牛
        return dict(regime_params.bull)
    else:  # 震荡
        return dict(regime_params.neutral)


__all__ = [
    "RegimeDetector",
    "RegimeParams",
    "HMMRegimeDetector",
    "get_regime_params",
]