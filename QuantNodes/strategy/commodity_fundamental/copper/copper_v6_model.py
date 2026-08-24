# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_v6_model.py

V6策略模型类

将v6/v6.1/v6.2策略封装为模型类，支持:
    - 参数配置
    - 自动参数优化
    - 信号生成
    - 回测

架构:
    v6 = avg(v4仓位, v5精简仓位)
    v6.1 = v6 + bull_enhance (条件性加仓)
    v6.2 = v6.1 + 波动率目标化

用法:
    model = V6Model(version="v6.2")
    signal = model.build_signal(con)
    result = model.backtest(signal)

    # 参数优化
    best_params = model.optimize(con)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .copper_model_v6 import (
    build_signal_v6, apply_bull_enhance, backtest_v6,
    _load_close_w, _load_tfapi, _rolling_z,
)

DB = Path("data/cmf/fundamental.duckdb")


@dataclass
class V6Config:
    """V6模型配置."""
    version: str = "v6.2"  # v6 / v6.1 / v6.2

    # v6参数
    with_carry: bool = False

    # bull_enhance参数 (v6.1/v6.2)
    threshold: int = 2
    bull_mom_weeks: int = 26
    bull_mom_thresh: float = 0.10

    # 波动率目标化参数 (v6.2)
    target_vol: float = 0.12
    max_leverage: float = 1.0

    # 回测参数
    cost_bps: float = 5.0


class V6Model:
    """V6策略模型.

    支持版本:
        - v6: 基础版 (avg v4+v5)
        - v6.1: bull_enhance版
        - v6.2: bull_enhance + 波动率目标化 (推荐)
    """

    def __init__(self, config: V6Config = None, **kwargs):
        """初始化模型."""
        if config is None:
            config = V6Config(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

    def build_signal(self, con=None) -> pd.DataFrame:
        """构建信号.

        参数:
            con: 数据库连接

        返回: DataFrame with position列
        """
        if con is None:
            con = duckdb.connect(str(DB), read_only=True)

        cfg = self.config

        # 构建v6基础信号
        signal = build_signal_v6("CU0", con, with_carry=cfg.with_carry)

        # 根据版本应用不同处理
        if cfg.version == "v6":
            # v6: 直接使用pos_main
            signal["position"] = signal["pos_main"]

        elif cfg.version == "v6.1":
            # v6.1: 应用bull_enhance
            signal = apply_bull_enhance(
                signal, con,
                threshold=cfg.threshold,
                bull_mom_weeks=cfg.bull_mom_weeks,
                bull_mom_thresh=cfg.bull_mom_thresh,
                target_vol=cfg.target_vol,
                max_leverage=cfg.max_leverage,
            )
            signal["position"] = signal["position_enhanced"]

        elif cfg.version == "v6.2":
            # v6.2: bull_enhance + 波动率目标化
            signal = apply_bull_enhance(
                signal, con,
                threshold=cfg.threshold,
                bull_mom_weeks=cfg.bull_mom_weeks,
                bull_mom_thresh=cfg.bull_mom_thresh,
                target_vol=cfg.target_vol,
                max_leverage=cfg.max_leverage,
            )
            signal["position"] = signal["position_sized"]

        return signal

    def backtest(self, signal: pd.DataFrame) -> dict:
        """回测.

        参数:
            signal: build_signal输出

        返回: 回测结果字典
        """
        return backtest_v6(signal, cost_bps=self.config.cost_bps)

    def optimize(self, con=None,
                 version_range: list = None,
                 threshold_range: list = None,
                 bull_mom_weeks_range: list = None,
                 bull_mom_thresh_range: list = None,
                 target_vol_range: list = None,
                 metric: str = "sharpe") -> dict:
        """参数优化 (网格搜索).

        参数:
            con: 数据库连接
            version_range: 版本搜索范围
            threshold_range: threshold搜索范围
            bull_mom_weeks_range: bull_mom_weeks搜索范围
            bull_mom_thresh_range: bull_mom_thresh搜索范围
            target_vol_range: target_vol搜索范围
            metric: 优化目标 (sharpe/ann_ret/max_dd)

        返回: 最优参数字典
        """
        if con is None:
            con = duckdb.connect(str(DB), read_only=True)

        # 默认搜索范围
        if version_range is None:
            version_range = ["v6", "v6.1", "v6.2"]
        if threshold_range is None:
            threshold_range = [1, 2, 3]
        if bull_mom_weeks_range is None:
            bull_mom_weeks_range = [13, 20, 26]
        if bull_mom_thresh_range is None:
            bull_mom_thresh_range = [0.05, 0.10, 0.15]
        if target_vol_range is None:
            target_vol_range = [0.10, 0.12, 0.15]

        results = []

        for version in version_range:
            for threshold in threshold_range:
                for bmw in bull_mom_weeks_range:
                    for bmt in bull_mom_thresh_range:
                        for tv in target_vol_range:
                            try:
                                cfg = V6Config(
                                    version=version,
                                    threshold=threshold,
                                    bull_mom_weeks=bmw,
                                    bull_mom_thresh=bmt,
                                    target_vol=tv,
                                )
                                model = V6Model(cfg)
                                signal = model.build_signal(con)
                                r = model.backtest(signal)

                                results.append({
                                    "version": version,
                                    "threshold": threshold,
                                    "bull_mom_weeks": bmw,
                                    "bull_mom_thresh": bmt,
                                    "target_vol": tv,
                                    "ann_ret": r["ann_ret"] * 100,
                                    "sharpe": r["sharpe"],
                                    "max_dd": r["max_dd"] * 100,
                                    "win_rate": r["win_rate"] * 100,
                                })
                            except Exception:
                                pass

        df = pd.DataFrame(results)

        if metric == "sharpe":
            best_idx = df["sharpe"].idxmax()
        elif metric == "ann_ret":
            best_idx = df["ann_ret"].idxmax()
        elif metric == "max_dd":
            best_idx = df["max_dd"].idxmax()
        else:
            best_idx = df["sharpe"].idxmax()

        best = df.loc[best_idx]

        return {
            "version": best["version"],
            "threshold": int(best["threshold"]),
            "bull_mom_weeks": int(best["bull_mom_weeks"]),
            "bull_mom_thresh": best["bull_mom_thresh"],
            "target_vol": best["target_vol"],
            "ann_ret": best["ann_ret"],
            "sharpe": best["sharpe"],
            "max_dd": best["max_dd"],
            "results": df,
        }

    def __repr__(self) -> str:
        return (f"V6Model(version={self.config.version}, "
                f"threshold={self.config.threshold}, "
                f"bull_mom_weeks={self.config.bull_mom_weeks}, "
                f"target_vol={self.config.target_vol})")


__all__ = ["V6Model", "V6Config"]
