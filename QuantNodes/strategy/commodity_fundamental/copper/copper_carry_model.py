# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_carry_model.py

Carry策略模型类

将carry策略封装为模型类，支持:
    - 参数配置
    - 信号生成
    - 回测
    - 参数优化

用法:
    model = CarryModel(variant="state_mom_curv2", z_window=26, mom_window=4)
    signal = model.build_signal(con)
    result = model.backtest(signal)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .copper_carry import (
    load_term_structure,
    compute_carry_signal,
    load_lme_inventory,
    backtest_carry,
)
from ..sign_constrained_lasso import rolling_zscore

DB = Path("data/cmf/fundamental.duckdb")


@dataclass
class CarryConfig:
    """Carry模型配置."""
    variant: str = "state_mom_curv2"
    z_window: int = 26
    min_duration: int = 6
    mom_window: int = 4
    cost_bps: float = 5.0

    # 状态分类阈值
    threshold_back: float = -0.5   # backwardation阈值
    threshold_contango: float = 0.5  # contango阈值


class CarryModel:
    """Carry策略模型.

    支持多种变体:
        - state: 纯carry状态
        - state_mom: carry + 动量
        - state_mom_curv2: carry + 动量 + 曲率 (默认)
        - state_mom_inv: carry + 动量 + 库存
    """

    def __init__(self, config: CarryConfig = None, **kwargs):
        """初始化模型.

        参数:
            config: CarryConfig配置对象
            **kwargs: 覆盖config的参数
        """
        if config is None:
            config = CarryConfig(**kwargs)
        elif kwargs:
            # 覆盖config中的参数
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

    def build_signal(self, con=None) -> pd.DataFrame:
        """构建信号.

        参数:
            con: 数据库连接

        返回: DataFrame with columns:
            - M: 周频收盘价
            - position: 仓位 [0, 1]
            - carry_slope: 期限结构斜率
            - carry_zscore: 期限结构z-score
            - carry_state: 状态 (+1/0/-1)
        """
        if con is None:
            con = duckdb.connect(str(DB), read_only=True)

        cfg = self.config

        # 加载期限结构
        ts = load_term_structure(con)
        sig = compute_carry_signal(
            ts,
            z_window=cfg.z_window,
            min_duration=cfg.min_duration,
        )

        # 加载价格
        close_near = ts["close_near"]
        close_w = close_near.resample("W-FRI").last()

        # 计算动量
        mom = close_near.pct_change(cfg.mom_window).rename("mom")

        # 合并信号
        df = sig.join(mom)

        # 库存增强变体需要库存数据
        if cfg.variant in ("state_mom_inv",):
            inv_df = load_lme_inventory(con)
            inv_change = inv_df["inv_change_4w"].reindex(df.index).fillna(0)
            inv_zscore = rolling_zscore(-inv_change, window=cfg.z_window).rename("inv_zscore")
            df = df.join(inv_zscore)

        # 生成仓位
        pos = self._compute_position(df, cfg.variant)

        # 构建输出
        out = pd.DataFrame({
            "M": close_w.reindex(df.index),
            "position": pos,
            "carry_slope": df["carry_slope"],
            "carry_zscore": df["carry_zscore"],
            "carry_state": df["carry_state"],
        })

        return out

    def _compute_position(self, df: pd.DataFrame, variant: str) -> pd.Series:
        """计算仓位."""
        pos = pd.Series(0.0, index=df.index)

        if variant == "state":
            pos[df["carry_state"] == 1.0] = 1.0

        elif variant == "state_mom":
            back = df["carry_state"] == 1.0
            pos[back & (df["mom"] > 0)] = 1.0
            pos[back & (df["mom"] <= 0)] = 0.5

        elif variant == "state_mom_curv":
            back = df["carry_state"] == 1.0
            curv_pos = df["curvature_zscore"] > 0
            mom_up = df["mom"] > 0
            agree3 = back & curv_pos & mom_up
            agree2 = back & ((curv_pos & ~mom_up) | (~curv_pos & mom_up))
            pos[agree3] = 1.0
            pos[agree2 & ~agree3] = 0.5

        elif variant == "state_mom_curv2":
            back = df["carry_state"] == 1.0
            mom_up = df["mom"] > 0
            curv = df["curvature_zscore"].clip(-1, 1)
            base = pd.Series(0.0, index=df.index)
            base[back & mom_up] = 1.0
            base[back & ~mom_up] = 0.5
            adj = 1.0 + curv * 0.3
            pos = (base * adj).clip(0, 1)

        elif variant == "state_mom_inv":
            back = df["carry_state"] == 1.0
            mom_up = df["mom"] > 0
            inv_low = df["inv_zscore"] < -0.5
            pos[back & mom_up & inv_low] = 1.0
            pos[back & mom_up & ~inv_low] = 0.7
            pos[back & ~mom_up & inv_low] = 0.5
            pos[back & ~mom_up & ~inv_low] = 0.3

        elif variant == "continuous":
            pos = (-df["carry_zscore"]).clip(lower=0.0, upper=1.0).fillna(0.0)

        return pos

    def backtest(self, signal: pd.DataFrame) -> dict:
        """回测.

        参数:
            signal: build_signal输出

        返回: 回测结果字典
        """
        return backtest_carry(signal, cost_bps=self.config.cost_bps)

    def optimize(self, con=None,
                 z_window_range: list = None,
                 mom_window_range: list = None,
                 min_duration_range: list = None,
                 metric: str = "sharpe") -> dict:
        """参数优化 (网格搜索).

        参数:
            con: 数据库连接
            z_window_range: z_window搜索范围
            mom_window_range: mom_window搜索范围
            min_duration_range: min_duration搜索范围
            metric: 优化目标 (sharpe/ann_ret/max_dd)

        返回: 最优参数字典
        """
        if con is None:
            con = duckdb.connect(str(DB), read_only=True)

        if z_window_range is None:
            z_window_range = [26, 39, 52, 65, 78]
        if mom_window_range is None:
            mom_window_range = [4, 8, 13, 20, 26]
        if min_duration_range is None:
            min_duration_range = [1, 2, 4, 6, 8]

        results = []

        for zw in z_window_range:
            for mw in mom_window_range:
                for md in min_duration_range:
                    try:
                        cfg = CarryConfig(
                            variant=self.config.variant,
                            z_window=zw,
                            mom_window=mw,
                            min_duration=md,
                        )
                        model = CarryModel(cfg)
                        signal = model.build_signal(con)
                        r = model.backtest(signal)

                        results.append({
                            "z_window": zw,
                            "mom_window": mw,
                            "min_duration": md,
                            "ann_ret": r["ann_ret"] * 100,
                            "sharpe": r["sharpe"],
                            "max_dd": r["max_dd"] * 100,
                            "win_rate": r["win_rate"] * 100,
                            "pct_active": r["pct_active"],
                        })
                    except Exception:
                        pass

        df = pd.DataFrame(results)

        if metric == "sharpe":
            best_idx = df["sharpe"].idxmax()
        elif metric == "ann_ret":
            best_idx = df["ann_ret"].idxmax()
        elif metric == "max_dd":
            best_idx = df["max_dd"].idxmax()  # max_dd是负数，越大越好
        else:
            best_idx = df["sharpe"].idxmax()

        best = df.loc[best_idx]

        return {
            "z_window": int(best["z_window"]),
            "mom_window": int(best["mom_window"]),
            "min_duration": int(best["min_duration"]),
            "ann_ret": best["ann_ret"],
            "sharpe": best["sharpe"],
            "max_dd": best["max_dd"],
            "results": df,
        }

    def __repr__(self) -> str:
        return (f"CarryModel(variant={self.config.variant}, "
                f"z_window={self.config.z_window}, "
                f"mom_window={self.config.mom_window}, "
                f"min_duration={self.config.min_duration})")


__all__ = ["CarryModel", "CarryConfig"]
