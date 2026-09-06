# coding: utf-8
"""Spanning Test vs 全部子策略 — P5 待实装.

论文 §6 关键结论:
  - Zookeeper 回归 → 单因子 (basis/mom/basis-mom/hp) 截距不显著
  - 反向: 单因子回归 Zookeeper → 年化 α 4.8%~6.0%

本项目对手 (用户决策 vs 全部):
  - §25 carry
  - §26 basis momentum / §26 vrp
  - ZN inventory / jjm_coke / CU premium TC / AU layered
  - OscAlpha (§19 已拒但保留对照)
  - 主 CTA baseline (E2-R + AU10%)

模型: target_alpha = α + β_zk · Zookeeper_factor + ε
  α 即 Zookeeper 无法解释的独立 alpha.
"""
from __future__ import annotations

import logging

import pandas as pd


LOG = logging.getLogger(__name__)


def spanning_regression(target_nav: pd.Series, zk_factor: pd.Series) -> dict:
    """单标的 Spanning Test.

    Returns:
        dict: alpha (年化) / beta / t_alpha / r_squared / decision
    """
    # TODO P5: OLS target_alpha = α + β·zk + ε → 年化 α / t 统计 / R²
    raise NotImplementedError("P5 待实装")


def run_all_spanning(zk_factor: pd.Series,
                     targets: dict[str, pd.Series]) -> pd.DataFrame:
    """对全部对手跑 Spanning Test.

    Args:
        zk_factor: Zookeeper 因子 NAV (pd.Series).
        targets: {name: nav_series} 子策略 + 主 CTA NAV 字典.

    Returns:
        DataFrame: name / alpha_ann / t_alpha / beta / r2 / decision
    """
    # TODO P5: 加载 reports/cta/*_nav.csv 各子策略 → spanning_regression → 汇总
    raise NotImplementedError("P5 待实装")


def decide(alpha_ann: float) -> str:
    """接入决策 (用户 §10.3 拍板)."""
    # TODO P5: < 2% 归档 / 2-5% 候选集成 / > 5% 谨慎审查
    raise NotImplementedError("P5 待实装")
