# coding=utf-8
"""v6.2 因子正交化 (Stage 27 v6.2).

v6.2 = v5.1.1 量价族 + IC 加权 + 因子正交化 (去除冗余).

设计动机:
- IC 诊断显示因子间高度相关: f8_pv_rankcov ↔ f9_pv_corr (0.78), f3_amt_vol ↔ f4_vol_vol (0.60)
- 高相关 → 权重"累加" → 实际像把这两个因子当 2 倍权重
- 正交化去除冗余, 让 IC 加权更纯净

算法选择:
- 残差化 (Gram-Schmidt): 给定顺序, 后续因子对前面做残差回归
  → 优点: 保留因子金融意义 (按 IC_IR 排序)
  → 缺点: 给定顺序不同时结果不同
- PCA 旋转: 数学最优, 但因子名解释力下降
  → 优点: 完全正交
  → 缺点: 解释不了第 N 主成分是哪个"原"因子

本模块采用残差化法, 顺序 = OOS IC_IR 降序 (含金融意义).

输入输出:
- 输入: factor_panel (dict[code] → DataFrame)
- 输出: factor_panel_orth (dict[code] → DataFrame)
- 算法:
  1. 跑一次 IC 时序, 得到 IR 排序
  2. 按 IR 降序, 对每个后续因子做残差化 (对前面的因子做截面回归, 取残差)
  3. 输出正交化后的因子 panel
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def get_factor_ir_order(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    factors: Sequence[str],
    horizon: int = 21,
    min_periods: int = 12,
) -> list[str]:
    """获取因子按 OOS IR 降序的排序 (用于正交化顺序).

    Returns:
        list of factor names (按 OOS IR 降序)
    """
    from ..v6_1.factor_weighting import compute_cross_section_ic

    records = {f: [] for f in factors}
    for d in rebalance_dates:
        ic = compute_cross_section_ic(factor_panel, panel_close, d, factors, horizon)
        for f in factors:
            if f in ic.index and not pd.isna(ic[f]):
                records[f].append(ic[f])

    ir = {}
    for f, vals in records.items():
        if len(vals) < min_periods:
            ir[f] = 0.0
            continue
        v = pd.Series(vals).dropna()
        m = v.mean()
        s = v.std()
        ir[f] = m / s if s > 0 else 0.0

    sorted_factors = sorted(ir.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_factors if ir[f] > 0]  # 只保留正向因子


def orthogonalize_factor_panel(
    factor_panel: dict[str, pd.DataFrame],
    factor_order: Sequence[str],
    rebalance_dates: Sequence[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """对因子 panel 做截面正交化 (按给定顺序, 残差化后续因子).

    算法:
    1. 按 factor_order 顺序处理因子
    2. 第 1 个因子保持原值
    3. 第 k 个因子 (k>1): 对每个截面日 d:
       - 取该日所有 ETF 在已处理因子上的值 (前面 k-1 个)
       - 用 OLS 回归: factor_k = Σ β_j × factor_j + residual
       - 取残差作为新因子值
    4. 返回正交化后的 panel

    Args:
        factor_panel: 原 panel (dict[code] → DataFrame)
        factor_order: 正交化顺序 (按 IR 降序)
        rebalance_dates: 用于计算残差的截面日 (subset)

    Returns:
        新 panel (dict[code] → DataFrame)
        仅返回 factor_order 中包含的因子 (其他因子不保留)
    """
    if not factor_order:
        return {}

    codes = sorted(factor_panel.keys())
    out = {code: pd.DataFrame(index=factor_panel[code].index) for code in codes}

    # 沿 IR 顺序处理
    processed = []  # 已正交化的因子名 (panel 输出列名)

    for k, fac in enumerate(factor_order):
        if k == 0:
            # 第 1 个因子保留原值
            for code in codes:
                if fac in factor_panel[code].columns:
                    out[code][fac] = factor_panel[code][fac]
            processed.append(fac)
            continue

        # k>1: 对每个截面日做残差化
        for code in codes:
            orig_fac_k = factor_panel[code].get(fac, pd.Series(dtype=float))
            if orig_fac_k.empty:
                continue

            # 对每个调仓日, 取该日的截面值
            new_vals = orig_fac_k.copy()

            for d in rebalance_dates:
                if d not in factor_panel[code].index:
                    continue
                # 取该 ETF 在 d 日, 前 k-1 个已处理因子的值
                X_row = []
                y_val = factor_panel[code][fac].loc[d] if fac in factor_panel[code].columns and d in factor_panel[code].index else np.nan
                if pd.isna(y_val):
                    continue
                for prev_fac in processed:
                    if prev_fac in factor_panel[code].columns and d in factor_panel[code].index:
                        v = factor_panel[code][prev_fac].loc[d]
                        X_row.append(v if pd.notna(v) else np.nan)

                if not X_row or any(pd.isna(v) for v in X_row):
                    # 无法回归, 保持原值
                    continue
                if len(X_row) != len(processed):
                    continue

                # 在多个 ETF 上做 OLS → 取该 ETF 的残差
                # 收集所有 ETF 在 d 日的 X / y
                Xs, ys = [], []
                for c2 in codes:
                    if d not in factor_panel[c2].index:
                        continue
                    if fac not in factor_panel[c2].columns:
                        continue
                    y2 = factor_panel[c2][fac].loc[d]
                    if pd.isna(y2):
                        continue
                    X2_row = []
                    valid = True
                    for prev_fac in processed:
                        if prev_fac in factor_panel[c2].columns and d in factor_panel[c2].index:
                            v2 = factor_panel[c2][prev_fac].loc[d]
                            if pd.isna(v2):
                                valid = False
                                break
                            X2_row.append(v2)
                    if valid and len(X2_row) == len(processed):
                        Xs.append(X2_row)
                        ys.append(y2)

                if len(Xs) < max(3, len(processed) + 1):
                    continue

                Xs = np.array(Xs)
                ys = np.array(ys)

                try:
                    # OLS: β = (X'X)^-1 X'y
                    XtX = Xs.T @ Xs
                    if np.linalg.det(XtX) < 1e-10:
                        # 退化情形跳过
                        continue
                    beta = np.linalg.solve(XtX, Xs.T @ ys)
                    # 该 ETF 的残差
                    pred = sum(X_row[i] * beta[i] for i in range(len(beta)))
                    resid = y_val - pred
                    new_vals.loc[d] = resid
                except Exception:
                    continue

            out[code][fac] = new_vals

        processed.append(fac)

    return out


__all__ = [
    "get_factor_ir_order",
    "orthogonalize_factor_panel",
]
