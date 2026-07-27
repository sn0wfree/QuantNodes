# coding=utf-8
"""v6.2 因子正交化 (Stage 27 v6.2, Stage 28/Phase1 look-ahead fix).

v6.2 = v5.1.1 量价族 + IC 加权 + 因子正交化 (去除冗余).

设计动机:
- IC 诊断显示因子间高度相关: f8_pv_rankcov ↔ f9_pv_corr (0.78), f3_amt_vol ↔ f4_vol_vol (0.60)
- 高相关 → 权重"累加" → 实际像把这两个因子当 2 倍权重
- 正交化去除冗余, 让 IC 加权更纯净

Gram-Schmidt 残差化数学:
  对每个截面日 d, 给定因子顺序 [f_1, f_2, ..., f_K]:
  f_1 (d) ← 原值
  f_k (d) ← f_k (d) - E[f_k | f_1, ..., f_{k-1}]  (OLS 残差)
  → 后续因子保留"独立于前面所有因子的增量信号".

[Phase 1 fix] 因子顺序策略由 4 种选 1:
  - "ir_expanding": 用截至 d_i 的 expanding IR 排序 (无 look-ahead, 默认)
  - "predefined": 永不变的金融预定义顺序 (Stage 28 试过, OOS 0.473 不理想)
  - "ir_full": 全样本 IR 排序 (DEPRECATED, 含 look-ahead, OOS 0.901 但有未来)
  - "symmetric_qr": QR 分解对称正交 (Phase 3 备选, 顺序无关)

输入输出:
- 输入: factor_panel (dict[code] → DataFrame)
- 输出: factor_panel_orth (dict[code] → DataFrame)
- 算法: 详见 orthogonalize_factor_panel / orthogonalize_factor_panel_qr
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


# ============================================================
# 预定义金融因子顺序 (Stage 28 试过但 OOS 不理想, 改用 expanding IR 默认)
# ============================================================
PREDEFINED_FACTOR_ORDER: tuple[str, ...] = (
    # ─ 动量族（基线, 不残差化） ─
    "f1_second_mom",       # 二阶动量
    "f2_mom_term",         # 动量期限差
    # ─ 反转族（残差化于动量） ─
    "f3_amt_vol",          # 成交金额波动 (反转)
    "f4_vol_vol",          # 成交量波动 (反转)
    # ─ 多空族（残差化于动量+反转） ─
    "f5_turnover",         # 换手率变化
    "f6_ls_total",         # 多空对比总量 (反转)
    "f7_ls_change",        # 多空对比变化
    # ─ 量价族（残差化于全部前面） ─
    "f8_pv_rankcov",       # 量价排序协方差 (量价)
    "f9_pv_corr",          # 量价相关系数 (量价)
    "f10_first_div",       # 一阶量价背离 (反转)
    "f11_vol_range",       # 量幅同向 (量价)
)
# 11 因子, 按"动量→反转→多空→量价"金融意义排序, 顺序永不变


# ============================================================
# 对外公开 API
# ============================================================
def get_factor_ir_order(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame = None,    # 仅保留接口兼容
    rebalance_dates: Sequence[pd.Timestamp] = None,
    factors: Sequence[str] = None,
    horizon: int = 21,
    min_periods: int = 12,
) -> list[str]:
    """[Phase 1 default] 返回预定义金融顺序, 不依赖任何数据.

    历史版本 (Stage 27) 曾用全样本 IR 排序 + 硬剔除 IR<=0 因子 (look-ahead).
    Stage 28 改为预定义金融顺序 (测试 OOS 0.473, 实际表现差).
    Phase 1 改为 expanding IR (Phase 1.1.1 实现), 此函数保留接口作 fallback.
    """
    if factors is None:
        return list(PREDEFINED_FACTOR_ORDER)
    seen = set(factors)
    return [f for f in PREDEFINED_FACTOR_ORDER if f in seen]


def get_factor_ir_order_warmup(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    factors: Sequence[str],
    horizon: int = 21,
    warmup_months: int = 24,
) -> list[str]:
    """[Phase 4 主推] 用早期 warmup 期 IR 序列算一次性固定顺序.

    算法:
      - 取 rebalance_dates 的前 warmup_months 个月 (w.r.t. 调仓日计数)
      - 算这些 warmup dates 上每个 factor 的 IC
      - 对每个 factor: IR = mean(IC) / std(IC)
      - 按 IR 降序排 (稳定的固定顺序, 后续所有调仓日共用)
      - 无剔除: IR<=0 因子也保留 (让 Gram-Schmidt 残差化"消化"它们)

    Args:
        factor_panel: 因子 panel
        panel_close: 收盘价
        rebalance_dates: 调仓日
        factors: 因子名列表
        horizon: IC 前瞻天数
        warmup_months: warmup 期月数 (默认 24)

    Returns:
        list[str]  按 IR 降序排, 固定长度 = len(factors)

    [Phase 4 关键] 这个函数只用 warmup 期 (调仓日第 1..warmup_months 个) 的 IC,
    不含 OOS 期或调仓日序列外的任何数据. 完全无 look-ahead.
    """
    from .factor_weighting import compute_cross_section_ic

    rebal_arr = list(rebalance_dates)
    warmup_window = rebal_arr[:warmup_months]
    if len(warmup_window) < 3:
        return list(factors)

    ic_records: dict[str, list[float]] = {f: [] for f in factors}
    for d in warmup_window:
        try:
            ic = compute_cross_section_ic(
                factor_panel, panel_close, d, factors, horizon,
            )
            for f in factors:
                if f in ic.index and not pd.isna(ic[f]):
                    ic_records[f].append(float(ic[f]))
        except Exception:
            continue

    ir: dict[str, float] = {}
    for f, vals in ic_records.items():
        if len(vals) < 3:
            ir[f] = 0.0
            continue
        v = pd.Series(vals).dropna()
        m = v.mean()
        s = v.std()
        ir[f] = (m / s) if s > 0 else 0.0

    sorted_factors = sorted(factors, key=lambda f: ir.get(f, 0.0), reverse=True)
    return sorted_factors


def get_factor_ir_order_expanding(
    factor_panel: dict[str, pd.DataFrame],
    panel_close: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    factors: Sequence[str],
    horizon: int = 21,
    min_periods: int = 12,
    lookback_months: int = 36,
) -> dict[pd.Timestamp, list[str]]:
    """[Phase 1 主推] 每个调仓日 d_i 用截至 d_{i-1} 的 expanding IR 排序.

    对每个调仓日 d_i:
      - 取过去 lookback_months 月 (含 i) 的 rebalance_dates
      - 对每个 past rebalance_date d_j (j < i) 算 IC
      - 算每个因子的 IR (mean/std)
      - 按 IR 降序排 (无剔除, IR<=0 因子也排, 仅影响顺序)

    Args:
        factor_panel: 因子 panel
        panel_close: 收盘价
        rebalance_dates: 调仓日 (升序)
        factors: 因子名列表
        horizon: IC 收益前瞻天数 (默认 21 = 1月)
        min_periods: 最少需要过去多少个月 IC 才算 IR (默认 12)
        lookback_months: 用最近几个月 IC (默认 36, expanding)

    Returns:
        dict[d_i] -> list[str]  按 IR 降序排的因子名
        d_i 早于 min_periods 个月时, 返回预设顺序的备份 (防止冷启动退化)

    [Phase 1 关键] 调仓日 d_i 的 IC 用 d_{i-1} 及之前的 past rebalance_dates,
    不含 d_i 之后的. 完全无 look-ahead.
    """
    from .factor_weighting import compute_cross_section_ic

    out: dict[pd.Timestamp, list[str]] = {}
    rebal_arr = list(rebalance_dates)

    for i, d_curr in enumerate(rebal_arr):
        # 取过去 lookback_months 月 (含 i, 但不含 d_curr 自己)
        past_window = rebal_arr[max(0, i - lookback_months):i]
        if len(past_window) < min_periods:
            # 数据不足, fallback 用 factors 原顺序 (而不是 PREDEFINED,
            # 因为 PREDEFINED_FACTOR_ORDER 可能不含全部 factors)
            out[d_curr] = list(factors)
            continue

        # 收集 past_window 中每个 past_date 的 IC
        ic_records: dict[str, list[float]] = {f: [] for f in factors}
        for d_past in past_window:
            try:
                ic = compute_cross_section_ic(
                    factor_panel, panel_close, d_past, factors, horizon,
                )
                for f in factors:
                    if f in ic.index and not pd.isna(ic[f]):
                        ic_records[f].append(float(ic[f]))
            except Exception:
                continue

        # 算 IR
        ir: dict[str, float] = {}
        for f, vals in ic_records.items():
            if len(vals) < 3:
                ir[f] = 0.0
                continue
            v = pd.Series(vals).dropna()
            m = v.mean()
            s = v.std()
            ir[f] = (m / s) if s > 0 else 0.0

        # 降序排, 缺失/0 因子排末尾
        sorted_factors = sorted(factors, key=lambda f: ir.get(f, 0.0), reverse=True)
        out[d_curr] = sorted_factors

    return out


# ============================================================
# 正交化核心: Gram-Schmidt 残差化 (per-order, 支持每期不同顺序)
# ============================================================
def orthogonalize_factor_panel(
    factor_panel: dict[str, pd.DataFrame],
    factor_order: Sequence[str],
    rebalance_dates: Sequence[pd.Timestamp],
    order_per_date: dict[pd.Timestamp, Sequence[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """对因子 panel 做截面正交化 (按给定顺序, 残差化后续因子).

    Args:
        factor_panel: 原 panel (dict[code] → DataFrame)
        factor_order: 默认正交化顺序 (若 order_per_date 为 None, 所有截面日共用)
        rebalance_dates: 用于计算残差的截面日 (subset)
        order_per_date: [Phase 1 新增] 可选, {d -> order}, 每个截面日单独用不同顺序
                          (用于 expanding IR 路线, 每调仓日 d_i 用截至 d_{i-1} 的 IR 排序)

    Returns:
        新 panel (dict[code] → DataFrame)
        仅返回 factor_order 中包含的因子 (其他因子不保留)
    """
    if not factor_order:
        return {}

    codes = sorted(factor_panel.keys())
    out = {code: pd.DataFrame(index=factor_panel[code].index) for code in codes}

    # [Phase 4 修复] sorted(keys())[0] 可能是后期上市的新 ETF (e.g. 159740 上 2021-05),
    # 它的 index 不含早期日期, 导致 d in codes[0].index 失败而跳过整个 rebal date.
    # 改为: 只在 "至少有一个 ETF 在 d 当天有 factor_panel 数据" 时才处理 d.

    for d in rebalance_dates:
        has_any = False
        for c_check in codes:
            if d in factor_panel[c_check].index:
                has_any = True
                break
        if not has_any:
            continue
        order_at_d = order_per_date.get(d) if order_per_date else factor_order
        order_at_d = [f for f in order_at_d if f in factor_order]  # 确保都在原 order 中
        if not order_at_d:
            continue

        # 处理当期顺序的每个因子
        for k, fac in enumerate(order_at_d):
            if k == 0:
                # 第 1 个因子保持原值
                for code in codes:
                    if fac in factor_panel[code].columns and d in factor_panel[code].index:
                        v = factor_panel[code][fac].loc[d]
                        if pd.notna(v):
                            out[code].loc[d, fac] = v
                continue

            # k>1: 在 d 日截面上, 取所有 ETF 的 (y, X) 做 OLS
            Xs, ys = [], []
            valid_codes = []
            for c2 in codes:
                if d not in factor_panel[c2].index:
                    continue
                if fac not in factor_panel[c2].columns:
                    continue
                y_val = factor_panel[c2][fac].loc[d]
                if pd.isna(y_val):
                    continue
                # 前 k-1 个 (来自当期 order) 的截面值
                X_row = []
                ok = True
                for prev_fac in order_at_d[:k]:
                    if prev_fac not in factor_panel[c2].columns:
                        ok = False
                        break
                    v = factor_panel[c2][prev_fac].loc[d]
                    if pd.isna(v):
                        ok = False
                        break
                    X_row.append(float(v))
                if not ok:
                    continue
                Xs.append(X_row)
                ys.append(float(y_val))
                valid_codes.append(c2)

            if len(Xs) < max(3, k + 1):
                # 样本不足, 保持原值
                for code in codes:
                    if fac in factor_panel[code].columns and d in factor_panel[code].index:
                        v = factor_panel[code][fac].loc[d]
                        if pd.notna(v):
                            out[code].loc[d, fac] = v
                continue

            Xs_arr = np.array(Xs)
            ys_arr = np.array(ys)
            try:
                XtX = Xs_arr.T @ Xs_arr
                cond = np.linalg.cond(XtX)
                if cond > 1e10:
                    # 退化, 保持原值
                    for code in codes:
                        if fac in factor_panel[code].columns and d in factor_panel[code].index:
                            v = factor_panel[code][fac].loc[d]
                            if pd.notna(v):
                                out[code].loc[d, fac] = v
                    continue
                beta = np.linalg.solve(XtX, Xs_arr.T @ ys_arr)
                # 给所有 valid_codes 写残差
                for c2, X_row in zip(valid_codes, Xs):
                    pred = sum(X_row[i] * beta[i] for i in range(len(beta)))
                    y_val = factor_panel[c2][fac].loc[d]
                    resid = y_val - pred
                    out[code := c2].loc[d, fac] = resid  # noqa: E501
            except Exception:
                for code in codes:
                    if fac in factor_panel[code].columns and d in factor_panel[code].index:
                        v = factor_panel[code][fac].loc[d]
                        if pd.notna(v):
                            out[code].loc[d, fac] = v

    # [Phase 4 修复] 残差值仅在 rebalance_date 处写入.
    # 调用者 (compute_composite_factor / cross_section_zscore) 在调仓日 as_of 不在
    # df.index 时会 ffill 到最近的非 NaN 日. 但 as_of 之前若所有日子都 NaN (例如
    # 第一调仓日之前), ffill 找不到前值; 即使之后遇到, 仅 ffill 到 as_of 当天前
    # 最近的非 NaN 日, 可能跨越多个调仓区间.
    # 现在用 ffill 沿每因子列补全: 在最近调仓日的残差值视为
    # "有效, 持续到下个调仓日", ffill 让所有期间都有值.
    for code in codes:
        for col in list(out[code].columns):
            out[code][col] = out[code][col].ffill()

    return out


# ============================================================
# 正交化备选: QR 分解对称正交 (Phase 3 fallback, 顺序无关)
# ============================================================
def orthogonalize_factor_panel_qr(
    factor_panel: dict[str, pd.DataFrame],
    rebalance_dates: Sequence[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """[Phase 3 fallback] 对称正交化: 对每个截面日 d 做 QR 分解.

    算法:
      对每个 d, 取所有 ETF 在 d 日的因子值矩阵 H (N × K)
      做列中心化: H_center = H - mean(H)
      QR: H_center = Q @ R
      Q 矩阵列正交, 用 Q 替换原 H 作为正交化后的因子值

    数学性质:
      Q 列正交: Qᵀ @ Q = I_K (K=11)
      完全顺序无关, 没有 Gram-Schmidt 残差化的信号损失问题

    缺点:
      因子失去原始金融含义 (Q 不是原因子的线性无关化, 而是旋转到正交方向)
      后续 IC 计算可能不稳定 (因子的物理意义变了)

    Args:
        factor_panel: 原 panel (dict[code] → DataFrame)
        rebalance_dates: 截面日

    Returns:
        新 panel (与 factor_panel 同结构, 列名 f_qr_0, f_qr_1, ..., f_qr_K-1)
    """
    codes = sorted(factor_panel.keys())
    out = {code: pd.DataFrame(index=factor_panel[code].index) for code in codes}

    # 取所有因子的列名 (假设各 code 的因子列一致)
    first_df = next(iter(factor_panel.values()))
    factor_cols = [c for c in first_df.columns]

    for d in rebalance_dates:
        if d not in first_df.index:
            continue

        # 收集 d 日所有 ETF 的因子矩阵 H (N × K)
        rows = []
        valid_codes = []
        for c in codes:
            if d not in factor_panel[c].index:
                continue
            row = []
            ok = True
            for fac in factor_cols:
                if fac not in factor_panel[c].columns:
                    ok = False
                    break
                v = factor_panel[c][fac].loc[d]
                if pd.isna(v):
                    ok = False
                    break
                row.append(float(v))
            if ok and len(row) == len(factor_cols):
                rows.append(row)
                valid_codes.append(c)

        if len(rows) < len(factor_cols) + 3:
            # 样本不足, 跳过
            continue

        H = np.array(rows)  # N × K
        K = H.shape[1]

        # 中心化
        H_center = H - H.mean(axis=0, keepdims=True)

        try:
            Q, R = np.linalg.qr(H_center, mode="reduced")
        except Exception:
            continue

        # Q 是 N × K 矩阵, 列正交
        # 给每个 valid_code 写 K 个新因子 f_qr_0, ..., f_qr_K-1
        for j, c in enumerate(valid_codes):
            for k in range(K):
                col_name = f"f_qr_{k}"
                out[c].loc[d, col_name] = float(Q[j, k])

    return out


__all__ = [
    "PREDEFINED_FACTOR_ORDER",
    "get_factor_ir_order",
    "get_factor_ir_order_expanding",
    "get_factor_ir_order_warmup",
    "orthogonalize_factor_panel",
    "orthogonalize_factor_panel_qr",
]
