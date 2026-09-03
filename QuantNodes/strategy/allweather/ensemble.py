# coding=utf-8
"""AllWeather Phase 5 — 双/三 sleeve 集成 (PLAN §十 架构决策).

Sleeve 定义:
    A  X2-L4   cap0.40 HRP+RMT, 杠杆≤4   (债牛友好, 结构 α)
    B  EW-vt   等权 + vol target cap1    (风险资产牛市参与, 压力段正)
    C  CTA     CTA v7 信号 × 29 期货     (long-short α, gross≤1, 独立于 β)

集成: 策略层滚动逆波动率加权 (Phase 18 ENSEMBLE_INV_VOL 形态, 因果 shift 1).

无前视: 权重[t] 用各 sleeve 截至 t-1 的收益波动.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import AllWeatherConfig, run_backtest
from .config import (
    FUTURES_COMMISSION_RATE,
    OUTPUT_DIR,
    PANEL_V2_OUTPUT,
)
from .run_replication import SEGMENTS

CTA_SIGNALS_PATH = OUTPUT_DIR / "cta_signals_v1.parquet"
ANN = 252.0


# ── Sleeve A / B: 引擎直接产出 ──────────────────────────────────────
def sleeve_a(panel: pd.DataFrame, max_leverage: int = 4):
    """X2-L4: cap0.40 HRP+RMT 无趋势, 杠杆≤4."""
    cfg = AllWeatherConfig(trend_mode="none", max_weight=0.40,
                           max_leverage=max_leverage)
    return run_backtest(panel, cfg, compute_metrics_fn=_noop)


def sleeve_b(panel: pd.DataFrame, max_leverage: int = 1):
    """EW-vt: 等权 + vol target cap1 (风险资产参与 sleeve)."""
    cfg = AllWeatherConfig(use_hrp=False, trend_mode="none",
                           use_vol_target=True, max_leverage=max_leverage)
    return run_backtest(panel, cfg, compute_metrics_fn=_noop)


# ── Sleeve C: CTA 信号 sleeve ───────────────────────────────────────
def cta_sleeve_returns(panel: pd.DataFrame,
                       signals_path: Path = CTA_SIGNALS_PATH,
                       momentum_filter_window: int = 60,
                       position_ema_alpha: float = 0.0,
                       min_hold_days: int = 1) -> tuple[pd.Series, pd.Series]:
    """CTA v7 信号 → (日净收益, 每日换手).

    信号 (T×N, -1/0/+1) 于 t 收盘生成, 赚 t→t+1 收益:
        w_i(t) = sign_i(t) / N_active(t)
        contrib(t+1) = Σ w_i(t) · r_i(t+1)
        cost(t+1) = Σ |Δw_i| · rate

    position_ema_alpha: 0=原状 (sign→权重的瞬时切换, 高换手);
        α>0 时, 权重在日间 EMA 平滑 → w_ema(t) = α·w_raw(t) + (1-α)·w_ema(t-1),
        减少 signal 反转触发的 turnover (CTA 模块 _positions() 内部亦用此机制).

    min_hold_days: 1=每日调仓; >1 时仅在每 N 天的调仓日重置权重 (其余日期
        持仓不变), 大幅降低换手. 经验上 5 ≈ 周频, 兼具信号敏感度.
    """
    if Path(signals_path).exists():
        sig = pd.read_parquet(signals_path)
    else:
        sig = _extract_cta_signals(panel, momentum_filter_window)
    sig.index = pd.to_datetime(sig.index)

    ret = panel.pct_change(fill_method=None)
    fut_cols = [c for c in panel.columns if c.startswith("fut_")]
    sig = sig.reindex(panel.index).ffill()
    sig = sig.rename(columns={c: f"fut_{c.lower()}" for c in sig.columns})
    sig = sig[[c for c in sig.columns if c in fut_cols]]

    n_active = (sig != 0).sum(axis=1).clip(lower=1)
    w_raw = sig.div(n_active, axis=0).fillna(0.0)

    # 应用调仓频率 (min_hold_days)
    if min_hold_days > 1:
        # 仅在调仓日 (每 N 天) 重置为目标权重, 中间持仓不变
        rb_mask = pd.Series(0, index=w_raw.index)
        rb_mask.iloc[::min_hold_days] = 1  # 第 0, N, 2N... 天调仓
        rb_mask = rb_mask.astype(bool)
        w_raw = w_raw.where(rb_mask.reindex(w_raw.index, fill_value=False)).ffill().fillna(0.0)

    if position_ema_alpha > 0:
        a = position_ema_alpha
        w = w_raw.ewm(alpha=a, adjust=False).mean()
    else:
        w = w_raw

    contrib = (w.shift(1) * ret[fut_cols].fillna(0.0)).sum(axis=1)
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    rate = FUTURES_COMMISSION_RATE
    out = (contrib - dw * rate).dropna()
    out.name = "cta_sleeve"
    return out, dw.rename("cta_turnover")


def _extract_cta_signals(panel: pd.DataFrame,
                         momentum_filter_window: int = 60) -> pd.DataFrame:
    """调用 CTA v7 模块提取信号面板 (慢, 结果缓存到 parquet)."""
    from QuantNodes.strategy.cta.strategies.factor_modules import (
        ComboConfig,
        FactorModuleComboStrategy,
    )
    prods = sorted({c.replace("fut_", "").upper()
                    for c in panel.columns if c.startswith("fut_")})
    combo = FactorModuleComboStrategy(
        products=prods, start=str(panel.index.min().date()),
        end=str(panel.index.max().date()),
        config=ComboConfig(momentum_filter_window=momentum_filter_window,
                           momentum_filter_coef=0.0))
    combo.load_panel()
    sig = combo._signals()
    sig.index = pd.to_datetime(sig.index)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(CTA_SIGNALS_PATH)
    return sig


def _noop(nav, freq=None, oos_start=None) -> dict:
    return {}


# ── 集成: 滚动逆波动率 ──────────────────────────────────────────────
def rolling_invvol_combine(navs: dict[str, pd.Series],
                           lookback: int = 60) -> pd.Series:
    """多 sleeve NAV → 滚动 inv-vol 加权组合 NAV (因果: 权重用截至 t-1 波动)."""
    rets = pd.DataFrame({k: v.pct_change().fillna(0.0) for k, v in navs.items()})
    vol = rets.rolling(lookback, min_periods=lookback).std().shift(1)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(1.0 / len(navs))
    port = (rets * w).sum(axis=1)
    out = (1 + port).cumprod()
    out.name = "ensemble"
    return out


def equal_combine(navs: dict[str, pd.Series]) -> pd.Series:
    """等权静态组合 (对照)."""
    rets = pd.DataFrame({k: v.pct_change().fillna(0.0) for k, v in navs.items()})
    return (1 + rets.mean(axis=1)).cumprod().rename("ensemble_ew")


SEG25 = SEGMENTS["stress_25+"]


def _metrics_fn():
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
        compute_metrics,
    )
    return compute_metrics


# ── Phase 5 编排 ────────────────────────────────────────────────────
def run_phase5(panel: pd.DataFrame, include_cta: bool = True,
               lookback: int = 60, cta_alpha: float = 0.0,
               a_cap: int = 4, b_cap: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建各 sleeve + 集成组合, 返回 (指标长表, 各 nav 的 DataFrame).

    a_cap / b_cap: A/B sleeve 的 max_leverage (T2 — 调控到文章 avg 2.3x).
    cta_alpha: position_ema_alpha (T1 — 降 CTA sleeve 换手).
    """
    mm = _metrics_fn()
    res_a = sleeve_a(panel, max_leverage=a_cap)
    res_b = sleeve_b(panel, max_leverage=b_cap)
    navs = {
        "A_X2L4": res_a.nav,
        "B_EWvt": res_b.nav,
    }
    if include_cta:
        cta_ret, _ = cta_sleeve_returns(panel, position_ema_alpha=cta_alpha)
        nav_cta = (1 + cta_ret).cumprod()
        navs["C_CTA"] = nav_cta

    combos: dict[str, pd.Series] = {"AB_invvol": rolling_invvol_combine(
        {"A": navs["A_X2L4"], "B": navs["B_EWvt"]}, lookback)}
    if include_cta:
        combos["AC_invvol"] = rolling_invvol_combine(
            {"A": navs["A_X2L4"], "C": navs["C_CTA"]}, lookback)
        combos["ABC_invvol"] = rolling_invvol_combine(navs, lookback)
        combos["ABC_ew"] = equal_combine(navs)

    all_navs = {**navs, **combos}
    rows = []
    for name, nav in all_navs.items():
        m = mm(nav.dropna(), freq="D")
        row = {"config": name}
        for k in ("Sharpe", "AnnRet", "Vol", "MaxDD", "WinRate", "Calmar"):
            row[k] = m.get(k)
        s25 = nav.loc[SEG25[0]:SEG25[1]]
        if len(s25) > 30:
            row["stress_Sharpe"] = mm(s25, freq="D")["Sharpe"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("config")
    return df, pd.DataFrame(all_navs)


# ── 成本分解: ABC_ew 换手/成本细算 ─────────────────────────────────
def analyze_costs(panel: pd.DataFrame, lookback: int = 60,
                 blended_rate: float = 0.0002,
                 cta_alpha: float = 0.0,
                 a_cap: int = 4, b_cap: int = 1) -> pd.DataFrame:
    """ABC_ew 三 sleeve 换手/成本细算.

    cta_alpha: 传入 cta_sleeve_returns 平滑 CTA 权重
    a_cap/b_cap: 调控 sleeve 杠杆 (T2: 整体 avg 杠杆 ≈ 2.3x)
    """
    res_a = sleeve_a(panel, max_leverage=a_cap)
    res_b = sleeve_b(panel, max_leverage=b_cap)
    cta_ret, cta_dw = cta_sleeve_returns(panel, position_ema_alpha=cta_alpha)
    mm = _metrics_fn()

    def ann_turn(t):
        t = t.dropna()
        return float(t.sum() / max(len(t), 1) * 252) if len(t) else 0.0

    ta, tb, tc = ann_turn(res_a.turnover), ann_turn(res_b.turnover), ann_turn(cta_dw)
    cost_a, cost_b = ta * blended_rate, tb * blended_rate
    cost_c = tc * FUTURES_COMMISSION_RATE    # CTA 已内部按 1bp 扣除
    cta_gross_ret = cta_ret + cta_dw * FUTURES_COMMISSION_RATE   # 反推 CTA 无成本

    navs_net = {"A": res_a.nav, "B": res_b.nav, "C": (1 + cta_ret).cumprod()}
    navs_gross = {
        "A": (1 + res_a.nav.pct_change().fillna(0).add(cost_a / 252)).cumprod(),
        "B": (1 + res_b.nav.pct_change().fillna(0).add(cost_b / 252)).cumprod(),
        "C": (1 + cta_gross_ret).cumprod(),
    }

    rows = []
    for s in ("A", "B", "C"):
        for tag, ns in (("net", navs_net), ("gross", navs_gross)):
            m = mm(ns[s].dropna(), freq="D")
            rows.append({"sleeve": s, "cost_basis": tag,
                         "Sharpe": m["Sharpe"], "AnnRet": m["AnnRet"],
                         "Vol": m["Vol"], "MaxDD": m["MaxDD"]})
    # portfolio ABC_ew net vs gross
    rets_net = pd.DataFrame({k: v.pct_change().fillna(0) for k, v in navs_net.items()})
    rets_gross = pd.DataFrame({k: v.pct_change().fillna(0) for k, v in navs_gross.items()})
    nav_abc_net = (1 + rets_net.mean(axis=1)).cumprod()
    nav_abc_gross = (1 + rets_gross.mean(axis=1)).cumprod()
    for tag, nav in (("net", nav_abc_net), ("gross", nav_abc_gross)):
        m = mm(nav.dropna(), freq="D")
        rows.append({"sleeve": "ABC_ew", "cost_basis": tag,
                     "Sharpe": m["Sharpe"], "AnnRet": m["AnnRet"],
                     "Vol": m["Vol"], "MaxDD": m["MaxDD"]})
    df = pd.DataFrame(rows)

    # turn/cost 表 (单 sleeve + 组合汇总)
    turn_cost = pd.DataFrame([
        {"scope": "Sleeve A (HRP+RMT, lev≤4)", "turn_ann": ta, "cost_rate": blended_rate,
         "cost_ann_pct": cost_a * 100},
        {"scope": "Sleeve B (EW + vt cap1)", "turn_ann": tb, "cost_rate": blended_rate,
         "cost_ann_pct": cost_b * 100},
        {"scope": "Sleeve C (CTA signal, gross<=1)", "turn_ann": tc,
         "cost_rate": FUTURES_COMMISSION_RATE, "cost_ann_pct": cost_c * 100},
        {"scope": "ABC_ew portfolio (1/3 each)", "turn_ann": (ta + tb + tc) / 3,
         "cost_rate": "blended", "cost_ann_pct": (cost_a + cost_b + cost_c) / 3 * 100},
    ])
    return df, turn_cost


# ── T1+T2 优化扫描: CTA 换手降 + 整体 avg 杠杆 2.3x ─────────────────
def phase5_optimize(panel: pd.DataFrame,
                    cta_alphas: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.30, 0.50),
                    a_caps: tuple[int, ...] = (3, 4, 5, 6, 7),
                    b_caps: tuple[int, ...] = (1, 2, 3),
                    include_b: bool = True) -> pd.DataFrame:
    """CTA α × A cap × B cap 三维扫描 → 输出 ABC_ew 全样本 + 压力段 Sharpe/MaxDD/换手/成本.

    目标: 选 Sharpe≥1.7 + AvgLeverage≈2.3x + 换手<80x + 压力段为正.
    """
    mm = _metrics_fn()
    # 先扫描 α (CTA 单独看, 只需算 CTA sleeve)
    cta_results = []
    for a in cta_alphas:
        ret, dw = cta_sleeve_returns(panel, position_ema_alpha=a)
        nav = (1 + ret).cumprod()
        m = mm(nav.dropna(), freq="D")
        turn = float(dw.sum() / max(len(dw.dropna()), 1) * 252)
        s25 = mm(nav.loc[SEG25[0]:SEG25[1]].dropna(), freq="D")
        cta_results.append({"alpha": a, "turn_ann": turn,
                            "Sharpe": m["Sharpe"], "AnnRet": m["AnnRet"],
                            "stress_Sharpe": s25["Sharpe"]})

    # 三维 (α, a_cap, b_cap) 扫描 ABC_ew (含 B 与否)
    combo_rows = []
    for a in cta_alphas:
        for ac in a_caps:
            bcs = b_caps if include_b else (1,)
            for bc in bcs:
                cfg_a = dict(trend_mode="none", max_weight=0.40, max_leverage=ac)
                cfg_b = dict(use_hrp=False, trend_mode="none",
                             use_vol_target=True, max_leverage=bc)
                res_a = run_backtest(panel, AllWeatherConfig(**cfg_a),
                                     compute_metrics_fn=_noop)
                res_b = run_backtest(panel, AllWeatherConfig(**cfg_b),
                                     compute_metrics_fn=_noop)
                cta_ret, cta_dw = cta_sleeve_returns(panel, position_ema_alpha=a)
                nav_a, nav_b = res_a.nav, res_b.nav
                nav_c = (1 + cta_ret).cumprod()
                abc_nav = equal_combine({"A": nav_a, "B": nav_b, "C": nav_c})

                # 平均杠杆 (按 sleeve 的平均 |sum(w)|)
                avg_lev = (res_a.weights.sum(axis=1).mean()
                           + res_b.weights.sum(axis=1).mean()
                           + 1.0) / 3  # CTA gross≤1
                turn = float((res_a.turnover.dropna().sum()
                             + res_b.turnover.dropna().sum()
                             + cta_dw.dropna().sum()) / max(len(cta_dw.dropna()), 1) * 252 / 3)

                m = mm(abc_nav.dropna(), freq="D")
                s25 = mm(abc_nav.loc[SEG25[0]:SEG25[1]].dropna(), freq="D")
                combo_rows.append({
                    "alpha": a, "a_cap": ac, "b_cap": bc,
                    "avg_leverage": round(avg_lev, 2),
                    "turn_ann": round(turn, 1),
                    "Sharpe": round(m["Sharpe"], 3),
                    "AnnRet": round(m["AnnRet"], 4),
                    "MaxDD": round(m["MaxDD"], 4),
                    "stress_Sharpe": round(s25["Sharpe"], 3),
                })
    cta_df = pd.DataFrame(cta_results)
    combo_df = pd.DataFrame(combo_rows)
    return cta_df, combo_df


def main():
    panel = pd.read_parquet(PANEL_V2_OUTPUT)
    df, _ = run_phase5(panel)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "phase5_ensemble.csv")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
