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


def sleeve_b(panel: pd.DataFrame):
    """EW-vt: 等权 + vol target cap1 (风险资产参与 sleeve)."""
    cfg = AllWeatherConfig(use_hrp=False, trend_mode="none",
                           use_vol_target=True, max_leverage=1.0)
    return run_backtest(panel, cfg, compute_metrics_fn=_noop)


# ── Sleeve C: CTA 信号 sleeve ───────────────────────────────────────
def cta_sleeve_returns(panel: pd.DataFrame,
                       signals_path: Path = CTA_SIGNALS_PATH,
                       momentum_filter_window: int = 60) -> pd.Series:
    """CTA v7 信号 → 日收益序列 (gross≤1, 含换手成本).

    信号 (T×N, -1/0/+1) 于 t 收盘生成, 赚 t→t+1 收益:
        w_i(t) = sign_i(t) / N_active(t)
        contrib(t+1) = Σ w_i(t) · r_i(t+1)
        cost(t+1) = Σ |Δw_i| · rate_i
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
    w = sig.div(n_active, axis=0)          # gross = Σ|sign|/N ≤ 1
    w = w.fillna(0.0)

    contrib = (w.shift(1) * ret[fut_cols].fillna(0.0)).sum(axis=1)
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = dw * FUTURES_COMMISSION_RATE
    out = (contrib - cost).dropna()
    out.name = "cta_sleeve"
    return out


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


# ── Phase 5 编排 ────────────────────────────────────────────────────
def run_phase5(panel: pd.DataFrame, include_cta: bool = True,
               lookback: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """构建各 sleeve + 集成组合, 返回 (指标长表, 各 nav 的 DataFrame)."""
    mm = _metrics_fn()
    res_a = sleeve_a(panel)
    res_b = sleeve_b(panel)
    navs = {
        "A_X2L4": res_a.nav,
        "B_EWvt": res_b.nav,
    }
    if include_cta:
        cta_ret = cta_sleeve_returns(panel)
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


SEG25 = SEGMENTS["stress_25+"]


def _metrics_fn():
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
        compute_metrics,
    )
    return compute_metrics


def main():
    panel = pd.read_parquet(PANEL_V2_OUTPUT)
    df, _ = run_phase5(panel)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "phase5_ensemble.csv")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
