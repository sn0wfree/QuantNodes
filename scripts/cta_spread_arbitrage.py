#!/usr/bin/env python3
"""cta_spread_arbitrage.py — 价差套利子策略 (基差55品种 + 跨品种26对).

单一子策略, 独立资金池, 与主CTA/锌库存/M经纪持仓并列.

信号池:
    A. 同品种基差 (spot_basis_daily, 55品种):
       B1 期限斜率 (F4-F1)/F1 z-score — 主CTA最强因子形态
       B2 基差水平 60日均值 — 极端水平
       B3 近月基差收敛 z-score — |z|>1反转
       B4 主力基差收敛 z-score — 同B3
    B. 跨品种价差 (cross_product_spread, 26对):
       X1 价差z-score — |z|>1反转
       X2 价差动量 20日 — 趋势延续

方法论: scripts/cta_shared/factor_research (IS/OOS方向一致性门槛)

用法:
    python3 scripts/cta_spread_arbitrage.py            # 全流程
    python3 scripts/cta_spread_arbitrage.py --ic-only  # 仅验证
"""
from __future__ import annotations

import argparse
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from cta_shared.factor_research import (  # noqa: E402
    REPORT_DIR, compute_ic, is_oos_direction_consistency,
    load_cost_bps, period_stats, rolling_zscore,
    summarize_factor_test, vol_target_backtest, yearly_returns,
)

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
IS_END = "2022-01-01"
MIN_IS_SPREAD = 0.05   # IS年化价差>5%才认定方向
Z_EXTREME = 1.0        # |z|>1才持仓


# ── 数据加载 ────────────────────────────────────────────────────────────────

def load_basis_panel() -> pd.DataFrame:
    """近月基差率宽表 (T, N品种)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, symbol, near_basis_rate FROM spot_basis_daily "
        "WHERE near_basis_rate IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot_table(index="trade_date", columns="symbol",
                          values="near_basis_rate")


def load_term_slope() -> pd.DataFrame:
    """期限斜率 (近月vs次月, 近月-次月)/近月 宽表."""
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, product, rank, close FROM term_structure_daily "
        "WHERE rank IN (1, 2) AND close IS NOT NULL"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    p1 = df[df["rank"] == 1].pivot_table(index="trade_date", columns="product", values="close")
    p2 = df[df["rank"] == 2].pivot_table(index="trade_date", columns="product", values="close")
    return (p1 - p2) / p1


def load_pair_spread() -> dict[str, pd.Series]:
    """26对跨品种价差 (bp), {pair_name: Series}."""
    con = duckdb.connect(CTA_DB, read_only=True)
    df = con.execute(
        "SELECT pair_name, trade_date, spread_bp FROM cross_product_spread "
        "WHERE spread_bp IS NOT NULL ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return {name: g.set_index("trade_date")["spread_bp"]
            for name, g in df.groupby("pair_name")}


# ── 因子工程 ────────────────────────────────────────────────────────────────

def build_basis_factors(basis: pd.DataFrame, slope: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """基差4因子 × 55品种. {factor_name: DataFrame(T, N)}.

    ⚠️ B2不做双重z-score (60日均值本身已平滑, 再z-score会因std→0爆炸出inf,
    拖累IC统计量) — 直接用60日均值水平 (业界BasisLevelFactor同形态).
    """
    out: dict[str, pd.DataFrame] = {}
    # B1 期限斜率 z-score
    out["B1_term_slope"] = slope.apply(rolling_zscore, axis=0, window=252)
    # B2 基差水平 (60日均值, 水平值无外层变换)
    out["B2_basis_level"] = basis.rolling(60, min_periods=30).mean()
    # B3 近月基差收敛
    out["B3_basis_conv"] = basis.apply(rolling_zscore, axis=0, window=60)
    return out


def build_pair_factors(pairs: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """跨品种2因子 × 26对. {factor_name: DataFrame(T, N对)}."""
    out: dict[str, pd.DataFrame] = {}
    z = pd.DataFrame({k: rolling_zscore(v, window=60) for k, v in pairs.items()})
    mom = pd.DataFrame({
        k: rolling_zscore(v.pct_change(20), window=252) for k, v in pairs.items()
    })
    out["X1_pair_conv"] = z
    out["X2_pair_mom"] = mom
    return out


# ── 验证 + 回测 ─────────────────────────────────────────────────────────────

def run_all(ic_only: bool) -> None:
    cn_close_cache: dict[str, pd.Series] = {}

    def get_close(product: str) -> pd.Series | None:
        if product not in cn_close_cache:
            con = duckdb.connect(CTA_DB, read_only=True)
            try:
                df = con.execute(
                    f"SELECT trade_date, close FROM smoothed_{product.lower()}_daily "
                    f"WHERE close IS NOT NULL ORDER BY trade_date"
                ).df()
            finally:
                con.close()
            if df.empty:
                cn_close_cache[product] = None  # type: ignore[assignment]
            else:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                cn_close_cache[product] = df.set_index("trade_date")["close"]
        return cn_close_cache[product]

    basis = load_basis_panel()
    slope = load_term_slope()
    pairs = load_pair_spread()
    basis_factors = build_basis_factors(basis, slope)
    pair_factors = build_pair_factors(pairs)
    all_factors = {**basis_factors, **pair_factors}

    # 因子→品种映射: 基差因子列=品种, 价差因子列=pair_name(取near_product腿)
    results = []
    for fname, fdf in all_factors.items():
        for col in fdf.columns:
            product = col if fname.startswith("B") else col.split("-")[0]
            close = get_close(product)
            if close is None:
                continue
            fwd5 = close.pct_change(5, fill_method=None).shift(-5)
            f = fdf[col].reindex(close.index).ffill()
            if f.notna().sum() < 400:
                continue
            ic = compute_ic(f, fwd5, min_obs=300)
            cons = is_oos_direction_consistency(f, fwd5, is_end=IS_END,
                                                min_is_spread=MIN_IS_SPREAD)
            results.append({"factor": fname, "leg": col, "product": product,
                            **ic, **(cons or {})})

    df = pd.DataFrame(results)
    df.to_csv(REPORT_DIR / "spread_arb_ic.csv", index=False)
    n_sig = (df["is_spread"].abs() > MIN_IS_SPREAD).sum()
    n_agree = df["agree"].sum()
    print(f"组合数: {len(df)}, IS显著: {n_sig}, 方向一致: {n_agree} "
          f"({n_agree / max(n_sig, 1):.0%})")
    print("\n=== 方向一致且IS显著的组合 ===")
    robust = df[df["agree"] & (df["is_spread"].abs() > MIN_IS_SPREAD)]
    robust = robust.sort_values("is_spread", key=abs, ascending=False)
    for _, r in robust.iterrows():
        print("  " + summarize_factor_test(
            {"ic": r["ic"], "ic_roll_mean": r["ic_roll_mean"], "t_stat": r["t_stat"]},
            {"is_spread": r["is_spread"], "oos_spread": r["oos_spread"],
             "agree": True}, f"{r['factor']}/{r['leg']}"))

    if n_sig > 0 and n_agree / n_sig < 0.6:
        print(f"\n❌ 一致率{n_agree / max(n_sig, 1):.0%}<60% → 止损归档")
        return
    if ic_only:
        return

    # IS定方向回测 (信号: sign(factor)×IS方向×极端化, 信号映射到品种级)
    print("\n=== IS定方向回测 ===")
    products = sorted(set(robust["product"]))
    close_df = pd.DataFrame({p: get_close(p) for p in products})
    signal = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for _, r in robust.iterrows():
        fdf = all_factors[r["factor"]]
        f = fdf[r["leg"]].reindex(close_df.index).ffill()
        sig = (np.sign(f) * np.sign(r["is_spread"])
               * (f.abs() > Z_EXTREME).astype(float)).fillna(0)
        signal[r["product"]] += sig
    signal = signal.clip(-1, 1)

    res = vol_target_backtest(signal, close_df, cost_bps=load_cost_bps(),
                              target_vol=0.08)
    st = period_stats(res["pnl"], is_end=IS_END)
    print(f"  全期: Sharpe={res['sharpe']:.3f} AnnRet={res['ann_ret']:+.2%} "
          f"MaxDD={res['max_dd']:.2%} 换手={res['turnover_annual']:.1f}x")
    print(f"  IS:   Sharpe={st['is']['sharpe']:.3f} AnnRet={st['is']['ann_ret']:+.2%}")
    print(f"  OOS:  Sharpe={st['oos']['sharpe']:.3f} AnnRet={st['oos']['ann_ret']:+.2%} "
          f"MaxDD={st['oos']['max_dd']:.2%}")

    if st["oos"]["sharpe"] < 0.15:
        print("❌ OOS Sharpe<0.15 → 止损归档")
        return

    yr = yearly_returns(res["pnl"])
    print("\n年度收益: " + "  ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
    pd.DataFrame({"nav": res["nav"]}).to_csv(REPORT_DIR / "spread_arb_nav.csv")
    yr.to_frame("ann_ret").to_csv(REPORT_DIR / "spread_arb_yearly.csv")
    print(f"\n✓ 通过OOS门槛, 已保存: {REPORT_DIR}/spread_arb_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()
    run_all(args.ic_only)
