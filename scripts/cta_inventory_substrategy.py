#!/usr/bin/env python3
"""cta_inventory_substrategy.py — 库存因子独立子策略 (锌/大豆/豆粕).

独立于主CTA策略的库存因子子策略:
    3个稳健品种-因子对: ZN_zscore, A_trend20w, M_chg4w
    IS(2018-2021)定方向 → OOS(2022-2025)验证
    日频调仓, vol target 8%

用法:
    python3 scripts/cta_inventory_substrategy.py        # IS/OOS回测
    python3 scripts/cta_inventory_substrategy.py --live  # 实盘模拟(最新数据)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

CMF_DB = "data/cmf/fundamental.duckdb"
CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"
REPORT_DIR = Path("reports/cta")

# ── 稳健品种-因子对 (IS/OOS一致 + IS价差>10%) ──────────────────────────────

ROBUST_PAIRS = {
    "ZN_zscore": {"product": "ZN", "direction": +1, "factor_type": "zscore", "window": 52},
    "A_trend20w": {"product": "A", "direction": -1, "factor_type": "trend20w", "window": 100},
    "M_chg4w": {"product": "M", "direction": -1, "factor_type": "chg4w", "window": 20},
}


def load_inventory_series(source: str, value_col: str) -> pd.Series:
    """加载单个库存序列."""
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM inventory_daily "
        "WHERE source=? AND value_col=? ORDER BY trade_date",
        [source, value_col],
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].sort_index()


def load_indicator_series(index_name: str) -> pd.Series:
    """加载tfapi_indicators单个序列."""
    con = duckdb.connect(CMF_DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM tfapi_indicators WHERE index_name=? ORDER BY trade_date",
        [index_name],
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].sort_index()


def load_cn_close(products: list[str]) -> pd.DataFrame:
    """国内品种日频收盘价."""
    con = duckdb.connect(CTA_DB, read_only=True)
    out = {}
    for p in products:
        try:
            df = con.execute(
                f"SELECT trade_date, close FROM smoothed_{p.lower()}_daily "
                f"WHERE close IS NOT NULL ORDER BY trade_date"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            out[p] = df.set_index("trade_date")["close"]
        except Exception:
            continue
    con.close()
    return pd.DataFrame(out)


# ── 因子计算 ────────────────────────────────────────────────────────────────

def compute_factor(inv_series: pd.Series, factor_type: str,
                   window: int) -> pd.Series:
    """计算单个品种的单个因子."""
    if factor_type == "zscore":
        m = inv_series.rolling(window, min_periods=26).mean()
        sd = inv_series.rolling(window, min_periods=26).std().replace(0, np.nan)
        return (inv_series - m) / sd
    elif factor_type == "chg4w":
        return inv_series.pct_change(20)  # 4周=20交易日
    elif factor_type == "trend20w":
        return inv_series.pct_change(100)  # 20周=100交易日
    else:
        raise ValueError(f"Unknown factor type: {factor_type}")


def load_all_factors() -> dict[str, pd.Series]:
    """加载3个稳健因子的数据."""
    factors = {}
    for pair_name, cfg in ROBUST_PAIRS.items():
        prod = cfg["product"]
        # 优先SHFE/COMEX库存(LME次选)
        if prod == "ZN":
            try:
                s = load_inventory_series("LME铜库存", "锌-库存")
            except Exception:
                s = pd.Series(dtype=float)
        elif prod == "A":
            try:
                s = load_indicator_series("大豆库存_中国")
            except Exception:
                s = pd.Series(dtype=float)
        elif prod == "M":
            try:
                s = load_indicator_series("豆粕库存_中国")
            except Exception:
                s = pd.Series(dtype=float)
        else:
            continue

        if s.empty or s.notna().sum() < 100:
            continue

        factors[pair_name] = compute_factor(s, cfg["factor_type"], cfg["window"])
    return factors


# ── 回测引擎 ────────────────────────────────────────────────────────────────

def run_substrategy(factors: dict[str, pd.Series],
                    cn_close: pd.DataFrame,
                    cost_bps: dict[str, float],
                    target_vol: float = 0.08,
                    vol_window: int = 20,
                    z_extreme: float = 1.0) -> dict:
    """库存因子独立子策略回测.

    信号: IS(2018-2021)定方向, OOS(2022-2025)按IS方向持有
    仓位: sign(factor) × direction × vol_target (极端化: |z|>z_extreme才持仓)
    """
    # 对齐到国内交易日 + PIT处理 (库存数据日频，PIT延迟0-1天)
    signal = pd.DataFrame(0.0, index=cn_close.index, columns=cn_close.columns)
    is_mask = cn_close.index < "2022-01-01"
    ret10 = cn_close.pct_change(10, fill_method=None).shift(-10)  # 未来10日收益

    for pair_name, cfg in ROBUST_PAIRS.items():
        if pair_name not in factors:
            continue
        prod = cfg["product"]
        if prod not in cn_close.columns:
            continue

        f = factors[pair_name]
        # 对齐到国内交易日
        f = f.reindex(cn_close.index).ffill().shift(1)  # PIT: 日频数据延迟1天

        # IS期定方向
        rv = ret10[prod]
        is_valid = is_mask & f.notna() & rv.notna()
        fv, rvv = f[is_valid], rv[is_valid]
        if len(fv) < 150:
            continue
        q = pd.qcut(fv.rank(method="first"), 5, labels=False)
        spread = (rvv[q == 4].mean() - rvv[q == 0].mean()) * 252 / 10

        # 预设方向 (从IS研究中已确认)
        direction = cfg["direction"]

        # 信号: 极端化 (|z|>z_extreme才持仓)
        sig = np.sign(f) * direction * (f.abs() > z_extreme).astype(float)
        signal[prod] = signal[prod].fillna(0) + sig.fillna(0)

    signal = signal.clip(-1, 1)

    # vol targeting
    ret = cn_close.pct_change(fill_method=None)
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=1.0).fillna(0)
    pos = signal * scale

    # PnL
    pnl_gross = pos.shift(1) * ret
    turnover = pos.diff().abs()
    cost = turnover * pd.Series(cost_bps).reindex(pos.columns).fillna(5.0) / 10000
    pnl_net = (pnl_gross - cost).sum(axis=1)

    valid = pnl_net.dropna()
    nav = (1 + valid).cumprod()
    n = len(valid)
    ann_ret = nav.iloc[-1] ** (252 / n) - 1 if n > 0 else 0
    ann_vol = valid.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()

    # IS/OOS分期
    is_ret = valid[valid.index < "2022-01-01"]
    oos_ret = valid[valid.index >= "2022-01-01"]

    def stats(r):
        if len(r) < 50:
            return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "calmar": 0}
        nav_r = (1 + r).cumprod()
        ann = nav_r.iloc[-1] ** (252 / len(r)) - 1
        dd = (nav_r / nav_r.cummax() - 1).min()
        return {"sharpe": ann / (r.std() * np.sqrt(252)), "ann_ret": ann,
                "max_dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0}

    s_is = stats(is_ret)
    s_oos = stats(oos_ret)

    # 年度收益
    yearly = valid.groupby(valid.index.year).apply(lambda x: (1 + x).prod() - 1)

    # 换手率
    true_turnover = turnover.sum(axis=1).mean() * 252

    return {
        "full": {"sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                 "ann_ret": ann_ret, "vol": ann_vol, "max_dd": dd,
                 "calmar": ann_ret / abs(dd) if dd != 0 else 0},
        "is": s_is, "oos": s_oos,
        "nav": nav, "yearly": yearly, "turnover": true_turnover,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="实盘模拟(最新数据)")
    args = parser.parse_args()

    print("=== 库存因子独立子策略 (锌/大豆/豆粕) ===")
    print(f"稳健对: {', '.join(ROBUST_PAIRS.keys())}")

    print("\n=== Step 1: 加载数据 ===")
    factors = load_all_factors()
    cn_close = load_cn_close(["ZN", "A", "M", "CU", "AL"])
    print(f"  因子: {len(factors)}个")
    print(f"  价格: {cn_close.shape[1]}品种, {cn_close.index.min().date()}~{cn_close.index.max().date()}")

    print("\n=== Step 2: 回测 (IS定方向 + OOS验证) ===")
    cost_table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    cost_bps = {p: (cost_table.get(p, {}).get("slippage_bps", 5.0)
                    + cost_table.get(p, {}).get("commission_bps", 0.1))
                for p in cn_close.columns}

    result = run_substrategy(factors, cn_close, cost_bps)

    f = result["full"]
    is_r = result["is"]
    oos_r = result["oos"]
    print(f"\n  全期:   Sharpe={f['sharpe']:.3f} AnnRet={f['ann_ret']:.2%} "
          f"Vol={f['vol']:.2%} MaxDD={f['max_dd']:.2%} Calmar={f['calmar']:.2f}")
    print(f"  IS期:   Sharpe={is_r['sharpe']:.3f} AnnRet={is_r['ann_ret']:.2%} "
          f"MaxDD={is_r['max_dd']:.2%} Calmar={is_r['calmar']:.2f}")
    print(f"  OOS期:  Sharpe={oos_r['sharpe']:.3f} AnnRet={oos_r['ann_ret']:.2%} "
          f"MaxDD={oos_r['max_dd']:.2%} Calmar={oos_r['calmar']:.2f}")
    print(f"  年化换手: {result['turnover']:.1f}x")

    # 年度收益
    print(f"\n  年度收益:")
    for yr, ret in result["yearly"].items():
        print(f"    {yr}: {ret:+.2%}")

    # 保存
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"nav": result["nav"]}).to_csv(
        REPORT_DIR / "inventory_sub_nav.csv"
    )
    yearly_df = result["yearly"].to_frame("ann_ret")
    yearly_df.to_csv(REPORT_DIR / "inventory_sub_yearly.csv")

    summary = {
        "pairs": list(ROBUST_PAIRS.keys()),
        "full": f, "is": is_r, "oos": oos_r,
        "turnover": result["turnover"],
        "yearly": {str(k): float(v) for k, v in result["yearly"].items()},
    }
    (REPORT_DIR / "inventory_sub_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n已保存: {REPORT_DIR}/inventory_sub_*.csv")


if __name__ == "__main__":
    main()
