#!/usr/bin/env python3
"""cta_liquidity_constraint_test.py — 流动性约束测试 (后处理, 三算法对比).

目的:
  factor_modules.py 无 max_volume_participation 参数 (不动别人的代码).
  本脚本对主CTA回测结果做流动性约束后处理, 测试4场景 × 2口径.

算法:
  A: 重算NAV (手×价×乘数累计, 假设初始1元权益单位)
  B: 增量成本法 (cap触发的额外滑点叠加到baseline nav)
  C: 简化重算 (与A同, 但忽略动态杠杆/融资, 仅手×价×乘数累计)

输入:
  - 主CTA基线回测 (run_maincta, baseline nav)
  - smooth_*_daily.volume / .close (回测期间每品种每日成交手数和收盘价)
  - contract_specs.contract_multiplier (品种元/手换算)
  - ADV窗口 = 20日滚动均量 (参数可调)

输出:
  reports/cta/liquidity_constraint_scenarios_*.csv   4场景×2口径×3算法对比
  reports/cta/liquidity_constraint_5pct_*.csv         5%约束NAV序列 (close+open)
  reports/cta/liquidity_constraint_yearly.csv         年度对比 (基础 vs 5%约束)
  reports/cta/liquidity_constraint_trigger_stats.csv  哪些品种触发cap多少次
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import fields

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "scripts")
from QuantNodes.strategy.cta.strategies.factor_modules import (  # noqa: E402
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS,
)
from cta_shared.factor_research import REPORT_DIR  # noqa: E402

CTA_DB = "/home/ll/Public/DataCache/futures_options_daily.duckdb"

# 4场景 (主CTA在5%/10%参与率下完全不被约束, 故采用更严苛档位)
SCENARIOS = [
    ("基础", np.inf),
    ("5%参与率", 0.05),
    ("10%参与率", 0.10),
    ("20%参与率", 0.20),
    ("0.5%参与率(极严)", 0.005),
    ("0.1%参与率(极限)", 0.001),
]

# ADV窗口(交易日) — 可调
ADV_WINDOWS = [20]

# FINAL_LUMP_CFG
CFG_KW = {
    "aggregation": "tvpr", "module_weighting": "inv_vol",
    "continuous": True, "cont_z_window": 252, "cont_z_clip": 3.0,
    "momentum_filter_window": 60, "momentum_filter_coef": 0.0,
    "target_vol": 1.5, "max_leverage": 3.0, "leverage_hard_cap": 3.0,
    "margin_call": True, "position_ema_alpha": 0.15,
    "roll_cost_bps": 10.0, "roll_model": "lump",
}


def load_cost_table() -> tuple[dict, dict]:
    """加载按品种手续费率和滑点bps."""
    raw = json.loads((REPORT_DIR / "cost_model.json").read_text())
    return (
        {p: v["commission_bps"] / 10000 for p, v in raw.items()},
        {p: v["slippage_bps"] for p, v in raw.items()},
    )


def load_multipliers(products: list[str]) -> dict[str, float]:
    """从contract_specs加载{product: 元/手}."""
    con = duckdb.connect(CTA_DB, read_only=True)
    rows = con.execute(
        f"SELECT product, contract_multiplier FROM contract_specs "
        f"WHERE product IN ({','.join([chr(39)+p+chr(39) for p in products])}) "
        f"AND contract_multiplier IS NOT NULL AND contract_multiplier > 0"
    ).fetchall()
    con.close()
    mult = {p: float(m) for p, m in rows}
    missing = [p for p in products if p not in mult]
    if missing:
        print(f"  [WARN] 缺少乘数品种(将用默认1): {missing}")
        for p in missing:
            mult[p] = 1.0
    return mult


def run_maincta_baseline(exec_mode: str):
    """跑主CTA基线, 返回 CtaBacktestResult."""
    cfg_kw = dict(CFG_KW)
    if exec_mode == "open":
        cfg_kw["execution_price_type"] = "open"
    commission_rate, slippage_bps = load_cost_table()
    cfg_kw["commission_rate"] = commission_rate
    cfg_kw["slippage_bps"] = slippage_bps
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(config=cfg, products=list(DEFAULT_PRODUCTS))
    return s.run()


def fetch_daily_volume(products: list[str], start: str = "2018-01-01",
                        end: str = "2025-12-31") -> pd.DataFrame:
    """从smoothed_*_daily取每个品种每日成交手数 (主连back-adjusted)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    frames = []
    for p in products:
        try:
            df = con.execute(
                f"SELECT trade_date, volume FROM smoothed_{p.lower()}_daily "
                f"WHERE trade_date BETWEEN '{start}' AND '{end}' AND volume IS NOT NULL"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")["volume"].rename(p)
            frames.append(df)
        except Exception:
            pass
    con.close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().ffill()


def fetch_daily_close(products: list[str], start: str = "2018-01-01",
                       end: str = "2025-12-31") -> pd.DataFrame:
    """从smoothed_*_daily取每个品种每日收盘价."""
    con = duckdb.connect(CTA_DB, read_only=True)
    frames = []
    for p in products:
        try:
            df = con.execute(
                f"SELECT trade_date, close FROM smoothed_{p.lower()}_daily "
                f"WHERE trade_date BETWEEN '{start}' AND '{end}' AND close IS NOT NULL"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")["close"].rename(p)
            frames.append(df)
        except Exception:
            pass
    con.close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().ffill()


def calc_adv(daily_volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """滚动N日均量 (ADV), 反映该品种真实可参与量."""
    return daily_volume.rolling(window, min_periods=1).mean()


def stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).fillna(0)
    if len(r) < 50:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "calmar": 0, "vol": 0,
                "final": 1.0, "n": len(nav)}
    ann = nav.iloc[-1] ** (252 / len(nav)) - 1
    dd = (nav / nav.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    return {
        "sharpe": ann / vol if vol > 0 else 0,
        "ann_ret": ann,
        "max_dd": dd,
        "calmar": ann / abs(dd) if dd != 0 else 0,
        "vol": vol,
        "final": float(nav.iloc[-1]),
        "n": len(nav),
    }


def trigger_analysis(pos: pd.DataFrame, prices: pd.DataFrame, mult: dict,
                     adv: pd.DataFrame, participation: float,
                     exec_mode: str) -> list[dict]:
    """统计每个品种触发cap的天数和最大超额手数 (用于close 5%场景报告).

    5%参与率 = 当日持仓手数 ≤ 当日成交量(手) × 5%.
    cap_lots = adv × participation (单位: 手).
    """
    if not np.isfinite(participation):
        return []
    cap_lots = (adv * participation).reindex(pos.index, method="ffill").fillna(0)
    delta = pos - pos.clip(lower=-cap_lots, upper=cap_lots)
    out = []
    for col in pos.columns:
        n_trig = int((delta[col].abs() > 0.001).sum())
        max_excess = float(delta[col].abs().max())
        if n_trig > 0:
            out.append({
                "product": col, "exec_mode": exec_mode,
                "n_trigger_days": n_trig,
                "max_excess_lots": max_excess,
                "median_adv_lots": float(adv[col].median()),
                "mult": mult.get(col, 1),
            })
    return out


def nav_from_pnl(pnl: pd.Series, init: float = 1.0) -> pd.Series:
    """pct change式pnl → NAV. pnl为对单位资金的日收益百分比."""
    pnl = pnl.fillna(0).clip(-0.99, 10.0)  # 防爆炸: 单日pnl不超过-99%/1000%
    return (1 + pnl).cumprod() * init


def algo_A_renav(res, pos: pd.DataFrame, prices: pd.DataFrame,
                 mult: dict, adv: pd.DataFrame, participation: float,
                 slippage_bps: dict) -> tuple[pd.Series, pd.DataFrame]:
    """算法A: 重算NAV (手×价×乘数累计, 用res.equity初始权益归一化).

    账户权益路径:
      daily_pnl[T] (元) = Σ [pos[T-1] × price_change[T] × multiplier]
      equity[T] = initial_equity + cumsum(daily_pnl)
      nav[T] = equity[T] / initial_equity

    约束: pos_capped[T-1] = sign(pos) × min(|pos|, ADV[T-1] × participation / (price[T-1] × mult))
    slippage cost = |pos - capped| × price[T-1] × mult × slippage_bps/1e4
    """
    init_eq = float(res.equity[0])
    mean_mult = pd.Series(mult).reindex(prices.columns)

    if not np.isfinite(participation):
        pnl_abs = (pos.shift(1) * prices.diff() * mean_mult).sum(axis=1).fillna(0)
        renav = (init_eq + pnl_abs.cumsum()) / init_eq
        return renav, pd.DataFrame()

    cap_lots = (adv * participation).reindex(pos.index, method="ffill").fillna(0)

    capped = pos.clip(lower=-cap_lots, upper=cap_lots)
    delta = pos - capped

    price_lag = prices.shift(1)
    pnl_capped = (capped.shift(1) * prices.diff() * mean_mult).sum(axis=1)

    slip_rate = pd.Series({p: slippage_bps.get(p, 0) / 10000
                            for p in pos.columns})
    cost = (delta.shift(1).abs() * price_lag
            * mean_mult.reindex(pos.columns) * slip_rate.reindex(pos.columns)).sum(axis=1)

    pnl_net = (pnl_capped - cost).fillna(0)
    renav = (init_eq + pnl_net.cumsum()) / init_eq
    trigger = (delta.abs() > 0.001).sum(axis=1)

    return renav, pd.DataFrame({
        "trigger_count": trigger,
        "daily_cost": cost,
        "daily_pnl_capped": pnl_capped,
    })


def algo_B_incremental(res, baseline_nav: pd.Series, pos: pd.DataFrame,
                        prices: pd.DataFrame, mult: dict, adv: pd.DataFrame,
                        participation: float, slippage_bps: dict
                        ) -> tuple[pd.Series, pd.DataFrame]:
    """算法B: 增量成本法 (在baseline nav基础上扣除cap触发的额外滑点).

    假设约束不改变PnL收入, 只增加cap部分的额外滑点:
      delta_slipped[T] = |pos[T-1] - capped[T-1]| × price[T-1] × mult × slip_bps/1e4
      baseline_daily_ret[T] = nav[T] / nav[T-1] - 1
      constrained_ret[T] = baseline_daily_ret[T] - delta_slipped[T] / (nav[T-1] × initial_equity)
    """
    if not np.isfinite(participation):
        return baseline_nav.copy(), pd.DataFrame()

    cap_lots = (adv * participation).reindex(pos.index, method="ffill").fillna(0)
    capped = pos.clip(lower=-cap_lots, upper=cap_lots)
    delta = pos - capped

    price_lag = prices.shift(1)
    slip_rate = pd.Series({p: slippage_bps.get(p, 0) / 10000
                            for p in pos.columns})
    cost = (delta.shift(1).abs() * price_lag
            * pd.Series(mult).reindex(pos.columns) * slip_rate.reindex(pos.columns)).sum(axis=1)

    init_eq = float(res.equity[0])
    base_ret = baseline_nav.pct_change(fill_method=None).fillna(0)
    nav_prev = baseline_nav.shift(1).fillna(1.0)
    constrained_ret = base_ret - cost / (nav_prev * init_eq)
    constrained_ret = constrained_ret.clip(-0.99, 10.0)

    trigger = (delta.abs() > 0.001).sum(axis=1)
    return (1 + constrained_ret).cumprod() * baseline_nav.iloc[0], pd.DataFrame({
        "trigger_count": trigger,
        "daily_cost": cost,
    })


def algo_C_simple_renav(res, pos: pd.DataFrame, prices: pd.DataFrame,
                        mult: dict, adv: pd.DataFrame, participation: float
                        ) -> tuple[pd.Series, pd.DataFrame]:
    """算法C: 简化重算 (同A但不考虑额外滑点成本, 只看cap对PnL的影响)."""
    init_eq = float(res.equity[0])
    mean_mult = pd.Series(mult).reindex(prices.columns)

    if not np.isfinite(participation):
        pnl_abs = (pos.shift(1) * prices.diff() * mean_mult).sum(axis=1).fillna(0)
        return (init_eq + pnl_abs.cumsum()) / init_eq, pd.DataFrame()

    cap_lots = (adv * participation).reindex(pos.index, method="ffill").fillna(0)
    capped = pos.clip(lower=-cap_lots, upper=cap_lots)
    pnl_capped = (capped.shift(1) * prices.diff() * mean_mult).sum(axis=1)
    delta = pos - capped
    trigger = (delta.abs() > 0.001).sum(axis=1)
    renav = (init_eq + pnl_capped.cumsum().fillna(0)) / init_eq
    return renav, pd.DataFrame({"trigger_count": trigger})


def main():
    print("=" * 100)
    print("流动性约束测试 — 4场景 × 2口径 × 3算法 (A重算 / B增量 / C简化)")
    print("=" * 100)

    products = list(DEFAULT_PRODUCTS)
    mult = load_multipliers(products)
    print(f"\n[1] 合约乘数加载完成: {len(mult)} 品种")

    print("[2] 加载日成交 + 收盘价...")
    daily_volume = fetch_daily_volume(products)
    daily_close = fetch_daily_close(products)
    print(f"  daily_volume: {daily_volume.shape}, daily_close: {daily_close.shape}")

    slippage_bps, _ = (lambda c, s: (s, c))(*load_cost_table())  # 仅取slippage
    commission_rate, _ = load_cost_table()  # noqa: F841 (未来如需)

    adv_window = 20
    adv = calc_adv(daily_volume, window=adv_window)
    print(f"  ADV窗口: {adv_window}日滚动均量, adv: {adv.shape}")

    results = []
    trigger_stats = []

    for exec_mode in ["close", "open"]:
        print(f"\n[3] 跑主CTA基线 (exec={exec_mode})...")
        res = run_maincta_baseline(exec_mode)
        nav_orig = res.nav.copy()
        nav_orig.name = "baseline"
        st_base = stats(nav_orig)
        print(f"  基线: Sharpe={st_base['sharpe']:.3f} Calmar={st_base['calmar']:.2f} "
              f"DD={st_base['max_dd']:.2%} AR={st_base['ann_ret']:.2%} "
              f"final={st_base['final']:.4f}")

        dates = pd.to_datetime(res.dates)
        pos = pd.DataFrame(
            {sym: res.positions[sym] for sym in res.positions},
            index=dates,
        )
        pos = pos.reindex(daily_close.index, method="ffill").fillna(0)
        prices = daily_close.reindex(pos.index, method="ffill")
        adv_aligned = adv.reindex(pos.index, method="ffill").fillna(0)

        # 持仓/ADV比例分析 (close口径一次性输出)
        if exec_mode == "close":
            ratio = (pos.abs() / adv_aligned.replace(0, np.nan)).dropna(how='all')
            peak_ratio_pct = (ratio.max() * 100).sort_values(ascending=False)
            print("\n  [持仓/ADV峰值] 各品种最高占比 (top10):")
            print(peak_ratio_pct.head(10).round(4).to_string())
            print(f"  平均peak ratio: {peak_ratio_pct.mean():.4f}%,  "
                  f"最大: {peak_ratio_pct.max():.4f}%")
            for th_pct in [10, 5, 1, 0.5, 0.1]:
                triggered = (peak_ratio_pct > th_pct).sum()
                print(f"  持仓/ADV>{th_pct}%触发品种: {triggered}/66")

        for label, participation in SCENARIOS:
            # 算法A
            nav_a, info_a = algo_A_renav(res, pos, prices, mult, adv_aligned,
                                          participation, slippage_bps)
            # 算法B
            nav_b, info_b = algo_B_incremental(res, nav_orig, pos, prices, mult,
                                                adv_aligned, participation, slippage_bps)
            # 算法C
            nav_c, info_c = algo_C_simple_renav(res, pos, prices, mult,
                                                  adv_aligned, participation)

            for algo_name, nav_x in [("A重算NAV", nav_a), ("B增量成本", nav_b),
                                       ("C简化重算", nav_c)]:
                st = stats(nav_x)
                results.append({
                    "exec_mode": exec_mode, "scenario": label, "algo": algo_name,
                    "participation": participation if np.isfinite(participation) else 1.0,
                    "sharpe": st["sharpe"], "calmar": st["calmar"],
                    "max_dd": st["max_dd"], "ann_ret": st["ann_ret"],
                    "vol": st["vol"], "final": st["final"],
                    "d_sharpe": st["sharpe"] - st_base["sharpe"],
                    "d_calmar": st["calmar"] - st_base["calmar"],
                    "d_max_dd": st["max_dd"] - st_base["max_dd"],
                    "d_ann_ret": st["ann_ret"] - st_base["ann_ret"],
                })

            # 触发统计 (close 0.1%极严场景, 最有意义)
            if exec_mode == "close" and participation == 0.001:
                trigger_stats.extend(
                    trigger_analysis(pos, prices, mult, adv_aligned,
                                       participation, exec_mode))

            # 保存5%约束后的NAV (close口径三种算法都保存)
            if exec_mode == "close" and participation == 0.05:
                nav_a.to_csv(REPORT_DIR / "liquidity_constraint_5pct_close_algoA.csv",
                              header=["nav"])
                nav_b.to_csv(REPORT_DIR / "liquidity_constraint_5pct_close_algoB.csv",
                              header=["nav"])
                nav_c.to_csv(REPORT_DIR / "liquidity_constraint_5pct_close_algoC.csv",
                              header=["nav"])
            if exec_mode == "open" and participation == 0.05:
                nav_a.to_csv(REPORT_DIR / "liquidity_constraint_5pct_open_algoA.csv",
                              header=["nav"])
                nav_b.to_csv(REPORT_DIR / "liquidity_constraint_5pct_open_algoB.csv",
                              header=["nav"])
                nav_c.to_csv(REPORT_DIR / "liquidity_constraint_5pct_open_algoC.csv",
                              header=["nav"])

    df = pd.DataFrame(results)
    df.to_csv(REPORT_DIR / "liquidity_constraint_scenarios.csv", index=False)

    # 输出对比表
    print("\n" + "=" * 100)
    print("[4] 4场景 × 3算法 对比 (close口径, 基线=主CTA close)")
    print("=" * 100)
    for algo_name in ["A重算NAV", "B增量成本", "C简化重算"]:
        sub = df[(df["exec_mode"] == "close") & (df["algo"] == algo_name)].set_index("scenario")
        print(f"\n  --- {algo_name} ---")
        out = sub[["sharpe", "calmar", "max_dd", "ann_ret", "d_sharpe", "d_calmar"]].round(4)
        print(out.to_string())
        suffix = algo_name.replace('算', '')
        out.to_csv(REPORT_DIR / f"liquidity_constraint_summary_close_{suffix}.csv")

    print("\n" + "=" * 100)
    print("[5] 4场景 × 3算法 对比 (open口径, 基线=主CTA open)")
    print("=" * 100)
    for algo_name in ["A重算NAV", "B增量成本", "C简化重算"]:
        sub = df[(df["exec_mode"] == "open") & (df["algo"] == algo_name)].set_index("scenario")
        print(f"\n  --- {algo_name} ---")
        out = sub[["sharpe", "calmar", "max_dd", "ann_ret", "d_sharpe", "d_calmar"]].round(4)
        print(out.to_string())
        suffix = algo_name.replace('算', '')
        out.to_csv(REPORT_DIR / f"liquidity_constraint_summary_open_{suffix}.csv")

    # 触发统计
    if trigger_stats:
        ts = pd.DataFrame(trigger_stats).sort_values("n_trigger_days", ascending=False)
        ts.to_csv(REPORT_DIR / "liquidity_constraint_trigger_stats.csv", index=False)
        print("\n[6] 5%约束 (close, A算法) 触发统计 top10:")
        print(ts.head(10).to_string(index=False))

    # 关键结论
    print("\n" + "=" * 100)
    print("[7] 关键结论")
    print("=" * 100)
    for mode in ["close", "open"]:
        sub = df[(df["exec_mode"] == mode) & (df["scenario"] == "5%参与率")]
        print(f"\n{mode}口径 5%参与率 vs 基础 (主CTA close/open):")
        for _, r in sub.iterrows():
            print(f"  {r['algo']:8s}: Sharpe {r['sharpe']:6.3f} (Δ{r['d_sharpe']:+.3f}), "
                  f"Calmar {r['calmar']:5.2f} (Δ{r['d_calmar']:+.2f}), "
                  f"DD {r['max_dd']:6.2%} (Δ{r['d_max_dd']:+.2%}), "
                  f"final {r['final']:6.4f}")

    print("\n输出文件:")
    print("  - liquidity_constraint_scenarios.csv              全场景全算法对比")
    print("  - liquidity_constraint_summary_close_{A,B,C}.csv  close口径各算法汇总")
    print("  - liquidity_constraint_summary_open_{A,B,C}.csv   open口径各算法汇总")
    print("  - liquidity_constraint_5pct_{close,open}_algo{A,B,C}.csv  5%约束NAV序列")
    print("  - liquidity_constraint_trigger_stats.csv          cap触发品种统计")


if __name__ == "__main__":
    main()
