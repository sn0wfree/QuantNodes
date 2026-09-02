# coding=utf-8
"""cta_cftc_strategy.py — CFTC持仓周频子策略 (Tier1).

数据: tfapi_cftc (国际品种COT周频持仓, report_date=周二持仓日)
映射: 10个国际品种 → 国内品种 (AU/AG/CU/C/A/B/Y/SR/CF/SC)
PIT:  t日可用数据 = report_date <= t - 3交易日 (周五发布)

因子 (3个):
    F1: mm_net_pct z-score (投机净持仓水平)
    F2: commercial净持仓 z-score (商业套保压力, 方向由IC确认)
    F3: MM-COM spread (投机-商业分歧)

回测 (路径甲: 纯周频):
    信号: 周五可用数据计算, 持有到下周五
    仓位: signal × vol_target (20周窗口, 目标8%)
    成本: cost_model.json 对应品种

用法:
    python3 scripts/cta_cftc_strategy.py              # 全流程: 因子+IC+回测
    python3 scripts/cta_cftc_strategy.py --ic-only    # 仅IC验证
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

# ── Step 1: 映射表 ──────────────────────────────────────────────────────────

CFTC_TO_CN = {
    "GOLD - COMMODITY EXCHANGE INC.": "AU",
    "SILVER - COMMODITY EXCHANGE INC.": "AG",
    "COPPER- #1 - COMMODITY EXCHANGE INC.": "CU",
    "CORN - CHICAGO BOARD OF TRADE": "C",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "A",
    "SOYBEAN OIL - CHICAGO BOARD OF TRADE": "Y",
    "SUGAR NO. 11 - ICE FUTURES U.S.": "SR",
    "COTTON NO. 2 - ICE FUTURES U.S.": "CF",
    "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE": "SC",
    # 弱映射 (标的差异): IC验证单独评估
    "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE": "PG",
}
WEAK_MAPS = {"PG"}  # 标注弱映射 (LPG vs NG 标的差异大)

PIT_LAG_DAYS = 3  # 周二持仓 → +3交易日 = 周五发布可用


def load_cftc_weekly() -> pd.DataFrame:
    """加载CFTC周频持仓 (仅映射品种), report_date为周二持仓日."""
    con = duckdb.connect(CMF_DB, read_only=True)
    markets = list(CFTC_TO_CN.keys())
    placeholders = ", ".join("?" for _ in markets)
    df = con.execute(
        f"SELECT market, report_date, oi, mm_long, mm_short, mm_net, mm_net_pct, "
        f"pm_long, pm_short, swap_long, swap_short "
        f"FROM tfapi_cftc WHERE market IN ({placeholders}) ORDER BY report_date",
        markets,
    ).df()
    con.close()
    df["product"] = df["market"].map(CFTC_TO_CN)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def load_cn_daily_close() -> pd.DataFrame:
    """国内品种日频收盘价 (smoothed连续合约)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    out = {}
    for prod in set(CFTC_TO_CN.values()):
        t = f"smoothed_{prod.lower()}_daily"
        try:
            df = con.execute(
                f"SELECT trade_date, close FROM {t} WHERE close IS NOT NULL ORDER BY trade_date"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            out[prod] = df.set_index("trade_date")["close"]
        except Exception:
            continue
    con.close()
    return pd.DataFrame(out)


def apply_pit(cftc: pd.DataFrame, cn_trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """PIT: report_date(周二持仓) +3个交易日(周五)才可用.

    实现: 按品种将周频数据重采样到国内日历, shift(PIT_LAG_DAYS)个交易日.
    """
    out = {}
    for prod, g in cftc.groupby("product"):
        g = g.set_index("report_date").sort_index()
        # 周频 → 日频 (前向填充到国内交易日历)
        daily = g.reindex(cn_trading_days.union(g.index)).ffill()
        daily = daily.reindex(cn_trading_days)
        # PIT: 滞后3个交易日
        daily = daily.shift(PIT_LAG_DAYS)
        out[prod] = daily
    return pd.concat(out, names=["product", "trade_date"])


# ── Step 2: 因子构建 ────────────────────────────────────────────────────────

ZSCORE_WINDOW = 52  # 52周窗口 (周频上计算, 非日频!)


def build_factors_weekly(cftc_weekly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """因子在周频上计算 (index=report_date, 52周z-score).

    ⚠️ 必须在周频上算z-score: 若在日频ffill面板上算52天窗口,
    因子值每天微变 → 信号每天翻转 → 换手爆炸+成本吞掉全部收益.

    F1: mm_net_pct z-score — 投机净持仓水平 (IC验证: 反向, 拥挤度反转)
    F2: commercial净持仓 z-score — 商业套保压力 (IC验证: 正向, smart money)
    F3: MM-COM spread — 投机与商业的分歧 (IC验证: 反向, MM主导)
    """
    factors = {"f1_mm": {}, "f2_com": {}, "f3_spread": {}}
    for prod, g in cftc_weekly.groupby("product"):
        g = g.set_index("report_date").sort_index()
        oi = g["oi"].replace(0, np.nan)
        mm_net_pct = g["mm_net"] / oi
        com_net = (g["pm_long"] - g["pm_short"]) / oi  # 商业净多占比

        def zscore(s: pd.Series) -> pd.Series:
            m = s.rolling(ZSCORE_WINDOW, min_periods=26).mean()
            sd = s.rolling(ZSCORE_WINDOW, min_periods=26).std().replace(0, np.nan)
            return (s - m) / sd

        factors["f1_mm"][prod] = zscore(mm_net_pct)
        factors["f2_com"][prod] = zscore(com_net)
        factors["f3_spread"][prod] = mm_net_pct + com_net

    return {k: pd.DataFrame(v) for k, v in factors.items()}


def pit_align_weekly(weekly_factors: dict[str, pd.DataFrame],
                     cn_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """周频因子 → 国内日历ffill → shift(3交易日) PIT."""
    out = {}
    for fname, fdf in weekly_factors.items():
        aligned = (
            fdf.reindex(cn_days.union(fdf.index))
            .ffill()
            .reindex(cn_days)
            .shift(PIT_LAG_DAYS)
        )
        out[fname] = aligned
    return out


# ── Step 3: IC验证 ──────────────────────────────────────────────────────────

def compute_ic(factors: dict[str, pd.DataFrame], cn_close: pd.DataFrame,
               horizon_days: int = 5) -> pd.DataFrame:
    """因子IC: 因子值(周频, PIT对齐) vs 国内品种未来N日收益.

    返回: 每因子×每品种的IC均值/t-stat/样本数
    """
    cn_ret = cn_close.pct_change(fill_method=None)
    fwd = cn_ret.shift(-horizon_days)  # 未来N日收益

    rows = []
    for fname, fdf in factors.items():
        for prod in fdf.columns:
            if prod not in fwd.columns:
                continue
            f = fdf[prod].dropna()
            r = fwd[prod].reindex(f.index)
            valid = f.notna() & r.notna()
            if valid.sum() < 100:
                continue
            # 时序相关 (单品种时间序列IC)
            ic = f[valid].corr(r[valid])
            # t-stat via 滚动IC检验
            roll_ic = f[valid].rolling(52).corr(r[valid]).dropna()
            if len(roll_ic) > 10:
                t_stat = roll_ic.mean() / (roll_ic.std() / np.sqrt(len(roll_ic)))
            else:
                t_stat = np.nan
            rows.append({
                "factor": fname, "product": prod,
                "ic_mean": ic, "ic_roll_mean": roll_ic.mean(),
                "t_stat": t_stat, "n": int(valid.sum()),
                "weak_map": prod in WEAK_MAPS,
            })
    return pd.DataFrame(rows)


# ── Step 4: 纯周频回测 (路径甲) ─────────────────────────────────────────────

def weekly_backtest(signal: pd.DataFrame, cn_close: pd.DataFrame,
                    cost_bps: dict[str, float], target_vol: float = 0.08,
                    vol_window: int = 20) -> dict:
    """周频回测: 周五信号 → 持有到下周五, vol target按品种.

    Args:
        signal: (T_daily, N) 日频信号面板 (周频不变, PIT已对齐)
        cn_close: (T_daily, N) 国内日频收盘价
        cost_bps: {product: 单边bp}
    """
    ret = cn_close.pct_change(fill_method=None)
    # 信号日频化已经完成 (PIT对齐时ffill), 直接日频PnL计算更精确
    pos_raw = signal.copy()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=1.0).fillna(0)
    pos = pos_raw * scale

    pnl_gross = (pos.shift(1) * ret)
    turnover = pos.diff().abs()
    cost = turnover * pd.Series(cost_bps).reindex(pos.columns).fillna(5.0) / 10000
    pnl_net = (pnl_gross - cost).sum(axis=1)

    valid = pnl_net.dropna()
    nav = (1 + valid).cumprod()
    n = len(valid)
    ann_ret = nav.iloc[-1] ** (252 / n) - 1 if n > 0 else 0
    ann_vol = valid.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "ann_ret": ann_ret, "vol": ann_vol, "max_dd": dd,
        "calmar": ann_ret / abs(dd) if dd != 0 else 0,
        "nav": nav, "turnover_annual": turnover.sum().mean() * 252,
    }


def build_final_signal(factors: dict[str, pd.DataFrame],
                       cn_close: pd.DataFrame,
                       z_extreme: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """最终信号: IS(2018-2021)定方向 + 极端化(|z|>阈值才持仓).

    方向由IS期五分位价差(Q5-Q1)符号决定, 避免OOS挑选偏差.
    OOS(2022-2025)按IS方向持有 → 诚实OOS.

    Returns:
        (signal, is_mask): 日频信号面板 + IS期布尔掩码
    """
    is_mask = cn_close.index < "2022-01-01"
    ret5 = cn_close.pct_change(5, fill_method=None).shift(-5)

    signal = pd.DataFrame(0.0, index=cn_close.index, columns=cn_close.columns)
    for fname, fdf in factors.items():
        for prod in fdf.columns:
            f = fdf[prod]
            rv = ret5[prod]
            # IS期五分位价差 → 方向
            is_valid = is_mask & f.notna() & rv.notna()
            fv, rvv = f[is_valid], rv[is_valid]
            if len(fv) < 150:
                continue
            q = pd.qcut(fv.rank(method="first"), 5, labels=False)
            spread = rvv[q == 4].mean() - rvv[q == 0].mean()
            if abs(spread) < 0.001:  # IS年化<5%价差 → 方向不可靠, 跳过
                continue
            direction = np.sign(spread)
            # 极端化: |z|>阈值才持仓 (中间区域噪声大)
            sig = np.sign(f) * direction * (f.abs() > z_extreme).astype(float)
            signal[prod] = signal[prod].fillna(0) + sig.fillna(0)
    return signal.clip(-1, 1), is_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    print("=== Step 1: 加载CFTC周频数据 + 映射 ===")
    cftc = load_cftc_weekly()
    print(f"  映射品种数: {cftc['product'].nunique()}, 行数: {len(cftc)}")
    print(f"  日期范围: {cftc['report_date'].min().date()} ~ {cftc['report_date'].max().date()}")

    print("\n=== Step 1b: 加载国内日频价格 ===")
    cn_close = load_cn_daily_close()
    d0, d1 = cn_close.index.min().date(), cn_close.index.max().date()
    print(f"  品种数: {cn_close.shape[1]}, 日期: {d0} ~ {d1}")

    print("\n=== Step 1c: PIT对齐 (t-3交易日) ===")
    cftc_pit = apply_pit(cftc, cn_close.index)
    print(f"  对齐后: {len(cftc_pit)}行")

    print("\n=== Step 2: 因子构建 (周频52周z-score → PIT对齐) ===")
    factors_weekly = build_factors_weekly(cftc)
    factors = pit_align_weekly(factors_weekly, cn_close.index)
    for k, v in factors.items():
        print(f"  {k}: {v.shape[1]}品种, 周内变化天数占比={(v.diff()!=0).sum().mean():.1%}")

    print("\n=== Step 3: IC验证 (国内品种传导, 未来5日) ===")
    ic = compute_ic(factors, cn_close)
    if len(ic) == 0:
        print("  ❌ 无有效IC数据 — 数据对齐问题, 终止")
        return
    ic = ic.sort_values(["factor", "ic_roll_mean"], ascending=[True, False])
    print(ic.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ic.to_csv(REPORT_DIR / "cftc_ic.csv", index=False)

    # 止损检查 (剔除弱映射)
    strong = ic[~ic["weak_map"]]
    f1_pass = strong[strong["factor"] == "f1_mm"]["ic_roll_mean"].abs().mean()
    print(f"\n强映射品种F1平均|IC|: {f1_pass:.4f} (止损线0.02)")

    if args.ic_only:
        return

    print("\n=== Step 4: 最终策略 (IS定方向 + 极端化) ===")
    cost_table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    cost_bps = {p: (cost_table.get(p, {}).get("slippage_bps", 5.0)
                    + cost_table.get(p, {}).get("commission_bps", 0.1))
                for p in cn_close.columns}

    signal, is_mask = build_final_signal(factors, cn_close)
    active = (signal != 0).sum(axis=1)
    is_max = (signal[is_mask] != 0).astype(int).sum(axis=1).max()
    print(f"  活跃品种-因子对均值: {active.mean():.1f}, IS期最大: {is_max}")

    res = weekly_backtest(signal, cn_close, cost_bps)
    nav = res["nav"]
    ret = nav.pct_change().dropna()

    # 正确换手算法: 日均总换手 × 252
    ret_d = cn_close.pct_change(fill_method=None)
    vol = ret_d.rolling(20).std() * np.sqrt(252)
    scale = (0.08 / vol).clip(upper=1.0).fillna(0)
    pos_check = signal * scale
    true_turnover = pos_check.diff().abs().sum(axis=1).mean() * 252

    is_ret = ret[ret.index < "2022-01-01"]
    oos_ret = ret[ret.index >= "2022-01-01"]

    def stats(r):
        if len(r) < 50:
            return {"sharpe": 0, "ann_ret": 0, "max_dd": 0}
        nav_r = (1 + r).cumprod()
        ann = nav_r.iloc[-1] ** (252 / len(r)) - 1
        dd = (nav_r / nav_r.cummax() - 1).min()
        return {"sharpe": ann / (r.std() * np.sqrt(252)), "ann_ret": ann, "max_dd": dd}

    s_is, s_oos = stats(is_ret), stats(oos_ret)
    print(f"  全期:   Sharpe={res['sharpe']:.3f} AnnRet={res['ann_ret']:.2%} "
          f"MaxDD={res['max_dd']:.2%}")
    print(f"  IS期:   Sharpe={s_is['sharpe']:.3f} AnnRet={s_is['ann_ret']:.2%} "
          f"MaxDD={s_is['max_dd']:.2%}")
    print(f"  OOS期:  Sharpe={s_oos['sharpe']:.3f} AnnRet={s_oos['ann_ret']:.2%} "
          f"MaxDD={s_oos['max_dd']:.2%}")
    print(f"  年化换手(正确口径): {true_turnover:.1f}x")

    results = {"final_is_direction": {**res, "sharpe_is": s_is["sharpe"],
                                      "sharpe_oos": s_oos["sharpe"]}}
    out = pd.DataFrame([{"strategy": k, **{kk: vv for kk, vv in v.items() if kk != "nav"}}
                        for k, v in results.items()])
    out.to_csv(REPORT_DIR / "cftc_weekly_strategy.csv", index=False)
    print(f"\n已保存: {REPORT_DIR}/cftc_{{ic,weekly_strategy}}.csv")


if __name__ == "__main__":
    main()
