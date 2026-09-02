#!/usr/bin/env python3
"""cta_inventory_strategy.py — 库存因子周频/日频子策略 (Tier2).

数据:
    inventory_daily: LME铜/铝/锌/镍库存 + EIA原油库存变动
    tfapi_indicators: LME/COMEX/SHFE铜库存, 豆粕/大豆库存, 压榨利润

因子 (6个):
    F1: 库存z-score (52周) — 极端库存→反转
    F2: 库存变化率 (4周) — 去库=涨价(趋势)
    F3: 变化率z-score (26周) — 标准化去库强度
    F4: 仓单比 (LME特有) — 供应紧张度
    F5: 趋势强度 (20周动量) — 去库/累库持续性
    F6: 季节性 (当月 vs 历史同月) — 季节性异常

用法:
    python3 scripts/cta_inventory_strategy.py          # 全流程
    python3 scripts/cta_inventory_strategy.py --ic-only # 仅IC验证
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

# ── Step 1: 数据加载 ────────────────────────────────────────────────────────

# 库存数据源映射: (source, value_col) → 国内品种
INVENTORY_MAP = {
    ("LME铜库存", "铜-库存"): "CU",
    ("LME铜库存", "铜-注册仓单"): "CU_warrant",
    ("LME铜库存", "铝-库存"): "AL",
    ("LME铜库存", "锌-库存"): "ZN",
    ("LME铜库存", "镍-库存"): "NI",
    ("EIA原油库存变动", "eia_chg_kbd"): "SC",
}

# tfapi_indicators映射: index_name → 国内品种+value_col
INDICATOR_MAP = {
    ("COMEX:铜:库存", "value"): "CU_comex",
    ("SHFE库存小计", "value"): "CU_shfe",
    ("豆粕库存_中国", "value"): "M",
    ("大豆库存_中国", "value"): "A",
}


def load_inventory_daily() -> dict[str, pd.Series]:
    """加载inventory_daily表, 按source+value_col拆分成Series."""
    con = duckdb.connect(CMF_DB, read_only=True)
    out = {}
    for (src, vcol), cn in INVENTORY_MAP.items():
        df = con.execute(
            "SELECT trade_date, value FROM inventory_daily "
            "WHERE source=? AND value_col=? ORDER BY trade_date",
            [src, vcol],
        ).df()
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        series = df.set_index("trade_date")["value"].sort_index()
        series.name = cn
        out[cn] = series
    con.close()
    return out


def load_indicators() -> dict[str, pd.Series]:
    """加载tfapi_indicators表, 按index_name拆分."""
    con = duckdb.connect(CMF_DB, read_only=True)
    out = {}
    for (idx_name, _), cn in INDICATOR_MAP.items():
        df = con.execute(
            "SELECT trade_date, value FROM tfapi_indicators "
            "WHERE index_name=? ORDER BY trade_date",
            [idx_name],
        ).df()
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        series = df.set_index("trade_date")["value"].sort_index()
        series.name = cn
        out[cn] = series
    con.close()
    return out


def load_cn_daily_close() -> pd.DataFrame:
    """国内品种日频收盘价 (smoothed连续合约)."""
    con = duckdb.connect(CTA_DB, read_only=True)
    out = {}
    products = ["CU", "AL", "ZN", "NI", "SC", "M", "A", "B"]
    for prod in products:
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


# ── Step 2: 因子工程 ────────────────────────────────────────────────────────

ZSCORE_WINDOW = 52


def build_inventory_factors(inv_data: dict[str, pd.Series],
                            ind_data: dict[str, pd.Series]) -> pd.DataFrame:
    """构建库存因子面板 (日频, 仅库存数据可用的品种).

    F1: 库存z-score — 极端库存→反转
    F2: 库存变化率 (4周) — 去库=涨价(趋势)
    F3: 变化率z-score — 标准化去库强度
    F4: 仓单比 (LME特有, 铜/铝)
    F5: 趋势强度 (20周动量) — 去库/累库持续性
    F6: 季节性 — 当月库存 vs 历史同月均值
    """
    # 优先用tfapi_indicators的铜库存（LME/COMEX/SHFE更权威）
    # 如果没有则用inventory_daily
    cu_inv = ind_data.get("CU_shfe") if "CU_shfe" in ind_data else (
        ind_data.get("CU_comex") if "CU_comex" in ind_data else inv_data.get("CU")
    )
    al_inv = inv_data.get("AL")
    zn_inv = inv_data.get("ZN")
    ni_inv = inv_data.get("NI")
    sc_inv = inv_data.get("SC")
    m_inv = ind_data.get("M") if "M" in ind_data else None
    a_inv = ind_data.get("A") if "A" in ind_data else None
    cu_warrant = inv_data.get("CU_warrant")

    # 转为DataFrame (不同频率/起始日，align到并集)
    all_series = []
    for s in [cu_inv, al_inv, zn_inv, ni_inv, sc_inv, m_inv, a_inv]:
        if s is not None:
            all_series.append(s)
    if cu_warrant is not None:
        all_series.append(cu_warrant)

    if not all_series:
        return pd.DataFrame()

    inv_df = pd.concat(all_series, axis=1)
    # 品种重命名
    inv_df.columns = [s.name for s in all_series]

    factors = pd.DataFrame(index=inv_df.index)

    # F1: 库存z-score (52周)
    for col in inv_df.columns:
        s = inv_df[col]
        m = s.rolling(ZSCORE_WINDOW, min_periods=26).mean()
        sd = s.rolling(ZSCORE_WINDOW, min_periods=26).std().replace(0, np.nan)
        factors[f"{col}_zscore"] = (s - m) / sd

    # F2: 库存变化率 (4周)
    for col in inv_df.columns:
        factors[f"{col}_chg4w"] = inv_df[col].pct_change(20)  # 4周=20交易日

    # F3: 变化率z-score (26周)
    for col in inv_df.columns:
        chg = inv_df[col].pct_change(20)
        m = chg.rolling(130, min_periods=26).mean()  # 26周=130天
        sd = chg.rolling(130, min_periods=26).std().replace(0, np.nan)
        factors[f"{col}_chg_zscore"] = (chg - m) / sd

    # F4: 仓单比 (仅铜/铝有warrant数据)
    for base in ["CU", "AL"]:
        warrant = inv_data.get(f"{base}_warrant")
        stock = inv_data.get(base)
        if warrant is not None and stock is not None:
            aligned = pd.concat([warrant, stock], axis=1).dropna()
            if len(aligned) > 50:
                factors[f"{base}_warrant_ratio"] = aligned.iloc[:, 0] / aligned.iloc[:, 1]

    # F5: 趋势强度 (20周动量 = 100交易日)
    for col in inv_df.columns:
        factors[f"{col}_trend20w"] = inv_df[col].pct_change(100)

    # F6: 季节性 (当月库存 vs 历史同月均值)
    for col in inv_df.columns:
        s = inv_df[col]
        monthly = s.groupby(s.index.month).transform("mean")
        historical = s.groupby(s.index.month).transform(
            lambda x: x.expanding(min_periods=24).mean()
        )
        factors[f"{col}_seasonal"] = (s - historical) / historical.replace(0, np.nan)

    return factors


# ── Step 3: IC检验 ──────────────────────────────────────────────────────────

def compute_ic(factors: pd.DataFrame, cn_close: pd.DataFrame,
               horizon_days: int = 5) -> pd.DataFrame:
    """因子IC: 因子值 vs 国内品种未来N日收益 (时序corr)."""
    ret = cn_close.pct_change(fill_method=None)
    fwd = ret.shift(-horizon_days)

    rows = []
    for fname in factors.columns:
        # 提取品种名 (去掉_f1后缀)
        parts = fname.rsplit("_", 1)
        prod = parts[0]
        if prod not in fwd.columns:
            continue
        f = factors[fname].dropna()
        r = fwd[prod].reindex(f.index)
        valid = f.notna() & r.notna()
        if valid.sum() < 200:
            continue
        # 全样本IC
        ic = f[valid].corr(r[valid])
        # 滚动IC均值 + t-stat
        roll_ic = f[valid].rolling(52).corr(r[valid]).dropna()
        t_stat = roll_ic.mean() / (roll_ic.std() / np.sqrt(len(roll_ic))) if len(roll_ic) > 10 else np.nan
        # 五分位价差 (年化)
        fv, rv = f[valid], r[valid]
        q = pd.qcut(fv.rank(method="first"), 5, labels=False)
        spread_ann = (rv[q == 4].mean() - rv[q == 0].mean()) * 252 / 5
        rows.append({
            "factor": fname, "product": prod, "ic": ic,
            "ic_roll_mean": roll_ic.mean(), "t_stat": t_stat,
            "spread_ann": spread_ann, "n": int(valid.sum()),
        })
    return pd.DataFrame(rows)


# ── Step 4: IS/OOS方向一致性 ────────────────────────────────────────────────

def is_oos_consistency(factors: pd.DataFrame, cn_close: pd.DataFrame,
                       is_end: str = "2022-01-01") -> pd.DataFrame:
    """IS/OOS五分位价差方向一致性检验 (每因子×品种)."""
    ret5 = cn_close.pct_change(5, fill_method=None).shift(-5)
    is_mask = factors.index < is_end
    oos_mask = factors.index >= is_end

    rows = []
    for fname in factors.columns:
        parts = fname.rsplit("_", 1)
        prod = parts[0]
        if prod not in ret5.columns:
            continue
        f = factors[fname]
        rv = ret5[prod]
        for mask, label in [(is_mask, "IS"), (oos_mask, "OOS")]:
            fv, rvv = f[mask & f.notna() & rv.notna()], rv[mask & f.notna() & rv.notna()]
            if len(fv) < 150:
                continue
            q = pd.qcut(fv.rank(method="first"), 5, labels=False)
            spread = (rvv[q == 4].mean() - rvv[q == 0].mean()) * 252 / 5
            rows.append({"factor": fname, "product": prod, "period": label, "spread_ann": spread})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    # pivot: IS vs OOS
    is_s = df[df["period"] == "IS"].set_index(["factor", "product"])["spread_ann"]
    oos_s = df[df["period"] == "OOS"].set_index(["factor", "product"])["spread_ann"]
    aligned = pd.DataFrame({"is_spread": is_s, "oos_spread": oos_s}).dropna()
    aligned["direction_agree"] = np.sign(aligned["is_spread"]) == np.sign(aligned["oos_spread"])
    aligned["is_significant"] = aligned["is_spread"].abs() > 0.05  # 年化5%价差
    return aligned


# ── Step 5: 回测 ──────────────────────────────────────────────────────────

def daily_backtest(signal: pd.DataFrame, cn_close: pd.DataFrame,
                   cost_bps: dict, target_vol: float = 0.08,
                   vol_window: int = 20) -> dict:
    """日频回测 (库存是日频数据)."""
    ret = cn_close.pct_change(fill_method=None)
    pos_raw = signal.copy()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=1.0).fillna(0)
    pos = pos_raw * scale

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
    return {
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "ann_ret": ann_ret, "vol": ann_vol, "max_dd": dd,
        "calmar": ann_ret / abs(dd) if dd != 0 else 0,
        "nav": nav,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true")
    args = parser.parse_args()

    print("=== Step 1: 加载库存数据 ===")
    inv = load_inventory_daily()
    ind = load_indicators()
    cn_close = load_cn_daily_close()
    print(f"  inventory_daily: {len(inv)}个Series")
    print(f"  tfapi_indicators: {len(ind)}个Series")
    print(f"  国内价格: {cn_close.shape[1]}品种, {cn_close.index.min().date()}~{cn_close.index.max().date()}")

    print("\n=== Step 2: 因子构建 (6维度) ===")
    factors = build_inventory_factors(inv, ind)
    # 对齐到国内交易日
    factors = factors.reindex(cn_close.index)
    # drop全NaN列
    factors = factors.dropna(axis=1, how="all")
    print(f"  因子数: {factors.shape[1]}, 日期: {factors.index.min().date()}~{factors.index.max().date()}")
    for col in factors.columns[:6]:
        print(f"    {col}: {factors[col].notna().sum()}天有效")

    print("\n=== Step 3: IC检验 ===")
    ic = compute_ic(factors, cn_close)
    if len(ic) == 0:
        print("  ❌ 无有效IC数据")
        return
    ic = ic.sort_values("ic_roll_mean", ascending=False)
    print(f"  因子-品种对数: {len(ic)}")
    # 过滤|IC|>0.02
    sig = ic[ic["ic_roll_mean"].abs() > 0.02]
    print(f"  |IC|>0.02: {len(sig)}个")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ic.to_csv(REPORT_DIR / "inventory_ic.csv", index=False)
    print(sig[["factor", "product", "ic_roll_mean", "t_stat", "spread_ann"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Step 4: IS/OOS方向一致性 ===")
    cons = is_oos_consistency(factors, cn_close)
    if len(cons) == 0:
        print("  ❌ 无有效数据")
        return
    sig_cons = cons[cons["is_significant"]]
    agree = sig_cons["direction_agree"].sum()
    total = len(sig_cons)
    print(f"  IS显著品种-因子对: {total}, 方向一致: {agree} ({agree/max(total,1):.0%})")
    cons.to_csv(REPORT_DIR / "inventory_is_oos.csv")
    print(sig_cons.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    if args.ic_only:
        return

    print("\n=== Step 5: IS/OOS定方向回测 ===")
    cost_table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    cost_bps = {p: (cost_table.get(p, {}).get("slippage_bps", 5.0)
                    + cost_table.get(p, {}).get("commission_bps", 0.1))
                for p in cn_close.columns}

    is_mask = factors.index < "2022-01-01"
    ret5 = cn_close.pct_change(5, fill_method=None).shift(-5)

    signal = pd.DataFrame(0.0, index=factors.index, columns=cn_close.columns)
    for fname in factors.columns:
        parts = fname.rsplit("_", 1)
        prod = parts[0]
        if prod not in cn_close.columns:
            continue
        f, rv = factors[fname], ret5[prod]
        is_valid = is_mask & f.notna() & rv.notna()
        fv, rvv = f[is_valid], rv[is_valid]
        if len(fv) < 150:
            continue
        q = pd.qcut(fv.rank(method="first"), 5, labels=False)
        spread = rvv[q == 4].mean() - rvv[q == 0].mean()
        if abs(spread) < 0.001:
            continue
        direction = np.sign(spread)
        signal[prod] += (np.sign(f) * direction).fillna(0)
    signal = signal.clip(-1, 1)

    res = daily_backtest(signal, cn_close, cost_bps)
    nav = res["nav"]
    ret = nav.pct_change().dropna()
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
    print(f"  全期:   Sharpe={res['sharpe']:.3f} AnnRet={res['ann_ret']:.2%} MaxDD={res['max_dd']:.2%}")
    print(f"  IS期:   Sharpe={s_is['sharpe']:.3f} AnnRet={s_is['ann_ret']:.2%} MaxDD={s_is['max_dd']:.2%}")
    print(f"  OOS期:  Sharpe={s_oos['sharpe']:.3f} AnnRet={s_oos['ann_ret']:.2%} MaxDD={s_oos['max_dd']:.2%}")


if __name__ == "__main__":
    main()
