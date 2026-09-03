#!/usr/bin/env python3
"""cta_dynamic_leverage.py — 动态杠杆风控 + 修正成本口径的完整优化.

核心修正 (vs e87a0ce0):
  杠杆成本口径错误 — 旧版把主CTA全部资金视为借款扣3%,
  正确口径只对借入的超额部分 (w_main × (lev-1) × cost) 扣除.

模块:
  1. 静态杠杆扫描 (修正成本后重跑) — 杠杆1.3-1.7 × 主CTA% × 子策略分配
  2. 动态杠杆参数扫描 — vol区间 × 杠杆上限/下限 × vol窗口
     IS期(2018-2021)调参, OOS(2022-2025)验证, 防过拟合
  3. 自适应子策略权重 — 低Sharpe子滚动Sharpe下滑减半 (增益项验证)
  4. 最终组合对比表

杠杆设定: 上限1.5 (用户确认)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import REPORT_DIR  # noqa: E402

warnings.filterwarnings("ignore")

IS_END = "2022-01-01"
LEV_MAX = 1.5      # 用户确认
LEV_COST = 0.03    # 借款年化成本

NAV_FILES = {
    "主CTA": "final_nav_lump.csv",
    "ZN库存": "zn_inventory_nav.csv",
    "CU溢价TC": "cu_premium_nav.csv",
    "IH50动40×OI10": "ih50_oi_nav.csv",
    "T10y OI变化5": "t10y_oi_nav.csv",
    "rbi_steel": "chain_rbi_steel_nav.csv",
    "jjm_coke": "chain_jjm_coke_nav.csv",
    "ym_oilmeal": "chain_ym_oilmeal_nav.csv",
}


def load_all_returns() -> pd.DataFrame:
    rets = {}
    common = None
    for name, fn in NAV_FILES.items():
        nav = pd.read_csv(REPORT_DIR / fn, index_col=0, parse_dates=True).iloc[:, 0]
        r = nav.pct_change(fill_method=None).fillna(0)
        rets[name] = r
        common = r.index if common is None else common.intersection(r.index)
    return pd.DataFrame({k: v.reindex(common) for k, v in rets.items()})


def stats(r: pd.Series, is_end: str = IS_END) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()

    def _one(s):
        if len(s) < 50:
            return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}
        nav_r = (1 + s).cumprod()
        a = nav_r.iloc[-1] ** (252 / len(s)) - 1
        v = s.std() * np.sqrt(252)
        d = (nav_r / nav_r.cummax() - 1).min()
        return {"sharpe": a / v if v > 0 else 0, "ann_ret": a, "max_dd": d}

    return {
        "sharpe": ann / vol if vol > 0 else 0,
        "ann_ret": ann, "vol": vol, "max_dd": dd,
        "calmar": ann / abs(dd) if dd != 0 else 0,
        "IS_Sharpe": _one(r[r.index < is_end])["sharpe"],
        "OOS_Sharpe": _one(r[r.index >= is_end])["sharpe"],
        "OOS_AnnRet": _one(r[r.index >= is_end])["ann_ret"],
        "OOS_MaxDD": _one(r[r.index >= is_end])["max_dd"],
    }


def levered_combo(rets: pd.DataFrame, weights: dict, leverage: float = 1.0,
                   cost_annual: float = LEV_COST) -> pd.Series:
    """正确成本口径的杠杆组合.

    组合 = Σ w_i × sub_i + w_main × lev × main_ret - w_main × (lev-1) × cost/252
    借款只对超额部分 (w_main × (lev-1)) 计息.
    """
    w = pd.Series(weights).reindex(rets.columns).fillna(0)
    combo = (rets * w).sum(axis=1)
    borrow = w.get("主CTA", 0) * max(leverage - 1.0, 0) * cost_annual / 252
    if "主CTA" in w.index and leverage > 1.0:
        combo = combo + w["主CTA"] * (leverage - 1.0) * rets["主CTA"] - borrow
    return combo.fillna(0)


def levered_combo_dynamic(rets: pd.DataFrame, weights: dict, lev_series: pd.Series,
                            cost_annual: float = LEV_COST) -> pd.Series:
    """动态杠杆组合: 逐日杠杆lev_t, 逐日借款成本."""
    w = pd.Series(weights).reindex(rets.columns).fillna(0)
    combo = (rets * w).sum(axis=1)
    if "主CTA" in w.index:
        w_main = w["主CTA"]
        combo = combo + w_main * (lev_series - 1.0) * rets["主CTA"]
        borrow = w_main * (lev_series - 1.0).clip(lower=0) * cost_annual / 252
        combo = combo - borrow
    return combo.fillna(0)


# ────────────────────────────────────────────────────────────────────────────
# Step 1: 修正成本口径后的静态杠杆扫描
# ────────────────────────────────────────────────────────────────────────────

SUB_MODES = {
    "CU+IH": {"CU溢价TC": 0.70, "IH50动40×OI10": 0.30},
    "CU+IH+ZN": {"CU溢价TC": 0.60, "IH50动40×OI10": 0.25, "ZN库存": 0.15},
    "CU+IH+ZN+T10y": {"CU溢价TC": 0.50, "IH50动40×OI10": 0.20,
                       "ZN库存": 0.15, "T10y OI变化5": 0.15},
    "全8策略": {"CU溢价TC": 0.30, "IH50动40×OI10": 0.20, "ZN库存": 0.15,
               "T10y OI变化5": 0.10, "rbi_steel": 0.10,
               "jjm_coke": 0.05, "ym_oilmeal": 0.10},
}


def step1_static_scan(rets: pd.DataFrame) -> pd.DataFrame:
    """修正成本口径后的静态杠杆扫描, 重点杠杆1.5."""
    print("=" * 100)
    print("Step 1: 静态杠杆扫描 (修正成本口径: 借款=w_main×(lev-1), 成本3%)")
    print("=" * 100)

    rows = []
    for lev in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
        for w_main in [0.50, 0.55, 0.60, 0.65, 0.70]:
            sub_pct = 1.0 - w_main
            for mode, sub_w in SUB_MODES.items():
                weights = {"主CTA": w_main}
                for k, v in sub_w.items():
                    weights[k] = v * sub_pct
                combo = levered_combo(rets, weights, leverage=lev)
                st = stats(combo)
                st.update({"杠杆": lev, "主CTA%": w_main, "模式": mode,
                            "总敞口": w_main * lev + sub_pct})
                rows.append(st)

    df = pd.DataFrame(rows)
    df["score"] = df["sharpe"] * 0.3 + df["ann_ret"] * 30 * 0.4 + df["calmar"] * 0.3
    df.to_csv(REPORT_DIR / "static_lev_scan_fixed.csv", index=False)

    # 展示: 杠杆1.5的Top结果
    for lev_show in [1.4, 1.5, 1.6]:
        sub = df[df["杠杆"] == lev_show].sort_values("score", ascending=False).head(5)
        print(f"\n--- 杠杆{lev_show} Top5 (score) ---")
        cols = ["主CTA%", "模式", "sharpe", "ann_ret", "max_dd", "calmar",
                "OOS_Sharpe", "OOS_MaxDD", "总敞口"]
        print(sub[cols].round(4).to_string(index=False))

    # 全局Top10
    print("\n--- 全局Top10 (score) ---")
    cols = ["杠杆", "主CTA%", "模式", "sharpe", "ann_ret", "max_dd", "calmar",
            "OOS_Sharpe", "OOS_MaxDD", "总敞口"]
    print(df.sort_values("score", ascending=False).head(10)[cols].round(4).to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Step 2: 动态杠杆参数扫描 (IS调参 → OOS验证)
# ────────────────────────────────────────────────────────────────────────────

def dynamic_leverage_series(main_ret: pd.Series, vol_window: int,
                              vol_lo: float, vol_hi: float,
                              lev_lo: float, lev_hi: float) -> pd.Series:
    """动态杠杆: vol低→lev_hi, vol高→lev_lo, 中间线性插值."""
    vol = main_ret.rolling(vol_window).std() * np.sqrt(252)
    frac = ((vol - vol_lo) / (vol_hi - vol_lo)).clip(0, 1)
    lev = lev_hi - frac * (lev_hi - lev_lo)
    return lev.fillna(lev_hi)


def step2_dynamic_scan(rets: pd.DataFrame, base_weights: dict) -> pd.DataFrame:
    """动态杠杆参数扫描. IS期选参数, OOS验证."""
    print("\n" + "=" * 100)
    print(f"Step 2: 动态杠杆参数扫描 (lev上限={LEV_MAX}, IS调参→OOS验证)")
    print("=" * 100)

    param_grid = [
        # (vol_window, vol_lo, vol_hi, lev_lo, lev_hi)
        (60, 0.08, 0.12, 1.0, LEV_MAX),   # 诊断默认
        (60, 0.08, 0.12, 0.8, LEV_MAX),
        (60, 0.09, 0.11, 1.0, LEV_MAX),
        (60, 0.07, 0.13, 1.0, LEV_MAX),
        (40, 0.08, 0.12, 1.0, LEV_MAX),
        (40, 0.09, 0.11, 1.0, LEV_MAX),
        (60, 0.08, 0.12, 1.0, 1.4),
        (60, 0.09, 0.11, 1.0, 1.4),
        (40, 0.09, 0.11, 1.0, 1.4),
    ]

    rows = []
    for vw, vlo, vhi, lvlo, lvhi in param_grid:
        lev_s = dynamic_leverage_series(rets["主CTA"], vw, vlo, vhi, lvlo, lvhi)
        combo = levered_combo_dynamic(rets, base_weights, lev_s)
        st = stats(combo)
        is_r = combo[combo.index < IS_END]
        oos_r = combo[combo.index >= IS_END]

        def _calmar(s):
            if len(s) < 50:
                return 0.0
            nav = (1 + s).cumprod()
            a = nav.iloc[-1] ** (252 / len(s)) - 1
            d = (nav / nav.cummax() - 1).min()
            return a / abs(d) if d != 0 else 0

        st.update({
            "参数": f"vw{vw} vol[{vlo},{vhi}] lev[{lvlo},{lvhi}]",
            "IS_Calmar": _calmar(is_r), "OOS_Calmar": _calmar(oos_r),
            "平均杠杆": lev_s.mean(),
        })
        rows.append(st)

    # 对照: 静态杠杆1.5 (无风控)
    combo_static = levered_combo(rets, base_weights, leverage=LEV_MAX)
    st_static = stats(combo_static)
    st_static.update({"参数": "静态1.5(无风控)", "IS_Calmar": None, "OOS_Calmar": None,
                       "平均杠杆": LEV_MAX})
    rows.insert(0, st_static)

    df = pd.DataFrame(rows)
    cols = ["参数", "sharpe", "ann_ret", "max_dd", "calmar",
            "IS_Sharpe", "OOS_Sharpe", "OOS_AnnRet", "OOS_MaxDD", "平均杠杆"]
    print()
    print(df[cols].round(4).to_string(index=False))
    df.to_csv(REPORT_DIR / "dynamic_lev_scan.csv", index=False)

    # 参数稳定性: OOS Calmar排名
    print("\n--- OOS_Calmar排序 (验证泛化) ---")
    valid = df[df["OOS_Calmar"].notna()].sort_values("OOS_Calmar", ascending=False)
    show_cols = ["参数", "IS_Calmar", "OOS_Calmar", "ann_ret", "max_dd"]
    print(valid[show_cols].round(4).to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────────
# Step 3: 自适应子策略权重
# ────────────────────────────────────────────────────────────────────────────

def step3_adaptive(rets: pd.DataFrame, base_weights: dict, lev_s: pd.Series) -> dict:
    """自适应子策略权重: 子策略滚动60日Sharpe较全期中位数下滑>50% → 权重减半."""
    print("\n" + "=" * 100)
    print("Step 3: 自适应子策略权重 (滚动60日Sharpe vs 全期中位, 下滑50%→减半)")
    print("=" * 100)

    modulated = {}
    for col in rets.columns:
        if col == "主CTA":
            modulated[col] = rets[col]
            continue
        r = rets[col]
        # 滚动60日Sharpe
        roll_sharpe = r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)
        full_median = roll_sharpe.expanding(min_periods=120).median()
        factor = pd.Series(1.0, index=r.index)
        factor[(roll_sharpe < full_median * 0.5) & (roll_sharpe < 0)] = 0.5
        modulated[col] = r * factor.fillna(1.0)

    rets_adj = pd.DataFrame(modulated)
    combo = levered_combo_dynamic(rets_adj, base_weights, lev_s)
    st = stats(combo)
    st["参数"] = "自适应子权重+动态杠杆"

    # 对照: 相同动态杠杆但无自适应
    combo_base = levered_combo_dynamic(rets, base_weights, lev_s)
    st_base = stats(combo_base)
    st_base["参数"] = "动态杠杆(无自适应)"

    df = pd.DataFrame([st_base, st])
    cols = ["参数", "sharpe", "ann_ret", "max_dd", "calmar",
            "IS_Sharpe", "OOS_Sharpe", "OOS_AnnRet", "OOS_MaxDD"]
    print()
    print(df[cols].round(4).to_string(index=False))
    df.to_csv(REPORT_DIR / "adaptive_weight_test.csv", index=False)
    return {"base": st_base, "adaptive": st}


# ────────────────────────────────────────────────────────────────────────────
# Step 4: 最终组合 (动态杠杆 + 最优静态配置) IS/OOS 双验证
# ────────────────────────────────────────────────────────────────────────────

def step4_final(rets: pd.DataFrame, base_weights: dict, best_params: tuple) -> dict:
    print("\n" + "=" * 100)
    print("Step 4: 最终组合 — 动态杠杆(最优参数) + 静态配置 IS/OOS双验证")
    print("=" * 100)

    vw, vlo, vhi, lvlo, lvhi = best_params
    lev_s = dynamic_leverage_series(rets["主CTA"], vw, vlo, vhi, lvlo, lvhi)

    results = []

    # 各基准
    results.append({**stats(rets["主CTA"]), "组合": "基准: 主CTA单独"})
    combo_s15 = levered_combo(rets, base_weights, leverage=LEV_MAX)
    results.append({**stats(combo_s15), "组合": f"静态杠杆{LEV_MAX}(无风控)"})

    combo_dyn = levered_combo_dynamic(rets, base_weights, lev_s)
    dyn_label = f"动态杠杆 vw{vw} vol[{vlo},{vhi}] lev[{lvlo},{lvhi}]"
    results.append({**stats(combo_dyn), "组合": dyn_label})

    # 动态杠杆 + 自适应
    modulated = {}
    for col in rets.columns:
        if col == "主CTA":
            modulated[col] = rets[col]
            continue
        r = rets[col]
        roll_sharpe = r.rolling(60).mean() / r.rolling(60).std() * np.sqrt(252)
        full_median = roll_sharpe.expanding(min_periods=120).median()
        factor = pd.Series(1.0, index=r.index)
        factor[(roll_sharpe < full_median * 0.5) & (roll_sharpe < 0)] = 0.5
        modulated[col] = r * factor.fillna(1.0)
    combo_full = levered_combo_dynamic(pd.DataFrame(modulated), base_weights, lev_s)
    results.append({**stats(combo_full), "组合": "动态杠杆+自适应子权重"})

    df = pd.DataFrame(results)
    cols = ["组合", "sharpe", "ann_ret", "max_dd", "calmar",
            "IS_Sharpe", "OOS_Sharpe", "OOS_AnnRet", "OOS_MaxDD"]
    print()
    print(df[cols].round(4).to_string(index=False))
    df.to_csv(REPORT_DIR / "final_dynamic_combo.csv", index=False)

    # 年度收益
    print("\n--- 最终组合年度收益 ---")
    yr = combo_dyn.groupby(combo_dyn.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr_main = rets["主CTA"].groupby(rets["主CTA"].index.year).apply(lambda x: (1 + x).prod() - 1)
    yr_df = pd.DataFrame({"动态杠杆组合": yr, "主CTA单独": yr_main}).round(4)
    print(yr_df.to_string())

    # 保存最终NAV
    (1 + combo_dyn).cumprod().to_csv(REPORT_DIR / "final_dynamic_lev_nav.csv", header=["nav"])
    print("\nNAV保存: reports/cta/final_dynamic_lev_nav.csv")
    return df


def main():
    rets = load_all_returns()
    print(f"数据范围: {rets.index.min().date()} ~ {rets.index.max().date()} ({len(rets)}天)\n")

    df_static = step1_static_scan(rets)

    # 用修正口径下的最优静态配置作为动态杠杆的基准权重
    best = df_static[df_static["杠杆"] == LEV_MAX].sort_values("score", ascending=False).iloc[0]
    w_main = best["主CTA%"]
    mode = best["模式"]
    sub_pct = 1.0 - w_main
    base_weights = {"主CTA": w_main}
    for k, v in SUB_MODES[mode].items():
        base_weights[k] = v * sub_pct
    print(f"\n★ 修正口径下 杠杆{LEV_MAX} 最优配置: 主CTA={w_main:.0%}, 模式={mode}")
    print(f"  weights: {base_weights}")

    df_dyn = step2_dynamic_scan(rets, base_weights)

    # IS_Calmar最优的参数 (IS调参)
    dyn_valid = df_dyn[df_dyn["IS_Calmar"].notna()]
    best_param_row = dyn_valid.sort_values("IS_Calmar", ascending=False).iloc[0]
    print(f"\n★ IS期最优动态杠杆参数: {best_param_row['参数']}")
    print(f"  IS_Calmar={best_param_row['IS_Calmar']:.2f} "
          f"→ OOS_Calmar={best_param_row['OOS_Calmar']:.2f}")

    # 解析参数
    p = best_param_row["参数"]
    import re
    m = re.match(r"vw(\d+) vol\[([\d.]+),([\d.]+)\] lev\[([\d.]+),([\d.]+)\]", p)
    best_params = (int(m.group(1)), float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5)))

    step3_adaptive(rets, base_weights, dynamic_leverage_series(rets["主CTA"], *best_params))
    step4_final(rets, base_weights, best_params)


if __name__ == "__main__":
    main()
