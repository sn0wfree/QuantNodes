#!/usr/bin/env python3
"""research_tree_model.py — 树模型预测黄金下一周期收益.

测试RF/GBR等非线性模型能否在12特征上提取比线性更多的信号
对比: 线性基线 (Ridge) vs RandomForest vs GradientBoosting
评估: Pseudo-OOS IC, OOS R², 方向准确率, 多horizon
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "reports/gold"

INDICATORS = {
    "tips10y":("gold_fred","tips10y"),"tips5y":("gold_fred","tips"),
    "rir":("gold_fred","rir"),"us_10y":("gold_macro","us_10y"),
    "us_2y":("gold_macro","us_2y"),"fed_total_assets":("gold_fred","fed_total_assets"),
    "m2":("gold_fred","m2_money_supply"),"comex_inv":("gold_position","comex_inv"),
    "cftc_oi":("gold_position","cftc_oi"),"brent":("fred_dcoil","brent"),
    "wti":("fred_dcoil","wti"),"us_cpi_yoy":("美国CPI","us_cpi_yoy"),
}


def load_data():
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    def load(src, vc):
        df = con.execute("SELECT trade_date, value FROM fundamentals_daily "
                        "WHERE source=? AND value_col=? ORDER BY trade_date", [src, vc]).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")["value"].astype(float)
    gold = load("gold_macro", "gold_spot")
    data = {name: load(s, v) for name, (s, v) in INDICATORS.items()}
    con.close()
    return gold, data


def build_features(gold, data, week_idx):
    """构造12特征的z-score矩阵 (PIT: 用过去数据标准化)."""
    feat = pd.DataFrame(index=week_idx)
    for name, s in data.items():
        s_w = s.reindex(week_idx, method="ffill")
        # 52周Z-score, 仅用过去数据
        m52 = s_w.rolling(52, min_periods=12).mean()
        sd52 = s_w.rolling(52, min_periods=12).std() + 1e-8
        z = (s_w - m52) / sd52
        # shift(1) 确保只用过去数据
        feat[f"{name}_z"] = z.shift(1)
    # 删除warmup期
    return feat.dropna()


def walk_forward(X, y, model_fn, min_train=104, step=52):
    """Walk-forward: 用过去min_train训练, 预测未来step周, 滚动."""
    preds = pd.Series(np.nan, index=y.index)
    common = X.dropna().index.intersection(y.dropna().index)
    X = X.reindex(common)
    y = y.reindex(common)
    for i in range(min_train, len(common)):
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        X_te, y_te = X.iloc[i:i+step], y.iloc[i:i+step]
        if len(X_tr) < 52 or len(X_te) == 0:
            continue
        try:
            m = model_fn()
            m.fit(X_tr.values, y_tr.values)
            preds.iloc[i:i+step] = m.predict(X_te.values)
        except Exception:
            pass
    return preds


def evaluate(y_true, y_pred):
    """评估预测: IC, R², 方向准确率."""
    valid = y_pred.dropna()
    if len(valid) < 10:
        return {"ic": np.nan, "r2": np.nan, "hit": np.nan, "up_hit": np.nan, "down_hit": np.nan}
    y = y_true.reindex(valid.index)
    ss_res = ((y - valid) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    ic = valid.corr(y, method="spearman")
    hit = ((valid > 0) == (y > 0)).mean()
    # 上涨市场方向准确率 (fwd > 0时预测>0的比例)
    up_mask = y > 0
    down_mask = y < 0
    up_hit = (valid[up_mask] > 0).mean() if up_mask.sum() > 0 else np.nan
    down_hit = (valid[down_mask] < 0).mean() if down_mask.sum() > 0 else np.nan
    return {"ic": ic, "r2": r2, "hit": hit, "up_hit": up_hit, "down_hit": down_hit}


def main():
    print("=" * 70)
    print("树模型预测下一周期黄金收益 — 12特征 × 3模型 × 3horizon")
    print("=" * 70)

    gold, data = load_data()
    w = gold.resample("W-FRI").last().dropna()

    # 构造特征
    X = build_features(gold, data, w.index)
    print(f"\n[1] 特征: {X.shape[1]}列, {X.shape[0]}行 (2018-2026)")

    # 多horizon目标
    targets = {
        "FWD1": w.pct_change().shift(-1),
        "FWD2": w.pct_change().shift(-2),
        "FWD4": w.pct_change().shift(-4),
    }

    # 模型 (快速版)
    models = {
        "Ridge (线性基线)": lambda: Ridge(alpha=1.0),
        "RandomForest(n=30)": lambda: RandomForestRegressor(
            n_estimators=30, max_depth=3, min_samples_leaf=40,
            random_state=42, n_jobs=-1),
        "GBR(n=20)": lambda: GradientBoostingRegressor(
            n_estimators=20, max_depth=2, learning_rate=0.1,
            min_samples_leaf=50, random_state=42, subsample=0.8),
    }

    print("\n[2] Walk-forward评估 (train=104周, step=26周, 全2018+窗口)")
    print("=" * 80)
    header = f"{'模型':<22}"
    for h in targets:
        header += f" {h+'IC':>10} {h+'R²':>9} {h+'Hit':>8}"
    print(header)
    print("-" * 80)

    results = {}
    for mname, mfn in models.items():
        row_str = f"{mname:<22}"
        results[mname] = {}
        for hname, y in targets.items():
            preds = walk_forward(X, y, mfn, min_train=104, step=52)
            m = evaluate(y, preds)
            results[mname][hname] = m
            ic_str = f"{m['ic']:+.4f}" if pd.notna(m['ic']) else "    nan"
            r2_str = f"{m['r2']:+.4f}" if pd.notna(m['r2']) else "    nan"
            hit_str = f"{m['hit']:.1%}" if pd.notna(m['hit']) else "    nan"
            row_str += f" {ic_str:>10} {r2_str:>9} {hit_str:>8}"
            print(f"  [{mname}][{hname}] IC={ic_str} R²={r2_str} Hit={hit_str}")
        print(row_str)

    # 详细FWD2的hit率分解
    print("\n[3] FWD2方向准确率分解 (上涨/下跌市场)")
    for mname, mfn in models.items():
        m = results[mname]["FWD2"]
        if pd.notna(m.get("up_hit")):
            print(f"  {mname:<22} 上涨市场 hit={m['up_hit']:.1%}  下跌市场 hit={m['down_hit']:.1%}")

    # 与单最强特征对比
    print("\n[4] 对比: 12特征 Ridge vs Top1单特征(tips10y_z) IC")
    for hname, y in targets.items():
        ic_tips = pd.Series(data["tips10y"]).reindex(w.index, method="ffill")
        ic_tips_z = ((ic_tips - ic_tips.rolling(52, min_periods=12).mean()) / (ic_tips.rolling(52, min_periods=12).std() + 1e-8)).shift(1)
        preds_tips = ic_tips_z.reindex(y.index)  # 不变(基准单特征信号)
        m_tips = evaluate(y, preds_tips)
        m_ridge = results["Ridge (线性基线)"][hname]
        m_rf = results["RandomForest(n=30)"][hname]
        print(f"  {hname}: tips10y单IC={m_tips['ic']:+.4f}  Ridge12={m_ridge['ic']:+.4f}  RF12={m_rf['ic']:+.4f}")

    # 输出CSV
    rows = []
    for mname in models:
        for hname in targets:
            m = results[mname][hname]
            rows.append({
                "model": mname, "horizon": hname,
                "ic": m["ic"], "r2": m["r2"], "hit_rate": m["hit"],
                "up_hit": m["up_hit"], "down_hit": m["down_hit"],
            })
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/tree_model.csv", index=False, float_format="%.4f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/tree_model.csv")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    # 找最佳模型×horizon组合
    valid_results = []
    for mn, hms in results.items():
        for hn, m in hms.items():
            if pd.notna(m["ic"]):
                valid_results.append((mn, hn, abs(m["ic"]), m["ic"]))
    if valid_results:
        best = max(valid_results, key=lambda x: x[2])
        print(f"  最佳IC: {best[0]} × {best[1]} = {best[3]:+.4f}")
    else:
        print("  无有效IC结果")

    # 比较树模型 vs 线性
    for hname in targets:
        ridge_ic = results["Ridge (线性基线)"][hname]["ic"]
        rf_ic = results["RandomForest(n=30)"][hname]["ic"]
        gbr_ic = results["GBR(n=20)"][hname]["ic"]
        best_rf = abs(rf_ic) > abs(ridge_ic)
        best_gbr = abs(gbr_ic) > abs(ridge_ic)
        print(f"  {hname}: Ridge={ridge_ic:+.4f}  RF={rf_ic:+.4f}  GBR={gbr_ic:+.4f}"
              f"  → RF {'胜' if best_rf else '输'}  GBR {'胜' if best_gbr else '输'}")


if __name__ == "__main__":
    main()
