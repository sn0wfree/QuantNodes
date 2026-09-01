#!/usr/bin/env python3
"""research_feature_engineering.py — 特征工程研究.

对每个主要指标提取: 变化率(1/5/20/60d) / Z-score / 分位数 / 偏度 / 峰度
测试每个特征对gold周收益的IC + walk-forward OLS联合预测力
输出: 哪些特征真正有增量预测力
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import duckdb
import numpy as np
import pandas as pd

OUTPUT_DIR = "reports/gold"

INDICATORS = {
    "vix": ("gold_fred", "vix"),
    "gvz": ("gold_fred", "gvz"),
    "tips10y": ("gold_fred", "tips10y"),
    "tips5y": ("gold_fred", "tips"),
    "dxy": ("gold_macro", "dxy"),
    "gold_spot": ("gold_macro", "gold_spot"),
    "etf_tonnes": ("gold_position", "etf_tonnes"),
    "fed_assets": ("gold_fred", "fed_total_assets"),
    "m2": ("gold_fred", "m2_money_supply"),
}


def load_data():
    """加载所有指标和gold周频."""
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    data = {}
    for name, (src, vc) in INDICATORS.items():
        df = con.execute(
            "SELECT trade_date, value FROM fundamentals_daily "
            "WHERE source=? AND value_col=? ORDER BY trade_date",
            [src, vc]
        ).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        s = df.set_index("trade_date")["value"].astype(float)
        data[name] = s
    con.close()
    widx = data["gold_spot"].resample("W-FRI").last().dropna().index
    weekly = {}
    for k, s in data.items():
        weekly[k] = s.reindex(widx, method="ffill")
    weekly["fwd_ret"] = weekly["gold_spot"].pct_change().shift(-1)
    return weekly


def build_features(w):
    """对每个指标生成7个特征."""
    features = {}
    for name, s in w.items():
        if name == "fwd_ret" or name == "gold_spot":
            continue
        s = s.dropna()
        if len(s) < 100:
            continue
        # 1. 水平
        features[f"{name}_level"] = s
        # 2-5. 变化率(1/5/20/60日)
        for n in [1, 5, 20, 60]:
            chg = s.pct_change(n)
            if chg.std() > 1e-10:
                features[f"{name}_chg{n}d"] = chg
        # 6. Z-score (52周滚动)
        m = s.rolling(52).mean()
        sd = s.rolling(52).std()
        if sd.notna().sum() > 0:
            features[f"{name}_zscore"] = (s - m) / (sd + 1e-8)
        # 7. 分位数(52周)
        features[f"{name}_pctile"] = s.rolling(52).rank(pct=True)
    return features


def weekly_ic(feature, fwd_ret):
    """Spearman IC (周频, 跳过前52行不稳定的zscore/pctile)."""
    common = feature.dropna().index.intersection(fwd_ret.dropna().index)
    if len(common) < 50:
        return np.nan
    return feature.reindex(common).corr(fwd_ret.reindex(common), method="spearman")


def walk_forward_ols(features, fwd_ret, train_min=104, step=52):
    """walk-forward OLS: 用过去train_min周训练, 预测接下来step周, 滚动.
    返回每期的预测值和OOS R²."""
    common = fwd_ret.dropna().index
    for f in features.values():
        common = common.intersection(f.dropna().index)
    common = common.sort_values()
    if len(common) < train_min + step:
        return np.nan, 0

    feat_df = pd.DataFrame({k: f.reindex(common) for k, f in features.items()})
    y = fwd_ret.reindex(common)

    # 标准化每个特征 (滚动窗口)
    pred = pd.Series(np.nan, index=common)
    for i in range(train_min, len(common)):
        # 训练集: 前i周
        X_train = feat_df.iloc[:i]
        y_train = y.iloc[:i]
        # 去除NaN行
        mask = X_train.notna().all(axis=1) & y_train.notna()
        if mask.sum() < train_min // 2:
            continue
        X_tr = X_train[mask].values
        y_tr = y_train[mask].values
        # 标准化
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-8
        X_tr_std = (X_tr - mu) / sd
        # OLS
        try:
            beta = np.linalg.lstsq(X_tr_std, y_tr, rcond=None)[0]
            # 预测本期
            x_today = (feat_df.iloc[i].values - mu) / sd
            pred.iloc[i] = x_today @ beta
        except Exception:
            pass

    valid = pred.dropna()
    y_valid = y.reindex(valid.index)
    if len(valid) < 10:
        return np.nan, 0
    ss_res = ((y_valid - valid) ** 2).sum()
    ss_tot = ((y_valid - y_valid.mean()) ** 2).sum()
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return valid.corr(y_valid, method="spearman"), r2_oos


def main():
    print("=" * 70)
    print("特征工程研究 — 从9个指标提取~40个特征, 测试预测力")
    print("=" * 70)

    print("\n[1] 加载数据...")
    w = load_data()
    fwd = w["fwd_ret"]
    print(f"  周频样本: {len(w['gold_spot'])}周 ({w['gold_spot'].index[0].date()} ~ {w['gold_spot'].index[-1].date()})")

    print("[2] 构建特征...")
    features = build_features(w)
    print(f"  共生成 {len(features)} 个特征")

    print("\n[3] 单特征IC (全期, 2018-2026)...")
    print(f"  {'特征':<35} {'IC':>8} {'|IC|':>8}")
    print("  " + "-" * 55)

    ic_table = []
    for name, feat in features.items():
        ic = weekly_ic(feat, fwd)
        ic_table.append({"feature": name, "ic": ic, "abs_ic": abs(ic) if pd.notna(ic) else 0})
    ic_df = pd.DataFrame(ic_table).sort_values("abs_ic", ascending=False)

    for _, r in ic_df.head(20).iterrows():
        marker = "🔵" if r["ic"] > 0.05 else "⚪"
        print(f"  {r['feature']:<35} {r['ic']:>+8.4f} {r['abs_ic']:>8.4f} {marker}")

    print(f"\n  ... (共{len(ic_df)}个特征, 完整数据已保存到CSV)")

    # 子周期稳健性检查
    print("\n[4] 稳健性检查: Top 10特征 × 3子周期...")
    top_features = ic_df.head(10)["feature"].tolist()
    periods = [("2018-2020", "2018-01-01", "2020-12-31"),
               ("2021-2023", "2021-01-01", "2023-12-31"),
               ("2024-2026", "2024-01-01", "2026-08-31")]
    print(f"  {'特征':<35} {'P1':>7} {'P2':>7} {'P3':>7} {'稳健'}")
    print("  " + "-" * 65)
    for fname in top_features:
        sics = []
        for _, s, e in periods:
            mask = (fwd.index >= s) & (fwd.index <= e)
            sub_fwd = fwd[mask]
            sub_feat = features[fname].reindex(sub_fwd.index)
            ic = weekly_ic(sub_feat, sub_fwd)
            sics.append(ic)
        robust = "✅" if all(pd.notna(s) and s > 0 for s in sics) else "⚠️"
        print(f"  {fname:<35} {sics[0]:>+7.3f} {sics[1]:>+7.3f} {sics[2]:>+7.3f} {robust}")

    # walk-forward OLS 联合预测
    print("\n[5] walk-forward OLS (Top 10特征联合预测gold周收益)...")
    top_10 = {k: features[k] for k in top_features}
    ic_oos, r2 = walk_forward_ols(top_10, fwd, train_min=104, step=52)
    print(f"  OOS Spearman IC: {ic_oos:+.4f}")
    print(f"  OOS R²: {r2:+.4f}")
    if pd.notna(ic_oos) and ic_oos > 0.1:
        print(f"  ⭐ Top 10特征联合有正OOS预测力")

    # 输出CSV
    out = ic_df.copy()
    out = out.sort_values("abs_ic", ascending=False)
    out.to_csv(f"{OUTPUT_DIR}/feature_ic.csv", index=False, float_format="%.6f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/feature_ic.csv ({len(out)}行)")

    print("\n" + "=" * 70)
    print("结论摘要")
    print("=" * 70)
    print(f"  单特征最佳: {ic_df.iloc[0]['feature']} (IC={ic_df.iloc[0]['ic']:+.4f})")
    print(f"  联合OOS IC: {ic_oos:+.4f}  R²: {r2:+.4f}")
    if r2 > 0.01:
        print(f"  ⭐ 联合特征解释了{abs(r2)*100:.1f}%的gold周收益方差")
    else:
        print(f"  ⚠️ 联合特征预测力弱 (< 1%方差), 因子构造已达天花板")


if __name__ == "__main__":
    main()
