# coding: utf-8
"""e1_production_eval.py — ZK vol_adj 的 production 口径评估 (独立脚本).

不修改任何现有脚本; 纯 import 复用:
  - cta_t10y_removal.load_rets / run_portfolio (§27 组合管线)
  - cta_full_assessment_export.stats_of (指标口径)
  - scripts.zookeeper.integrate (ZK 月度状态变量)

方法 (P7-B simulate 的 production 正式化):
  1. baseline: 主CTA收益 = production NAV (overfit_final_navs/基线_*.parquet)
  2. ZK版:     主CTA收益 × 月度调节因子 (多stat集成, α=0.5, dir=-1)
  3. 两版分别跑 §27 五版本组合 (run_portfolio), 输出 5版本 × 3窗口 对比表
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_full_assessment_export import stats_of  # noqa: E402
from cta_t10y_removal import load_rets, run_portfolio  # noqa: E402

from scripts.zookeeper.integrate import (  # noqa: E402
    _load_zk_state_variable,
    _normalize_state,
)

REPORT_DIR = "reports/cta"
ZK_DIR = f"{REPORT_DIR}/zookeeper"

# ZK 调节配置 (E3 推荐: 多stat等权集成, α=0.5, dir=-1)
ZK_CFG = {
    "method": "weighted",
    "alpha": 0.5,
    "direction": -1,
    "weights": {"signal_strength": 1/3, "signal_dispersion": 1/3,
                "signal_momentum": 1/3},
}

# §27 五版本 + E3-R + E4 (与 cta_full_assessment_export.main 一致)
VERSIONS = {
    "主CTA本体(golden)": None,  # 特殊: 纯主CTA收益序列, 不走 run_portfolio
    "R2-R(55/45)": {"主CTA": 0.55, "CU溢价TC": 32.14, "jjm_coke": 12.86},
    "E2-R(45/55)": {"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69},
    "E2-R+AU10%": {"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69,
                   "AU_layered": 10},
    "E3-R(主CTA70)": {"主CTA": 0.70, "CU溢价TC": 10, "jjm_coke": 10,
                      "AU_layered": 10},
    "E4(主CTA75/CU5)": {"主CTA": 0.75, "CU溢价TC": 5, "jjm_coke": 10,
                        "AU_layered": 10},
    "B2满配(子≤10%)": {"主CTA": 0.60, "CU溢价TC": 10, "jjm_coke": 10,
                       "AU_layered": 10, "ZN库存": 10},
}

WINDOWS = [("全样本19-25", None, None), ("IS 19-21", None, "2022-01-01"),
           ("OOS 22-25", "2022-01-01", None)]


def build_zk_adjust_factor() -> pd.Series:
    """ZK 多 stat 等权集成 → 月度调节因子 (P7-B 公式)."""
    frames = {}
    for stat in ZK_CFG["weights"]:
        s = _load_zk_state_variable(method=ZK_CFG["method"], stat=stat)
        frames[stat] = _normalize_state(s)
    common = frames[next(iter(ZK_CFG["weights"]))].index
    for s in frames.values():
        common = common.intersection(s.index)
    composite = pd.Series(0.0, index=common)
    for stat, w in ZK_CFG["weights"].items():
        composite = composite + frames[stat].reindex(common) * w
    adj = 1 + ZK_CFG["direction"] * ZK_CFG["alpha"] * (composite - 0.5) * 2
    return adj.clip(lower=0.3, upper=2.0)


def scale_main_cta(rets: pd.DataFrame, adj_monthly: pd.Series) -> pd.DataFrame:
    """主CTA日收益 × 月度调节因子 (月度→日频映射, 调节因子缺失段=1.0)."""
    out = rets.copy()
    idx = rets.index
    month_key = idx.to_period("M")
    # 月度调节因子对齐到日频: 当月所有日收益乘同一因子
    adj_map = adj_monthly.copy()
    adj_map.index = adj_map.index.to_period("M")
    factor_by_day = pd.Series(
        [adj_map.get(k, 1.0) for k in month_key], index=idx)
    out["主CTA"] = out["主CTA"] * factor_by_day
    return out


def build_versions(rets: pd.DataFrame, au_ret: pd.Series) -> dict[str, pd.Series]:
    """按 §27 组装五版本净收益序列 (rets 已含 AU_layered 列).

    与 §27 一致: 各版本先切 >= 2019-01-01 (剔除 2018 warmup 年) 再统计.
    """
    versions = {}
    for lab, w in VERSIONS.items():
        if lab == "主CTA本体(golden)":
            r_v = rets["主CTA"]
        else:
            w = {k: v * 0.9 for k, v in w.items()} if lab == "E2-R+AU10%" else w
            r_v = run_portfolio(rets, w)
        versions[lab] = r_v[r_v.index >= "2019-01-01"]
    return versions


def main() -> int:
    # ── baseline rets ──
    rets_base = load_rets()
    au = pd.read_parquet(f"{REPORT_DIR}/gold_layered_layered.parquet")["ret"]
    rets_base = rets_base.copy()
    rets_base["AU_layered"] = au.reindex(rets_base.index).fillna(0)
    print(f"baseline rets: {rets_base.shape}, {rets_base.index[0]} ~ {rets_base.index[-1]}")

    # ── ZK 缩放版 rets ──
    adj = build_zk_adjust_factor()
    rets_zk = scale_main_cta(rets_base, adj)
    n_changed = (rets_zk["主CTA"] != rets_base["主CTA"]).sum()
    print(f"ZK 调节覆盖: {n_changed} 天收益被缩放 "
          f"(因子范围 {adj.min():.3f}~{adj.max():.3f})")

    # ── 两版 × 五版本 ──
    rows = []
    for tag, versions in [("baseline", build_versions(rets_base, au)),
                          ("zk_vol_adj", build_versions(rets_zk, au))]:
        for lab, r_v in versions.items():
            for wlab, lo, hi in WINDOWS:
                x = r_v.dropna()
                if lo:
                    x = x[x.index >= lo]
                if hi:
                    x = x[x.index < hi]
                st = stats_of(x)
                rows.append({"tag": tag, "version": lab, "window": wlab, **st})
    df = pd.DataFrame(rows)

    # ── 输出: 对比总表 (baseline vs zk 并排) ──
    pivot = df.pivot_table(index=["version", "window"], columns="tag",
                           values=["AR", "Vol", "Sharpe", "MaxDD", "Calmar"])
    pivot.columns = [f"{m}_{t}" for m, t in pivot.columns]
    pivot["dSharpe"] = pivot["Sharpe_zk_vol_adj"] - pivot["Sharpe_baseline"]
    pivot["dCalmar"] = pivot["Calmar_zk_vol_adj"] - pivot["Calmar_baseline"]
    pivot = pivot.round(4).reset_index()

    out_full = f"{ZK_DIR}/e1_production_full.csv"
    out_pivot = f"{ZK_DIR}/e1_production_evaluation.csv"
    df.to_csv(out_full, index=False)
    pivot.to_csv(out_pivot, index=False)
    print(f"输出: {out_full} ({len(df)}行) + {out_pivot} ({len(pivot)}行)")
    print()
    show = ["version", "window", "Sharpe_baseline", "Sharpe_zk_vol_adj",
            "dSharpe", "Calmar_baseline", "Calmar_zk_vol_adj", "dCalmar",
            "MaxDD_baseline", "MaxDD_zk_vol_adj"]
    print(pivot[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
