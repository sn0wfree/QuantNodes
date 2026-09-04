# coding=utf-8
"""cta_full_assessment_export.py — §27 主CTA完整评估导出.

口径修正: 2018为warmup占位 (242日全零), 真实窗口 2019-2025.
pandas注意: 聚合用 .prod() 方法, 禁用 apply(lambda (1+s).prod()-1)
            (本环境多层分组下该写法产出荒谬值, 已验证 get_group正常).

产出:
    reports/cta/main_cta_full_assessment.csv  指标总表 (版本×窗口)
    reports/cta/main_cta_curve.csv            日频NAV+回撤+月度矩阵
    reports/cta/main_cta_curve.html/.png      plotly三联图
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_t10y_removal import load_rets, run_portfolio  # noqa: E402

REPORT_DIR = "reports/cta"


def stats_of(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    downside = r[r < 0].std() * np.sqrt(252)
    return {"AR": round(ann, 4), "Vol": round(vol, 4),
            "Sharpe": round(ann / vol, 3) if vol > 0 else np.nan,
            "MaxDD": round(dd, 4), "Calmar": round(ann / abs(dd), 2),
            "Sortino": round(ann / downside, 2) if downside > 0 else np.nan,
            "NAV_end": round(nav.iloc[-1], 3)}


def main() -> None:
    g = pd.read_parquet(f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet")["nav"]
    r_full = g.pct_change(fill_method=None).fillna(0)
    r = r_full[r_full.index >= "2019-01-01"]

    rets = load_rets()
    au = pd.read_parquet(f"{REPORT_DIR}/gold_layered_layered.parquet")["ret"]
    rets2 = rets.copy()
    rets2["AU_layered"] = au.reindex(rets2.index).fillna(0)
    versions = {
        "主CTA本体(golden)": r,
        "R2-R(55/45)": pd.read_parquet(f"{REPORT_DIR}/t10y_free_R2-R放大.parquet")["net_ret"],
        "E2-R(45/55)": pd.read_parquet(f"{REPORT_DIR}/t10y_free_E2-R放大.parquet")["net_ret"],
        "E2-R+AU10%": run_portfolio(rets2, {k: v * 0.9 for k, v in
            {"主CTA": 0.45, "CU溢价TC": 39.31, "jjm_coke": 15.69}.items()}
            | {"AU_layered": 10}),
        "B2满配(子≤10%)": run_portfolio(rets2, {"主CTA": 0.60, "CU溢价TC": 10,
            "jjm_coke": 10, "AU_layered": 10, "ZN库存": 10}),
    }
    for k in versions:
        versions[k] = versions[k][versions[k].index >= "2019-01-01"]

    # ── 1. 指标总表 ──
    rows = []
    windows = [("全样本19-25", None, None), ("IS 19-21", None, "2022-01-01"),
               ("OOS 22-25", "2022-01-01", None)]
    for lab, r_v in versions.items():
        for wlab, lo, hi in windows:
            x = r_v.dropna()
            if lo:
                x = x[x.index >= lo]
            if hi:
                x = x[x.index < hi]
            st = stats_of(x)
            rows.append({"version": lab, "window": wlab, **st})
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(f"{REPORT_DIR}/main_cta_full_assessment.csv", index=False)
    print(df_stats.to_string(index=False))

    # ── 2. 曲线数据 ──
    main_nav = (1 + versions["主CTA本体(golden)"]).cumprod()
    main_dd = main_nav / main_nav.cummax() - 1
    curve = main_nav.to_frame("nav_main")
    for lab, r_v in versions.items():
        curve[f"nav_{lab}"] = (1 + r_v).cumprod()
    curve["dd_main"] = main_dd
    r_m = r.groupby([r.index.year, r.index.month]).prod() - 1
    mt = r_m.rename("ret").reset_index()
    mt.columns = ["y", "m", "ret"]
    curve.to_csv(f"{REPORT_DIR}/main_cta_curve.csv")
    mt.to_csv(f"{REPORT_DIR}/main_cta_curve_monthly.csv", index=False)

    # ── 3. plotly 三联图 ──
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.05,
        subplot_titles=("NAV (2019-2025, 剔除2018 warmup)",
                        "主CTA回撤", "分年收益"))
    colors = {"主CTA本体(golden)": "#d62728", "R2-R(55/45)": "#7f7f7f",
              "E2-R(45/55)": "#1f77b4", "E2-R+AU10%": "#2ca02c",
              "B2满配(子≤10%)": "#9467bd"}
    for lab, r_v in versions.items():
        fig.add_trace(go.Scatter(x=(1 + r_v).cumprod().index,
                                 y=(1 + r_v).cumprod().values,
                                 name=lab, line=dict(width=2.2 if "E2-R+AU" in lab else 1.4,
                                                     color=colors[lab])),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=main_dd.index, y=main_dd.values, name="DD",
                             fill="tozeroy", line=dict(color="#d62728", width=1)),
                  row=2, col=1)
    yr = (1 + r).groupby(r.index.year).prod() - 1
    fig.add_trace(go.Bar(x=yr.index.astype(str), y=yr.values, name="分年收益",
                         marker_color=["#2ca02c" if v > 0 else "#d62728" for v in yr],
                         text=[f"{v:+.1%}" for v in yr], textposition="outside"),
                  row=3, col=1)
    fig.update_layout(title="主CTA 完整评估曲线 (真实窗口 2019-2025)",
                      height=900, legend=dict(orientation="h", y=1.06))
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1, tickformat=".0%")
    fig.update_yaxes(title_text="年收益", row=3, col=1, tickformat=".0%")
    fig.write_html(f"{REPORT_DIR}/main_cta_curve.html")
    try:
        fig.write_image(f"{REPORT_DIR}/main_cta_curve.png", width=1400, height=900)
        print("PNG 导出成功")
    except Exception as e:
        print(f"PNG 跳过 (kaleido): {e}")
    print("导出完成: main_cta_full_assessment.csv / main_cta_curve.csv / "
          "main_cta_curve_monthly.csv / main_cta_curve.html(.png)")


if __name__ == "__main__":
    main()
