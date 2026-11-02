# coding=utf-8
"""完整业绩对比报告: v1 (修改前) vs v8 (当前默认), 2020-01-01 ~ 2026-09-04.

输出:
    - 逐品种: 统计区间 / 年化收益 / 波动 / Sharpe / MaxDD / Calmar / 交易数
    - 逐年收益: 2020 ~ 2025 全年 + 2026 YTD, 及 2026 YTD MaxDD
    - 等权组合: 同套指标
    - 落盘: reports/cta_ts_mom/performance_full_report.csv + PNG

用法:
    .venv-mig/bin/python scripts/perf_report_v1_vs_v8.py
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.cta.data_loader import CtaDataLoader  # noqa: E402
from QuantNodes.strategy.cta.data_loader_ext import (  # noqa: E402
    load_continuous_main_ext,
)
from QuantNodes.strategy.cta.strategies.huatai_timing_cta import (  # noqa: E402
    HUATAI_PRODUCTS,
    HuataiTimingConfig,
    HuataiTimingCTAStrategy,
)

OUT_DIR = "/home/ll/Public/QuantNodes/reports/cta_ts_mom"
START, END = "2020-01-01", "2026-09-04"

# v1 复现配置 (修改前): 无RSI回落 / 无TR截断 / 无冷却 / 熔断禁用
V1_CONFIG = HuataiTimingConfig(
    tr_clip_enabled=False,
    cooldown_days=0,
    range_rsi_exit_enabled=False,
    drawdown_threshold=-1.0,
)


class _ExtLoader:
    """CtaDataLoader 适配器: 主力连续经 export 2026 库扩展至 2026-09-04."""

    def __init__(self, inner: CtaDataLoader):
        self._inner = inner

    def load_continuous_main(self, product, start, end):
        return load_continuous_main_ext(product, start, end, loader=self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _metrics_from_nav(nav: pd.Series, trades: int = 0) -> dict:
    """从 NAV 计算全套指标."""
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if nav.iloc[0] > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "start": nav.index[0], "end": nav.index[-1],
        "AnnRet": cagr, "Vol": ann_vol, "Sharpe": sharpe,
        "MaxDD": max_dd, "Calmar": calmar, "Trades": trades,
    }


def _yearly(nav: pd.Series) -> dict[str, float]:
    """逐年收益: 2020..2025 全年 + 2026 YTD."""
    nav = nav.dropna()
    out = {}
    for y in range(2020, 2027):
        sub = nav[nav.index.year == y]
        if len(sub) < 2:
            continue
        prev = nav[nav.index.year < y]
        base = prev.iloc[-1] if len(prev) else sub.iloc[0]
        out[str(y)] = sub.iloc[-1] / base - 1
    # 2026 YTD MaxDD
    sub26 = nav[nav.index.year == 2026]
    if len(sub26) >= 2:
        dd26 = (sub26 / sub26.cummax() - 1).min()
        out["MaxDD26YTD"] = dd26
    return out


def run_portfolio(nav_df: pd.DataFrame) -> pd.Series:
    """等权组合: 各品种日收益横截面均值 (skipna), 再累乘."""
    ret = nav_df.sort_index().pct_change()
    port_ret = ret.mean(axis=1, skipna=True)
    return (1 + port_ret.fillna(0)).cumprod()


def main() -> None:
    dl = CtaDataLoader()
    ext = _ExtLoader(dl)

    configs = {
        "v1": V1_CONFIG,
        "v8": HuataiTimingConfig(),  # 默认: ρ可选+μ过滤+观察期熔断+RSI回落+TR截断+冷却
    }
    navs_all: dict[str, pd.DataFrame] = {}
    trades_all: dict[str, dict[str, int]] = {}

    t0 = time.time()
    for cfg_name, cfg in configs.items():
        navs: dict[str, pd.Series] = {}
        trades: dict[str, int] = {}
        for p in HUATAI_PRODUCTS:
            try:
                strat = HuataiTimingCTAStrategy(
                    products=[p], config=cfg, data_loader=ext,
                    start=START, end=END,
                )
                result = strat.run()
                navs[p] = result.nav
                trades[p] = len(result.trades)
            except Exception as e:  # noqa: BLE001
                print(f"[{cfg_name} {p}] ERROR: {e}")
        navs_all[cfg_name] = pd.DataFrame(navs)
        trades_all[cfg_name] = trades
        print(f"[{cfg_name}] done ({time.time() - t0:.0f}s)")

    # ── 逐品种统计 ──
    rows = []
    for cfg_name, nav_df in navs_all.items():
        for p in nav_df.columns:
            nav = nav_df[p].dropna()
            m = _metrics_from_nav(nav, trades_all[cfg_name].get(p, 0))
            y = _yearly(nav)
            rows.append({"config": cfg_name, "product": p, **m, **y})
    per_product = pd.DataFrame(rows)
    per_product.to_csv(f"{OUT_DIR}/performance_full_report.csv", index=False)

    # ── 等权组合 ──
    port_rows = []
    port_navs = {}
    for cfg_name, nav_df in navs_all.items():
        port = run_portfolio(nav_df)
        port_navs[cfg_name] = port
        m = _metrics_from_nav(port)
        y = _yearly(port)
        port_rows.append({"config": cfg_name, "product": "PORTFOLIO_EW", **m, **y})
    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(f"{OUT_DIR}/performance_portfolio_ew.csv", index=False)

    combined = pd.concat([port_df, per_product], ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/performance_full_report.csv", index=False)

    # ── 打印 ──
    show_cols = ["AnnRet", "Vol", "Sharpe", "MaxDD", "Calmar", "Trades"]
    yr_cols = ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "MaxDD26YTD"]

    for cfg_name in configs:
        sub = combined[combined["config"] == cfg_name].set_index("product")
        print(f"\n{'=' * 118}")
        print(f"  {cfg_name} 全期指标 (统计区间见 CSV)  年化收益 / 波动 / Sharpe / MaxDD / Calmar")
        print(f"{'=' * 118}")
        print(f"{'品种':6s} " + " ".join(f"{c:>9s}" for c in show_cols)
              + "  " + " ".join(f"{c:>8s}" for c in yr_cols[:-1]) + f" {'26YTD_MDD':>9s}")
        print("-" * 118)
        order = ["PORTFOLIO_EW"] + HUATAI_PRODUCTS
        for p in order:
            if p not in sub.index:
                continue
            r = sub.loc[p]
            interval = f"{r['start']:%Y.%m.%d}~{r['end']:%Y.%m.%d}"
            cells = " ".join(f"{r[c]:+9.3f}" for c in show_cols[:5]) + f" {int(r['Trades']):9d}"
            yrs = " ".join(
                f"{r[c] * 100:+7.1f}%" if pd.notna(r.get(c)) else "     - "
                for c in yr_cols[:-1]
            )
            mdd26 = r.get("MaxDD26YTD", np.nan)
            mdd26_s = f"{mdd26 * 100:+8.1f}%" if pd.notna(mdd26) else "     -  "
            print(f"{p:6s} {cells}  {yrs} {mdd26_s}   [{interval}]")

    # ── 对比汇总 ──
    print(f"\n{'=' * 118}")
    print("  v1 vs v8 同品种对比 (全期 Calmar / 2026 YTD)")
    print(f"{'=' * 118}")
    v1 = combined[combined["config"] == "v1"].set_index("product")
    v8 = combined[combined["config"] == "v8"].set_index("product")
    for p in ["PORTFOLIO_EW"] + HUATAI_PRODUCTS:
        if p not in v1.index or p not in v8.index:
            continue
        c1, c8 = v1.loc[p, "Calmar"], v8.loc[p, "Calmar"]
        y1 = v1.loc[p].get("2026", np.nan)
        y8 = v8.loc[p].get("2026", np.nan)
        print(f"{p:14s}: Calmar {c1:+.3f} → {c8:+.3f} ({c8 - c1:+.3f}) | "
              f"26YTD {y1 * 100:+6.1f}% → {y8 * 100:+6.1f}%")

    # ── 图: 组合 NAV 对比 + 逐年柱状 ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [3, 1, 1.2]})
    ax1 = axes[0]
    for name, color in [("v1", "steelblue"), ("v8", "green")]:
        nv = port_navs[name]
        ax1.plot(nv.index, nv, label=f"{name} portfolio", lw=2, color=color)
    ax1.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax1.set_title("Equal-Weight Portfolio NAV: v1 vs v8 (2020-01 ~ 2026-09)",
                  fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2 = axes[1]
    for name, color in [("v1", "steelblue"), ("v8", "green")]:
        dd = port_navs[name] / port_navs[name].cummax() - 1
        ax2.fill_between(dd.index, dd, 0, alpha=0.35, color=color, label=name)
    ax2.set_title("Portfolio Drawdown")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax3 = axes[2]
    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    x = np.arange(len(years))
    w = 0.38
    for k, (name, color) in enumerate([("v1", "steelblue"), ("v8", "green")]):
        sub = combined[combined["config"] == name].set_index("product").loc["PORTFOLIO_EW"]
        vals = [sub.get(y, np.nan) * 100 if pd.notna(sub.get(y, np.nan)) else np.nan
                for y in years]
        ax3.bar(x + (k - 0.5) * w, vals, w, label=name, color=color, alpha=0.85)
    ax3.set_xticks(x, years)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Portfolio Yearly Returns (%)")
    ax3.legend()
    ax3.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    png = f"{OUT_DIR}/performance_v1_vs_v8_full.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图已保存: {png}")
    print(f"表已保存: {OUT_DIR}/performance_full_report.csv")
    print(f"         {OUT_DIR}/performance_portfolio_ew.csv")


if __name__ == "__main__":
    main()
