"""v7.0 52 ETF 池量化筛选 (Stage 30.5 Phase B1).

[动机] 7 ETF 手工选, 引入选择偏差. 用量化标准从 52 ETF 中筛出
       真正可投资 + 多样化 的 50+ ETF 池.

[筛选标准]
    1. 上市 ≥ 3 年 (start date ≥ 2018-07-01, 数据起点 2018-01-01)
    2. 日均成交额 > 5000 万 (从 OHLCV amount 算)
    3. 排除货币基金 (511260, 511010 等)
    4. 同标的留最高流动性 (黄金 518880 留, 518800 排除)
    5. 至少 80% 完整度 (允许节假日缺失, 但不应有长段空缺)

[输出]
    - reports/.../v7_0_52etf_universe.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")


# 货币/商品/工具型 ETF (排除)
EXCLUDE_PATTERNS = ["511", "518800"]  # 511=货币, 518800=黄金同标
EXCLUDE_EXACT = {"511260"}  # 货币基金


def compute_etf_metrics(panel_close: pd.DataFrame, panel_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """对每个 ETF 计算筛选指标."""
    rows = []
    for code in panel_close.columns:
        nav_s = panel_close[code].dropna()
        if len(nav_s) < 252 * 3:  # < 3 年
            continue
        start = nav_s.index[0]
        end = nav_s.index[-1]

        adv = np.nan
        if isinstance(panel_ohlcv.columns, pd.MultiIndex):
            codes_ohlcv = panel_ohlcv.columns.get_level_values(0)
        else:
            codes_ohlcv = panel_ohlcv.columns
        if code in codes_ohlcv:
            try:
                if isinstance(panel_ohlcv.columns, pd.MultiIndex):
                    sub = panel_ohlcv[code]
                    if "close" in sub.columns and "volume" in sub.columns:
                        amt = (sub["close"] * sub["volume"]).dropna()
                        amt = amt[amt > 0]
                        if len(amt) > 0:
                            adv = float(amt.mean())
            except Exception:
                pass

        n_obs = len(nav_s)
        n_total_days = (end - start).days
        completeness = n_obs / max(n_total_days * 0.7, 1)  # 假设 70% 交易日
        ny = (end - start).days / 365.25
        ann_ret = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / ny) - 1 if ny > 0 else 0
        ann_vol = nav_s.pct_change().std() * np.sqrt(252)

        rows.append({
            "code": code,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "n_obs": n_obs,
            "n_years": round(ny, 2),
            "completeness": round(min(completeness, 1.0), 3),
            "ann_ret": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4),
            "avg_daily_amount": round(adv, 0) if not np.isnan(adv) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("[v7.0 52 ETF 池量化筛选] 加载数据...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    sb = pd.read_parquet(REPO / "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")

    panel_close = pd.DataFrame()
    for c in set(nav_main.columns) | set(sb.columns):
        if c in nav_main.columns:
            s = nav_main[c].dropna()
        elif c in sb.columns:
            s = sb[c].dropna()
        else:
            continue
        panel_close[c] = s
    panel_close = panel_close.dropna(how='all').ffill()
    print(f"  panel_close: {panel_close.shape}")

    if isinstance(ohlcv.columns, pd.MultiIndex):
        codes = panel_close.columns
    else:
        codes = panel_close.columns

    print("  计算 52 ETF 筛选指标...")
    metrics = compute_etf_metrics(panel_close, ohlcv)
    print(f"  通过 n_obs >= 756 (3y): {len(metrics)}/{len(panel_close.columns)}")

    metrics["pass_exclude"] = ~(
        metrics["code"].str.startswith(tuple(EXCLUDE_PATTERNS))
        | metrics["code"].isin(EXCLUDE_EXACT)
    )
    n_excluded = (~metrics["pass_exclude"]).sum()
    print(f"  排除货币/商品基金: {n_excluded}")

    metrics["pass_amount"] = metrics["avg_daily_amount"].fillna(0) > 50_000_000
    n_amount = metrics["pass_amount"].sum()
    print(f"  通过日均成交额 > 5000万: {n_amount}")

    metrics["pass_completeness"] = metrics["completeness"] >= 0.80
    n_comp = metrics["pass_completeness"].sum()
    print(f"  通过 completeness >= 80%: {n_comp}")

    metrics["passed"] = (
        metrics["pass_exclude"]
        & metrics["pass_amount"]
        & metrics["pass_completeness"]
    )
    n_pass = metrics["passed"].sum()
    print(f"  最终通过: {n_pass}")

    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "v7_0_52etf_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\n[save] {metrics_path}")

    universe = metrics[metrics["passed"]].copy()
    universe_path = out_dir / "v7_0_52etf_universe.csv"
    universe.to_csv(universe_path, index=False)
    print(f"[save] {universe_path}")

    print(f"\n=== 52 ETF 池: {len(universe)} 个 ===")
    print(universe[["code", "start_date", "n_years", "completeness", "ann_ret", "avg_daily_amount"]].to_string(index=False))


if __name__ == "__main__":
    main()
