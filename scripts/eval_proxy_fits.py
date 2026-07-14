# coding: utf-8
"""评估 17 只 late-listing ETF 与其 iFinD proxy 的同步涨跌 (R² + corr).

输入:
  data/high_freq_macro/v56_expanded_daily.parquet
  data/high_freq_macro/v56_proxy_indices_daily.parquet
  data/high_freq_macro/_proxy_nan_table.csv

输出:
  data/high_freq_macro/_proxy_r2_table.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

V56_PATH  = PROJECT_ROOT / "data/high_freq_macro/v56_expanded_daily.parquet"
PROXY_PANEL = PROJECT_ROOT / "data/high_freq_macro/v56_proxy_indices_daily.parquet"
NAN_TABLE  = PROJECT_ROOT / "data/high_freq_macro/_proxy_nan_table.csv"
R2_TABLE   = PROJECT_ROOT / "data/high_freq_macro/_proxy_r2_table.csv"

# ETF → 选定的 proxy code (从 ETF_PROXY_MAP 同步硬拷贝, 避免再次引入)
ETF_PROXY = {
    "159786": "399324.SZ_深证红利",
    "159740": "HSTECH.HK_恒生科技",
    "515790": "931151.CSI_光伏产业",
    "588000": "000688.SH_科创50",
    "513300": "NDX.GI_纳斯达克100",
    "515100": "930850.CSI_智能制造",
    "515220": "399998.SZ_中证煤炭",
    "515030": "399976.SZ_CS新能车",
    "159996": "931450.CSI_新能车主题",
    "515080": "931152.CSI_CS创新药",
    "515900": "000906.SH_中证800",
    "512170": "399989.SZ_中证医疗",
    "512760": "980017.SZ_国证芯片",
    "512480": "980017.SZ_国证芯片",
    "512690": "399987.SZ_中证酒",
    "512890": "H30269.CSI_红利低波",
    "512260": "000922.CSI_中证红利",
}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def monthly_returns(daily: pd.Series) -> pd.Series:
    """月收益 = 月末值 / 月初值 - 1 (level-based compound)."""
    s = daily.dropna()
    if s.empty:
        return s
    monthly = (1 + s.pct_change().fillna(0.0)).resample("ME").prod() - 1.0
    return monthly


def returns(s: pd.Series) -> pd.Series:
    """日度收益."""
    return s.pct_change()


def fit_r2(y: pd.Series, x: pd.Series) -> tuple[float, float, int]:
    """OLS y ~ a + b*x, 返回 (R², corr, N). 输入应为 returns, 不是 level."""
    valid = y.notna() & x.notna()
    yv = y[valid].values
    xv = x[valid].values
    if len(yv) < 12:
        return np.nan, np.nan, len(yv)
    # demean
    yv = yv - yv.mean()
    xv = xv - xv.mean()
    denom = float((yv**2).sum())
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan, len(yv)
    b = float((yv * xv).sum() / (xv**2).sum()) if (xv**2).sum() > 0 else 0.0
    if not np.isfinite(b):
        return np.nan, np.nan, len(yv)
    intercept = yv.mean() - b * xv.mean()
    resid = yv - (b * xv + intercept)
    ss_res = float((resid**2).sum())
    ss_tot = denom
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if (xv**2).sum() == 0 or (yv**2).sum() == 0:
        corr = np.nan
    else:
        corr = float((yv * xv).sum() / np.sqrt((xv**2).sum() * (yv**2).sum()))
    return float(r2), float(corr), int(len(yv))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    v56 = pd.read_parquet(V56_PATH)
    v56.index = pd.DatetimeIndex(v56.index)
    proxy = pd.read_parquet(PROXY_PANEL)
    proxy.index = pd.DatetimeIndex(proxy.index)
    nan_df = pd.read_csv(NAN_TABLE)

    rows = []
    for etf, proxy_col in ETF_PROXY.items():
        if etf not in v56.columns:
            continue
        if proxy_col not in proxy.columns:
            continue
        y_d = v56[etf]
        x_d = proxy[proxy_col]
        # 共有日 (同期)
        y_m = monthly_returns(y_d)
        x_m = monthly_returns(x_d)

        r2_d, corr_d, n_d = fit_r2(y_d, x_d)
        r2_m, corr_m, n_m = fit_r2(y_m, x_m)
        # 上市后段 (ETF 有数据段)
        etf_row = nan_df[nan_df["etf"].astype(str) == etf]
        if etf_row.empty:
            continue
        etf_row = etf_row.iloc[0]
        ipo = pd.Timestamp(etf_row["ipo"])
        y_d_post = y_d.loc[y_d.index >= ipo]
        r2_d_post, corr_d_post, n_d_post = fit_r2(y_d_post, x_d)

        rows.append({
            "etf": etf,
            "name": etf_row["name"],
            "ipo": ipo.strftime("%Y-%m-%d"),
            "proxy": proxy_col,
            "R2_daily_full":      round(r2_d, 3) if not np.isnan(r2_d) else None,
            "R2_daily_postIPO":   round(r2_d_post, 3) if not np.isnan(r2_d_post) else None,
            "R2_monthly":         round(r2_m, 3) if not np.isnan(r2_m) else None,
            "corr_daily":         round(corr_d, 3) if not np.isnan(corr_d) else None,
            "corr_monthly":       round(corr_m, 3) if not np.isnan(corr_m) else None,
            "n_days_full":        n_d,
            "n_days_postIPO":     n_d_post,
            "n_months":           n_m,
        })

    df = pd.DataFrame(rows)
    df.to_csv(R2_TABLE, index=False)

    # 输出精炼表
    print("\n=== ETF ↔ iFinD Proxy 同期 R² 表 (月频) ===")
    print(df[["etf","name","proxy","R2_daily_full","R2_daily_postIPO",
              "R2_monthly","corr_monthly","n_months"]].to_string(index=False))

    print("\n=== NaN 现状总结 (v56 + v56_proxy 合并后) ===")
    full_nan = v56.isna().mean().mean() * 100
    proxy_nan = proxy.isna().mean().mean() * 100
    print(f"  v56 daily NaN:        {full_nan:.2f}%")
    print(f"  v56_proxy daily NaN:  {proxy_nan:.2f}% (前 3.5 年 NaN)")
    print(f"  proxy 共同段(2021-07-14~) daily NaN: "
          f"{proxy.loc['2021-07-14':].isna().mean().mean() * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
