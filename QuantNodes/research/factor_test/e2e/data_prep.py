# coding: utf-8
"""E2E 数据准备 — 生成 H5 格式合成数据, 模拟 iFinD 真实数据结构。

输出目录结构 (符合 LoadDataNode 期望):
    {output_dir}/
    ├── stk_daily.h5     # 股票日数据
    │   ├── cp           # 收盘价
    │   ├── st           # ST 标记
    │   ├── suspend      # 停牌
    │   ├── ud_limit     # 涨跌停
    │   ├── ipo_days     # 上市天数
    │   ├── id_citic1    # 行业
    │   └── mv_float     # 自由流通市值
    ├── index_daily.h5   # 指数日数据
    │   ├── index_cp     # 指数收盘价
    │   └── ...
    ├── stklist.h5
    ├── trade_dt.h5
    └── {factor_name}.h5  # 因子数据 (如 momentum_20d)

用法:
    python -m QuantNodes.research.factor_test.e2e.data_prep \\
           --output-dir /tmp/e2e_data/ \\
           --n-days 120 --n-stocks 30 \\
           --factors momentum_20d,reversal_5d,volatility_60d
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _gen_factor_data(rng: np.random.RandomState, n_days: int, n_stocks: int,
                     name: str) -> pd.DataFrame:
    """生成特定形态的因子值 (与名称挂钩, 让不同因子有不同 IC)。"""
    dates = _gen_dates(n_days)
    stocks = _gen_stocks(n_stocks)
    if "momentum" in name.lower():
        # 因子值与未来收益正相关 (正 IC)
        trend = np.linspace(0, 0.5, n_days).reshape(-1, 1)
        base = rng.randn(n_days, n_stocks) + trend
    elif "reversal" in name.lower():
        # 因子值与未来收益负相关 (负 IC)
        trend = -np.linspace(0, 0.5, n_days).reshape(-1, 1)
        base = rng.randn(n_days, n_stocks) + trend
    elif "volatility" in name.lower():
        # 因子值与波动率相关 (弱 IC)
        base = rng.randn(n_days, n_stocks) * 0.5
    else:
        # 默认: 随机因子 (无 IC)
        base = rng.randn(n_days, n_stocks)
    return pd.DataFrame(base, index=dates, columns=stocks)


def _gen_dates(n_days: int) -> list[int]:
    # H11: 不再硬编码 '2026-01-04', 默认从 1 年前开始 (滚动)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    return [int(d.strftime('%Y%m%d'))
            for d in pd.bdate_range(start_date, periods=n_days)]


def _gen_stocks(n_stocks: int) -> list[int]:
    return list(range(100001, 100001 + n_stocks))


def _gen_index_cp(rng: np.random.RandomState, n_days: int) -> pd.DataFrame:
    """沪深 300 + 中证 500 指数收盘价。"""
    dates = _gen_dates(n_days)
    return pd.DataFrame({
        '000300.SH': 3500 + np.cumsum(rng.randn(n_days) * 10),
        '000905.SH': 6000 + np.cumsum(rng.randn(n_days) * 15),
    }, index=dates)


def _gen_stk_daily(rng: np.random.RandomState, n_days: int, n_stocks: int) -> dict:
    """生成 stk_daily.h5 的所有 key。"""
    dates = _gen_dates(n_days)
    stocks = _gen_stocks(n_stocks)
    # 收盘价: 几何布朗运动
    price = 100 * np.exp(np.cumsum(rng.randn(n_days, n_stocks) * 0.02, axis=0))
    # 行业 (申万一级 1-30)
    industry = rng.randint(1, 31, (n_days, n_stocks))
    # 自由流通市值
    mv = rng.lognormal(10, 1, (n_days, n_stocks))
    # ST / 停牌 / 涨跌停 — 稀疏
    st = np.zeros((n_days, n_stocks), dtype=int)
    st[:, :min(2, n_stocks)] = 1
    suspend = np.zeros((n_days, n_stocks), dtype=int)
    if n_stocks > 3:
        suspend[5:8, 3] = 1
    ud_limit = np.zeros((n_days, n_stocks), dtype=int)
    if n_stocks > 5:
        ud_limit[10:12, 5] = 1
    # 上市天数 (>360 不剔除)
    ipo_days = np.ones((n_days, n_stocks), dtype=int) * 500
    ipo_days[0, 0] = 100  # 第一只新股, 会被 IPO < 360 剔除
    return {
        "cp": pd.DataFrame(price, index=dates, columns=stocks),
        "st": pd.DataFrame(st, index=dates, columns=stocks),
        "suspend": pd.DataFrame(suspend, index=dates, columns=stocks),
        "ud_limit": pd.DataFrame(ud_limit, index=dates, columns=stocks),
        "ipo_days": pd.DataFrame(ipo_days, index=dates, columns=stocks),
        "id_citic1": pd.DataFrame(industry, index=dates, columns=stocks),
        "mv_float": pd.DataFrame(mv, index=dates, columns=stocks),
    }


def main():
    parser = argparse.ArgumentParser(description="E2E 数据准备")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--n-days", type=int, default=120, help="天数 (默认 120)")
    parser.add_argument("--n-stocks", type=int, default=30, help="股票数 (默认 30)")
    parser.add_argument("--factors", default="momentum_20d,reversal_5d,volatility_60d",
                        help="逗号分隔的因子列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    n_days, n_stocks = args.n_days, args.n_stocks
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]

    print(f"生成数据: {n_days} 天 × {n_stocks} 股票, 因子={factors}")
    print(f"输出目录: {out}")

    # 1. stklist / trade_dt
    dates = _gen_dates(n_days)
    stocks = _gen_stocks(n_stocks)
    pd.DataFrame(stocks, columns=[0]).to_hdf(out / "stklist.h5", key="data", mode="w")
    pd.DataFrame(dates, columns=[0]).to_hdf(out / "trade_dt.h5", key="data", mode="w")
    print(f"  ✓ stklist.h5, trade_dt.h5 ({len(stocks)} stocks × {len(dates)} dates)")

    # 2. stk_daily.h5
    stk_daily = _gen_stk_daily(rng, n_days, n_stocks)
    with pd.HDFStore(out / "stk_daily.h5", mode="w") as store:
        for key, df in stk_daily.items():
            store.put(key, df, format="table")
    print(f"  ✓ stk_daily.h5 ({len(stk_daily)} keys)")

    # 3. index_daily.h5
    index_cp = _gen_index_cp(rng, n_days)
    with pd.HDFStore(out / "index_daily.h5", mode="w") as store:
        store.put("index_cp", index_cp, format="table")
    print(f"  ✓ index_daily.h5 (index_cp: {index_cp.shape})")

    # 4. 因子 H5 (LoadDataNode 用 factor_dir 加载)
    for fname in factors:
        factor = _gen_factor_data(rng, n_days, n_stocks, fname)
        factor.to_hdf(out / f"{fname}.h5", key="data", mode="w")
        print(f"  ✓ {fname}.h5 (shape: {factor.shape})")

    print("\n✓ 数据准备完成")
    print(f"  data_path: {out}")


if __name__ == "__main__":
    main()
