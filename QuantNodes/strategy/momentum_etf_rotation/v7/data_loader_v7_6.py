# coding=utf-8
"""v7.6 数据加载: 9 macro + 11 量价, 月频.

v7.6 = v7.3 (9 macro) + v5 (11 量价), 月频对齐.

数据流:
  1. 9 macro factors: 周频 → 月频 (resample('M').last())
  2. 11 量价 factors: 日频 → 月频 (resample('M').last())
  3. Y (asset returns): 日频 → 月频 (resample('M').last())

输入:
  - ~/Public/高频宏观因子/高频宏观因子跟踪_output_2026-06-01.xlsx (9 macro)
  - data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet (ETF OHLCV)
  - data/real/etf_nav_2018-01-01_2026-06-30.parquet (ETF NAV)

输出:
  - data/high_freq_macro/v7_6_X_macro_monthly.parquet (T_monthly, 9)
  - data/high_freq_macro/v7_6_X_pv_monthly.parquet (T_monthly, 56×11)
  - data/high_freq_macro/v7_6_Y_monthly.parquet (T_monthly, 56)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"
HF_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR = REPO / "data" / "real"

# 复用 v7.3 的数据加载函数
from .data_loader import (
    FACTOR_COLS,
    EXPANDED_COLS,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
    EXPANDED_BOND_INDICES,
    load_macro_factors,
)

# 复用 v5 的因子计算
from ..v5.industry_factors import (
    FactorEngineConfig,
    compute_all_factors_panel,
)


# ============================================================
# 1. 9 macro factors: 周频 → 月频
# ============================================================
def load_monthly_macro_factors() -> pd.DataFrame:
    """加载 9 宏观因子, 周频 → 月频.

    Returns:
        DataFrame (T_monthly, 9) 月频宏观因子.
    """
    cache = HF_DIR / "v7_6_X_macro_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 加载周频
    weekly = load_macro_factors()

    # 周频 → 月频 (月末)
    monthly = weekly.resample("M").last()

    # 保存缓存
    monthly.to_parquet(cache)
    return monthly


# ============================================================
# 2. 11 量价 factors: 日频 → 月频
# ============================================================
def load_monthly_pv_factors(
    factor_cfg: FactorEngineConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """计算 11 量价因子, 日频 → 月频.

    Args:
        factor_cfg: 因子引擎配置 (None = 默认)

    Returns:
        dict, code → DataFrame (T_monthly, 11) 月频量价因子.
    """
    cache_dir = HF_DIR / "v7_6_pv_monthly"
    cache_dir.mkdir(parents=True, exist_ok=True)

    factor_cfg = factor_cfg or FactorEngineConfig()
    codes = sorted(set(EXPANDED_COLS) - set(EXPANDED_BOND_INDICES))  # 51 ETFs

    result = {}
    for code in codes:
        cache_path = cache_dir / f"{code.replace('/', '_')}.parquet"
        if cache_path.exists():
            result[code] = pd.read_parquet(cache_path)
            continue

        # 加载 ETF OHLCV
        ohlcv_path = REAL_DIR / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
        if not ohlcv_path.exists():
            continue

        ohlcv_panel = pd.read_parquet(ohlcv_path)
        if code not in ohlcv_panel.columns.get_level_values(0):
            continue

        try:
            sub = ohlcv_panel[code].copy()
            sub = sub.dropna(how="all")
            if len(sub) < 252:
                continue

            # 计算 11 因子 (日频)
            from ..v5.industry_factors import compute_all_factors
            daily_factors = compute_all_factors(sub, factor_cfg)

            # 日频 → 月频 (月末)
            monthly_factors = daily_factors.resample("M").last()

            # 保存缓存
            monthly_factors.to_parquet(cache_path)
            result[code] = monthly_factors
        except Exception as e:
            print(f"  [{code}] 量价因子计算失败: {e}")
            continue

    return result


# ============================================================
# 3. Y (asset returns): 日频 → 月频
# ============================================================
def load_monthly_asset_returns() -> pd.DataFrame:
    """计算资产收益, 日频 → 月频.

    Returns:
        DataFrame (T_monthly, 56) 月频资产收益.
    """
    cache = HF_DIR / "v7_6_Y_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 加载 ETF NAV
    nav_path = REAL_DIR / "etf_nav_2018-01-01_2026-06-30.parquet"
    if not nav_path.exists():
        raise FileNotFoundError(f"NAV file not found: {nav_path}")

    nav = pd.read_parquet(nav_path)

    # 计算日频收益
    daily_returns = nav.pct_change()

    # 日频 → 月频 (月末)
    monthly_returns = daily_returns.resample("M").last()

    # 选择 expanded 池
    want = [c for c in EXPANDED_COLS if c in monthly_returns.columns]
    monthly_returns = monthly_returns[want]

    # 保存缓存
    monthly_returns.to_parquet(cache)
    return monthly_returns


# ============================================================
# 4. 合并因子面板
# ============================================================
def build_mixed_factor_panel(
    X_macro: pd.DataFrame,
    X_pv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """合并 9 macro + 11 量价 → 20 维因子.

    对于每个时间点, 量价因子取所有 ETF 的均值 (截面平均).

    Args:
        X_macro: (T_monthly, 9) 月频宏观因子
        X_pv: dict, code → (T_monthly, 11) 月频量价因子

    Returns:
        DataFrame (T_monthly, 20) 月频混合因子.
    """
    # 截面平均量价因子
    if X_pv:
        # 对齐时间索引
        common_idx = X_macro.index
        pv_dfs = []
        for code, df in X_pv.items():
            aligned = df.reindex(common_idx)
            pv_dfs.append(aligned)

        # 截面平均
        pv_stack = np.stack([df.values for df in pv_dfs], axis=0)  # (N_codes, T, 11)
        pv_mean = np.nanmean(pv_stack, axis=0)  # (T, 11)

        # 转为 DataFrame
        pv_cols = list(pv_dfs[0].columns)
        X_pv_mean = pd.DataFrame(pv_mean, index=common_idx, columns=pv_cols)
    else:
        X_pv_mean = pd.DataFrame(index=X_macro.index, columns=["pv_dummy"] * 11)

    # 合并
    X_mixed = pd.concat([X_macro, X_pv_mean], axis=1)

    return X_mixed


# ============================================================
# 5. 端到端加载
# ============================================================
def load_v7_6_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 v7.6 全部数据.

    Returns:
        X: (T_monthly, 20) 月频混合因子
        Y: (T_monthly, 56) 月频资产收益
    """
    # 1. 加载 9 macro (月频)
    X_macro = load_monthly_macro_factors()

    # 2. 加载 11 量价 (月频)
    X_pv = load_monthly_pv_factors()

    # 3. 合并因子
    X = build_mixed_factor_panel(X_macro, X_pv)

    # 4. 加载 Y (月频)
    Y = load_monthly_asset_returns()

    # 5. 对齐时间
    common_idx = X.index.intersection(Y.index)
    X = X.loc[common_idx]
    Y = Y.loc[common_idx]

    return X, Y


__all__ = [
    "load_monthly_macro_factors",
    "load_monthly_pv_factors",
    "load_monthly_asset_returns",
    "build_mixed_factor_panel",
    "load_v7_6_data",
]
