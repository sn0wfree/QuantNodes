# coding=utf-8
"""v7.6 数据加载: 12 macro + 17 量价, 周频.

v7.6 = v7.3 (9 macro) + v5 (11 量价) + 增强因子 (3 macro + 6 量价), 周频对齐.

数据流:
  1. 12 macro factors: 周频 (9 原始 + 3 新增)
  2. 17 量价 factors: 日频 → 周频 (resample('W').last())
  3. Y (asset returns): 日频 → 周频 (resample('W').last())

输入:
  - ~/Public/高频宏观因子/高频宏观因子跟踪_output_2026-06-01.xlsx (9 macro)
  - data/high_freq_macro/macro_*.parquet (3 新增宏观因子)
  - data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet (ETF OHLCV)
  - data/real/etf_nav_2018-01-01_2026-06-30.parquet (ETF NAV)

输出:
  - data/high_freq_macro/v7_6_X_macro_weekly.parquet (T_weekly, 12)
  - data/high_freq_macro/v7_6_X_pv_weekly.parquet (T_weekly, 56×17)
  - data/high_freq_macro/v7_6_Y_weekly.parquet (T_weekly, 56)
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

# 增强因子
from .enhanced_factors import (
    EnhancedFactorConfig,
    compute_all_enhanced_factors_panel,
    load_enhanced_macro_factors,
)


# ============================================================
# 1. 12 macro factors: 周频 (9 原始 + 3 新增)
# ============================================================
def load_weekly_macro_factors() -> pd.DataFrame:
    """加载 12 宏观因子, 周频.

    9 个原始宏观因子 (NAV levels) + 3 个新增宏观因子 (DXY, VIX, 实际利率)

    Returns:
        DataFrame (T_weekly, 12) 周频宏观因子.
    """
    cache = HF_DIR / "v7_6_X_macro_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 加载 9 原始宏观因子 (周频)
    weekly = load_macro_factors()

    # 加载 3 新增宏观因子 (周频)
    enhanced_macro = load_enhanced_macro_factors()

    # 对齐时间
    common_idx = weekly.index.intersection(enhanced_macro.index)
    weekly = weekly.loc[common_idx]
    enhanced_macro = enhanced_macro.loc[common_idx]

    # 合并
    merged = pd.concat([weekly, enhanced_macro], axis=1)

    # 保存缓存
    merged.to_parquet(cache)
    return merged


# ============================================================
# 2. 17 量价 factors: 日频 → 周频 (11 原始 + 6 增强)
# ============================================================
def load_weekly_pv_factors(
    factor_cfg: FactorEngineConfig | None = None,
    enhanced_cfg: EnhancedFactorConfig | None = None,
    include_enhanced: bool = True,
) -> dict[str, pd.DataFrame]:
    """计算 17 量价因子, 日频 → 周频.

    Args:
        factor_cfg: 原始因子引擎配置 (None = 默认)
        enhanced_cfg: 增强因子配置 (None = 默认)
        include_enhanced: 是否包含增强因子 (默认 True)

    Returns:
        dict, code → DataFrame (T_weekly, K_pv) 周频量价因子.
    """
    cache_dir = HF_DIR / "v7_6_pv_weekly"
    cache_dir.mkdir(parents=True, exist_ok=True)

    factor_cfg = factor_cfg or FactorEngineConfig()
    enhanced_cfg = enhanced_cfg or EnhancedFactorConfig()
    codes = sorted(set(EXPANDED_COLS) - set(EXPANDED_BOND_INDICES))  # 51 ETFs

    # 加载市场基准 (用于特质波动率)
    market_close = None
    if include_enhanced:
        try:
            benchmark = pd.read_parquet(HF_DIR / "v9_benchmark_沪深300.parquet")
            market_close = benchmark.iloc[:, 0]
        except Exception as e:
            print(f"  加载市场基准失败: {e}")

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

            # 计算 11 原始因子 (日频)
            from ..v5.industry_factors import compute_all_factors
            daily_factors = compute_all_factors(sub, factor_cfg)

            # 计算 6 增强因子 (日频)
            if include_enhanced:
                enhanced_factors = compute_all_enhanced_factors_panel(
                    ohlcv_panel[[code]], market_close, enhanced_cfg
                )
                if code in enhanced_factors:
                    enhanced_df = enhanced_factors[code]
                    # 合并原始因子和增强因子
                    daily_factors = pd.concat([daily_factors, enhanced_df], axis=1)

            # 日频 → 周频 (周末)
            weekly_factors = daily_factors.resample("W").last()

            # 保存缓存
            weekly_factors.to_parquet(cache_path)
            result[code] = weekly_factors
        except Exception as e:
            print(f"  [{code}] 量价因子计算失败: {e}")
            continue

    return result


# ============================================================
# 3. Y (asset returns): 日频 → 周频
# ============================================================
def load_weekly_asset_returns() -> pd.DataFrame:
    """计算资产收益, 日频 → 周频.

    Returns:
        DataFrame (T_weekly, 56) 周频资产收益.
    """
    cache = HF_DIR / "v7_6_Y_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 加载 ETF NAV
    nav_path = REAL_DIR / "etf_nav_2018-01-01_2026-06-30.parquet"
    if not nav_path.exists():
        raise FileNotFoundError(f"NAV file not found: {nav_path}")

    nav = pd.read_parquet(nav_path)

    # 先采样净值再算收益 (正确方法)
    weekly_nav = nav.resample("W").last()
    weekly_returns = weekly_nav.pct_change()

    # 选择 expanded 池
    want = [c for c in EXPANDED_COLS if c in weekly_returns.columns]
    weekly_returns = weekly_returns[want]

    # 保存缓存
    weekly_returns.to_parquet(cache)
    return weekly_returns


# ============================================================
# 4. 截面标准化
# ============================================================
def cross_sectional_standardize(
    X_panel: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """截面标准化.

    对每个时间点、每个因子独立标准化。

    Args:
        X_panel: (T, N, K) 因子面板
        method: "zscore" 或 "rank"

    Returns:
        X_std: 标准化后的因子面板
    """
    T, N, K = X_panel.shape
    X_std = X_panel.copy()

    for t in range(T):
        for k in range(K):
            vals = X_panel[t, :, k]
            valid = ~np.isnan(vals)
            if valid.sum() <= 1:
                continue

            if method == "zscore":
                mean = vals[valid].mean()
                std = vals[valid].std()
                if std > 1e-10:
                    X_std[t, valid, k] = (vals[valid] - mean) / std
            elif method == "rank":
                from scipy.stats import rankdata
                ranks = rankdata(vals[valid], method='average')
                X_std[t, valid, k] = ranks / len(ranks)

    return X_std


# ============================================================
# 5. 合并因子面板
# ============================================================
def build_mixed_factor_panel(
    X_macro: pd.DataFrame,
    X_pv: dict[str, pd.DataFrame],
    asset_codes: list[str],
    macro_use_log_return: bool = True,
    standardize: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """合并 macro + 量价 → 面板格式.

    对于每个资产 i, 构造 x_{i,t} = [macro_t, pv_{i,t}]
    其中 macro_t 是全局因子 (所有资产相同), pv_{i,t} 是资产特异因子.

    Args:
        X_macro: (T_weekly, K_macro) 周频宏观因子 (NAV levels)
        X_pv: dict, code → (T_weekly, K_pv) 周频量价因子
        asset_codes: 资产代码列表
        macro_use_log_return: 是否对宏观因子用对数收益率 (默认 True)
        standardize: 截面标准化方法 ("zscore", "rank", None)

    Returns:
        X_panel: (T_weekly, N_assets, K) 周频混合因子面板
        valid_codes: 有效资产代码列表
    """
    common_idx = X_macro.index
    N = len(asset_codes)
    K_macro = X_macro.shape[1]

    # 获取量价因子维度
    first_valid_code = next(iter(X_pv), None)
    if first_valid_code is not None:
        K_pv = X_pv[first_valid_code].shape[1]
    else:
        K_pv = 0

    K = K_macro + K_pv

    # 初始化面板
    X_panel = np.full((len(common_idx), N, K), np.nan)

    # 填充宏观因子 (所有资产相同)
    if macro_use_log_return:
        # 用对数收益率: r_t = ln(NAV_t / NAV_{t-1})
        # 只对 FACTOR_COLS 中的 NAV 指数做 log 变换
        # 排除已是收益率/差值/rank 的增强宏观因子 (如 dxy_logret, vix, real_rate_diff 等)
        from .data_loader import FACTOR_COLS
        _nav_cols = set(FACTOR_COLS)

        X_macro_logret = X_macro.copy()
        for col in X_macro.columns:
            if col in _nav_cols:
                # NAV 指数 (全部正值): log(NAV_t / NAV_{t-1})
                X_macro_logret[col] = np.log(X_macro[col] / X_macro[col].shift(1))
            # 非 NAV 列 (已是 return/rate/rank/diff): 保持原值
        # 第一行是 NaN (shift 导致), 从第二行开始填充
        X_panel[1:, :, :K_macro] = X_macro_logret.values[1:, np.newaxis, :]
    else:
        # 用原始 NAV levels
        for i in range(N):
            X_panel[:, i, :K_macro] = X_macro.values

    # 填充量价因子 (资产特异)
    valid_codes = []
    for j, code in enumerate(asset_codes):
        if code in X_pv:
            df = X_pv[code]
            aligned = df.reindex(common_idx)
            X_panel[:, j, K_macro:] = aligned.values
            valid_codes.append(code)

    # 截面 NaN 填充: 量价因子的 NaN (ETF 停牌/未上市) 用截面中位数填充
    # 对每个时间点 t 和因子 k, 用当天其他资产的中位数填充 NaN
    for t in range(len(common_idx)):
        for k in range(K_macro, K):
            col = X_panel[t, :, k]
            if np.any(np.isnan(col)):
                median_val = np.nanmedian(col)
                if not np.isnan(median_val):
                    col[np.isnan(col)] = median_val

    # 截面标准化 (可选)
    if standardize is not None:
        X_panel = cross_sectional_standardize(X_panel, method=standardize)

    return X_panel, valid_codes


# ============================================================
# 5. 端到端加载
# ============================================================
def load_v7_6_data(
    macro_use_log_return: bool = True,
    standardize: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.6 全部数据.

    Args:
        macro_use_log_return: 是否对宏观因子用对数收益率 (默认 True)
        standardize: 截面标准化方法 ("zscore", "rank", None)

    Returns:
        X_panel: (T_weekly, N_assets, K) 周频混合因子面板
        Y: (T_weekly, N_assets) 周频资产收益
        valid_codes: 有效资产代码列表
    """
    # 1. 加载 9 macro (周频)
    X_macro = load_weekly_macro_factors()

    # 2. 加载 11 量价 (周频)
    X_pv = load_weekly_pv_factors()

    # 3. 加载 Y (周频)
    Y = load_weekly_asset_returns()

    # 4. 对齐时间 (以 Y 的时间为准)
    common_idx = X_macro.index.intersection(Y.index)
    X_macro = X_macro.loc[common_idx]
    Y = Y.loc[common_idx]

    # 5. 合并因子
    asset_codes = list(Y.columns)
    X_panel, valid_codes = build_mixed_factor_panel(
        X_macro, X_pv, asset_codes,
        macro_use_log_return=macro_use_log_return,
        standardize=standardize,
    )

    # 6. 过滤有效资产
    Y = Y[valid_codes]

    return X_panel, Y, valid_codes


# ============================================================
# 4. 日频 ETF 收益 (用于日频 NAV 计算)
# ============================================================
def load_daily_etf_returns() -> pd.DataFrame:
    """加载日频 ETF 收益 (用于日频 NAV 计算).

    Returns:
        DataFrame (T_daily, N_etf) 日频 ETF 收益.
    """
    cache = HF_DIR / "v7_6_daily_etf_returns.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 加载 ETF NAV
    nav_path = REAL_DIR / "etf_nav_2018-01-01_2026-06-30.parquet"
    if not nav_path.exists():
        raise FileNotFoundError(f"NAV file not found: {nav_path}")

    nav = pd.read_parquet(nav_path)

    # 计算日频收益
    daily_returns = nav.pct_change()

    # 保存缓存
    daily_returns.to_parquet(cache)
    return daily_returns


def load_weekly_monday_open_returns(codes: list[str]) -> pd.DataFrame:
    """加载周一开盘到周五收盘的周频收益 (不含周末隔夜).

    使用 OHLCV 数据源 (同一数据源), 计算:
      ret[t] = OHLCV_friday_close[t] / OHLCV_monday_open[t] - 1

    Returns:
        DataFrame (T_weekly, N_etf) 周一开盘到周五收盘的收益.
        index 与 Y_df 对齐 (每周日).
    """
    cache = HF_DIR / "v7_10_monday_open_returns.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    ohlc_path = REAL_DIR / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
    ohlc = pd.read_parquet(ohlc_path)
    opens = ohlc.xs("open", level=1, axis=1)
    closes = ohlc.xs("close", level=1, axis=1)

    # 只保留 v7.10 的 43 个 code
    valid_codes = [c for c in codes if c in opens.columns and c in closes.columns]

    # 每周一开盘价 (W-MON: 周一所在的周)
    monday_open = opens[valid_codes].resample("W-MON").first()
    # 每周五收盘价
    friday_close = closes[valid_codes].resample("W-FRI").last()

    # 对齐: monday_open index 是周一, friday_close 是周五
    # 将 monday_open 的 index +4 天 → 周五, 表示"本周一的开盘价, 对齐到上周五"
    monday_open.index = monday_open.index + pd.Timedelta(days=4)  # 周一 → 周五
    monday_open = monday_open.reindex(friday_close.index)

    # 收益 = 周五收盘 / 周一开盘 - 1 (周一到周五收益, 不含周末隔夜)
    monday_open_returns = friday_close / monday_open - 1

    # 对齐到 Y_df 的 index (Y 是周日, friday_close 是周五, 差 2 天)
    monday_open_returns.index = monday_open_returns.index + pd.Timedelta(days=2)

    monday_open_returns.to_parquet(cache)
    return monday_open_returns


def load_weekly_ohlcv_returns(codes: list[str]) -> pd.DataFrame:
    """加载 OHLCV 周频收益 (周五到周五, 用于与 monday_open_returns 同源对比).

    计算: ret[t] = OHLCV_friday_close[t] / OHLCV_friday_close[t-1] - 1

    Returns:
        DataFrame (T_weekly, N_etf) 周五到周五收益.
        index 与 Y_df 对齐 (每周日).
    """
    cache = HF_DIR / "v7_10_weekly_ohlcv_returns.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    ohlc_path = REAL_DIR / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
    ohlc = pd.read_parquet(ohlc_path)
    closes = ohlc.xs("close", level=1, axis=1)

    valid_codes = [c for c in codes if c in closes.columns]
    friday_close = closes[valid_codes].resample("W-FRI").last()
    weekly_returns = friday_close.pct_change()

    # 对齐到 Y_df 的 index (Y 是周日, friday_close 是周五, 差 2 天)
    weekly_returns.index = weekly_returns.index + pd.Timedelta(days=2)

    weekly_returns.to_parquet(cache)
    return weekly_returns


def load_v7_9_data() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.9 数据 (去重 36 因子 + log 变换).

    v7.9 = v7.8 - {f4_vol_vol, f9_pv_corr, f21_reversal}
         + log transform on f3_amt_vol, f6_ls_total, f7_ls_change, f12_amihud, f22_rsi

    Returns:
        X_panel: (T_weekly, N_assets, 36) 周频因子面板
        Y: (T_weekly, N_assets) 周频资产收益
        valid_codes: 有效资产代码列表
    """
    X_panel = np.load(HF_DIR / "v7_9_X_panel.npy")
    Y = pd.read_parquet(HF_DIR / "v7_9_Y_weekly.parquet")
    codes = (HF_DIR / "v7_9_codes.csv").read_text().strip().split("\n")[1:]
    return X_panel, Y, codes


# ============================================================
# v7.10 混合标准化
# ============================================================
MACRO_K = 17  # 宏观因子数量 (k=0..16)

# 极端偏度因子 (需要 Winsorize)
SKEWED_FACTORS = {"f13_rv", "f17_idio_vol", "f15_max5", "f1_first_mom"}


def winsorize_factor(X_panel: np.ndarray, factor_names: list[str],
                     lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """对极端偏度因子做 Winsorize (截断异常值).

    Parameters:
        X_panel: (T, N, K) 因子面板
        factor_names: 因子名称列表
        lower, upper: 分位数截断点

    Returns:
        X_out: Winsorize 后的因子面板
    """
    T, N, K = X_panel.shape
    X_out = X_panel.copy()

    for k in range(K):
        fname = factor_names[k] if k < len(factor_names) else f"f{k}"
        if fname not in SKEWED_FACTORS:
            continue

        vals = X_out[:, :, k].ravel()
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue

        q_low = np.quantile(valid, lower)
        q_high = np.quantile(valid, upper)
        X_out[:, :, k] = np.clip(X_out[:, :, k], q_low, q_high)

    return X_out


def standardize_v7_10(X_panel: np.ndarray, factor_names: list[str]) -> np.ndarray:
    """v7.10 混合标准化: 宏观=时间序列Z-score, PV=截面Z-score.

    处理顺序:
      1. Winsorize 极端偏度因子 (f13_rv, f17_idio_vol, f15_max5, f1_first_mom)
      2. 宏观因子 (k=0-16): 时间序列 Z-score (每个因子在时间维度上标准化)
      3. PV 因子 (k=17-35): 截面 Z-score (每个时间点、每个因子在截面上标准化)

    Parameters:
        X_panel: (T, N, K) 因子面板 (v7.9 原始)
        factor_names: 因子名称列表

    Returns:
        X_std: 标准化后的因子面板
    """
    T, N, K = X_panel.shape

    # Step 1: Winsorize 极端偏度因子
    X_std = winsorize_factor(X_panel, factor_names)

    # Step 2: 宏观因子 - 时间序列 Z-score
    for k in range(min(MACRO_K, K)):
        vals = X_std[:, :, k].ravel()
        valid = vals[~np.isnan(vals)]
        if len(valid) < 2:
            continue
        mean_t = np.mean(valid)
        std_t = np.std(valid)
        if std_t > 1e-10:
            X_std[:, :, k] = (X_std[:, :, k] - mean_t) / std_t

    # Step 3: PV 因子 - 截面 Z-score
    for t in range(T):
        for k in range(MACRO_K, K):
            vals = X_std[t, :, k]
            valid = vals[~np.isnan(vals)]
            if len(valid) <= 1:
                continue
            mean_cs = np.mean(valid)
            std_cs = np.std(valid)
            if std_cs > 1e-10:
                X_std[t, :, k] = (X_std[t, :, k] - mean_cs) / std_cs

    return X_std


def generate_v7_10_data() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """生成 v7.10 数据 (v7.9 + 混合标准化).

    Returns:
        X_panel: (T_weekly, N_assets, 36) 标准化后因子面板
        Y: (T_weekly, N_assets) 周频资产收益
        valid_codes: 有效资产代码列表
    """
    X_raw, Y, codes = load_v7_9_data()
    factor_names = (HF_DIR / "v7_9_factor_names.csv").read_text().strip().split("\n")[1:]

    X_std = standardize_v7_10(X_raw, factor_names)

    # 保存
    np.save(HF_DIR / "v7_10_X_panel.npy", X_std)
    (HF_DIR / "v7_10_codes.csv").write_text("\n".join(["code"] + codes))
    (HF_DIR / "v7_10_factor_names.csv").write_text("\n".join(["factor"] + factor_names))
    Y.to_parquet(HF_DIR / "v7_10_Y_weekly.parquet")

    return X_std, Y, codes


def load_v7_10_data() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.10 数据 (混合标准化).

    如果数据文件不存在, 自动调用 generate_v7_10_data() 生成.

    Returns:
        X_panel: (T_weekly, N_assets, 36) 标准化后因子面板
        Y: (T_weekly, N_assets) 周频资产收益
        valid_codes: 有效资产代码列表
    """
    npy_path = HF_DIR / "v7_10_X_panel.npy"
    parquet_path = HF_DIR / "v7_10_Y_weekly.parquet"
    codes_path = HF_DIR / "v7_10_codes.csv"

    if not npy_path.exists() or not parquet_path.exists() or not codes_path.exists():
        return generate_v7_10_data()

    X_panel = np.load(npy_path)
    Y = pd.read_parquet(parquet_path)
    codes = codes_path.read_text().strip().split("\n")[1:]
    return X_panel, Y, codes


__all__ = [
    "load_weekly_macro_factors",
    "load_weekly_pv_factors",
    "load_weekly_asset_returns",
    "load_daily_etf_returns",
    "load_weekly_monday_open_returns",
    "build_mixed_factor_panel",
    "load_v7_6_data",
    "load_v7_9_data",
    "standardize_v7_10",
    "generate_v7_10_data",
    "load_v7_10_data",
]
