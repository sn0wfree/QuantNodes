# coding=utf-8
"""v7.14 相关性距离因子: 6 个截面相关性因子.

全部是截面因子 (每个资产在每个时间步有不同的值).

六个因子:
  1. distance_to_centroid  — 相关性空间中到中心的 L2 距离
  2. avg_pairwise_corr     — 与所有其他资产的真实平均相关性
  3. local_clustering_coeff — 阈值化相关性网络的局部聚类系数
  4. corr_diff             — 同类别相关性 - 跨类别相关性 (权益/商品/债券)
  5. avg_tail_dep          — 平均下尾依赖 (10% 分位数)
  6. corr_momentum         — 相关性变化 (corr_60d - corr_60d_lag20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# 1. distance_to_centroid
# ============================================================
def compute_distance_to_centroid(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """L2 distance from asset's correlation profile to mean profile.

    对每个资产 i, 在每个时间步 t:
      corr_i = [rho(i,1), rho(i,2), ..., rho(i,N)]  (与所有资产的相关性)
      mean_corr = 所有资产的平均相关性 profile
      distance_i = ||corr_i - mean_corr||_2

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口 (默认 60 天)

    Returns:
        (T, N) DataFrame, 每个资产的距离
    """
    T, N = returns.shape
    codes = returns.columns.tolist()
    result = pd.DataFrame(np.nan, index=returns.index, columns=codes)

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        corr_mat = window_data.corr().values  # (N, N)
        mean_profile = corr_mat.mean(axis=0)  # (N,)
        for i in range(N):
            result.iloc[t, i] = np.sqrt(((corr_mat[i] - mean_profile) ** 2).sum())

    return result


# ============================================================
# 2. avg_pairwise_corr
# ============================================================
def compute_avg_pairwise_corr(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """与所有其他资产的真实平均相关性.

    对每个资产 i, 在每个时间步 t:
      avg_corr_i = mean(|rho(i,j)| for all j != i)

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口 (默认 60 天)

    Returns:
        (T, N) DataFrame, 每个资产的平均相关性
    """
    T, N = returns.shape
    codes = returns.columns.tolist()
    result = pd.DataFrame(np.nan, index=returns.index, columns=codes)

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        corr_mat = window_data.corr().values  # (N, N)
        for i in range(N):
            others = list(range(i)) + list(range(i + 1, N))
            result.iloc[t, i] = np.abs(corr_mat[i, others]).mean()

    return result


# ============================================================
# 3. local_clustering_coeff
# ============================================================
def compute_local_clustering_coeff(
    returns: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """阈值化相关性网络的局部聚类系数.

    构建阈值化无向图: 边 = |rho(i,j)| > threshold
    局部聚类系数 = 资产 i 的邻居之间实际边数 / 最大可能边数

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口 (默认 60 天)
        threshold: 相关性阈值 (默认 0.3)

    Returns:
        (T, N) DataFrame, 每个资产的局部聚类系数
    """
    T, N = returns.shape
    codes = returns.columns.tolist()
    result = pd.DataFrame(np.nan, index=returns.index, columns=codes)

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        corr_mat = np.abs(window_data.corr().values)  # (N, N)
        adj = (corr_mat > threshold).astype(float)
        np.fill_diagonal(adj, 0)

        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                result.iloc[t, i] = 0.0
                continue
            # 邻居之间的实际边数
            edges = 0
            for a in range(k):
                for b in range(a + 1, k):
                    if adj[neighbors[a], neighbors[b]] > 0:
                        edges += 1
            result.iloc[t, i] = 2.0 * edges / (k * (k - 1))

    return result


# ============================================================
# 4. corr_diff (同类别 - 跨类别)
# ============================================================
def compute_corr_diff(
    returns: pd.DataFrame,
    category_map: dict[str, int],
    window: int = 60,
) -> pd.DataFrame:
    """同类别相关性 - 跨类别相关性.

    对每个资产 i:
      same_corr = mean(|rho(i,j)| for j in same category)
      cross_corr = mean(|rho(i,j)| for j in other categories)
      corr_diff_i = same_corr - cross_corr

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        category_map: {code: category_id} 资产分类映射
                      category_id: 0=权益, 1=商品, 2=债券
        window: 滚动窗口 (默认 60 天)

    Returns:
        (T, N) DataFrame, 每个资产的 corr_diff
    """
    T, N = returns.shape
    codes = returns.columns.tolist()
    result = pd.DataFrame(np.nan, index=returns.index, columns=codes)

    # 构建分类向量
    cats = np.array([category_map.get(c, 0) for c in codes])

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        corr_mat = np.abs(window_data.corr().values)  # (N, N)

        for i in range(N):
            same_mask = (cats == cats[i])
            same_mask[i] = False  # 排除自身
            cross_mask = (cats != cats[i])

            same_corr = corr_mat[i, same_mask].mean() if same_mask.any() else 0.0
            cross_corr = corr_mat[i, cross_mask].mean() if cross_mask.any() else 0.0
            result.iloc[t, i] = same_corr - cross_corr

    return result


# ============================================================
# 5. avg_tail_dep
# ============================================================
def compute_avg_tail_dep(
    returns: pd.DataFrame,
    window: int = 60,
    quantile: float = 0.10,
) -> pd.DataFrame:
    """平均下尾依赖 (10% 分位数).

    对每个资产 i, 在每个时间步 t:
      用经验 copula: rank 变换后, P(两者都 < quantile) / P(其他 < quantile)
      avg_tail_dep_i = mean(tail_dep(i,j) for all j != i)

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口 (默认 60 天)
        quantile: 尾部分位数 (默认 10%)

    Returns:
        (T, N) DataFrame, 每个资产的平均尾部依赖
    """
    T, N = returns.shape
    codes = returns.columns.tolist()
    result = pd.DataFrame(np.nan, index=returns.index, columns=codes)

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        # Rank 变换 (经验 copula)
        ranked = window_data.rank(pct=True)
        q = quantile

        for i in range(N):
            tail_deps = []
            extreme_i = (ranked.iloc[:, i] < q).astype(float)
            for j in range(N):
                if j == i:
                    continue
                extreme_j = (ranked.iloc[:, j] < q).astype(float)
                joint = (extreme_i * extreme_j).sum()
                marginal = extreme_j.sum()
                if marginal > 0:
                    tail_deps.append(joint / marginal)
                else:
                    tail_deps.append(0.0)
            result.iloc[t, i] = np.mean(tail_deps)

    return result


# ============================================================
# 6. corr_momentum
# ============================================================
def compute_corr_momentum(
    returns: pd.DataFrame,
    window: int = 60,
    lag: int = 20,
) -> pd.DataFrame:
    """相关性变化 (corr_60d - corr_60d_lag20).

    对每个资产 i:
      corr_now = rolling correlation to market mean (window days)
      corr_past = corr_now shifted by lag days
      corr_momentum_i = corr_now - corr_past

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口 (默认 60 天)
        lag: 滞后期 (默认 20 天)

    Returns:
        (T, N) DataFrame, 每个资产的相关性动量
    """
    market_ret = returns.mean(axis=1)
    corr_to_market = returns.rolling(window).corr(market_ret)
    corr_past = corr_to_market.shift(lag)
    return corr_to_market - corr_past


# ============================================================
# 综合入口
# ============================================================
def compute_all_corr_factors(
    daily_returns: pd.DataFrame,
    category_map: dict[str, int],
    original_codes: list[str],
    window: int = 60,
    threshold: float = 0.3,
    quantile: float = 0.10,
    lag: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """计算 6 个相关性因子, 输出 (T, N_original, 6).

    用扩展后的资产池计算相关性, 但只输出原始资产的因子值.

    Parameters:
        daily_returns: (T, N_extended) 日频收益 DataFrame (包含扩展资产)
        category_map: {code: category_id} 资产分类映射
        original_codes: 原始资产代码列表 (只输出这些资产的因子)
        window: 滚动窗口
        threshold: 相关性网络阈值
        quantile: 尾部分位数
        lag: 相关性动量滞后期

    Returns:
        X_corr: (T, N_original, 6) 因子面板
        names: 因子名称列表
    """
    names = [
        'distance_to_centroid',
        'avg_pairwise_corr',
        'local_clustering_coeff',
        'corr_diff',
        'avg_tail_dep',
        'corr_momentum',
    ]

    print("  计算 distance_to_centroid...")
    f1 = compute_distance_to_centroid(daily_returns, window)

    print("  计算 avg_pairwise_corr...")
    f2 = compute_avg_pairwise_corr(daily_returns, window)

    print("  计算 local_clustering_coeff...")
    f3 = compute_local_clustering_coeff(daily_returns, window, threshold)

    print("  计算 corr_diff...")
    f4 = compute_corr_diff(daily_returns, category_map, window)

    print("  计算 avg_tail_dep...")
    f5 = compute_avg_tail_dep(daily_returns, window, quantile)

    print("  计算 corr_momentum...")
    f6 = compute_corr_momentum(daily_returns, window, lag)

    # 只取原始资产
    T = len(daily_returns)
    N_orig = len(original_codes)
    X_corr = np.full((T, N_orig, 6), np.nan)

    for k, df in enumerate([f1, f2, f3, f4, f5, f6]):
        for i, code in enumerate(original_codes):
            if code in df.columns:
                X_corr[:, i, k] = df[code].values

    # 周频对齐: 取每周最后一个值 (与 v7.10 一致)
    # 这里返回日频, 由调用方做周频对齐
    return X_corr, names
