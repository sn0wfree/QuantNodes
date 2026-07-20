# coding=utf-8
"""v7.13 图谱距离因子: DCC Z-Score + 网络拓扑 + 跨类别尾部依赖.

来源: ~/Public/comovement/resonance_warning/data/ (只读, 不修改)

三个组件:
  A. DCC Z-Score (每对资产, 截面因子): dcc_z_mean, dcc_z_max
  B. 网络拓扑 (7 维, 时序因子): avg_path, clustering, centrality_entropy,
                                density, largest_comp, spectral_radius, modularity
  C. 跨类别尾部依赖 (时序因子): cross_class_tail_dep
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# A. DCC Z-Score (每对资产)
# ============================================================
def compute_dcc_zscore_pairwise(
    returns: pd.DataFrame,
    short_window: int = 60,
    long_window: int = 252,
    min_periods: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算每对资产的 DCC Z-Score.

    z(i,j) = (rho_short - rho_long) / sigma_rho
    rho_short: short_window 天滚动相关性
    rho_long: long_window 天滚动相关性
    sigma_rho: 滚动相关性的标准差

    对每个资产 i:
      dcc_z_mean[i] = mean(|z(i,j)| for all j != i)
      dcc_z_max[i] = max(|z(i,j)| for all j != i)

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        short_window: 短期窗口 (默认 60 天)
        long_window: 长期窗口 (默认 252 天)
        min_periods: 最小观测数

    Returns:
        dcc_z_mean: (T, N) 每个资产的平均 DCC Z-Score
        dcc_z_max: (T, N) 每个资产的最大 DCC Z-Score
    """
    T, N = returns.shape
    assets = returns.columns.tolist()

    # 初始化输出
    dcc_z_mean = pd.DataFrame(np.nan, index=returns.index, columns=assets)
    dcc_z_max = pd.DataFrame(np.nan, index=returns.index, columns=assets)

    # 预计算滚动相关性矩阵 (T, N, N)
    # 为了效率, 只计算需要的时间步
    for t in range(long_window, T):
        # 短期相关性
        short_data = returns.iloc[t - short_window:t]
        if len(short_data) < min_periods:
            continue
        corr_short = short_data.corr().values

        # 长期相关性
        long_data = returns.iloc[t - long_window:t]
        corr_long = long_data.corr().values

        # 相关性标准差 (用长期窗口)
        # 计算滚动相关性的时间序列, 然后取标准差
        corr_std = np.full((N, N), np.nan)
        for i in range(N):
            for j in range(i + 1, N):
                # 计算滚动相关性的时间序列
                corr_series = []
                for s in range(long_window, t + 1):
                    window_data = returns.iloc[s - short_window:s]
                    if len(window_data) >= min_periods:
                        c = window_data.iloc[:, i].corr(window_data.iloc[:, j])
                        corr_series.append(c)
                if len(corr_series) > 5:
                    corr_std[i, j] = np.std(corr_series)
                    corr_std[j, i] = corr_std[i, j]

        # 计算 Z-Score
        z_matrix = np.full((N, N), np.nan)
        for i in range(N):
            for j in range(i + 1, N):
                if not np.isnan(corr_std[i, j]) and corr_std[i, j] > 1e-10:
                    z_matrix[i, j] = (corr_short[i, j] - corr_long[i, j]) / corr_std[i, j]
                    z_matrix[j, i] = z_matrix[i, j]

        # 计算每个资产的 mean 和 max
        for i in range(N):
            z_row = np.abs(z_matrix[i, :])
            z_row = z_row[~np.isnan(z_row)]
            if len(z_row) > 0:
                dcc_z_mean.iloc[t, i] = np.mean(z_row)
                dcc_z_max.iloc[t, i] = np.max(z_row)

    return dcc_z_mean, dcc_z_max


def compute_dcc_zscore_fast(
    returns: pd.DataFrame,
    short_window: int = 60,
    long_window: int = 252,
    min_periods: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """快速版 DCC Z-Score (用向量化计算).

    简化: 用全局相关性标准差近似每对的标准差.
    """
    T, N = returns.shape
    assets = returns.columns.tolist()

    dcc_z_mean = pd.DataFrame(np.nan, index=returns.index, columns=assets)
    dcc_z_max = pd.DataFrame(np.nan, index=returns.index, columns=assets)

    # 预计算短期和长期相关性
    for t in range(long_window, T):
        short_data = returns.iloc[t - short_window:t]
        long_data = returns.iloc[t - long_window:t]

        if len(short_data) < min_periods:
            continue

        corr_short = short_data.corr().values
        corr_long = long_data.corr().values

        # 用长期窗口内多个短期窗口的标准差近似
        n_windows = max(3, (long_window - short_window) // short_window)
        corr_diffs = []
        for w in range(n_windows):
            start = w * short_window
            end = start + short_window
            if end <= long_window:
                window_data = long_data.iloc[start:end]
                c = window_data.corr().values
                corr_diffs.append(c)

        if len(corr_diffs) >= 3:
            corr_std = np.std(corr_diffs, axis=0)
            # Z-Score
            z_matrix = np.where(corr_std > 1e-10,
                               (corr_short - corr_long) / corr_std,
                               0.0)
            np.fill_diagonal(z_matrix, np.nan)

            for i in range(N):
                z_row = np.abs(z_matrix[i, :])
                valid = ~np.isnan(z_row)
                if valid.sum() > 0:
                    dcc_z_mean.iloc[t, i] = np.mean(z_row[valid])
                    dcc_z_max.iloc[t, i] = np.max(z_row[valid])

    return dcc_z_mean, dcc_z_max


# ============================================================
# B. 网络拓扑特征 (7 维)
# ============================================================
def compute_network_topology(
    returns: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.3,
    min_periods: int = 30,
) -> pd.DataFrame:
    """计算 7 维网络拓扑特征 (时序因子).

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        window: 滚动窗口
        threshold: 相关性阈值 (|corr| > threshold → 边存在)
        min_periods: 最小观测数

    Returns:
        DataFrame (T-window+1, 7):
          avg_path: 平均最短路径长度
          clustering_coeff: 聚类系数
          centrality_entropy: 特征向量中心性熵
          density: 网络密度
          largest_component: 最大连通分量比例
          spectral_radius: 谱半径
          modularity: 模块度
    """
    try:
        import networkx as nx
        HAS_NX = True
    except ImportError:
        HAS_NX = False

    T, N = returns.shape
    features = []
    dates = []

    for t in range(window, T):
        window_data = returns.iloc[t - window:t]
        if len(window_data) < min_periods:
            features.append([0.0] * 7)
            dates.append(returns.index[t])
            continue

        corr = window_data.corr().values
        adj = (np.abs(corr) > threshold).astype(float)
        np.fill_diagonal(adj, 0)

        if HAS_NX:
            G = nx.from_numpy_array(adj)
            try:
                # 平均最短路径长度
                if nx.is_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                else:
                    # 取最大连通分量
                    largest_cc = max(nx.connected_components(G), key=len)
                    subG = G.subgraph(largest_cc)
                    avg_path = nx.average_shortest_path_length(subG)

                # 聚类系数
                clustering = nx.average_clustering(G)

                # 特征向量中心性
                try:
                    centrality = nx.eigenvector_centrality(G, max_iter=1000)
                    cent_values = np.array(list(centrality.values()))
                    cent_values = cent_values / (cent_values.sum() + 1e-10)
                    centrality_entropy = -np.sum(cent_values * np.log(cent_values + 1e-10))
                except Exception:
                    centrality_entropy = 0.0

                # 网络密度
                density = nx.density(G)

                # 最大连通分量比例
                largest_component = len(max(nx.connected_components(G), key=len)) / N

                # 谱半径
                try:
                    eigenvalues = np.linalg.eigvalsh(adj)
                    spectral_radius = np.max(np.abs(eigenvalues))
                except Exception:
                    spectral_radius = 0.0

                # 模块度
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                    modularity = nx.community.modularity(G, communities)
                except Exception:
                    modularity = 0.0

            except Exception:
                avg_path = 0.0
                clustering = 0.0
                centrality_entropy = 0.0
                density = 0.0
                largest_component = 0.0
                spectral_radius = 0.0
                modularity = 0.0
        else:
            # 无 networkx, 用简化计算
            density = np.sum(adj) / (N * (N - 1))
            clustering = 0.0
            avg_path = 0.0
            centrality_entropy = 0.0
            largest_component = 0.0
            spectral_radius = 0.0
            modularity = 0.0

        features.append([
            avg_path, clustering, centrality_entropy, density,
            largest_component, spectral_radius, modularity,
        ])
        dates.append(returns.index[t])

    return pd.DataFrame(
        features, index=dates,
        columns=["avg_path", "clustering_coeff", "centrality_entropy",
                 "density", "largest_component", "spectral_radius", "modularity"],
    )


# ============================================================
# C. 跨资产类别尾部依赖
# ============================================================
def compute_cross_class_tail_dep(
    returns: pd.DataFrame,
    equity_codes: list[str],
    non_equity_codes: list[str],
    window: int = 60,
    quantile: float = 0.05,
) -> pd.Series:
    """计算权益 vs 非权益的尾部依赖 (时序因子).

    tail_dep = P(两者同时 < q% 分位数) / P(任一 < q% 分位数)

    Parameters:
        returns: (T, N) 日频收益 DataFrame
        equity_codes: 权益 ETF 代码列表
        non_equity_codes: 非权益 ETF 代码列表
        window: 滚动窗口
        quantile: 分位数 (默认 5%)

    Returns:
        Series (T-window+1,), 尾部依赖值
    """
    eq_codes = [c for c in equity_codes if c in returns.columns]
    neq_codes = [c for c in non_equity_codes if c in returns.columns]

    if not eq_codes or not neq_codes:
        return pd.Series(np.nan, index=returns.index[window - 1:])

    eq_returns = returns[eq_codes]
    neq_returns = returns[neq_codes]

    # 权益和非权益的平均收益
    eq_mean = eq_returns.mean(axis=1)
    neq_mean = neq_returns.mean(axis=1)

    T = len(returns)
    tail_deps = []

    for t in range(window, T):
        eq_window = eq_mean.iloc[t - window:t]
        neq_window = neq_mean.iloc[t - window:t]

        eq_q = eq_window.quantile(quantile)
        neq_q = neq_window.quantile(quantile)

        eq_extreme = (eq_window < eq_q).astype(float)
        neq_extreme = (neq_window < neq_q).astype(float)

        joint = (eq_extreme * neq_extreme).sum()
        marginal = eq_extreme.sum() + neq_extreme.sum() - joint

        if marginal > 0:
            tail_deps.append(joint / marginal)
        else:
            tail_deps.append(0.0)

    return pd.Series(tail_deps, index=returns.index[window:])


# ============================================================
# 综合计算
# ============================================================
def get_graph_distance_factor_names() -> dict[str, list[str]]:
    """返回图谱距离因子名."""
    return {
        "dcc_zscore": ["dcc_z_mean", "dcc_z_max"],
        "topology": ["avg_path", "clustering_coeff", "centrality_entropy",
                     "density", "largest_component", "spectral_radius", "modularity"],
        "tail_dep": ["cross_class_tail_dep"],
    }


def compute_all_graph_distance_factors(
    daily_returns: pd.DataFrame,
    equity_codes: list[str],
    non_equity_codes: list[str],
    fast_mode: bool = True,
) -> dict[str, pd.DataFrame]:
    """计算所有图谱距离因子.

    Parameters:
        daily_returns: (T, N) 日频收益 DataFrame
        equity_codes: 权益 ETF 代码列表
        non_equity_codes: 非权益 ETF 代码列表
        fast_mode: 是否用快速版 DCC Z-Score

    Returns:
        dict, 组件名 → DataFrame
    """
    factors = {}

    # A. DCC Z-Score
    print("  计算 DCC Z-Score...")
    if fast_mode:
        dcc_z_mean, dcc_z_max = compute_dcc_zscore_fast(daily_returns)
    else:
        dcc_z_mean, dcc_z_max = compute_dcc_zscore_pairwise(daily_returns)
    factors["dcc_z_mean"] = dcc_z_mean
    factors["dcc_z_max"] = dcc_z_max

    # B. 网络拓扑
    print("  计算网络拓扑...")
    topo = compute_network_topology(daily_returns, window=60, threshold=0.3)
    # 广播到所有资产 (时序因子)
    for col in topo.columns:
        factors[col] = topo[col]

    # C. 跨类别尾部依赖
    print("  计算跨类别尾部依赖...")
    tail_dep = compute_cross_class_tail_dep(
        daily_returns, equity_codes, non_equity_codes, window=60, quantile=0.05,
    )
    factors["cross_class_tail_dep"] = tail_dep

    return factors
