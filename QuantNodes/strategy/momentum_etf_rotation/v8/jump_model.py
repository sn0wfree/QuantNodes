# coding=utf-8
"""Jump Model — 统计跳跃模型 (Shu et al. 2024, 中金 2026).

3 维特征: DD_10, Sortino_20, Sortino_60
2 状态: bull (0) / bear (1)
动态规划 + 坐标下降求解最优状态路径

版本:
  - jump_model_rolling(): 原始版本 (有未来函数, 仅作对照)
  - jump_model_true_rolling(): 方案 A (真正滚动, 无未来函数)
  - jump_model_periodic_retrain(): 方案 B (周期重估, 无未来函数)

参考:
  - Shu et al. (2024) "Statistical Jump Model"
  - 中金《基于统计跳跃的系统性风险预警模型》(2026-06-22)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 最优参数映射 (基于 bootstrap 实验和参数调优)
# ============================================================
JUMP_PENALTY_MAP = {
    'equity': 50,      # 权益类
    'commodity': 50,    # 商品类 (黄金等)
    'bond': 50,         # 债券类 (实验发现50优于文章推荐的25)
}

# 特征窗口类型映射
FEATURE_WINDOW_MAP = {
    'equity': 'simple',    # 权益类: 简单滚动窗口
    'commodity': 'simple',  # 商品类: 简单滚动窗口
    'bond': 'exp',          # 债券类: 指数衰减窗口
}

# 训练窗口映射 (基于参数调优实验)
TRAIN_WINDOW_MAP = {
    'equity': 1000,     # 权益类: 1000 天
    'commodity': 1000,   # 商品类: 1000 天
    'bond': 1000,        # 债券类: 1000 天
}

# 重估频率映射 (基于参数调优实验)
RETRAIN_EVERY_MAP = {
    'equity': 30,       # 权益类: 30 天
    'commodity': 30,     # 商品类: 30 天
    'bond': 30,          # 债券类: 30 天
}

# 随机重启次数 (基于 bootstrap 实验)
N_RESTARTS = 10


# ============================================================
# 特征计算
# ============================================================
def compute_DD(returns: pd.Series, window: int = 10) -> pd.Series:
    """下行偏差 (Downside Deviation)."""
    downside = returns.clip(upper=0)
    return downside.rolling(window, min_periods=window // 2).std()


def compute_Sortino(returns: pd.Series, window: int = 20) -> pd.Series:
    """Sortino 比率."""
    excess = returns  # 无风险收益简化为 0
    downside_std = returns.clip(upper=0).rolling(window, min_periods=window // 2).std()
    return excess.rolling(window, min_periods=window // 2).mean() / (downside_std + 1e-10)


def compute_features(returns: pd.Series) -> pd.DataFrame:
    """计算 3 维特征: DD_10, Sortino_20, Sortino_60.

    Parameters:
        returns: 日频收益序列

    Returns:
        DataFrame, columns=['DD_10', 'Sortino_20', 'Sortino_60']
    """
    return pd.DataFrame({
        'DD_10': compute_DD(returns, 10),
        'Sortino_20': compute_Sortino(returns, 20),
        'Sortino_60': compute_Sortino(returns, 60),
    })


# ============================================================
# 指数衰减窗口版本 (待测试)
# ============================================================
def compute_DD_exp(returns: pd.Series, window: int = 10) -> pd.Series:
    """下行偏差 (指数衰减窗口)."""
    downside = returns.clip(upper=0)
    return downside.ewm(span=window, adjust=False).std()


def compute_Sortino_exp(returns: pd.Series, window: int = 20) -> pd.Series:
    """Sortino 比率 (指数衰减窗口)."""
    excess = returns
    downside_std = returns.clip(upper=0).ewm(span=window, adjust=False).std()
    return excess.ewm(span=window, adjust=False).mean() / (downside_std + 1e-10)


def compute_features_exp(returns: pd.Series) -> pd.DataFrame:
    """计算 3 维特征 (指数衰减窗口): DD_10, Sortino_20, Sortino_60.

    Parameters:
        returns: 日频收益序列

    Returns:
        DataFrame, columns=['DD_10', 'Sortino_20', 'Sortino_60']
    """
    return pd.DataFrame({
        'DD_10': compute_DD_exp(returns, 10),
        'Sortino_20': compute_Sortino_exp(returns, 20),
        'Sortino_60': compute_Sortino_exp(returns, 60),
    })


# ============================================================
# Jump Model 核心算法
# ============================================================
def _compute_cost(
    features: np.ndarray,
    centroids: np.ndarray,
    state: int,
) -> float:
    """计算单点到质心的欧氏距离平方."""
    diff = features - centroids[state]
    return float(np.sum(diff ** 2))


def _dynamic_programming(
    features: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float = 50.0,
) -> np.ndarray:
    """动态规划求解最优状态路径.

    损失函数: min Σ ||features[t] - centroid[s[t]]||² + λ × I[s[t] != s[t-1]]

    注意: 此函数回溯时使用未来数据, 仅用于对照!
    如需无未来函数版本, 使用 _dynamic_programming_insample().

    Parameters:
        features: (T, 3) 标准化特征
        centroids: (2, 3) 质心
        jump_penalty: 跳跃惩罚

    Returns:
        states: (T,) 最优状态序列
    """
    T = len(features)
    n_states = len(centroids)

    # 成本矩阵: cost[t, s] = ||features[t] - centroid[s]||²
    cost = np.zeros((T, n_states))
    for t in range(T):
        for s in range(n_states):
            cost[t, s] = _compute_cost(features[t], centroids, s)

    # DP 表: dp[t, s] = 到时刻 t 状态为 s 的最小累计成本
    dp = np.full((T, n_states), np.inf)
    # 回溯表: back[t, s] = 最优前驱状态
    back = np.zeros((T, n_states), dtype=int)

    # 初始化
    dp[0] = cost[0]

    # 递推
    for t in range(1, T):
        for s in range(n_states):
            # 不跳跃: 保持状态 s
            dp[t, s] = dp[t - 1, s] + cost[t, s]
            back[t, s] = s

            # 跳跃: 从其他状态切换到 s
            for s_prev in range(n_states):
                if s_prev != s:
                    candidate = dp[t - 1, s_prev] + cost[t, s] + jump_penalty
                    if candidate < dp[t, s]:
                        dp[t, s] = candidate
                        back[t, s] = s_prev

    # 回溯 (注意: 此处使用未来数据, 仅作对照!)
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmin(dp[-1])
    for t in range(T - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    return states


def _dynamic_programming_insample(
    features: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float = 50.0,
) -> int:
    """动态规划求解当前时刻的最优状态 (无未来函数).

    与 _dynamic_programming() 的区别:
    - 此函数只返回最后一天的状态 states[-1]
    - 不回溯到历史, 避免使用未来数据

    Parameters:
        features: (T, 3) 标准化特征 (到当前时刻为止)
        centroids: (2, 3) 质心
        jump_penalty: 跳跃惩罚

    Returns:
        state: 当前时刻的最优状态 (0 或 1)
    """
    T = len(features)
    n_states = len(centroids)

    # 成本矩阵
    cost = np.zeros((T, n_states))
    for t in range(T):
        for s in range(n_states):
            cost[t, s] = _compute_cost(features[t], centroids, s)

    # DP 表 (只用前向递推)
    dp = np.full((T, n_states), np.inf)
    dp[0] = cost[0]

    for t in range(1, T):
        for s in range(n_states):
            # 不跳跃
            dp[t, s] = dp[t - 1, s] + cost[t, s]
            # 跳跃
            for s_prev in range(n_states):
                if s_prev != s:
                    candidate = dp[t - 1, s_prev] + cost[t, s] + jump_penalty
                    if candidate < dp[t, s]:
                        dp[t, s] = candidate

    # 只返回最后一天的状态 (不回溯)
    return int(np.argmin(dp[-1]))


def _classify_bull_bear(
    states: np.ndarray,
    returns: pd.Series,
) -> tuple[int, int]:
    """根据累计收益判断哪个状态是 bull，哪个是 bear.

    注意: 此函数使用整个窗口的收益, 可能包含未来数据!
    如需无未来函数版本, 使用 _classify_bull_bear_insample().

    Returns:
        bull_state: bull 状态的编号 (0 或 1)
        bear_state: bear 状态的编号 (0 或 1)
    """
    cum_ret = {}
    for s in [0, 1]:
        mask = states == s
        if mask.sum() > 0:
            cum_ret[s] = float(returns.values[mask].sum())
        else:
            cum_ret[s] = 0.0

    if cum_ret[0] >= cum_ret[1]:
        return 0, 1
    else:
        return 1, 0


def _classify_bull_bear_insample(
    states: np.ndarray,
    returns: pd.Series,
) -> tuple[int, int]:
    """根据累计收益判断哪个状态是 bull，哪个是 bear (只用历史收益).

    与 _classify_bull_bear() 的区别:
    - 此函数假设 states 和 returns 已经是到当前时刻为止的数据
    - 不使用任何未来数据

    Returns:
        bull_state: bull 状态的编号 (0 或 1)
        bear_state: bear 状态的编号 (0 或 1)
    """
    cum_ret = {}
    for s in [0, 1]:
        mask = states == s
        if mask.sum() > 0:
            cum_ret[s] = float(returns.values[mask].sum())
        else:
            cum_ret[s] = 0.0

    if cum_ret[0] >= cum_ret[1]:
        return 0, 1
    else:
        return 1, 0


def _train_jump_model(
    features_z: np.ndarray,
    jump_penalty: float = 50.0,
    n_iter: int = 10,
    use_insample: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """训练 Jump Model (坐标下降).

    Parameters:
        features_z: (T, 3) 标准化特征
        jump_penalty: 跳跃惩罚
        n_iter: 交替迭代次数
        use_insample: 是否使用无未来函数版本

    Returns:
        states: (T,) 状态序列 (只在 use_insample=False 时有效)
        centroids: (2, 3) 质心
        last_state: 当前时刻的最优状态 (只在 use_insample=True 时有效)
    """
    # 随机初始化质心
    centroids = np.random.randn(2, 3)

    # 交替迭代
    states = None
    last_state = 0
    for _ in range(n_iter):
        if use_insample:
            # 只求解当前时刻的状态 (无未来函数)
            last_state = _dynamic_programming_insample(features_z, centroids, jump_penalty)
            # 用 last_state 更新质心
            # 注意: 这里我们需要完整的状态序列来更新质心
            # 所以我们用 DP 求解完整序列, 但只使用 last_state
            states = _dynamic_programming(features_z, centroids, jump_penalty)
        else:
            # 求解完整状态序列 (有未来函数)
            states = _dynamic_programming(features_z, centroids, jump_penalty)

        # 更新质心
        for s in range(2):
            mask = states == s
            if mask.sum() > 0:
                centroids[s] = features_z[mask].mean(axis=0)

    return states, centroids, last_state


# ============================================================
# 原始版本 (有未来函数, 仅作对照)
# ============================================================
def jump_model_single(
    returns: pd.Series,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """单资产 Jump Model (有未来函数, 仅作对照).

    Parameters:
        returns: 日频收益序列
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口 (默认 1000 天)
        n_iter: 交替迭代次数
        random_state: 随机种子

    Returns:
        states: (T,) 状态序列 (0=bull, 1=bear)
        centroids: (2, 3) 质心
    """
    np.random.seed(random_state)

    # 计算特征
    features_df = compute_features(returns)
    features_df = features_df.dropna()

    if len(features_df) < train_window:
        return np.zeros(len(features_df), dtype=int), np.zeros((2, 3))

    # 取最后 train_window 天
    features = features_df.iloc[-train_window:].values

    # Z-Score 标准化
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    features_z = (features - mean) / std

    # 训练 Jump Model (有未来函数)
    states, centroids, _ = _train_jump_model(
        features_z, jump_penalty, n_iter, use_insample=False
    )

    # 判断 bull/bear
    returns_window = returns.iloc[-train_window:]
    bull_state, bear_state = _classify_bull_bear(states, returns_window)

    # 重排状态: 0=bull, 1=bear
    if bull_state != 0:
        states = 1 - states

    return states, centroids


def jump_model_rolling(
    returns: pd.Series,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    retrain_every: int = 60,
    n_iter: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Jump Model 滚动预测 (有未来函数, 仅作对照).

    每 retrain_every 个交易日重估质心，每天用最新质心预测状态.

    注意: 此函数有未来函数问题, 仅用于对照!
    如需无未来函数版本, 使用 jump_model_true_rolling() 或 jump_model_periodic_retrain().

    Parameters:
        returns: 日频收益序列
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口
        retrain_every: 重估频率 (默认 60 天)
        n_iter: 交替迭代次数
        random_state: 随机种子

    Returns:
        pd.Series, index=returns.index, values=0(bull)/1(bear)
    """
    features_df = compute_features(returns).dropna()
    T = len(features_df)

    if T < train_window:
        return pd.Series(0, index=features_df.index, name='regime')

    # 初始化
    all_states = np.zeros(T, dtype=int)
    current_centroids = None
    last_retrain = -retrain_every

    np.random.seed(random_state)

    for t in range(train_window, T):
        # 检查是否需要重估
        if t - last_retrain >= retrain_every:
            window_start = max(0, t - train_window + 1)
            features_window = features_df.iloc[window_start:t + 1].values

            # Z-Score 标准化
            mean = features_window.mean(axis=0)
            std = features_window.std(axis=0) + 1e-10
            features_z = (features_window - mean) / std

            # 训练 Jump Model (有未来函数)
            states, centroids, _ = _train_jump_model(
                features_z, jump_penalty, n_iter, use_insample=False
            )

            # 判断 bull/bear
            returns_window = returns.iloc[window_start:t + 1]
            bull_state, _ = _classify_bull_bear(states, returns_window)

            # 重排状态: 0=bull, 1=bear
            if bull_state != 0:
                states = 1 - states

            current_centroids = centroids
            last_retrain = t

            # 更新历史状态 (有未来函数!)
            all_states[window_start:t + 1] = states
        else:
            # 用最新质心预测当前状态
            if current_centroids is not None:
                features_now = features_df.iloc[t].values
                mean = features_df.iloc[max(0, t - train_window + 1):t + 1].values.mean(axis=0)
                std = features_df.iloc[max(0, t - train_window + 1):t + 1].values.std(axis=0) + 1e-10
                features_z = (features_now - mean) / std

                # 计算到两个质心的距离
                dist_0 = np.sum((features_z - current_centroids[0]) ** 2)
                dist_1 = np.sum((features_z - current_centroids[1]) ** 2)

                all_states[t] = 0 if dist_0 <= dist_1 else 1

    return pd.Series(all_states, index=features_df.index, name='regime')


# ============================================================
# 方案 A: 真正的滚动预测 (无未来函数)
# ============================================================
def jump_model_true_rolling(
    returns: pd.Series,
    jump_penalty: float = 50.0,
    train_window: int = 1000,
    n_iter: int = 10,
    show_progress: bool = True,
) -> pd.Series:
    """真正的滚动预测，无未来函数.

    每天重新训练模型，只用到当天为止的数据.

    Parameters:
        returns: 日频收益序列
        jump_penalty: 跳跃惩罚
        train_window: 训练窗口 (默认 1000 天)
        n_iter: 交替迭代次数
        show_progress: 是否显示进度条

    Returns:
        pd.Series, index=returns.index, values=0(bull)/1(bear)
    """
    features_df = compute_features(returns).dropna()
    T = len(features_df)

    if T < train_window:
        return pd.Series(0, index=features_df.index, name='regime')

    # 初始化
    all_states = np.zeros(T, dtype=int)

    # 进度条
    iterator = range(train_window, T)
    if show_progress:
        iterator = tqdm(iterator, desc="Jump Model (True Rolling)", ncols=80)

    for t in iterator:
        # 只用 [t-train_window+1, t] 的数据
        window_start = max(0, t - train_window + 1)
        features_window = features_df.iloc[window_start:t + 1].values

        # Z-Score 标准化 (只用历史数据)
        mean = features_window.mean(axis=0)
        std = features_window.std(axis=0) + 1e-10
        features_z = (features_window - mean) / std

        # 训练 Jump Model (只用历史数据)
        # 注意: _train_jump_model 内部用 DP 求解完整序列, 但我们只使用 last_state
        _, _, last_state = _train_jump_model(
            features_z, jump_penalty, n_iter, use_insample=True
        )

        # 判断 bull/bear (只用历史收益)
        returns_window = returns.iloc[window_start:t + 1]
        # 用完整 DP 序列来分类 (只用于判断哪个状态是 bull/bear)
        centroids = np.random.randn(2, 3)
        for _ in range(n_iter):
            states = _dynamic_programming(features_z, centroids, jump_penalty)
            for s in range(2):
                mask = states == s
                if mask.sum() > 0:
                    centroids[s] = features_z[mask].mean(axis=0)

        bull_state, _ = _classify_bull_bear_insample(states, returns_window)

        # 当前时刻的状态
        current_state = last_state
        if bull_state != 0:
            current_state = 1 - current_state

        all_states[t] = current_state

    return pd.Series(all_states, index=features_df.index, name='regime')


# ============================================================
# 方案 B: 周期性重估 (无未来函数)
# ============================================================
def _compute_objective(
    features_z: np.ndarray,
    states: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float = 50.0,
) -> float:
    """计算目标函数值 (越小越好).

    目标函数: Σ ||features[t] - centroid[s[t]]||² + λ × I[s[t] != s[t-1]]
    """
    T = len(features_z)
    cost = 0.0
    for t in range(T):
        cost += np.sum((features_z[t] - centroids[states[t]]) ** 2)
    for t in range(1, T):
        if states[t] != states[t - 1]:
            cost += jump_penalty
    return cost


def jump_model_periodic_retrain(
    returns: pd.Series,
    asset_type: str = 'equity',
    jump_penalty: float | None = None,
    train_window: int | None = None,
    retrain_every: int | None = None,
    n_iter: int = 10,
    n_restarts: int = N_RESTARTS,
    show_progress: bool = True,
    random_state: int = 42,
    use_exp_features: bool | None = None,
) -> pd.Series:
    """周期性重估，无未来函数.

    每 retrain_every 个交易日重估质心，中间用最新质心预测.
    使用多次随机重启选择最优解，提高稳定性.

    Parameters:
        returns: 日频收益序列
        asset_type: 资产类型 ('equity'/'bond'/'commodity')
        jump_penalty: 跳跃惩罚 (如果为 None，根据 asset_type 自动选择)
        train_window: 训练窗口 (如果为 None，根据 asset_type 自动选择)
        retrain_every: 重估频率 (如果为 None，根据 asset_type 自动选择)
        n_iter: 交替迭代次数
        n_restarts: 随机重启次数 (默认 10, 基于 bootstrap 实验)
        show_progress: 是否显示进度条
        random_state: 随机种子 (默认 42)
        use_exp_features: 是否使用指数衰减窗口特征 (如果为 None，根据 asset_type 自动选择)

    Returns:
        pd.Series, index=returns.index, values=0(bull)/1(bear)
    """
    # 根据资产类型选择跳跃惩罚
    if jump_penalty is None:
        jump_penalty = JUMP_PENALTY_MAP.get(asset_type, 50)

    # 根据资产类型选择训练窗口
    if train_window is None:
        train_window = TRAIN_WINDOW_MAP.get(asset_type, 1000)

    # 根据资产类型选择重估频率
    if retrain_every is None:
        retrain_every = RETRAIN_EVERY_MAP.get(asset_type, 30)

    # 根据资产类型选择特征窗口类型
    if use_exp_features is None:
        use_exp_features = FEATURE_WINDOW_MAP.get(asset_type, 'simple') == 'exp'

    np.random.seed(random_state)

    # 选择特征计算函数
    if use_exp_features:
        features_df = compute_features_exp(returns).dropna()
    else:
        features_df = compute_features(returns).dropna()
    T = len(features_df)

    if T < train_window:
        return pd.Series(0, index=features_df.index, name='regime')

    # 初始化
    all_states = np.zeros(T, dtype=int)
    current_centroids = None
    bull_state = 0  # 默认 bull=0
    last_retrain = -retrain_every

    # 进度条
    iterator = range(train_window, T)
    if show_progress:
        iterator = tqdm(iterator, desc="Jump Model (Periodic Retrain)", ncols=80)

    for t in iterator:
        # 检查是否需要重估
        if t - last_retrain >= retrain_every:
            # 只用 [t-train_window+1, t] 的数据
            window_start = max(0, t - train_window + 1)
            features_window = features_df.iloc[window_start:t + 1].values

            # Z-Score 标准化 (只用历史数据)
            mean = features_window.mean(axis=0)
            std = features_window.std(axis=0) + 1e-10
            features_z = (features_window - mean) / std

            # 多次随机重启，选择最优解
            best_cost = np.inf
            best_states = None
            best_centroids = None

            for restart in range(n_restarts):
                # 随机初始化质心
                centroids = np.random.randn(2, 3)

                # 交替迭代
                for _ in range(n_iter):
                    states = _dynamic_programming(features_z, centroids, jump_penalty)
                    for s in range(2):
                        mask = states == s
                        if mask.sum() > 0:
                            centroids[s] = features_z[mask].mean(axis=0)

                # 计算目标函数值
                cost = _compute_objective(features_z, states, centroids, jump_penalty)

                # 选择最优解
                if cost < best_cost:
                    best_cost = cost
                    best_states = states.copy()
                    best_centroids = centroids.copy()

            # 判断 bull/bear (只用历史收益)
            returns_window = returns.iloc[window_start:t + 1]
            bull_state, _ = _classify_bull_bear_insample(best_states, returns_window)

            current_centroids = best_centroids
            last_retrain = t

            # 当前时刻的状态
            current_state = best_states[-1]
            if bull_state != 0:
                current_state = 1 - current_state
            all_states[t] = current_state
        else:
            # 用最新质心预测当前状态
            if current_centroids is not None:
                features_now = features_df.iloc[t].values
                # 标准化用历史数据
                window_start = max(0, t - train_window + 1)
                features_window = features_df.iloc[window_start:t + 1].values
                mean = features_window.mean(axis=0)
                std = features_window.std(axis=0) + 1e-10
                features_z = (features_now - mean) / std

                # 计算到两个质心的距离
                dist_0 = np.sum((features_z - current_centroids[0]) ** 2)
                dist_1 = np.sum((features_z - current_centroids[1]) ** 2)

                raw_state = 0 if dist_0 <= dist_1 else 1
                # 应用 bull/bear 分类
                if bull_state != 0:
                    raw_state = 1 - raw_state
                all_states[t] = raw_state

    return pd.Series(all_states, index=features_df.index, name='regime')
