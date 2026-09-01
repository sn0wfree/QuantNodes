"""ML异常检测归因模块 - 为L1方案提供SHAP级特征贡献解释.

支持3个模型的归因:
- ZScore: 直接分解 (|z_i| * 标准化权重)
- IF: TreeExplainer (SHAP精确值)
- LOF: k近邻偏差近似 (替代慢速KernelExplainer)

融合公式: feature_contrib = 0.40*IF_SHAP + 0.35*LOF_approx + 0.25*Z_decomp
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 从v3导入FEATURES常量
from research_ml_anomaly_v3 import FEATURES


# 模型融合权重 (与 v3 的 compute_r_scores 一致)
MODEL_WEIGHTS = {
    "if": 0.40,
    "lof": 0.35,
    "zscore": 0.25,
}

# 模型归一化阈值 (用于展示时的颜色标记)
R_HIGH_THRESHOLD = 80
R_MID_THRESHOLD = 60


def compute_zscore_decomposition(X_scaled, mu, std):
    """ZScore模型归因: 每个特征的|z|值, 直接可解释.

    Args:
        X_scaled: 标准化后的特征矩阵 (n_samples, n_features)
        mu: 训练集均值向量
        std: 训练集标准差向量

    Returns:
        contributions: (n_samples, n_features) 每个特征的绝对z贡献
    """
    z_matrix = np.abs((X_scaled - mu) / std)
    return z_matrix


def compute_if_shap(m_if, X_scaled, max_samples=2000):
    """IF模型SHAP归因: TreeExplainer精确值.

    Args:
        m_if: 已训练的IsolationForest模型
        X_scaled: 标准化后的特征矩阵
        max_samples: 最大样本数 (超过则subsample, 避免过慢)

    Returns:
        shap_values: (n_samples, n_features) SHAP值
        sample_idx: 实际计算的样本索引 (None=全部)
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap未安装, 无法计算IF归因")

    if len(X_scaled) > max_samples:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X_scaled), size=max_samples, replace=False)
        X_sub = X_scaled.iloc[sample_idx]
    else:
        sample_idx = None
        X_sub = X_scaled

    explainer = shap.TreeExplainer(m_if)
    shap_values = explainer.shap_values(X_sub.values)

    # shap_values shape: (n_samples, n_features)
    # 取绝对值 (归因是"贡献", 不关心方向)
    return np.abs(shap_values), sample_idx


def compute_lof_approx(X_scaled, k=20):
    """LOF模型近似归因: 当前点与k近邻均值的偏差.

    核心思路:
    - LOF异常分数 = 当前点密度 / k近邻平均密度
    - 密度低 ≈ 与近邻距离大
    - 所以"贡献大的特征" ≈ 与近邻偏差大的特征

    Args:
        X_scaled: 标准化后的特征矩阵
        k: 近邻数 (默认20, 与LOF训练一致)

    Returns:
        contributions: (n_samples, n_features) 每个特征的偏差贡献
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled.values)
    _, indices = nbrs.kneighbors(X_scaled.values)

    n_samples, n_features = X_scaled.shape
    contributions = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        neighbor_mean = X_scaled.values[indices[i]].mean(axis=0)
        contributions[i] = (X_scaled.values[i] - neighbor_mean) ** 2

    return contributions


def normalize_contributions(contrib, reference_min=0, reference_max=None):
    """归一化到0-100区间, 与R_score可比较."""
    contrib_arr = np.asarray(contrib)
    if reference_max is None:
        reference_max = float(contrib_arr.max())
    if reference_max <= reference_min:
        return np.zeros_like(contrib_arr)
    return ((contrib_arr - reference_min) / (reference_max - reference_min + 1e-8) * 100).clip(0, 100)


def combine_contributions(if_contrib, lof_contrib, z_contrib, weights=None):
    """融合3个模型的归因, 加权求和.

    Args:
        if_contrib: (n, n_features) IF归因
        lof_contrib: (n, n_features) LOF归因
        z_contrib: (n, n_features) ZScore归因
        weights: 模型权重字典, 默认MODEL_WEIGHTS

    Returns:
        combined: (n, n_features) 融合归因
    """
    if weights is None:
        weights = MODEL_WEIGHTS
    return (
        weights["if"] * if_contrib +
        weights["lof"] * lof_contrib +
        weights["zscore"] * z_contrib
    )


def compute_model_agreement(if_score, lof_score, z_score, threshold=70):
    """模型一致性检查: 3个模型是否都认为异常.

    Returns:
        agreement: dict {agree_high, agree_low, disagree, scores}
    """
    if_norm = normalize_contributions(if_score)
    lof_norm = normalize_contributions(lof_score)
    z_norm = normalize_contributions(z_score)

    if_high = if_norm > threshold
    lof_high = lof_norm > threshold
    z_high = z_norm > threshold

    agree_high = if_high & lof_high & z_high
    agree_low = (~if_high) & (~lof_high) & (~z_high)
    disagree = ~(agree_high | agree_low)

    return {
        "if_score": float(if_norm),
        "lof_score": float(lof_norm),
        "zscore_score": float(z_norm),
        "agree_high": bool(agree_high),
        "agree_low": bool(agree_low),
        "disagree": bool(disagree),
    }


def generate_natural_language(date, r_score, top_features, model_agreement, feature_values):
    """生成自然语言解读.

    Args:
        date: 异常日期
        r_score: R分数
        top_features: [(name, contribution, direction)] top3特征
        model_agreement: 模型一致性字典
        feature_values: {feature: z_value} 各特征z值

    Returns:
        text: str 自然语言解读
    """
    severity = "🔴 高" if r_score > R_HIGH_THRESHOLD else "🟡 中" if r_score > R_MID_THRESHOLD else "🟢 低"

    if model_agreement["disagree"]:
        consensus = "模型分歧, 单一指标驱动, 需谨慎判断"
    elif model_agreement["agree_high"]:
        consensus = "3模型一致, 多指标共振, 系统性异常"
    else:
        consensus = "3模型一致, 信号中等"

    top_text = ", ".join([f"{name}({direction})" for name, _, direction in top_features[:3]])

    return f"{severity}异常 | {consensus} | Top3: {top_text}"


def explain_anomaly_day(date, X_scaled_row, feature_names, contributions_row, model_agreement, raw_z_values):
    """为单个异常日生成完整解释.

    Args:
        date: 日期
        X_scaled_row: 当日的标准化特征向量
        feature_names: 特征名列表
        contributions_row: 当日融合归因向量
        model_agreement: 模型一致性字典
        raw_z_values: 各特征原始z值

    Returns:
        explanation: dict 含全部字段
    """
    contrib_series = pd.Series(contributions_row, index=feature_names)
    top3 = contrib_series.nlargest(3)

    top_features = []
    for feat_name, contrib_val in top3.items():
        z_val = raw_z_values[feat_name]
        direction = "+" if z_val > 0 else "-"
        top_features.append((feat_name, float(contrib_val), f"{direction}{abs(z_val):.1f}σ"))

    r_score = (model_agreement["if_score"] * MODEL_WEIGHTS["if"] +
               model_agreement["lof_score"] * MODEL_WEIGHTS["lof"] +
               model_agreement["zscore_score"] * MODEL_WEIGHTS["zscore"])

    text = generate_natural_language(date, r_score, top_features, model_agreement, raw_z_values)

    return {
        "date": date,
        "r_score": float(r_score),
        "model_if": model_agreement["if_score"],
        "model_lof": model_agreement["lof_score"],
        "model_zscore": model_agreement["zscore_score"],
        "consensus": "high" if model_agreement["agree_high"] else "low" if model_agreement["agree_low"] else "disagree",
        "top1_feature": top_features[0][0] if len(top_features) > 0 else None,
        "top1_z": top_features[0][2] if len(top_features) > 0 else None,
        "top1_contrib": float(top_features[0][1]) if len(top_features) > 0 else 0,
        "top2_feature": top_features[1][0] if len(top_features) > 1 else None,
        "top3_feature": top_features[2][0] if len(top_features) > 2 else None,
        "interpretation": text,
    }


def full_explain_pipeline(df, scaler, m_if, m_lof, m_zscore, r_threshold=60):
    """完整归因流程: 从数据到逐日解释.

    Args:
        df: 完整数据 (含FEATURES列)
        scaler: 已训练的StandardScaler
        m_if: 已训练的IsolationForest
        m_lof: 已训练的LocalOutlierFactor
        m_zscore: ZScore模型 (含mu, std属性)
        r_threshold: R分数筛选阈值 (默认60)

    Returns:
        explanations: list of dict 逐日解释
        meta: dict 运行信息
    """
    import time

    t0 = time.time()
    X = df[FEATURES].dropna()
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    # 1. 计算R_score (复用v3逻辑)
    from research_ml_anomaly_v3 import compute_r_scores
    r_score = compute_r_scores(df, scaler, m_if, m_lof, m_zscore)

    # 2. 筛选R>threshold日
    high_r_dates = r_score[r_score > r_threshold].index
    print(f"[Explain] 筛选R>{r_threshold}: {len(high_r_dates)}天 / 总{len(r_score)}天 ({len(high_r_dates)/len(r_score):.1%})")

    # 3. 计算3个模型的归因 (向量化)
    t1 = time.time()
    z_decomp = compute_zscore_decomposition(X_scaled, m_zscore.mu, m_zscore.std)
    t_z = time.time() - t1

    t1 = time.time()
    if_shap, if_sample_idx = compute_if_shap(m_if, X_scaled, max_samples=len(X_scaled))
    t_if = time.time() - t1

    t1 = time.time()
    lof_approx = compute_lof_approx(X_scaled, k=20)
    t_lof = time.time() - t1

    # 4. 处理IF样本索引 (如果subsample了, 需要对齐)
    if if_sample_idx is not None:
        if_shap_full = np.zeros((len(X_scaled), X_scaled.shape[1]))
        if_shap_full[if_sample_idx] = if_shap
        if_shap = if_shap_full

    # 5. 归一化
    if_norm = normalize_contributions(if_shap)
    lof_norm = normalize_contributions(lof_approx)
    z_norm = normalize_contributions(z_decomp)

    # 6. 融合
    combined = combine_contributions(if_norm, lof_norm, z_norm)

    # 7. 逐日生成解释
    t1 = time.time()
    explanations = []
    feature_names = list(X_scaled.columns)

    for date in high_r_dates:
        if date not in X_scaled.index:
            continue
        idx = X_scaled.index.get_loc(date)
        if not isinstance(idx, (int, np.integer)):
            idx = idx

        # 原始z值 (用于方向标注)
        raw_z = pd.Series(X_scaled.iloc[idx].values, index=feature_names)

        # 模型一致性
        agreement = compute_model_agreement(
            if_score=np.array([if_norm[idx].sum()]),
            lof_score=np.array([lof_norm[idx].sum()]),
            z_score=np.array([z_norm[idx].sum()]),
        )

        # 生成解释
        explanation = explain_anomaly_day(
            date=date,
            X_scaled_row=X_scaled.iloc[idx],
            feature_names=feature_names,
            contributions_row=combined[idx],
            model_agreement=agreement,
            raw_z_values=raw_z,
        )
        explanations.append(explanation)

    t_explain = time.time() - t1

    meta = {
        "total_time": time.time() - t0,
        "zscore_time": t_z,
        "if_shap_time": t_if,
        "lof_approx_time": t_lof,
        "explain_time": t_explain,
        "n_explained": len(explanations),
        "r_threshold": r_threshold,
    }

    return explanations, meta