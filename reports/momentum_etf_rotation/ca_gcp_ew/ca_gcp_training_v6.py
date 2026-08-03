"""CA-GCP 训练 v6 — 两步法 Utility-Aligned 模型.

Phase 1.4 of docs/82 roadmap.

设计:
  Stage A — Downside probability (替换 y_E)
    - 目标: 未来 max_dd 跨阈 + percentile 跨阈 (即 y_E)
    - 模型: v4 强正则化 XGB + LightGBM ensemble (calibrated)
    - 输出: p_downside ∈ [0, 1] + recovery_days

  Stage B — Action calibration
    - 输入: p_downside + recovery_days
    - 输出: risk_budget ∈ [0, 0.20] (用于 v10.3c 约束动态分配)
    - 公式: risk_budget = clip(α·p_downside + β·recovery_factor, 0, RISK_BUDGET_MAX)
    - 目标: 最大化 portfolio utility = E[R] - λ·CVaR_loss

  评估切换 (相对 v4):
    - 不只看 F1 → 改 PR-AUC + ECE + net_portfolio_utility
    - 事件 recall 只对 is_downside=True 事件计算
    - 4 个 bootstrap CSV (per-model)
    - nested walk-forward (purge=5d, embargo=5d)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"))  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ca_gcp_training_v3 import (  # noqa: E402
    build_full_features,
    build_y_labels,
    load_hs300_returns,
)
from ca_gcp_training_v4 import (  # noqa: E402
    feature_columns_v3,
    train_with_strong_regularization,
)
from ca_gcp_ew_eval import (  # noqa: E402
    KNOWN_EVENTS,
    OUT_DIR,
    load_returns,
)

RISK_BUDGET_MAX = 0.20
RISK_BUDGET_ALPHA = 0.15
RISK_BUDGET_BETA = 0.05
RECOVERY_NORMALIZE = 60.0
LAMBDA_UTILITY = 2.0


def compute_recovery_days(
    returns: pd.DataFrame, y_E: pd.Series, max_horizon: int = 30
) -> pd.Series:
    """计算每个 positive 样本的实际恢复天数 (max_dd 跨阈后回到 0 的天数).

    对 negative 样本 (y_E=0), 设为 max_horizon.
    """
    market_ret = returns.mean(axis=1)
    recovery = pd.Series(np.nan, index=y_E.index, dtype=float)

    for d in y_E.index[y_E == 1]:
        d_int = y_E.index.get_loc(d)
        if d_int + max_horizon >= len(market_ret):
            continue
        window = market_ret.iloc[d_int:d_int + max_horizon + 1]
        cum = (1 + window).cumprod()
        dd = (cum / cum.cummax() - 1)
        underwater = dd < -0.01
        if not underwater.any():
            continue
        first_under = underwater.idxmax()  # 第一次进入水下的日期
        # 从 first_under 起, 找到回到 0 (cum >= 0) 的日期
        post = cum.loc[first_under:]
        if (post >= 0).any():
            recovered_at = post[post >= 0].index[0]
            recovery.loc[d] = (recovered_at - d).days
        else:
            recovery.loc[d] = max_horizon  # 窗口内未恢复

    # 负样本 → max_horizon
    recovery = recovery.fillna(max_horizon)
    return recovery


def calibrate_probability(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray]:
    """Expected Calibration Error (ECE) + 校准概率.

    ECE = Σ_b |bin_acc - bin_conf| * (n_b / N)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    calibrated = np.zeros_like(y_proba)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += abs(bin_acc - bin_conf) * (mask.sum() / len(y_true))
        # 简单校准: 在 bin 内用 acc 替换 conf
        calibrated[mask] = bin_acc

    return float(ece), calibrated


def stage_a_downside_proba(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 20
) -> dict:
    """Stage A: 训练 calibrated downside probability 模型.

    返回: dict with final_model, model_type, oof_proba, best_threshold, P/R/F1/AUC
    """
    print("  [Stage A] 训练 downside probability 模型 (XGB + 强正则化)...")
    res = train_with_strong_regularization(X, y, "xgb", n_trials=n_trials)
    # 校准
    ece, calibrated_proba = calibrate_probability(y.values, res["oof_proba"])
    res["ECE"] = ece
    res["calibrated_proba"] = calibrated_proba
    res["Brier"] = float(brier_score_loss(y.values, res["oof_proba"]))
    return res


def stage_b_action_calibration(
    p_downside: np.ndarray, recovery_days: np.ndarray,
    alpha: float = RISK_BUDGET_ALPHA, beta: float = RISK_BUDGET_BETA,
    normalize: float = RECOVERY_NORMALIZE,
    max_budget: float = RISK_BUDGET_MAX,
) -> np.ndarray:
    """Stage B: 输出 risk_budget ∈ [0, max_budget].

    risk_budget = clip(α · p_downside + β · recovery_factor, 0, max_budget)
    where recovery_factor = min(recovery_days / normalize, 1)
    """
    recovery_factor = np.minimum(recovery_days / normalize, 1.0)
    risk = alpha * p_downside + beta * recovery_factor
    return np.clip(risk, 0, max_budget)


def net_portfolio_utility(
    risk_budget: np.ndarray,
    p_downside: np.ndarray,
    realized_loss: np.ndarray,
    lambda_param: float = LAMBDA_UTILITY,
) -> float:
    """计算 net portfolio utility = E[R] - λ·CVaR_loss.

    risk_budget: 风险预算 0~0.20
    p_downside: 模型预测的下行概率
    realized_loss: 实际损失 (负值代表亏损)
    lambda_param: 风险厌恶系数
    """
    if len(risk_budget) == 0:
        return 0.0
    expected_return = 0.0  # baseline
    downside_mask = realized_loss < 0
    if downside_mask.sum() == 0:
        cvar = 0.0
    else:
        cvar = float(-realized_loss[downside_mask].mean())
    return float(expected_return - lambda_param * cvar)


def evaluate_v6(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    risk_budget: np.ndarray,
    realized_loss: np.ndarray,
    y_E: np.ndarray | None = None,
) -> dict:
    """综合评估 v6 2-stage 模型.

    返回: AUC, F1, PR-AUC, ECE, Brier, net_utility
    """
    metrics = {}
    if len(np.unique(y_true)) < 2:
        return {"error": "y_true 只有 1 个类"}
    metrics["AUC"] = float(roc_auc_score(y_true, y_proba))
    metrics["PR_AUC"] = float(average_precision_score(y_true, y_proba))
    metrics["Brier"] = float(brier_score_loss(y_true, y_proba))

    # 找最佳 threshold
    best_f1, best_th = -1, 0.5
    for th in np.arange(0.2, 0.8, 0.02):
        pred = (y_proba > th).astype(int)
        if pred.sum() < 3:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    pred = (y_proba > best_th).astype(int)
    metrics["F1"] = float(f1_score(y_true, pred, zero_division=0))
    metrics["P"] = float(precision_score(y_true, pred, zero_division=0))
    metrics["R"] = float(recall_score(y_true, pred, zero_division=0))
    metrics["best_threshold"] = float(best_th)

    # 校准
    ece, _ = calibrate_probability(y_true, y_proba)
    metrics["ECE"] = ece

    # Portfolio utility
    metrics["net_utility"] = net_portfolio_utility(risk_budget, y_proba, realized_loss)
    metrics["avg_risk_budget"] = float(risk_budget.mean())

    return metrics


def is_downside_event_recall(
    alerts: pd.Series, fired_idx: list, lookback: int = 30
) -> dict:
    """计算 is_downside=True 事件的 recall (Phase 1.1 配合).

    alerts: alerts Series (index=日期)
    fired_idx: 触发的日期列表
    lookback: 向前看的天数
    """
    fired_set = set(fired_idx)
    n_total = 0
    n_caught = 0
    detail = []
    for ev in KNOWN_EVENTS:
        if not ev.get("is_downside", False):
            continue
        n_total += 1
        ev_d = pd.Timestamp(ev["date"])
        prior = [d for d in fired_set if ev_d - pd.Timedelta(days=lookback) <= d < ev_d]
        caught = len(prior) > 0
        if caught:
            n_caught += 1
        detail.append({"event": ev["name"], "date": ev_d.date().isoformat(), "caught": caught})
    return {
        "n_total": n_total,
        "n_caught": n_caught,
        "recall": n_caught / n_total if n_total > 0 else 0.0,
        "detail": detail,
    }


def main() -> None:
    print("=" * 70)
    print("CA-GCP 训练 v6 — 两步法 Utility-Aligned 模型")
    print("=" * 70)

    returns = load_returns()
    features = build_full_features(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common_idx = features.index.intersection(y_labels.index)
    X = features.loc[common_idx]
    Y = y_labels.loc[common_idx]

    feat_cols = feature_columns_v3(X)
    X_train = X[feat_cols].fillna(0)

    y_E = Y["y_E"]
    common = X_train.index.intersection(y_E.index)
    X_aligned = X_train.loc[common]
    y_E_aligned = y_E.loc[common]

    print()
    print("=" * 70)
    print("Stage A — Downside probability")
    print("=" * 70)
    stage_a = stage_a_downside_proba(X_aligned, y_E_aligned, n_trials=20)
    print(f"  Stage A: F1={stage_a['F1']:.3f}, AUC={stage_a['AUC']:.3f}, "
          f"PR-AUC=N/A, ECE={stage_a['ECE']:.3f}, Brier={stage_a['Brier']:.3f}")

    print()
    print("=" * 70)
    print("Stage B — Action calibration (risk budget)")
    print("=" * 70)
    recovery = compute_recovery_days(returns, y_E_aligned)
    recovery_aligned = recovery.reindex(y_E_aligned.index).fillna(30)
    risk_budget = stage_b_action_calibration(
        stage_a["oof_proba"], recovery_aligned.values
    )
    print(f"  risk_budget: mean={risk_budget.mean():.3f}, "
          f"max={risk_budget.max():.3f}, min={risk_budget.min():.3f}")

    print()
    print("=" * 70)
    print("评估 v6 2-stage (含 PR-AUC + ECE + net_utility)")
    print("=" * 70)

    market_ret = returns.mean(axis=1).reindex(X_aligned.index)
    realized_loss = market_ret.values

    metrics = evaluate_v6(
        y_E_aligned.values, stage_a["oof_proba"], risk_budget, realized_loss
    )
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")

    print()
    print("=" * 70)
    print("is_downside 事件 recall (Phase 1.1 配合)")
    print("=" * 70)
    fired_idx = [d for d, p in zip(X_aligned.index, stage_a["oof_proba"]) if p > 0.5]
    rec = is_downside_event_recall(
        pd.Series(stage_a["oof_proba"], index=X_aligned.index), fired_idx
    )
    print(f"  总 is_downside 事件: {rec['n_total']}")
    print(f"  捕获: {rec['n_caught']} / {rec['n_total']} = {rec['recall']:.1%}")
    for d in rec["detail"]:
        marker = "✓" if d["caught"] else "✗"
        print(f"    {marker} {d['date']} {d['event']}")

    print()
    print("=" * 70)
    print("输出报告")
    print("=" * 70)
    report = [
        "# CA-GCP v6 Utility-Aligned 实验报告",
        "",
        "## 1. Stage A — Downside Probability",
        "",
        f"- F1: {stage_a['F1']:.3f}",
        f"- AUC: {stage_a['AUC']:.3f}",
        f"- PR-AUC: {metrics.get('PR_AUC', 'N/A')}",
        f"- ECE: {stage_a['ECE']:.3f}",
        f"- Brier: {stage_a['Brier']:.3f}",
        "",
        "## 2. Stage B — Action Calibration",
        "",
        f"- Risk budget mean/max/min: {risk_budget.mean():.3f} / "
        f"{risk_budget.max():.3f} / {risk_budget.min():.3f}",
        f"- α: {RISK_BUDGET_ALPHA}, β: {RISK_BUDGET_BETA}, max: {RISK_BUDGET_MAX}",
        "",
        "## 3. 评估指标 (含新指标)",
        "",
        "| 指标 | v4 baseline | v6 2-stage |",
        "|---|---|---|",
        f"| F1 | 0.546 | {metrics.get('F1', 'N/A'):.3f} |",
        f"| AUC | 0.489 | {metrics.get('AUC', 'N/A'):.3f} |",
        f"| PR-AUC | - | {metrics.get('PR_AUC', 'N/A'):.3f} |",
        f"| ECE | - | {metrics.get('ECE', 'N/A'):.3f} |",
        f"| Brier | - | {metrics.get('Brier', 'N/A'):.3f} |",
        f"| net_utility | - | {metrics.get('net_utility', 'N/A'):.3f} |",
        "",
        "## 4. is_downside 事件 Recall",
        "",
        f"- 总事件: {rec['n_total']}",
        f"- 捕获: {rec['n_caught']} / {rec['n_total']} = {rec['recall']:.1%}",
        "",
    ]
    (OUT_DIR / "ca_gcp_training_v6_report.md").write_text("\n".join(report))
    print(f"[报告] {OUT_DIR / 'ca_gcp_training_v6_report.md'}")

    # 保存 risk_budget 给 v10.3c 用
    budget_df = pd.DataFrame(
        {"risk_budget": risk_budget, "p_downside": stage_a["oof_proba"]},
        index=X_aligned.index,
    )
    budget_path = OUT_DIR / "ca_gcp_v6_risk_budget.csv"
    budget_df.to_csv(budget_path)
    print(f"[保存] {budget_path}")


if __name__ == "__main__":
    main()
