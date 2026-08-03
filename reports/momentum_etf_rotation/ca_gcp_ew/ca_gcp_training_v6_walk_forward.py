"""CA-GCP v6 Nested Walk-Forward (Phase 1.6).

Purge/embargo 防止标签泄露, 评估 v6 在 OOS 上的稳定性.

设计:
  outer fold: 4 年 train, 1 年 test, 1 年 step
  inner: 用 train 数据做 TimeSeriesSplit 验证 (Stage A 训练)
  purge: 在 train/test 边界前后剔除 5 天
  embargo: 在 train/test 边界前 5 天不再用于训练 (purged)
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

from ca_gcp_training_v3 import (  # noqa: E402
    build_full_features,
    build_y_labels,
    load_hs300_returns,
)
from ca_gcp_training_v4 import train_with_strong_regularization  # noqa: E402
from ca_gcp_ew_eval import OUT_DIR, load_returns  # noqa: E402
from ca_gcp_training_v6 import (  # noqa: E402
    evaluate_v6,
    is_downside_event_recall,
    stage_b_action_calibration,
)

PURGE_DAYS = 5
EMBARGO_DAYS = 5
TRAIN_YEARS = 4
TEST_YEARS = 1
STEP_YEARS = 1
N_TRIALS = 5  # 快速模式


def make_folds(start: str, end: str, train_years: int = TRAIN_YEARS,
               test_years: int = TEST_YEARS, step_years: int = STEP_YEARS):
    """生成 walk-forward folds."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    folds = []
    fold_start = start
    fold_id = 0
    while True:
        train_start = fold_start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append({
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "label": f"F{fold_id:02d} {train_start.year}-{test_end.year}",
        })
        fold_start = fold_start + pd.DateOffset(years=step_years)
        fold_id += 1
    return folds


def nested_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    market_ret: pd.Series,
    folds: list,
    purge: int = PURGE_DAYS,
    embargo: int = EMBARGO_DAYS,
    n_trials: int = N_TRIALS,
) -> pd.DataFrame:
    """Nested walk-forward 评估 v6.

    每个 fold:
      1. train: train_start+purge 到 train_end-embargo
      2. test: test_start+purge 到 test_end (只评估, 不调参)
      3. inner: train 数据 TimeSeriesSplit 4 折
    """
    rows = []
    for f in folds:
        print(f"\n[fold {f['fold_id']}] {f['label']}")

        # Apply purge/embargo
        train_mask = (X.index >= f["train_start"] + pd.Timedelta(days=purge)) & \
                     (X.index <= f["train_end"] - pd.Timedelta(days=embargo))
        test_mask = (X.index >= f["test_start"] + pd.Timedelta(days=purge)) & \
                    (X.index <= f["test_end"])

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            print(f"  跳过: train={len(X_train)}, test={len(X_test)}")
            continue

        print(f"  train: {len(X_train)} 天, test: {len(X_test)} 天")

        try:
            res = train_with_strong_regularization(X_train, y_train, "xgb", n_trials=n_trials)
        except Exception as e:
            print(f"  训练失败: {e}")
            continue

        # Predict on test
        final_model = res["final_model"]
        from xgboost import DMatrix
        dtest = DMatrix(X_test.values)
        proba_test = final_model.predict(dtest)

        # Risk budget
        risk_test = stage_b_action_calibration(
            proba_test, np.full(len(X_test), 30.0)
        )
        loss_test = market_ret.loc[X_test.index].fillna(0).values

        m = evaluate_v6(y_test.values, proba_test, risk_test, loss_test)

        # is_downside event recall
        fired_idx = [d for d, p in zip(X_test.index, proba_test) if p > 0.5]
        rec = is_downside_event_recall(
            pd.Series(proba_test, index=X_test.index), fired_idx
        )
        m.update({
            "fold_id": f["fold_id"],
            "fold_label": f["label"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "is_downside_recall": rec["recall"],
            "is_downside_caught": rec["n_caught"],
            "is_downside_total": rec["n_total"],
        })
        rows.append(m)
        print(f"  F1={m['F1']:.3f}, AUC={m['AUC']:.3f}, "
              f"PR-AUC={m['PR_AUC']:.3f}, ECE={m['ECE']:.3f}")
        print(f"  is_downside recall: {rec['n_caught']}/{rec['n_total']} = {rec['recall']:.1%}")

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("CA-GCP v6 Nested Walk-Forward (Phase 1.6)")
    print("=" * 70)
    print(f"  Purge: {PURGE_DAYS}d, Embargo: {EMBARGO_DAYS}d")
    print(f"  Train: {TRAIN_YEARS}y, Test: {TEST_YEARS}y, Step: {STEP_YEARS}y")

    returns = load_returns()
    features = build_full_features(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common = features.index.intersection(y_labels.index)
    X = features.loc[common]
    Y = y_labels.loc[common]
    market_ret = returns.mean(axis=1)

    folds = make_folds("2018-01-01", "2026-05-06")
    print(f"\n  共 {len(folds)} 个 fold: {[f['label'] for f in folds]}")

    df = nested_walk_forward(X, Y["y_E"], market_ret, folds)

    if len(df) == 0:
        print("\n[无有效 fold]")
        return

    print("\n" + "=" * 70)
    print("Walk-Forward 汇总")
    print("=" * 70)
    for col in ["F1", "AUC", "PR_AUC", "ECE", "Brier", "net_utility", "is_downside_recall"]:
        print(f"  {col}: mean={df[col].mean():.3f}, "
              f"std={df[col].std():.3f}, "
              f"min={df[col].min():.3f}, max={df[col].max():.3f}")

    out_path = OUT_DIR / "ca_gcp_v6_walk_forward_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[保存] {out_path}")

    # 汇总报告
    report = [
        "# CA-GCP v6 Nested Walk-Forward 结果",
        "",
        f"**配置**: purge={PURGE_DAYS}d, embargo={EMBARGO_DAYS}d, "
        f"train={TRAIN_YEARS}y, test={TEST_YEARS}y, step={STEP_YEARS}y",
        "",
        f"**fold 数**: {len(df)}",
        "",
        "## 1. 各 Fold 详情",
        "",
        "| Fold | Train | Test | F1 | AUC | PR-AUC | ECE | Brier | is_downside Recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        report.append(
            f"| {r['fold_label']} | {int(r['n_train'])} | {int(r['n_test'])} | "
            f"{r['F1']:.3f} | {r['AUC']:.3f} | {r['PR_AUC']:.3f} | "
            f"{r['ECE']:.3f} | {r['Brier']:.3f} | {r['is_downside_recall']:.1%} |"
        )
    report += [
        "",
        "## 2. 汇总",
        "",
        f"- **F1**: {df['F1'].mean():.3f} ± {df['F1'].std():.3f}",
        f"- **AUC**: {df['AUC'].mean():.3f} ± {df['AUC'].std():.3f}",
        f"- **PR-AUC**: {df['PR_AUC'].mean():.3f} ± {df['PR_AUC'].std():.3f}",
        f"- **ECE**: {df['ECE'].mean():.3f} ± {df['ECE'].std():.3f}",
        f"- **Brier**: {df['Brier'].mean():.3f} ± {df['Brier'].std():.3f}",
        f"- **is_downside Recall**: {df['is_downside_recall'].mean():.1%} ± {df['is_downside_recall'].std():.1%}",
    ]
    (OUT_DIR / "ca_gcp_v6_walk_forward_report.md").write_text("\n".join(report))
    print(f"[报告] {OUT_DIR / 'ca_gcp_v6_walk_forward_report.md'}")


if __name__ == "__main__":
    main()
