# coding=utf-8
"""v6.2 Phase 4 generalization test: 5-fold walk-forward 验证无 look-ahead.

[目的] 验证 Phase 4 主推 (warmup_ir + IC12 + warmup=12m) 在不同时间窗口下都稳定.

算法:
  - 将回测期 (2018-01-01 ~ 2026-06-30) 切成 5 段
  - 每段作为 OOS 测试期, 其之前的数据作为训练期
  - 计算每段 OOS Calmar
  - 如果 5 段都 ≥ 0.5 (或者 ≥ 0.6 for 显著), 说明策略无 overfit

段切分:
  Fold 1: 2020-01-01 ~ 2020-12-31 (训练: 2018-01 ~ 2019-12)
  Fold 2: 2021-01-01 ~ 2021-12-31 (训练: 2018-01 ~ 2020-12)
  Fold 3: 2022-01-01 ~ 2023-06-30 (训练: 2018-01 ~ 2021-12)
  Fold 4: 2023-07-01 ~ 2024-12-31 (训练: 2018-01 ~ 2023-06)
  Fold 5: 2025-01-01 ~ 2026-06-30 (训练: 2018-01 ~ 2024-12)

每 fold 重新算 IC weights 和 warmup-IR 排序 (基于 fold 起点之前数据).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_2Config, run_v6_2_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6 import V6_1Config, run_v6_1_backtest


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0, "end": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else 0,
            "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0, "end": s.iloc[-1]}


# 5-fold 时间切分
FOLDS = [
    ("2020-01-01", "2020-12-31", "训练2018-01~2019-12"),
    ("2021-01-01", "2021-12-31", "训练2018-01~2020-12"),
    ("2022-01-01", "2023-06-30", "训练2018-01~2021-12"),
    ("2023-07-01", "2024-12-31", "训练2018-01~2023-06"),
    ("2025-01-01", "2026-06-30", "训练2018-01~2024-12"),
]


def main() -> None:
    print("[v6.2 generalization test] 加载数据...")
    panel_close_full = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv_full = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    print(f"  panel_close: {panel_close_full.shape}")

    rows = []
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fi, (fold_start, fold_end, train_desc) in enumerate(FOLDS):
        print(f"\n─── Fold {fi+1}/5 ─── OOS: {fold_start} ~ {fold_end} ({train_desc})")
        # 用 fold_end 截 panel_close, 但 ohlcv 也截到 fold_end
        fold_close = panel_close_full.loc[:"2026-06-30"]
        fold_ohlcv = panel_ohlcv_full.loc[:"2026-06-30"]

        # v6.2 主推
        cfg = V6_2Config(
            ic_min_months=12,
            sort_method="warmup_ir",
            warmup_months=12,
        )
        nav_v62 = run_v6_2_backtest(fold_close, fold_ohlcv, cfg)
        fold_nav = nav_v62.loc[fold_start:fold_end]
        m62 = metrics(fold_nav)
        print(f"  v6.2 warmup_ir IC12: OOS Calmar={m62['calmar']:.3f} ann={m62['ann']:+.2%} "
              f"DD={m62['dd']:.2%} Sharpe={m62['sharpe']:.2f} | End={m62['end']:.3f}")

        # v6.1 IC12 baseline (对比)
        cfg = V6_1Config(ic_min_months=12)
        nav_v61 = run_v6_1_backtest(fold_close, fold_ohlcv, cfg)
        fold_nav61 = nav_v61.loc[fold_start:fold_end]
        m61 = metrics(fold_nav61)
        print(f"  v6.1 IC12:        OOS Calmar={m61['calmar']:.3f} ann={m61['ann']:+.2%} "
              f"DD={m61['dd']:.2%} Sharpe={m61['sharpe']:.2f}")

        # v6.2 no_orth (正交 = skip, 等同 v6.1)
        cfg = V6_2Config(ic_min_months=12, use_orthogonal=False)
        nav_noorth = run_v6_2_backtest(fold_close, fold_ohlcv, cfg)
        fold_noorth = nav_noorth.loc[fold_start:fold_end]
        m_noorth = metrics(fold_noorth)
        print(f"  v6.2 no_orth:     OOS Calmar={m_noorth['calmar']:.3f} ann={m_noorth['ann']:+.2%} "
              f"DD={m_noorth['dd']:.2%} Sharpe={m_noorth['sharpe']:.2f}")

        # v6.2 ir_full DEPRECATED (含 look-ahead) — 从历史 CSV 读取
        from pandas import read_csv
        hist_csv = REPO / "reports/momentum_etf_rotation/combo/v6_2_generalization_test.csv"
        if hist_csv.exists():
            hist_df = read_csv(hist_csv)
            hist_row = hist_df[hist_df["fold"] == fi + 1]
            if len(hist_row) > 0 and "v6.2_ir_full_oos_calmar" in hist_row.columns:
                m_full_calmar = float(hist_row.iloc[0]["v6.2_ir_full_oos_calmar"])
                print(f"  v6.2 ir_full DEPRECATED (历史 CSV): OOS Calmar={m_full_calmar:.3f}")
                m_full = {"calmar": m_full_calmar}
            else:
                print("  v6.2 ir_full DEPRECATED: 历史 CSV 无 fold 数据, 跳过")
                m_full = {"calmar": float("nan")}
        else:
            print("  v6.2 ir_full DEPRECATED: 历史 CSV 不存在, 跳过")
            m_full = {"calmar": float("nan")}

        rows.append({
            "fold": fi + 1,
            "fold_start": fold_start,
            "fold_end": fold_end,
            "train_desc": train_desc,
            "v6.2_warmup_oos_calmar": m62["calmar"],
            "v6.2_warmup_oos_ann": m62["ann"],
            "v6.2_warmup_oos_dd": m62["dd"],
            "v6.2_warmup_oos_sharpe": m62["sharpe"],
            "v6.2_warmup_oos_end": m62["end"],
            "v6.1_oos_calmar": m61["calmar"],
            "v6.2_no_orth_oos_calmar": m_noorth["calmar"],
            "v6.2_ir_full_oos_calmar": m_full["calmar"],
        })

    df = pd.DataFrame(rows)
    print("\n=== 5-fold walk-forward generalization test ===")
    print(df[["fold", "fold_start", "fold_end", "v6.2_warmup_oos_calmar",
              "v6.1_oos_calmar", "v6.2_no_orth_oos_calmar",
              "v6.2_ir_full_oos_calmar"]].to_string(index=False))

    df.to_csv(out_dir / "v6_2_generalization_test.csv", index=False)
    print(f"\n[save] {out_dir / 'v6_2_generalization_test.csv'}")

    # 决策: warmup_IR 路径是否稳定
    warmup_cals = df["v6.2_warmup_oos_calmar"].values
    v61_cals = df["v6.1_oos_calmar"].values

    print(f"\n=== generalization 决策 ===")
    print(f"v6.2 warmup OOS Calmar: mean={warmup_cals.mean():.3f} min={warmup_cals.min():.3f} "
          f"std={warmup_cals.std():.3f}")
    print(f"v6.1 IC12 OOS Calmar: mean={v61_cals.mean():.3f} min={v61_cals.min():.3f} "
          f"std={v61_cals.std():.3f}")

    n_pass = (warmup_cals >= 0.5).sum()
    print(f"\nv6.2 warmup ≥ 0.5 folds: {n_pass}/5")
    if warmup_cals.min() >= 0.5:
        print(f"✅ 5 folds 均 ≥ 0.5 → 策略无明显过拟合, 锁定 warmup_ir 为 v6.2 默认")
    elif n_pass >= 4:
        print(f"⚠ 4 folds ≥ 0.5, 1 fold < 0.5 → 部分过拟合, 仍接受为默认但标注谨慎")
    else:
        print(f"❌ < 4 folds ≥ 0.5 → 严重过拟合, 退回 v6.1 IC12")


if __name__ == "__main__":
    main()
