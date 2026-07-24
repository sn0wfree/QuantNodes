# coding=utf-8
"""Phase E: 最佳组合 (zwin=4, coef=1.5, 保守 clip) × 4 成本档验证."""
import sys, time, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_v9_macro_factors,
    compute_factor_score_from_macro,
    compute_risk_scalar,
)
import importlib.util
SCRIPT = Path('scripts/combo/regenerate_v8_dynamic_position.py').resolve()
spec = importlib.util.spec_from_file_location('regen_dyn', SCRIPT)
regen_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regen_mod)
compute_nav_two_layer = regen_mod.compute_nav_two_layer

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
HF_DIR = REPO / "data" / "high_freq_macro"
SIGNAL_PKL = Path(__file__).resolve().parent / "signals_prob.pkl"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase E: 最佳组合 × 4 成本档验证")
    log("=" * 70)

    v9_weekly = pd.read_parquet(HF_DIR / "v9_factors_weekly.parquet")
    daily_returns = pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")
    weekly_weights, _, _ = load_v7_14_portfolio()
    with open(SIGNAL_PKL, 'rb') as f:
        signals = pickle.load(f)

    # 最佳: zwin=4, coef=1.5, 保守 clip [0.5, 1.2]
    zwin = 4
    coef = 1.5
    clip_low, clip_high = 0.5, 1.2
    costs = [5, 10, 15, 20]

    log(f"参数: zwin={zwin}, coef={coef}, clip=[{clip_low},{clip_high}]")
    log(f"成本档: {costs}")

    factors = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=False)
    fs = compute_factor_score_from_macro(factors)
    rs = compute_risk_scalar(fs, coef=coef, clip_low=clip_low, clip_high=clip_high)
    log(f"  fs shape: {fs.shape}, rs shape: {rs.shape}, range: [{rs.min():.3f}, {rs.max():.3f}]")

    results = []
    for cost_bp in costs:
        t0 = time.time()
        nav = compute_nav_two_layer(
            weekly_weights, daily_returns, signals, rs,
            cost_bp=cost_bp,
            clip_low=clip_low, clip_high=clip_high,
        )
        elapsed = time.time() - t0

        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()

        total = oos.iloc[-1] / oos.iloc[0] - 1
        n_years = len(rets) / 252
        ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0.0
        peak = oos.cummax()
        max_dd = float((oos / peak - 1).min())
        calmar = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0

        log(f"\n  cost={cost_bp:>2d}bp | Sharpe={sharpe:.3f} Calmar={calmar:.3f} "
            f"AnnRet={ann_ret:.2%} MaxDD={max_dd:.2%} ({elapsed:.1f}s)")

        results.append({
            'config': f'zwin={zwin}_coef={coef}_clip=[{clip_low},{clip_high}]',
            'cost_bp': cost_bp,
            'Sharpe': float(sharpe),
            'Calmar': float(calmar),
            'AnnRet': float(ann_ret),
            'MaxDD': max_dd,
        })

        out_path = OUT_DIR / f"v9_macro_best_C{cost_bp}.parquet"
        nav.to_frame('nav').to_parquet(out_path)

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v9_macro_best_costs.csv"
    df.to_csv(csv_path, index=False)

    log(f"\n✅ 完成")
    log(f"对比表: {csv_path}")
    log(df.to_string(index=False))

    log("\n=== 最终综合对比 ===")
    log("| 策略 | Sharpe | Calmar | MaxDD | AnnRet |")
    log("|------|--------|--------|-------|--------|")
    log("| v7.10 TV-PR 5bp       | 0.922 | 0.871 | -20.54% | 17.89% |")
    log("| v8 per-asset 5bp      | 0.871 | 0.739 | -18.14% | 12.98% |")
    for _, row in df.iterrows():
        log(f"| **NEW** {row['config']} cost={row['cost_bp']}bp | **{row['Sharpe']:.3f}** | {row['Calmar']:.3f} | {row['MaxDD']:.2%} | {row['AnnRet']:.2%} |")


if __name__ == "__main__":
    main()
