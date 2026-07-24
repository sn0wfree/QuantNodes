# coding=utf-8
"""v9 macro LEVEL 因子集的参数网格 (Phase D).

找到最佳 zwin × coef × clip 组合.
"""
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
    log("Phase D: v9 macro LEVEL 参数网格")
    log("=" * 70)

    v9_weekly = pd.read_parquet(HF_DIR / "v9_factors_weekly.parquet")
    daily_returns = pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")
    weekly_weights, _, _ = load_v7_14_portfolio()
    with open(SIGNAL_PKL, 'rb') as f:
        signals = pickle.load(f)

    zscore_windows = [4, 8, 13, 26, 52]
    coefs = [0.3, 0.5, 0.8, 1.0, 1.5]
    clip_profiles = [('激进', 0.1, 2.0), ('标准', 0.3, 1.5), ('保守', 0.5, 1.2)]
    cost_bp = 5

    n = len(zscore_windows) * len(coefs) * len(clip_profiles)
    log(f"参数空间: {len(zscore_windows)} × {len(coefs)} × {len(clip_profiles)} = {n} 组合")
    log(f"成本: {cost_bp}bp")

    results = []
    t0_all = time.time()

    for zwin in zscore_windows:
        # 重算 v9 macro LEVEL with new zwin
        factors = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=False)
        fs = compute_factor_score_from_macro(factors)

        for coef in coefs:
            for clip_name, clip_low, clip_high in clip_profiles:
                t0 = time.time()
                rs = compute_risk_scalar(fs, coef=coef, clip_low=clip_low, clip_high=clip_high)

                nav = compute_nav_two_layer(
                    weekly_weights, daily_returns, signals, rs,
                    cost_bp=cost_bp,
                    clip_low=clip_low, clip_high=clip_high,
                )

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

                elapsed = time.time() - t0
                log(f"  zwin={zwin:>3d} coef={coef:.1f} clip=[{clip_low:.1f},{clip_high:.1f}] ({clip_name:>4s}) | "
                    f"Sharpe={sharpe:.3f} Calmar={calmar:.3f} MaxDD={max_dd:.2%} ({elapsed:.1f}s)")

                results.append({
                    'zscore_window': zwin,
                    'coef': coef,
                    'clip_name': clip_name,
                    'clip_low': clip_low,
                    'clip_high': clip_high,
                    'Sharpe': float(sharpe),
                    'Calmar': float(calmar),
                    'AnnRet': float(ann_ret),
                    'MaxDD': max_dd,
                })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v9_macro_level_grid.csv"
    df.to_csv(csv_path, index=False)

    elapsed_all = time.time() - t0_all
    log(f"\n✅ 完成 ({elapsed_all:.1f}s, {n} 组合)")
    log(f"对比表: {csv_path}")

    log("\n=== Top 15 最佳组合 (按 Sharpe) ===")
    log(df.sort_values('Sharpe', ascending=False).head(15).to_string(index=False))

    log("\n=== Sharpe 矩阵 (zwin × coef, 标准 clip) ===")
    pivot = df[df['clip_name'] == '标准'].pivot_table(
        index='zscore_window', columns='coef', values='Sharpe'
    )
    log(pivot.to_string())

    log("\n=== MaxDD 矩阵 (zwin × coef, 标准 clip) ===")
    pivot_dd = df[df['clip_name'] == '标准'].pivot_table(
        index='zscore_window', columns='coef', values='MaxDD'
    )
    log(pivot_dd.to_string())

    # 综合 baseline 对比
    log("\n=== 综合 baseline 对比 ===")
    log(f"  v7.10 TV-PR 5bp:                        Sharpe=0.922 (MaxDD -20.54%)")
    log(f"  v8 per-asset 5bp:                       Sharpe=0.871 (MaxDD -18.14%)")
    best = df.loc[df['Sharpe'].idxmax()]
    log(f"  v8 + v9 macro LEVEL (best {best['zscore_window']:.0f}w {best['coef']:.1f}c): Sharpe={best['Sharpe']:.3f} (MaxDD {best['MaxDD']:.2%})")


if __name__ == "__main__":
    main()
