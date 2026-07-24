# coding=utf-8
"""测试所有可用宏观因子集, 寻找最佳 Layer 2 信号.

宏观因子候选:
    1. 5 ETF (Phase A 用了, baseline)
    2. 8 v9 macro (level: 水平值滚动 zscore)
    3. 8 v9 macro (flow: 周收益率)
    4. 13 混合 (5 ETF + 8 v9)
    5. 9 (5 ETF + 4 extra: VIX/DXY/real_rate/spread)
    6. 17 ALL (5 ETF + 8 v9 + 4 extra)

参数: 用 Phase C 最佳 zwin=4, coef=0.8, cost=5bp
"""
import sys, time, pickle
from pathlib import Path
from datetime import datetime
import importlib.util

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_five_macro_factors,
    compute_v9_macro_factors,
    compute_extra_macro_factors,
    compute_factor_score_from_macro,
    compute_risk_scalar,
)
SCRIPT = Path('scripts/combo/regenerate_v8_dynamic_position.py').resolve()
spec = importlib.util.spec_from_file_location(
    'regen_dyn', SCRIPT
)
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
    log("扩展宏观因子集测试 - 6 种组合")
    log("=" * 70)

    daily_returns = pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")
    v9_weekly = pd.read_parquet(HF_DIR / "v9_factors_weekly.parquet")

    # 额外宏观数据
    vix_d = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")['vix']
    dxy_d = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")['dxy']
    real_d = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")['real_rate']
    spread_df = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
    spread_d = pd.Series(spread_df['cn_us_spread'].values, index=pd.to_datetime(spread_df['date']))

    weekly_weights, _, _ = load_v7_14_portfolio()
    with open(SIGNAL_PKL, 'rb') as f:
        signals = pickle.load(f)

    # 用 Phase C 最佳参数: zwin=4, coef=0.8
    zwin = 4
    coef = 0.8
    cost_bp = 5
    clip_low, clip_high = 0.3, 1.5

    log(f"参数: zwin={zwin}, coef={coef}, cost={cost_bp}bp, clip=[{clip_low},{clip_high}]")

    # 6 因子集
    factor_sets = {}

    # 1. 5 ETF only (Phase A baseline)
    log("\n[1/6] 5 ETF only ...")
    t0 = time.time()
    five = compute_five_macro_factors(daily_returns, zscore_window=zwin)
    fs1 = compute_factor_score_from_macro(five)
    rs1 = compute_risk_scalar(fs1, coef=coef)
    log(f"  shape={five.shape}, factor_score: {len(fs1)}, risk_scalar: {len(rs1)} ({time.time()-t0:.1f}s)")
    factor_sets['1_5_ETF_only'] = rs1

    # 2. 8 v9 macro level
    log("\n[2/6] 8 v9 macro level ...")
    t0 = time.time()
    v9_level = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=False)
    fs2 = compute_factor_score_from_macro(v9_level)
    rs2 = compute_risk_scalar(fs2, coef=coef)
    log(f"  shape={v9_level.shape}, factor_score: {len(fs2)}, risk_scalar: {len(rs2)} ({time.time()-t0:.1f}s)")
    factor_sets['2_v9_macro_LEVEL'] = rs2

    # 3. 8 v9 macro flow
    log("\n[3/6] 8 v9 macro flow (weekly returns) ...")
    t0 = time.time()
    v9_flow = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=True)
    fs3 = compute_factor_score_from_macro(v9_flow)
    rs3 = compute_risk_scalar(fs3, coef=coef)
    log(f"  shape={v9_flow.shape}, factor_score: {len(fs3)}, risk_scalar: {len(rs3)} ({time.time()-t0:.1f}s)")
    factor_sets['3_v9_macro_FLOW'] = rs3

    # 4. 13 混合 (5 ETF + 8 v9 flow)
    log("\n[4/6] 13 混合 (5 ETF + 8 v9 flow) ...")
    t0 = time.time()
    mixed_13 = pd.concat([five, v9_flow], axis=1).dropna()
    fs4 = compute_factor_score_from_macro(mixed_13)
    rs4 = compute_risk_scalar(fs4, coef=coef)
    log(f"  shape={mixed_13.shape}, factor_score: {len(fs4)}, risk_scalar: {len(rs4)} ({time.time()-t0:.1f}s)")
    factor_sets['4_5ETF_v9_flow'] = rs4

    # 5. 9 (5 ETF + 4 extra: VIX/DXY/real_rate/spread)
    log("\n[5/6] 9 (5 ETF + 4 extra) ...")
    t0 = time.time()
    extra = compute_extra_macro_factors(vix_d, dxy_d, real_d, spread_d, zscore_window=zwin)
    combined_9 = pd.concat([five, extra], axis=1).dropna()
    fs5 = compute_factor_score_from_macro(combined_9)
    rs5 = compute_risk_scalar(fs5, coef=coef)
    log(f"  shape={combined_9.shape}, factor_score: {len(fs5)}, risk_scalar: {len(rs5)} ({time.time()-t0:.1f}s)")
    factor_sets['5_5ETF_4EXTRA'] = rs5

    # 6. ALL = 17
    log("\n[6/6] ALL = 17 (5 ETF + 8 v9 flow + 4 extra) ...")
    t0 = time.time()
    all_17 = pd.concat([five, v9_flow, extra], axis=1).dropna()
    fs6 = compute_factor_score_from_macro(all_17)
    rs6 = compute_risk_scalar(fs6, coef=coef)
    log(f"  shape={all_17.shape}, factor_score: {len(fs6)}, risk_scalar: {len(rs6)} ({time.time()-t0:.1f}s)")
    factor_sets['6_ALL_17'] = rs6

    # 验证 v9 数据可读到 2021-08 (OOS start)
    log(f"\n[校验] 6 组 rs 覆盖率 (vs OOS_START={OOS_START.date()}):")
    for name, rs in factor_sets.items():
        oos_rs = rs.loc[OOS_START:]
        oos_n = oos_rs.notna().sum()
        early_n = rs.loc[:OOS_START].notna().sum()
        log(f"  {name:25s} | 预 OOS: {early_n:>3d} | OOS: {oos_n:>3d} | rs range [{rs.min():.2f}, {rs.max():.2f}]")

    # 跑 6 组 NAV
    log("\n" + "=" * 70)
    log("跑 6 组 NAV (zwin=4, coef=0.8, cost=5bp)")
    log("=" * 70)

    results = []
    for name, rs in factor_sets.items():
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

        # 924 期间 risk_scalar
        rs_924 = rs.loc['2024-09-23':'2024-10-11']
        rs_924_mean = rs_924.mean() if len(rs_924) > 0 else float('nan')
        rs_924_min = rs_924.min() if len(rs_924) > 0 else float('nan')
        rs_924_max = rs_924.max() if len(rs_924) > 0 else float('nan')

        log(f"  {name:25s} | Sharpe={sharpe:.3f} Calmar={calmar:.3f} AnnRet={ann_ret:.2%} MaxDD={max_dd:.2%}")
        log(f"    rs overall range [{rs.min():.2f}, {rs.max():.2f}] | 924 period range [{rs_924_min:.2f}, {rs_924_max:.2f}] mean={rs_924_mean:.2f}")
        log(f"    ({elapsed:.1f}s)")

        results.append({
            'factor_set': name,
            'n_factors': len(factor_sets) - 0,  # placeholder
            'Sharpe': sharpe,
            'Calmar': calmar,
            'AnnRet': float(ann_ret),
            'MaxDD': max_dd,
            'rs_min': float(rs.min()),
            'rs_max': float(rs.max()),
            'rs_924_mean': float(rs_924_mean) if pd.notna(rs_924_mean) else 0,
            'rs_924_range': f'[{rs_924_min:.2f}, {rs_924_max:.2f}]' if pd.notna(rs_924_min) else 'N/A',
        })

        out_path = OUT_DIR / f"v8_dyn_{name}.parquet"
        nav.to_frame('nav').to_parquet(out_path)

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_extended_factors_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("✅ 完成")
    log(f"对比表: {csv_path}")
    log("=" * 70)
    log("\n=== 6 因子集 Sharpe 排序 ===")
    log(df.sort_values('Sharpe', ascending=False).to_string(index=False))

    log("\n=== baseline ===")
    log(f"  v7.10 TV-PR 5bp:                Sharpe=0.922")
    log(f"  v8 per-asset 5bp:                Sharpe=0.871")
    log(f"  v8 + dynamic 5 ETF (zwin=4 coef=0.8): Sharpe={df.iloc[0]['Sharpe']:.3f}")


if __name__ == "__main__":
    main()
