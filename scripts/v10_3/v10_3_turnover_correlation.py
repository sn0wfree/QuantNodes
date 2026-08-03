"""Phase 2 追加验证: turnover + 相关性 + bootstrap p-value.

1. Turnover: max turnover, avg turnover, rebalance count
2. Correlation: G returns vs v1.0 returns
3. Bootstrap p-value: G vs D Sharpe diff
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (  # noqa: E402
    scheme_g_trend_adaptive,
    scheme_d_signal_weighted,
    scheme_f_conservative,
    load_navs,
    compute_nav,
    BASE_WEIGHTS,
)

OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-06-30")
N_BOOT = 5000


def turnover_analysis(prices: pd.DataFrame, scheme_fn, name: str):
    """计算 scheme 的换手率统计."""
    weights = scheme_fn(prices)
    diffs = weights.diff().abs().sum(axis=1)
    # 月末调仓日 (权重变化 > 0.01)
    rebal_mask = diffs > 0.01
    n_rebal = rebal_mask.sum()
    turnover_series = diffs[rebal_mask]

    # OOS 期间
    oos = turnover_series.loc[OOS_START:OOS_END]
    all_turn = turnover_series

    return {
        "name": name,
        "n_rebal_total": int(n_rebal),
        "n_rebal_oos": int(len(oos)),
        "avg_turnover_total": float(all_turn.mean()) if len(all_turn) > 0 else 0,
        "max_turnover_total": float(all_turn.max()) if len(all_turn) > 0 else 0,
        "avg_turnover_oos": float(oos.mean()) if len(oos) > 0 else 0,
        "max_turnover_oos": float(oos.max()) if len(oos) > 0 else 0,
        "p95_turnover_oos": float(oos.quantile(0.95)) if len(oos) > 0 else 0,
    }


def correlation_analysis(prices: pd.DataFrame, scheme_fn, name: str):
    """计算 scheme 日收益 vs v1.0 日收益的相关性."""
    nav = compute_nav(prices, scheme_fn(prices))
    rets = nav.pct_change().dropna()

    # v1.0 日收益
    v1_rets = prices["v1.0"].pct_change().dropna()
    common = rets.index.intersection(v1_rets.index)
    corr = rets.loc[common].corr(v1_rets.loc[common])

    # v7.10
    v7_rets = prices["v7.10"].pct_change().dropna()
    common2 = rets.index.intersection(v7_rets.index)
    corr_v7 = rets.loc[common2].corr(v7_rets.loc[common2])

    # DualMom
    dm_rets = prices["DualMom"].pct_change().dropna()
    common3 = rets.index.intersection(dm_rets.index)
    corr_dm = rets.loc[common3].corr(dm_rets.loc[common3])

    return {
        "name": name,
        "corr_v1.0": float(corr),
        "corr_v7.10": float(corr_v7),
        "corr_DualMom": float(corr_dm),
    }


def bootstrap_sharpe_test(
    prices: pd.DataFrame,
    scheme_a_fn,
    scheme_b_fn,
    n_boot: int = N_BOOT,
    block_size: int = 21,
    seed: int = 42,
) -> dict:
    """Block bootstrap Sharpe diff 检验 (centered at H0).

    H0: Sharpe_A - Sharpe_B = 0
    方法: 中心化 bootstrap — 从各策略日收益中减去均值,
          重新采样 block, 计算 Sharpe diff (此时 H0 下期望为 0).
    p-value = P(bootstrap_diff >= observed_diff | H0)
    """
    nav_a = compute_nav(prices, scheme_a_fn(prices))
    nav_b = compute_nav(prices, scheme_b_fn(prices))

    common = nav_a.index.intersection(nav_b.index)
    oos_a = nav_a.loc[OOS_START:OOS_END].pct_change().dropna()
    oos_b = nav_b.loc[OOS_START:OOS_END].pct_change().dropna()
    common_oos = oos_a.index.intersection(oos_b.index)
    oos_a = oos_a.loc[common_oos]
    oos_b = oos_b.loc[common_oos]

    # Observed Sharpe diff
    obs_sharpe_a = oos_a.mean() / oos_a.std() * np.sqrt(252) if oos_a.std() > 0 else 0
    obs_sharpe_b = oos_b.mean() / oos_b.std() * np.sqrt(252) if oos_b.std() > 0 else 0
    obs_diff = obs_sharpe_a - obs_sharpe_b

    # Centered bootstrap (H0: mean=0)
    # 从每个策略的日收益中减去均值, 使 H0 成立
    centered_a = oos_a - oos_a.mean()
    centered_b = oos_b - oos_b.mean()

    rng = np.random.default_rng(seed)
    n = len(oos_a)
    n_blocks = n // block_size
    diffs = []

    for _ in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot_a, boot_b = [], []
        for bs in block_starts:
            boot_a.extend(centered_a.iloc[bs:bs + block_size].values)
            boot_b.extend(centered_b.iloc[bs:bs + block_size].values)
        boot_a = np.array(boot_a[:n])
        boot_b = np.array(boot_b[:n])
        # 用原始 std (因为中心化不改变方差)
        sa = boot_a.mean() / oos_a.std() * np.sqrt(252) if oos_a.std() > 0 else 0
        sb = boot_b.mean() / oos_b.std() * np.sqrt(252) if oos_b.std() > 0 else 0
        diffs.append(sa - sb)

    diffs = np.array(diffs)
    # 两尾检验: p-value = P(|diff| >= |observed|)
    p_value_two = float((np.abs(diffs) >= np.abs(obs_diff)).mean())
    # 单尾检验 (G > D): p-value = P(diff >= observed)
    p_value_one = float((diffs >= obs_diff).mean())

    return {
        "observed_sharpe_a": float(obs_sharpe_a),
        "observed_sharpe_b": float(obs_sharpe_b),
        "observed_diff": float(obs_diff),
        "bootstrap_mean_diff": float(diffs.mean()),
        "bootstrap_std_diff": float(diffs.std()),
        "ci_low_2.5": float(np.percentile(diffs, 2.5)),
        "ci_high_97.5": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": float(p_value_one),
        "p_value_two_sided": float(p_value_two),
    }


def main():
    prices = load_navs()
    print(f"Prices: {prices.index[0].date()} ~ {prices.index[-1].date()}, {len(prices)} days")

    # === 1. Turnover Analysis ===
    print("\n" + "=" * 70)
    print("1. Turnover Analysis (换手率分析)")
    print("=" * 70)

    schemes = {
        "D (signal-weighted)": scheme_d_signal_weighted,
        "F (conservative)": scheme_f_conservative,
        "G (trend-adaptive)": scheme_g_trend_adaptive,
    }

    for name, fn in schemes.items():
        t = turnover_analysis(prices, fn, name)
        print(f"\n  {name}:")
        print(f"    调仓次数: {t['n_rebal_total']} (OOS: {t['n_rebal_oos']})")
        print(f"    平均换手: {t['avg_turnover_total']:.3f} (OOS: {t['avg_turnover_oos']:.3f})")
        print(f"    最大换手: {t['max_turnover_total']:.3f} (OOS: {t['max_turnover_oos']:.3f})")
        print(f"    P95 换手: {t['p95_turnover_oos']:.3f} (OOS)")

    # === 2. Correlation Analysis ===
    print("\n" + "=" * 70)
    print("2. Correlation Analysis (相关性分析)")
    print("=" * 70)

    for name, fn in schemes.items():
        c = correlation_analysis(prices, fn, name)
        print(f"\n  {name}:")
        print(f"    vs v1.0:    {c['corr_v1.0']:.3f}")
        print(f"    vs v7.10:   {c['corr_v7.10']:.3f}")
        print(f"    vs DualMom: {c['corr_DualMom']:.3f}")

    # === 3. Bootstrap Sharpe Diff ===
    print("\n" + "=" * 70)
    print(f"3. Bootstrap Sharpe Diff (block bootstrap, n={N_BOOT}, block=21)")
    print("=" * 70)

    boot = bootstrap_sharpe_test(prices, scheme_g_trend_adaptive, scheme_d_signal_weighted)
    print(f"\n  G vs D:")
    print(f"    Observed Sharpe G: {boot['observed_sharpe_a']:.3f}")
    print(f"    Observed Sharpe D: {boot['observed_sharpe_b']:.3f}")
    print(f"    Observed diff:     {boot['observed_diff']:+.3f}")
    print(f"    Bootstrap mean:    {boot['bootstrap_mean_diff']:+.3f}")
    print(f"    Bootstrap std:     {boot['bootstrap_std_diff']:.3f}")
    print(f"    95% CI:            [{boot['ci_low_2.5']:+.3f}, {boot['ci_high_97.5']:+.3f}]")
    print(f"    p-value (one-tail): {boot['p_value_one_sided']:.4f}")
    print(f"    p-value (two-tail): {boot['p_value_two_sided']:.4f}")
    print(f"    Significant:        {'✓ YES' if boot['p_value_two_sided'] < 0.05 else '✗ NO'}")

    # === Save ===
    out_dir = ROOT / "reports" / "momentum_etf_rotation" / "v10_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, fn in schemes.items():
        t = turnover_analysis(prices, fn, name)
        c = correlation_analysis(prices, fn, name)
        rows.append({**t, **c})
    pd.DataFrame(rows).to_csv(out_dir / "v10_3_turnover_correlation.csv", index=False)

    boot_df = pd.DataFrame([boot])
    boot_df.to_csv(out_dir / "v10_3_bootstrap_sharpe_diff.csv", index=False)

    print(f"\n[保存] {out_dir / 'v10_3_turnover_correlation.csv'}")
    print(f"[保存] {out_dir / 'v10_3_bootstrap_sharpe_diff.csv'}")


if __name__ == "__main__":
    main()
