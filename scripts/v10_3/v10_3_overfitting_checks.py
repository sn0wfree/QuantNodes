"""Overfitting checks for v10.3 G (Trend-Adaptive) scheme.

5 项检查:
  1. 参数敏感性: TREND_THRESH = 0.015/0.020/0.025/0.030
  2. 随机基线: 等权 + 100 随机权重方案
  3. Sub-period: 2018-2020/2020-2022/2022-2024/2024-2026
  4. Walk-forward: 3y/1y + 5y/1y splits
  5. Bootstrap block size: 5/21/63
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics  # noqa: E402
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


def make_g_at_threshold(prices: pd.DataFrame, threshold: float):
    """创建指定阈值的 G 方案."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_dates = prices.resample('M').last().index
    market_ret_5d = prices['v7.10'].pct_change(5)
    prev_w = dict(BASE_WEIGHTS)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w
    for i in range(252, len(prices)):
        date = prices.index[i]
        trend_up = i >= 5 and market_ret_5d.iloc[i] > threshold
        if trend_up:
            sharpes = {}
            for col in prices.columns:
                rets_12m = prices[col].iloc[i-252:i].pct_change().dropna()
                if len(rets_12m) > 50:
                    ann_ret = prices[col].iloc[i] / prices[col].iloc[i-252] - 1
                    vol = rets_12m.std() * np.sqrt(252)
                    sharpes[col] = ann_ret / vol if vol > 0 else 0
                else:
                    sharpes[col] = 0
            sharpes_pos = {k: max(v, 0) + 0.1 for k, v in sharpes.items()}
            total_s = sum(sharpes_pos.values())
            w_target = {
                col: (sharpes_pos[col] / total_s) * BASE_WEIGHTS[col]
                for col in prices.columns
            }
            total_w = sum(w_target.values())
            w_target = {k: v / total_w for k, v in w_target.items()}
            prev_w = w_target
        else:
            if date in rebal_dates:
                sharpes = {}
                for col in prices.columns:
                    rets_12m = prices[col].iloc[i-252:i].pct_change().dropna()
                    if len(rets_12m) > 50:
                        ann_ret = prices[col].iloc[i] / prices[col].iloc[i-252] - 1
                        vol = rets_12m.std() * np.sqrt(252)
                        sharpes[col] = ann_ret / vol if vol > 0 else 0
                    else:
                        sharpes[col] = 0
                sharpes_pos = {k: max(v, 0) + 0.1 for k, v in sharpes.items()}
                total_s = sum(sharpes_pos.values())
                w_target = {
                    col: (sharpes_pos[col] / total_s) * BASE_WEIGHTS[col]
                    for col in prices.columns
                }
                total_w = sum(w_target.values())
                w_target = {k: v / total_w for k, v in w_target.items()}
                from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import _apply_constraints
                prev_w = _apply_constraints(w_target, prev_w)
        weights.loc[date] = prev_w
    return weights


def eval_oos(prices: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """计算 OOS 指标."""
    nav = compute_nav(prices, weights)
    oos = nav.loc[OOS_START:OOS_END].dropna()
    if len(oos) < 50:
        return {}
    m = compute_metrics(oos)
    return {
        "Sharpe": m.get("Sharpe", 0),
        "Calmar": m.get("Calmar", 0),
        "AnnRet": m.get("AnnRet", 0),
        "MaxDD": m.get("MaxDD", 0),
    }


def main():
    prices = load_navs()
    print(f"Prices: {prices.index[0].date()} ~ {prices.index[-1].date()}, {len(prices)} days")

    all_results = []

    # === 1. Parameter Sensitivity ===
    print("\n" + "=" * 70)
    print("1. Parameter Sensitivity (TREND_THRESH)")
    print("=" * 70)
    thresholds = [0.015, 0.020, 0.025, 0.030]
    for t in thresholds:
        w = make_g_at_threshold(prices, t)
        m = eval_oos(prices, w)
        tag = " (current)" if t == 0.020 else ""
        print(f"  THRESH={t:.3f}{tag}: Sharpe={m['Sharpe']:.3f}, Calmar={m['Calmar']:.3f}, "
              f"MaxDD={m['MaxDD']*100:.2f}%")
        all_results.append({"check": "param_sensitivity", "threshold": t, **m})

    # === 2. Random Baseline ===
    print("\n" + "=" * 70)
    print("2. Random Baseline Comparison")
    print("=" * 70)

    # Equal weight
    w_eq = pd.DataFrame(BASE_WEIGHTS, index=prices.index, columns=prices.columns)
    m_eq = eval_oos(prices, w_eq)
    print(f"  Equal-weight:  Sharpe={m_eq['Sharpe']:.3f}, Calmar={m_eq['Calmar']:.3f}")
    all_results.append({"check": "random_baseline", "scheme": "equal_weight", **m_eq})

    # 100 random schemes
    random_sharpes = []
    random_calmars = []
    for seed in range(100):
        r = np.random.default_rng(seed + 100)
        # 月频随机权重
        w_rand = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_w = dict(BASE_WEIGHTS)
        for i in range(min(252, len(prices))):
            w_rand.iloc[i] = prev_w
        rebal_dates = prices.resample('M').last().index
        for i in range(252, len(prices)):
            date = prices.index[i]
            if date in rebal_dates:
                raw = r.dirichlet(np.ones(4))
                prev_w = {col: float(raw[j]) for j, col in enumerate(prices.columns)}
            w_rand.loc[date] = prev_w
        m_r = eval_oos(prices, w_rand)
        random_sharpes.append(m_r['Sharpe'])
        random_calmars.append(m_r['Calmar'])

    g_w = scheme_g_trend_adaptive(prices)
    m_g = eval_oos(prices, g_w)
    g_sharpe_pctile = (np.array(random_sharpes) < m_g['Sharpe']).mean() * 100
    g_calmar_pctile = (np.array(random_calmars) < m_g['Calmar']).mean() * 100

    print(f"  Random 100: Sharpe mean={np.mean(random_sharpes):.3f}, "
          f"std={np.std(random_sharpes):.3f}, "
          f"max={np.max(random_sharpes):.3f}")
    print(f"  G scheme:  Sharpe={m_g['Sharpe']:.3f}, "
          f"percentile={g_sharpe_pctile:.1f}%")
    print(f"  Random 100: Calmar mean={np.mean(random_calmars):.3f}, "
          f"std={np.std(random_calmars):.3f}, "
          f"max={np.max(random_calmars):.3f}")
    print(f"  G scheme:  Calmar={m_g['Calmar']:.3f}, "
          f"percentile={g_calmar_pctile:.1f}%")
    all_results.append({
        "check": "random_baseline", "scheme": "G_vs_random",
        "g_sharpe": m_g['Sharpe'], "g_sharpe_pctile": g_sharpe_pctile,
        "g_calmar": m_g['Calmar'], "g_calmar_pctile": g_calmar_pctile,
    })

    # === 3. Sub-period Analysis ===
    print("\n" + "=" * 70)
    print("3. Sub-period Analysis")
    print("=" * 70)
    sub_periods = [
        ("2018-01-01", "2019-12-31", "2018-2020 牛熊"),
        ("2020-01-01", "2021-12-31", "2020-2022 疫情+加息"),
        ("2022-01-01", "2023-12-31", "2022-2024 加息尾声"),
        ("2024-01-01", "2026-05-06", "2024-2026 多事件"),
    ]
    for start, end, label in sub_periods:
        nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices))
        nav_d = compute_nav(prices, scheme_d_signal_weighted(prices))
        seg_g = nav_g.loc[start:end].dropna()
        seg_d = nav_d.loc[start:end].dropna()
        m_g_sub = compute_metrics(seg_g) if len(seg_g) > 50 else {}
        m_d_sub = compute_metrics(seg_d) if len(seg_d) > 50 else {}
        g_sharpe = m_g_sub.get("Sharpe", 0)
        d_sharpe = m_d_sub.get("Sharpe", 0)
        g_calmar = m_g_sub.get("Calmar", 0)
        d_calmar = m_d_sub.get("Calmar", 0)
        delta_sharpe = g_sharpe - d_sharpe
        delta_calmar = g_calmar - d_calmar
        marker_s = "✓" if delta_sharpe > 0 else "✗"
        marker_c = "✓" if delta_calmar > 0 else "✗"
        print(f"  {label:25s}: G Sharpe={g_sharpe:.3f} vs D {d_sharpe:.3f} "
              f"(Δ={delta_sharpe:+.3f} {marker_s}) | "
              f"G Calmar={g_calmar:.3f} vs D {d_calmar:.3f} "
              f"(Δ={delta_calmar:+.3f} {marker_c})")
        all_results.append({
            "check": "sub_period", "period": label,
            "g_sharpe": g_sharpe, "d_sharpe": d_sharpe,
            "g_calmar": g_calmar, "d_calmar": d_calmar,
        })

    # === 4. Walk-forward Robustness ===
    print("\n" + "=" * 70)
    print("4. Walk-forward Robustness (different train/test splits)")
    print("=" * 70)

    def run_wf(prices, train_years, test_years):
        folds = []
        fold_start = pd.Timestamp("2018-01-01")
        while True:
            train_end = fold_start + pd.DateOffset(years=train_years)
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=test_years)
            if test_end > pd.Timestamp("2026-05-06"):
                break
            folds.append({
                "train_start": fold_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end,
            })
            fold_start += pd.DateOffset(years=test_years)

        results_g, results_d = [], []
        for f in folds:
            nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices))
            nav_d = compute_nav(prices, scheme_d_signal_weighted(prices))
            test_s = f["test_start"] + pd.Timedelta(days=5)  # purge
            test_e = f["test_end"]
            for nav, label, res in [(nav_g, "G", results_g), (nav_d, "D", results_d)]:
                seg = nav.loc[test_s:test_e].dropna()
                if len(seg) > 50:
                    m = compute_metrics(seg)
                    res.append({"Sharpe": m.get("Sharpe", 0), "Calmar": m.get("Calmar", 0)})
        return results_g, results_d

    splits = [(4, 1, "4y/1y (current)"), (3, 1, "3y/1y"), (5, 1, "5y/1y")]
    for train_y, test_y, label in splits:
        rg, rd = run_wf(prices, train_y, test_y)
        if rg and rd:
            g_sharpes = [r['Sharpe'] for r in rg]
            d_sharpes = [r['Sharpe'] for r in rd]
            g_calmars = [r['Calmar'] for r in rg]
            d_calmars = [r['Calmar'] for r in rd]
            n_beat_s = sum(1 for g, d in zip(g_sharpes, d_sharpes) if g > d)
            n_beat_c = sum(1 for g, d in zip(g_calmars, d_calmars) if g > d)
            print(f"  {label:15s}: {len(rg)} folds, G beats D: "
                  f"Sharpe {n_beat_s}/{len(rg)} ({n_beat_s/len(rg)*100:.0f}%), "
                  f"Calmar {n_beat_c}/{len(rg)} ({n_beat_c/len(rg)*100:.0f}%)")
            print(f"    G: Sharpe={np.mean(g_sharpes):.3f}±{np.std(g_sharpes):.3f}, "
                  f"Calmar={np.mean(g_calmars):.3f}±{np.std(g_calmars):.3f}")
            print(f"    D: Sharpe={np.mean(d_sharpes):.3f}±{np.std(d_sharpes):.3f}, "
                  f"Calmar={np.mean(d_calmars):.3f}±{np.std(d_calmars):.3f}")
            all_results.append({
                "check": "wf_robustness", "split": label,
                "g_sharpe_mean": np.mean(g_sharpes), "d_sharpe_mean": np.mean(d_sharpes),
                "g_calmar_mean": np.mean(g_calmars), "d_calmar_mean": np.mean(d_calmars),
                "n_beat_sharpe": n_beat_s, "n_beat_calmar": n_beat_c, "n_folds": len(rg),
            })

    # === 5. Bootstrap Block Size Sensitivity ===
    print("\n" + "=" * 70)
    print("5. Bootstrap Block Size Sensitivity")
    print("=" * 70)

    nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices))
    nav_d = compute_nav(prices, scheme_d_signal_weighted(prices))
    oos_g = nav_g.loc[OOS_START:OOS_END].pct_change().dropna()
    oos_d = nav_d.loc[OOS_START:OOS_END].pct_change().dropna()
    common = oos_g.index.intersection(oos_d.index)
    oos_g = oos_g.loc[common]
    oos_d = oos_d.loc[common]

    obs_sharpe_g = oos_g.mean() / oos_g.std() * np.sqrt(252)
    obs_sharpe_d = oos_d.mean() / oos_d.std() * np.sqrt(252)
    obs_diff = obs_sharpe_g - obs_sharpe_d

    block_sizes = [5, 21, 63]
    for bs in block_sizes:
        centered_g = oos_g - oos_g.mean()
        centered_d = oos_d - oos_d.mean()
        rng_boot = np.random.default_rng(42)
        n = len(oos_g)
        n_blocks = n // bs
        diffs = []
        for _ in range(5000):
            starts = rng_boot.integers(0, n - bs + 1, size=n_blocks)
            bg, bd = [], []
            for s in starts:
                bg.extend(centered_g.iloc[s:s+bs].values)
                bd.extend(centered_d.iloc[s:s+bs].values)
            bg = np.array(bg[:n])
            bd = np.array(bd[:n])
            sg = bg.mean() / oos_g.std() * np.sqrt(252) if oos_g.std() > 0 else 0
            sb = bd.mean() / oos_d.std() * np.sqrt(252) if oos_d.std() > 0 else 0
            diffs.append(sg - sb)
        diffs = np.array(diffs)
        p_one = float((diffs >= obs_diff).mean())
        p_two = float((np.abs(diffs) >= np.abs(obs_diff)).mean())
        ci_low = float(np.percentile(diffs, 2.5))
        ci_high = float(np.percentile(diffs, 97.5))
        sig = "✓" if p_two < 0.05 else "✗"
        print(f"  block={bs:2d}: observed_diff={obs_diff:+.3f}, "
              f"CI=[{ci_low:+.3f}, {ci_high:+.3f}], "
              f"p_one={p_one:.4f}, p_two={p_two:.4f} {sig}")
        all_results.append({
            "check": "bootstrap_block", "block_size": bs,
            "obs_diff": obs_diff, "ci_low": ci_low, "ci_high": ci_high,
            "p_one_sided": p_one, "p_two_sided": p_two,
        })

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    df = pd.DataFrame(all_results)
    out_dir = ROOT / "reports" / "momentum_etf_rotation" / "v10_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "v10_3_overfitting_checks.csv", index=False)
    print(f"[保存] {out_dir / 'v10_3_overfitting_checks.csv'}")


if __name__ == "__main__":
    main()
