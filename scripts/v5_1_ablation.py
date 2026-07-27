# coding=utf-8
"""v5.1 4 项改动消融实验 — 评估每项的边际贡献.

对比:
- A: 原 v5 (等权, baseline)
- B: 原 v5.1 (逆波动, baseline)
- C: v5.1 + S1 (T+1 调仓, 其他 baseline)
- D: v5.1 + S2 (winsorize, 其他 baseline) — 但 S2 在 cross_section_zscore 共享, 需临时回退
- E: v5.1 + S3 (vol 60+floor 0.01)
- F: v5.1 + S4 (max_weight 0.25)
- G: v5.1 + S1+S3+S4 (不含 S2)
- H: v5.1 + S1+S2+S3+S4 (全部)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.momentum_etf_rotation.v5 import (
    IndustryRotationV5Config,
    IndustryRotationV5SubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v5 import (
    IndustryRotationV5_1Config,
    inverse_vol_weights_v5_1,
)
import QuantNodes.strategy.momentum_etf_rotation.v5.industry_rotation_v5 as v5_mod
import QuantNodes.strategy.momentum_etf_rotation.v5.industry_factors as v5_factors

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"


def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav):
    return float((nav / nav.cummax() - 1.0).min())


def sharpe(rets):
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def metrics(nav):
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def backtest_v5_1_custom(panel, top_n=5, vol_window=60, vol_floor=0.01,
                        max_weight=0.25, rebal_lag=1, use_winsorize=True):
    """v5.1 自定义回测 (S1-S4 各自可控)."""
    dates = panel.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    from QuantNodes.strategy.momentum_etf_rotation.v5 import (
        compute_all_factors_panel, compute_composite_factor,
    )
    cfg = IndustryRotationV5_1Config(
        top_n=top_n, vol_window=vol_window, vol_floor=vol_floor,
        max_weight=max_weight, rebal_lag=rebal_lag,
    )
    factor_panel = compute_all_factors_panel(panel, cfg.factor_cfg)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            if len(composite) >= cfg.top_n:
                top = composite.nlargest(cfg.top_n)
                chosen = list(top.index)
                last_weights = inverse_vol_weights_v5_1(
                    panel, chosen, date, cfg.vol_window, cfg.vol_floor, cfg.rebal_lag,
                )
                # max_weight 约束
                max_w = cfg.max_weight
                capped = dict(last_weights)
                for _ in range(10):
                    excess = 0.0
                    for c, w in capped.items():
                        if w > max_w:
                            excess += w - max_w
                            capped[c] = max_w
                    if excess <= 1e-6:
                        break
                    non_capped = [c for c, w in capped.items() if w < max_w]
                    nc_sum = sum(capped[c] for c in non_capped)
                    if nc_sum > 0 and non_capped:
                        for c in non_capped:
                            capped[c] += excess * (capped[c] / nc_sum)
                last_weights = capped
                total = sum(last_weights.values())
                if total > 0:
                    last_weights = {k: v / total for k, v in last_weights.items()}

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel.columns.get_level_values(0):
                    p_t = panel[code]["close"].iloc[i]
                    p_prev = panel[code]["close"].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates)


def main():
    print("[data] 加载 OHLCV 面板 ...")
    panel = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel = panel.loc[START:END]
    print(f"[data] {panel.shape[0]} 天 × {panel.shape[1]} 列")

    # baseline 数据 (来自 git tag v5.1-baseline 时的代码, 当前为 S2 已生效)
    # 注意: S2 改 cross_section_zscore, 已全局生效
    # 消融: 临时把 cross_section_zscore 回退为原始 z-score
    original_zscore = v5_mod.cross_section_zscore

    def original_zscore_fn(factor_panel, factor, as_of):
        values = {}
        for code, df in factor_panel.items():
            if factor not in df.columns:
                continue
            if as_of not in df.index:
                idx = df.index.get_indexer([as_of], method="ffill")[0]
                if idx < 0:
                    continue
                val = df[factor].iloc[idx]
            else:
                val = df[factor].loc[as_of]
            if pd.isna(val):
                continue
            values[code] = float(val)
        s = pd.Series(values)
        if s.empty:
            return s
        return (s - s.mean()) / s.std() if s.std() > 0 else s * 0

    # v5 等权 (baseline)
    print("\n========= 0. v5 baseline (等权, 旧选股) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 用旧选股
    nav_v5 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=21, vol_floor=1e-4,
        max_weight=0.30, rebal_lag=0,
    )
    full = metrics(nav_v5)
    oos = metrics(nav_v5.loc["2022-01-01":])
    print(f"  v5 baseline: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5 baseline: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 baseline (逆波动, 旧选股) = S2 关
    print("\n========= 1. v5.1 baseline (逆波动, 旧选股) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 用旧选股
    nav_v51_base = backtest_v5_1_custom(
        panel, top_n=5, vol_window=21, vol_floor=1e-4,
        max_weight=0.30, rebal_lag=0,
    )
    full = metrics(nav_v51_base)
    oos = metrics(nav_v51_base.loc["2022-01-01":])
    print(f"  v5.1 baseline: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1 baseline: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S1 (T+1 调仓)
    print("\n========= 2. v5.1 + S1 (T+1 调仓) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 旧选股
    nav_s1 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=21, vol_floor=1e-4,
        max_weight=0.30, rebal_lag=1,
    )
    full = metrics(nav_s1)
    oos = metrics(nav_s1.loc["2022-01-01":])
    print(f"  v5.1+S1: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1+S1: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S2 (winsorize)
    print("\n========= 3. v5.1 + S2 (winsorize) =========")
    v5_mod.cross_section_zscore = original_zscore  # 用新选股 (winsorize)
    nav_s2 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=21, vol_floor=1e-4,
        max_weight=0.30, rebal_lag=0,
    )
    full = metrics(nav_s2)
    oos = metrics(nav_s2.loc["2022-01-01":])
    print(f"  v5.1+S2: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1+S2: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S3 (vol 60+floor 0.01)
    print("\n========= 4. v5.1 + S3 (vol 60+0.01) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 旧选股
    nav_s3 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=60, vol_floor=0.01,
        max_weight=0.30, rebal_lag=0,
    )
    full = metrics(nav_s3)
    oos = metrics(nav_s3.loc["2022-01-01":])
    print(f"  v5.1+S3: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1+S3: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S4 (max_weight 0.25)
    print("\n========= 5. v5.1 + S4 (max_weight 0.25) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 旧选股
    nav_s4 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=21, vol_floor=1e-4,
        max_weight=0.25, rebal_lag=0,
    )
    full = metrics(nav_s4)
    oos = metrics(nav_s4.loc["2022-01-01":])
    print(f"  v5.1+S4: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1+S4: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S1+S3+S4 (无 S2)
    print("\n========= 6. v5.1 + S1+S3+S4 (无 S2) =========")
    v5_mod.cross_section_zscore = original_zscore_fn  # 旧选股
    nav_134 = backtest_v5_1_custom(
        panel, top_n=5, vol_window=60, vol_floor=0.01,
        max_weight=0.25, rebal_lag=1,
    )
    full = metrics(nav_134)
    oos = metrics(nav_134.loc["2022-01-01":])
    print(f"  v5.1+S1+S3+S4: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1+S1+S3+S4: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # v5.1 + S1+S2+S3+S4 (全部)
    print("\n========= 7. v5.1.1 (S1+S2+S3+S4 全部) =========")
    v5_mod.cross_section_zscore = original_zscore  # 新选股 (winsorize)
    nav_all = backtest_v5_1_custom(
        panel, top_n=5, vol_window=60, vol_floor=0.01,
        max_weight=0.25, rebal_lag=1,
    )
    full = metrics(nav_all)
    oos = metrics(nav_all.loc["2022-01-01":])
    print(f"  v5.1.1: 全期 Calmar={full['calmar']:.3f}  OOS Calmar={oos['calmar']:.3f}")
    print(f"  v5.1.1: 全期 Sharpe={full['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}")

    # 恢复
    v5_mod.cross_section_zscore = original_zscore

    # 保存所有 NAV
    out = pd.DataFrame({
        "v5_baseline": nav_v5,
        "v5_1_baseline": nav_v51_base,
        "v5_1_S1": nav_s1,
        "v5_1_S2": nav_s2,
        "v5_1_S3": nav_s3,
        "v5_1_S4": nav_s4,
        "v5_1_S134": nav_134,
        "v5_1_all": nav_all,
    })
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_path = out_dir / "v5_1_ablation_navs.parquet"
    out.to_parquet(out_path)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
