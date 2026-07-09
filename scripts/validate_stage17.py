# coding=utf-8
"""v4 6 模式回测验证 (Stage 17, v4.0).

6 模式:
- v3_baseline: Stage 16A 多策略 (动量+反转+行业轮动)  // 用 v3 multi_strategy
- v4A_style: 仅风格轮动
- v4B_smartbeta: 仅 Smart β
- v4C_combo: 风格 + Smart β (无因子择时)
- v4D_ic: + IC 因子择时
- v4E_hmm: + HMM 因子择时 (退化为 v4C, 待实施距离先验)
- v4F_fusion: + IC + HMM 融合 (待实施)

数据:
- v3: 44 ETF 主面板 (data/real/etf_nav_2018-01-01_2026-06-30.parquet)
- v4: 12 Smart β ETF 面板 (data/real/etf_nav_smartbeta_*.parquet)

输出:
- reports/momentum_etf_rotation/v4/stage17_*.{md,json,parquet}
- 4 个 NAV + 1 个 IC 时序图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation.common.data import load_etf_nav_panel
from QuantNodes.strategy.momentum_etf_rotation.v3 import (
    MultiStrategyConfig,
    ReversionConfig,
    IndustryRotationConfig,
    DEFAULT_POOL,
    run_multi_strategy_backtest as run_v3_backtest,
)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*not supported.*")
from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    FactorTimingConfig,
    RegimeConfig,
    RegimeDetector,
    load_smartbeta_panel,
    run_v4_mode,
)


def metrics(nav: pd.Series) -> dict:
    """计算关键业绩指标."""
    n = len(nav)
    if n < 2:
        return {}
    ann_ret = nav.iloc[-1] ** (252 / n) - 1
    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) < 2:
        return {"ann_return": float(ann_ret)}
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    max_dd = float((nav / nav.cummax() - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    return {
        "final_nav": float(nav.iloc[-1]),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def period_return(nav: pd.Series, start: str, end: str) -> float | None:
    """计算指定区间的累计收益."""
    nav_in = nav.loc[start:end]
    if len(nav_in) < 2:
        return None
    return float(nav_in.iloc[-1] / nav_in.iloc[0] - 1)


def main():
    print("=" * 70)
    print("Stage 17 v4 — 6 模式回测验证")
    print("=" * 70)

    # 数据准备
    print("\n[1/6] 加载数据...")
    panel_v3 = load_etf_nav_panel(
        start="2018-01-01", end="2026-06-30",
        data_dir=PROJECT_ROOT / "data" / "real",
        codes=None, ffill_limit=5,
    )
    panel_v4 = load_smartbeta_panel()
    print(f"  v3 panel: {panel_v3.shape} ({panel_v3.index[0].date()} ~ {panel_v3.index[-1].date()})")
    print(f"  v4 panel: {panel_v4.shape} ({panel_v4.index[0].date()} ~ {panel_v4.index[-1].date()})")

    # 跑 v3 baseline
    print("\n[2/6] 跑 v3 baseline (Stage 16A)...")
    r_v3 = run_v3_backtest(panel_v3, DEFAULT_POOL, MultiStrategyConfig())
    m_v3 = metrics(r_v3.nav)
    print(f"  v3: Sharpe={m_v3.get('sharpe', 0):.3f}, Calmar={m_v3.get('calmar', 0):.3f}, Nav={m_v3.get('final_nav', 0):.3f}")

    # 跑 5 个 v4 模式
    print("\n[3/6] 跑 4 个 v4 模式 (A/B/C/D)...")
    cfg_ft = FactorTimingConfig(forward_window=10)  # best from IC validation

    results = {"v3_baseline": (r_v3.nav, m_v3)}
    for mode in ["v4A_style", "v4B_smartbeta", "v4C_combo", "v4D_ic"]:
        r = run_v4_mode(panel_v4, mode, factor_timing_cfg=cfg_ft)
        m = metrics(r.nav)
        results[mode] = (r.nav, m)
        print(f"  {mode}: Sharpe={m.get('sharpe', 0):.3f}, Calmar={m.get('calmar', 0):.3f}, Nav={m.get('final_nav', 0):.3f}")

    # v4E / v4F: HMM 训练
    print("\n[4/6] v4E_hmm / v4F_fusion (HMM 距离先验)...")
    hmm_detector = RegimeDetector(RegimeConfig(n_iter=30))
    hmm_detector.fit(panel_v4, panel_v4.index[-1])
    print(f"  HMM label_map: {hmm_detector.label_map}")

    # 打印 HMM regime 分布 (用更短的 min_duration 让 3 状态都出现)
    hmm_series = hmm_detector.predict_series(
        panel_v4, panel_v4.index[0], panel_v4.index[-1], step=5, min_duration=6,
    )
    print(f"  HMM regime 分布: {hmm_series.value_counts().to_dict()}")

    r_E = run_v4_mode(panel_v4, "v4E_hmm", factor_timing_cfg=cfg_ft, hmm_detector=hmm_detector)
    m_E = metrics(r_E.nav)
    results["v4E_hmm"] = (r_E.nav, m_E)
    print(f"  v4E: Sharpe={m_E.get('sharpe', 0):.3f}, Calmar={m_E.get('calmar', 0):.3f}")

    r_F = run_v4_mode(panel_v4, "v4F_fusion", factor_timing_cfg=cfg_ft, hmm_detector=hmm_detector)
    m_F = metrics(r_F.nav)
    results["v4F_fusion"] = (r_F.nav, m_F)
    print(f"  v4F: Sharpe={m_F.get('sharpe', 0):.3f}, Calmar={m_F.get('calmar', 0):.3f}")

    # 关键区间分析
    print("\n[5/6] 关键区间分析...")
    periods = {
        "924": ("2024-09-23", "2024-10-31"),
        "2025_H2": ("2025-07-01", "2025-12-31"),
        "2026_H1": ("2026-01-01", "2026-06-30"),
    }
    period_results = {}
    for period_name, (start, end) in periods.items():
        period_results[period_name] = {}
        for name, (nav, _) in results.items():
            ret = period_return(nav, start, end)
            if ret is not None:
                period_results[period_name][name] = ret
        # 打印
        print(f"  {period_name} ({start} ~ {end}):")
        for name, ret in period_results[period_name].items():
            print(f"    {name}: {ret:+.3%}")

    # 落盘
    print("\n[6/6] 落盘...")
    out_dir = Path("reports/momentum_etf_rotation/v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # NAV 合并
    nav_all = pd.DataFrame({name: nav for name, (nav, _) in results.items()})
    nav_all.to_parquet(out_dir / "stage17_navs.parquet")
    print(f"  NAV: {out_dir / 'stage17_navs.parquet'}, shape={nav_all.shape}")

    # Save HMM regime history
    hmm_series.to_csv(out_dir / "hmm_regime_history.csv", header=True)

    # summary JSON
    summary = {
        "config": {
            "v3_pool": "44 ETFs (data/real/etf_nav_2018-01-01_2026-06-30.parquet)",
            "v4_pool": "12 Smart β ETFs (data/real/etf_nav_smartbeta_*.parquet)",
            "factor_timing_cfg": {
                "ic_window": cfg_ft.ic_window,
                "forward_window": cfg_ft.forward_window,
                "lookback": cfg_ft.lookback,
                "base": cfg_ft.base,
                "power": cfg_ft.power,
            },
            "HMM": "待实施 (距离先验方案, 后续 Stage 17E)",
        },
        "metrics": {name: m for name, (_, m) in results.items()},
        "period_returns": period_results,
        "best_v4": max(
            (k for k in results if k.startswith("v4") and k != "v4E_hmm" and k != "v4F_fusion"),
            key=lambda k: results[k][1].get("calmar", 0),
        ),
    }
    with open(out_dir / "stage17_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary: {out_dir / 'stage17_summary.json'}")

    # 打印对比表
    print("\n" + "=" * 70)
    print("Stage 17 6 模式对比 (全周期 2018-2026)")
    print("=" * 70)
    print(f"  {'模式':<18} {'Sharpe':>8} {'Calmar':>8} {'AnnRet':>8} {'DD':>8} {'Nav':>8}")
    print(f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, (_, m) in results.items():
        print(f"  {name:<18} {m.get('sharpe', 0):>8.3f} {m.get('calmar', 0):>8.3f} {m.get('ann_return', 0):>8.3%} {m.get('max_drawdown', 0):>8.3%} {m.get('final_nav', 0):>8.3f}")

    print("\n完成!")


if __name__ == "__main__":
    main()
