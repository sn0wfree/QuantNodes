# coding=utf-8
"""统一收集所有策略日净值 → 对齐 → 用 common.metrics 算指标 → 存 parquet.

所有策略净值归一化到 1.0 起点, 对齐到同一日期范围.
指标计算复用 common.metrics.compute_metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COMBO_DIR = REPO / "reports/momentum_etf_rotation/combo"
V10_DIR = REPO / "reports/momentum_etf_rotation/v10"
V11_DIR = REPO / "reports/momentum_etf_rotation/v11"
DATA_DIR = REPO / "data/real"

OUTPUT_PATH = COMBO_DIR / "all_strategy_navs.parquet"
METRICS_PATH = COMBO_DIR / "all_strategy_metrics.parquet"

OOS_START = pd.Timestamp("2021-08-01")

STRATEGY_SOURCES: list[tuple[str, Path, dict]] = [
    # (显示名称, 文件路径, 加载参数)
    ("v1.0 locked",
     COMBO_DIR / "unified_v1v5_navs_calA.parquet",
     {"col": "v1.0 locked"}),

    ("v5.1 量价 (逆波动)",
     COMBO_DIR / "unified_v1v5_navs_calA.parquet",
     {"col": "v5.1 量价 (逆波动)"}),

    ("v7.10 TV-PR (标准化+CV)",
     COMBO_DIR / "v7_10_v56_5bp.parquet",
     {"col": 0}),

    ("银河方案-动态仓位",
     COMBO_DIR / "v9_navs.parquet",
     {"col": "银河方案-动态仓位"}),

    ("基础风险平价",
     COMBO_DIR / "v9_navs.parquet",
     {"col": "基础风险平价"}),

    ("等权基准",
     COMBO_DIR / "equal_weight_baseline.parquet",
     {"col": 0}),

    ("v10 DualMom (4资产)",
     V10_DIR / "dual_momentum_nav.parquet",
     {"col": 0}),

    ("v10 4策略Vol-parity",
     V10_DIR / "vol_parity_4strat_nav.parquet",
     {"col": 0}),

    ("v10-DynD 信号加权",
     V10_DIR / "dynamic_nav_D_signal_weighted.parquet",
     {"col": 0}),
]


def load_nav(path: Path, col: str | int = 0) -> pd.Series:
    """加载单个策略 NAV 序列, 归一化到 1.0."""
    df = pd.read_parquet(path)
    if isinstance(col, int):
        s = df.iloc[:, col]
    else:
        s = df[col]
    s.index = pd.to_datetime(s.index)
    s = s.dropna()
    if len(s) == 0:
        raise ValueError(f"{path} has no valid data")
    s = s / s.iloc[0]
    s.name = None
    return s


def load_v11_daily() -> pd.Series:
    """v11 周频 NAV → 日频 (前向填充)."""
    weekly = pd.read_parquet(V11_DIR / "v11_weekly_nav.parquet")
    s = weekly.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s = s.dropna()
    s = s / s.iloc[0]
    # 周频 → 日频 (工作日)
    daily_idx = pd.bdate_range(start=s.index[0], end=s.index[-1])
    daily = s.reindex(daily_idx, method="ffill")
    daily.name = None
    return daily


def load_hs300() -> pd.Series:
    """HS300 基准."""
    df = pd.read_parquet(DATA_DIR / "per_etf" / "510300.parquet")
    s = df["close"] if "close" in df.columns else df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    s = s.dropna()
    s = s / s.iloc[0]
    s.name = None
    return s


def compute_all_metrics(navs: pd.DataFrame, oos_start: pd.Timestamp) -> pd.DataFrame:
    """用 common.metrics.compute_metrics 计算全期和 OOS 指标."""
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

    rows = []
    for col in navs.columns:
        nav = navs[col].dropna()
        if len(nav) < 2:
            continue

        full = compute_metrics(nav)
        oos_nav = nav.loc[oos_start:]
        oos = compute_metrics(oos_nav) if len(oos_nav) > 2 else {}

        rows.append({
            "strategy": col,
            "full_ann_return": full.get("AnnRet", np.nan),
            "full_ann_vol": full.get("Vol", np.nan),
            "full_sharpe": full.get("Sharpe", np.nan),
            "full_sortino": full.get("Sortino", np.nan),
            "full_max_dd": full.get("MaxDD", np.nan),
            "full_calmar": full.get("Calmar", np.nan),
            "full_win_rate": full.get("WinRate", np.nan),
            "oos_ann_return": oos.get("AnnRet", np.nan),
            "oos_ann_vol": oos.get("Vol", np.nan),
            "oos_sharpe": oos.get("Sharpe", np.nan),
            "oos_sortino": oos.get("Sortino", np.nan),
            "oos_max_dd": oos.get("MaxDD", np.nan),
            "oos_calmar": oos.get("Calmar", np.nan),
            "oos_win_rate": oos.get("WinRate", np.nan),
            "n_days": int(len(nav)),
            "oos_days": int(len(oos_nav)),
            "first_date": nav.index[0],
            "last_date": nav.index[-1],
        })

    return pd.DataFrame(rows).set_index("strategy")


def main() -> int:
    print("=" * 60)
    print("收集所有策略日净值...")
    print("=" * 60)

    all_navs: dict[str, pd.Series] = {}

    # 1. 核心策略
    for name, path, kwargs in STRATEGY_SOURCES:
        if not path.exists():
            print(f"  ⚠️  {name}: 文件不存在 ({path.name})")
            continue
        try:
            nav = load_nav(path, kwargs["col"])
            all_navs[name] = nav
            print(f"  ✅ {name:30s} {len(nav):4d}天  {nav.index[0].date()} ~ {nav.index[-1].date()}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    # 2. v11 (周频转日频)
    try:
        nav = load_v11_daily()
        all_navs["v11 5层架构"] = nav
        print(f"  ✅ {'v11 5层架构':30s} {len(nav):4d}天  {nav.index[0].date()} ~ {nav.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ v11: {e}")

    # 3. HS300 基准
    try:
        nav = load_hs300()
        all_navs["HS300 基准"] = nav
        print(f"  ✅ {'HS300 基准':30s} {len(nav):4d}天  {nav.index[0].date()} ~ {nav.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ HS300: {e}")

    # 对齐到公共日期范围 (取交集)
    print(f"\n共 {len(all_navs)} 个策略, 对齐中...")
    df = pd.DataFrame(all_navs)

    # 找有效起始日期 (每个策略都有数据的第一个日期)
    first_valid = df.apply(lambda c: c.first_valid_index()).max()
    last_valid = df.apply(lambda c: c.last_valid_index()).min()
    print(f"  公共区间: {first_valid.date()} ~ {last_valid.date()}")

    # 前向填充 (策略起始日不同, 早的策略填充到对齐起点)
    df = df.ffill()
    df = df.loc[first_valid:last_valid]

    # 确保起点都是 1.0 (用 first_valid 那天归一化)
    df = df.div(df.loc[first_valid])

    print(f"  最终 shape: {df.shape}")
    print("  策略列表:")
    for col in df.columns:
        print(f"    - {col}")

    # 保存 NAV
    df.to_parquet(OUTPUT_PATH)
    print(f"\nNAV 已保存: {OUTPUT_PATH} ({df.shape})")

    # 计算指标
    print("\n计算业绩指标 (common.metrics.compute_metrics)...")
    metrics_df = compute_all_metrics(df, OOS_START)
    metrics_df.to_parquet(METRICS_PATH)
    print(f"指标已保存: {METRICS_PATH} ({metrics_df.shape})")

    # 打印 OOS Calmar 排名
    print("\n" + "=" * 60)
    print("OOS Calmar 排名:")
    print("=" * 60)
    ranked = metrics_df.sort_values("oos_calmar", ascending=False)
    for i, (name, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i:2d}. {name:30s}  Sharpe={row['oos_sharpe']:.3f}  "
              f"Calmar={row['oos_calmar']:.3f}  "
              f"AnnRet={row['oos_ann_return']:+.2%}  "
              f"MaxDD={row['oos_max_dd']:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
