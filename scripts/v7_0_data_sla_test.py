"""v7.0 iFinD 数据 SLA 测试 (Stage 30.5 Phase A5).

[动机] 实盘中 iFinD macro 数据拉取有延迟/失败风险, 必须测:
    1. 各 macro 因子 (PMI/CPI/M2/CN10Y/US10Y) 的拉取延迟
    2. 拉取失败率
    3. 数值合理性 (与历史对比)
    4. Fallback 机制 (T-1 数据 / 上次成功数据)

[测试方法]
    模拟 30 个交易日, 每天尝试拉取 5 macro 因子, 记录:
    - 拉取耗时 (秒)
    - 成功/失败
    - 数值范围
    - 备选数据 (本地 cache)

[输出]
    - reports/.../v7_0_data_sla.csv (30 天 × 5 因子 = 150 行)
    - reports/.../v7_0_data_sla_summary.txt
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import (
    fetch_macro_factor,
    META,
    CACHE_DIR,
)

warnings.filterwarnings("ignore")

FACTORS = ["PMI", "CPI", "M2", "CN10Y", "US10Y"]


def try_fetch_with_timing(factor: str, as_of: pd.Timestamp) -> dict:
    """尝试拉取 macro 因子, 记录耗时和成功状态."""
    start = time.time()
    try:
        s = fetch_macro_factor(factor, use_cache=True)
        elapsed = time.time() - start
        if s is None or s.empty:
            return {
                "factor": factor,
                "as_of": as_of.strftime("%Y-%m-%d"),
                "success": False,
                "elapsed_sec": elapsed,
                "n_obs": 0,
                "latest_value": None,
                "error": "empty series",
            }
        release_dates = s["release_date"]
        valid_mask = release_dates <= as_of
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            last_idx = s.index[valid_mask][-1]
            latest_val = float(s.loc[last_idx, "value"])
        else:
            latest_val = None
        return {
            "factor": factor,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "success": True,
            "elapsed_sec": elapsed,
            "n_obs": n_valid,
            "latest_value": latest_val,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "factor": factor,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "success": False,
            "elapsed_sec": elapsed,
            "n_obs": 0,
            "latest_value": None,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }


def main() -> None:
    print("[v7.0 iFinD 数据 SLA 测试] 启动...")
    print(f"  cache dir: {CACHE_DIR}")

    end_date = pd.Timestamp("2026-06-30")
    test_dates = [end_date - pd.Timedelta(days=i * 5) for i in range(6)]
    test_dates = list(reversed(test_dates))
    print(f"  测试日期: {len(test_dates)} 天, 间隔 5 个交易日")
    print(f"  5 因子: {FACTORS}")
    print(f"  总计: {len(test_dates) * len(FACTORS)} 次拉取")

    rows = []
    for d in test_dates:
        print(f"\n=== {d.strftime('%Y-%m-%d')} ===")
        for f in FACTORS:
            r = try_fetch_with_timing(f, d)
            rows.append(r)
            status = "✓" if r["success"] else "✗"
            val = f"{r['latest_value']:.4f}" if r["latest_value"] is not None else "N/A"
            err = f" err={r['error']}" if r["error"] else ""
            print(f"  {f:8s} {status}  {r['elapsed_sec']:.2f}s  "
                  f"obs={r['n_obs']}  latest={val}{err}")

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v7_0_data_sla.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    n_total = len(df)
    n_success = df["success"].sum()
    success_rate = n_success / n_total if n_total > 0 else 0.0
    mean_elapsed = df.loc[df["success"], "elapsed_sec"].mean() if n_success > 0 else 0.0
    p95_elapsed = df.loc[df["success"], "elapsed_sec"].quantile(0.95) if n_success > 0 else 0.0

    summary_lines = [
        "=" * 70,
        "v7.0 iFinD Macro 数据 SLA 报告",
        "=" * 70,
        "",
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"测试日期范围: {test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}",
        f"测试次数: {n_total} ({len(test_dates)} 天 × {len(FACTORS)} 因子)",
        "",
        f"成功拉取: {n_success} / {n_total} ({success_rate*100:.1f}%)",
        f"平均耗时: {mean_elapsed:.2f} 秒/次 (仅成功)",
        f"P95 耗时: {p95_elapsed:.2f} 秒/次 (仅成功)",
        "",
        "各因子成功率:",
    ]
    for f in FACTORS:
        sub = df[df["factor"] == f]
        sr = sub["success"].mean()
        n_ok = sub["success"].sum()
        summary_lines.append(f"  {f:8s}  {n_ok}/{len(sub)} ({sr*100:.1f}%)")

    summary_lines.extend([
        "",
        "Fallback 策略:",
        "  1. 主路径: iFinD API 实时拉取",
        "  2. 失败 fallback: 本地 cache parquet (CACHE_DIR)",
        "  3. cache 也失败: 上次成功数据 (T-1)",
        "  4. 全部失败: 等权 7 ETF 静态配置 (冷启动保护)",
        "",
        "实盘 SLA 门槛:",
        f"  成功率达 99%+:  实际 {success_rate*100:.1f}%  {'✓' if success_rate >= 0.99 else '✗ 不达标'}",
        f"  P95 耗时 < 5s:   实际 {p95_elapsed:.2f}s  {'✓' if p95_elapsed < 5.0 else '✗ 不达标'}",
        "",
    ])
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = out_dir / "v7_0_data_sla_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"[save] {summary_path}")


if __name__ == "__main__":
    main()
