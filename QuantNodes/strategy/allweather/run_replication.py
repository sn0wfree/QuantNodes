# coding=utf-8
"""AllWeather — 消融复现入口 (PLAN Step 6 / Phase 2).

跑批六配置 × 分段 (全样本 / 三段 WF / 2025-2026 压力段),
输出 data/allweather/ablation_results.csv + 终端 markdown 表.

用法:
    python -m QuantNodes.strategy.allweather.run_replication            # 全部配置
    python -m QuantNodes.strategy.allweather.run_replication --configs 0,3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import AllWeatherConfig, run_backtest
from .config import ABLATION_OUTPUT, OUTPUT_DIR, PANEL_V2_OUTPUT

# 消融配置 (PLAN §五; 3b 与 2 同构故并入 2 的解读)
ABLATION: dict[str, dict] = {
    "0_等权+vol":     dict(use_hrp=False, denoise=None, trend_mode="none"),
    "1_HRP_only":     dict(use_hrp=True,  denoise=None, trend_mode="none"),
    "2_HRP+RMT":      dict(use_hrp=True,  denoise="mp",  trend_mode="none"),
    "3a_等权+trendV1": dict(use_hrp=False, denoise=None, trend_mode="v1"),
    "3_HRP+RMT+V1":   dict(use_hrp=True,  denoise="mp",  trend_mode="v1"),
    "3v4_HRP+RMT+V4": dict(use_hrp=True,  denoise="mp",  trend_mode="v4"),
}

# 分段 (固定参数下的分段评估, 不重新优化)
SEGMENTS: dict[str, tuple[str, str]] = {
    "full":      ("2018-01-08", "2026-07-21"),
    "wf_18-20":  ("2018-01-08", "2020-12-31"),
    "wf_21-23":  ("2021-01-01", "2023-12-31"),
    "wf_24-26":  ("2024-01-01", "2026-07-21"),
    "stress_25+": ("2025-01-01", "2026-07-21"),  # 全天候行业承压段
}

METRIC_COLS = ["Sharpe", "AnnRet", "Vol", "MaxDD", "WinRate", "Calmar", "ann_turnover"]


def _metrics_from_nav(nav: pd.Series, compute_metrics_fn) -> dict:
    m = compute_metrics_fn(nav, freq="D")
    return {k: m.get(k) for k in METRIC_COLS}


def run_ablation(panel: pd.DataFrame, config_names: list[str] | None = None,
                 out_dir: Path = OUTPUT_DIR, compute_metrics_fn=None) -> pd.DataFrame:
    """跑批消融配置 × 分段, 返回长表."""
    if compute_metrics_fn is None:
        from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
            compute_metrics as compute_metrics_fn,
        )

    names = config_names or list(ABLATION)
    rows: list[dict] = []
    for name in names:
        cfg = AllWeatherConfig(**ABLATION[name])
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        for seg, (s, e) in SEGMENTS.items():
            seg_nav = res.nav.loc[s:e]
            if len(seg_nav) < 30:
                continue
            m = _metrics_from_nav(seg_nav, compute_metrics_fn)
            rows.append({"config": name, "segment": seg, **m})
        print(f"  [{name}] done")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_OUTPUT, index=False)
    return df


def _noop(nav, freq=None, oos_start=None) -> dict:
    """run_backtest 内部指标占位 (分段指标由本模块统一重算)."""
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default=None,
                    help="逗号分隔配置名, 默认全部")
    ap.add_argument("--panel", type=str, default=None, help="面板路径 (默认 panel_v2)")
    args = ap.parse_args()

    panel_path = Path(args.panel).expanduser() if args.panel else PANEL_V2_OUTPUT
    panel = pd.read_parquet(panel_path)

    names = args.configs.split(",") if args.configs else None
    df = run_ablation(panel, names, out_dir=OUTPUT_DIR)

    # markdown 输出: 每段一张表
    print()
    for seg in SEGMENTS:
        sub = df[df["segment"] == seg].set_index("config")[METRIC_COLS[:6]]
        if sub.empty:
            continue
        print(f"### {seg}")
        print(sub.round(4).to_markdown())
        print()


if __name__ == "__main__":
    main()
