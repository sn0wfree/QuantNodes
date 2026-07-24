# coding=utf-8
"""风格轮动 (StyleRotation) 独立子策略参数网格搜索.

目标: 找到最优 (lookback, top_n_styles, trend_weight, max_weight) 组合, 
让风格轮动子策略作为独立可交易子策略运行.

输出: 控制台打印 top 20 网格结果 + JSON 保存到 reports/.../v4/style_rotation_grid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import (
    StyleRotationConfig,
    StyleRotationSubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.sub_strategy_v4 import run_sub_strategy
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    STYLE_GROUP_CODES,
    load_smartbeta_panel,
)


REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"


def ann_return(nav: pd.Series) -> float:
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def ann_vol(daily_ret: pd.Series) -> float:
    return float(daily_ret.std() * np.sqrt(252))


def max_dd(nav: pd.Series) -> float:
    pk = nav.cummax()
    dd = nav / pk - 1.0
    return float(dd.min())


def sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))


def calmar(nav: pd.Series) -> float:
    ar = ann_return(nav)
    dd = max_dd(nav)
    if dd == 0:
        return 0.0
    return ar / abs(dd)


def metrics_from_nav(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    return {
        "ann_return": ann_return(nav),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets),
        "max_dd": max_dd(nav),
        "calmar": calmar(nav),
        "n_days": len(nav),
    }


def main():
    sb_panel = load_smartbeta_panel(START, END)
    print(f"[data] smartbeta panel: {sb_panel.shape[0]} days × {sb_panel.shape[1]} codes")
    print(f"[data] codes: {list(sb_panel.columns)}")

    lookbacks = [60, 90, 120, 144, 180, 252]
    top_n_styles_list = [1, 2, 3, 5]
    top_n_per_style_list = [1, 2]
    trend_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    max_weights = [1.0, 0.30, 0.20, 0.15]
    rebalance_freqs = ["M", "W-FRI"]
    min_history_options = [144, 252]

    results = []
    total = (len(lookbacks) * len(top_n_styles_list) * len(top_n_per_style_list)
             * len(trend_weights) * len(max_weights) * len(rebalance_freqs)
             * len(min_history_options))
    print(f"[grid] {total} combinations")
    cnt = 0

    for L in lookbacks:
        for tn_s in top_n_styles_list:
            for tn_ps in top_n_per_style_list:
                if tn_s * tn_ps > 6:
                    continue
                for tw in trend_weights:
                    for mw in max_weights:
                        for freq in rebalance_freqs:
                            for mh in min_history_options:
                                cfg = StyleRotationConfig(
                                    lookback=L,
                                    trend_lookback=L,
                                    trend_weight=tw,
                                    top_n_styles=tn_s,
                                    top_n_per_style=tn_ps,
                                    min_history=mh,
                                    max_weight=mw,
                                    rebalance_freq=freq,
                                )
                                strat = StyleRotationSubStrategy(cfg)
                                nav, _ = run_sub_strategy(strat, sb_panel)
                                if nav is None or len(nav) < 252:
                                    cnt += 1
                                    continue
                                m = metrics_from_nav(nav)
                                m.update({
                                    "L": L, "top_n_styles": tn_s, "top_n_per_style": tn_ps,
                                    "trend_weight": tw, "max_weight": mw, "freq": freq,
                                    "min_history": mh,
                                })
                                results.append(m)
                                cnt += 1

    print(f"[grid] {len(results)} valid backtests")

    df = pd.DataFrame(results)
    df = df.sort_values("calmar", ascending=False)
    print("\n[Top 25 by Calmar]")
    cols = ["L", "top_n_styles", "top_n_per_style", "trend_weight",
            "max_weight", "freq", "min_history", "ann_return", "ann_vol", "sharpe",
            "max_dd", "calmar"]
    print(df.head(25)[cols].to_string(index=False))

    print("\n[Top 15 by Sharpe]")
    df_s = df.sort_values("sharpe", ascending=False)
    print(df_s.head(15)[cols].to_string(index=False))

    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "style_rotation_grid.csv", index=False)
    with open(out_dir / "style_rotation_grid.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[save] {out_dir / 'style_rotation_grid.csv'}")
    print(f"[save] {out_dir / 'style_rotation_grid.json'}")

    return df


if __name__ == "__main__":
    main()
