#!/usr/bin/env python3
"""gold_backtest.py — 黄金CTA O2O日频回测分析工具.

固化生产链构建 + 业绩统计 + 逐年/事件段分析, 消除分析脚本中的重复内联代码.

数据口径 (重要):
    - 价格: futures_main_close_au_adj (渐进式换月调整), 实际数据从 2018-06-20 起
    - 收益: open-to-open, 信号 shift(1), 成本 cost_bps
    - 一层风控: compute_layered_ensemble_signal 内置 (daily_cap 可调)
    - 二层风控: compute_risk_overlay (L倍杠杆后)
    - 指标区间: 2018-06-20 ~ 数据末日 (所有对比必须用同一窗口)

用法:
    python3 scripts/cmf/gold_backtest.py                        # 生产现状链报告
    python3 scripts/cmf/gold_backtest.py --fc -0.05 --hard -0.20  # 自定义链配置
    python3 scripts/cmf/gold_backtest.py --stack                # 全执行栈 (B3默认)
    python3 scripts/cmf/gold_backtest.py --stack --compare      # 现状 vs 全栈 vs B3
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts/cmf")

import argparse
import math

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes/QuantNodes")
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal,
    compute_risk_overlay,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    _load_indicator_d,
)

DB_PATH = "data/cmf/fundamental.duckdb"
COST_BPS = 5.0

# 生产现状参数 (锚点: CAGR 15.42% / Sharpe 1.52 / MaxDD -13.97% @ 2018-2026)
DEFAULTS = dict(
    L=1.5,
    first_cap=-0.02,
    second_cap=-0.03,
    second_hard=-0.15,
    second_soft=-0.12,
    vol_cap=0.20,
    cooldown_days=5,
    deadband=0.05,
)

# 全执行栈参数 B3 (锚点: CAGR 18.47% / Sharpe 1.56 / MaxDD -16.84% @ 2018-2026)
# 结构: 链(-5/-5/-20) → 核心裁剪1.25 → +周网格库存(3档×0.083, sp=2%) → 总封顶1.5
#       → 日平滑cap0.20 → 应急(单日<-5%压向0)
DEFAULTS_STACK = dict(
    # 链层 (被测参数, 非生产现状)
    L=1.5,
    first_cap=-0.05,
    second_cap=-0.05,
    second_hard=-0.20,
    second_soft=-0.12,
    vol_cap=0.20,
    cooldown_days=5,
    deadband=0.05,
    # 预算制网格层
    core_cap=1.25,
    grid_sp=0.02,
    grid_step=0.0833,
    grid_tiers=3,
    budget=1.5,
    # 日平滑层
    smooth_cap=0.20,
    emergency=-0.05,
)


def load_au_ohlcv() -> tuple[pd.Series, pd.Series]:
    """加载调整后AU期货 (close, open), 数据从2018-06-20起."""
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        "SELECT trade_date, open, close FROM futures_main_close_au_adj ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df["close"], df["open"]


def load_vix_gvz(close_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """vix/gvz, 生产口径: reindex + ffill + shift(2)."""
    vix = _load_indicator_d("vix", "gold_fred").reindex(close_index).ffill().shift(2)
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close_index).ffill().shift(2)
    return vix, gvz


def build_chain(
    L: float = 1.5,
    first_cap: float = -0.02,
    second_cap: float = -0.03,
    second_hard: float = -0.15,
    second_soft: float = -0.12,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    deadband: float = 0.05,
) -> pd.Series:
    """生产链: 一层风控信号 → ×L杠杆 → 二层overlay → 死带平滑."""
    close, _ = load_au_ohlcv()
    vix, gvz = load_vix_gvz(close.index)

    sig = compute_layered_ensemble_signal(use_risk_overlay=True, daily_cap=first_cap)
    pos_base = sig["position"]
    pos_lev = (pos_base * L).clip(0, L * 1.2)

    pos = compute_risk_overlay(
        close=close, position=pos_lev, vix=vix, gvz=gvz,
        hard_cap=second_hard, soft_cap=second_soft, daily_cap=second_cap,
        vol_cap=vol_cap, cooldown_days=cooldown_days,
    )
    for i in range(1, len(pos)):
        if abs(pos.iloc[i] - pos.iloc[i - 1]) < deadband:
            pos.iloc[i] = pos.iloc[i - 1]
    return pos


def weekly_grid_inventory(
    close: pd.Series,
    target_index: pd.DatetimeIndex,
    spacing_pct: float = 0.02,
    step: float = 0.0833,
    max_grids: int = 3,
) -> pd.Series:
    """周频网格库存 (预算制): 价格在锚点下方每跌1档, 库存+step, 回到锚点上方归零.

    周五收盘按档位计库存, 下周执行 (reindex ffill + shift(1) 避免当日决策当日用).
    锚点漂移超过 (max_grids+1) 档时重置 (库存清零, 防止趋势中无界累积).

    Args:
        close: 日频收盘价
        target_index: 输出的日频index (与目标仓位对齐)
        spacing_pct: 网格间距 (价格百分比)
        step: 每档库存 (占NAV比例)
        max_grids: 最大档数 (库存上限 = max_grids * step)

    Returns:
        日频库存序列 (与target_index对齐, 已含执行时滞)
    """
    w_close = close.resample("W-FRI").last().dropna()
    anchor = None
    inventories = []
    for dt, px in w_close.items():
        if anchor is None or np.isnan(px):
            anchor = px
            inventories.append(0.0)
            continue
        spacing = px * spacing_pct
        n_below = math.floor((anchor - px) / spacing)
        inv = float(np.clip(n_below * step, 0, max_grids * step))
        if abs(px - anchor) > (max_grids + 1) * spacing:
            anchor = px
            inv = 0.0
        inventories.append(inv)
    inv_w = pd.Series(inventories, index=w_close.index)
    return inv_w.reindex(target_index, method="ffill").shift(1).fillna(0.0)


def smooth_execution(
    pos_target: pd.Series,
    close: pd.Series,
    cap: float = 0.20,
    emergency: float = -0.05,
) -> pd.Series:
    """日平滑执行: 仓位向目标限速收敛; 单日暴跌触发应急立即压向0.

    Args:
        pos_target: 目标仓位 (含网格库存)
        close: 日频收盘价 (计算应急触发的当日收益)
        cap: 每日最大调仓幅度
        emergency: 单日跌幅应急阈值 (None=关闭)

    Returns:
        实际执行仓位序列
    """
    ret = close.pct_change()
    out = pos_target.copy()
    prev = pos_target.iloc[0]
    for i in range(len(pos_target)):
        tgt = pos_target.iloc[i]
        gap = tgt - prev
        if abs(gap) <= cap:
            new = tgt
        else:
            new = prev + cap if gap > 0 else prev - cap
        if emergency is not None and ret.iloc[i] < emergency:
            new = min(new, 0.0)
        out.iloc[i] = new
        prev = new
    return out


def build_full_stack(
    L: float = 1.5,
    first_cap: float = -0.05,
    second_cap: float = -0.05,
    second_hard: float = -0.20,
    second_soft: float = -0.12,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    deadband: float = 0.05,
    core_cap: float = 1.25,
    grid_sp: float = 0.02,
    grid_step: float = 0.0833,
    grid_tiers: int = 3,
    budget: float = 1.5,
    smooth_cap: float = 0.20,
    emergency: float = -0.05,
) -> pd.Series:
    """全执行栈 (预算制, B3参数): 链 → 核心裁剪 → +周网格库存 → 总封顶 → 日平滑 → 应急.

    杠杆预算分配: 总敞口硬封顶 budget (默认1.5),
    核心信号最多用 core_cap (默认1.25), 网格库存吃闲置预算 (≤0.25).
    """
    close, _ = load_au_ohlcv()
    chain = build_chain(
        L=L, first_cap=first_cap, second_cap=second_cap,
        second_hard=second_hard, second_soft=second_soft,
        vol_cap=vol_cap, cooldown_days=cooldown_days, deadband=deadband,
    )
    core = chain.clip(0, core_cap)
    grid = weekly_grid_inventory(
        close, chain.index,
        spacing_pct=grid_sp, step=grid_step, max_grids=grid_tiers,
    )
    pos = (core + grid).clip(0, budget)
    return smooth_execution(pos, close, cap=smooth_cap, emergency=emergency)


def o2o_returns(
    pos: pd.Series, open_price: pd.Series, cost_bps: float = COST_BPS
) -> pd.Series:
    """持仓 → 策略日收益 (open-to-open, shift(1), 扣成本)."""
    o = open_price.reindex(pos.index)
    ret_fwd = o.pct_change().shift(-1)
    sr = (pos.shift(1) * ret_fwd).dropna()
    cost = pos.diff().abs() * cost_bps / 10000
    return (sr - cost).dropna()


def perf_metrics(returns: pd.Series) -> dict:
    """六指标 + nav + 回撤谷底日期. 空数据安全."""
    empty = {
        "cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan,
        "calmar": np.nan, "total": np.nan, "mdd_date": None,
        "nav": pd.Series(dtype=float),
    }
    if returns is None or len(returns) < 2:
        return empty
    nav = (1 + returns).cumprod()
    n = len(returns)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    vol = returns.std() * np.sqrt(252)
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": cagr / vol if vol > 0 else np.nan,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd != 0 else np.nan,
        "total": nav.iloc[-1] - 1,  # 期间真实总收益 (nav[-1]/nav[0]-1 会吞掉首日收益)
        "mdd_date": dd.idxmin(),
        "nav": nav,
    }


def yearly_table(returns: pd.Series) -> pd.DataFrame:
    """逐年: 收益 / 波动 / Sharpe / MaxDD."""
    if returns is None or len(returns) < 2:
        return pd.DataFrame()
    rows = []
    for y in sorted(set(returns.index.year)):
        r = returns[returns.index.year == y]
        nav_y = (1 + r).cumprod()
        n = len(r)
        cagr = nav_y.iloc[-1] ** (252 / n) - 1
        vol = r.std() * np.sqrt(252)
        dd = (nav_y / nav_y.cummax() - 1).min()
        rows.append({
            "year": y, "return": nav_y.iloc[-1] - 1,
            "vol": vol, "sharpe": cagr / vol if vol > 0 else np.nan, "maxdd": dd,
        })
    return pd.DataFrame(rows).set_index("year")


SEGMENTS_2026 = {
    "2026-03崩盘": ("2026-03-01", "2026-03-31"),
    "2026-08暴跌": ("2026-08-25", "2026-08-31"),
    "2026 YTD": ("2026-01-01", None),
}


def segment_table(
    pos: pd.Series,
    open_price: pd.Series,
    segments: dict | None = None,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """事件段收益表. 自动跳过数据范围外的窗口."""
    if segments is None:
        segments = SEGMENTS_2026
    rows = []
    for name, (s, e) in segments.items():
        e = e or pos.index[-1]
        sub = pos[(pos.index >= s) & (pos.index <= e)]
        if len(sub) < 2:
            continue
        r = o2o_returns(sub, open_price, cost_bps)
        if len(r) == 0:
            continue
        rows.append({"segment": name, "return": (1 + r).prod() - 1, "days": len(r)})
    return pd.DataFrame(rows).set_index("segment")


def print_compare(configs: dict[str, pd.Series], open_price: pd.Series | None = None):
    """多配置对比打印: 全期六指标 + 2026YTD."""
    if open_price is None:
        _, open_price = load_au_ohlcv()
    header = (
        f'{"配置":<34} {"CAGR":>8} {"Sharpe":>7} {"MaxDD":>8} '
        f'{"Calmar":>7} {"26YTD":>8} {"26MDD":>8}'
    )
    print(header)
    print("-" * len(header.expandtabs()))
    for name, pos in configs.items():
        full = o2o_returns(pos, open_price)
        m = perf_metrics(full)
        m26 = perf_metrics(full[full.index >= "2026-01-01"])
        print(
            f"{name:<34} {m['cagr']:>+8.2%} {m['sharpe']:>7.2f} "
            f"{m['mdd']:>+8.2%} {m['calmar']:>7.2f} "
            f"{m26['total']:>+8.2%} {m26['mdd']:>+8.2%}"
        )


def _fmt_params(**kw):
    parts = []
    for k, v in kw.items():
        parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return "/".join(parts)


def main():
    parser = argparse.ArgumentParser(description="黄金CTA O2O回测分析")
    parser.add_argument("--L", type=float, default=None)
    parser.add_argument("--fc", type=float, default=None,
                        help="一层daily_cap")
    parser.add_argument("--sc", type=float, default=None,
                        help="二层daily_cap")
    parser.add_argument("--hard", type=float, default=None,
                        help="二层hard_cap")
    parser.add_argument("--soft", type=float, default=None,
                        help="二层soft_cap")
    parser.add_argument("--stack", action="store_true",
                        help="全执行栈模式 (链+预算制网格+日平滑+应急, B3参数)")
    parser.add_argument("--core-cap", type=float, default=DEFAULTS_STACK["core_cap"],
                        help="核心仓位上限")
    parser.add_argument("--sp", type=float, default=DEFAULTS_STACK["grid_sp"],
                        help="网格间距")
    parser.add_argument("--step", type=float, default=DEFAULTS_STACK["grid_step"],
                        help="网格每档仓位")
    parser.add_argument("--tiers", type=int, default=DEFAULTS_STACK["grid_tiers"],
                        help="网格最大档数")
    parser.add_argument("--budget", type=float, default=DEFAULTS_STACK["budget"],
                        help="总仓位上限")
    parser.add_argument("--smooth-cap", type=float, default=DEFAULTS_STACK["smooth_cap"],
                        help="日平滑限速")
    parser.add_argument("--no-emergency", action="store_true",
                        help="关闭暴跌应急通道")
    parser.add_argument("--compare", action="store_true", help="与生产现状对比")
    args = parser.parse_args()

    close, open_price = load_au_ohlcv()

    # 链层参数: stack模式用DEFAULTS_STACK, chain模式用DEFAULTS
    src = DEFAULTS_STACK if args.stack else DEFAULTS
    args.L = args.L if args.L is not None else src["L"]
    args.fc = args.fc if args.fc is not None else src["first_cap"]
    args.sc = args.sc if args.sc is not None else src["second_cap"]
    args.hard = args.hard if args.hard is not None else src["second_hard"]
    args.soft = args.soft if args.soft is not None else src["second_soft"]

    if args.stack:
        pos = build_full_stack(
            L=args.L, first_cap=args.fc, second_cap=args.sc,
            second_hard=args.hard, second_soft=args.soft,
            core_cap=args.core_cap, grid_sp=args.sp, grid_step=args.step,
            grid_tiers=args.tiers, budget=args.budget,
            smooth_cap=args.smooth_cap,
            emergency=None if args.no_emergency else DEFAULTS_STACK["emergency"],
        )
        label = (f"全栈 L={args.L:g} 链={args.fc:.0%}/{args.sc:.0%}/{args.hard:.0%} "
                 f"核心≤{args.core_cap:g}+网格{args.step:g}×{args.tiers}档 "
                 f"sp={args.sp:.1%} 上限{args.budget:g} 平滑{args.smooth_cap:g}")
    else:
        kw = dict(L=args.L, first_cap=args.fc, second_cap=args.sc,
                  second_hard=args.hard, second_soft=args.soft)
        pos = build_chain(**kw)
        label = f"L={args.L:g} fc={args.fc:.0%} sc={args.sc:.0%} hard={args.hard:.0%}"

    if args.compare:
        configs = {"现状(生产)": build_chain(), f"被测({label})": pos}
        if args.stack:
            configs["全栈B3(默认参数)"] = build_full_stack()
        print_compare(configs, open_price)
    else:
        print(f"=== 配置: {label} ===")
        m = perf_metrics(o2o_returns(pos, open_price))
        print(f"区间: {m['nav'].index[0].date()} ~ {m['nav'].index[-1].date()}, "
              f"MaxDD谷底: {m['mdd_date'].date()}")
        print(f"CAGR={m['cagr']:+.2%} Sharpe={m['sharpe']:.2f} "
              f"MaxDD={m['mdd']:+.2%} Calmar={m['calmar']:.2f} 累计={m['total']:+.1%}")
        print("\n--- 逐年 ---")
        print(yearly_table(o2o_returns(pos, open_price)).to_string(
            float_format=lambda x: f"{x:+.3f}"))
        print("\n--- 事件段 ---")
        print(segment_table(pos, open_price).to_string(
            float_format=lambda x: f"{x:+.3f}"))
        print(f"\n最新仓位 ({pos.index[-1].date()}): {pos.iloc[-1]:+.4f}")


if __name__ == "__main__":
    main()
