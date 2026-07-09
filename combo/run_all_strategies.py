# coding=utf-8
"""统一 ETF 池对比回测 — v3 / v4 / v5 公平对比.

所有策略使用相同的 52 只 ETF 池 (44 主池 + 8 SmartBeta).
v4 的 style/smart_beta 子策略仍使用各自的子集 (设计如此),
但因子择时可以利用完整 52 只池.

输出:
- 各策略全期 + OOS 指标
- 相关性矩阵
- 组合推荐
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combo.load_unified_data import load_unified_data, ALL_52, SMARTBETA_8, MAIN_44
from QuantNodes.strategy.momentum_etf_rotation.common.universe import ETFPool, Category, ETFMeta
from QuantNodes.strategy.momentum_etf_rotation.v3.multi_strategy_v3 import (
    MultiStrategyConfig,
    run_multi_strategy_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.multi_strategy_v4 import (
    V4Config,
    run_v4_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v5 import (
    IndustryRotationV5Config,
    IndustryRotationV5SubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v5_1 import (
    IndustryRotationV5_1Config,
    IndustryRotationV5_1SubStrategy,
    inverse_vol_weights_v5_1,
)

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"
OOS_START = "2022-01-01"


# ============================================================
# 指标计算
# ============================================================
def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav):
    pk = nav.cummax()
    return float((nav / pk - 1.0).min())


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
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
        "final": float(nav.iloc[-1]),
    }


def yearly_returns(nav):
    yr = nav.resample("YE").last() / nav.resample("YE").first() - 1
    return {d.year: float(v) for d, v in yr.items()}


# ============================================================
# 扩展 ETFPool (加入 SmartBeta)
# ============================================================
def make_expanded_pool():
    """构建包含 52 只 ETF 的扩展池."""
    from QuantNodes.strategy.momentum_etf_rotation.common.universe import _DEFAULT_ETFS, DEFAULT_POOL

    # SmartBeta ETF 的分类
    SB_METAS = [
        ETFMeta(code="510880", name="华泰柏瑞红利", category=Category.A_BROAD, index_code="红利", liquidity_rank=1),
        ETFMeta(code="512890", name="红利低波", category=Category.A_SECTOR, index_code="红利低波", liquidity_rank=1),
        ETFMeta(code="512260", name="300低波", category=Category.A_SECTOR, index_code="低波", liquidity_rank=1),
        ETFMeta(code="515900", name="中证质量", category=Category.A_SECTOR, index_code="质量", liquidity_rank=1),
        ETFMeta(code="512040", name="国泰价值", category=Category.A_SECTOR, index_code="价值", liquidity_rank=1),
        ETFMeta(code="159786", name="现金流", category=Category.A_SECTOR, index_code="现金流", liquidity_rank=1),
        ETFMeta(code="515080", name="中信红利", category=Category.A_SECTOR, index_code="红利100", liquidity_rank=1),
        ETFMeta(code="515100", name="红利低波100", category=Category.A_SECTOR, index_code="红利低波100", liquidity_rank=1),
    ]

    existing_codes = set(DEFAULT_POOL.codes)
    members = list(DEFAULT_POOL.members)
    for m in SB_METAS:
        if m.code not in existing_codes:
            members.append(m)
            existing_codes.add(m.code)

    pool = ETFPool(tuple(members))
    return pool


# ============================================================
# v5 回测 (与 v5_backtest.py 相同逻辑, 等权 — 保留 v5 旧实现)
# ============================================================
def backtest_v5(panel_close, panel_ohlcv, top_n=5):
    """用 close 面板 + OHLCV 回测 v5 (等权, 论文做法, 保留旧实现)."""
    dates = panel_close.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    from QuantNodes.strategy.momentum_etf_rotation.v5 import (
        compute_all_factors_panel,
        compute_composite_factor,
    )
    cfg = IndustryRotationV5Config(top_n=top_n)
    factor_panel = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)

    nav = np.ones(len(dates))
    last_weights = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            valid = [c for c in composite.index if c in panel_close.columns]
            composite = composite[valid]
            if len(composite) >= top_n:
                top = composite.nlargest(top_n)
                last_weights = {code: 1.0 / top_n for code in top.index}

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v5")


# ============================================================
# v5.1 回测 (逆波动率加权 — Stage 25 升级版)
# ============================================================
def backtest_v5_1(panel_close, panel_ohlcv, top_n=5):
    """v5.1: 11 量价因子 + 逆波动率加权 (与 v1/v3 一致)."""
    dates = panel_close.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    from QuantNodes.strategy.momentum_etf_rotation.v5 import (
        compute_all_factors_panel,
        compute_composite_factor,
    )
    cfg = IndustryRotationV5_1Config(top_n=top_n)
    factor_panel = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)

    nav = np.ones(len(dates))
    last_weights = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            valid = [c for c in composite.index if c in panel_close.columns]
            composite = composite[valid]
            if len(composite) >= top_n:
                top = composite.nlargest(top_n)
                chosen = list(top.index)
                # 逆波动率加权 (与 v1/v3 一致)
                last_weights = inverse_vol_weights_v5_1(
                    panel_close, chosen, date, cfg.vol_window, cfg.vol_floor,
                )
                # max_weight 约束 (默认 0.30)
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
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v5_1")


# ============================================================
# v4 回测 (使用统一面板)
# ============================================================
def backtest_v4_unified(panel_close):
    """v4 回测, 使用统一 close 面板 (v4 子策略只用 12 只)."""
    # v4 需要 12 只 ETF 的面板
    from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES
    v4_cols = [c for c in ALL_V4_CODES if c in panel_close.columns]
    v4_panel = panel_close[v4_cols].copy()

    cfg = V4Config(mode="v4C_combo", style_enabled=True, smart_beta_enabled=True)
    result = run_v4_backtest(v4_panel, cfg)
    return result.nav


def backtest_v4_style(panel_close):
    """v4 风格轮动 only."""
    from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES
    v4_cols = [c for c in ALL_V4_CODES if c in panel_close.columns]
    v4_panel = panel_close[v4_cols].copy()

    cfg = V4Config(mode="v4A_style", style_enabled=True, smart_beta_enabled=False)
    result = run_v4_backtest(v4_panel, cfg)
    return result.nav


def backtest_v4_factor(panel_close):
    """v4 因子择时 only, 修复 warmup 期 IC 为空导致权重归零的 bug."""
    from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES
    from QuantNodes.strategy.momentum_etf_rotation.v4.factor_timing_v4 import (
        compute_factor_weights,
        compute_strategy_weights,
    )

    v4_cols = [c for c in ALL_V4_CODES if c in panel_close.columns]
    v4_panel = panel_close[v4_cols].copy()

    # 跑 v4 combo (style+smart_beta), 然后叠加因子择时调整子策略权重
    cfg = V4Config(mode="v4C_combo", style_enabled=True, smart_beta_enabled=True)
    result = run_v4_backtest(v4_panel, cfg)

    # 如果需要因子择时, 重新跑带因子择时的版本, 但修复 warmup bug
    cfg2 = V4Config(mode="v4D_ic", style_enabled=True, smart_beta_enabled=True,
                    factor_timing_enabled=True)

    # 手动跑因子择时回测, 修复 warmup 期权重归零
    import numpy as np
    from QuantNodes.strategy.momentum_etf_rotation.v4.multi_strategy_v4 import (
        _get_rebal_dates, _apply_max_weight, _combine_sub_results, _performance_metrics,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import StyleRotationSubStrategy
    from QuantNodes.strategy.momentum_etf_rotation.v4.smart_beta_v4 import SmartBetaSubStrategy
    from QuantNodes.strategy.momentum_etf_rotation.v4.sub_strategy_v4 import SubStrategyResult

    panel = v4_panel.dropna(how="all")
    dates = panel.index
    rebal_dates = _get_rebal_dates(panel, cfg2.main_rebal_freq)
    style_sub = StyleRotationSubStrategy(cfg2.style)
    sb_sub = SmartBetaSubStrategy(cfg2.smart_beta)

    ic_history = pd.DataFrame()
    from QuantNodes.strategy.momentum_etf_rotation.v4.factor_timing_v4 import backtest_factor_timing
    ic_history = backtest_factor_timing(panel, list(panel.columns), cfg2.factor_timing)

    sub_weights = {"style_rotation": 0.5, "smart_beta": 0.5}
    nav = np.ones(len(dates))
    weights_combined = {}
    last_sub_results = {}

    for i, date in enumerate(dates):
        if date in rebal_dates:
            sub_results = []
            r = style_sub.run_step(panel, date)
            sub_results.append(r)
            last_sub_results["style_rotation"] = r
            r = sb_sub.run_step(panel, date)
            sub_results.append(r)
            last_sub_results["smart_beta"] = r

            # 因子择时 (修复: IC 全0时保持默认权重)
            if not ic_history.empty:
                idx = ic_history.index.get_indexer([date], method="ffill")[0]
                if idx >= 0:
                    ic_dict = ic_history.iloc[idx].to_dict()
                    f_w = compute_factor_weights(
                        pd.DataFrame([ic_dict], index=[date]),
                        cfg2.factor_timing,
                    )
                    # 关键修复: 如果因子权重全为0 (warmup期), 保持默认
                    if sum(f_w.values()) > 0:
                        s_w = compute_strategy_weights(
                            f_w, cfg2.factor_timing.factor_to_strategy,
                        )
                        if s_w:
                            sub_weights = s_w
                            total = sum(sub_weights.values())
                            if total > 0:
                                sub_weights = {k: v / total for k, v in sub_weights.items()}

            combined = _combine_sub_results(sub_results, sub_weights)
            combined = _apply_max_weight(combined, cfg2.max_weight)
            total = sum(combined.values())
            if total > 0:
                combined = {k: v / total for k, v in combined.items()}

            cost = 0.0
            if i > 0 and weights_combined:
                all_codes = set(weights_combined.keys()) | set(combined.keys())
                turnover = sum(
                    abs(combined.get(c, 0) - weights_combined.get(c, 0))
                    for c in all_codes
                ) / 2
                cost = turnover * cfg2.cost_bps / 10000

            weights_combined = combined
            if i > 0:
                nav[i] = nav[i - 1] * (1 - cost)
            else:
                nav[i] = 1.0
        else:
            if i > 0 and weights_combined:
                daily_ret = 0.0
                for code, w in weights_combined.items():
                    if code in panel.columns:
                        a, b = panel[code].iloc[i], panel[code].iloc[i - 1]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

    return pd.Series(nav, index=dates, name="v4_factor")


# ============================================================
# 主对比
# ============================================================
def main():
    print("=" * 70)
    print("统一 ETF 池对比回测 (52 只 ETF)")
    print("=" * 70)

    # 1. 加载数据
    data = load_unified_data(START, END)
    close_52 = data.close_52
    ohlcv_44 = data.ohlcv_44

    # 2. 构建扩展池
    pool_52 = make_expanded_pool()
    print(f"\n池大小: {len(pool_52.codes)} 只 ETF")

    # 3. v3 回测 (52 只统一池)
    print("\n" + "=" * 50)
    print("1. v3 Baseline (52 只 ETF)")
    print("=" * 50)
    cfg_v3 = MultiStrategyConfig(
        weight_method="equal",
        a_share_total=5,  # 放宽 A 股限制 (52 只池更大)
        max_weight=0.15,
    )
    result_v3 = run_multi_strategy_backtest(close_52, pool_52, cfg_v3)
    n_v3 = result_v3.nav

    # 4. v4 回测 (子策略用各自 12 只)
    print("\n" + "=" * 50)
    print("2. v4 Style (12 只 ETF)")
    print("=" * 50)
    n_v4s = backtest_v4_style(close_52)

    print("\n" + "=" * 50)
    print("3. v4 Factor (12 只 ETF)")
    print("=" * 50)
    n_v4f = backtest_v4_factor(close_52)

    print("\n" + "=" * 50)
    print("4. v4 Combo (12 只 ETF)")
    print("=" * 50)
    n_v4c = backtest_v4_unified(close_52)

    # 5. v5 回测 (44 只 OHLCV)
    print("\n" + "=" * 50)
    print("5. v5 量价 (44 只 ETF, 等权)")
    print("=" * 50)
    n_v5 = backtest_v5(close_52, ohlcv_44, top_n=5)

    # 5.1 v5.1 回测 (44 只 OHLCV, 逆波动率)
    print("\n" + "=" * 50)
    print("5.1 v5.1 量价 (44 只 ETF, 逆波动率加权)")
    print("=" * 50)
    n_v5_1 = backtest_v5_1(close_52, ohlcv_44, top_n=5)

    # ============================================================
    # 汇总
    # ============================================================
    navs = pd.DataFrame({
        "v3 baseline": n_v3,
        "v4 风格": n_v4s,
        "v4 因子": n_v4f,
        "v4 combo": n_v4c,
        "v5 量价": n_v5,
        "v5.1 量价 (逆波动)": n_v5_1,
    }).dropna()

    print("\n" + "=" * 70)
    print("全期业绩 (2018-2026)")
    print("=" * 70)
    print(f"{'策略':<20s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 65)
    for col in navs.columns:
        m = metrics(navs[col])
        print(f"{col:<20s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    # 年度收益
    print(f"\n{'年度收益':}")
    print(f"{'年份':>6s}", end="")
    for col in navs.columns:
        print(f" {col:>12s}", end="")
    print()
    for year in sorted(set(navs.index.year)):
        print(f"{year:>6d}", end="")
        for col in navs.columns:
            yr_nav = navs[col].loc[str(year)]
            if len(yr_nav) > 1:
                ret = yr_nav.iloc[-1] / yr_nav.iloc[0] - 1
                print(f" {ret*100:>+11.2f}%", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()

    # OOS
    oos = navs.loc[OOS_START:]
    print(f"\n{'='*70}")
    print(f"OOS 业绩 ({OOS_START} ~ {END})")
    print("=" * 70)
    print(f"{'策略':<20s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 65)
    for col in oos.columns:
        m = metrics(oos[col])
        print(f"{col:<20s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    # 相关性
    print(f"\n{'='*70}")
    print("日收益相关性")
    print("=" * 70)
    rets = navs.pct_change().dropna()
    corr = rets.corr()
    print(corr.round(2).to_string())

    # ============================================================
    # 组合推荐
    # ============================================================
    print(f"\n{'='*70}")
    print("组合推荐 (统一池)")
    print("=" * 70)
    combos = {
        "v3 80% + v5 20%": 0.8 * navs["v3 baseline"] + 0.2 * navs["v5 量价"],
        "v3 80% + v5.1 20%": 0.8 * navs["v3 baseline"] + 0.2 * navs["v5.1 量价 (逆波动)"],
        "v3 70% + v5 30%": 0.7 * navs["v3 baseline"] + 0.3 * navs["v5 量价"],
        "v3 70% + v5.1 30%": 0.7 * navs["v3 baseline"] + 0.3 * navs["v5.1 量价 (逆波动)"],
        "v3 50% + v4f 25% + v5 25%": 0.5 * navs["v3 baseline"] + 0.25 * navs["v4 因子"] + 0.25 * navs["v5 量价"],
        "v3 50% + v4f 25% + v5.1 25%": 0.5 * navs["v3 baseline"] + 0.25 * navs["v4 因子"] + 0.25 * navs["v5.1 量价 (逆波动)"],
        "v3 33% + v4f 33% + v5 34%": 0.33 * navs["v3 baseline"] + 0.33 * navs["v4 因子"] + 0.34 * navs["v5 量价"],
        "v3 33% + v4f 33% + v5.1 34%": 0.33 * navs["v3 baseline"] + 0.33 * navs["v4 因子"] + 0.34 * navs["v5.1 量价 (逆波动)"],
        "v3 60% + v4c 20% + v5 20%": 0.6 * navs["v3 baseline"] + 0.2 * navs["v4 combo"] + 0.2 * navs["v5 量价"],
        "v3 60% + v4c 20% + v5.1 20%": 0.6 * navs["v3 baseline"] + 0.2 * navs["v4 combo"] + 0.2 * navs["v5.1 量价 (逆波动)"],
        "等权 5 策略": navs[["v3 baseline", "v4 风格", "v4 因子", "v4 combo", "v5 量价"]].mean(axis=1),
        "等权 6 策略 (含 v5.1)": navs.mean(axis=1),
    }

    print(f"\n全期:")
    print(f"{'组合':<30s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 75)
    for name, nav in combos.items():
        m = metrics(nav)
        print(f"{name:<30s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    oos_combos = {k: v.loc[OOS_START:] for k, v in combos.items()}
    print(f"\nOOS ({OOS_START}~{END}):")
    print(f"{'组合':<30s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 75)
    for name, nav in oos_combos.items():
        m = metrics(nav)
        print(f"{name:<30s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    # 保存
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    navs.to_parquet(out_dir / "combo_navs_unified52.parquet")
    print(f"\n[save] {out_dir / 'combo_navs_unified52.parquet'}")


if __name__ == "__main__":
    main()
