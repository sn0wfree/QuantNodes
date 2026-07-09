# coding=utf-8
"""v1-v5 统一 ETF 池 + 统一时间对比 (2018-2026).

统一口径:
- 时间: 2018-01-01 ~ 2026-06-30 (8.5y), OOS 2022-01-01 ~ 2026-06-30 (4.5y)
- 池: 52 只 (44 主池 + 8 SmartBeta)
- 数据: 前复权 close
- 调仓: 月末
- 单边成本: 5 bp (口径 A 启用, 口径 B 用各自配置)
- A 股 cap: 3 (CICC 规则)

两口径:
- A (裸 alpha): 全部 v1/v2 关掉 VT/TF/Cost, 只保留 5bp 成本 + CICC 规则
- B (生产配置): v1.0 保留 VT/Cost, v2 保留 VT, v3 保留 cost
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(REPO))

from combo.load_unified_data import load_unified_data
from QuantNodes.strategy.momentum_etf_rotation.common.universe import (
    Category, ETFMeta, ETFPool, DEFAULT_POOL,
)
from QuantNodes.strategy.momentum_etf_rotation.portfolio import (
    RotationConfig, DiversificationCaps, TrendFilter, VolTargeting, CostModel,
)
from QuantNodes.strategy.momentum_etf_rotation.backtest import (
    BacktestConfig, run_rotation_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v3.multi_strategy_v3 import (
    MultiStrategyConfig, run_multi_strategy_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.multi_strategy_v4 import (
    V4Config, run_v4_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES

START = "2018-01-01"
END = "2026-06-30"
OOS_START = "2022-01-01"
COST_BPS = 5.0  # 5 bp 单边 (口径 A)

# ============================================================
# 工具函数
# ============================================================
def ann_return(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return 0.0
    r = valid.iloc[-1] / valid.iloc[0]
    n = (valid.index[-1] - valid.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return 0.0
    pk = valid.cummax()
    return float((valid / pk - 1.0).min())


def sharpe(rets):
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def metrics(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "final": float(nav.iloc[-1]) if pd.notna(nav.iloc[-1]) else 0.0}
    rets = valid.pct_change().dropna()
    ar = ann_return(valid)
    dd = max_dd(valid)
    return {
        "ann_return": ar,
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
        "final": float(valid.iloc[-1]),
    }


def trim_flat_prefix(nav_dict):
    """每条策略切掉前段 NAV=1.0 的 flat 期, 从首次交易日归一化."""
    out = {}
    for name, nav in nav_dict.items():
        non_one = nav[nav != 1.0]
        if len(non_one) > 0:
            ft = non_one.index[0]
            trimmed = nav.loc[ft:]
            out[name] = trimmed / trimmed.iloc[0]
        else:
            out[name] = nav / nav.iloc[0]
    return out


def make_expanded_pool():
    """52 只 ETF 扩展池 (44 主池 + 8 SmartBeta)."""
    SB_METAS = [
        ETFMeta(code="510880", name="华泰柏瑞红利", category=Category.A_BROAD,
                index_code="红利", liquidity_rank=1),
        ETFMeta(code="512890", name="红利低波", category=Category.A_SECTOR,
                index_code="红利低波", liquidity_rank=1),
        ETFMeta(code="512260", name="300低波", category=Category.A_SECTOR,
                index_code="低波", liquidity_rank=1),
        ETFMeta(code="515900", name="中证质量", category=Category.A_SECTOR,
                index_code="质量", liquidity_rank=1),
        ETFMeta(code="512040", name="国泰价值", category=Category.A_SECTOR,
                index_code="价值", liquidity_rank=1),
        ETFMeta(code="159786", name="现金流", category=Category.A_SECTOR,
                index_code="现金流", liquidity_rank=1),
        ETFMeta(code="515080", name="中信红利", category=Category.A_SECTOR,
                index_code="红利100", liquidity_rank=1),
        ETFMeta(code="515100", name="红利低波100", category=Category.A_SECTOR,
                index_code="红利低波100", liquidity_rank=1),
    ]
    existing = set(DEFAULT_POOL.codes)
    members = list(DEFAULT_POOL.members)
    for m in SB_METAS:
        if m.code not in existing:
            members.append(m)
            existing.add(m.code)
    return ETFPool(tuple(members))


# ============================================================
# v1.0 / v2 / v1 / v0.x 在 52 池中的回测
# ============================================================
def cicc_caps():
    """CICC 强制分散 (A 股 cap=3, HK=1, 必含商品+海外)."""
    return DiversificationCaps(
        a_share_broad=2, a_share_sector=2, a_share_total=3,
        hk=1, require_commodity=True, require_overseas=True,
    )


def v0_baseline_52(pool, etf_nav, cost_enabled=True):
    """v0.0 baseline (Stage 8) 在 52 池中: 144d 动量, 无增强."""
    cfg = RotationConfig(
        lookback=144, top_n=10,
        diversification=cicc_caps(),
        weight_method="inv_vol",
        vol_window=21,
        cost_model=CostModel(
            enabled=cost_enabled, commission_bp=COST_BPS,
            slippage_bp=(10 if cost_enabled else 0),
            impact_factor=(0.1 if cost_enabled else 0),
        ),
    )
    res = run_rotation_backtest(etf_nav, pool, BacktestConfig(rotation=cfg))
    return res.nav


def v0_1_vt_only_52(pool, etf_nav, cost_enabled=True):
    """v0.1 + VT (Stage 9-C) 在 52 池中."""
    cfg = RotationConfig(
        lookback=144, top_n=10,
        diversification=cicc_caps(),
        weight_method="inv_vol",
        vol_targeting=VolTargeting(enabled=True, target_vol=0.15,
                                    lookback=60, min_scale=0.3, max_scale=1.5),
        cost_model=CostModel(
            enabled=cost_enabled, commission_bp=COST_BPS,
            slippage_bp=(10 if cost_enabled else 0),
            impact_factor=(0.1 if cost_enabled else 0),
        ),
    )
    res = run_rotation_backtest(etf_nav, pool, BacktestConfig(rotation=cfg))
    return res.nav


def v0_2_tf_only_52(pool, etf_nav, cost_enabled=True):
    """v0.2 + TF (Stage 9-B) 在 52 池中."""
    cfg = RotationConfig(
        lookback=144, top_n=10,
        diversification=cicc_caps(),
        weight_method="inv_vol",
        trend_filter=TrendFilter(enabled=True, benchmark_code="510300",
                                  ma_window=200, exposure_bear=0.7,
                                  bond_code="511260"),
        cost_model=CostModel(
            enabled=cost_enabled, commission_bp=COST_BPS,
            slippage_bp=(10 if cost_enabled else 0),
            impact_factor=(0.1 if cost_enabled else 0),
        ),
    )
    res = run_rotation_backtest(etf_nav, pool, BacktestConfig(rotation=cfg))
    return res.nav


def v1_0_hybrid_52(pool, etf_nav, cost_enabled=True):
    """v1.0 locked: 斜率×R² 混合 + VT + Cost (Stage 12A)."""
    cfg = RotationConfig(
        lookback=90, top_n=10,
        diversification=cicc_caps(),
        weight_method="inv_vol",
        momentum_type="hybrid", momentum_fused_weight=0.5,
        vol_targeting=VolTargeting(enabled=True, target_vol=0.15,
                                    lookback=60, min_scale=0.3, max_scale=1.5),
        cost_model=CostModel(
            enabled=cost_enabled, commission_bp=5,
            slippage_bp=(10 if cost_enabled else 0),
            impact_factor=(0.1 if cost_enabled else 0),
        ),
    )
    res = run_rotation_backtest(etf_nav, pool, BacktestConfig(rotation=cfg))
    return res.nav


# ============================================================
# v3 在 52 池中 (CICC cap 替代为 3)
# ============================================================
def v3_52(close_52, pool_52):
    cfg = MultiStrategyConfig(
        weight_method="equal",
        a_share_total=3,
        max_weight=0.15,
    )
    res = run_multi_strategy_backtest(close_52, pool_52, cfg)
    return res.nav


# ============================================================
# v4 在 12 池中 (保持 v4 设计意图)
# ============================================================
def v4_style(close_52):
    from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES
    v4_cols = [c for c in ALL_V4_CODES if c in close_52.columns]
    v4_panel = close_52[v4_cols].copy()
    cfg = V4Config(mode="v4A_style", style_enabled=True, smart_beta_enabled=False)
    res = run_v4_backtest(v4_panel, cfg)
    return res.nav


def v4_factor_patched(close_52):
    """v4 因子择时: 修复 warmup 期 IC 为空导致权重归零的 bug."""
    import numpy as np
    from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import ALL_V4_CODES
    from QuantNodes.strategy.momentum_etf_rotation.v4.factor_timing_v4 import (
        compute_factor_weights, compute_strategy_weights, backtest_factor_timing,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v4.multi_strategy_v4 import (
        _get_rebal_dates, _apply_max_weight, _combine_sub_results,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import StyleRotationSubStrategy
    from QuantNodes.strategy.momentum_etf_rotation.v4.smart_beta_v4 import SmartBetaSubStrategy

    v4_cols = [c for c in ALL_V4_CODES if c in close_52.columns]
    panel = close_52[v4_cols].dropna(how="all").copy()
    dates = panel.index
    rebal_dates = _get_rebal_dates(panel, "ME")

    cfg = V4Config(mode="v4D_ic", style_enabled=True, smart_beta_enabled=True,
                   factor_timing_enabled=True)
    style_sub = StyleRotationSubStrategy(cfg.style)
    sb_sub = SmartBetaSubStrategy(cfg.smart_beta)
    ic_history = backtest_factor_timing(panel, list(panel.columns), cfg.factor_timing)

    sub_weights = {"style_rotation": 0.5, "smart_beta": 0.5}
    nav = np.ones(len(dates))
    weights_combined = {}

    for i, date in enumerate(dates):
        if date in rebal_dates:
            sub_results = []
            r1 = style_sub.run_step(panel, date)
            sub_results.append(r1)
            r2 = sb_sub.run_step(panel, date)
            sub_results.append(r2)

            if not ic_history.empty:
                idx = ic_history.index.get_indexer([date], method="ffill")[0]
                if idx >= 0:
                    ic_dict = ic_history.iloc[idx].to_dict()
                    f_w = compute_factor_weights(
                        pd.DataFrame([ic_dict], index=[date]), cfg.factor_timing,
                    )
                    if sum(f_w.values()) > 0:
                        s_w = compute_strategy_weights(
                            f_w, cfg.factor_timing.factor_to_strategy,
                        )
                        if s_w:
                            sub_weights = s_w
                            total = sum(sub_weights.values())
                            if total > 0:
                                sub_weights = {k: v / total for k, v in sub_weights.items()}

            combined = _combine_sub_results(sub_results, sub_weights)
            combined = _apply_max_weight(combined, cfg.max_weight)
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
                cost = turnover * COST_BPS / 10000

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
# v5 在 OHLCV 面板中 (等权, 保留旧实现)
# ============================================================
def v5_52(close_52, ohlcv_44, top_n=5):
    from QuantNodes.strategy.momentum_etf_rotation.v5 import (
        compute_all_factors_panel, compute_composite_factor,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v5.industry_rotation_v5 import (
        IndustryRotationV5Config,
    )

    dates = close_52.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    cfg = IndustryRotationV5Config(top_n=top_n)
    factor_panel = compute_all_factors_panel(ohlcv_44, cfg.factor_cfg)

    nav = np.ones(len(dates))
    last_weights = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            valid = [c for c in composite.index if c in close_52.columns]
            composite = composite[valid]
            if len(composite) >= top_n:
                top = composite.nlargest(top_n)
                last_weights = {code: 1.0 / top_n for code in top.index}

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in close_52.columns:
                    p_t = close_52[code].iloc[i]
                    p_prev = close_52[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v5")


# ============================================================
# v5.1 在 OHLCV 面板中 (逆波动率加权)
# ============================================================
def v5_1_52(close_52, ohlcv_44, top_n=5):
    from QuantNodes.strategy.momentum_etf_rotation.v5 import (
        compute_all_factors_panel, compute_composite_factor,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v5_1 import (
        IndustryRotationV5_1Config, inverse_vol_weights_v5_1,
    )

    dates = close_52.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    cfg = IndustryRotationV5_1Config(top_n=top_n)
    factor_panel = compute_all_factors_panel(ohlcv_44, cfg.factor_cfg)

    nav = np.ones(len(dates))
    last_weights = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            valid = [c for c in composite.index if c in close_52.columns]
            composite = composite[valid]
            if len(composite) >= top_n:
                top = composite.nlargest(top_n)
                chosen = list(top.index)
                # 逆波动率加权
                last_weights = inverse_vol_weights_v5_1(
                    close_52, chosen, date, cfg.vol_window, cfg.vol_floor,
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
                if code in close_52.columns:
                    p_t = close_52[code].iloc[i]
                    p_prev = close_52[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v5_1")


# ============================================================
# 主对比
# ============================================================
def run_caliber(cost_enabled: bool, label: str):
    """跑一个口径 (A: cost on, B: cost off)."""
    print("\n" + "=" * 70)
    print(f"口径 {label} (成本 {'开' if cost_enabled else '关'})")
    print("=" * 70)

    data = load_unified_data(START, END)
    close_52 = data.close_52
    ohlcv_44 = data.ohlcv_44
    pool_52 = make_expanded_pool()
    print(f"池: {len(pool_52.codes)} 只 | 时间: {START} ~ {END}")

    navs = {}

    print("[1] v0.0 baseline (144d price mom)...")
    navs["v0.0 baseline"] = v0_baseline_52(pool_52, close_52, cost_enabled)

    print("[2] v0.1 + VT (Stage 9-C)...")
    navs["v0.1 +VT"] = v0_1_vt_only_52(pool_52, close_52, cost_enabled)

    print("[3] v0.2 + TF (Stage 9-B)...")
    navs["v0.2 +TF"] = v0_2_tf_only_52(pool_52, close_52, cost_enabled)

    print("[4] v1.0 locked (斜率×R² + VT + Cost)...")
    navs["v1.0 locked"] = v1_0_hybrid_52(pool_52, close_52, cost_enabled)

    print("[5] v3 multi-strategy (52 池, cap=3)...")
    navs["v3 (52 池)"] = v3_52(close_52, pool_52)

    print("[6] v4 style (12 池)...")
    navs["v4 style"] = v4_style(close_52)

    print("[7] v4 factor (12 池, patched)...")
    navs["v4 factor"] = v4_factor_patched(close_52)

    print("[8] v5 量价 (44 OHLCV)...")
    navs["v5 量价"] = v5_52(close_52, ohlcv_44, top_n=5)

    print("[8.1] v5.1 量价 (44 OHLCV, 逆波动)...")
    navs["v5.1 量价 (逆波动)"] = v5_1_52(close_52, ohlcv_44, top_n=5)

    navs_df = pd.DataFrame(navs)

    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"unified_v1v5_navs_cal{label}.parquet"
    navs_df.to_parquet(out_dir / fname)
    print(f"\n[save] {out_dir / fname}")

    # 全期 + OOS 指标
    print(f"\n--- 全期业绩 (2018-2026) ---")
    print(f"{'策略':<22s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 65)
    full_metrics = {}
    for col in navs_df.columns:
        m = metrics(navs_df[col])
        full_metrics[col] = m
        print(f"{col:<22s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    oos = navs_df.loc[OOS_START:]
    print(f"\n--- OOS 业绩 ({OOS_START} ~ {END}) ---")
    print(f"{'策略':<22s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 65)
    oos_metrics = {}
    for col in oos.columns:
        m = metrics(oos[col])
        oos_metrics[col] = m
        print(f"{col:<22s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    return navs_df, full_metrics, oos_metrics


def run_all():
    print("=" * 70)
    print("v1-v5 统一 ETF 池 (52) + 统一时间 (2018-2026) 对比")
    print("=" * 70)

    # 口径 A: 5bp 成本开启
    navs_A, full_A, oos_A = run_caliber(cost_enabled=True, label="A")

    # 口径 B: 全部关掉成本 (pure alpha)
    navs_B, full_B, oos_B = run_caliber(cost_enabled=False, label="B")

    # ============================================================
    # 双口径对比表
    # ============================================================
    print("\n" + "=" * 70)
    print("双口径 OOS Calmar 对比 (A: 5bp 成本, B: 无成本)")
    print("=" * 70)
    print(f"{'策略':<22s} {'OOS Calmar A':>14s} {'OOS Calmar B':>14s} {'差值':>8s}")
    print("-" * 60)
    for col in oos_A:
        a = oos_A[col]['calmar']
        b = oos_B[col]['calmar']
        diff = a - b
        print(f"{col:<22s} {a:>14.3f} {b:>14.3f} {diff:>+8.3f}")

    # 双口径全期
    print(f"\n--- 双口径全期 Calmar 对比 ---")
    print(f"{'策略':<22s} {'Calmar A':>10s} {'Calmar B':>10s} {'差值':>8s}")
    print("-" * 55)
    for col in full_A:
        a = full_A[col]['calmar']
        b = full_B[col]['calmar']
        diff = a - b
        print(f"{col:<22s} {a:>10.3f} {b:>10.3f} {diff:>+8.3f}")

    # 组合推荐 (口径 A)
    print(f"\n{'='*70}")
    print("组合推荐 (口径 A: 含成本)")
    print("=" * 70)
    combos = {
        "v1.0 80% + v5 20%": 0.8 * navs_A["v1.0 locked"] + 0.2 * navs_A["v5 量价"],
        "v1.0 80% + v5.1 20%": 0.8 * navs_A["v1.0 locked"] + 0.2 * navs_A["v5.1 量价 (逆波动)"],
        "v1.0 70% + v5 30%": 0.7 * navs_A["v1.0 locked"] + 0.3 * navs_A["v5 量价"],
        "v1.0 70% + v5.1 30%": 0.7 * navs_A["v1.0 locked"] + 0.3 * navs_A["v5.1 量价 (逆波动)"],
        "v3 50% + v5 50%":   0.5 * navs_A["v3 (52 池)"] + 0.5 * navs_A["v5 量价"],
        "v3 50% + v5.1 50%": 0.5 * navs_A["v3 (52 池)"] + 0.5 * navs_A["v5.1 量价 (逆波动)"],
        "v1.0 50% + v3 25% + v5 25%":
            0.5 * navs_A["v1.0 locked"] + 0.25 * navs_A["v3 (52 池)"] + 0.25 * navs_A["v5 量价"],
        "v1.0 50% + v3 25% + v5.1 25%":
            0.5 * navs_A["v1.0 locked"] + 0.25 * navs_A["v3 (52 池)"] + 0.25 * navs_A["v5.1 量价 (逆波动)"],
    }
    print(f"\n全期:")
    print(f"{'组合':<28s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 75)
    for name, nav in combos.items():
        m = metrics(nav)
        print(f"{name:<28s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    oos_combos = {k: v.loc[OOS_START:] for k, v in combos.items()}
    print(f"\nOOS:")
    print(f"{'组合':<28s} {'年化':>7s} {'波动':>7s} {'Sharpe':>7s} {'DD':>8s} {'Calmar':>7s}")
    print("-" * 75)
    for name, nav in oos_combos.items():
        m = metrics(nav)
        print(f"{name:<28s} {m['ann_return']*100:6.2f}% {m['ann_vol']*100:6.2f}% "
              f"{m['sharpe']:6.2f} {m['max_dd']*100:7.2f}% {m['calmar']:6.3f}")

    # 相关性
    print(f"\n{'='*70}")
    print("日收益相关性 (口径 A)")
    print("=" * 70)
    rets = navs_A.pct_change().dropna()
    print(rets.corr().round(2).to_string())


if __name__ == "__main__":
    run_all()
