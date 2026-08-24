# coding=utf-8
"""scripts/cmf/16_v6_3_comprehensive_test.py

v6.3 综合测试: 因子筛选 + 无前视回测 + bull_enhance参数 + 因子权重 + 对比
"""
import sys
sys.path.insert(0, ".")

import duckdb
import numpy as np
import pandas as pd
from QuantNodes.strategy.commodity_fundamental.copper.copper_factors import (
    compute_all_factors, _load_weekly_close, FACTOR_NAMES,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6_3 import (
    build_signal_v6_3, backtest_v6_3,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import (
    build_signal_v6, apply_bull_enhance, backtest_v6,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry_model import CarryModel

DB = "data/cmf/fundamental.duckdb"


def test1_factor_selection():
    """测试1: 因子筛选."""
    print("=" * 70)
    print("测试1: 因子筛选")
    print("=" * 70)

    con = duckdb.connect(DB, read_only=True)
    close_w = _load_weekly_close(con, "CU0").dropna()
    factors = compute_all_factors(con)

    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)
    ret_w = close_w.pct_change()

    # 单因子测试
    print("\n单因子Sharpe:")
    print("-" * 50)
    factor_sharpes = {}
    for col in factors.columns:
        pos = factors[col].rank(pct=True).fillna(0.5)
        strat = pos.shift(1) * ret_w - pos.diff().abs() * 5e-4
        strat = strat.dropna()
        m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
        s = strat[m]
        if len(s) > 0:
            nav = (1 + s).cumprod()
            ann = nav.iloc[-1] ** (52 / len(s)) - 1
            vol = s.std() * np.sqrt(52)
            sh = ann / vol if vol > 0 else 0
            factor_sharpes[col] = sh
            print(f"  {col:<15} Sharpe={sh:.3f}")

    # 按Sharpe排序
    sorted_factors = sorted(factor_sharpes.items(), key=lambda x: x[1], reverse=True)
    print(f"\n因子排序 (Sharpe):")
    for i, (name, sh) in enumerate(sorted_factors):
        print(f"  {i+1}. {name:<15} {sh:.3f}")

    # 测试不同因子组合
    print("\n因子组合测试:")
    print("-" * 70)
    print(f"{'组合':<40} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 70)

    combos = {
        "全部12因子": FACTOR_NAMES,
        "Top8 (去掉最弱4个)": [f for f, _ in sorted_factors[:8]],
        "Top6 (去掉最弱6个)": [f for f, _ in sorted_factors[:6]],
        "Top4 (去掉最弱8个)": [f for f, _ in sorted_factors[:4]],
        "宏观组": ["dxy_z", "pmi", "industry_dz"],
        "动量+宏观": ["mom_2w", "mom_13w", "dxy_z", "pmi"],
        "最强3因子": [f for f, _ in sorted_factors[:3]],
    }

    for name, cols in combos.items():
        valid_cols = [c for c in cols if c in factors.columns]
        if not valid_cols:
            continue
        pos = factors[valid_cols].mean(axis=1).rank(pct=True).fillna(0.5)
        strat = pos.shift(1) * ret_w - pos.diff().abs() * 5e-4
        strat = strat.dropna()
        m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
        s = strat[m]
        if len(s) > 0:
            nav = (1 + s).cumprod()
            ann = nav.iloc[-1] ** (52 / len(s)) - 1
            vol = s.std() * np.sqrt(52)
            sh = ann / vol if vol > 0 else 0
            dd = (nav / nav.cummax() - 1).min()
            print(f"{name:<40} {sh:<10.3f} {dd*100:<10.1f}")


def test2_no_lookahead():
    """测试2: 无前视回测."""
    print("\n" + "=" * 70)
    print("测试2: 无前视回测")
    print("=" * 70)

    con = duckdb.connect(DB, read_only=True)
    close_w = _load_weekly_close(con, "CU0").dropna()
    factors = compute_all_factors(con)

    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)
    ret_w = close_w.pct_change()

    min_history = 104
    positions = []

    for i in range(min_history, len(close_w)):
        current_date = close_w.index[i]

        # 只用历史数据
        factors_train = factors[factors.index <= current_date]
        close_train = close_w[close_w.index <= current_date]

        if len(factors_train) < 52:
            positions.append(0.5)
            continue

        # 等权合成
        pos_raw = factors_train.mean(axis=1).rank(pct=True).fillna(0.5)

        # bull_enhance
        mom26 = close_train.pct_change(26)
        bull = mom26 > 0.10

        # 增强条件
        from QuantNodes.strategy.commodity_fundamental.copper.copper_factors import (
            compute_lme_z, compute_carry, compute_cu_crude,
        )
        lme_z = compute_lme_z(con, close_train)
        carry = compute_carry(con, close_train)
        cu_crude_pos = compute_cu_crude(con, close_train)

        factor_score = (
            (lme_z < -0.5).astype(int) +
            (carry > 0.7).astype(int) +
            (cu_crude_pos > 0.6).astype(int)
        )
        enhance = bull & (factor_score >= 2)

        pos_enhanced = pos_raw.copy()
        pos_enhanced[enhance] = np.maximum(pos_enhanced[enhance], 0.9)

        # 波动率目标化
        ret_train = close_train.pct_change()
        vol_4w = ret_train.rolling(4).std() * np.sqrt(52)
        scale = (0.12 / vol_4w).fillna(1.0).clip(upper=1.0)
        pos_sized = (pos_enhanced * scale).clip(upper=1.0)

        positions.append(pos_sized.iloc[-1])

    # 回测
    pos_series = pd.Series(positions, index=close_w.index[min_history:])
    strat = pos_series.shift(1) * ret_w.reindex(pos_series.index) - pos_series.diff().abs() * 5e-4
    strat = strat.dropna()
    m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
    strat = strat[m]

    nav = (1 + strat).cumprod()
    ann = nav.iloc[-1] ** (52 / len(strat)) - 1
    vol = strat.std() * np.sqrt(52)
    sh = ann / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()

    print(f"\n无前视回测结果:")
    print(f"  年化: {ann*100:.1f}%")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  MaxDD: {dd*100:.1f}%")

    # 分年
    print(f"\n分年收益:")
    for y in range(2018, 2026):
        yn = nav[nav.index.year == y]
        if len(yn) > 1:
            ret = (yn.iloc[-1] / yn.iloc[0] - 1) * 100
            print(f"  {y}: {ret:.1f}%")

    # 对比有前视
    print(f"\n对比:")
    sig_lookahead = build_signal_v6_3("CU0", con)
    r_lookahead = backtest_v6_3(sig_lookahead)
    print(f"  有前视: ann={r_lookahead['ann_ret']*100:.1f}%  sh={r_lookahead['sharpe']:.2f}  dd={r_lookahead['max_dd']*100:.1f}%")
    print(f"  无前视: ann={ann*100:.1f}%  sh={sh:.2f}  dd={dd*100:.1f}%")


def test3_bull_enhance_params():
    """测试3: bull_enhance参数优化."""
    print("\n" + "=" * 70)
    print("测试3: bull_enhance参数优化")
    print("=" * 70)

    con = duckdb.connect(DB, read_only=True)

    print(f"\n{'threshold':<12} {'mom_thresh':<12} {'mom_weeks':<12} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 60)

    for thresh in [1, 2, 3]:
        for mt in [0.05, 0.10, 0.15]:
            for mw in [13, 20, 26]:
                sig = build_signal_v6_3(
                    "CU0", con,
                    enhance_threshold=thresh,
                    bull_mom_thresh=mt,
                    bull_mom_weeks=mw,
                )
                r = backtest_v6_3(sig)
                print(f"{thresh:<12} {mt:<12} {mw:<12} {r['sharpe']:<10.3f} {r['max_dd']*100:<10.1f}")


def test4_factor_weight_optimization():
    """测试4: 因子权重优化."""
    print("\n" + "=" * 70)
    print("测试4: 因子权重优化")
    print("=" * 70)

    con = duckdb.connect(DB, read_only=True)
    close_w = _load_weekly_close(con, "CU0").dropna()
    factors = compute_all_factors(con)

    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)
    ret_w = close_w.pct_change()

    # 单因子Sharpe用于权重分配
    factor_sharpes = {}
    for col in factors.columns:
        pos = factors[col].rank(pct=True).fillna(0.5)
        strat = pos.shift(1) * ret_w - pos.diff().abs() * 5e-4
        strat = strat.dropna()
        m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
        s = strat[m]
        if len(s) > 0:
            nav = (1 + s).cumprod()
            ann = nav.iloc[-1] ** (52 / len(s)) - 1
            vol = s.std() * np.sqrt(52)
            sh = ann / vol if vol > 0 else 0
            factor_sharpes[col] = max(sh, 0)  # 负Sharpe权重为0

    # 方法1: Sharpe加权
    total_sharpe = sum(factor_sharpes.values())
    weights_sharpe = {k: v / total_sharpe for k, v in factor_sharpes.items()}
    pos_sharpe = sum(factors[k].rank(pct=True).fillna(0.5) * w for k, w in weights_sharpe.items())
    strat_sharpe = pos_sharpe.shift(1) * ret_w - pos_sharpe.diff().abs() * 5e-4
    strat_sharpe = strat_sharpe.dropna()
    m = (strat_sharpe.index >= "2018-01-01") & (strat_sharpe.index <= "2025-12-31")
    s_sharpe = strat_sharpe[m]
    nav_sharpe = (1 + s_sharpe).cumprod()
    ann_sharpe = nav_sharpe.iloc[-1] ** (52 / len(s_sharpe)) - 1
    vol_sharpe = s_sharpe.std() * np.sqrt(52)
    sh_sharpe = ann_sharpe / vol_sharpe if vol_sharpe > 0 else 0
    dd_sharpe = (nav_sharpe / nav_sharpe.cummax() - 1).min()

    # 方法2: 等权 (基准)
    pos_equal = factors.mean(axis=1).rank(pct=True).fillna(0.5)
    strat_equal = pos_equal.shift(1) * ret_w - pos_equal.diff().abs() * 5e-4
    strat_equal = strat_equal.dropna()
    m = (strat_equal.index >= "2018-01-01") & (strat_equal.index <= "2025-12-31")
    s_equal = strat_equal[m]
    nav_equal = (1 + s_equal).cumprod()
    ann_equal = nav_equal.iloc[-1] ** (52 / len(s_equal)) - 1
    vol_equal = s_equal.std() * np.sqrt(52)
    sh_equal = ann_equal / vol_equal if vol_equal > 0 else 0
    dd_equal = (nav_equal / nav_equal.cummax() - 1).min()

    # 方法3: Top8因子Sharpe加权
    sorted_factors = sorted(factor_sharpes.items(), key=lambda x: x[1], reverse=True)[:8]
    top8_sharpe = sum(v for _, v in sorted_factors)
    weights_top8 = {k: v / top8_sharpe for k, v in sorted_factors}
    pos_top8 = sum(factors[k].rank(pct=True).fillna(0.5) * w for k, w in weights_top8.items())
    strat_top8 = pos_top8.shift(1) * ret_w - pos_top8.diff().abs() * 5e-4
    strat_top8 = strat_top8.dropna()
    m = (strat_top8.index >= "2018-01-01") & (strat_top8.index <= "2025-12-31")
    s_top8 = strat_top8[m]
    nav_top8 = (1 + s_top8).cumprod()
    ann_top8 = nav_top8.iloc[-1] ** (52 / len(s_top8)) - 1
    vol_top8 = s_top8.std() * np.sqrt(52)
    sh_top8 = ann_top8 / vol_top8 if vol_top8 > 0 else 0
    dd_top8 = (nav_top8 / nav_top8.cummax() - 1).min()

    print(f"\n因子权重优化结果:")
    print(f"{'方法':<25} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 50)
    print(f"{'等权 (基准)':<25} {sh_equal:<10.3f} {dd_equal*100:<10.1f}")
    print(f"{'Sharpe加权 (全部)':<25} {sh_sharpe:<10.3f} {dd_sharpe*100:<10.1f}")
    print(f"{'Sharpe加权 (Top8)':<25} {sh_top8:<10.3f} {dd_top8*100:<10.1f}")

    # 显示权重
    print(f"\nSharpe加权权重 (Top8):")
    for k, v in sorted(weights_top8.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:<15} {v:.3f}")


def test5_compare_v62_carry():
    """测试5: 与v6.2+carry对比."""
    print("\n" + "=" * 70)
    print("测试5: 与v6.2+carry对比")
    print("=" * 70)

    con = duckdb.connect(DB, read_only=True)

    # v6.3
    sig_v63 = build_signal_v6_3("CU0", con)
    r_v63 = backtest_v6_3(sig_v63)

    # v6.2
    sig6 = build_signal_v6("CU0", con)
    sig_e = apply_bull_enhance(sig6, con, threshold=2)
    sig_e["position"] = sig_e["position_sized"]
    r_v62 = backtest_v6(sig_e)

    # carry
    model_carry = CarryModel(z_window=26, mom_window=4, min_duration=6)
    sig_carry = model_carry.build_signal(con)
    r_carry = model_carry.backtest(sig_carry)

    # v6.2+carry 50/50
    import numpy as np
    p_v62 = sig_e["position_sized"].reindex(sig_carry.index).fillna(0)
    p_carry = sig_carry["position"].reindex(sig_e.index).fillna(0)
    common = p_v62.index.intersection(p_carry.index)
    combo = 0.5 * p_v62[common] + 0.5 * p_carry[common]

    close_w = _load_weekly_close(con, "CU0").reindex(common).fillna(method="ffill")
    ret_w = close_w.pct_change()
    strat = combo.shift(1) * ret_w - combo.diff().abs() * 5e-4
    strat = strat.dropna()
    m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
    s = strat[m]
    nav = (1 + s).cumprod()
    ann = nav.iloc[-1] ** (52 / len(s)) - 1
    vol = s.std() * np.sqrt(52)
    sh = ann / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()

    print(f"\n{'策略':<30} {'年化%':<10} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 60)
    print(f"{'v6.2':<30} {r_v62['ann_ret']*100:<10.1f} {r_v62['sharpe']:<10.2f} {r_v62['max_dd']*100:<10.1f}")
    print(f"{'carry(优化)':<30} {r_carry['ann_ret']*100:<10.1f} {r_carry['sharpe']:<10.2f} {r_carry['max_dd']*100:<10.1f}")
    print(f"{'v6.3':<30} {r_v63['ann_ret']*100:<10.1f} {r_v63['sharpe']:<10.2f} {r_v63['max_dd']*100:<10.1f}")
    print(f"{'v6.2+carry 50/50':<30} {ann*100:<10.1f} {sh:<10.2f} {dd*100:<10.1f}")

    # 分年对比
    print(f"\n分年对比:")
    print(f"{'年份':<8} {'v6.2':<12} {'carry':<12} {'v6.3':<12} {'v6.2+carry':<12}")
    print("-" * 55)

    nav_v62 = r_v62["nav"]
    nav_carry = r_carry["nav"]
    nav_v63 = r_v63["nav"]

    for y in range(2018, 2026):
        y62 = nav_v62[nav_v62.index.year == y]
        yc = nav_carry[nav_carry.index.year == y]
        y63 = nav_v63[nav_v63.index.year == y]
        ycombo = nav[nav.index.year == y]

        r62 = (y62.iloc[-1] / y62.iloc[0] - 1) * 100 if len(y62) > 1 else 0
        rc = (yc.iloc[-1] / yc.iloc[0] - 1) * 100 if len(yc) > 1 else 0
        r63 = (y63.iloc[-1] / y63.iloc[0] - 1) * 100 if len(y63) > 1 else 0
        rcombo = (ycombo.iloc[-1] / ycombo.iloc[0] - 1) * 100 if len(ycombo) > 1 else 0

        print(f"{y:<8} {r62:<12.1f} {rc:<12.1f} {r63:<12.1f} {rcombo:<12.1f}")


def main():
    print("v6.3 综合测试")
    print("=" * 70)

    test1_factor_selection()
    test2_no_lookahead()
    test3_bull_enhance_params()
    test4_factor_weight_optimization()
    test5_compare_v62_carry()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
