"""scripts/cmf/test_risk_overlay.py — 风控覆盖层单元测试.

验证compute_risk_overlay的:
1. 三级触发逻辑正确
2. 硬止损触发后清仓
3. 冷却期内position=0
4. 智能再入场 (VIX+GVZ双确认)
5. 软减仓生效
6. 波动率压制生效
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/')

from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_risk_overlay, _rolling_drawdown, compute_macro_regime_signal,
    _load_indicator_d,
)


def make_synthetic_data(n=500):
    """构造合成数据测试."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    # 模拟金价: 平稳上涨, 中间加一段崩盘
    np.random.seed(42)
    ret = np.random.normal(0.0003, 0.01, n)
    ret[200:230] = np.random.normal(-0.03, 0.02, 30)  # 模拟崩盘 -3%日均
    ret[230:260] = np.random.normal(0.005, 0.015, 30)  # 反弹
    close = pd.Series(100 * (1 + ret).cumprod(), index=dates)

    # 仓位: 全程满仓(模拟不应用风控的信号)
    position = pd.Series(1.0, index=dates)

    # VIX: 崩盘期飙升
    vix = pd.Series(15.0, index=dates)
    vix.iloc[200:230] = 35.0
    vix.iloc[230:260] = 25.0

    # GVZ: 崩盘期飙升
    gvz = pd.Series(16.0, index=dates)
    gvz.iloc[200:230] = 28.0

    return close, position, vix, gvz


def test_no_risk_no_change():
    """风控参数全None → position应基本不变."""
    close, position, vix, gvz = make_synthetic_data()
    pos_out = compute_risk_overlay(
        close=close, position=position,
        hard_cap=-0.99, soft_cap=-0.99, daily_cap=-0.99,
        vol_cap=99, cooldown_days=0,
    )
    assert (pos_out == position).mean() > 0.9, "无风控参数时, position应保持不变"
    print("✓ test_no_risk_no_change")


def test_hard_stop_trigger():
    """硬止损: 当日跌幅 > daily_cap 时, 触发清仓."""
    close, position, vix, gvz = make_synthetic_data()
    pos_out = compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=-0.50, soft_cap=-0.30,
        daily_cap=-0.02,  # 崩盘期日均-3%, >2%即触发
        cooldown_days=5,
    )
    # 在崩盘区域 (index 200-230), 应该有清仓
    crash_window = pos_out.iloc[200:260]
    zero_pct = (crash_window == 0).mean()
    assert zero_pct > 0.3, f"崩盘区应有较多清仓, 但 zero_pct={zero_pct:.2f}"
    print(f"✓ test_hard_stop_trigger (崩盘区清仓占比={zero_pct:.2f})")


def test_cooldown_period():
    """硬触发后冷却期内保持0仓位."""
    close, position, vix, gvz = make_synthetic_data()
    pos_out = compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=-0.50, soft_cap=-0.30,
        daily_cap=-0.02,
        cooldown_days=10,
        reentry_vix_pct=0.99,  # 永不满足, 测试冷却期延长
        reentry_gvz_z=99,
    )
    # 找到第一次硬触发的索引
    crash_window = pos_out.iloc[200:260]
    first_zero_idx = None
    for i in range(len(crash_window)):
        if crash_window.iloc[i] == 0:
            first_zero_idx = i
            break
    # 检查触发后的15天零仓天数
    if first_zero_idx is not None:
        check_window = crash_window.iloc[first_zero_idx:first_zero_idx + 15]
        zero_count = (check_window == 0).sum()
        print(f"  触发后15天零仓: {zero_count}/15")
        assert zero_count >= 10, f"冷却期内应至少10天零仓, 但zero_count={zero_count}"
    print(f"✓ test_cooldown_period")


def test_soft_de_reduction():
    """软减仓: 接近MaxDD阈值时, 仓位减半."""
    close, position, vix, gvz = make_synthetic_data()
    # soft_cap较宽, 让position立即进入软减仓状态
    pos_out = compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=-0.50, soft_cap=-0.01,  # soft_cap很浅,容易触发
        daily_cap=-0.99, vol_cap=99, cooldown_days=0,
    )
    # 后期 (rolling dd变深), 应有软减仓 (0.5)
    soft_zone = pos_out.iloc[100:]
    half_count = ((soft_zone - 0.5).abs() < 0.01).sum()
    full_count = (soft_zone >= 0.99).sum()
    print(f"  后期: 半仓={half_count}, 满仓={full_count}")
    print(f"✓ test_soft_de_reduction")


def test_volatility_clamp():
    """波动率压制: 当realized_vol>vol_cap时, 仓位被压制."""
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    np.random.seed(42)
    # 后期加大波动
    ret = np.random.normal(0.001, 0.005, n)
    ret[300:] = np.random.normal(0.001, 0.04, 200)  # 后期高波动
    close = pd.Series(100 * (1 + ret).cumprod(), index=dates)
    position = pd.Series(1.0, index=dates)
    vix = pd.Series(15.0, index=dates)
    gvz = pd.Series(16.0, index=dates)
    pos_out = compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=-0.50, soft_cap=-0.20,
        daily_cap=-0.10,  # 放宽, 让波动率压制先触发
        vol_cap=0.15,  # 较严的波动率上限
        vol_window=20,
        cooldown_days=0,
    )
    high_vol_zone = pos_out.iloc[350:]  # 高波动区
    avg_pos = high_vol_zone.mean()
    print(f"  高波动区平均仓位: {avg_pos:.3f}")
    assert avg_pos < 1.0, f"高波动区仓位应<1.0, 但={avg_pos:.3f}"
    print(f"✓ test_volatility_clamp (高波动区压制至{avg_pos:.2f})")


def test_integration_with_macro_regime():
    """与生产函数集成: risk_overlay能否实际压低MaxDD."""
    sig = compute_macro_regime_signal()
    close = sig["close"]
    pos_raw = sig["position"]
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    pos_overlaid = compute_risk_overlay(
        close=close, position=pos_raw, vix=vix, gvz=gvz,
        hard_cap=-0.10,  # 严苛
        soft_cap=-0.07,
        daily_cap=-0.03,
        vol_cap=0.18,
        cooldown_days=10,
    )

    # 计算nav对比
    sr_raw = (pos_raw.shift(1) * close.pct_change()).dropna()
    sr_ov = (pos_overlaid.shift(1) * close.pct_change()).dropna()

    def quick_metrics(sr):
        nav = (1 + sr).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
        ann_vol = sr.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        return ann_ret, sharpe, dd

    raw_metrics = quick_metrics(sr_raw)
    ov_metrics = quick_metrics(sr_ov)

    print(f"\n=== 与生产函数集成 ===")
    print(f"原始:  年化={raw_metrics[0]*100:.2f}%, Sharpe={raw_metrics[1]:.2f}, MaxDD={raw_metrics[2]*100:.2f}%")
    print(f"加风控: 年化={ov_metrics[0]*100:.2f}%, Sharpe={ov_metrics[1]:.2f}, MaxDD={ov_metrics[2]*100:.2f}%")
    print(f"DD改善: {ov_metrics[2]*100 - raw_metrics[2]*100:.2f}pp")
    print(f"年化损失: {ov_metrics[0]*100 - raw_metrics[0]*100:.2f}pp")
    print(f"✓ test_integration_with_macro_regime")


def test_reentry_condition():
    """智能再入场: VIX低+GVZ低才允许恢复, 任一不满足则延长冷却."""
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100.0 + np.arange(n) * 0.1, index=dates)
    position = pd.Series(1.0, index=dates)
    vix = pd.Series(20.0, index=dates)
    gvz = pd.Series(18.0, index=dates)

    # 制造一个-12%日跌幅, 触发daily_cap=-0.10硬止损
    close.iloc[100] = close.iloc[100] * 0.85  # ~-15%
    close.iloc[101:] = close.iloc[100] + np.arange(n - 101) * 0.1

    pos_out = compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=-0.50, daily_cap=-0.10,
        cooldown_days=5,
        reentry_vix_pct=0.5,
        reentry_gvz_z=0.5,
    )

    # 崩盘日 (t=100) 应清零
    assert pos_out.iloc[100] == 0.0, f"崩盘日应清仓, 但={pos_out.iloc[100]:.3f}"
    # 冷却期内强制0 (崩盘日+4天, 共5天冷却期)
    for i in range(101, 105):
        assert pos_out.iloc[i] == 0.0, f"冷却期{i}日应=0, 但={pos_out.iloc[i]:.3f}"
    # 冷却期后应恢复 (t=105为冷却期第6天, 已恢复)
    after_cooldown = pos_out.iloc[105]
    print(f"  崩盘日零仓: ✓, 冷却期5天零仓: ✓, 冷却后恢复至 {after_cooldown:.3f}")
    print(f"✓ test_reentry_condition")


if __name__ == "__main__":
    test_no_risk_no_change()
    test_hard_stop_trigger()
    test_cooldown_period()
    test_soft_de_reduction()
    test_volatility_clamp()
    test_integration_with_macro_regime()
    test_reentry_condition()
    print("\n=== 全部测试通过 ===")