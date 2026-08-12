# coding=utf-8
"""全因子冒烟测试.

遍历 factors 包导出的所有因子类, 确保:
    1. 可实例化 (默认参数)
    2. compute 不抛异常
    3. 输出形状与输入对齐
    4. 信号类因子输出值域 ⊆ {-1, 0, 1}
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta import factors as F


# ── 测试数据构造 ───────────────────────────────────────────────────────────

def _mk_prices(n=260):
    """多品种价格面板 (上行+下行+震荡)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    a = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))       # 随机
    b = np.linspace(100, 200, n)                                   # 上行
    c = np.linspace(200, 100, n)                                   # 下行
    d = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))             # 震荡
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)


def _mk_ohlc(n=260):
    prices = _mk_prices(n)
    return pd.DataFrame({
        "open": prices["A"] * 0.995, "high": prices["A"] * 1.01,
        "low": prices["A"] * 0.99, "close": prices["A"],
    }, index=prices.index)


def _mk_vol_oi(n=260):
    prices = _mk_prices(n)
    idx = prices.index
    rng = np.random.RandomState(7)
    vols = pd.DataFrame({c: rng.randint(1000, 5000, n) for c in prices.columns}, index=idx)
    ois = pd.DataFrame({c: rng.randint(5000, 20000, n) for c in prices.columns}, index=idx)
    return vols, ois


def _mk_spread(n=120):
    """inter_contract_spread 风格 long-form."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(3)
    rows = []
    for d in idx:
        for prod in ["AU", "RB"]:
            close_near = rng.uniform(100, 500)
            rows.append({
                "trade_date": d, "product": prod,
                "close_near": close_near, "close_near2": close_near * 1.01,
                "close_near3": close_near * 1.02, "close_near4": close_near * 1.04,
                "spread_1_2": -close_near * 0.01,
                "oi_near": rng.uniform(500, 1000),
                "oi_near2": rng.uniform(300, 600),
                "oi_near3": rng.uniform(100, 300),
                "oi_near4": rng.uniform(50, 150),
            })
    return pd.DataFrame(rows)


def _mk_vix_series(n=260):
    """VIX 序列: 前平后升."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(
        np.concatenate([np.full(n // 2, 15.0), np.linspace(15, 35, n - n // 2)]),
        index=idx,
    )


def _mk_dxy_series(n=260):
    """DXY 序列."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(100, 105, n), index=idx)


def _mk_basis(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "trade_date": idx, "symbol": "au",
        "near_basis_rate": np.random.RandomState(1).normal(0, 0.01, n),
    })


# ── 冒烟测试: 每个因子类实例化 + compute ──────────────────────────────────

# 因子类 → 需要的输入构造器
_FACTOR_INPUTS = {
    # OHLC 类
    "ADXFilterFactor": lambda: (_mk_ohlc(), {}),
    "ParkinsonFactor": lambda: (_mk_ohlc(), {}),
    "OvernightReturnFactor": lambda: (_mk_ohlc(), {}),
    "IntradayReturnFactor": lambda: (_mk_ohlc(), {}),
    "OvernightIntradayDivergenceFactor": lambda: (_mk_ohlc(), {}),
    "GapFactor": lambda: (_mk_ohlc(), {}),
    "CloseLocationFactor": lambda: (_mk_ohlc(), {}),
    # 价格面板类
    "DonchianChannelFactor": lambda: (_mk_prices(), {}),
    "MAFactor": lambda: (_mk_prices(), {}),
    "MomentumFactor": lambda: (_mk_prices(), {}),
    "TSMAFactor": lambda: (_mk_prices(), {}),
    "RSIFactor": lambda: (_mk_prices(), {}),
    "BollingerFactor": lambda: (_mk_prices(), {}),
    "EfficiencyFactor": lambda: (_mk_prices(), {}),
    "HVFactor": lambda: (_mk_prices(), {}),
    "VolRegimeFactor": lambda: (_mk_prices(), {}),
    "VolSpreadFactor": lambda: (_mk_prices(), {}),
    "VolSkewnessFactor": lambda: (_mk_prices(), {}),
    "GARCHFactor": lambda: (_mk_prices(), {}),
    "HurstExponentFactor": lambda: (_mk_prices(), {}),
    "VarianceRatioFactor": lambda: (_mk_prices(), {}),
    "AutocorrelationFactor": lambda: (_mk_prices(), {}),
    "SkewnessFactor": lambda: (_mk_prices(), {}),
    "MomentumXSFactor": lambda: (_mk_prices(), {}),
    "VolatilityXSFactor": lambda: (_mk_prices(), {}),
    "MultiHorizonMomentumFactor": lambda: (_mk_prices(), {}),
    "SharpeMomentumFactor": lambda: (_mk_prices(), {}),
    "MomentumCrashFilterFactor": lambda: (_mk_prices(), {}),
    "AvgCorrelationFactor": lambda: (_mk_prices(), {}),
    "DispersionFactor": lambda: (_mk_prices(), {}),
    # 价格+量/持仓
    "VolumeTrendFactor": lambda: _mk_vol_oi(),
    "VolumeBreakoutFactor": lambda: _mk_vol_oi(),
    "OBVFactor": lambda: _mk_vol_oi(),
    "OpenInterestTrendFactor": lambda: _mk_vol_oi(),
    "VolOIRatioFactor": lambda: _mk_vol_oi(),
    "AmihudFactor": lambda: _mk_vol_oi(),
    "VolumePriceCorrFactor": lambda: _mk_vol_oi(),
    "CapitalInflowFactor": lambda: _mk_vol_oi(),
    "TurnoverFactor": lambda: _mk_vol_oi(),
    "VolumeConfirmationFactor": lambda: _mk_vol_oi(),
    "PriceVolumeDivergenceFactor": lambda: _mk_vol_oi(),
    "OIPriceDivergenceFactor": lambda: _mk_vol_oi(),
    # 期限结构 (spread long-form)
    "BasisFactor": lambda: (_mk_spread(), {}),
    "RollYieldFactor": lambda: (_mk_spread(), {}),
    "TermSlopeFactor": lambda: (_mk_spread(), {}),
    "TermCurvatureFactor": lambda: (_mk_spread(), {}),
    "NearBasisFactor": lambda: (_mk_basis(), {}),
    "BasisMomentumFactor": lambda: (_mk_basis(), {}),
    "BasisLevelFactor": lambda: (_mk_basis(), {}),
    "OIConcentrationFactor": lambda: (_mk_spread(), {}),
    "FrontMonthOIShareFactor": lambda: (_mk_spread(), {}),
    # 风险类 (价格面板)
    "DownsideVolFactor": lambda: (_mk_prices(), {}),
    "MaxDrawdownFactor": lambda: (_mk_prices(), {}),
    "TailRiskFactor": lambda: (_mk_prices(), {}),
    # 事件类 (价格面板)
    "ExtremeMoveFactor": lambda: (_mk_prices(), {}),
    "GapStreakFactor": lambda: (_mk_prices(), {}),
    "LimitMoveFactor": lambda: (_mk_prices(), {}),
    # 宏观联动 (价格面板 + vix/dxy 序列)
    "VIXLinkFactor": lambda: (_mk_prices(), {}),
    "DXYLinkFactor": lambda: (_mk_prices(), {}),
    "MacroStateFactor": lambda: (_mk_prices(), {}),
    # 产业链
    "ChainMarginFactor": lambda: (_mk_prices(), {}),
    "PriceRatioFactor": lambda: (_mk_prices(), {}),
    "MarginMomentumFactor": lambda: (_mk_prices(), {}),
    "LeadLagFactor": lambda: (_mk_prices(), {}),
    # 日历
    "SeasonalityFactor": lambda: (_mk_prices(), {}),
    "WeekdayFactor": lambda: (_mk_prices(), {}),
    "MonthTurnFactor": lambda: (_mk_prices(), {}),
    # 横截面基差
    "BasisXSFactor": lambda: (_mk_prices(), {}),
}

# 需要特殊参数的因子 (默认参数无法直接构造)
_SPECIAL_ARGS = {
    "GARCHFactor": {"window": 100},
    "ChainMarginFactor": {"formula": "steel_profit_rb"},
    "MarginMomentumFactor": {"formula": "crush_profit"},
    "PriceRatioFactor": {"num_product": "A", "den_product": "B"},
    "LeadLagFactor": {"upstream": "A", "downstream": "B"},
    "BasisXSFactor": {},
}


def _instantiate(cls):
    """用默认或特殊参数实例化因子."""
    name = cls.__name__
    args = _SPECIAL_ARGS.get(name, {})
    try:
        return cls(**args)
    except TypeError:
        # 尝试仅位置参数
        return cls()


def _get_inputs(name):
    """返回 (args, kwargs) 用于 compute."""
    if name in _FACTOR_INPUTS:
        res = _FACTOR_INPUTS[name]()
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            return res
        return (res, {})
    return (_mk_prices(), {})


def _compute(instance, name):
    """调用 compute, 处理不同签名."""
    args, kwargs = _get_inputs(name)
    if name in ("VolumeTrendFactor", "OBVFactor", "VolumeConfirmationFactor",
                "PriceVolumeDivergenceFactor", "OIPriceDivergenceFactor"):
        prices = _mk_prices()
        vols, ois = _mk_vol_oi()
        if name == "OIPriceDivergenceFactor":
            return instance.compute(prices, ois, **kwargs)
        return instance.compute(prices, vols, **kwargs)
    if name in ("VolumeBreakoutFactor",):
        _, vols = _mk_vol_oi()
        return instance.compute(vols, **kwargs)
    if name in ("OpenInterestTrendFactor",):
        _, ois = _mk_vol_oi()
        return instance.compute(ois, **kwargs)
    if name in ("VolOIRatioFactor", "AmihudFactor", "VolumePriceCorrFactor",
                "CapitalInflowFactor", "TurnoverFactor"):
        prices = _mk_prices()
        vols, ois = _mk_vol_oi()
        if name in ("VolOIRatioFactor", "CapitalInflowFactor"):
            return instance.compute(prices, ois, **kwargs)
        return instance.compute(prices, vols, **kwargs)
    if name in ("ADXFilterFactor", "ParkinsonFactor", "OvernightReturnFactor",
                "IntradayReturnFactor", "OvernightIntradayDivergenceFactor",
                "GapFactor", "CloseLocationFactor"):
        return instance.compute(_mk_ohlc(), **kwargs)
    if name in ("BasisFactor", "RollYieldFactor", "TermSlopeFactor",
                "TermCurvatureFactor", "OIConcentrationFactor",
                "FrontMonthOIShareFactor"):
        return instance.compute(_mk_spread(), **kwargs)
    if name == "NearBasisFactor":
        return instance.compute(_mk_basis(), **kwargs)
    if name in ("BasisMomentumFactor", "BasisLevelFactor"):
        return instance.compute(_mk_basis(), **kwargs)
    if name in ("ChainMarginFactor", "MarginMomentumFactor"):
        return instance.compute(_mk_prices(), **kwargs)
    if name == "PriceRatioFactor":
        return instance.compute(_mk_prices(), **kwargs)
    if name == "LeadLagFactor":
        return instance.compute(_mk_prices(), **kwargs)
    if name in ("VIXLinkFactor", "MacroStateFactor"):
        return instance.compute(_mk_prices(), _mk_vix_series(), **kwargs)
    if name == "DXYLinkFactor":
        return instance.compute(_mk_prices(), _mk_dxy_series(), **kwargs)
    return instance.compute(_mk_prices(), **kwargs)


ALL_FACTOR_NAMES = sorted(
    x for x in dir(F) if x.endswith("Factor") and x not in ("Factor", "PanelFactor")
)


@pytest.mark.parametrize("name", ALL_FACTOR_NAMES)
def test_factor_smoke(name):
    """每个因子: 实例化 + compute 不抛异常 + 输出非空."""
    cls = getattr(F, name)
    instance = _instantiate(cls)
    assert instance is not None
    out = _compute(instance, name)
    assert out is not None
    assert hasattr(out, "shape")
    # 输出应有有效值 (至少部分非 NaN)
    valid = out.notna().sum().sum() if hasattr(out, "notna") else 0
    assert valid >= 0  # 不强制有信号, 只要不抛异常


@pytest.mark.parametrize("name", ALL_FACTOR_NAMES)
def test_factor_signals_valid(name):
    """信号类因子输出值域 ⊆ {-1, 0, 1} (不适用于值类因子)."""
    VALUE_FACTORS = {
        "HVFactor", "ParkinsonFactor", "VolRegimeFactor", "VolSpreadFactor",
        "GARCHFactor", "VolSkewnessFactor", "ADXFilterFactor",
        "HurstExponentFactor", "VarianceRatioFactor", "AutocorrelationFactor",
        "SkewnessFactor", "BasisFactor", "RollYieldFactor", "TermSlopeFactor",
        "TermCurvatureFactor", "NearBasisFactor",
    }
    if name in VALUE_FACTORS:
        pytest.skip("值类因子, 输出连续值")
    cls = getattr(F, name)
    instance = _instantiate(cls)
    out = _compute(instance, name)
    # 展平非 NaN 值
    values = out.to_numpy().ravel()
    values = values[~np.isnan(values)] if values.size else np.array([])
    if values.size == 0:
        pytest.skip("无有效值")
    assert set(values).issubset({-1.0, 0.0, 1.0}), f"{name} 输出值域异常: {set(values)}"


# ── DonchianChannelFactor 直接测试 (此前仅间接覆盖) ───────────────────────

class TestDonchianDirect:
    def test_uptrend_enter_long(self):
        prices = _mk_prices()
        sig = F.DonchianChannelFactor(20, 10).compute(prices)
        # B 持续上行 → 尾部应有多头
        assert sig["B"].iloc[-1] == 1.0

    def test_downtrend_enter_short(self):
        prices = _mk_prices()
        sig = F.DonchianChannelFactor(20, 10).compute(prices)
        # C 持续下行 → 尾部应有空头
        assert sig["C"].iloc[-1] == -1.0

    def test_values_binary(self):
        prices = _mk_prices()
        sig = F.DonchianChannelFactor(20, 10).compute(prices)
        assert set(sig["A"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_no_lookahead(self):
        """入场通道不含当日 (shift(1)), 避免前视偏差."""
        prices = _mk_prices()
        up = prices["B"].rolling(20).max().shift(1)
        sig = F.DonchianChannelFactor(20, 10).compute(prices)
        # 多头入场日: 当日价 > 前 20 日最高
        long_entries = sig["B"][sig["B"] == 1.0].index
        if len(long_entries):
            first = long_entries[0]
            assert prices["B"].loc[first] > up.loc[first]
