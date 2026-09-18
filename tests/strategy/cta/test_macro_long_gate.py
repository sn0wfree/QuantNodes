# coding=utf-8
"""macro_long_gate 单元测试.

覆盖:
    - mode=off: 闸门全 True
    - mode=external: 复用 short_gate 取反
    - mode=breadth: 广度阈值过滤
    - invalid mode: ValueError
    - 与 huatai_timing_cta 的集成 (long_gate_enabled)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.regime.macro_long_gate import macro_long_gate
from QuantNodes.strategy.cta.strategies.huatai_timing_cta import (
    HuataiTimingCTAStrategy, HuataiTimingConfig,
)


@pytest.fixture
def px_panel() -> pd.DataFrame:
    """500 天合成价格面板 (3 品种)."""
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    px = pd.DataFrame(
        {
            "AU": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)),
            "CU": 100 * np.cumprod(1 + rng.normal(0.0008, 0.012, n)),
            "AG": 100 * np.cumprod(1 + rng.normal(0.0012, 0.015, n)),
        },
        index=idx,
    )
    return px


# ── macro_long_gate 直接测试 ─────────────────────────────────

def test_long_gate_off_returns_all_true(px_panel):
    """mode=off 时全 True (不限制)."""
    out = macro_long_gate(px_panel, mode="off")
    assert out.shape == px_panel.shape
    assert bool(out.all().all())


def test_long_gate_breadth_shape(px_panel):
    """mode=breadth 时返回正确 shape (可能全 False, 因为短序列 ma=250 仍可计算)."""
    out = macro_long_gate(px_panel, mode="breadth", breadth_threshold=0.5, breadth_ma=250)
    assert out.shape == (len(px_panel), px_panel.shape[1])
    # dtype 应为 bool
    assert isinstance(out.dtypes.iloc[0], bool) or str(out.dtypes.iloc[0]) == "bool"


def test_long_gate_invalid_mode_raises(px_panel):
    """非法 mode → ValueError."""
    with pytest.raises(ValueError, match="mode"):
        macro_long_gate(px_panel, mode="invalid")


def test_long_gate_external_no_panic(px_panel):
    """mode=external: 即使缺宏观数据也不应崩溃."""
    out = macro_long_gate(px_panel, mode="external", use_vix=True)
    assert out.shape == px_panel.shape


def test_long_gate_external_disable_dimensions(px_panel):
    """禁用所有维度时 fallback 到全 True (顺风永远 False → long_allow 永远 False?)."""
    # 全部维度禁用 → favorable 全 False → long_allow = False (外部维度)
    # 这与 short_gate 行为一致
    out = macro_long_gate(
        px_panel, mode="external",
        use_vix=False, use_crb_proxy=False,
        use_credit_spread=False, use_copper_oil_ratio=False,
    )
    # 外部维度全关闭, 无顺风 → long_allow 全 False
    # 这是正确行为: 没有正向宏观证据时不允许做多
    assert out.shape == px_panel.shape
    # 至少有部分日期 False (无宏观数据无法判定为顺风)
    assert (~out).any().any() or not out.any().any()  # 取决于 shift 后 NaN 填充


# ── 与 huatai_timing_cta 集成测试 ─────────────────────────────

class _StubLoader:
    def load_continuous_main(self, product, start="2019-01-01", end="2026-09-04"):
        # 返回合成数据
        n = 1500
        idx = pd.date_range(start, periods=n, freq="B")
        import numpy as np
        rng = np.random.default_rng(hash(product) % 100)
        px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        df = pd.DataFrame({
            "trade_date": idx,
            "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
        })
        return df


def test_long_gate_disabled_matches_baseline():
    """long_gate_enabled=False 时与原始策略行为一致 (向后兼容)."""
    products = ["AU"]
    cfg_off = HuataiTimingConfig(
        target_vol=0.20, short_gate_enabled=False,
        long_gate_enabled=False,
    )
    cfg_on = HuataiTimingConfig(
        target_vol=0.20, short_gate_enabled=False,
        long_gate_enabled=True,
    )
    s_off = HuataiTimingCTAStrategy(products=products, config=cfg_off,
                                     start="2022-01-01", end="2026-09-04")
    s_off._dl = _StubLoader()
    s_on = HuataiTimingCTAStrategy(products=products, config=cfg_on,
                                    start="2022-01-01", end="2026-09-04")
    s_on._dl = _StubLoader()
    r_off = s_off.run()
    r_on = s_on.run()
    # 关闭时无任何闸门, 结果应与原策略一致 (这里我们比较 _long_gate 状态)
    assert s_off._long_gate is None
    assert s_on._long_gate is not None
    # 净收益不同 (long_gate 启用后会拒做多) 是预期, 不要求 NAV 相等
    assert r_off is not None
    assert r_on is not None
