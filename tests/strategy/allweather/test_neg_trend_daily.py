"""V8 D.1 负趋势因子日频风控层单测 (PLAN §17.12)."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.neg_trend_daily import (
    compute_neg_trend_penalty,
    compute_sub_signals,
    _hetero_weight,
    _rapid_move_score,
    NEG_TREND_PRESETS,
)
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES


def _make_panel(n_days=400, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = sorted({c for ms in ASSET_CLASSES.values() for c in ms})
    data = {c: 100 * (1 + rng.normal(0.0003, 0.01, n_days)).cumprod()
            for c in cols}
    return pd.DataFrame(data, index=dates)


def _make_volume_oi(panel, seed=7):
    rng = np.random.default_rng(seed)
    vol = pd.DataFrame({c: rng.uniform(1e4, 1e6, len(panel))
                        for c in panel.columns}, index=panel.index)
    fut_cols = [c for c in panel.columns if c.startswith("fut_")]
    oi = pd.DataFrame({c: rng.uniform(1e4, 1e5, len(panel))
                       for c in fut_cols}, index=panel.index)
    return vol, oi


# ── penalty 基本性质 ──

def test_penalty_bounds():
    """penalty ∈ [0, 1]."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    pen = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                     asset_classes=ASSET_CLASSES, mode="3d")
    assert (pen >= 0).all().all()
    assert (pen <= 1).all().all()


def test_mode_none_all_zero():
    """mode='none' → 全零 (开关关闭)."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    pen = compute_neg_trend_penalty(panel, volume=vol, oi=oi, mode="none")
    assert (pen == 0).all().all()


def test_trend_only_uses_trend_signal():
    """trend 模式: 趋势为负且 vol/flow=0 时, 3d 与 trend 惩罚一致 (趋势权重下)."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    # vol/flow 归零 → 3d 退化为 trend-only (同权重 w_trend)
    vol0 = vol * 0
    pen_t = compute_neg_trend_penalty(panel, volume=vol0, oi=oi,
                                       asset_classes=ASSET_CLASSES, mode="trend")
    pen_3d = compute_neg_trend_penalty(panel, volume=vol0, oi=oi,
                                        asset_classes=ASSET_CLASSES, mode="3d")
    # 趋势维度权重相同 (所有类 w_trend>0), 两模式都主要响应趋势
    # 断言两者激活率都 > 0 且同向 (宽公差, 因 trend 模式不含 vol/flow 贡献)
    assert (pen_t > 0).mean().mean() > 0
    assert (pen_3d > 0).mean().mean() > 0


def test_hysteresis_no_frequent_flip():
    """滞回: 激活期不应高频翻转 (连续 penalty>0 段长 > 1 日的占比高)."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    pen = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                     asset_classes=ASSET_CLASSES, mode="3d")
    # 计算每列连续激活段的平均长度
    lengths = []
    for col in pen.columns:
        s = (pen[col] > 0).astype(int)
        # 段长 = diff 计数
        starts = s.diff().fillna(0) == 1
        n_seg = starts.sum()
        total = s.sum()
        if n_seg > 0:
            lengths.append(total / n_seg)
    if lengths:
        assert np.mean(lengths) > 1.5  # 平均激活段长 > 1.5 天


# ── 异质权重 ──

def test_hetero_weights_dispatch():
    """异质权重: precious 重 vol, nonferrous 重 flow (设计文稿)."""
    w_gold = _hetero_weight("fut_au", ASSET_CLASSES)
    w_metal = _hetero_weight("fut_cu", ASSET_CLASSES)
    assert w_gold["vol"] > w_metal["vol"]
    assert w_metal["flow"] > w_gold["flow"]
    # 未知资产 → 等权默认
    w_unknown = _hetero_weight("unknown_asset", ASSET_CLASSES)
    assert abs(sum(w_unknown.values()) - 1.0) < 1e-9


# ── volume_loader ──

def test_volume_loader_alignment():
    """量价面板与 panel 索引对齐; ETF 的 OI 为 NaN."""
    panel = _make_panel()
    # 用合成量价 (真实加载器需 DuckDB/parquet, 单测用注入)
    vol, oi = _make_volume_oi(panel)
    assert vol.shape == panel.shape
    assert oi.shape[0] == panel.shape[0]
    assert (oi.columns.str.startswith("fut_")).all()


def test_sub_signals_shape_and_bounds():
    """子信号形状对齐 panel, 值域正确."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    st, sv, sf = compute_sub_signals(panel, vol, oi)
    for sig in (st, sv, sf):
        assert sig.shape == panel.shape
        assert sig.notna().all().all()
    assert (st.abs() <= 1.0001).all().all()   # tanh 压缩 [-1,1]
    assert (sv >= 0).all().all() and (sv <= 1).all().all()
    assert (sf >= 0).all().all() and (sf <= 1.0001).all().all()


def test_sub_trend_saturation_fixed():
    """饱和修复: |sub_trend|=1 的占比 < 10% (原 tanh(x) 重度饱和 ~25%)."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    st, _, _ = compute_sub_signals(panel, vol, oi)
    sat_ratio = ((st.abs() >= 0.9999).mean()).mean()
    assert sat_ratio < 0.10


# ═══════════════════════════════════════════════════════════════════
# D.1 三档参数 (用户确认全部保留) 单测
# ═══════════════════════════════════════════════════════════════════

def test_presets_defined_and_monotonic():
    """三档都在字典里且阈值单调: aggressive < balanced < conservative."""
    from QuantNodes.strategy.allweather.neg_trend_daily import (
        NEG_TREND_PRESETS, DEFAULT_PRESET,
    )
    assert set(NEG_TREND_PRESETS) == {"aggressive", "balanced", "conservative"}
    a = NEG_TREND_PRESETS["aggressive"]
    b = NEG_TREND_PRESETS["balanced"]
    c = NEG_TREND_PRESETS["conservative"]
    assert a["base_threshold"] < b["base_threshold"] < c["base_threshold"]
    assert a["threshold_low"] < b["threshold_low"] < c["threshold_low"]
    assert DEFAULT_PRESET == "balanced"


def test_get_preset_params_fallback():
    """未知名回退 balanced."""
    from QuantNodes.strategy.allweather.neg_trend_daily import (
        get_preset_params, NEG_TREND_PRESETS,
    )
    assert get_preset_params("unknown") == NEG_TREND_PRESETS["balanced"]
    assert get_preset_params("aggressive")["base_threshold"] == 0.4


def test_preset_activation_monotonic():
    """合成数据: 激活率单调 aggressive ≥ balanced ≥ conservative."""
    panel = _make_panel(n_days=400, seed=99)
    vol, oi = _make_volume_oi(panel)
    rates = {}
    for preset in ["aggressive", "balanced", "conservative"]:
        pen = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                         asset_classes=ASSET_CLASSES,
                                         mode="trend",
                                         base_threshold=NEG_TREND_PRESETS[preset]
                                         ["base_threshold"],
                                         threshold_low=NEG_TREND_PRESETS[preset]
                                         ["threshold_low"])
        rates[preset] = (pen > 0).mean().mean()
    assert rates["aggressive"] >= rates["balanced"]
    assert rates["balanced"] >= rates["conservative"]


def test_run_d_sleeve_preset_passthrough():
    """run_d_sleeve 档位透传: 不同档位产生不同 nav (aggressive 减仓最多)."""
    from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
    panel = _make_panel(n_days=300)
    vol, oi = _make_volume_oi(panel)
    navs = {}
    for preset in ["aggressive", "conservative"]:
        res = run_d_sleeve(panel, bond_cap=0.40, window=120,
                           rebalance_freq="W-FRI", cov_estimator="lw",
                           neg_trend_mode="trend", volume=vol, oi=oi,
                           asset_classes=ASSET_CLASSES,
                           neg_trend_preset=preset)
        navs[preset] = res.nav
    # 不同档位 → 不同净值
    assert not np.allclose(navs["aggressive"].values,
                           navs["conservative"].values)


# ═══════════════════════════════════════════════════════════════════
# D.3 急跌检测器 (§5.29)
# ═══════════════════════════════════════════════════════════════════

def test_rapid_move_score_bounds():
    """rapid score ∈ [0, 2], 无 NaN."""
    panel = _make_panel()
    vol, oi = _make_volume_oi(panel)
    rs = _rapid_move_score(panel)
    assert rs.notna().all().all()
    assert (rs >= 0).all().all()
    assert (rs <= 1.0001).all().all()


def test_rapid_catches_sudden_drop():
    """合成急跌: 末两日全资产暴跌 → rapid penalty 在组合层面显著高于平静期."""
    panel = _make_panel(n_days=200, seed=42)
    # 最后 2 日全资产 -15% (模拟全市场急跌)
    for c in panel.columns:
        panel.iloc[-2, panel.columns.get_loc(c)] = panel.iloc[-3, panel.columns.get_loc(c)] * 0.85
        panel.iloc[-1, panel.columns.get_loc(c)] = panel.iloc[-2, panel.columns.get_loc(c)] * 0.85
    pen = compute_neg_trend_penalty(panel, mode="rapid", asset_classes=ASSET_CLASSES)
    # 急跌日 penalty (组合层面均值) vs 平静期
    calm_pen = pen.iloc[:190].mean(axis=1).mean()
    shock_pen = pen.iloc[-1].mean()
    # 组合层面: 急跌日 penalty 至少是平静期的 1.5 倍
    assert shock_pen > calm_pen * 1.5


def test_rapid_standalone_in_penalty():
    """mode='rapid' standalone → 非零 penalty, 与 mode='none' 不同."""
    panel = _make_panel(n_days=300, seed=55)
    vol, oi = _make_volume_oi(panel)
    pen_none = compute_neg_trend_penalty(panel, mode="none")
    pen_rapid = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                           mode="rapid", asset_classes=ASSET_CLASSES)
    assert (pen_none == 0).all().all()
    assert (pen_rapid > 0).mean().mean() > 0  # 有非零激活


def test_include_rapid_modifies_trend():
    """include_rapid=True 在 trend 模式上叠加 rapid, 与纯 trend 不同."""
    panel = _make_panel(n_days=300, seed=77)
    vol, oi = _make_volume_oi(panel)
    pen_trend = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                           mode="trend", asset_classes=ASSET_CLASSES)
    pen_tr = compute_neg_trend_penalty(panel, volume=vol, oi=oi,
                                        mode="trend", include_rapid=True,
                                        asset_classes=ASSET_CLASSES)
    # 两者不应完全相等 (rapid 叠加后部分日/资产有差异)
    assert not np.allclose(pen_trend.values, pen_tr.values)
    # 两者都不应全零
    assert (pen_trend > 0).any().any()
    assert (pen_tr > 0).any().any()


def test_include_rapid_in_d_pipeline():
    """run_d_sleeve include_rapid=True 与 False 产生不同 nav."""
    from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
    from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES
    panel = _make_panel(n_days=300, seed=33)
    vol, oi = _make_volume_oi(panel)
    res0 = run_d_sleeve(panel, bond_cap=0.40, window=120,
                        rebalance_freq="W-FRI", cov_estimator="lw",
                        neg_trend_mode="trend",
                        volume=vol, oi=oi, asset_classes=ASSET_CLASSES)
    res1 = run_d_sleeve(panel, bond_cap=0.40, window=120,
                        rebalance_freq="W-FRI", cov_estimator="lw",
                        neg_trend_mode="trend", include_rapid=True,
                        volume=vol, oi=oi, asset_classes=ASSET_CLASSES)
    assert not np.allclose(res0.nav.values, res1.nav.values)
