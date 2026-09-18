# coding=utf-8
"""宏观/广度因子做空闸门 (macro short gate).

背景 (研报原文 §"系统性全局风控" 的宏观延伸 + 归因证据):
    2020-2026 为商品超级牛市周期 (通胀+放水), 做空存在结构性逆风
    (通胀漂移 + 展期收益偏置)。trend_down 做空整体 -0.153% (t=-3.50),
    亏损集中于长期上涨品种 (AL/SC/AU/P); 唯一盈利窗口是真实熊市段
    (ZN/RB 2022H2、NI 2023, 恰逢商品广度跌至 0.36)。

设计:
    商品广度 (内部): 20品种中 close > MA250 的占比。
        breadth >= threshold (0.6) → 商品结构性牛市 → 禁空
        breadth <  threshold        → 存在真实熊市段 → 允许空
    外部宏观 (可选): DXY 120日动量<0 (美元弱=商品顺风) 或
                     US10Y 120日动量>0 (通胀上行=商品顺风) → 禁空

四种模式:
    breadth   仅广度
    external  仅外部宏观
    both_and  广度放行 AND 非顺风 (双闸最严)
    both_or   广度放行 OR 非顺风 (任一闸)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

US10Y_PARQUET = (
    "/home/ll/Public/QuantNodes/data/high_freq_macro/us_10y_yield_fred.parquet"
)

_GATE_MODES = ("breadth", "external", "both_and", "both_or")


def commodity_breadth(px_panel: pd.DataFrame, ma: int = 250) -> pd.Series:
    """商品广度: close > MA(ma) 的品种占比 (0~1).

    Args:
        px_panel: (T, N) 收盘价面板
        ma: 广度均线窗口 (默认 250, 约一年)

    Returns:
        pd.Series[float], 与 px_panel 同索引; 某日无任何有效品种时为 NaN.
    """
    ma_vals = px_panel.rolling(ma, min_periods=ma // 2).mean()
    above = px_panel > ma_vals
    n_valid = above.notna().sum(axis=1).clip(lower=1)
    return above.sum(axis=1) / n_valid


def _us10y_series() -> pd.Series:
    """US 10Y 收益率序列 (FRED parquet)."""
    df = pd.read_parquet(US10Y_PARQUET)
    s = pd.Series(df["us_10y_yield"].values, index=pd.to_datetime(df["date"]))
    return s.dropna()


def external_commodity_favorable(
    dxy_mom_window: int = 120,
    us10y_mom_window: int = 120,
    vix_mom_window: int = 120,
    crb_mom_window: int = 120,
    cs_mom_window: int = 120,
    cor_mom_window: int = 120,
    cor_change_threshold: float = -0.3,
    use_vix: bool = True,
    use_crb_proxy: bool = True,
    use_credit_spread: bool = True,
    use_copper_oil_ratio: bool = True,
    px_panel: pd.DataFrame | None = None,
) -> pd.Series:
    """外部宏观商品顺风指数 (bool, True=顺风=禁空).

    条件 (任一满足即顺风):
        DXY 120日动量 < 0  (美元走弱 → 商品计价顺风)
        US10Y 120日动量 > 0 (通胀预期上行 → 商品顺风)
        VIX 120日动量 > 0  (风险规避上升期做空 = 接救市反弹的刀;
                            归因: trend_down 空头 × VIX上升 -0.562%, t=-3.17)
        CRB等权价格指数 120日动量 > 0 (商品复合体整体上行 → 做空逆风;
                                       比离散型广度因子更细腻的连续动量信号)
        cn_us_spread 120日动量 > 0  (中美10Y利差扩大 → 人民币贬值压力
                                       → 商品计价顺风 → 禁空)
        gold_oil_corr 120日变化 < -0.3 (金油相关性骤降 = 避险资产跑赢工业品
                                         = 经济衰退预期 → 商品承压 →
                                         顺风逻辑反转, 仍禁空)

    Args:
        dxy_mom_window / us10y_mom_window / vix_mom_window / crb_mom_window /
        cs_mom_window / cor_mom_window:
            各指标动量计算窗口 (默认 120)
        cor_change_threshold: 金油比相关性 120日变化阈值 (默认 -0.3)
        use_vix: 是否包含 VIX 维度 (默认 True)
        use_crb_proxy: 是否包含 CRB 等权价格指数代理 (默认 True)
        use_credit_spread: 是否包含中美利差维度 (默认 True)
        use_copper_oil_ratio: 是否包含金油比相关性维度 (默认 True)
        px_panel: (T, N) 品种收盘价面板 (use_crb_proxy=True 时必填;
                  若 None 则跳过 CRB 维度)

    Returns:
        pd.Series[bool], 索引为各序列日期的并集.
    """
    from QuantNodes.strategy.cta.data_loader import CtaDataLoader

    dl = CtaDataLoader()
    dxy = dl.load_macro_series(name="dxy")
    if dxy is not None and not isinstance(dxy, pd.Series):
        dxy = dxy.iloc[:, 0]
    us10y = _us10y_series()
    vix = None
    if use_vix:
        try:
            vix = dl.load_macro_series(name="vix")
            if vix is not None and not isinstance(vix, pd.Series):
                vix = vix.iloc[:, 0]
            vix = vix.dropna() if vix is not None else None
        except Exception:
            vix = None

    cs = None
    if use_credit_spread:
        try:
            cs = dl.load_macro_series(name="cn_us_spread").dropna()
        except Exception:
            cs = None

    cor = None
    if use_copper_oil_ratio:
        try:
            cor = dl.load_macro_series(name="gold_oil_corr").dropna()
        except Exception:
            cor = None

    favorable = pd.Series(False, dtype=bool, index=[])
    frames = []
    if dxy is not None and len(dxy):
        frames.append(dxy.pct_change(dxy_mom_window) < 0)
    if len(us10y):
        frames.append(us10y.diff(us10y_mom_window) > 0)
    if vix is not None and len(vix):
        frames.append(vix.pct_change(vix_mom_window) > 0)
    if use_crb_proxy and px_panel is not None:
        # CRB 等权价格指数代理: 20品种等权构建连续指数
        equal_idx = px_panel.mean(axis=1).dropna()
        if len(equal_idx) > crb_mom_window:
            crb_mom = equal_idx / equal_idx.shift(crb_mom_window) - 1
            frames.append(crb_mom > 0)  # CRB上行 → 商品顺风 → 禁空
    if cs is not None and len(cs) > cs_mom_window:
        # 中美利差扩大 → 人民币贬值 → 商品顺风 → 禁空
        frames.append(cs.diff(cs_mom_window) > 0)
    if cor is not None and len(cor) > cor_mom_window:
        # 金油相关性 120 日下降超过阈值 = 经济衰退预期 → 禁空
        cor_change = cor - cor.shift(cor_mom_window)
        frames.append(cor_change < cor_change_threshold)
    for f in frames:
        favorable = favorable.reindex(favorable.index.union(f.index), fill_value=False)
        favorable = favorable | f.reindex(favorable.index, fill_value=False)
    return favorable


def macro_short_gate(
    px_panel: pd.DataFrame,
    mode: str = "breadth",
    breadth_threshold: float = 0.6,
    breadth_ma: int = 250,
    use_vix: bool = True,
    use_crb_proxy: bool = True,
    use_credit_spread: bool = True,
    use_copper_oil_ratio: bool = True,
) -> pd.DataFrame:
    """做空闸门: 返回 (T, N) bool, True = 允许做空.

    Args:
        px_panel: (T, N) 收盘价面板 (广度因子输入)
        mode: 'breadth' / 'external' / 'both_and' / 'both_or'
        breadth_threshold: 广度禁空阈值 (默认 0.6)
        breadth_ma: 广度均线 (默认 250)
        use_vix: 是否使用 VIX 维度 (默认 True)
        use_crb_proxy: 是否使用 CRB 等权价格指数代理 (默认 True)
        use_credit_spread: 是否使用中美利差维度 (默认 True)
        use_copper_oil_ratio: 是否使用金油比相关性维度 (默认 True)

    Returns:
        pd.DataFrame[bool], 与 px_panel 同形. trend_down 做空信号
        仅在该闸门为 True 的日子放行.
    """
    if mode not in _GATE_MODES:
        raise ValueError(f"mode must be one of {_GATE_MODES}, got {mode!r}")

    breadth = commodity_breadth(px_panel, breadth_ma)
    breadth_allow = (breadth < breadth_threshold).fillna(False)

    if mode == "breadth":
        out = breadth_allow
        return pd.DataFrame(
            np.repeat(out.values[:, None], px_panel.shape[1], axis=1),
            index=px_panel.index, columns=px_panel.columns,
        )

    ext_fav = external_commodity_favorable(
        use_vix=use_vix, use_crb_proxy=use_crb_proxy,
        use_credit_spread=use_credit_spread,
        use_copper_oil_ratio=use_copper_oil_ratio,
        px_panel=px_panel,
    )
    ext_allow = (~ext_fav)
    # 对齐到品种日历, lag 1 日防前视 (用 t-1 的宏观状态判定 t 日闸门)
    # 注意: bool Series shift(1) 会退化为 object dtype, 必须显式 astype(bool)
    ext_allow = (
        ext_allow.reindex(px_panel.index).shift(1).fillna(False).astype(bool)
    )

    def _broadcast(mask: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            np.repeat(mask.values[:, None], px_panel.shape[1], axis=1),
            index=px_panel.index, columns=px_panel.columns, dtype=bool,
        )

    if mode == "external":
        return _broadcast(ext_allow)
    # 注意: Series & DataFrame 会按 Series.index ↔ DataFrame.columns 对齐
    # (日期 vs 品种名 = 空集), 必须先将两侧都广播为 DataFrame
    if mode == "both_and":
        return _broadcast(breadth_allow) & _broadcast(ext_allow)
    return _broadcast(breadth_allow) | _broadcast(ext_allow)
