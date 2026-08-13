# coding=utf-8
"""因子选择跨段稳定性检验.

对全部方向因子做 IS (2018-2021) 与 OOS (2022-2025) 分段单因子回测,
对比全样本 Sharpe, 验证因子选择决策 (剔除/保留) 的跨段稳健性.
判据: IS/OOS 符号一致 = 稳健; 符号反转 = 选择决策有风险.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    EventCalendarModuleStrategy,
    IndustrialChainModuleStrategy,
    MacroModuleStrategy,
    MeanReversionModuleStrategy,
    TermBasisModuleStrategy,
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
)

SEGS = {
    "full": ("2018-01-01", "2025-12-31"),
    "is": ("2018-01-01", "2021-12-31"),
    "oos": ("2022-01-01", "2025-12-31"),
}

MODULES = [
    TrendMomentumModuleStrategy,
    MeanReversionModuleStrategy,
    VolumeFlowModuleStrategy,
    EventCalendarModuleStrategy,
    TermBasisModuleStrategy,
    MacroModuleStrategy,
    IndustrialChainModuleStrategy,
]

# 被剔除的因子 (选择决策检验): (因子类, 输入类型, 所属模块类)
DELETED_FACTORS = [
    ("donchian", "panel", TrendMomentumModuleStrategy),
    ("basis", "spread", TermBasisModuleStrategy),
    ("roll_yield", "spread", TermBasisModuleStrategy),
    ("term_curvature", "spread", TermBasisModuleStrategy),
    ("near_basis", "basis", TermBasisModuleStrategy),
    ("basis_momentum", "basis", TermBasisModuleStrategy),
    ("oi_concentration", "spread", TermBasisModuleStrategy),
    ("front_month_oi", "spread", TermBasisModuleStrategy),
]

DELETED_CLASSES = {
    "donchian": ("DonchianChannelFactor", "trend"),
    "basis": ("BasisFactor", "term_structure"),
    "roll_yield": ("RollYieldFactor", "term_structure"),
    "term_curvature": ("TermCurvatureFactor", "term_structure"),
    "near_basis": ("NearBasisFactor", "term_structure"),
    "basis_momentum": ("BasisMomentumFactor", "basis_momentum"),
    "oi_concentration": ("OIConcentrationFactor", "position"),
    "front_month_oi": ("FrontMonthOIShareFactor", "position"),
}


def _deleted_factor(name: str):
    """构造被删因子实例."""
    cls_name, mod = DELETED_CLASSES[name]
    mod = __import__(f"QuantNodes.strategy.cta.factors.{mod}", fromlist=[cls_name])
    return getattr(mod, cls_name)()


def sharpe_of(nav: pd.Series) -> float:
    ret = nav.pct_change().dropna()
    if len(ret) < 30 or ret.std() == 0:
        return np.nan
    return float(ret.mean() / ret.std() * np.sqrt(252))


def main():
    rows = []
    for seg_name, (start, end) in SEGS.items():
        for cls in MODULES:
            s = cls(start=start, end=end)
            s.load_panel()
            s._signals_raw()
            for name, panel in s.factor_signals.items():
                nav = s._run_signal(np.sign(panel)).nav
                rows.append({"factor": name, "seg": seg_name,
                             "sharpe": sharpe_of(nav)})
        # 被删因子: 用其所属模块实例计算信号面板
        for name, input_type, mod_cls in DELETED_FACTORS:
            s = mod_cls(start=start, end=end)
            s.load_panel()
            panel = s._signal_panel(_deleted_factor(name), input_type, name)
            nav = s._run_signal(np.sign(panel)).nav
            rows.append({"factor": f"{name}*", "seg": seg_name,
                         "sharpe": sharpe_of(nav)})
    df = pd.DataFrame(rows).pivot(index="factor", columns="seg", values="sharpe")
    df = df.reindex(columns=["full", "is", "oos"])
    df["sign_consistent"] = np.sign(df["is"]) == np.sign(df["oos"])
    df["flip"] = ~df["sign_consistent"].fillna(False)

    n_consistent = df["sign_consistent"].sum()
    n_flip = df["flip"].sum()
    print(f"因子总数: {len(df)}  符号一致: {n_consistent}  反转: {n_flip} "
          f"({n_flip/len(df)*100:.0f}%)")
    print()
    print("=== 被删因子跨段表现 (决策检验, * 标记) ===")
    deleted = df[df.index.str.endswith("*")]
    print(deleted.round(3).to_string() if len(deleted) else "(无)")
    print()
    print("=== 符号反转因子 (选择决策风险) ===")
    flipped = df[df["flip"]]
    if len(flipped):
        print(flipped.round(3).to_string())
    else:
        print("(无)")
    print()
    print("=== 全样本与分段 Sharpe 全部同号且为负 ===")
    neg = df[(df["full"] < 0) & (df["is"] < 0) & (df["oos"] < 0)]
    print(neg.round(3).to_string() if len(neg) else "(无)")


if __name__ == "__main__":
    main()
