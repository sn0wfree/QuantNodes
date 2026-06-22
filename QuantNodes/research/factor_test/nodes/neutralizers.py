# coding=utf-8
"""Neutralizer 抽象与具体实现 (Chain of Responsibility, Phase 2.1)。

将原 factor_neutralize_node.py::_neutralize 中 3 个几乎相同的 if/elif
分支 (industry only / risk only / both) 抽象为:

  Neutralizer (ABC)
    ├── IndustryNeutralizer   # 行业哑变量
    └── RiskNeutralizer       # 风险因子列

  build_neutralizer_chain(if_industry, if_risk, industry, risk_data) -> list
  apply_neutralizer_chain(factor_i, chain) -> factor_neut

每个 neutralizer 负责自己的"设计矩阵 X 组装", apply_neutralizer_chain
负责统一的"按日期循环 + OLS + 写残差"流程。新增中性化类型
(如 StyleNeutralizer) 只需新增一个 Neutralizer 子类, _execute
无需修改。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================================
# Abstract base
# ============================================================================

class Neutralizer(ABC):
    """中性化器抽象基类 (Chain of Responsibility 的一环).

    子类实现 build_design_matrix() 返回指定日期的 X (设计矩阵).
    is_active() 用于 build_neutralizer_chain 过滤无效环节.
    """

    name: str = ""

    @abstractmethod
    def build_design_matrix(
        self,
        date: pd.Timestamp,
        factor_i: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """为指定日期组装设计矩阵 X (index=股票代码, columns=回归变量).

        Returns:
            X: 索引 = 股票代码, 列 = 哑变量/风险因子值
            None: 跳过此日期 (无足够数据)
        """
        raise NotImplementedError

    def is_active(self) -> bool:
        """默认: neutralizer 始终 active. 子类可覆盖 (如缺数据时关闭)."""
        return True


# ============================================================================
# Concrete: Industry
# ============================================================================

class IndustryNeutralizer(Neutralizer):
    """行业中性化: 对 industry Series 做 one-hot dummy encoding.

    行为与原 _neutralize branch 2 (lines 98-114) 一致:
      - industry NaN 替换为 0
      - 每个日期生成 dummies
      - 去掉全 0 列 (sum > 0 过滤)
    """

    name = "industry"

    def __init__(self, industry: Optional[pd.Series]) -> None:
        self.industry = (
            industry.copy().replace(np.nan, 0) if industry is not None else None
        )

    def is_active(self) -> bool:
        return self.industry is not None

    def build_design_matrix(
        self, date: pd.Timestamp, factor_i: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        if self.industry is None or date not in self.industry.index:
            return None
        ind_j = self.industry.loc[date]
        dum_ind = pd.get_dummies(ind_j)
        # 与原代码一致: 去掉全 0 列
        return dum_ind.loc[:, dum_ind.sum() > 0]


# ============================================================================
# Concrete: Risk
# ============================================================================

class RiskNeutralizer(Neutralizer):
    """风险因子中性化: 把加载的 risk_data 横向 concat 为 X.

    输出 X 形状: index=股票代码, columns=risk_factors (与 IndustryNeutralizer 一致).
    这样 apply_neutralizer_chain 中的 pd.merge(..., left_index=True, right_index=True) 能正确合并.

    注: 原 _neutralize branch 3 (lines 116-135) 用 pd.concat(..., axis=1) 组装 X,
    得到 (n_risks, n_stocks) 形状 (index=range(n_risks), columns=stock_codes),
    与后续 merge 不匹配, 是 latent bug. 本实现修正.
    """

    name = "risk"

    def __init__(self, risk_data: list) -> None:
        self.risk_data = risk_data

    def is_active(self) -> bool:
        return bool(self.risk_data)

    def build_design_matrix(
        self, date: pd.Timestamp, factor_i: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        cols: List[pd.DataFrame] = []
        for i, rf in enumerate(self.risk_data):
            if date not in rf.index:
                continue
            # rf.loc[date] 是 Series, index=股票代码 (因为 rf.columns=股票代码)
            # 转为 1 列 DataFrame (index=股票代码, column=rf_i)
            col = rf.loc[date].to_frame(name=f"rf_{i}")
            cols.append(col)
        if not cols:
            return None
        # 横向 concat (axis=1) 沿股票代码 index 对齐
        return pd.concat(cols, axis=1)


# ============================================================================
# Chain construction & execution
# ============================================================================

def build_neutralizer_chain(
    if_industry: bool,
    if_risk: bool,
    industry: Optional[pd.Series],
    risk_data: list,
) -> List[Neutralizer]:
    """根据配置构造 chain, 自动过滤 is_active() == False 的环节.

    顺序固定: [Industry, Risk] (与原代码 if/elif 优先级一致).

    Args:
        if_industry: 是否启用行业中性化
        if_risk: 是否启用风险因子中性化
        industry: 行业 Series (None 时 IndustryNeutralizer 自动 inactive)
        risk_data: 风险因子 list (空时 RiskNeutralizer 自动 inactive)

    Returns:
        List[Neutralizer]: 启用的 neutralizer 列表 (空表示无需中性化)
    """
    chain: List[Neutralizer] = []
    if if_industry:
        chain.append(IndustryNeutralizer(industry))
    if if_risk:
        chain.append(RiskNeutralizer(risk_data))
    return [n for n in chain if n.is_active()]


def apply_neutralizer_chain(
    factor_i: pd.DataFrame, chain: List[Neutralizer],
) -> pd.DataFrame:
    """执行 chain: 每个 neutralizer 顺序回归, 取残差作为新因子值.

    行为与原 _neutralize 三分支 (lines 74-135) 等价:
      - 3 个分支的差异在 X 组装 (build_design_matrix 各自负责)
      - 公共的"按日期循环 + merge + OLS + 写残差"在此统一

    Args:
        factor_i: 每日一行, index=日期, columns=股票代码
        chain: build_neutralizer_chain 返回的列表

    Returns:
        factor_neut: 残差矩阵 (空 chain 时返回 factor_i.copy() 保留 nan 模式)
    """
    factor_neut = factor_i.copy() * np.nan
    if not chain:
        return factor_neut

    for date_j in factor_i.index:
        if factor_i.loc[date_j].notna().sum() == 0:
            continue
        # 收集所有 neutralizer 的 X
        X_parts: List[pd.DataFrame] = []
        for neutralizer in chain:
            X_part = neutralizer.build_design_matrix(date_j, factor_i)
            if X_part is not None:
                X_parts.append(X_part)
        if not X_parts:
            continue
        # 合并 X (与原 branch 1 一致: merge suffixes=('', '_rf'))
        X = X_parts[0]
        for xp in X_parts[1:]:
            X = pd.merge(
                X, xp, left_index=True, right_index=True, suffixes=("", "_rf"),
            )
        # y + X 对齐 + dropna
        lm_data = pd.merge(
            factor_i.loc[date_j].to_frame(), X,
            left_index=True, right_index=True,
            suffixes=("_y", "_x"),
        ).dropna()
        # OLS 需要 lm_data 长度 > 参数数 (含常数项)
        if len(lm_data) > X.shape[1]:
            # 转 float: IndustryNeutralizer 的 dummies 是 bool, sm.add_constant
            # 在 bool 上报 "numpy boolean subtract" 错误. 转 float 修复此 bug
            # (原 _neutralize branch 2 同样会失败, 是 latent bug)
            X_values = lm_data.iloc[:, 1:].values.astype(float)
            model = sm.OLS(
                lm_data.iloc[:, 0].values,
                sm.add_constant(X_values),
            )
            resid = model.fit().resid
            factor_neut.loc[date_j, lm_data.index.values] = resid

    return factor_neut
