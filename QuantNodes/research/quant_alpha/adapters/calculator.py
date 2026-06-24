# coding=utf-8
"""
adapters/calculator.py - AlphaCalculator ABC + PolarsAlphaCalculator

参考：AlphaGen (KDD 2023, github.com/ICT-FinD-Lab/alphagen)
源文件: alphagen/data/calculator.py::AlphaCalculator

AlphaGen 用 7 个抽象方法把"数据后端"和"RL 训练"解耦：
- calc_single_IC_ret: 单因子 IC（与单期前瞻收益）
- calc_single_rIC_ret: 单因子 rank IC
- calc_single_all_ret: 单因子 vs 多期前瞻
- calc_mutual_IC: 两因子互 IC（用于去重）
- calc_pool_IC_ret: 因子集合 ensemble IC（joint 优化用）
- calc_pool_rIC_ret: 因子集合 ensemble rank IC
- calc_pool_all_ret: 因子集合 vs 多期前瞻

AlphaGen 默认实现用 qlib 数据后端，本模块用 QuantNodes
的 OperatorVocab + polars 重新实现，让 QuantNodes 算子可被
任何 RL 训练脚本使用（无需 qlib 依赖）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl

from QuantNodes.research.quant_alpha.adapters.expression import (
    Expression,
    expression_to_formula,
)
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab

logger = logging.getLogger(__name__)


# ==============================================================================
# 抽象基类
# ==============================================================================


class BaseAlphaCalculator(ABC):
    """AlphaGen 兼容的 AlphaCalculator 抽象基类

    任何 RL 训练（AlphaGen / Alpha² / 自研）只需实现这 7 个方法，
    即可用 QuantNodes 算子 + factor_test 评估。

    M4 范围：仅提供 ABC 接口。
    PolarsAlphaCalculator 是参考实现。
    """

    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression, ret_offset: int = 1) -> np.ndarray:
        """单因子 IC（与单期前瞻收益的 Pearson 相关系数序列）

        Args:
            expr: 因子表达式
            ret_offset: 前瞻期数（1 = 1 日后，5 = 5 日后）

        Returns:
            np.ndarray of shape (n_dates,): 每个日期的截面 IC
        """
        raise NotImplementedError

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression, ret_offset: int = 1) -> np.ndarray:
        """单因子 rank IC（Spearman 秩相关）

        Args:
            expr: 因子表达式
            ret_offset: 前瞻期数

        Returns:
            np.ndarray of shape (n_dates,): 每个日期的截面 rank IC
        """
        raise NotImplementedError

    @abstractmethod
    def calc_single_all_ret(self, expr: Expression) -> np.ndarray:
        """单因子 vs 多期前瞻（多步 IC）

        Args:
            expr: 因子表达式

        Returns:
            np.ndarray of shape (n_dates, n_ret_offsets): 每对 (date, offset) 的 IC
        """
        raise NotImplementedError

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> np.ndarray:
        """两因子互 IC（用于去重）

        Args:
            expr1, expr2: 两个因子表达式

        Returns:
            np.ndarray of shape (n_dates,): 每个日期的两因子 Pearson 相关
        """
        raise NotImplementedError

    @abstractmethod
    def calc_pool_IC_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
        ret_offset: int = 1,
    ) -> np.ndarray:
        """因子集合的 ensemble IC（joint 优化用）

        ensemble = sum(weight * factor_i)

        Args:
            exprs: 因子表达式列表
            weights: 因子权重（None=等权）
            ret_offset: 前瞻期数

        Returns:
            np.ndarray of shape (n_dates,): 每个日期的 ensemble IC
        """
        raise NotImplementedError

    @abstractmethod
    def calc_pool_rIC_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
        ret_offset: int = 1,
    ) -> np.ndarray:
        """因子集合的 ensemble rank IC

        Args:
            exprs: 因子表达式列表
            weights: 因子权重（None=等权）
            ret_offset: 前瞻期数

        Returns:
            np.ndarray of shape (n_dates,): 每个日期的 ensemble rank IC
        """
        raise NotImplementedError

    @abstractmethod
    def calc_pool_all_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """因子集合 vs 多期前瞻（多步 ensemble IC）

        Args:
            exprs: 因子表达式列表
            weights: 因子权重（None=等权）

        Returns:
            np.ndarray of shape (n_dates, n_ret_offsets)
        """
        raise NotImplementedError


# ==============================================================================
# Polars 实现
# ==============================================================================


class PolarsAlphaCalculator(BaseAlphaCalculator):
    """基于 polars + OperatorVocab 的 AlphaCalculator 参考实现

    设计原则：
    - 用 OperatorVocab.evaluate() 计算因子值（162 算子 + per-date over()）
    - 用 polars 原生 corr() 计算截面 IC
    - 用 per-date cross_sectional 语义（与 M1 修复一致）

    示例：
        >>> from QuantNodes.research.quant_alpha.adapters import PolarsAlphaCalculator
        >>> from QuantNodes.research.quant_alpha.adapters.expression import *
        >>> import polars as pl
        >>> df = pl.DataFrame({...})  # 行情数据
        >>> forward_returns = {1: df["forward_return_1d"]}  # 前瞻收益
        >>> calc = PolarsAlphaCalculator(df, forward_returns)
        >>> expr = Sub(Ref(Feature("close"), 5), Feature("close"))
        >>> ic = calc.calc_single_IC_ret(expr, ret_offset=1)
    """

    def __init__(
        self,
        data: pl.DataFrame,
        forward_returns: Dict[int, pl.Series],
        date_column: str = "date",
        code_column: str = "code",
        vocab: Optional[OperatorVocab] = None,
        cross_sectional: bool = True,
    ):
        """
        Args:
            data: 行情数据（必须包含 date_column）
            forward_returns: {前瞻期: 前瞻收益 Series} 映射
                例: {1: ret_1d_series, 5: ret_5d_series}
            date_column: 日期列名
            code_column: 股票代码列名
            vocab: OperatorVocab 实例（None=默认）
            cross_sectional: rank/zscore 是否 per-date 截面
        """
        # 重要：按 (code_column, date_column) 排序，确保 ts_*/rolling_* 算子
        # 在每只股票内独立计算（否则跨股票滚动会出错）
        self.data = data.sort([code_column, date_column])
        self.forward_returns = forward_returns
        self.date_column = date_column
        self.code_column = code_column
        self.vocab = vocab or OperatorVocab.default()
        self.cross_sectional = cross_sectional
        # 缓存：避免重复计算
        self._factor_cache: Dict[str, pl.Series] = {}

    # ==================================================================
    # 核心：因子计算（带缓存）
    # ==================================================================

    def _evaluate_factor(
        self,
        expr: Expression,
    ) -> Optional[pl.Series]:
        """计算因子值（带缓存）"""
        formula = expression_to_formula(expr)
        if formula in self._factor_cache:
            return self._factor_cache[formula]
        try:
            result = self.vocab.evaluate(
                formula=formula,
                data=self.data,
                date_column=self.date_column,
                code_column=self.code_column,
                cross_sectional=self.cross_sectional,
            )
            if result is not None:
                self._factor_cache[formula] = result
            return result
        except Exception as e:
            logger.debug("Factor eval failed: formula=%r, error=%s", formula, e)
            return None

    def _evaluate_factors(
        self,
        exprs: List[Expression],
    ) -> List[Optional[pl.Series]]:
        """计算多个因子值"""
        return [self._evaluate_factor(e) for e in exprs]

    # ==================================================================
    # 截面 IC 计算（per-date）
    # ==================================================================

    def _per_date_pearson_ic(
        self,
        factor: pl.Series,
        target: pl.Series,
    ) -> np.ndarray:
        """per-date Pearson IC

        Returns:
            np.ndarray: 每个日期的 IC（无效日期为 NaN）
        """
        df = self.data.select([
            pl.col(self.date_column).alias("_d"),
        ]).with_columns([
            factor.alias("_f"),
            target.alias("_t"),
        ]).filter(
            pl.col("_f").is_not_null() & pl.col("_t").is_not_null()
        )
        if len(df) == 0:
            return np.array([], dtype=np.float64)

        per_date = df.group_by("_d").agg(
            pl.corr("_f", "_t").alias("_ic")
        ).sort("_d")

        return per_date["_ic"].to_numpy().astype(np.float64)

    def _per_date_spearman_ic(
        self,
        factor: pl.Series,
        target: pl.Series,
    ) -> np.ndarray:
        """per-date Spearman rank IC

        Returns:
            np.ndarray: 每个日期的 rank IC
        """
        df = self.data.select([
            pl.col(self.date_column).alias("_d"),
        ]).with_columns([
            factor.rank().alias("_f"),
            target.rank().alias("_t"),
        ]).filter(
            pl.col("_f").is_not_null() & pl.col("_t").is_not_null()
        )
        if len(df) == 0:
            return np.array([], dtype=np.float64)

        per_date = df.group_by("_d").agg(
            pl.corr("_f", "_t").alias("_ic")
        ).sort("_d")

        return per_date["_ic"].to_numpy().astype(np.float64)

    def _per_date_pearson_corr(
        self,
        f1: pl.Series,
        f2: pl.Series,
    ) -> np.ndarray:
        """per-date 两因子互 IC（Pearson）"""
        df = self.data.select([
            pl.col(self.date_column).alias("_d"),
        ]).with_columns([
            f1.alias("_f1"),
            f2.alias("_f2"),
        ]).filter(
            pl.col("_f1").is_not_null() & pl.col("_f2").is_not_null()
        )
        if len(df) == 0:
            return np.array([], dtype=np.float64)

        per_date = df.group_by("_d").agg(
            pl.corr("_f1", "_f2").alias("_ic")
        ).sort("_d")

        return per_date["_ic"].to_numpy().astype(np.float64)

    # ==================================================================
    # 7 个 ABC 方法实现
    # ==================================================================

    def calc_single_IC_ret(
        self, expr: Expression, ret_offset: int = 1,
    ) -> np.ndarray:
        factor = self._evaluate_factor(expr)
        if factor is None:
            return np.array([], dtype=np.float64)
        ret = self.forward_returns.get(ret_offset)
        if ret is None:
            raise ValueError(
                f"No forward return for offset {ret_offset}. "
                f"Available: {list(self.forward_returns.keys())}"
            )
        return self._per_date_pearson_ic(factor, ret)

    def calc_single_rIC_ret(
        self, expr: Expression, ret_offset: int = 1,
    ) -> np.ndarray:
        factor = self._evaluate_factor(expr)
        if factor is None:
            return np.array([], dtype=np.float64)
        ret = self.forward_returns.get(ret_offset)
        if ret is None:
            raise ValueError(
                f"No forward return for offset {ret_offset}. "
                f"Available: {list(self.forward_returns.keys())}"
            )
        return self._per_date_spearman_ic(factor, ret)

    def calc_single_all_ret(
        self, expr: Expression,
    ) -> np.ndarray:
        """多步 IC：(n_dates, n_ret_offsets)"""
        factor = self._evaluate_factor(expr)
        if factor is None:
            n_dates = self.data[self.date_column].n_unique()
            return np.full(
                (n_dates, len(self.forward_returns)),
                np.nan, dtype=np.float64,
            )
        results = []
        for offset in sorted(self.forward_returns.keys()):
            ret = self.forward_returns[offset]
            ic = self._per_date_pearson_ic(factor, ret)
            results.append(ic)
        # 对齐长度
        max_len = max(len(r) for r in results)
        padded = []
        for r in results:
            if len(r) < max_len:
                r = np.concatenate([r, np.full(max_len - len(r), np.nan)])
            padded.append(r)
        return np.stack(padded, axis=-1)  # (n_dates, n_offsets)

    def calc_mutual_IC(
        self, expr1: Expression, expr2: Expression,
    ) -> np.ndarray:
        f1 = self._evaluate_factor(expr1)
        f2 = self._evaluate_factor(expr2)
        if f1 is None or f2 is None:
            return np.array([], dtype=np.float64)
        return self._per_date_pearson_corr(f1, f2)

    def _build_ensemble(
        self,
        factors: List[pl.Series],
        weights: List[float],
    ) -> pl.Series:
        """构造 ensemble = sum(weight_i * factor_i)"""
        if not factors:
            raise ValueError("No factors to ensemble")
        # 过滤掉 None
        valid = [(f, w) for f, w in zip(factors, weights) if f is not None]
        if not valid:
            raise ValueError("No valid factors to ensemble")
        # 等权归一化
        total_w = sum(w for _, w in valid)
        if total_w == 0:
            weights = [1.0 / len(valid)] * len(valid)
        else:
            weights = [w / total_w for _, w in valid]
        # 加权求和
        ensemble = None
        for (factor, _), w in zip(valid, weights):
            if ensemble is None:
                ensemble = w * factor.fill_null(0.0)
            else:
                ensemble = ensemble + w * factor.fill_null(0.0)
        return ensemble

    def calc_pool_IC_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
        ret_offset: int = 1,
    ) -> np.ndarray:
        if not exprs:
            raise ValueError("exprs is empty")
        if weights is None:
            weights = [1.0 / len(exprs)] * len(exprs)
        factors = self._evaluate_factors(exprs)
        ensemble = self._build_ensemble(factors, weights)
        ret = self.forward_returns.get(ret_offset)
        if ret is None:
            raise ValueError(
                f"No forward return for offset {ret_offset}. "
                f"Available: {list(self.forward_returns.keys())}"
            )
        return self._per_date_pearson_ic(ensemble, ret)

    def calc_pool_rIC_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
        ret_offset: int = 1,
    ) -> np.ndarray:
        if not exprs:
            raise ValueError("exprs is empty")
        if weights is None:
            weights = [1.0 / len(exprs)] * len(exprs)
        factors = self._evaluate_factors(exprs)
        ensemble = self._build_ensemble(factors, weights)
        ret = self.forward_returns.get(ret_offset)
        if ret is None:
            raise ValueError(
                f"No forward return for offset {ret_offset}. "
                f"Available: {list(self.forward_returns.keys())}"
            )
        return self._per_date_spearman_ic(ensemble, ret)

    def calc_pool_all_ret(
        self,
        exprs: List[Expression],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        if not exprs:
            raise ValueError("exprs is empty")
        if weights is None:
            weights = [1.0 / len(exprs)] * len(exprs)
        factors = self._evaluate_factors(exprs)
        ensemble = self._build_ensemble(factors, weights)
        # 多步前瞻
        results = []
        for offset in sorted(self.forward_returns.keys()):
            ret = self.forward_returns[offset]
            ic = self._per_date_pearson_ic(ensemble, ret)
            results.append(ic)
        max_len = max(len(r) for r in results)
        padded = []
        for r in results:
            if len(r) < max_len:
                r = np.concatenate([r, np.full(max_len - len(r), np.nan)])
            padded.append(r)
        return np.stack(padded, axis=-1)

    # ==================================================================
    # 工具方法
    # ==================================================================

    def stats(self) -> Dict[str, Any]:
        """计算器统计"""
        return {
            "n_factors_cached": len(self._factor_cache),
            "n_data_rows": len(self.data),
            "n_dates": self.data[self.date_column].n_unique(),
            "n_codes": self.data[self.code_column].n_unique(),
            "forward_returns": sorted(self.forward_returns.keys()),
            "cross_sectional": self.cross_sectional,
        }
