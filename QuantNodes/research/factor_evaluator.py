# coding=utf-8
"""
因子评估器 - 6维度评估 + 相关性去重

维度: 收益、稳定性、分散度、换手率、单调性、覆盖率
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from QuantNodes.research.factor_miner import FactorCandidate


@dataclass
class FactorEvaluationResult:
    """6维度因子评估结果"""
    candidate: FactorCandidate
    factor_values: Optional[pl.Series] = None

    # 维度1: 收益
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    rank_ic_mean: float = 0.0

    # 维度2: 稳定性
    rolling_ic_mean: float = 0.0
    rolling_ic_std: float = 0.0
    stability_score: float = 0.0

    # 维度3: 分散度
    avg_corr_with_existing: float = 0.0
    diversification_score: float = 1.0

    # 维度4: 换手率
    turnover: float = 0.0
    turnover_cost: float = 0.0

    # 维度5: 单调性
    group_returns: List[float] = field(default_factory=list)
    monotonicity_score: float = 0.0

    # 维度6: 覆盖率
    coverage: float = 0.0

    # 综合判定
    is_valid: bool = False
    fail_reasons: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """评估配置"""
    ic_threshold: float = 0.03
    icir_threshold: float = 0.5
    stability_threshold: float = 0.6
    corr_threshold: float = 0.7
    turnover_threshold: float = 0.5
    monotonicity_threshold: float = 0.7
    coverage_threshold: float = 0.8
    n_groups: int = 5
    rolling_window: int = 20

    # 综合评分权重
    weights: Dict[str, float] = field(default_factory=lambda: {
        "return": 0.30,
        "stability": 0.20,
        "diversification": 0.20,
        "turnover": 0.15,
        "monotonicity": 0.10,
        "coverage": 0.05,
    })


class FactorEvaluator:
    """6维度因子评估器"""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()

    def evaluate(
        self,
        candidate: FactorCandidate,
        data: pl.DataFrame,
        date_column: str = "date",
        code_column: str = "code",
        forward_return_column: str = "forward_return",
        existing_factors: Optional[List[pl.Series]] = None,
    ) -> FactorEvaluationResult:
        """评估单个候选因子"""
        result = FactorEvaluationResult(candidate=candidate)

        try:
            # 计算因子值
            factor_values = self._compute_factor(candidate.formula, data)
            if factor_values is None:
                result.fail_reasons.append("公式计算失败")
                return result

            result.factor_values = factor_values

            # 合并因子值和收益率
            eval_df = data.with_columns(factor_values.alias("_factor"))

            # 过滤无效值
            eval_df = eval_df.filter(
                pl.col("_factor").is_not_null() &
                pl.col(forward_return_column).is_not_null()
            )

            if len(eval_df) < 10:
                result.fail_reasons.append("有效数据不足")
                return result

            # 维度6: 覆盖率
            result.coverage = self._compute_coverage(data, factor_values)
            result.dimension_scores["coverage"] = result.coverage

            # 维度1: 收益 (IC/IR)
            self._compute_return_dimension(result, eval_df, date_column)
            result.dimension_scores["return"] = min(abs(result.icir) / 1.0, 1.0)

            # 维度2: 稳定性
            self._compute_stability_dimension(result, eval_df, date_column)
            result.dimension_scores["stability"] = result.stability_score

            # 维度4: 换手率
            self._compute_turnover_dimension(result, eval_df, date_column, code_column)
            result.dimension_scores["turnover"] = max(1.0 - result.turnover, 0.0)

            # 维度5: 单调性
            self._compute_monotonicity_dimension(result, eval_df, forward_return_column)
            result.dimension_scores["monotonicity"] = result.monotonicity_score

            # 维度3: 分散度
            if existing_factors:
                self._compute_diversification_dimension(
                    result, factor_values, existing_factors
                )
            else:
                result.avg_corr_with_existing = 0.0
                result.diversification_score = 1.0
            result.dimension_scores["diversification"] = result.diversification_score

            # 综合评分
            result.overall_score = self._compute_overall_score(result)
            result.is_valid = self._check_validity(result)

        except Exception as e:
            result.fail_reasons.append(f"评估异常: {e}")

        return result

    def deduplicate(
        self,
        results: List[FactorEvaluationResult],
        corr_threshold: float = None,
    ) -> List[FactorEvaluationResult]:
        """相关性去重 (贪心聚类)"""
        threshold = corr_threshold or self.config.corr_threshold

        # 按综合评分降序排序
        sorted_results = sorted(
            results, key=lambda r: r.overall_score, reverse=True
        )

        selected: List[FactorEvaluationResult] = []
        for r in sorted_results:
            if r.factor_values is None:
                continue

            # 检查与已选因子的相关性
            is_duplicate = False
            for s in selected:
                if s.factor_values is None:
                    continue
                corr = self._spearman_corr(r.factor_values, s.factor_values)
                if abs(corr) > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(r)

        return selected

    # ==================== 因子计算 ====================

    def _compute_factor(
        self, formula: str, data: pl.DataFrame
    ) -> Optional[pl.Series]:
        """安全计算因子值"""
        try:
            namespace = {
                "pl": pl,
                "ts_mean": lambda col, w: col.rolling_mean(w),
                "ts_std": lambda col, w: col.rolling_std(w),
                "ts_max": lambda col, w: col.rolling_max(w),
                "ts_min": lambda col, w: col.rolling_min(w),
                "ts_delta": lambda col, w: col - col.shift(w),
                "ts_lag": lambda col, w: col.shift(w),
                "ts_pct_change": lambda col, w: col.pct_change(w),
                "ts_corr": lambda c1, c2, w: c1.rolling_corr(c2, w),
                "ts_cov": lambda c1, c2, w: c1.rolling_cov(c2, w),
                "rank": lambda col: col.rank(),
                "zscore": lambda col: (col - col.mean()) / (col.std() + 1e-8),
            }
            for col_name in data.columns:
                namespace[col_name] = data[col_name]

            result = eval(formula, {"__builtins__": {}}, namespace)
            if isinstance(result, pl.Series):
                return result
            if isinstance(result, pl.Expr):
                return data.select(result).to_series()
            return None
        except Exception:
            return None

    # ==================== 维度1: 收益 ====================

    def _compute_return_dimension(
        self,
        result: FactorEvaluationResult,
        eval_df: pl.DataFrame,
        date_column: str,
    ):
        """计算 IC/IR/Rank IC"""
        # 按日期分组计算 IC
        ic_series = (
            eval_df
            .group_by(date_column)
            .agg(
                pl.corr("_factor", "forward_return").alias("ic")
            )
            .sort(date_column)
        )

        ic_values = ic_series["ic"].drop_nulls()
        if len(ic_values) == 0:
            return

        result.ic_mean = float(ic_values.mean())
        result.ic_std = float(ic_values.std()) if len(ic_values) > 1 else 0.0
        result.icir = (
            result.ic_mean / (result.ic_std + 1e-8) if result.ic_std > 0 else 0.0
        )

        # Rank IC
        rank_ic_series = (
            eval_df
            .group_by(date_column)
            .agg(
                pl.corr(
                    pl.col("_factor").rank(),
                    pl.col("forward_return").rank(),
                ).alias("rank_ic")
            )
            .sort(date_column)
        )
        rank_ic_values = rank_ic_series["rank_ic"].drop_nulls()
        if len(rank_ic_values) > 0:
            result.rank_ic_mean = float(rank_ic_values.mean())

    # ==================== 维度2: 稳定性 ====================

    def _compute_stability_dimension(
        self,
        result: FactorEvaluationResult,
        eval_df: pl.DataFrame,
        date_column: str,
    ):
        """计算滚动IC的稳定性"""
        # 按日期分组计算 IC
        ic_by_date = (
            eval_df
            .group_by(date_column)
            .agg(pl.corr("_factor", "forward_return").alias("ic"))
            .sort(date_column)
        )

        ic_values = ic_by_date["ic"].drop_nulls().to_list()
        if len(ic_values) < self.config.rolling_window:
            result.stability_score = 0.0
            return

        # 滚动IC
        rolling_ics = []
        for i in range(self.config.rolling_window, len(ic_values) + 1):
            window = ic_values[i - self.config.rolling_window : i]
            rolling_ics.append(np.mean(window))

        if rolling_ics:
            result.rolling_ic_mean = float(np.mean(rolling_ics))
            result.rolling_ic_std = float(np.std(rolling_ics))

            # 稳定性 = 1 - (滚动IC的标准差 / 均值的绝对值)
            mean_abs = abs(result.rolling_ic_mean) + 1e-8
            result.stability_score = max(0.0, 1.0 - result.rolling_ic_std / mean_abs)

    # ==================== 维度3: 分散度 ====================

    def _compute_diversification_dimension(
        self,
        result: FactorEvaluationResult,
        factor_values: pl.Series,
        existing_factors: List[pl.Series],
    ):
        """计算与已有因子的平均相关性"""
        if not existing_factors:
            result.avg_corr_with_existing = 0.0
            result.diversification_score = 1.0
            return

        corrs = []
        for ef in existing_factors:
            corr = self._spearman_corr(factor_values, ef)
            if not math.isnan(corr):
                corrs.append(abs(corr))

        if corrs:
            result.avg_corr_with_existing = float(np.mean(corrs))
            result.diversification_score = max(0.0, 1.0 - result.avg_corr_with_existing)
        else:
            result.avg_corr_with_existing = 0.0
            result.diversification_score = 1.0

    # ==================== 维度4: 换手率 ====================

    def _compute_turnover_dimension(
        self,
        result: FactorEvaluationResult,
        eval_df: pl.DataFrame,
        date_column: str,
        code_column: str,
    ):
        """计算因子排名变化率 (换手率)"""
        # 按日期计算排名
        ranked = eval_df.with_columns(
            pl.col("_factor").rank().alias("_rank")
        )

        # 按股票分组，计算排名变化
        dates = sorted(ranked[date_column].unique().to_list())
        if len(dates) < 2:
            result.turnover = 0.0
            return

        rank_changes = []
        for i in range(1, min(len(dates), 50)):  # 最多取50天
            prev = (
                ranked
                .filter(pl.col(date_column) == dates[i - 1])
                .select([code_column, "_rank"])
                .rename({"_rank": "prev_rank"})
            )
            curr = (
                ranked
                .filter(pl.col(date_column) == dates[i])
                .select([code_column, "_rank"])
                .rename({"_rank": "curr_rank"})
            )
            merged = prev.join(curr, on=code_column, how="inner")
            if len(merged) > 0:
                change = (
                    (merged["curr_rank"] - merged["prev_rank"]).abs().mean()
                    / len(merged)
                )
                rank_changes.append(float(change))

        if rank_changes:
            result.turnover = float(np.mean(rank_changes))
            result.turnover_cost = result.turnover * 0.001  # 简单估计

    # ==================== 维度5: 单调性 ====================

    def _compute_monotonicity_dimension(
        self,
        result: FactorEvaluationResult,
        eval_df: pl.DataFrame,
        forward_return_column: str,
    ):
        """计算5组分位收益的单调性"""
        n_groups = self.config.n_groups

        # 按因子值分组
        quantiles = eval_df.select(
            pl.col("_factor").quantile_segmentation(n_groups).alias("_group")
        )
        eval_df = eval_df.with_columns(quantiles["_group"])

        # 计算每组平均收益
        group_returns = (
            eval_df
            .group_by("_group")
            .agg(pl.col(forward_return_column).mean().alias("avg_return"))
            .sort("_group")
        )

        returns = group_returns["avg_return"].to_list()
        result.group_returns = [float(r) for r in returns if r is not None]

        # 计算单调性评分
        if len(result.group_returns) < 2:
            result.monotonicity_score = 0.0
            return

        # 计算 Spearman 秩相关 (组号 vs 平均收益)
        n = len(result.group_returns)
        x_rank = list(range(n))
        y_sorted = sorted(range(n), key=lambda i: result.group_returns[i])
        y_rank = [0] * n
        for rank, idx in enumerate(y_sorted):
            y_rank[idx] = rank

        # Spearman 相关
        d_sq_sum = sum((x_rank[i] - y_rank[i]) ** 2 for i in range(n))
        spearman = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))
        result.monotonicity_score = max(0.0, spearman)

    # ==================== 维度6: 覆盖率 ====================

    def _compute_coverage(
        self, data: pl.DataFrame, factor_values: pl.Series
    ) -> float:
        """计算因子值覆盖率"""
        total = len(factor_values)
        if total == 0:
            return 0.0
        non_null = factor_values.drop_nulls()
        return len(non_null) / total

    # ==================== 综合评分 ====================

    def _compute_overall_score(self, result: FactorEvaluationResult) -> float:
        """计算综合评分"""
        weights = self.config.weights
        score = 0.0

        for dim, weight in weights.items():
            if dim in result.dimension_scores:
                score += weight * result.dimension_scores[dim]

        return score

    def _check_validity(self, result: FactorEvaluationResult) -> bool:
        """检查是否通过所有维度阈值"""
        reasons = []

        if abs(result.ic_mean) < self.config.ic_threshold:
            reasons.append(f"IC {result.ic_mean:.4f} < {self.config.ic_threshold}")

        if abs(result.icir) < self.config.icir_threshold:
            reasons.append(f"ICIR {result.icir:.4f} < {self.config.icir_threshold}")

        if result.stability_score < self.config.stability_threshold:
            reasons.append(
                f"稳定性 {result.stability_score:.4f} < {self.config.stability_threshold}"
            )

        if result.avg_corr_with_existing > self.config.corr_threshold:
            reasons.append(
                f"相关性 {result.avg_corr_with_existing:.4f} > {self.config.corr_threshold}"
            )

        if result.turnover > self.config.turnover_threshold:
            reasons.append(
                f"换手率 {result.turnover:.4f} > {self.config.turnover_threshold}"
            )

        if result.monotonicity_score < self.config.monotonicity_threshold:
            reasons.append(
                f"单调性 {result.monotonicity_score:.4f} < {self.config.monotonicity_threshold}"
            )

        if result.coverage < self.config.coverage_threshold:
            reasons.append(
                f"覆盖率 {result.coverage:.4f} < {self.config.coverage_threshold}"
            )

        result.fail_reasons = reasons
        return len(reasons) == 0

    # ==================== 工具方法 ====================

    @staticmethod
    def _spearman_corr(a: pl.Series, b: pl.Series) -> float:
        """计算 Spearman 秩相关"""
        min_len = min(len(a), len(b))
        if min_len < 3:
            return 0.0

        a_vals = a[:min_len].to_list()
        b_vals = b[:min_len].to_list()

        # 过滤 None/NaN
        pairs = [(x, y) for x, y in zip(a_vals, b_vals) if x is not None and y is not None]
        if len(pairs) < 3:
            return 0.0

        a_clean, b_clean = zip(*pairs)
        n = len(a_clean)

        # 排序得到秩
        a_ranked = _rank(list(a_clean))
        b_ranked = _rank(list(b_clean))

        # Spearman 公式
        d_sq = sum((a_ranked[i] - b_ranked[i]) ** 2 for i in range(n))
        return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _rank(values: List[float]) -> List[float]:
    """计算平均秩"""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks
