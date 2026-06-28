# coding=utf-8
"""
polars_evaluator.py - Stage 1/2 通用 Polars 评估器

内部使用 OperatorVocab.evaluate() 直接评估公式，支持复杂表达式。
Stage 1 + Stage 2 共用此实现（不依赖 mock / real 数据）。

复用：
- QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab：162 算子
- contracts.FactorMetrics：统一输出 schema
- contracts.FactorSpec：输入因子列表
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl

from ..contracts import Evaluator, FactorMetrics, FactorSpec, VerifyConfig

logger = logging.getLogger(__name__)

__all__ = ["PolarsAlphaCalculatorEvaluator", "deduplicate_mutual_ic"]


# ==============================================================================
# 辅助函数
# ==============================================================================


def _spearman_corr(x: pl.Series, y: pl.Series) -> float:
    """Spearman 秩相关

    Args:
        x: 第一个序列
        y: 第二个序列

    Returns:
        相关系数 [-1, 1]
    """
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x_rank = x.head(n).rank()
    y_rank = y.head(n).rank()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def deduplicate_mutual_ic(
    factors: List[FactorMetrics],
    get_values: Callable[[FactorMetrics], Optional[pl.Series]],
    threshold: float = 0.7,
) -> List[FactorMetrics]:
    """贪心互信息去重

    按 overall_score 降序排序，逐个检查与已选因子的 Spearman 相关性。
    如果 |corr| > threshold，跳过该因子；否则加入已选集合。

    优化：缓存 get_values 结果，避免重复计算。

    Args:
        factors: 候选因子列表
        get_values: 获取因子值的函数
        threshold: 相关性阈值（默认 0.7）

    Returns:
        去重后的因子列表
    """
    values_cache: Dict[str, Optional[pl.Series]] = {}
    sorted_f = sorted(factors, key=lambda f: abs(f.overall_score), reverse=True)
    selected = []
    for f in sorted_f:
        if f.formula_id not in values_cache:
            values_cache[f.formula_id] = get_values(f)
        vals = values_cache[f.formula_id]
        if vals is None:
            continue
        is_dup = False
        for s in selected:
            if s.formula_id not in values_cache:
                values_cache[s.formula_id] = get_values(s)
            s_vals = values_cache[s.formula_id]
            if s_vals is None:
                continue
            corr = _spearman_corr(vals, s_vals)
            if abs(corr) > threshold:
                is_dup = True
                break
        if not is_dup:
            selected.append(f)
    return selected


class PolarsAlphaCalculatorEvaluator(Evaluator):
    """Stage 1/2 通用 Polars 评估器

    使用 OperatorVocab.evaluate() 直接评估公式，支持复杂表达式。
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._vocab = None  # lazy init

    def _get_vocab(self):
        """懒加载 OperatorVocab"""
        if self._vocab is None:
            from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
            self._vocab = OperatorVocab.default()
        return self._vocab

    def evaluate(
        self,
        factors: List[FactorSpec],
        data: Any,
        forward_returns: Optional[List[int]] = None,
    ) -> List[FactorMetrics]:
        """批量评估因子

        Args:
            factors: FactorSpec 列表
            data: polars.DataFrame（含 date / code / OHLCV 等列）
            forward_returns: 前瞻期列表（默认 [1]）

        Returns:
            FactorMetrics 列表（顺序与 factors 一一对应）
        """
        if not factors:
            return []

        vocab = self._get_vocab()
        fr = forward_returns or [1]
        date_column = "date"
        code_column = "code"

        logger.info(
            "[PolarsAlphaCalculatorEvaluator] 评估 %d 个公式 (forward_returns=%s)",
            len(factors),
            fr,
        )

        # 计算前瞻收益
        forward_return_series = {}
        for offset in fr:
                    col_name = f"forward_return_{offset}d"
                    if col_name in data.columns:
                        forward_return_series[offset] = data[col_name]
                    else:
                        # 计算前瞻收益: close(t+offset) / close(t) - 1
                        sorted_data = data.sort([code_column, date_column])
                        fwd = sorted_data.with_columns(
                            pl.col("close")
                            .shift(-offset)
                            .over(code_column)
                            .alias("_fwd_close")
                        )
                        forward_return_series[offset] = (
                            (fwd["_fwd_close"] / sorted_data["close"]) - 1.0
                        )

        out: List[FactorMetrics] = []
        for factor in factors:
            try:
                # 评估因子值
                factor_values = vocab.evaluate(
                    formula=factor.formula,
                    data=data,
                    date_column=date_column,
                    code_column=code_column,
                )

                if factor_values is None or len(factor_values) != len(data):
                    out.append(FactorMetrics(
                        formula_id=factor.formula_id,
                        status="failed",
                        error_msg="Factor evaluation returned None or wrong length",
                    ))
                    continue

                # 计算 IC (向量化 per-date)
                ic_results = {}
                for offset in fr:
                    fwd_ret = forward_return_series[offset]
                    
                    # 构造临时 DataFrame 计算 per-date IC
                    tmp = pl.DataFrame({
                        "_date": data[date_column],
                        "_factor": factor_values,
                        "_fwd": fwd_ret,
                    }).drop_nulls()

                    if len(tmp) == 0:
                        continue

                    # per-date corr (groupby)
                    try:
                        daily_corr = (
                            tmp
                            .group_by("_date")
                            .agg([
                                pl.corr("_factor", "_fwd").alias("_corr"),
                                pl.len().alias("_n"),
                            ])
                            .filter(pl.col("_n") >= 3)
                            .drop_nulls()
                        )

                        if len(daily_corr) == 0:
                            continue

                        ics = daily_corr["_corr"].to_list()
                        ic_mean = float(np.mean(ics))
                        ic_std = float(np.std(ics))
                        ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
                        ic_results[offset] = {
                            "ic_mean": ic_mean,
                            "ic_std": ic_std,
                            "ir": ir,
                        }

                        # 计算 Rank IC（Spearman 秩相关）
                        try:
                            rank_corr = (
                                tmp
                                .with_columns([
                                    pl.col("_factor").rank().alias("_factor_rank"),
                                    pl.col("_fwd").rank().alias("_fwd_rank"),
                                ])
                                .group_by("_date")
                                .agg(pl.corr("_factor_rank", "_fwd_rank").alias("_rank_corr"))
                                .drop_nulls()
                            )
                            if len(rank_corr) > 0:
                                rank_ics = rank_corr["_rank_corr"].to_list()
                                ic_results[offset]["rank_ic_mean"] = float(np.mean(rank_ics))
                            else:
                                ic_results[offset]["rank_ic_mean"] = 0.0
                        except Exception as e:
                            logger.debug("Rank IC calc failed for offset %d: %s", offset, e)
                            ic_results[offset]["rank_ic_mean"] = 0.0

                    except Exception as e:
                        logger.debug("IC calc failed for offset %d: %s", offset, e)
                        continue

                if ic_results:
                    primary = ic_results[fr[0]]
                    out.append(FactorMetrics(
                        formula_id=factor.formula_id,
                        status="success",
                        ic_mean=primary["ic_mean"],
                        ic_std=primary["ic_std"],
                        ir=primary["ir"],
                        ic_decay={str(k): v["ic_mean"] for k, v in ic_results.items()},
                        rank_ic_mean=primary.get("rank_ic_mean", 0.0),
                    ))
                else:
                    out.append(FactorMetrics(
                        formula_id=factor.formula_id,
                        status="failed",
                        error_msg="No valid IC computed",
                    ))

            except Exception as e:
                logger.debug("[PolarsAlphaCalculatorEvaluator] 公式评估失败: %s - %s", factor.formula, e)
                out.append(FactorMetrics(
                    formula_id=factor.formula_id,
                    status="failed",
                    error_msg=str(e),
                ))

        n_success = sum(1 for m in out if m.status == "success")
        logger.info(
            "[PolarsAlphaCalculatorEvaluator] 完成: %d/%d success",
            n_success,
            len(out),
        )

        return out

    def verify(
        self,
        metrics: FactorMetrics,
        data: Any,
        factor_values: Any = None,
        existing_factors: Optional[List[Any]] = None,
        config: Optional[VerifyConfig] = None,
    ) -> FactorMetrics:
        """6 维验证（从 _legacy_3c 迁移）

        Args:
            metrics: 已计算的 IC/IR 指标
            data: 原始数据 DataFrame
            factor_values: 因子值 Series（可选，用于计算 turnover/monotonicity）
            existing_factors: 已有因子值列表（可选，用于计算 diversification）
            config: 验证阈值配置

        Returns:
            更新后的 FactorMetrics（含 6 维分数和 is_valid）
        """
        if metrics.status != "success":
            metrics.is_valid = False
            metrics.fail_reasons = [metrics.error_msg or "IC 计算失败"]
            return metrics

        cfg = config or VerifyConfig()
        fail_reasons = []

        # 1. 收益维度（已有 IC/IR）
        return_score = min(abs(metrics.ir) / 1.0, 1.0)
        if abs(metrics.ic_mean) < cfg.ic_threshold:
            fail_reasons.append(f"IC {abs(metrics.ic_mean):.4f} < {cfg.ic_threshold}")
        if abs(metrics.ir) < cfg.icir_threshold:
            fail_reasons.append(f"IR {abs(metrics.ir):.4f} < {cfg.icir_threshold}")

        # 2. 稳定性维度（滚动 IC）
        stability_score = self._compute_stability(data, factor_values, cfg)
        metrics.stability_score = stability_score
        if stability_score < cfg.stability_threshold:
            fail_reasons.append(f"稳定性 {stability_score:.4f} < {cfg.stability_threshold}")

        # 3. 分散度维度（与已有因子相关性）
        diversification_score = self._compute_diversification(
            factor_values, existing_factors, cfg
        )
        metrics.diversification_score = diversification_score
        if diversification_score < (1.0 - cfg.corr_threshold):
            fail_reasons.append(f"分散度 {diversification_score:.4f} < {1.0 - cfg.corr_threshold}")

        # 4. 换手率维度
        turnover = self._compute_turnover(data, factor_values, cfg)
        metrics.turnover = turnover
        if turnover > cfg.turnover_threshold:
            fail_reasons.append(f"换手率 {turnover:.4f} > {cfg.turnover_threshold}")

        # 5. 单调性维度
        monotonicity_score = self._compute_monotonicity(data, factor_values, cfg)
        metrics.monotonicity_score = monotonicity_score
        if monotonicity_score < cfg.monotonicity_threshold:
            fail_reasons.append(f"单调性 {monotonicity_score:.4f} < {cfg.monotonicity_threshold}")

        # 6. 覆盖率维度
        coverage = self._compute_coverage(data, factor_values)
        metrics.coverage = coverage
        if coverage < cfg.coverage_threshold:
            fail_reasons.append(f"覆盖率 {coverage:.4f} < {cfg.coverage_threshold}")

        # 计算综合分数
        weights = cfg.weights
        metrics.overall_score = (
            weights["return"] * return_score
            + weights["stability"] * stability_score
            + weights["diversification"] * diversification_score
            + weights["turnover"] * (1.0 - turnover)
            + weights["monotonicity"] * monotonicity_score
            + weights["coverage"] * coverage
        )

        metrics.is_valid = len(fail_reasons) == 0
        metrics.fail_reasons = fail_reasons

        return metrics

    def _compute_stability(
        self, data: Any, factor_values: Any, config: VerifyConfig
    ) -> float:
        """计算稳定性分数（滚动 IC 标准差）"""
        if factor_values is None:
            return 0.0

        try:
            # 计算每日 IC
            tmp = pl.DataFrame({
                "_date": data["date"],
                "_factor": factor_values,
                "_fwd": data.get("forward_return", pl.Series([0.0] * len(data))),
            }).drop_nulls()

            daily_ic = (
                tmp
                .group_by("_date")
                .agg(pl.corr("_factor", "_fwd").alias("_ic"))
                .sort("_date")
                .drop_nulls()
            )

            if len(daily_ic) < config.rolling_window:
                return 0.0

            # 滚动窗口计算稳定性
            ics = daily_ic["_ic"].to_numpy()
            rolling_std = np.array([
                np.std(ics[max(0, i - config.rolling_window):i + 1])
                for i in range(len(ics))
            ])
            rolling_mean = np.array([
                np.mean(ics[max(0, i - config.rolling_window):i + 1])
                for i in range(len(ics))
            ])

            # 稳定性 = 1 - 滚动标准差 / |滚动均值|
            stability = np.mean(
                np.maximum(0, 1 - rolling_std / (np.abs(rolling_mean) + 1e-8))
            )
            return float(stability)

        except Exception as e:
            logger.debug("稳定性计算失败: %s", e)
            return 0.0

    def _compute_diversification(
        self, factor_values: Any, existing_factors: Optional[List[Any]], config: VerifyConfig
    ) -> float:
        """计算分散度分数（与已有因子的相关性）"""
        if factor_values is None or not existing_factors:
            return 1.0  # 没有已有因子时默认满分

        try:
            corrs = []
            for existing in existing_factors:
                if existing is not None and len(existing) == len(factor_values):
                    # 计算 Spearman 相关性
                    corr = np.corrcoef(
                        np.argsort(np.argsort(factor_values)),
                        np.argsort(np.argsort(existing))
                    )[0, 1]
                    corrs.append(abs(corr))

            if not corrs:
                return 1.0

            avg_corr = np.mean(corrs)
            return float(max(0, 1 - avg_corr))

        except Exception as e:
            logger.debug("分散度计算失败: %s", e)
            return 0.0

    def _compute_turnover(
        self, data: Any, factor_values: Any, config: VerifyConfig
    ) -> float:
        """计算换手率（排名变化率）"""
        if factor_values is None:
            return 1.0

        try:
            # 按日期分组计算排名
            tmp = pl.DataFrame({
                "_date": data["date"],
                "_code": data["code"],
                "_factor": factor_values,
            })

            # 全局排名
            tmp = tmp.with_columns(pl.col("_factor").rank().alias("_rank"))

            # 计算相邻日期的排名变化
            dates = tmp["_date"].unique().sort()
            if len(dates) < 2:
                return 0.0

            turnovers = []
            for i in range(min(len(dates) - 1, 50)):  # 最多计算 50 天
                # 按股票代码 join（正确处理股票池变化）
                curr_df = tmp.filter(pl.col("_date") == dates[i]).select(["_code", "_rank"])
                prev_df = tmp.filter(pl.col("_date") == dates[i + 1]).select(["_code", "_rank"])
                merged = curr_df.join(prev_df, on="_code", suffix="_prev")
                
                if len(merged) > 0:
                    turnover = (merged["_rank"] - merged["_rank_prev"]).abs().mean() / len(merged)
                    turnovers.append(turnover)

            return float(max(0, np.mean(turnovers))) if turnovers else 0.0

        except Exception as e:
            logger.debug("换手率计算失败: %s", e)
            return 1.0

    def _compute_monotonicity(
        self, data: Any, factor_values: Any, config: VerifyConfig
    ) -> float:
        """计算单调性分数（分组收益单调性）"""
        if factor_values is None:
            return 0.0

        try:
            # 构造临时 DataFrame
            tmp = pl.DataFrame({
                "_factor": factor_values,
                "_fwd": data.get("forward_return", pl.Series([0.0] * len(data))),
            }).drop_nulls()

            if len(tmp) < config.n_groups * 10:
                return 0.0

            # 分组
            tmp = tmp.with_columns(
                pl.col("_factor")
                .qcut(config.n_groups, labels=[str(i) for i in range(config.n_groups)])
                .alias("_group")
            )

            # 计算每组平均收益
            group_returns = (
                tmp
                .group_by("_group")
                .agg(pl.col("_fwd").mean().alias("_mean_return"))
                .sort("_group")
            )

            if len(group_returns) < config.n_groups:
                return 0.0

            # 计算 Spearman 相关性（组号 vs 组收益）
            groups = list(range(len(group_returns)))
            returns = group_returns["_mean_return"].to_list()
            corr = np.corrcoef(groups, returns)[0, 1]

            return float(max(0, corr))

        except Exception as e:
            logger.debug("单调性计算失败: %s", e)
            return 0.0

    def _compute_coverage(self, data: Any, factor_values: Any) -> float:
        """计算覆盖率（非空比例）"""
        if factor_values is None:
            return 0.0

        try:
            if isinstance(factor_values, pl.Series):
                non_null = factor_values.drop_nulls().len()
                total = len(factor_values)
            else:
                # numpy array
                non_null = np.count_nonzero(~np.isnan(factor_values))
                total = len(factor_values)

            return float(non_null / total) if total > 0 else 0.0

        except Exception as e:
            logger.debug("覆盖率计算失败: %s", e)
            return 0.0