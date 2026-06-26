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
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..contracts import Evaluator, FactorMetrics, FactorSpec

logger = logging.getLogger(__name__)

__all__ = ["PolarsAlphaCalculatorEvaluator"]


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
                forward_return_series[offset] = data[col_name].to_list()
            else:
                # 计算前瞻收益: close(t+offset) / close(t) - 1
                sorted_data = data.sort([code_column, date_column])
                fwd_returns = [None] * len(sorted_data)
                
                # 按股票分组计算
                for code in sorted_data[code_column].unique().sort():
                    mask = sorted_data[code_column] == code
                    stock_indices = [i for i, x in enumerate(sorted_data[code_column].to_list()) if x == code]
                    stock_closes = sorted_data.filter(mask)['close'].to_list()
                    
                    for j, idx in enumerate(stock_indices):
                        if j + offset < len(stock_closes):
                            fwd_returns[idx] = (stock_closes[j + offset] / stock_closes[j]) - 1.0
                
                forward_return_series[offset] = fwd_returns

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

                # 计算 IC
                ic_results = {}
                for offset in fr:
                    fwd_ret = forward_return_series.get(offset)
                    if fwd_ret is None:
                        continue

                    # per-date IC
                    dates = data[date_column].unique().sort()
                    daily_ics = []
                    for d in dates:
                        mask = data[date_column] == d
                        # 获取当前日期的因子值和前瞻收益
                        fv = factor_values.filter(mask).to_list()
                        # fwd_ret 是 list，用索引获取对应日期的值
                        mask_indices = [i for i, x in enumerate(data[date_column].to_list()) if x == d]
                        rv = [fwd_ret[i] for i in mask_indices if i < len(fwd_ret)]

                        # 过滤 NaN
                        valid = [(f, r) for f, r in zip(fv, rv)
                                 if f is not None and r is not None
                                 and not (isinstance(f, float) and np.isnan(f))
                                 and not (isinstance(r, float) and np.isnan(r))]

                        if len(valid) >= 3:
                            fv_valid, rv_valid = zip(*valid)
                            corr = np.corrcoef(fv_valid, rv_valid)[0, 1]
                            if not np.isnan(corr):
                                daily_ics.append(corr)

                    if daily_ics:
                        ic_mean = float(np.mean(daily_ics))
                        ic_std = float(np.std(daily_ics))
                        ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
                        ic_results[offset] = {
                            "ic_mean": ic_mean,
                            "ic_std": ic_std,
                            "ir": ir,
                        }

                if ic_results:
                    primary = ic_results[fr[0]]
                    out.append(FactorMetrics(
                        formula_id=factor.formula_id,
                        status="success",
                        ic_mean=primary["ic_mean"],
                        ic_std=primary["ic_std"],
                        ir=primary["ir"],
                        ic_decay={str(k): v["ic_mean"] for k, v in ic_results.items()},
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