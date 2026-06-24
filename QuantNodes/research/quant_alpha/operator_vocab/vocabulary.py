# coding=utf-8
"""
OperatorVocab 主类 - 统一算子查询/调用/元数据化接口

修复 factor_evaluator._compute_factor 的 3 个 latent bug：
1. ts_corr/ts_cov 的 Series.rolling_corr 不存在 → 改用 .rolling_corr() 在 DataFrame
2. rank/zscore 全局而非 per-date → 默认 over(date_column)
3. 异常被静默吞掉 → 完整错误抛出（带上下文）
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import polars as pl

from QuantNodes.research.quant_alpha.operator_vocab.config import (
    OperatorVocabConfig,
)
from QuantNodes.research.quant_alpha.operator_vocab.metadata import (
    OperatorCategory,
    OperatorMetadata,
    _infer_category_tags,
    _infer_default_window,
    _infer_difficulty,
    _infer_output_dtype,
    _infer_requires_group_by,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# 模块级单例（线程安全）
_default_vocab: Optional["OperatorVocab"] = None
_default_vocab_lock = threading.Lock()


def _max_nesting_depth(formula: str) -> int:
    """计算公式的最大嵌套深度

    例如：
        "rank(close)" → 1
        "rank(ts_mean(close, 5))" → 2
        "rank(ts_mean(ts_delta(close, 1), 5))" → 3
    """
    max_depth = 0
    current_depth = 0
    for char in formula:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ")":
            current_depth -= 1
            if current_depth < 0:
                return -1  # 不平衡
    return max_depth


class OperatorVocab:
    """统一算子词表 — 自动挖掘的统一接口

    把分散在 3 处的 285 个算子整合成统一查询/调用/元数据化接口。
    详见 docs/quant_alpha/PROJECT_PLAN.md §3 路线 0。
    """

    def __init__(self, config: Optional[OperatorVocabConfig] = None):
        self.config = config or OperatorVocabConfig()
        self._l0_registry: Optional[Dict[str, Dict[str, Any]]] = None
        self._l1_specs: Optional[Dict[str, Any]] = None
        self._metadata_cache: Dict[str, OperatorMetadata] = {}
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> "OperatorVocab":
        """模块级默认实例（线程安全单例）"""
        global _default_vocab
        if _default_vocab is None:
            with _default_vocab_lock:
                if _default_vocab is None:
                    _default_vocab = cls()
        return _default_vocab

    def reset_default_cache(self) -> None:
        """重置默认实例缓存（仅用于测试）"""
        global _default_vocab
        with _default_vocab_lock:
            _default_vocab = None

    # ==================================================================
    # 注册表加载
    # ==================================================================

    def _ensure_l0_loaded(self) -> Dict[str, Dict[str, Any]]:
        """懒加载 L0 注册表"""
        if self._l0_registry is not None:
            return self._l0_registry

        with self._lock:
            if self._l0_registry is not None:
                return self._l0_registry

            from QuantNodes.factor_node.factor_functions._helpers import (
                _OPERATOR_REGISTRY,
            )

            self._l0_registry = {}
            for cat in self.config.enabled_categories:
                if cat == "talib" and not self.config.talib_enabled:
                    continue
                if cat in _OPERATOR_REGISTRY:
                    self._l0_registry.update(_OPERATOR_REGISTRY[cat])

            return self._l0_registry

    def _ensure_l1_loaded(self) -> Dict[str, Any]:
        """懒加载 L1 composite 注册表"""
        if self._l1_specs is not None:
            return self._l1_specs

        with self._lock:
            if self._l1_specs is not None:
                return self._l1_specs

            try:
                from QuantNodes.operators.composite_dag import (
                    list_composite_ops as _list_composite,
                )

                self._l1_specs = {
                    spec.name: spec
                    for spec in (_list_composite() or [])
                }
            except (ImportError, AttributeError) as e:
                logger.debug("L1 composite registry unavailable: %s", e)
                self._l1_specs = {}

            return self._l1_specs

    # ==================================================================
    # 公开 API：查询
    # ==================================================================

    def list_operators(self, category: Optional[str] = None) -> List[str]:
        """列出所有算子

        Args:
            category: 可选过滤（point/time/section/multi_section/talib）

        Returns:
            算子名列表
        """
        l0 = self._ensure_l0_loaded()
        if category is None:
            return sorted(l0.keys())
        if not OperatorCategory.is_valid(category):
            raise ValueError(f"Invalid category: {category}")
        return sorted(name for name, entry in l0.items() if entry.get("category") == category)

    def get_operator(self, name: str) -> Optional[Callable]:
        """按名称获取算子函数"""
        l0 = self._ensure_l0_loaded()
        entry = l0.get(name)
        if entry is not None:
            return entry.get("func")
        return None

    def get_metadata(self, name: str) -> Optional[OperatorMetadata]:
        """按名称获取算子元数据（带缓存）"""
        if name in self._metadata_cache:
            return self._metadata_cache[name]

        l0 = self._ensure_l0_loaded()
        entry = l0.get(name)
        if entry is None:
            return None

        meta = OperatorMetadata.from_registry_entry(entry)
        self._metadata_cache[name] = meta
        return meta

    def list_metadata(
        self, category: Optional[str] = None
    ) -> List[OperatorMetadata]:
        """列出所有算子的元数据"""
        names = self.list_operators(category)
        return [self.get_metadata(n) for n in names if self.get_metadata(n) is not None]

    # ==================================================================
    # 公开 API：build_namespace
    # ==================================================================

    def build_namespace(
        self,
        data: pl.DataFrame,
        date_column: str = "date",
        code_column: str = "code",
        cross_sectional: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """构造 eval 沙箱 namespace

        修复因子：
        1. ts_corr/ts_cov：使用正确 Series API（pl.DataFrame.rolling_corr 不行，
           改用 (c1 - c1.mean()).rolling_corr((c2 - c2.mean()), window)）
        2. rank/zscore 默认 per-date over(date_column)
        3. 异常不静默吞掉（eval 错误直接抛）

        Args:
            data: 行情数据
            date_column: 日期列名
            code_column: 股票代码列名（暂未使用，预留）
            cross_sectional: 是否 per-date 截面
                None = 用 config 默认（默认 True，修复旧 BUG 2）
                True = per-date over(date_column)
                False = 全局（与旧 12-lambda 行为一致）

        Returns:
            eval 沙箱 namespace dict
        """
        if cross_sectional is None:
            cross_sectional = self.config.cross_sectional

        namespace: Dict[str, Any] = {"pl": pl, "data": data}

        # 注入所有列（Series 形式）
        for col in data.columns:
            namespace[col] = data[col]

        # 注入 L0 算子（基于注册表）
        l0 = self._ensure_l0_loaded()
        for name, entry in l0.items():
            func = entry.get("func")
            if func is not None:
                namespace[name] = func

        # 注入 per-date over() 修复的截面算子
        if cross_sectional:
            date_col = date_column

            def _to_series(x: Any) -> pl.Series:
                """Expr → Series（用 data 物化）；Series → 自身"""
                if isinstance(x, pl.Expr):
                    return data.select(x).to_series()
                return x

            def _rank_per_date(x: Any) -> pl.Series:
                """per-date rank（修复 BUG 2）

                支持 Series 或 Expr 输入：
                - Series: 直接使用
                - Expr: 用 data.select 物化
                """
                s = _to_series(x)
                tmp = pl.DataFrame({
                    "_x": s,
                    "_d": data[date_col],
                })
                return tmp.select(
                    pl.col("_x").rank(method="average").over("_d").alias("_r")
                )["_r"]

            def _zscore_per_date(x: Any) -> pl.Series:
                """per-date zscore（修复 BUG 2）"""
                s = _to_series(x)
                tmp = pl.DataFrame({
                    "_x": s,
                    "_d": data[date_col],
                })
                return tmp.select(
                    (
                        (pl.col("_x") - pl.col("_x").mean().over("_d"))
                        / (pl.col("_x").std().over("_d") + 1e-8)
                    ).alias("_z")
                )["_z"]

            def _winsorize_per_date(
                x: Any, lower: float = 0.01, upper: float = 0.99
            ) -> pl.Series:
                """per-date winsorize（修复 BUG 2）"""
                s = _to_series(x)
                tmp = pl.DataFrame({
                    "_x": s,
                    "_d": data[date_col],
                })
                lo = pl.col("_x").quantile(lower).over("_d")
                hi = pl.col("_x").quantile(upper).over("_d")
                return tmp.select(
                    pl.col("_x").clip(lo, hi).alias("_w")
                )["_w"]

            # 覆盖 section_ops 的 rank/zscore/winsorize（per-date 修复）
            namespace["rank"] = _rank_per_date
            namespace["zscore"] = _zscore_per_date
            namespace["winsorize"] = _winsorize_per_date
            # IndNeutralize per-date 修复：在 L0 注册表基础上
            # 加一层 over([date_col, ind_class])
            l0 = self._ensure_l0_loaded()
            base_ind_neutralize = l0.get("IndNeutralize", {}).get("func")

            def _IndNeutralize_per_date(x, ind_class="citic_1"):
                s = _to_series(x)
                tmp = pl.DataFrame({
                    "_x": s,
                    "_d": data[date_col],
                    "_ind": data[ind_class],
                })
                return tmp.select(
                    (pl.col("_x") - pl.col("_x").mean().over(["_d", "_ind"])).alias("_r")
                )["_r"]

            namespace["IndNeutralize"] = _IndNeutralize_per_date
            # ts_corr / ts_cov 走 L0 注册表（用 rolling_corr on Expr）
        else:
            # 全局（与旧 12-lambda 行为一致）
            # 同样支持 Expr 输入
            def _rank_global(x: Any) -> pl.Series:
                s = x.to_series() if isinstance(x, pl.Expr) else x
                return s.rank(method="average")

            def _zscore_global(x: Any) -> pl.Series:
                s = x.to_series() if isinstance(x, pl.Expr) else x
                return (s - s.mean()) / (s.std() + 1e-8)

            def _winsorize_global(
                x: Any, lower: float = 0.01, upper: float = 0.99
            ) -> pl.Series:
                s = x.to_series() if isinstance(x, pl.Expr) else x
                return s.quantile(lower).clip(s.quantile(lower), s.quantile(upper))

            namespace["rank"] = _rank_global
            namespace["zscore"] = _zscore_global
            namespace["winsorize"] = _winsorize_global

        return namespace

    # ==================================================================
    # 公开 API：端到端 evaluate
    # ==================================================================

    def evaluate(
        self,
        formula: str,
        data: pl.DataFrame,
        date_column: str = "date",
        code_column: str = "code",
        cross_sectional: Optional[bool] = None,
    ) -> Optional[pl.Series]:
        """端到端评估公式

        Args:
            formula: 因子公式字符串
            data: 行情数据
            date_column: 日期列名
            code_column: 股票代码列名
            cross_sectional: 是否 per-date 截面

        Returns:
            因子值（pl.Series）或 None（评估失败）
        """
        # 安全检查：公式长度
        if (
            self.config.max_formula_length is not None
            and len(formula) > self.config.max_formula_length
        ):
            raise ValueError(
                f"Formula length {len(formula)} exceeds limit "
                f"{self.config.max_formula_length}"
            )

        # 安全检查：嵌套深度（用栈计算最大嵌套层数）
        depth = _max_nesting_depth(formula)
        if depth > self.config.max_formula_depth:
            raise ValueError(
                f"Formula nesting depth {depth} exceeds limit "
                f"{self.config.max_formula_depth}"
            )

        try:
            namespace = self.build_namespace(
                data=data,
                date_column=date_column,
                code_column=code_column,
                cross_sectional=cross_sectional,
            )
            result = eval(formula, {"__builtins__": {}}, namespace)

            if isinstance(result, pl.Series):
                return result
            if isinstance(result, pl.Expr):
                return data.select(result).to_series()
            if isinstance(result, (int, float, bool)):
                # 标量结果：扩展为 Series
                return pl.Series("_scalar", [result] * len(data))
            return None

        except Exception as e:
            # 不再静默吞掉（修复 BUG 3），但允许调用方捕获
            logger.debug(
                "Formula evaluation failed: formula=%r, error=%s",
                formula,
                e,
            )
            raise

    # ==================================================================
    # 统计
    # ==================================================================

    def stats(self) -> Dict[str, Any]:
        """词表统计信息"""
        l0 = self._ensure_l0_loaded()
        l1 = self._ensure_l1_loaded()

        by_category: Dict[str, int] = {}
        for entry in l0.values():
            cat = entry.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "l0_total": len(l0),
            "l1_total": len(l1),
            "by_category": by_category,
            "config": {
                "cross_sectional": self.config.cross_sectional,
                "talib_enabled": self.config.talib_enabled,
                "enabled_categories": list(self.config.enabled_categories),
            },
        }


# ==================================================================
# 模块级便捷函数
# ==================================================================

def build_namespace(
    data: pl.DataFrame,
    date_column: str = "date",
    code_column: str = "code",
    cross_sectional: bool = True,
) -> Dict[str, Any]:
    """便捷函数：使用默认 vocab 构建 namespace"""
    return OperatorVocab.default().build_namespace(
        data=data,
        date_column=date_column,
        code_column=code_column,
        cross_sectional=cross_sectional,
    )


def list_vocab_operators(category: Optional[str] = None) -> List[str]:
    """便捷函数：列出默认 vocab 算子"""
    return OperatorVocab.default().list_operators(category=category)


def get_vocab_operator(name: str) -> Optional[Callable]:
    """便捷函数：按名称获取默认 vocab 算子"""
    return OperatorVocab.default().get_operator(name)


def get_vocab_metadata(name: str) -> Optional[OperatorMetadata]:
    """便捷函数：按名称获取默认 vocab 元数据"""
    return OperatorVocab.default().get_metadata(name)
