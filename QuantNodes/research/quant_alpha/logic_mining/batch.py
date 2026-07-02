# coding=utf-8
"""
batch.py - 并发批处理逻辑挖掘 (v3.0.2 Step 2)

提供:
- LogicMiningBatchResult: 一次批量挖掘的结果汇总
- mine_logic_library_v2: 并发批量入口 (ThreadPoolExecutor)
- 自动幂等: 读 wiki 现有 Logic pages → 跳过已挖
- 进度回调 + 跨线程 PipelineMetrics (Lock)
- alpha158 template-only 自动 warning 跳过

对比 build_initial_logic_library (v1, 单线程):
- 并发执行
- 幂等跳过
- 自动构建 FactorPool
- 失败/跳过计数
- 跨线程 metrics 锁
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from QuantNodes.research.quant_alpha.factor_pool import FactorEntry, FactorPool
from QuantNodes.research.quant_alpha.logic_mining.metrics import (
    LogicMiningStrictError,
    PipelineMetrics,
    StrictConfig,
)
from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicAbstractionResult,
)
from QuantNodes.research.quant_alpha.logic_mining.pipelines import (
    LogicMiningPipeline,
)
from QuantNodes.research.quant_alpha.logic_mining.sources import (
    get_formulas_from_source,
    list_available_sources,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LogicMiningBatchResult",
    "mine_logic_library_v2",
    "ThreadSafeMetrics",
]


class ThreadSafeMetrics:
    """PipelineMetrics 跨线程包装

    所有 record_* 方法加锁; 否则 ThreadPool 并发下 count 会漂移
    """

    def __init__(self, inner: Optional[PipelineMetrics] = None) -> None:
        self._inner = inner or PipelineMetrics()
        self._lock = threading.Lock()

    def record_call_failure(self, agent_id: str) -> None:
        with self._lock:
            self._inner.record_call_failure(agent_id)

    def record_parse_failure(self, agent_id: str, layer_reached: int) -> None:
        with self._lock:
            self._inner.record_parse_failure(agent_id, layer_reached)

    def record_structured_failure(self, agent_id: str) -> None:
        with self._lock:
            self._inner.record_structured_failure(agent_id)

    def record_wiki_failure(self) -> None:
        with self._lock:
            self._inner.record_wiki_failure()

    def record_inner_loop_failure(self) -> None:
        with self._lock:
            self._inner.record_inner_loop_failure()

    @property
    def inner(self) -> PipelineMetrics:
        return self._inner

    def to_dict(self) -> Dict[str, Any]:
        return self._inner.to_dict()

    def total_failures(self) -> int:
        return self._inner.total_failures()


@dataclass
class LogicMiningBatchResult:
    """批量挖掘结果

    Attributes:
        results:        成功提取的 LogicAbstractionResult 列表
        pool:           FactorPool 实例 (含本次新加入 + from_wiki 预加载)
        metrics:        ThreadSafeMetrics (跨线程统计)
        skipped_ids:    跳过 (已存在) 的 formula_id 集合
        attempted_ids:  尝试挖掘 (未跳过) 的 formula_id 列表
        failed_ids:     失败 (LLM/parse/structured) 的 (formula_id, error) 列表
        wall_clock_s:   总耗时秒数
        warnings:       warning 信息 (如 alpha158 无公式)
    """
    results: List[LogicAbstractionResult] = field(default_factory=list)
    pool: Optional[FactorPool] = None
    metrics: Optional[ThreadSafeMetrics] = None
    skipped_ids: Set[str] = field(default_factory=set)
    attempted_ids: List[str] = field(default_factory=list)
    failed_ids: List[Tuple[str, str]] = field(default_factory=list)
    wall_clock_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def n_mined(self) -> int:
        return len(self.results)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_ids)

    @property
    def n_failed(self) -> int:
        return len(self.failed_ids)

    @property
    def is_success(self) -> bool:
        return self.n_failed == 0 and self.n_mined > 0

    @property
    def is_partial(self) -> bool:
        return self.n_failed > 0 and self.n_mined > 0

    @property
    def is_empty(self) -> bool:
        return self.n_mined == 0 and self.n_skipped == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_attempted": len(self.attempted_ids),
            "n_mined": self.n_mined,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "warnings": list(self.warnings),
            "failed_ids": [{"formula_id": fid, "error": err} for fid, err in self.failed_ids],
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "pool_summary": self.pool.summary() if self.pool else {},
        }


def mine_logic_library_v2(
    source_libs: Sequence[str] = ("alpha101", "alpha158", "alpha191"),
    llm_client: Any = None,
    max_per_lib: int = 10,
    only_volume_price: bool = True,
    workers: int = 4,
    metrics: Optional[ThreadSafeMetrics] = None,
    strict: Optional[StrictConfig] = None,
    wiki_path: str = "wiki_auto",
    skip_existing: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    proxy: Any = None,
) -> LogicMiningBatchResult:
    """并发批量挖掘逻辑库 (v3.0.2 入口)

    Args:
        source_libs:      要挖掘的来源库列表
        llm_client:       LLM 客户端 (None → NullLLMClient)
        max_per_lib:      每个库最多提取多少条
        only_volume_price: 仅量价类 (过滤财务关键词)
        workers:          并发线程数 (>=1)
        metrics:          可观测性指标 (None → 新建 ThreadSafeMetrics)
        strict:           严格模式开关 (默认 all_off)
        wiki_path:        Wiki 根目录
        skip_existing:    是否跳过已存在的 logic pages (默认 True)
        on_progress:      进度回调 (done, total, current_id)
        proxy:            可选注入的 WikiFactorProxy (避免重复创建)

    Returns:
        LogicMiningBatchResult
    """
    workers = max(1, workers)
    metrics = metrics or ThreadSafeMetrics()
    result = LogicMiningBatchResult(metrics=metrics)

    t0 = time.perf_counter()

    # Step 1: 创建池并预加载 wiki 现有 logics (幂等性)
    pool = FactorPool(wiki_path=wiki_path)
    proxy_obj = proxy if proxy is not None else _try_create_proxy(wiki_path)
    if proxy_obj is not None:
        try:
            n_pre = pool.from_wiki(proxy_obj)
            logger.info("Pre-loaded %d existing logics from %s", n_pre, wiki_path)
        except Exception as exc:
            result.warnings.append(f"from_wiki failed: {exc!r}")
            logger.warning("from_wiki failed: %s", exc)
    result.pool = pool

    # Step 2: 收集所有待挖掘 (lib, formula_id, formula)
    todo: List[Tuple[str, str, str]] = []
    for lib in source_libs:
        formulas = get_formulas_from_source(
            lib, max_count=max_per_lib, only_volume_price=only_volume_price
        )
        if not formulas:
            if lib == "alpha158":
                result.warnings.append(
                    f"alpha158 returns 0 formulas (template-only source); skipping"
                )
            else:
                result.warnings.append(
                    f"{lib} returns 0 formulas (may be empty source or all filtered out)"
                )
            continue
        for f in formulas:
            fid = f"{lib}-{f['id']}"
            if skip_existing and pool.contains(fid):
                result.skipped_ids.add(fid)
            else:
                todo.append((lib, fid, f["formula"]))

    total = len(todo)
    logger.info(
        "Mining %d formulas across %s (skip=%d, workers=%d)",
        total, source_libs, len(result.skipped_ids), workers,
    )

    if total == 0:
        result.wall_clock_s = time.perf_counter() - t0
        return result

    # Step 3: ThreadPoolExecutor 并发执行
    done_count = 0
    lock = threading.Lock()

    def _worker(lib: str, fid: str, formula: str) -> Tuple[str, Optional[LogicAbstractionResult], Optional[str]]:
        try:
            pipeline = LogicMiningPipeline(
                llm_client=llm_client, metrics=metrics.inner, strict=strict,
            )
            r = pipeline.run(formula, lib)
            return (fid, r, None)
        except LogicMiningStrictError:
            raise
        except Exception as exc:
            return (fid, None, repr(exc))

    strict_error: Optional[LogicMiningStrictError] = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_worker, lib, fid, formula): (lib, fid)
            for lib, fid, formula in todo
        }
        for fut in as_completed(futures):
            lib, fid = futures[fut]
            try:
                fid_ret, r, err = fut.result()
            except LogicMiningStrictError as exc:
                strict_error = exc
                break

            done_count += 1
            with lock:
                result.attempted_ids.append(fid_ret)
            if on_progress is not None:
                try:
                    on_progress(done_count, total, fid_ret)
                except Exception:
                    pass

            if err is not None:
                result.failed_ids.append((fid_ret, err))
                logger.warning("Mining failed for %s: %s", fid_ret, err)
                continue

            if r is None:
                result.failed_ids.append((fid_ret, "null result"))
                continue

            result.results.append(r)

            structured_dict = r.structured_logic.to_dict() if r.structured_logic else None
            pool.add(FactorEntry(
                formula_id=fid_ret,
                formula=r.source_formula,
                source_lib=lib,
                source_id=fid_ret.split("-", 1)[-1] if "-" in fid_ret else fid_ret,
                ir=0.0,
                ic_mean=0.0,
                rank_ic=0.0,
                tags=[f"source={lib}", f"parse_layer={r.parse_layer}"],
                structured=structured_dict,
                evidence=None,
            ))

    if strict_error is not None:
        raise strict_error

    result.wall_clock_s = time.perf_counter() - t0
    logger.info(
        "Batch done: mined=%d skipped=%d failed=%d in %.2fs",
        result.n_mined, result.n_skipped, result.n_failed, result.wall_clock_s,
    )
    return result


def _try_create_proxy(wiki_path: str) -> Any:
    """安全构造 WikiFactorProxy; 失败返回 None"""
    try:
        from QuantNodes.research.wiki import WikiFactorProxy
        return WikiFactorProxy(wiki_path=wiki_path)
    except Exception as exc:
        logger.debug("WikiFactorProxy unavailable: %s", exc)
        return None