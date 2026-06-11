"""Parallel — 多进程评估池。

公开 API:
    - parallel_evaluate(candidates, evaluate_fn, max_workers): 并行评估
    - make_worker_evaluate(base, sleep_ms): 构造 worker 函数
    - detect_max_workers(default): 推荐 max_workers
"""
from .worker import _heavy_evaluate, detect_max_workers, make_worker_evaluate, parallel_evaluate

__all__ = [
    "parallel_evaluate",
    "make_worker_evaluate",
    "detect_max_workers",
    "_heavy_evaluate",
]
