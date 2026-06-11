"""多进程评估池 — 并行 evaluate 多个 candidate。

设计:
    - evaluate_fn 必须是顶层函数 (模块级), 才能被 pickle 传给子进程
    - 子进程内执行的 evaluate_fn 接受 FactorCandidate, 返回 (passed, metrics, feedback)
    - EvolutionLoop 在 round 0 / round N 中, 把所有 candidate 通过 ThreadPoolExecutor.map 批量评估
    - 使用 ThreadPool 而非 ProcessPool (runner._evaluate_candidate 是 bound method, 不可 pickle)
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Optional


# ── 顶层 pickle-safe evaluate 函数 ──────────────────────────

def _heavy_evaluate(
    candidate_dict: dict,
    sleep_ms: int = 0,
) -> dict:
    """在子进程中执行 evaluate, 接受 dict (pickle 友好), 返回 dict。

    Args:
        candidate_dict: FactorCandidate 的 dict 形式
        sleep_ms: 模拟重计算 (测试用, ms)

    Returns:
        dict 含 passed / metrics / feedback_dict / error
    """
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    import random
    random.seed(hash(candidate_dict.get("expression", "")) & 0xFFFF)
    sharpe = random.uniform(0.0, 2.0)
    return {
        "passed": True,
        "metrics": {"sharpe": sharpe, "arr": sharpe * 0.1, "ic_mean": 0.04},
        "feedback_dict": {
            "factor_id": candidate_dict.get("factor_id", ""),
            "factor_name": candidate_dict.get("name", ""),
            "decision": True,
            "summary": f"sharpe={sharpe:.2f}",
            "metadata": {},
            "channels": {},
        },
        "error": None,
    }


def make_worker_evaluate(
    base_evaluate: Optional[Callable] = None,
    sleep_ms: int = 0,
) -> Callable:
    """构造 workers>1 时的 evaluate 函数。

    base_evaluate=None → 使用 _heavy_evaluate (mock, pickle-safe)。
    base_evaluate 提供 → 使用它 (线程池, 不需 pickle)。
    """
    if base_evaluate is None:
        return lambda c: _heavy_evaluate(c.__dict__ if hasattr(c, "__dict__") else c, sleep_ms)
    return base_evaluate


# ── 并行评估 (ThreadPoolExecutor) ──────────────────────────

def parallel_evaluate(
    candidates: list,
    evaluate_fn: Callable,
    max_workers: int = 4,
    snapshot_path: str | None = None,
) -> list[dict]:
    """并行评估多个 candidate, 返回 list。

    - max_workers=1: 串行
    - snapshot_path 提供: ProcessPoolExecutor (真实并行, 适合大回测)
    - 否则: ThreadPoolExecutor (无需 pickle, 适合 I/O 密集)

    Args:
        candidates: FactorCandidate 列表
        evaluate_fn: 评估函数 (ThreadPool 时用)
        max_workers: 并行数 (1=串行)
        snapshot_path: 预序列化的 config+context 路径 (ProcessPool 模式)

    Returns:
        list[tuple|dict|result_dict] 顺序与 candidates 对应
    """
    if max_workers <= 1:
        return [evaluate_fn(c) for c in candidates]

    if snapshot_path is not None:
        # ProcessPool 模式 (真实并行)
        from .worker_process import subprocess_evaluate
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(subprocess_evaluate, c.__dict__, snapshot_path): i
                for i, c in enumerate(candidates)
            }
            results: list[Optional[dict]] = [None] * len(candidates)
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = {"passed": False, "metrics": {}, "feedback_dict": None, "error": str(e)}
            return results  # type: ignore

    # ThreadPool 模式 (无需 pickle)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_fn, c): i for i, c in enumerate(candidates)}
        results = [None] * len(candidates)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = (False, {}, None)
        return results  # type: ignore


# ── 进程数检测 ──────────────────────────────────────────────

def detect_max_workers(default: int = 4) -> int:
    """检测可用 CPU 核数, 返回推荐 max_workers。"""
    import multiprocessing as mp
    try:
        cpu_count = mp.cpu_count()
        if cpu_count is None or cpu_count < 1:
            return default
        return min(cpu_count, default * 2)
    except Exception:
        return default
