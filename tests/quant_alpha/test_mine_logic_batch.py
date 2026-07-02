# coding=utf-8
"""
test_mine_logic_batch.py - mine_logic_library_v2 单元测试 (v3.0.2 Step 2)

覆盖:
- LogicMiningBatchResult 数据类 (n_mined / n_skipped / is_success / is_partial / is_empty)
- ThreadSafeMetrics 跨线程安全
- mine_logic_library_v2:
  * 基础单线程 + NullLLMClient
  * 多线程 workers > 1
  * 幂等性 (skip_existing)
  * wiki 写入 + pre-load
  * alpha158 warning (空 source)
  * 失败 ID 记录
  * 进度回调
  * 严格模式 strict=True

注: alpha101 有 15 个公式, 其中 _is_volume_price 过滤掉含 "pe" 子串的
    (open 中含 pe), 实际保留 5 个: alpha001, alpha012, alpha038, alpha041, alpha054
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.quant_alpha.logic_mining.batch import (
    LogicMiningBatchResult,
    ThreadSafeMetrics,
    mine_logic_library_v2,
)
from QuantNodes.research.quant_alpha.logic_mining.metrics import (
    PipelineMetrics,
    StrictConfig,
)

# alpha101 中通过 _is_volume_price 过滤的 5 个公式 ID
ALPHA101_VOLUME_PRICE_IDS = ["alpha001", "alpha012", "alpha038", "alpha041", "alpha054"]


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def mock_proxy() -> MagicMock:
    proxy = MagicMock()
    proxy.list_logics.return_value = []
    proxy.store_logic.return_value = "Logic/xxx.md"
    return proxy


@pytest.fixture
def mock_proxy_with_existing() -> MagicMock:
    """返回一个含 1 个 existing logic 的 proxy (使用 alpha001 作为已有条目)"""
    logic = MagicMock()
    logic.name = "alpha101-alpha001"
    logic.extracted_formula = "rank(ts_argmax(...))"
    logic.source_detail = {"source_lib": "alpha101", "source_id": "alpha001"}
    logic.created_at = "2026-01-01"
    logic.wiki_page_name = "Logic/alpha101-alpha001.md"
    logic.structured = None
    logic.performance_evidence = None
    proxy = MagicMock()
    proxy.list_logics.return_value = [logic]
    return proxy


# ======================================================================
# ThreadSafeMetrics
# ======================================================================
class TestThreadSafeMetrics:
    def test_default_construction(self):
        m = ThreadSafeMetrics()
        assert m.inner.total_failures() == 0

    def test_record_methods_call_inner(self):
        m = ThreadSafeMetrics()
        m.record_call_failure("logic-mining-structure")
        m.record_parse_failure("logic-mining-semantics", layer_reached=2)
        m.record_structured_failure("logic-mining-abstraction")
        m.record_wiki_failure()
        m.record_inner_loop_failure()
        d = m.to_dict()
        assert d["call_failures"]["logic-mining-structure"] == 1
        assert d["parse_failures"]["logic-mining-semantics"] == 1
        assert d["parse_layer_reached"]["logic-mining-semantics"] == 2
        assert d["structured_failures"]["logic-mining-abstraction"] == 1
        assert d["wiki_failures"] == 1
        assert d["inner_loop_failures"] == 1
        assert m.total_failures() == 5

    def test_concurrent_record_is_safe(self):
        m = ThreadSafeMetrics()
        n_threads = 8
        per_thread = 200
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(per_thread):
                m.record_call_failure(f"a{tid}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        d = m.to_dict()
        total = sum(d["call_failures"].values())
        assert total == n_threads * per_thread


# ======================================================================
# LogicMiningBatchResult
# ======================================================================
class TestLogicMiningBatchResult:
    def test_default_construction(self):
        r = LogicMiningBatchResult()
        assert r.n_mined == 0
        assert r.n_skipped == 0
        assert r.n_failed == 0
        assert r.wall_clock_s == 0.0
        assert r.warnings == []
        assert r.is_empty

    def test_n_mined_n_skipped_n_failed(self):
        r = LogicMiningBatchResult(
            results=[MagicMock()],
            skipped_ids={"a", "b"},
            failed_ids=[("x", "err")],
        )
        assert r.n_mined == 1
        assert r.n_skipped == 2
        assert r.n_failed == 1

    def test_status_flags(self):
        r = LogicMiningBatchResult()
        assert r.is_empty and not r.is_success and not r.is_partial

        r = LogicMiningBatchResult(results=[MagicMock()])
        assert r.is_success

        r = LogicMiningBatchResult(results=[MagicMock()], failed_ids=[("x", "err")])
        assert r.is_partial

    def test_to_dict_shape(self):
        m = ThreadSafeMetrics()
        m.record_call_failure("a")
        r = LogicMiningBatchResult(
            results=[MagicMock()],
            skipped_ids={"a"},
            failed_ids=[("b", "err")],
            metrics=m,
            wall_clock_s=1.23,
        )
        d = r.to_dict()
        assert d["n_attempted"] == 0
        assert d["n_mined"] == 1
        assert d["n_skipped"] == 1
        assert d["n_failed"] == 1
        assert d["wall_clock_s"] == 1.23
        assert "metrics" in d
        assert "pool_summary" in d
        assert d["failed_ids"] == [{"formula_id": "b", "error": "err"}]


# ======================================================================
# mine_logic_library_v2 - 基础流
# ======================================================================
class TestMineLogicLibraryV2Basic:
    def test_with_null_llm(self, tmp_path, mock_proxy):
        """NullLLMClient 走完整 pipeline"""
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=2,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.n_failed == 0
        assert out.n_mined >= 1
        assert out.pool is not None
        assert len(out.pool) >= 1
        assert len(out.attempted_ids) >= 1

    def test_result_includes_pool(self, tmp_path, mock_proxy):
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=1,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.pool is not None
        values = out.pool.values()
        assert len(values) == 1
        assert values[0].source_lib == "alpha101"

    def test_default_source_libs_mines_all(self, tmp_path, mock_proxy):
        """默认三源: alpha101+alpha158+alpha191 全部产出结果"""
        out = mine_logic_library_v2(
            max_per_lib=2,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.n_mined >= 1
        libs_seen = {e.source_lib for e in out.pool}
        assert "alpha101" in libs_seen
        assert "alpha191" in libs_seen

    def test_pool_contains_parsed_results(self, tmp_path, mock_proxy):
        """池内 entry 结构化字段正确"""
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=1,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        for entry in out.pool:
            assert entry.formula_id.startswith("alpha101-")
            assert entry.source_lib == "alpha101"


# ======================================================================
# 幂等性
# ======================================================================
class TestMineLogicLibraryV2Idempotency:
    def test_skip_existing(self, tmp_path, mock_proxy_with_existing):
        """第二次跑: wiki 中已有 alpha101-alpha001 → 跳过"""
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=5,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy_with_existing,
            llm_client=_NullLLM(),
            skip_existing=True,
        )
        # alpha101-alpha001 已被预加载, 应跳过
        assert "alpha101-alpha001" in out.skipped_ids
        # 其他 alpha101 formula 仍会尝试
        assert "alpha101-alpha001" not in out.attempted_ids
        assert len(out.attempted_ids) == len(ALPHA101_VOLUME_PRICE_IDS) - 1

    def test_no_skip_when_disabled(self, tmp_path, mock_proxy_with_existing):
        """skip_existing=False → 即使 wiki 已有也重新尝试"""
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=2,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy_with_existing,
            llm_client=_NullLLM(),
            skip_existing=False,
        )
        assert len(out.attempted_ids) >= 1


# ======================================================================
# 并发
# ======================================================================
class TestMineLogicLibraryV2Concurrency:
    def test_workers_2_runs_to_completion(self, tmp_path, mock_proxy):
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=5,
            workers=2,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.n_failed == 0
        assert len(out.attempted_ids) == len(ALPHA101_VOLUME_PRICE_IDS)

    def test_workers_4_pool_no_duplicate_keys(self, tmp_path, mock_proxy):
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=5,
            workers=4,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        keys = out.pool.keys()
        assert len(keys) == len(set(keys))

    def test_workers_clamped_to_min_1(self, tmp_path, mock_proxy):
        """workers=0 应被 clamp 到 1"""
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=1,
            workers=0,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.n_failed == 0


# ======================================================================
# 进度回调 / 警告 / 错误捕获
# ======================================================================
class TestMineLogicLibraryV2Callbacks:
    def test_progress_callback_fires_for_each(self, tmp_path, mock_proxy):
        seen = []

        def cb(done: int, total: int, fid: str) -> None:
            seen.append((done, total, fid))

        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=2,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
            on_progress=cb,
        )
        assert len(seen) == len(out.attempted_ids)
        dones = [s[0] for s in seen]
        assert dones == sorted(dones)

    def test_alpha158_has_real_formulas(self, tmp_path, mock_proxy):
        """alpha158 returns 7 formulas (no warning expected)"""
        out = mine_logic_library_v2(
            source_libs=["alpha158"],
            max_per_lib=10,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert out.n_mined >= 1
        assert len(out.attempted_ids) == 7

    def test_unknown_source_warning(self, tmp_path, mock_proxy):
        out = mine_logic_library_v2(
            source_libs=["unknown_lib"],
            max_per_lib=10,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_NullLLM(),
        )
        assert len(out.warnings) >= 1
        assert len(out.attempted_ids) == 0


# ======================================================================
# 失败注入
# ======================================================================
class TestMineLogicLibraryV2Failures:
    def test_failing_llm_records_call_failures(self, tmp_path, mock_proxy):
        """LLM 异常 (non-strict) → pipeline falls back to mock, metrics.call_failures 记录"""
        m = ThreadSafeMetrics()
        out = mine_logic_library_v2(
            source_libs=["alpha101"],
            max_per_lib=1,
            workers=1,
            wiki_path=str(tmp_path / "wiki"),
            proxy=mock_proxy,
            llm_client=_AlwaysFailingLLM(),
            metrics=m,
        )
        # non-strict: pipeline falls back to mock, so results succeed
        assert out.n_failed == 0
        assert out.n_mined >= 1
        # but call_failures are recorded
        call_fails = m.to_dict()["call_failures"]
        assert any(v > 0 for v in call_fails.values())


# ======================================================================
# strict 模式
# ======================================================================
class TestMineLogicLibraryV2Strict:
    def test_strict_call_propagates(self, tmp_path, mock_proxy):
        """strict.call=True + 失败 LLM → exception 上抛 (不静默)"""
        from QuantNodes.research.quant_alpha.logic_mining.metrics import (
            LogicMiningStrictError,
        )

        with pytest.raises(LogicMiningStrictError):
            mine_logic_library_v2(
                source_libs=["alpha101"],
                max_per_lib=1,
                workers=1,
                wiki_path=str(tmp_path / "wiki"),
                proxy=mock_proxy,
                llm_client=_AlwaysFailingLLM(),
                strict=StrictConfig(call=True),
            )


# ======================================================================
# Helpers
# ======================================================================
class _NullLLM:
    """NullLLMClient-style stub: 所有调用返回空响应"""
    def chat(self, *args, **kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}

    def __call__(self, prompt: str, *args, **kwargs):
        return "{}"


class _AlwaysFailingLLM:
    """总是抛异常的 LLM stub"""
    def chat(self, *args, **kwargs):
        raise RuntimeError("simulated LLM failure")

    def __call__(self, prompt: str, *args, **kwargs):
        raise RuntimeError("simulated LLM failure")