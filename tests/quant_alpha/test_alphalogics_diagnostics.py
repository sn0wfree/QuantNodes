# coding=utf-8
"""
test_alphalogics_diagnostics.py — AlphaLogicsDiagnostics (v3.0.1)

覆盖 P-14 / P-15 / P-16 / P-17: wiki.store_logic 与 inner loop 失败
不再静默, 而是通过 AlphaLogicsDiagnostics 暴露
"""
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
    AlphaLogicsConfig,
    AlphaLogicsDiagnostics,
    AlphaLogicsWorkflow,
)


def _make_workflow(wiki_fail: bool = False, inner_fail: bool = False) -> AlphaLogicsWorkflow:
    cfg = AlphaLogicsConfig(
        max_outer_rounds=2,
        inner_iterations=1,
        inner_pool_size=2,
        initial_logic_sources=(),  # 不实际 mining, 强制 empty
    )
    wf = AlphaLogicsWorkflow(config=cfg, llm_client=MagicMock())
    if wiki_fail:
        wf.wiki.store_logic = MagicMock(side_effect=IOError("wiki down"))
    if inner_fail:
        # 用 monkey-patch 强制 _run_inner_loop 抛错
        wf._run_inner_loop = MagicMock(side_effect=RuntimeError("inner down"))
    return wf


class TestDiagnosticsBasics:
    def test_default_empty(self):
        d = AlphaLogicsDiagnostics()
        assert d.wiki_failures == 0
        assert d.inner_loop_failures == 0
        assert d.by_round_wiki_failures == []
        assert d.strict_raised == 0
        assert d.to_dict()["wiki_failures"] == 0

    def test_record_wiki_increments(self):
        d = AlphaLogicsDiagnostics()
        d.record_wiki_failure(round_idx=1)
        d.record_wiki_failure(round_idx=2)
        d.record_wiki_failure()  # no round
        assert d.wiki_failures == 3
        assert d.by_round_wiki_failures == [1, 2]

    def test_record_inner_loop_increments(self):
        d = AlphaLogicsDiagnostics()
        d.record_inner_loop_failure(round_idx=1)
        d.record_inner_loop_failure(round_idx=2)
        assert d.inner_loop_failures == 2

    def test_record_strict(self):
        d = AlphaLogicsDiagnostics()
        d.record_strict("first fail")
        d.record_strict("second fail")
        assert d.strict_raised == 2
        assert d.strict_raised_messages == ["first fail", "second fail"]


class TestEmptyLibraryDoesNotInvokeDiagnostics:
    def test_run_with_empty_sources_returns_cleanly(self):
        wf = _make_workflow()
        result = wf.run()
        assert result.summary.get("error") == "initial_library_empty"
        # 此时还没进入外层循环 — diagnostics 应保持空
        assert result.diagnostics.wiki_failures == 0
        assert result.diagnostics.inner_loop_failures == 0
