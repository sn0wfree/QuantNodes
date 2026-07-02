# coding=utf-8
"""
test_p0_logic_mining_fixes.py — Phase 3 P0 修复 (v3.0.1)

覆盖:
- P0-1: alpha191 source 不再静默返回空 — 实现 18 条 OHLCV-only 公式
- P0-2: best_ic 与 IR 解耦 — _compute_best_ic 取 FactorMetrics.ic_mean
- P0-3: IR 上升分支窗口收窄 — MarketLogicGenerator._mock_generate_response 实现
"""
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.quant_alpha.logic_mining import (
    LogicMiningPipeline,
    WikiLogicStructured,
    LogicBehavior,
    LogicCondition,
    LogicPerformanceEvidence,
)
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
)
from QuantNodes.research.quant_alpha.logic_mining.sources import (
    ALPHA191_OHLCV_FORMULAS,
    SOURCES,
    get_formulas_from_source,
)


# =====================================================================
# P0-1: alpha191 source
# =====================================================================


class TestAlpha191Source:
    def test_alpha191_no_longer_silent(self):
        """alpha191 不再静默 — 至少 8 条公式"""
        r = get_formulas_from_source("alpha191", max_count=20)
        assert len(r) >= 8, f"alpha191 returned {len(r)}, expected >= 8"
        assert r[0]["lib"] == "alpha191"

    def test_alpha191_formulas_constant_populated(self):
        """ALPHA191_OHLCV_FORMULAS 至少 18 条"""
        assert len(ALPHA191_OHLCV_FORMULAS) >= 18

    def test_alpha191_all_ohlcv_only(self):
        """所有 alpha191 公式必须不含财务关键字"""
        financial_keywords = ["earnings", "revenue", "profit", "roe", "roa"]
        for fid, formula in ALPHA191_OHLCV_FORMULAS.items():
            lower = formula.lower()
            for kw in financial_keywords:
                assert kw not in lower, f"{fid} contains {kw}: {formula}"

    def test_alpha191_in_sources_registry(self):
        """SOURCES registry 含 alpha191, 且 count 正确"""
        assert "alpha191" in SOURCES
        assert SOURCES["alpha191"]["count"] == len(ALPHA191_OHLCV_FORMULAS)

    def test_alpha191_only_volume_price_filter_works(self):
        """only_volume_price=True 时仍能返回公式 (它们本来就是量价类)"""
        r = get_formulas_from_source("alpha191", max_count=10, only_volume_price=True)
        assert len(r) >= 8, "alpha191 should not be filtered out by only_volume_price"


# =====================================================================
# P0-2: best_ic 解耦
# =====================================================================


class TestBestICDecoupled:
    def test_compute_best_ic_from_real_factor_metrics(self):
        """_compute_best_ic 取 FactorMetrics.ic_mean, 与 ir 无关"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _compute_best_ic

        class _F:
            def __init__(self, ir, ic_mean):
                self.ir = ir
                self.ic_mean = ic_mean
                self.formula_id = "x"

        # IR 很高, IC 几乎为零 → best_ic 应 ≈ 0.01 (|ic_mean|), 不应等于 2.0
        result = MagicMock()
        result.final_pool = [_F(ir=2.0, ic_mean=0.01), _F(ir=1.5, ic_mean=-0.005)]
        assert _compute_best_ic(result) == pytest.approx(0.01)

    def test_compute_best_ic_empty_pool(self):
        """final_pool 为空 → 0.0"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _compute_best_ic

        result = MagicMock()
        result.final_pool = []
        assert _compute_best_ic(result) == 0.0

    def test_compute_best_ic_none_result(self):
        """alphagpt_result is None → 0.0"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _compute_best_ic

        assert _compute_best_ic(None) == 0.0

    def test_build_inner_evidence_uses_real_ic_not_ir_proxy(self):
        """_build_inner_evidence.best_ic 应使用 _compute_best_ic, 而非 best_ir"""
        from QuantNodes.research.quant_alpha.workflow.alpha_logics import _build_inner_evidence

        class _F:
            def __init__(self, ir, ic_mean, fid):
                self.ir = ir
                self.ic_mean = ic_mean
                self.formula_id = fid

        result = MagicMock()
        result.summary = {}
        result.final_pool = [
            _F(ir=2.0, ic_mean=0.04, fid="A"),
            _F(ir=1.0, ic_mean=-0.02, fid="B"),
        ]
        # 构建证据
        ev = _build_inner_evidence("test_logic", result, round_idx=1)
        # best_ir 取 max(2.0, 1.0) == 2.0
        assert ev.best_ir == pytest.approx(2.0)
        # best_ic 必须与 best_ir 不再相等; 取 max(0.04, 0.02) == 0.04
        assert ev.best_ic == pytest.approx(0.04)
        assert ev.best_ic != ev.best_ir, "best_ic proxy bug regressed!"


# =====================================================================
# P0-3: IR 上升分支窗口收窄
# =====================================================================


class TestIRImprovingBranchTightens:
    """MarketLogicGenerator._mock_generate_response 在 IR 上升时窗口收窄"""

    def _make_logic(self, param_ranges):
        logic = MagicMock()
        logic.structured = MagicMock()
        logic.structured.predicates = [
            LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
        ]
        logic.structured.behavior = LogicBehavior(target="forward_return_5", direction=-1, horizon=5)
        logic.structured.operator_whitelist = ["ts_mean", "rank"]
        logic.structured.parameter_ranges = param_ranges
        logic.structured.sign_constraint = -1
        return logic

    def test_improving_ir_keeps_sign_tightens_window(self):
        """evidence[-1].best_ir > evidence[-2].best_ir → 窗口 [10, 40] (从 [5, 60] 收 20%)"""
        gen = MarketLogicGenerator(llm_client=None, base_name="improving")
        logic = self._make_logic({"ts_mean": (5, 60)})

        # 构造 evidence 链: prev best_ir=0.3, cur best_ir=0.5 (提升)
        prev = LogicPerformanceEvidence(refinement_round=1, best_ir=0.3, n_factors_explored=5)
        cur = LogicPerformanceEvidence(refinement_round=2, best_ir=0.5, n_factors_explored=4)

        mock_resp = gen._mock_generate_response(
            library=[logic], current_logic=logic,
            history=[logic], evidence=[prev, cur], round_idx=2,
        )
        import json
        data = json.loads(mock_resp)
        # 期望 window 收窄: 5 → 5+(60-5)*0.2 = 16, 60 → 60-(60-5)*0.2 = 49
        rng = data["parameter_ranges"]["ts_mean"]
        assert rng[0] == pytest.approx(16.0)
        assert rng[1] == pytest.approx(49.0)
        # sign 应保持 -1
        assert data["sign_constraint"] == -1

    def test_declining_ir_flips_sign(self):
        """evidence 下降 → sign 反转, 窗口不收窄 (沿用前一轮)"""
        gen = MarketLogicGenerator(llm_client=None, base_name="declining")
        logic = self._make_logic({"ts_mean": (5, 60)})

        prev = LogicPerformanceEvidence(refinement_round=1, best_ir=0.5, n_factors_explored=5)
        cur = LogicPerformanceEvidence(refinement_round=2, best_ir=0.3, n_factors_explored=4)

        mock_resp = gen._mock_generate_response(
            library=[logic], current_logic=logic,
            history=[logic], evidence=[prev, cur], round_idx=2,
        )
        import json
        data = json.loads(mock_resp)
        # sign 应反转
        assert data["sign_constraint"] == 1
        # window 应保持原状 [5, 60]
        rng = data["parameter_ranges"]["ts_mean"]
        assert list(rng) == [5, 60]

    def test_zero_factors_explored_skips_tightening(self):
        """cur.n_factors_explored == 0 → 即使 IR 上升也不收窄 (无样本)"""
        gen = MarketLogicGenerator(llm_client=None, base_name="zero")
        logic = self._make_logic({"ts_mean": (5, 60)})

        prev = LogicPerformanceEvidence(refinement_round=1, best_ir=0.0, n_factors_explored=0)
        cur = LogicPerformanceEvidence(refinement_round=2, best_ir=0.5, n_factors_explored=0)

        mock_resp = gen._mock_generate_response(
            library=[logic], current_logic=logic,
            history=[logic], evidence=[prev, cur], round_idx=2,
        )
        import json
        data = json.loads(mock_resp)
        # IR 升但 n_factors_explored==0 → 不收窄
        rng = data["parameter_ranges"]["ts_mean"]
        assert list(rng) == [5, 60]

    def test_single_evidence_keeps_untouched(self):
        """evidence 长度 < 2 → 保持原状不动"""
        gen = MarketLogicGenerator(llm_client=None, base_name="single")
        logic = self._make_logic({"ts_mean": (10, 50)})

        cur = LogicPerformanceEvidence(refinement_round=1, best_ir=0.5, n_factors_explored=3)
        mock_resp = gen._mock_generate_response(
            library=[logic], current_logic=logic,
            history=[logic], evidence=[cur], round_idx=1,
        )
        import json
        data = json.loads(mock_resp)
        rng = data["parameter_ranges"]["ts_mean"]
        assert list(rng) == [10, 50]
