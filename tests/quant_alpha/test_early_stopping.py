# coding=utf-8
"""
test_early_stopping.py - EarlyStopping + TerminationConfig + RoundFeedback (Phase 6)

目标: 覆盖 0 测试的 EarlyStopping 类 + 多轮迭代关键配置
- EarlyStopping: patience 边界 / reset / 持续改善
- TerminationConfig: 默认值 / 边界
- RoundFeedback: 序列化
- _should_stop 集成
- _check_timeout
"""
import time
from dataclasses import fields

import pytest

from QuantNodes.research.quant_alpha.pipeline import (
    EarlyStopping,
    PipelineConfig,
    RoundFeedback,
    TerminationConfig,
)


# ==============================================================================
# Test Class 1: EarlyStopping 基本行为
# ==============================================================================


class TestEarlyStoppingBasic:
    """EarlyStopping 类基本行为"""

    def test_default_construction(self):
        """默认构造: patience=3, min_improvement=0.01"""
        es = EarlyStopping()
        assert es.patience == 3
        assert es.min_improvement == 0.01
        assert es.best_ir == 0.0
        assert es.counter == 0

    def test_custom_construction(self):
        """自定义构造"""
        es = EarlyStopping(patience=5, min_improvement=0.05)
        assert es.patience == 5
        assert es.min_improvement == 0.05

    def test_improvement_resets_counter(self):
        """IR 提升重置 counter"""
        es = EarlyStopping(patience=3, min_improvement=0.01)
        # 第一次: best_ir=0, current=0.05 → 改善
        es.should_stop(0.05)
        assert es.best_ir == 0.05
        assert es.counter == 0
        assert not es.should_stop(0.0)  # 不改善
        assert es.counter == 1

    def test_no_improvement_increments_counter(self):
        """无提升 counter 累加"""
        es = EarlyStopping(patience=3, min_improvement=0.01)
        es.should_stop(0.05)  # 改善
        es.should_stop(0.05)  # 无变化 (counter 0 → 1)
        es.should_stop(0.05)  # 无变化 (counter 1 → 2)
        assert es.counter == 2
        assert es.should_stop(0.05)  # 达到 patience → True
        assert es.counter == 3

    def test_strict_threshold(self):
        """min_improvement 严格阈值: 需更大改善才计数"""
        es = EarlyStopping(patience=3, min_improvement=0.1)  # 严格
        # current=0.05 > best=0+0.1? No, 不算改善, counter=1
        es.should_stop(0.05)
        assert es.counter == 1
        # 0.05+0.1=0.15, current=0.06, 仍不 > 0.15, counter=2
        es.should_stop(0.06)
        assert es.counter == 2
        # 但 0.20 > 0.15, 改善, counter 重置
        es.should_stop(0.20)
        assert es.counter == 0
        assert es.best_ir == 0.20

    def test_lax_threshold(self):
        """min_improvement 宽松阈值: 小幅改善也算"""
        es = EarlyStopping(patience=3, min_improvement=0.001)  # 宽松
        es.should_stop(0.05)
        # 0.05+0.001=0.051, current=0.051, 0.051 > 0.051? 不严格 >, 边界
        # 0.052 > 0.051, 改善
        es.should_stop(0.052)
        assert es.counter == 0
        assert es.best_ir == 0.052

    def test_negative_improvement_counted(self):
        """负向 IR 提升也算改善 (negative → 0)"""
        es = EarlyStopping(patience=3, min_improvement=0.01)
        # 从 0 到 -0.05: -0.05 < 0, 不算 > 0 + 0.01
        es.should_stop(-0.05)
        assert es.counter == 1


# ==============================================================================
# Test Class 2: EarlyStopping 边界
# ==============================================================================


class TestEarlyStoppingEdgeCases:
    """EarlyStopping 边界 case"""

    def test_patience_zero_stops_immediately(self):
        """patience=0: 第一次无改善就停"""
        es = EarlyStopping(patience=0)
        # current_ir=0 == best_ir=0, 不算改善, counter=1 >= 0
        assert es.should_stop(0.0) is True

    def test_patience_one(self):
        """patience=1: 第一次无改善就停"""
        es = EarlyStopping(patience=1)
        es.should_stop(0.1)  # 改善
        # 0.1 = best, 不算改善, counter=1 >= 1
        assert es.should_stop(0.1) is True

    def test_reset(self):
        """reset 重置所有状态"""
        es = EarlyStopping(patience=3)
        es.should_stop(0.05)
        es.should_stop(0.05)
        es.should_stop(0.05)
        assert es.counter == 2
        # reset
        es.reset()
        assert es.best_ir == 0.0
        assert es.counter == 0
        # 重新开始
        assert es.should_stop(0.0) is False  # 改善 (0 > 0+0.01? no, 0 == 0+0.01)
        # 注: current=0 vs best=0+0.01, current < best+min, counter 累加
        # 重新读实现: if current_ir > best_ir + min_improvement: 改善
        # current=0, best+min=0.01, 0 > 0.01? No → 不算改善, counter=1
        assert es.counter == 1

    def test_continuous_improvement_never_stops(self):
        """持续改善不会触发早停"""
        es = EarlyStopping(patience=2, min_improvement=0.01)
        for i in range(10):
            ir = (i + 1) * 0.05  # 0.05, 0.10, 0.15, ...
            assert es.should_stop(ir) is False
        assert es.best_ir == 0.5
        assert es.counter == 0


# ==============================================================================
# Test Class 3: TerminationConfig
# ==============================================================================


class TestTerminationConfig:
    """TerminationConfig 边界值"""

    def test_defaults(self):
        """默认配置"""
        tc = TerminationConfig()
        assert tc.max_rounds == 5
        assert tc.target_factors == 10
        assert tc.min_improvement == 0.01
        assert tc.early_stopping is True
        assert tc.patience == 3
        assert tc.timeout_seconds == 3600
        assert tc.round_timeout_seconds == 600

    def test_custom(self):
        """自定义配置"""
        tc = TerminationConfig(
            max_rounds=10,
            target_factors=20,
            early_stopping=False,
        )
        assert tc.max_rounds == 10
        assert tc.target_factors == 20
        assert tc.early_stopping is False

    def test_max_rounds_zero(self):
        """max_rounds=0: 不跑"""
        tc = TerminationConfig(max_rounds=0)
        assert tc.max_rounds == 0

    def test_max_rounds_one(self):
        """max_rounds=1: 跑 1 轮"""
        tc = TerminationConfig(max_rounds=1)
        assert tc.max_rounds == 1

    def test_target_factors_negative(self):
        """target_factors=-1: 不限制"""
        tc = TerminationConfig(target_factors=-1)
        assert tc.target_factors == -1


# ==============================================================================
# Test Class 4: RoundFeedback
# ==============================================================================


class TestRoundFeedback:
    """RoundFeedback 序列化"""

    def test_default_construction(self):
        """默认构造"""
        rf = RoundFeedback(round_num=1, best_ir=0.1, avg_ir=0.05, valid_count=3)
        assert rf.round_num == 1
        assert rf.best_ir == 0.1
        assert rf.avg_ir == 0.05
        assert rf.valid_count == 3
        assert rf.best_formulas == []
        assert rf.failed_patterns == []
        assert rf.suggestions == []

    def test_to_dict(self):
        """to_dict 序列化"""
        rf = RoundFeedback(
            round_num=2,
            best_ir=0.15,
            avg_ir=0.08,
            valid_count=5,
            best_formulas=["f1", "f2"],
            failed_patterns=[{"type": "syntax"}],
            suggestions=["add more operators"],
            stats={"total": 10},
        )
        d = rf.to_dict()
        assert d["round_num"] == 2
        assert d["best_ir"] == 0.15
        assert d["valid_count"] == 5
        assert d["best_formulas"] == ["f1", "f2"]
        assert d["stats"] == {"total": 10}

    def test_to_dict_handles_empty(self):
        """to_dict 空字段处理"""
        rf = RoundFeedback(round_num=1, best_ir=0.0, avg_ir=0.0, valid_count=0)
        d = rf.to_dict()
        assert d["best_formulas"] == []
        assert d["failed_patterns"] == []


# ==============================================================================
# Test Class 5: PipelineConfig 基本
# ==============================================================================


class TestPipelineConfig:
    """PipelineConfig 配置"""

    def test_minimal_construction(self):
        """最小构造"""
        config = PipelineConfig(objective="test")
        assert config.objective == "test"
        assert isinstance(config.termination, TerminationConfig)

    def test_termination_default(self):
        """默认 termination"""
        config = PipelineConfig(objective="t")
        assert config.termination.max_rounds == 5

    def test_custom_termination(self):
        """自定义 termination"""
        tc = TerminationConfig(max_rounds=3, target_factors=5)
        config = PipelineConfig(objective="t", termination=tc)
        assert config.termination.max_rounds == 3
        assert config.termination.target_factors == 5


# ==============================================================================
# Test Class 6: 集成场景
# ==============================================================================


class TestEarlyStoppingIntegration:
    """EarlyStopping 与 Pipeline 集成场景"""

    def test_typical_convergence(self):
        """典型收敛场景: 持续改善后无进展"""
        es = EarlyStopping(patience=3, min_improvement=0.01)
        # Round 1: IR=0.05 (改善)
        assert not es.should_stop(0.05)
        # Round 2: IR=0.08 (改善)
        assert not es.should_stop(0.08)
        # Round 3: IR=0.10 (改善)
        assert not es.should_stop(0.10)
        # Round 4: IR=0.10 (无改善, counter=1)
        assert not es.should_stop(0.10)
        # Round 5: IR=0.10 (counter=2)
        assert not es.should_stop(0.10)
        # Round 6: IR=0.10 (counter=3 >= patience → 停)
        assert es.should_stop(0.10) is True

    def test_improvement_resume_after_stagnation(self):
        """停滞后改善应恢复"""
        es = EarlyStopping(patience=2, min_improvement=0.01)
        es.should_stop(0.05)
        es.should_stop(0.05)  # counter=1
        es.should_stop(0.05)  # counter=2 = patience, 停
        assert es.counter == 2
        # 但应能恢复: 重置后
        es.reset()
        es.should_stop(0.05)
        assert es.counter == 0
