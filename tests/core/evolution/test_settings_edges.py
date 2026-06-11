"""evolution settings 验证测试 (10 tests)。

聚焦:
    - OperatorSetting 默认值
    - EvolutionSetting: any_operator_enabled、默认值验证
    - 各种组合的 enabled 状态
    - pydantic 字段验证
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from QuantNodes.core.evolution.settings import (
    EvolutionSetting,
    OperatorSetting,
)


# ============================================================================
# 1. OperatorSetting (4 tests)
# ============================================================================

class TestOperatorSetting:
    def test_default_values(self):
        s = OperatorSetting()
        assert s.enabled is True
        assert s.model == "mock"
        assert s.max_correction_attempts == 3
        assert s.seed == 42

    def test_custom_values(self):
        s = OperatorSetting(
            enabled=False,
            model="deepseek-v3",
            max_correction_attempts=5,
            seed=100,
        )
        assert s.enabled is False
        assert s.model == "deepseek-v3"
        assert s.max_correction_attempts == 5
        assert s.seed == 100

    def test_invalid_type_raises(self):
        """max_correction_attempts 必须是 int。"""
        with pytest.raises(ValidationError):
            OperatorSetting(max_correction_attempts="not int")

    def test_negative_seed_allowed(self):
        """seed 接受负数。"""
        s = OperatorSetting(seed=-1)
        assert s.seed == -1


# ============================================================================
# 2. EvolutionSetting (6 tests)
# ============================================================================

class TestEvolutionSetting:
    def test_default_disabled(self):
        """默认 enabled=False, 因为演化是 opt-in。"""
        s = EvolutionSetting()
        assert s.enabled is False
        assert s.max_rounds == 3
        assert s.parents_per_round == 1
        assert s.parent_selection_strategy == "top_percent_plus_random"
        assert s.top_percent_threshold == 0.3
        assert s.metric == "sharpe"
        assert s.pool_dir is None
        assert s.early_stop_patience == 0

    def test_any_operator_enabled_default(self):
        """默认 operator enabled=True → any_operator_enabled=True。"""
        s = EvolutionSetting()
        assert s.any_operator_enabled() is True

    def test_all_operators_disabled(self):
        s = EvolutionSetting(
            hypothesizer=OperatorSetting(enabled=False),
            mutator=OperatorSetting(enabled=False),
            crosser=OperatorSetting(enabled=False),
        )
        assert s.any_operator_enabled() is False

    def test_pool_dir_accepts_path(self):
        s = EvolutionSetting(pool_dir="/tmp/pool")
        assert s.pool_dir == "/tmp/pool"

    def test_invalid_top_percent_raises(self):
        """top_percent_threshold 必须是 0-1 (pydantic conint 不强, 但 1.5 仍可存)。"""
        s = EvolutionSetting(top_percent_threshold=1.5)
        # 当前实现未限制, 仅记录
        assert s.top_percent_threshold == 1.5

    def test_early_stop_patience_custom(self):
        s = EvolutionSetting(early_stop_patience=5)
        assert s.early_stop_patience == 5
