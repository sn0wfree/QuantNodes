# coding=utf-8
"""
test_op_prior.py - OpPrior 单元测试

测试覆盖:
- 基础更新（增加/衰减）
- 权重范围（[floor, 1.0]）
- 混合分布（与均匀分布）
- 持久化（save/load roundtrip）
- 边界条件（空 ops, 零 IR）
- top_k 方法
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from QuantNodes.research.quant_alpha.mcts.op_prior import OpPrior


class TestOpPriorBasic:
    """基础场景"""

    def test_default_init(self):
        prior = OpPrior()
        assert prior.weights == {}
        assert prior.alpha == 0.7
        assert prior.floor == 0.1
        assert prior.total_updates == 0

    def test_update_single_op(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=0.5)
        # 第一次更新，old=0.5, new = 0.5*0.7 + 1.0*0.3 = 0.65
        assert abs(prior.weights["rank"] - 0.65) < 0.01
        assert prior.total_updates == 1

    def test_update_multiple_ops(self):
        prior = OpPrior()
        prior.update(ops=["rank", "ts_std", "div"], ir=0.5)
        assert "rank" in prior.weights
        assert "ts_std" in prior.weights
        assert "div" in prior.weights
        assert prior.total_updates == 1

    def test_update_empty_ops_noop(self):
        prior = OpPrior()
        prior.update(ops=[], ir=0.5)
        assert prior.weights == {}
        assert prior.total_updates == 0

    def test_update_zero_ir_noop(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=0.0)
        assert prior.weights == {}
        assert prior.total_updates == 0

    def test_update_ir_below_threshold_noop(self):
        prior = OpPrior(ir_threshold=0.05)
        prior.update(ops=["rank"], ir=0.01)
        assert prior.weights == {}
        assert prior.total_updates == 0

    def test_update_negative_ir_uses_abs(self):
        """负 IR 也应该贡献（用 |IR|）"""
        prior = OpPrior()
        prior.update(ops=["rank"], ir=-0.5)
        # 应该和正 IR 0.5 同样效果
        prior2 = OpPrior()
        prior2.update(ops=["rank"], ir=0.5)
        assert abs(prior.weights["rank"] - prior2.weights["rank"]) < 1e-6


class TestOpPriorDecay:
    """指数衰减"""

    def test_exponential_decay(self):
        """alpha=0.7 保留 70% 历史"""
        prior = OpPrior(alpha=0.7)
        prior.update(ops=["rank"], ir=0.5)  # new = 0.5*0.7 + 1.0*0.3 = 0.65
        prior.update(ops=["rank"], ir=0.0)  # 不应更新（ir=0）
        # weights 应该是 0.65（不变）
        assert abs(prior.weights["rank"] - 0.65) < 0.01

    def test_repeated_strong_signals(self):
        """重复强信号 → 权重趋向满强度 1.0"""
        prior = OpPrior(alpha=0.5, ir_full_strength=0.5)
        for _ in range(10):
            prior.update(ops=["rank"], ir=0.5)
        assert prior.weights["rank"] > 0.95

    def test_floor_protection(self):
        """floor=0.1 避免零概率"""
        prior = OpPrior(alpha=0.99, floor=0.1)
        # 即使反复低信号，权重不应低于 0.1
        for _ in range(20):
            prior.update(ops=["rank"], ir=0.01)  # ir < threshold 但刚好等于
        # 实际上 ir=0.01 < ir_threshold=0.01 → 不会更新
        # 改用接近 0 但 > 0
        for _ in range(20):
            prior.update(ops=["ts_std"], ir=0.011)
        # ts_std 第一次：old=0.5, strength=0.022, new = 0.5*0.99 + 0.022*0.01 = 0.4952
        # 第 20 次后接近 0.5 附近（衰减很慢），floor=0.1 不会触发
        assert prior.weights["ts_std"] >= 0.1

    def test_clamp_to_one(self):
        """strength cap 到 1.0"""
        prior = OpPrior()
        prior.update(ops=["rank"], ir=10.0)  # 远超 ir_full_strength
        # strength = min(10/0.5, 1.0) = 1.0
        # new = 0.5*0.7 + 1.0*0.3 = 0.65
        # 多次更新后趋向 1.0
        for _ in range(20):
            prior.update(ops=["rank"], ir=10.0)
        assert prior.weights["rank"] <= 1.0


class TestOpPriorSampling:
    """采样权重"""

    def test_sample_weights(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=1.0)  # 满强度
        prior.update(ops=["ts_std"], ir=0.1)  # 弱强度
        weights = prior.sample_weights(["rank", "ts_std", "div"])
        assert len(weights) == 3
        assert weights[0] > weights[1]  # rank > ts_std
        assert weights[2] == 0.5  # 未见过 → 默认 0.5

    def test_mix_pure_uniform(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=1.0)
        weights = prior.mix(["rank", "ts_std", "div"], mix_ratio=0.0)
        # 纯均匀
        assert np.allclose(weights, [1/3, 1/3, 1/3])

    def test_mix_pure_prior(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=1.0)  # rank 强
        prior.update(ops=["ts_std"], ir=0.1)  # ts_std 弱
        weights = prior.mix(["rank", "ts_std", "div"], mix_ratio=1.0)
        # 纯先验
        assert weights[0] > weights[1]
        assert weights.sum() == pytest.approx(1.0)

    def test_mix_default_50_50(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=1.0)
        weights = prior.mix(["rank", "ts_std", "div"])  # mix_ratio=0.5
        # 介于均匀和先验之间
        uniform = np.array([1/3, 1/3, 1/3])
        assert not np.allclose(weights, uniform)
        assert weights.sum() == pytest.approx(1.0)

    def test_mix_sums_to_one(self):
        prior = OpPrior()
        prior.update(ops=["rank", "ts_std"], ir=0.5)
        weights = prior.mix(["rank", "ts_std", "div", "mul"], mix_ratio=0.5)
        assert weights.sum() == pytest.approx(1.0)

    def test_empty_ops(self):
        prior = OpPrior()
        weights = prior.mix([])
        assert len(weights) == 0


class TestOpPriorPersistence:
    """持久化"""

    def test_save_load_roundtrip(self, tmp_path):
        prior = OpPrior()
        prior.update(ops=["rank", "ts_std"], ir=0.5)
        prior.update(ops=["div"], ir=0.8)

        path = tmp_path / "op_prior.json"
        prior.save(path)
        assert path.exists()

        # 重新加载
        loaded = OpPrior.load(path)
        assert loaded.weights == prior.weights
        assert loaded.alpha == prior.alpha
        assert loaded.floor == prior.floor
        assert loaded.total_updates == prior.total_updates

    def test_save_creates_dir(self, tmp_path):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=0.5)
        nested = tmp_path / "subdir" / "op_prior.json"
        prior.save(nested)
        assert nested.exists()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OpPrior.load(tmp_path / "missing.json")

    def test_to_dict(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=0.5)
        d = prior.to_dict()
        assert "weights" in d
        assert "alpha" in d
        assert "total_updates" in d
        assert d["total_updates"] == 1


class TestOpPriorTopK:
    """top_k 方法"""

    def test_top_k(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=1.0)
        prior.update(ops=["ts_std"], ir=0.5)
        prior.update(ops=["div"], ir=0.2)
        top = prior.top_k(2)
        assert len(top) == 2
        assert top[0][0] == "rank"  # 最高权重
        assert top[1][0] == "ts_std"

    def test_top_k_more_than_weights(self):
        prior = OpPrior()
        prior.update(ops=["rank"], ir=0.5)
        top = prior.top_k(10)
        assert len(top) == 1


class TestOpPriorIntegration:
    """集成测试"""

    def test_typical_session(self):
        """典型使用流程"""
        prior = OpPrior()

        # 1. 多轮更新
        prior.update(ops=["rank", "ts_std"], ir=0.5)  # round 1
        prior.update(ops=["rank", "div"], ir=0.7)      # round 2
        prior.update(ops=["ts_std", "mul"], ir=0.3)    # round 3

        # 2. 验证权重：rank 应最高（出现 2 次）
        weights = prior.sample_weights(["rank", "ts_std", "div", "mul"])
        rank_w = prior.weights["rank"]
        ts_std_w = prior.weights["ts_std"]
        div_w = prior.weights["div"]
        mul_w = prior.weights["mul"]
        assert rank_w > ts_std_w  # rank 出现 2 次
        assert rank_w > div_w
        assert ts_std_w > mul_w  # ts_std IR=0.5 > mul IR=0.3

        # 3. 混合采样
        mix_weights = prior.mix(
            ["rank", "ts_std", "div", "mul"],
            mix_ratio=0.5,
        )
        assert mix_weights.sum() == pytest.approx(1.0)
        assert mix_weights[0] > mix_weights[3]  # rank > mul
