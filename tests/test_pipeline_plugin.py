"""PR-QN-2: PipelineRunner plugin 机制测试

锁定 PR-QN-2 (2026-06-21) 行为:
- __init__ 接受 specs 自定义 phase 列表
- from_dict 接受 extra_phases (追加到标准 12 阶段之后)
- 默认参数行为与 PR 之前完全一致 (向后兼容)
- run() 遍历 self._specs 而非硬编码 PIPELINE_SPEC
"""
from __future__ import annotations

import inspect

from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
from QuantNodes.research.factor_test.pipeline_spec import PIPELINE_SPEC, PhaseSpec


def _make_min_config():
    """构造最小化 SingleFactorTestConfig (不触发 I/O)."""
    from QuantNodes.research.factor_test.config import (
        SingleFactorTestConfig, FactorSetting, PreprocessSetting,
    )
    return SingleFactorTestConfig(
        factor=FactorSetting(name="x", factor_dir="/tmp/fake"),
        preprocess=PreprocessSetting(),
    )


class TestPipelinePluginDefaults:
    """默认参数行为不变 (向后兼容)."""

    def test_default_specs_is_pipline_spec(self):
        """不传 specs 时, _specs 应为 PIPELINE_SPEC 的拷贝."""
        cfg = _make_min_config()
        runner = PipelineRunner(cfg)
        assert len(runner._specs) == len(PIPELINE_SPEC)
        assert runner._specs is not PIPELINE_SPEC  # 拷贝, 避免污染全局

    def test_default_specs_preserves_phase_order(self):
        """默认 12 阶段顺序保持."""
        cfg = _make_min_config()
        runner = PipelineRunner(cfg)
        # 第一个 phase 是 LoadData
        assert runner._specs[0].name == "LoadData"
        # 最后一个是 FactorTestReport
        assert runner._specs[-1].name == "FactorTestReport"

    def test_from_dict_default_no_extra(self):
        """from_dict 不传 extra_phases 时 _specs 为标准 12 阶段."""
        cfg = _make_min_config()
        data = cfg.model_dump()
        runner = PipelineRunner.from_dict(data)
        assert len(runner._specs) == len(PIPELINE_SPEC)


class TestExtraPhases:
    """extra_phases 注入自定义 stage."""

    def test_extra_phase_appended(self):
        """extra_phases 应追加到标准 12 阶段之后."""
        from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode

        # 最小化自定义 phase: 用一个 no-op 节点
        class _NoOpNode(PydanticConfigNode):
            ConfigSchema = None
            _ALIASES: dict = {}

            def execute(self, context):
                return context

        custom = PhaseSpec(
            name="MockStage", phase_no=99,
            title="Mock", node_cls=_NoOpNode,
            build_cfg=lambda cfg: {},
        )
        data = _make_min_config().model_dump()
        runner = PipelineRunner.from_dict(data, extra_phases=[custom])
        assert len(runner._specs) == len(PIPELINE_SPEC) + 1
        assert runner._specs[-1].name == "MockStage"

    def test_specs_direct_init(self):
        """__init__ 直接传 specs 也可: specs 追加到标准 12 阶段之后."""
        runner = PipelineRunner(_make_min_config(), specs=[])
        # 空 list 等价于不传 (因为 falsy)
        assert len(runner._specs) == len(PIPELINE_SPEC)

    def test_extra_phases_preserves_standard_order(self):
        """extra_phases 不应破坏标准 12 阶段顺序."""
        from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode

        class _NoOpNode(PydanticConfigNode):
            ConfigSchema = None
            _ALIASES: dict = {}

            def execute(self, context):
                return context

        custom = PhaseSpec(
            name="Mock", phase_no=99,
            title="t", node_cls=_NoOpNode,
            build_cfg=lambda cfg: {},
        )
        data = _make_min_config().model_dump()
        runner = PipelineRunner.from_dict(data, extra_phases=[custom])
        # LoadData 仍是 phase 0
        assert runner._specs[0].name == "LoadData"
        # 倒数第二个仍是 FactorTestReport, 最后一个是 Mock
        assert runner._specs[-2].name == "FactorTestReport"
        assert runner._specs[-1].name == "Mock"

    def test_multiple_extra_phases(self):
        """多个 extra_phase 按顺序追加."""
        from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode

        class _NoOpNode(PydanticConfigNode):
            ConfigSchema = None
            _ALIASES: dict = {}

            def execute(self, context):
                return context

        a = PhaseSpec(
            name="A", phase_no=13, title="A", node_cls=_NoOpNode,
            build_cfg=lambda cfg: {},
        )
        b = PhaseSpec(
            name="B", phase_no=14, title="B", node_cls=_NoOpNode,
            build_cfg=lambda cfg: {},
        )
        data = _make_min_config().model_dump()
        runner = PipelineRunner.from_dict(data, extra_phases=[a, b])
        assert runner._specs[-2].name == "A"
        assert runner._specs[-1].name == "B"

    def test_run_uses_self_specs_not_module(self):
        """run() 应使用 self._specs 而非硬编码 PIPELINE_SPEC.

        通过 inspect.signature 验证 run() 内只读 self._specs, 不引用
        模块级 PIPELINE_SPEC (compile-time 静态检查不易, 这里用
        monkey-patching 方式验证).
        """
        # 静态检查: 源码中不应再出现 `for spec in PIPELINE_SPEC`
        from QuantNodes.research.factor_test import pipeline_runner as pr_mod
        source = inspect.getsource(pr_mod)
        # PR-QN-2 之后, 唯一对 PIPELINE_SPEC 的引用应限定在类外
        # (默认参数 list(PIPELINE_SPEC) / 模块级导入), 不在 run() 循环内
        run_block = source.split("def run(self)")[1].split("def ")[0]
        assert "PIPELINE_SPEC" not in run_block, (
            "run() 内仍含硬编码 PIPELINE_SPEC, 应改为 self._specs"
        )


class TestSignature:
    """新签名兼容性."""

    def test_init_accepts_specs(self):
        """__init__ 第二参数为 specs (Optional[List[PhaseSpec]])."""
        sig = inspect.signature(PipelineRunner.__init__)
        assert "specs" in sig.parameters
        # 默认 None
        assert sig.parameters["specs"].default is None

    def test_from_dict_accepts_extra_phases(self):
        """from_dict 第二参数为 extra_phases."""
        sig = inspect.signature(PipelineRunner.from_dict)
        assert "extra_phases" in sig.parameters
        assert sig.parameters["extra_phases"].default is None
