# coding: utf-8
"""Pipeline Runner / 管线编排器

混合模式数据传递:
- Phase 1 (严格串联): LoadData >> SampleFilter >> TradabilityFilter >> AdjustDate >> Preprocess >> Neutralize
- Phase 2 (Context 共享): ICAnalyzer / GroupAnalyzer / FactorScore / RiskCorrelation
- Phase 3 (依赖分析): LongShort
- Phase 4 (输出): FactorTestReport

FactorFeedback 集成 (Week 1.5, 可选):
- feedback.enabled=False: 现有行为完全不变 (向后兼容)
- feedback.enabled=True:  5 个分析节点返回值自动包装为 FactorFeedback,
  聚合到 ctx['Feedback'], 可选持久化到 feedback.output_dir
"""

import sys
import uuid
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from QuantNodes.core.feedback import (
    FactorFeedback,
    FeedbackChannel,
    FeedbackCollector,
    LLMJudge,
    ensure_feedback,
)
from QuantNodes.research.factor_test.config import SingleFactorTestConfig
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
from QuantNodes.research.factor_test.nodes.risk_correlation_node import RiskCorrelationNode
from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode

_ANALYSIS_NODES = ("ICAnalyzer", "GroupAnalyzer", "LongShort", "FactorScore", "RiskCorrelation")


class PipelineRunner:
    """单因子回测管线编排器

    用法:
        config = SingleFactorTestConfig(...)
        runner = PipelineRunner(config)
        result = runner.run()

        # 或从 YAML:
        runner = PipelineRunner.from_yaml("config.yaml")
        result = runner.run()

    FactorFeedback 集成:
        当 config.feedback.enabled=True 时, 5 个分析节点返回值会自动包装为
        FactorFeedback, 聚合到 ctx['Feedback'] = {node_name: FactorFeedback}。
        若 config.feedback.output_dir 不为 None, 还会持久化到该目录。
    """

    def __init__(self, config: SingleFactorTestConfig):
        self.config = config
        self._context = {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineRunner':
        """从 YAML 配置文件创建 Runner"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        config = SingleFactorTestConfig(**raw)
        return cls(config)

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineRunner':
        """从 dict 创建 Runner"""
        config = SingleFactorTestConfig(**data)
        return cls(config)

    def run(self) -> dict:
        """执行完整单因子回测管线

        Returns:
            dict: 完整结果, 包含各节点输出
                  当 config.feedback.enabled=True 时, 还包含 'Feedback' 键
        """
        cfg = self.config
        ctx = {}
        feedback_enabled = cfg.feedback.enabled
        factor_id = str(uuid.uuid4())
        factor_name = cfg.factor.name
        judge = self._maybe_build_judge(cfg) if feedback_enabled else None

        print("=" * 60)
        print(f"单因子回测: {cfg.factor.name}")
        print(f"时间范围: {cfg.preprocess.adj_date_beg} ~ {cfg.preprocess.adj_date_end}")
        if feedback_enabled:
            print(f"FactorFeedback: ENABLED (factor_id={factor_id[:8]}...)")
        print("=" * 60)

        # ============================================================
        # Phase 1: 严格串联 — 数据层
        # ============================================================
        print("\n[Phase 1] 数据加载...")
        load_data_cfg = {
            'factor': cfg.factor.model_dump(),
            'data_path': cfg.data_path,
            'load_keys': cfg.load_keys,
        }
        load_data = LoadDataNode(config=load_data_cfg)
        ctx['LoadData'] = load_data.execute()
        print(f"  因子形状: {ctx['LoadData'].get('factor', pd.DataFrame()).shape}")

        print("\n[Phase 2] 样本池筛选...")
        sample_filter = SamplePoolFilterNode(config={
            'sample_index': cfg.preprocess.sample_index,
            'sample_industry': cfg.preprocess.sample_industry,
            'sample_index_customdir': cfg.preprocess.sample_index_customdir,
        })
        ctx['SamplePoolFilter'] = sample_filter.execute(context=ctx)

        print("\n[Phase 3] 可交易性筛选...")
        tradability = TradabilityFilterNode(config={
            'tradable': cfg.preprocess.tradable.model_dump(),
        })
        ctx['TradabilityFilter'] = tradability.execute(context=ctx)

        print("\n[Phase 4] 调仓日生成...")
        adjust_date = AdjustDateNode(config={
            'adj_date_beg': cfg.preprocess.adj_date_beg,
            'adj_date_end': cfg.preprocess.adj_date_end,
            'adj_mode': cfg.preprocess.adj_mode,
        })
        ctx['AdjustDate'] = adjust_date.execute(context=ctx)
        print(f"  调仓日数: {len(ctx['AdjustDate'])}")

        print("\n[Phase 5] 因子预处理...")
        preprocess = FactorPreprocessNode(config={
            'missing': cfg.preprocess.missing,
            'extreme': cfg.preprocess.extreme,
            'norm': cfg.preprocess.norm,
        })
        ctx['FactorPreprocess'] = preprocess.execute(context=ctx)
        print(f"  预处理后因子形状: {ctx['FactorPreprocess'].shape}")

        print("\n[Phase 6] 因子中性化...")
        neutralize = FactorNeutralizeNode(config={
            'industry_neutral': cfg.preprocess.industry_neutral,
            'risk_neutral': cfg.preprocess.risk_neutral,
            'risk_factors': cfg.preprocess.risk_factors,
        })
        ctx['FactorNeutralize'] = neutralize.execute(context=ctx)

        # ============================================================
        # Phase 2: Context 共享 — 分析层
        # ============================================================
        print("\n[Phase 7] IC 分析...")
        ic_analyzer = ICAnalyzerNode(config={
            'min_group_size': cfg.analysis.ic.min_group_size,
        })
        ctx['ICAnalyzer'] = ic_analyzer.execute(context=ctx)
        ic_result = ctx['ICAnalyzer'].get('ic_result')
        if ic_result is not None:
            print(f"  IC均值: {ic_result.get('IC均值', 'N/A'):.4f}")
            print(f"  ICIR: {ic_result.get('ICIR', 'N/A'):.4f}")

        print("\n[Phase 8] 分组分析...")
        group_analyzer = GroupAnalyzerNode(config={
            'groups': cfg.analysis.group.groups,
            'factor_direction': cfg.analysis.group.factor_direction,
            'floor_mode': cfg.analysis.group.floor_mode,
            'hedge': cfg.analysis.group.hedge,
            'hedge_path': cfg.analysis.group.hedge_path,
        })
        ctx['GroupAnalyzer'] = group_analyzer.execute(context=ctx)
        print(f"  分组数: {cfg.analysis.group.groups}")

        print("\n[Phase 9] 多空组合...")
        long_short = LongShortNode(config={
            'factor_direction': cfg.analysis.longshort.factor_direction,
        })
        ctx['LongShort'] = long_short.execute(context=ctx)

        print("\n[Phase 10] 市值行业分层打分...")
        score = FactorScoreNode(config={
            'enabled': cfg.analysis.score.enabled,
        })
        ctx['FactorScore'] = score.execute(context=ctx)

        print("\n[Phase 11] 风险因子相关性...")
        risk_corr = RiskCorrelationNode(config={
            'factors': cfg.analysis.risk_corr.factors,
        })
        ctx['RiskCorrelation'] = risk_corr.execute(context=ctx)

        # ============================================================
        # Phase 3: 输出
        # ============================================================
        print("\n[Phase 12] 生成报告...")
        report = FactorTestReportNode(config={
            'dir': cfg.output.dir,
            'format': cfg.output.format,
        })
        ctx['FactorTestReport'] = report.execute(context=ctx)

        # ============================================================
        # Phase 4: FactorFeedback 自动包装 (可选)
        # ============================================================
        if feedback_enabled:
            ctx['Feedback'] = self._build_feedback(
                ctx, factor_id, factor_name, judge,
            )
            self._maybe_persist_feedback(ctx['Feedback'], cfg)
            n_passed = sum(1 for fb in ctx['Feedback'].values() if fb.decision)
            print(f"\n[Feedback] 包装完成: {len(ctx['Feedback'])} 节点, {n_passed} 通过")

        print("\n" + "=" * 60)
        print("单因子回测完成!")
        print("=" * 60)

        return ctx

    def _build_feedback(
        self,
        ctx: dict,
        factor_id: str,
        factor_name: str,
        judge: Optional[LLMJudge],
    ) -> dict:
        """包装 5 个分析节点返回值为 FactorFeedback。

        包装策略:
            - 节点返回 dict → ensure_feedback() 创建 metadata-only FactorFeedback
            - 节点已是 FactorFeedback → 直接用其 channels
            - 都通过同一个 FeedbackCollector 聚合, 共享 factor_id
            - judge 不为 None 时追加 LLM 一致性通道
        """
        feedbacks: dict[str, FactorFeedback] = {}
        for node_name in _ANALYSIS_NODES:
            result = ctx.get(node_name)
            if result is None:
                continue
            collector = FeedbackCollector(factor_id, factor_name)
            fb = ensure_feedback(result, factor_id, factor_name)
            for _ch, ch_fb in fb.channels.items():
                collector.add_feedback(ch_fb)
            if not fb.channels:
                collector.add(
                    channel=FeedbackChannel.VALUE,
                    passed=fb.decision,
                    detail=fb.summary or f"{node_name} 节点无显式通道反馈",
                    score=1.0 if fb.decision else 0.0,
                )
            if judge is not None:
                hypothesis = getattr(self.config.factor, "hypothesis", "") or ""
                description = getattr(self.config.factor, "description", "") or ""
                expression = getattr(self.config.factor, "expression", "") or ""
                if hypothesis or description or expression:
                    llm_fb = judge.judge(hypothesis, description, expression)
                    collector.add_feedback(llm_fb)
            feedbacks[node_name] = collector.finalize(
                summary=fb.summary or f"{node_name} 节点执行完成",
            )
        return feedbacks

    def _maybe_persist_feedback(
        self,
        feedbacks: dict[str, FactorFeedback],
        cfg: SingleFactorTestConfig,
    ) -> None:
        """可选: 持久化 Feedback 到 Parquet。"""
        if cfg.feedback.output_dir is None:
            return
        out = Path(cfg.feedback.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        parquet_path = out / "feedback.parquet"
        for node_name, fb in feedbacks.items():
            fb.save_parquet(parquet_path)

    @staticmethod
    def _maybe_build_judge(cfg: SingleFactorTestConfig) -> Optional[LLMJudge]:
        """若 judge_enabled, 构建 LLMJudge; 否则 None。"""
        if not cfg.feedback.judge_enabled:
            return None
        return LLMJudge(
            model=cfg.feedback.judge_model,
            max_correction_attempts=cfg.feedback.judge_max_attempts,
        )

    @property
    def context(self) -> dict:
        """获取当前上下文"""
        return self._context


# ============================================================
# 便捷函数
# ============================================================

def run_single_factor_test(config: dict) -> dict:
    """便捷函数: 从 dict 配置运行单因子回测

    Args:
        config: 配置字典

    Returns:
        dict: 完整结果
    """
    runner = PipelineRunner.from_dict(config)
    return runner.run()


def run_single_factor_test_yaml(yaml_path: str) -> dict:
    """便捷函数: 从 YAML 配置运行单因子回测

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        dict: 完整结果
    """
    runner = PipelineRunner.from_yaml(yaml_path)
    return runner.run()


# ============================================================
# 需要 import pandas 用于 type hints
# ============================================================
import pandas as pd
