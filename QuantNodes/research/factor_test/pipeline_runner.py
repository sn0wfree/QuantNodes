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
from QuantNodes.core.evolution import (
    EvolutionLoop,
    EvolutionResult,
    FactorCandidate,
)
from QuantNodes.core.quality_gate import QualityGateNode
from QuantNodes.core.trajectory import TrajectoryPool
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
        # 若 _context 已注入 LoadData, 复用, 否则调 LoadDataNode
        if "LoadData" in self._context:
            ctx = dict(self._context)
            print(f"  [LoadData 跳过] 使用已注入数据 (factor shape: {ctx['LoadData'].get('factor', pd.DataFrame()).shape})")
        else:
            ctx = {}
            print("\n[Phase 1] 数据加载...")
            load_data_cfg = {
                'factor': cfg.factor.model_dump(),
                'data_path': cfg.data_path,
                'load_keys': cfg.load_keys,
            }
            load_data = LoadDataNode(config=load_data_cfg)
            ctx['LoadData'] = load_data.execute()
            print(f"  因子形状: {ctx['LoadData'].get('factor', pd.DataFrame()).shape}")
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

    # ============================================================
    # Week 4: 演化集成
    # ============================================================

    def _build_quality_gate(self) -> Optional["QualityGateNode"]:
        """根据 config.quality_gate.enabled 构造 QualityGateNode。"""
        if not self.config.quality_gate.enabled:
            return None
        from QuantNodes.core.quality_gate import FactorZoo, QualityGateNode, QualityGateSetting
        from pathlib import Path as _P
        zoo_path = (
            _P(self.config.quality_gate.zoo_path)
            if self.config.quality_gate.zoo_path
            else None
        )
        zoo = FactorZoo(zoo_path) if zoo_path is not None else FactorZoo()
        return QualityGateNode(QualityGateSetting(), zoo=zoo)

    def _build_trajectory_pool(self) -> Optional["TrajectoryPool"]:
        """根据 config.evolution.pool_dir 构造 TrajectoryPool。"""
        if not self.config.evolution.enabled:
            return None
        from QuantNodes.core.trajectory import TrajectoryPool
        from pathlib import Path as _P
        if self.config.evolution.pool_dir:
            base = _P(self.config.evolution.pool_dir)
        else:
            base = _P(self.config.output.dir) / "trajectory"
        return TrajectoryPool(base)

    def _build_evolution_loop(
        self,
        pool: "TrajectoryPool",
        quality_gate: Optional["QualityGateNode"],
        workers: int = 1,
    ) -> "EvolutionLoop":
        """构造 EvolutionLoop, evaluate_fn 默认委托给 self._run_candidate。"""
        from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting
        settings = EvolutionSetting(
            enabled=True,
            max_rounds=self.config.evolution.max_rounds,
            parents_per_round=self.config.evolution.parents_per_round,
            parent_selection_strategy=self.config.evolution.parent_selection_strategy,
            top_percent_threshold=self.config.evolution.top_percent_threshold,
            metric=self.config.evolution.metric,
            early_stop_patience=self.config.evolution.early_stop_patience,
        )
        return EvolutionLoop(
            settings=settings,
            pool=pool,
            quality_gate=quality_gate,
            evaluate_fn=self._evaluate_candidate,
            workers=workers,
        )

    def _evaluate_candidate(
        self,
        candidate: "FactorCandidate",
    ) -> tuple[bool, dict, "FactorFeedback"]:
        """EvolutionLoop 评估回调: 执行单次回测 + 提取 metrics + 构造 feedback。

        Returns:
            (passed, metrics, feedback)
        """
        from QuantNodes.core.evolution import FactorCandidate as _FC
        from QuantNodes.core.feedback import FactorFeedback
        if not isinstance(candidate, _FC):
            raise TypeError(f"expected FactorCandidate, got {type(candidate)}")

        try:
            ctx = self._run_one_factor(candidate)
        except Exception as e:  # noqa: BLE001
            return False, {}, FactorFeedback(
                factor_id=candidate.factor_id,
                factor_name=candidate.name,
                decision=False,
                summary=f"evaluate failed: {e}",
            )

        passed = bool(ctx.get("status") != "rejected")
        metrics = _extract_metrics_from_ctx(ctx)
        factor_id = str(candidate.factor_id)
        factor_name = str(candidate.name)
        feedback = FactorFeedback(
            factor_id=factor_id,
            factor_name=factor_name,
            decision=passed,
            summary="ok" if passed else "rejected",
            metadata=metrics,
        )
        if self.config.feedback.enabled and "Feedback" in ctx:
            for node_fb in ctx["Feedback"].values():
                for ch, ch_fb in node_fb.channels.items():
                    feedback.channels[ch] = ch_fb
        return passed, metrics, feedback

    def _run_one_factor(self, candidate: "FactorCandidate") -> dict:
        """执行单次回测 (Phase 1-12), 不写 TrajectoryPool。

        临时把 candidate 的 expression / name 注入 config.factor 后跑 12 节点。
        """
        from QuantNodes.core.evolution import FactorCandidate as _FC
        from QuantNodes.core.feedback import FactorFeedback
        if not isinstance(candidate, _FC):
            raise TypeError(f"expected FactorCandidate, got {type(candidate)}")
        # 临时覆盖 factor.name / expression
        original_name = self.config.factor.name
        self.config.factor.name = candidate.name
        if not hasattr(self.config.factor, "expression") or not self.config.factor.expression:
            try:
                self.config.factor.expression = candidate.expression  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            ctx = self.run()
            return ctx
        finally:
            self.config.factor.name = original_name

    def run_evolution(
        self,
        initial_directions: list[str] | None = None,
        initial_candidates: list["FactorCandidate"] | None = None,
        workers: int = 1,
    ) -> "EvolutionResult":
        """多轮演化主入口。

        Args:
            initial_directions: round 0 用的研究假设列表 (Hypothesizer 处理)
            initial_candidates: round 0 用的直接候选 (跳过 Hypothesizer)
            workers: 并行数 (1=串行, >1=ThreadPool/ProcessPool 并行)

        Returns:
            EvolutionResult: best entries + 统计

        Raises:
            ValueError: config.evolution.enabled=False
        """
        from QuantNodes.core.evolution import EvolutionResult
        if not self.config.evolution.enabled:
            raise ValueError("config.evolution.enabled=False, 无法运行演化")

        pool = self._build_trajectory_pool()
        quality_gate = self._build_quality_gate()
        loop = self._build_evolution_loop(pool, quality_gate, workers=workers)

        # workers > 1 + 有 _loader → 预序列化供 ProcessPool
        if workers > 1 and "_loader" in self._context.get("LoadData", {}):
            from QuantNodes.core.parallel.worker_process import prepare_snapshot
            snapshot = prepare_snapshot(
                self.config, self._context,
                factor_path=getattr(self.config.factor, "factor_dir", None),
            )
            from pathlib import Path as _P
            snap_path = _P(pool.base_dir) / "_snapshot.pkl"
            snapshot.save(snap_path)
            loop.snapshot_path = str(snap_path)
            print(f"  [ProcessPool] 预序列化快照: {snap_path}")

        # 连接 MetricCollector (streaming)
        from QuantNodes.core.monitoring import MetricCollector
        collector = MetricCollector()
        loop.metric_collector = collector

        result = loop.run(
            initial_directions=initial_directions,
            initial_candidates=initial_candidates,
        )

        # 演化结束后, 追加写入 JSON + 生成 streaming dashboard
        if pool.size > 0:
            metrics_json = pool.base_dir / "metrics.json"
            collector.append_json(metrics_json)
            print(f"  [Streaming] 指标追加: {metrics_json}")
            # 生成 streaming dashboard (可选: 若 feedback.enabled)
            from QuantNodes.core.monitoring import generate_dashboard_html
            dashboard_html = pool.base_dir.parent / "dashboard_streaming.html"
            generate_dashboard_html(
                collector,
                title=f"演化 Dashboard (Streaming): {self.config.factor.name}",
                output_path=str(dashboard_html),
                streaming=True,
            )
            print(f"  [Streaming] Dashboard: {dashboard_html}")

        return result

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


# ============================================================
# 工具: 从 ctx 提取指标 (IC, Sharpe, ARR, MDD, Calmar)
# ============================================================

def _extract_metrics_from_ctx(ctx: dict) -> dict:
    """从单次回测 ctx 提取关键指标, 供 TrajectoryEntry.metrics。"""
    metrics: dict = {}
    ic = ctx.get("ICAnalyzer") or {}
    ic_result = ic.get("ic_result") if isinstance(ic, dict) else None
    if isinstance(ic_result, dict):
        for src_key, dst_key in (
            ("IC均值", "ic_mean"),
            ("Rank IC均值", "rank_ic_mean"),
            ("ICIR", "ic_ir"),
        ):
            if src_key in ic_result and ic_result[src_key] is not None:
                try:
                    metrics[dst_key] = float(ic_result[src_key])
                except (TypeError, ValueError):
                    pass
    ls = ctx.get("LongShort") or {}
    if isinstance(ls, dict):
        for src_key, dst_key in (
            ("sharpe", "sharpe"),
            ("annualized_return", "arr"),
            ("max_drawdown", "mdd"),
            ("calmar", "calmar"),
        ):
            if src_key in ls and ls[src_key] is not None:
                try:
                    metrics[dst_key] = float(ls[src_key])
                except (TypeError, ValueError):
                    pass
    return metrics
