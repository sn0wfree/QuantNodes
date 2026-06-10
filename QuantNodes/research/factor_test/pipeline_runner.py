# coding: utf-8
"""Pipeline Runner / 管线编排器

混合模式数据传递:
- Phase 1 (严格串联): LoadData >> SampleFilter >> TradabilityFilter >> AdjustDate >> Preprocess >> Neutralize
- Phase 2 (Context 共享): ICAnalyzer / GroupAnalyzer / FactorScore / RiskCorrelation
- Phase 3 (依赖分析): LongShort
- Phase 4 (输出): FactorTestReport
"""

import sys
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


class PipelineRunner:
    """单因子回测管线编排器

    用法:
        config = SingleFactorTestConfig(...)
        runner = PipelineRunner(config)
        result = runner.run()

        # 或从 YAML:
        runner = PipelineRunner.from_yaml("config.yaml")
        result = runner.run()
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
        """
        cfg = self.config
        ctx = {}

        print("=" * 60)
        print(f"单因子回测: {cfg.factor.name}")
        print(f"时间范围: {cfg.preprocess.adj_date_beg} ~ {cfg.preprocess.adj_date_end}")
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

        print("\n" + "=" * 60)
        print("单因子回测完成!")
        print("=" * 60)

        return ctx

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
