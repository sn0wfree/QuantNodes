# coding: utf-8
"""声明式 12 节点 Pipeline / Pipeline Spec.

把 ``pipeline_runner.run()`` 中 137 行手写阶段拆为 ``PIPELINE_SPEC`` 数据驱动:

- ``PhaseSpec`` 描述每阶段的 ``(name, node_cls, build_cfg, log_summary)``
- 增 / 删 / 调序节点 = 改 ``PIPELINE_SPEC`` 列表 (单点)
- ``run()`` 主循环 ~10 行

Phase R2 (2026-06-19): 从 pipeline_runner.py 抽出, 单一职责.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from QuantNodes.research.factor_test.config import SingleFactorTestConfig
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
from QuantNodes.research.factor_test.nodes.risk_correlation_node import RiskCorrelationNode
from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode


@dataclass(frozen=True)
class PhaseSpec:
    """单个 Pipeline 阶段描述.

    Attributes:
        name: 节点名 (用作 ``ctx[name]`` 键)
        phase_no: 1-12 阶段编号 (仅显示)
        title: 阶段中文标题 (打印 banner 用)
        node_cls: 节点类 (``BaseNode`` 子类)
        build_cfg: ``cfg -> dict`` 构造节点 config
        skip_if_in_ctx: True 时若 ``name in ctx`` 则跳过 (LoadData 注入场景)
        log_summary: ``(cfg, output) -> str`` 阶段成功后的额外日志, 可为 None
    """
    name: str
    phase_no: int
    title: str
    node_cls: type
    build_cfg: Callable[[SingleFactorTestConfig], dict]
    skip_if_in_ctx: bool = False
    log_summary: Optional[Callable[[SingleFactorTestConfig, Any], str]] = None


# ── 12 节点 config builder ────────────────────────────────────

def _cfg_load(cfg: SingleFactorTestConfig) -> dict:
    return {
        "factor": cfg.factor.model_dump(),
        "data_path": cfg.data_path,
        "load_keys": cfg.load_keys,
    }


def _cfg_sample(cfg: SingleFactorTestConfig) -> dict:
    return {
        "sample_index": cfg.preprocess.sample_index,
        "sample_industry": cfg.preprocess.sample_industry,
        "sample_index_customdir": cfg.preprocess.sample_index_customdir,
    }


def _cfg_trad(cfg: SingleFactorTestConfig) -> dict:
    return {"tradable": cfg.preprocess.tradable.model_dump()}


def _cfg_adj(cfg: SingleFactorTestConfig) -> dict:
    return {
        "adj_date_beg": cfg.preprocess.adj_date_beg,
        "adj_date_end": cfg.preprocess.adj_date_end,
        "adj_mode": cfg.preprocess.adj_mode,
    }


def _cfg_preprocess(cfg: SingleFactorTestConfig) -> dict:
    return {
        "missing": cfg.preprocess.missing,
        "extreme": cfg.preprocess.extreme,
        "norm": cfg.preprocess.norm,
    }


def _cfg_neutralize(cfg: SingleFactorTestConfig) -> dict:
    return {
        "industry_neutral": cfg.preprocess.industry_neutral,
        "risk_neutral": cfg.preprocess.risk_neutral,
        "risk_factors": cfg.preprocess.risk_factors,
    }


def _cfg_ic(cfg: SingleFactorTestConfig) -> dict:
    return {"min_group_size": cfg.analysis.ic.min_group_size}


def _cfg_group(cfg: SingleFactorTestConfig) -> dict:
    return {
        "groups": cfg.analysis.group.groups,
        "factor_direction": cfg.analysis.group.factor_direction,
        "floor_mode": cfg.analysis.group.floor_mode,
        "hedge": cfg.analysis.group.hedge,
        "hedge_path": cfg.analysis.group.hedge_path,
    }


def _cfg_longshort(cfg: SingleFactorTestConfig) -> dict:
    return {"factor_direction": cfg.analysis.longshort.factor_direction}


def _cfg_score(cfg: SingleFactorTestConfig) -> dict:
    return {"enabled": cfg.analysis.score.enabled}


def _cfg_risk_corr(cfg: SingleFactorTestConfig) -> dict:
    return {"factors": cfg.analysis.risk_corr.factors}


def _cfg_report(cfg: SingleFactorTestConfig) -> dict:
    return {"dir": cfg.output.dir, "format": cfg.output.format}


# ── log_summary 辅助 ──────────────────────────────────────────

def _log_adj(cfg, out) -> str:
    return f"  调仓日数: {len(out)}"


def _log_preprocess(cfg, out) -> str:
    shape = getattr(out, "shape", "?")
    return f"  预处理后因子形状: {shape}"


def _log_ic(cfg, out) -> str:
    if not isinstance(out, dict):
        return ""
    ic_result = out.get("ic_result")
    if not isinstance(ic_result, dict):
        return ""
    parts = []
    if ic_result.get("IC均值") is not None:
        parts.append(f"IC均值: {ic_result['IC均值']:.4f}")
    if ic_result.get("ICIR") is not None:
        parts.append(f"ICIR: {ic_result['ICIR']:.4f}")
    return "  " + " | ".join(parts) if parts else ""


def _log_group(cfg, out) -> str:
    return f"  分组数: {cfg.analysis.group.groups}"


# ── 12 节点 Pipeline 声明 ─────────────────────────────────────

PIPELINE_SPEC: list[PhaseSpec] = [
    PhaseSpec(
        name="LoadData", phase_no=1, title="数据加载",
        node_cls=LoadDataNode, build_cfg=_cfg_load,
        skip_if_in_ctx=True,
    ),
    PhaseSpec(
        name="SamplePoolFilter", phase_no=2, title="样本池筛选",
        node_cls=SamplePoolFilterNode, build_cfg=_cfg_sample,
    ),
    PhaseSpec(
        name="TradabilityFilter", phase_no=3, title="可交易性筛选",
        node_cls=TradabilityFilterNode, build_cfg=_cfg_trad,
    ),
    PhaseSpec(
        name="AdjustDate", phase_no=4, title="调仓日生成",
        node_cls=AdjustDateNode, build_cfg=_cfg_adj,
        log_summary=_log_adj,
    ),
    PhaseSpec(
        name="FactorPreprocess", phase_no=5, title="因子预处理",
        node_cls=FactorPreprocessNode, build_cfg=_cfg_preprocess,
        log_summary=_log_preprocess,
    ),
    PhaseSpec(
        name="FactorNeutralize", phase_no=6, title="因子中性化",
        node_cls=FactorNeutralizeNode, build_cfg=_cfg_neutralize,
    ),
    PhaseSpec(
        name="ICAnalyzer", phase_no=7, title="IC 分析",
        node_cls=ICAnalyzerNode, build_cfg=_cfg_ic,
        log_summary=_log_ic,
    ),
    PhaseSpec(
        name="GroupAnalyzer", phase_no=8, title="分组分析",
        node_cls=GroupAnalyzerNode, build_cfg=_cfg_group,
        log_summary=_log_group,
    ),
    PhaseSpec(
        name="LongShort", phase_no=9, title="多空组合",
        node_cls=LongShortNode, build_cfg=_cfg_longshort,
    ),
    PhaseSpec(
        name="FactorScore", phase_no=10, title="市值行业分层打分",
        node_cls=FactorScoreNode, build_cfg=_cfg_score,
    ),
    PhaseSpec(
        name="RiskCorrelation", phase_no=11, title="风险因子相关性",
        node_cls=RiskCorrelationNode, build_cfg=_cfg_risk_corr,
    ),
    PhaseSpec(
        name="FactorTestReport", phase_no=12, title="生成报告",
        node_cls=FactorTestReportNode, build_cfg=_cfg_report,
    ),
]
