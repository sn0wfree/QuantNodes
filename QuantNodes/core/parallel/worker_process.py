"""ProcessPool 工作器 — 顶层 pickle-safe 函数。

设计:
    主进程:
        1. 序列化 config dict + context dict (DataFrames) → 临时 pickle 文件
        2. 提交 subprocess_evaluate(candidate_dict, snapshot_path) 到 ProcessPool
    子进程:
        1. 从 snapshot_path 加载 config + context
        2. 重建 DataLoader (从 context 中的 DataFrames)
        3. 跑 Phase 2-11 (节点 2-11)
        4. 返回结果 dict
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from pathlib import Path
from typing import Any


# ============================================================================
# 顶层 pickle-safe 函数 (子进程入口)
# ============================================================================

def subprocess_evaluate(
    candidate_dict: dict,
    snapshot_path: str,
) -> dict:
    """在子进程中执行评估, 接受候选 dict + 快照路径, 返回结果 dict。

    这是顶层函数, 可被 pickle 传给 ProcessPoolExecutor。

    Args:
        candidate_dict: FactorCandidate 的 dict 形式
        snapshot_path: 预序列化的 config+context 路径

    Returns:
        dict: {passed, metrics, feedback_dict, error}
    """
    try:
        return _evaluate_in_subprocess(candidate_dict, snapshot_path)
    except Exception as e:
        import traceback
        return {
            "passed": False,
            "metrics": {},
            "feedback_dict": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
        }


def _evaluate_in_subprocess(candidate_dict: dict, snapshot_path: str) -> dict:
    """子进程内部实现。"""
    import pandas as pd
    import numpy as np

    # 1. 加载快照
    snapshot = pickle.loads(Path(snapshot_path).read_bytes())
    config = snapshot["config"]
    context = snapshot["context"]
    factor_override_name = snapshot["factor_name"]
    factor_override_expr = snapshot.get("factor_expression", "")

    # 2. 重建因子数据
    factor_path = snapshot.get("factor_path")
    if factor_path and Path(factor_path).exists():
        factor = pd.read_hdf(factor_path, key="data")
    else:
        # 从 context 中的 factor 字段
        factor = context.get("LoadData", {}).get("factor")
        if factor is None:
            factor = pd.DataFrame(np.zeros((10, 10)))

    # 3. 临时覆盖 factor name / expression
    if "factor" not in config:
        config["factor"] = {}
    config["factor"]["name"] = factor_override_name
    if factor_override_expr:
        config["factor"]["expression"] = factor_override_expr

    # 4. 重建 context
    ctx = {}
    ctx["LoadData"] = {
        "factor": factor,
        "price": context.get("LoadData", {}).get("price", pd.DataFrame()),
        "id_citic1": context.get("LoadData", {}).get("id_citic1", pd.DataFrame()),
        "mv_float": context.get("LoadData", {}).get("mv_float", pd.DataFrame()),
        "st": context.get("LoadData", {}).get("st", pd.DataFrame()),
        "suspend": context.get("LoadData", {}).get("suspend", pd.DataFrame()),
        "ud_limit": context.get("LoadData", {}).get("ud_limit", pd.DataFrame()),
        "ipo_days": context.get("LoadData", {}).get("ipo_days", pd.DataFrame()),
        "index_cp": context.get("LoadData", {}).get("index_cp", pd.DataFrame()),
        "stklist": context.get("LoadData", {}).get("stklist", pd.DataFrame()),
        "trade_dt": context.get("LoadData", {}).get("trade_dt", pd.DataFrame()),
        "_loader": context.get("LoadData", {}).get("_loader"),
    }

    # 5. 跑节点 2-11 (Phase 2-11, 跳过 LoadData 和 Report)
    try:
        _run_analysis_nodes(ctx, config)
    except Exception as e:
        return {"passed": False, "metrics": {}, "feedback_dict": None, "error": str(e)}

    # 6. 提取 metrics
    metrics = _extract_metrics(ctx)
    passed = True

    return {
        "passed": passed,
        "metrics": metrics,
        "feedback_dict": {
            "factor_id": candidate_dict.get("factor_id", ""),
            "factor_name": candidate_dict.get("name", ""),
            "decision": passed,
            "summary": f"ProcessPool: sharpe={metrics.get('sharpe', 0):.2f}",
            "metadata": metrics,
            "channels": {},
        },
        "error": None,
    }


def _run_analysis_nodes(ctx: dict, config: dict) -> None:
    """在子进程中跑节点 2-11 (与 PipelineRunner.run 相同逻辑)。"""
    from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
    from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
    from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
    from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
    from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
    from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
    from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
    from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
    from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode

    pp = config.get("preprocess", {})

    # Node 2: SampleFilter
    ctx["SamplePoolFilter"] = SamplePoolFilterNode(config={
        "sample_index": pp.get("sample_index", "all"),
        "sample_industry": pp.get("sample_industry", "all"),
        "sample_index_customdir": pp.get("sample_index_customdir"),
    }).execute(context=ctx)

    # Node 3: TradabilityFilter
    tradable = pp.get("tradable", {})
    if isinstance(tradable, dict):
        tradable_dict = tradable
    elif hasattr(tradable, "model_dump"):
        tradable_dict = tradable.model_dump()
    else:
        tradable_dict = {}
    ctx["TradabilityFilter"] = TradabilityFilterNode(config={
        "tradable": tradable_dict,
    }).execute(context=ctx)

    # Node 4: AdjustDate
    ctx["AdjustDate"] = AdjustDateNode(config={
        "adj_date_beg": pp.get("adj_date_beg", 20260101),
        "adj_date_end": pp.get("adj_date_end", 20260630),
        "adj_mode": pp.get("adj_mode", ["M", "end"]),
    }).execute(context=ctx)

    # Node 5: Preprocess
    ctx["FactorPreprocess"] = FactorPreprocessNode(config={
        "missing": pp.get("missing", ""),
        "extreme": pp.get("extreme", "median"),
        "norm": pp.get("norm", "zscore"),
    }).execute(context=ctx)

    # Node 6: Neutralize
    ctx["FactorNeutralize"] = ctx["FactorPreprocess"]

    # Node 7-10: Analysis
    analysis = config.get("analysis", {})
    ctx["ICAnalyzer"] = ICAnalyzerNode(config={
        "min_group_size": analysis.get("ic", {}).get("min_group_size", 5),
    }).execute(context=ctx)
    ctx["GroupAnalyzer"] = GroupAnalyzerNode(config={
        "groups": analysis.get("group", {}).get("groups", 5),
        "factor_direction": analysis.get("group", {}).get("factor_direction", 1),
        "floor_mode": analysis.get("group", {}).get("floor_mode", "group"),
        "hedge": analysis.get("group", {}).get("hedge", "equal"),
        "hedge_path": analysis.get("group", {}).get("hedge_path"),
    }).execute(context=ctx)
    ctx["LongShort"] = LongShortNode(config={
        "factor_direction": analysis.get("longshort", {}).get("factor_direction", 1),
    }).execute(context=ctx)
    ctx["FactorScore"] = FactorScoreNode(config={
        "enabled": analysis.get("score", {}).get("enabled", True),
    }).execute(context=ctx)


def _extract_metrics(ctx: dict) -> dict:
    """从 ctx 提取指标 (与 PipelineRunner._extract_metrics_from_ctx 相同)。"""
    metrics: dict = {}
    ic = ctx.get("ICAnalyzer") or {}
    ic_result = ic.get("ic_result") if isinstance(ic, dict) else None
    if isinstance(ic_result, dict):
        for src, dst in (("IC均值", "ic_mean"), ("Rank IC均值", "rank_ic_mean"), ("ICIR", "ic_ir")):
            if src in ic_result and ic_result[src] is not None:
                try:
                    metrics[dst] = float(ic_result[src])
                except (TypeError, ValueError):
                    pass
    ls = ctx.get("LongShort") or {}
    if isinstance(ls, dict):
        for src, dst in (("sharpe", "sharpe"), ("annualized_return", "arr"),
                         ("max_drawdown", "mdd"), ("calmar", "calmar")):
            if src in ls and ls[src] is not None:
                try:
                    metrics[dst] = float(ls[src])
                except (TypeError, ValueError):
                    pass
    return metrics


# ============================================================================
# 序列化 / 反序列化 (主进程侧)
# ============================================================================

class RunnerSnapshot:
    """预序列化 config + context, 供子进程读取。"""

    def __init__(self, config_dict: dict, context: dict, factor_path: str | None = None):
        self.config = config_dict
        self.context = context
        self.factor_name = config_dict.get("factor", {}).get("name", "")
        self.factor_path = factor_path or ""

    def save(self, path: Path) -> None:
        """序列化到 pickle 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "config": self.config,
            "context": self.context,
            "factor_name": self.factor_name,
            "factor_path": self.factor_path,
        }
        path.write_bytes(pickle.dumps(snapshot))

    @staticmethod
    def load(path: Path) -> dict:
        """从 pickle 文件加载。"""
        return pickle.loads(Path(path).read_bytes())


def prepare_snapshot(
    config,
    context: dict,
    factor_path: str | None = None,
) -> RunnerSnapshot:
    """从 PipelineRunner 准备序列化快照。

    Args:
        config: SingleFactorTestConfig (会 model_dump)
        context: runner._context (含 DataFrames)
        factor_path: 因子 H5 文件路径 (可选, 若 context 中有)

    Returns:
        RunnerSnapshot (可 save 到文件)
    """
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    else:
        config_dict = dict(config)
    return RunnerSnapshot(config_dict, context, factor_path)
