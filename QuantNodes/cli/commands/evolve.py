# coding=utf-8
"""``quantnodes evolve`` command (Week 5 multi-round evolution entry)."""

import json
from pathlib import Path


def _load_runner_from_config(config_path: str):
    """从 YAML 配置构造 PipelineRunner (延迟 import 避免顶层依赖)。"""
    from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
    return PipelineRunner.from_yaml(config_path)


def cmd_evolve(args) -> int:
    """多轮演化主入口。

    用法:
        quantnodes evolve --config configs/single_factor.yaml \\
                         --directions momentum,reversal,volatility \\
                         --max-rounds 3
    """
    from QuantNodes.core.evolution import FactorCandidate

    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return 1

    directions = [d.strip() for d in (args.directions or "").split(",") if d.strip()]
    initial = None
    if args.initial_json:
        try:
            raw = json.loads(args.initial_json)
            initial = [FactorCandidate(**c) for c in raw]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"错误: --initial-json 解析失败: {e}")
            return 1

    try:
        runner = _load_runner_from_config(config_path)
    except Exception as e:
        print(f"错误: 加载配置失败: {e}")
        return 1

    # CLI 参数覆盖 config (如指定) — M13-M15
    # 用 getattr 防御性取值 (Args 子类可能未定义这些属性)
    _min_ipo_days = getattr(args, "min_ipo_days", None)
    _min_group_size = getattr(args, "min_group_size", None)
    _groups = getattr(args, "groups", None)
    if _min_ipo_days is not None:
        runner.config.preprocess.tradable.min_ipo_days = _min_ipo_days
    if _min_group_size is not None:
        runner.config.analysis.ic.min_group_size = _min_group_size
    if _groups is not None:
        runner.config.analysis.group.groups = _groups

    print("=" * 60)
    print(f"演化实验: {config_path}")
    print(f"  方向: {directions or '(无, 走 initial_candidates)'}")
    print(f"  max_rounds: {runner.config.evolution.max_rounds}")
    print(f"  early_stop: {runner.config.evolution.early_stop_patience}")
    if _min_ipo_days is not None:
        print(f"  min_ipo_days: {runner.config.preprocess.tradable.min_ipo_days} (CLI override)")
    if _min_group_size is not None:
        print(f"  min_group_size: {runner.config.analysis.ic.min_group_size} (CLI override)")
    if _groups is not None:
        print(f"  groups: {runner.config.analysis.group.groups} (CLI override)")
    print("=" * 60)

    try:
        result = runner.run_evolution(
            initial_directions=directions or None,
            initial_candidates=initial,
            workers=getattr(args, 'workers', 1),
        )
    except Exception as e:
        print(f"错误: 演化失败: {e}")
        return 1

    print()
    print("=" * 60)
    print(f"演化完成: {result.rounds_completed} 轮")
    print(f"  总数: {result.total_count}, 拒绝: {result.rejected_count}")
    print(f"  Top {len(result.best_entries)} entries:")
    for i, e in enumerate(result.best_entries[:5], 1):
        metric_val = e.metrics.get(runner.config.evolution.metric, 0)
        name = e.feedback.factor_name if e.feedback else e.entry_id[:8]
        print(f"    {i}. {name} [{e.operation} r{e.round_idx}] "
              f"{runner.config.evolution.metric}={metric_val:.4f}")
    print("=" * 60)
    return 0
