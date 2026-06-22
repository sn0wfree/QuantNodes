# coding=utf-8
"""``quantnodes factor-*`` commands.

Week 5~13 entry points for trajectory pool inspection / RAG evaluation /
visual reporting / iFinD data fetching / live monitoring dashboards.
"""

from pathlib import Path

from QuantNodes.cli._helpers import cli_safe_run
from QuantNodes.cli.command import Command


@cli_safe_run
def cmd_factor_info(args) -> int:
    """显示 TrajectoryPool 统计信息。

    用法:
        quantnodes factor-info --pool-dir output/trajectory/
    """
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    print("=" * 60)
    print(f"TrajectoryPool: {pool_dir}")
    print(f"  size: {pool.size}")
    by_round: dict = {}
    for e in pool.all():
        by_round.setdefault(e.round_idx, 0)
        by_round[e.round_idx] += 1
    print(f"  by_round: {by_round}")
    by_op: dict = {}
    for e in pool.all():
        by_op.setdefault(e.operation, 0)
        by_op[e.operation] += 1
    print(f"  by_operation: {by_op}")
    n_passed = sum(1 for e in pool.all() if e.feedback and e.feedback.decision)
    print(f"  passed: {n_passed} / {pool.size}")
    print("=" * 60)
    return 0


@cli_safe_run
def cmd_factor_best(args) -> int:
    """显示 Top-N 最佳 entry (按 metric 排序)。

    用法:
        quantnodes factor-best --pool-dir output/trajectory/ --top 5 --metric sharpe
    """
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    top = pool.best(top_n=args.top, metric=args.metric)
    print("=" * 60)
    print(f"Top {len(top)} entries by {args.metric}:")
    for i, e in enumerate(top, 1):
        metric_val = e.metrics.get(args.metric, 0)
        name = e.feedback.factor_name if e.feedback else e.entry_id[:8]
        print(f"  {i}. {name} [{e.operation} r{e.round_idx}] "
              f"{args.metric}={metric_val:.4f}")
    print("=" * 60)
    return 0


@cli_safe_run
def cmd_factor_visual(args) -> int:
    """生成可视化 HTML 报告 (谱系 DAG + 指标分布 + 拦截率 + 趋势)。

    用法:
        quantnodes factor-visual --pool-dir output/trajectory/ \\
                                --output report.html --metric sharpe
    """
    from QuantNodes.core.trajectory import TrajectoryPool
    from QuantNodes.core.visualization import generate_html

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无 entry 可视化")
        return 1

    output = args.output or str(Path(pool_dir).parent / f"{Path(pool_dir).name}_report.html")
    title = args.title or f"QuantNodes 演化报告: {pool_dir}"
    generate_html(pool, metric=args.metric, title=title, output_path=output)
    print(f"✓ HTML 报告已生成: {output}")
    print(f"  size: {pool.size}, metric: {args.metric}")
    return 0


@cli_safe_run
def cmd_factor_dashboard(args) -> int:
    """生成 3 类指标 dashboard (Week 13)。

    从 TrajectoryPool 提取 RAG + Evolution + Quality Gate 指标,
    生成 Plotly 6 图 + 概览表 HTML 报告。

    用法:
        quantnodes factor-dashboard --pool-dir output/trajectory/ \\
                                     --output dashboard.html
    """
    from QuantNodes.core.monitoring import (
        MetricCollector,
        generate_dashboard_html,
    )
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无指标可显示")
        return 1

    output = args.output or str(Path(pool_dir).parent / f"{Path(pool_dir).name}_dashboard.html")
    title = args.title or f"QuantNodes 演化 Dashboard: {Path(pool_dir).name}"

    # 收集 3 类指标
    collector = MetricCollector()

    # RAG: 从 TrajectoryEntry.feedback 元数据中提取 (兼容 rag_metrics_history 缺失)
    rounds = sorted({e.round_idx for e in pool.all()})
    for r in rounds:
        round_entries = [e for e in pool.all() if e.round_idx == r]
        n_total = len(round_entries)
        n_passed = sum(1 for e in round_entries if e.feedback and e.feedback.decision)
        # RAG 指标的简单代理: pass rate 作为 HR@5
        if n_total > 0:
            from QuantNodes.core.monitoring import RagMetrics
            collector.add_rag(RagMetrics(
                round=r, n_queries=n_total,
                hit_at_5=n_passed / n_total, hit_at_10=n_passed / n_total,
                ndcg_at_5=n_passed / n_total, ndcg_at_10=n_passed / n_total,
                mrr=n_passed / n_total,
                lineage_coverage=0.0,
                diversity=1.0,
            ))

    # Evolution: 累积统计
    from QuantNodes.core.monitoring import EvolutionMetrics
    for r in rounds:
        round_entries = [e for e in pool.all() if e.round_idx <= r]
        n_passed = sum(1 for e in round_entries if e.feedback and e.feedback.decision)
        n_total = len(round_entries)
        n_rejected = n_total - n_passed
        best_metric = 0.0
        best_name = ""
        for e in round_entries:
            sharpe = (e.metrics or {}).get("sharpe", 0)
            if sharpe > best_metric:
                best_metric = sharpe
                if e.feedback:
                    best_name = e.feedback.factor_name
        collector.add_evolution(EvolutionMetrics(
            round=r, pool_size=n_total,
            total_count=n_passed, rejected_count=n_rejected,
            best_metric=best_metric, best_factor_name=best_name,
        ))

    # Quality: 每 round 通道统计
    for r in rounds:
        collector.update_quality_from_pool(pool, round_idx=r)

    print("=" * 60)
    print(f"Dashboard 收集 ({len(collector)} metrics):")
    print(f"  RAG:    {len(collector.rag_history)} rounds")
    print(f"  Evo:    {len(collector.evolution_history)} rounds")
    print(f"  Quality: {len(collector.quality_history)} rounds")
    print("=" * 60)

    try:
        streaming = getattr(args, "streaming", False) or getattr(args, "watch", False)
        refresh_sec = getattr(args, "refresh", 10)
        generate_dashboard_html(
            collector, title=title, output_path=output,
            streaming=streaming, refresh_interval_sec=refresh_sec,
        )
    except Exception as e:
        # 嵌套: 这里包了 generate_dashboard_html 的内部异常。
        # 整体 cmd_factor_dashboard 由 @cli_safe_run 包了顶层异常,
        # 这里只针对 generate_dashboard_html 单独报错以便区分。
        print(f"错误: 生成 dashboard 失败: {e}")
        return 1

    # 同时保存 JSON (供后续分析)
    metrics_json = output.replace(".html", "_metrics.json")
    collector.save(metrics_json)
    print(f"✓ Dashboard: {output}")
    print(f"✓ Metrics JSON: {metrics_json}")

    # Watch 模式: 后台定时刷新
    if getattr(args, "watch", False):
        import time as _time
        refresh_sec = getattr(args, "refresh", 10)
        print(f"\n[Watch] 每 {refresh_sec}s 刷新 dashboard (Ctrl+C 退出)...")
        try:
            while True:
                _time.sleep(refresh_sec)
                # 重载 pool + 重新生成
                try:
                    pool = TrajectoryPool(pool_dir)
                    collector = MetricCollector()
                    for r in rounds:
                        round_entries = [e for e in pool.all() if e.round_idx == r]
                        if not round_entries:
                            continue
                        collector.update_quality_from_pool(pool, round_idx=r)
                    generate_dashboard_html(
                        collector, title=title, output_path=output,
                        streaming=True, refresh_interval_sec=refresh_sec,
                    )
                except Exception:
                    pass  # pool 可能被其他进程写入, 忽略暂时错误
        except KeyboardInterrupt:
            print("\n[Watch] 停止监控")
    return 0


@cli_safe_run
def cmd_factor_data_fetch(args) -> int:
    """从 iFinD 拉取数据 + 写为 HDF5 格式 (Week 12)。

    用法:
        quantnodes factor-data-fetch --output-dir /tmp/real_data/ \\
                                    --universe all \\
                                    --date-beg 20260101 --date-end 20260630 \\
                                    --factors momentum_20d,reversal_5d
    """
    try:
        from QuantNodes.research.factor_test.ifind_db import IFinDDatabase
    except (ImportError, FileNotFoundError) as e:
        print(f"错误: 无法导入 IFinDDatabase: {e}")
        return 1

    output_dir = Path(args.output_dir)
    try:
        db = IFinDDatabase(
            date_beg=args.date_beg,
            date_end=args.date_end,
            universe=args.universe,
        )
    except FileNotFoundError as e:
        print(f"错误: iFinD 配置缺失: {e}")
        return 1
    except ValueError as e:
        print(f"错误: iFinD auth_token 无效: {e}")
        return 1

    factor_names = [f.strip() for f in (args.factors or "").split(",") if f.strip()]

    print("=" * 60)
    print("iFinD 数据拉取")
    print(f"  universe: {args.universe}")
    print(f"  date range: {args.date_beg} ~ {args.date_end}")
    print(f"  output_dir: {output_dir}")
    print(f"  factors: {factor_names or '(none)'}")
    print("=" * 60)

    stats = db.fetch_to_h5(output_dir, factor_names=factor_names)

    print()
    print("=" * 60)
    print("✓ 完成, 统计:")
    for fname, file_stats in stats.items():
        if isinstance(file_stats, dict):
            keys_info = ", ".join(
                f"{k}={v}" for k, v in file_stats.items() if v
            )
            print(f"  {fname}: {keys_info or '(empty)'}")
    print("=" * 60)
    return 0


@cli_safe_run
def cmd_factor_rag_eval(args) -> int:
    """批量评估 RAG 检索质量 (Week 10)。

    用法:
        quantnodes factor-rag-eval --pool-dir output/trajectory/ \\
                                   --queries "momentum,reversal,volatility" \\
                                   --top 5 \\
                                   --output eval.json
    """
    from QuantNodes.core.knowledge import (
        IdentityRetriever,
        KnowledgeBase,
        RAGEvaluator,
        expand_lineage,
    )
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无可评估内容")
        return 1

    queries = [q.strip() for q in (args.queries or "").split(",") if q.strip()]
    if not queries:
        print("错误: --queries 至少需要 1 个 query")
        return 1

    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    n = kb.sync_from_pool()

    # 构造评估输入
    all_ids = {e.entry_id for e in pool.all()}
    retrieved: list[list[str]] = []
    relevant: list[list[str]] = []
    relevance_scores: list[dict[str, float]] = []
    lineage_ids: list[list[str]] = []
    token_lists: list[list[list[str]]] = []

    for q in queries:
        results = kb.query(q, top_k=args.top)
        ids = [e.entry_id for e, _ in results]
        retrieved.append(ids)
        relevant.append(list(all_ids))
        relevance_scores.append({eid: 1.0 for eid in ids})
        lin_set: set[str] = set()
        tokens_per_entry: list[list[str]] = []
        for e, _ in results:
            expanded = expand_lineage(
                pool, e.entry_id,
                max_ancestor_depth=args.ancestor_depth,
                max_descendant_depth=args.descendant_depth,
            )
            for _, ee in expanded["ancestors"] + expanded["descendants"]:
                lin_set.add(ee.entry_id)
            cfg = (e.config_snapshot or {}).get("factor", {}) if e else {}
            toks = []
            if cfg.get("name"):
                toks += cfg["name"].lower().split("_")
            if cfg.get("hypothesis"):
                toks += cfg["hypothesis"].lower().split()
            tokens_per_entry.append(toks)
        lineage_ids.append(list(lin_set))
        token_lists.append(tokens_per_entry)

    ev = RAGEvaluator()
    report = ev.evaluate(
        queries=queries,
        retrieved=retrieved,
        relevant=relevant,
        relevance_scores=relevance_scores,
        lineage_ids=lineage_ids,
        token_lists=token_lists,
    )

    if args.output:
        ev.save(report, args.output)
        print(f"✓ EvalReport 已保存: {args.output}")

    print("=" * 60)
    print(f"RAG 评估报告 ({report.n_queries} queries, indexed {n} entries)")
    print(f"  HitRate@5:   {report.hit_at_5:.3f}")
    print(f"  HitRate@10:  {report.hit_at_10:.3f}")
    print(f"  NDCG@5:      {report.ndcg_at_5:.3f}")
    print(f"  NDCG@10:     {report.ndcg_at_10:.3f}")
    print(f"  MRR:         {report.mrr:.3f}")
    print(f"  LineageCov:  {report.lineage_coverage:.3f}")
    print(f"  Diversity:   {report.diversity:.3f}")
    print("=" * 60)
    return 0


@cli_safe_run
def cmd_factor_rag_show(args) -> int:
    """从 TrajectoryPool 检索相似因子 (RAG demo)。

    用法:
        quantnodes factor-rag-show --pool-dir output/trajectory/ \\
                                   --query "momentum effect" --top 5
    """
    from QuantNodes.core.knowledge import (
        Compressor,
        IdentityRetriever,
        KnowledgeBase,
        expand_lineage,
    )
    from QuantNodes.core.trajectory import TrajectoryPool

    pool_dir = args.pool_dir
    if not Path(pool_dir).exists():
        print(f"错误: pool 目录不存在: {pool_dir}")
        return 1

    pool = TrajectoryPool(pool_dir)
    if pool.size == 0:
        print("错误: pool 为空, 无可检索内容")
        return 1

    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    n = kb.sync_from_pool()
    print(f"索引了 {n} 个 entry")

    results = kb.query(args.query, top_k=args.top)
    if not results:
        print(f"无匹配结果 (query: {args.query!r})")
        return 0

    use_compress = getattr(args, "compress", False)
    compressor = Compressor(model="mock", max_tokens=args.max_tokens) if use_compress else None

    print("=" * 60)
    print(f"Top {len(results)} 检索结果 (query: {args.query!r}):")
    for i, (entry, score) in enumerate(results, 1):
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        name = factor_cfg.get("name", entry.entry_id[:8])
        expr = factor_cfg.get("expression", "")[:50]
        sharpe = (entry.metrics or {}).get("sharpe", 0)
        print(f"  {i}. {name}  score={score:.3f}  sharpe={sharpe:.2f}")
        print(f"     expression: {expr}")
        if use_compress and compressor is not None:
            expanded = expand_lineage(
                pool, entry.entry_id,
                max_ancestor_depth=args.ancestor_depth,
                max_descendant_depth=args.descendant_depth,
            )
            c_anc = compressor.compress(expanded["ancestors"], relation="ancestors")
            c_desc = compressor.compress(expanded["descendants"], relation="descendants")
            print(f"     ↑ ancestors ({c_anc.original_count}): {c_anc.summary[:80]}")
            print(f"     ↓ descendants ({c_desc.original_count}): {c_desc.summary[:80]}")
    print("=" * 60)
    return 0


class FactorInfoCommand(Command):
    name = "factor-info"
    description = "显示 TrajectoryPool 统计"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import add_pool_dir_arg
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)

    def run(self, args) -> int:
        return cmd_factor_info(args)


class FactorBestCommand(Command):
    name = "factor-best"
    description = "显示 Top-N 最佳 entry"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import add_pool_dir_arg, add_top_arg, add_metric_arg
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)
        add_top_arg(p)
        add_metric_arg(p)

    def run(self, args) -> int:
        return cmd_factor_best(args)


class FactorVisualCommand(Command):
    name = "factor-visual"
    description = "生成可视化 HTML 报告 (Week 6)"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import (
            add_pool_dir_arg, add_output_arg, add_metric_arg, add_title_arg,
        )
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)
        add_output_arg(p)
        add_metric_arg(p)
        add_title_arg(p)

    def run(self, args) -> int:
        return cmd_factor_visual(args)


class FactorDashboardCommand(Command):
    name = "factor-dashboard"
    description = "生成 3 类指标 dashboard (Week 13/16)"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import add_pool_dir_arg, add_output_arg, add_title_arg
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)
        add_output_arg(p)
        add_title_arg(p)
        p.add_argument(
            "--streaming", action="store_true", help="启用 streaming 模式 (自动刷新 10s)"
        )
        p.add_argument("--refresh", type=int, default=10, help="streaming 刷新间隔秒数 (默认 10)")
        p.add_argument("--watch", action="store_true", help="后台模式: 每 10s 刷新 dashboard")

    def run(self, args) -> int:
        return cmd_factor_dashboard(args)


class FactorDataFetchCommand(Command):
    name = "factor-data-fetch"
    description = "从 iFinD 拉取数据 + 写 H5 (Week 12)"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import add_cli_overrides
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("--output-dir", required=True, help="HDF5 输出目录")
        p.add_argument("--date-beg", required=True, help="起始日期 (YYYYMMDD)")
        p.add_argument("--date-end", default="", help="截止日期 (空=今天)")
        p.add_argument("--universe", default="all", help="股票池 (默认 all, 与 iFinD API 兼容)")
        p.add_argument("--factors", default="", help="逗号分隔的因子列表")
        add_cli_overrides(p)

    def run(self, args) -> int:
        return cmd_factor_data_fetch(args)


class FactorRagEvalCommand(Command):
    name = "factor-rag-eval"
    description = "批量评估 RAG 检索质量 (Week 10)"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import (
            add_pool_dir_arg, add_top_arg, add_lineage_depth_args, add_output_arg,
        )
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)
        p.add_argument("--queries", required=True, help="逗号分隔的 query 列表")
        add_top_arg(p)
        add_lineage_depth_args(p)
        add_output_arg(p)

    def run(self, args) -> int:
        return cmd_factor_rag_eval(args)


class FactorRagShowCommand(Command):
    name = "factor-rag-show"
    description = "RAG 检索演示 (Week 7)"

    def add_arguments(self, subparsers) -> None:
        from .._helpers import (
            add_pool_dir_arg, add_top_arg, add_lineage_depth_args,
        )
        p = subparsers.add_parser(self.name, help=self.description)
        add_pool_dir_arg(p)
        p.add_argument("--query", required=True, help="查询文本")
        add_top_arg(p)
        p.add_argument("--compress", action="store_true", help="启用谱系压缩 (Week 9)")
        add_lineage_depth_args(p)
        p.add_argument("--max-tokens", type=int, default=200, help="压缩最大字符数 (默认 200)")

    def run(self, args) -> int:
        return cmd_factor_rag_show(args)
