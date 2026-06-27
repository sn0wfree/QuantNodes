# coding=utf-8
"""
CLI 命令: alpha-mcts 等 QuantAlpha 子包相关命令

M2 PR 新增：
- `quantnodes alpha-mcts` —— MCTS 因子搜索

后续 M3-M6 PR 会陆续添加：
- `quantnodes alpha-101` (M3)
- `quantnodes alpha-158` (M3)
- `quantnodes alpha-360` (M3)
- `quantnodes alpha-gpt` (M6) ← 新增
- `quantnodes alpha-compare` (M7)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from QuantNodes.cli.command import Command
from QuantNodes.research.quant_alpha.mcts import (
    MCTSFeedbackConfig,
    MCTSSearch,
    MCTSSearchConfig,
)


class AlphaMctsCommand(Command):
    """quantnodes alpha-mcts - MCTS 因子搜索

    用法:
        quantnodes alpha-mcts --iterations 50
        quantnodes alpha-mcts --iterations 100 --data data.parquet
        quantnodes alpha-mcts --iterations 50 --date-column date
        quantnodes alpha-mcts --iterations 50 --max-depth 6
    """
    name = "alpha-mcts"
    description = "MCTS 因子搜索（QuantAlpha M2）"

    def add_arguments(self, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description=(
                "基于 OperatorVocab（162 算子）+ 5 通道反馈的 MCTS 因子搜索。\n"
                "详见 docs/quant_alpha/PROJECT_PLAN.md M2 PR。"
            ),
        )
        parser.add_argument(
            "--iterations", "-n",
            type=int, default=50,
            help="MCTS 迭代次数（默认 50）",
        )
        parser.add_argument(
            "--data",
            type=str, default=None,
            help="行情数据路径（Parquet/CSV/JSON），None=用合成数据",
        )
        parser.add_argument(
            "--date-column",
            type=str, default="date",
            help="日期列名（默认 date）",
        )
        parser.add_argument(
            "--code-column",
            type=str, default="code",
            help="股票代码列名（默认 code）",
        )
        parser.add_argument(
            "--max-depth",
            type=int, default=4,
            help="MCTS 树最大深度（默认 4）",
        )
        parser.add_argument(
            "--exploration-weight",
            type=float, default=1.414,
            help="UCB1 exploration weight（默认 sqrt(2) ≈ 1.414）",
        )
        parser.add_argument(
            "--seed",
            type=int, default=42,
            help="随机种子（默认 42）",
        )
        parser.add_argument(
            "--top-k",
            type=int, default=10,
            help="返回 top-k 节点（默认 10）",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="安静模式（不打印每个节点详情）",
        )

    def run(self, args: argparse.Namespace) -> int:
        # 1. 加载数据
        try:
            df = self._load_data(args.data, args.date_column, args.code_column)
        except Exception as e:
            print(f"❌ 数据加载失败: {e}", file=sys.stderr)
            return 1

        print(f"📊 数据加载完成: {len(df)} 行, {len(df.columns)} 列")
        if not args.quiet:
            print(f"  列: {df.columns[:10].to_list()}{'...' if len(df.columns) > 10 else ''}")

        # 2. 配置 MCTS
        config = MCTSSearchConfig(
            iterations=args.iterations,
            max_depth=args.max_depth,
            exploration_weight=args.exploration_weight,
            seed=args.seed,
            feedback_config=MCTSFeedbackConfig(),
        )
        mcts = MCTSSearch(config=config)

        # 3. 执行搜索
        print(f"🔍 开始 MCTS 搜索: iterations={args.iterations}, max_depth={args.max_depth}")
        result = mcts.search(data=df, date_column=args.date_column)

        # 4. 输出结果
        print()
        print(f"✅ MCTS 搜索完成: {result.elapsed_seconds:.2f}s")
        print(f"  迭代: {result.total_iterations}")
        print(f"  候选公式: {result.formula_count}")
        print(f"  通过验证: {result.valid_count}")
        print(f"  拒绝: {result.rejected_count}")
        print(f"  剪枝: {result.pruned_count}")
        print()
        print(f"🏆 Top {min(args.top_k, len(result.best_k_nodes))} 节点:")
        for i, n in enumerate(result.best_k_nodes[:args.top_k], 1):
            print(
                f"  {i:2d}. score={n.overall_score:.3f} "
                f"depth={n.depth} visits={n.visits} "
                f"formula={n.formula!r}"
            )
            if not args.quiet and n.metadata:
                if "feedback_summary" in n.metadata:
                    print(f"      feedback: {n.metadata['feedback_summary']}")

        return 0

    def _load_data(
        self,
        path: Optional[str],
        date_column: str,
        code_column: str,
    ):
        """加载数据：路径 → Parquet/CSV/JSON；None → 合成"""
        import polars as pl
        import numpy as np

        if path is None:
            # 合成 5 票 × 20 日
            np.random.seed(42)
            dates = (
                [f"2024-01-{d:02d}" for d in range(1, 21)]
            )
            rows = []
            for date in dates:
                for code in ["A", "B", "C", "D", "E"]:
                    rows.append({
                        date_column: date,
                        code_column: code,
                        "close": float(np.random.randn() * 5 + 100),
                        "open": float(np.random.randn() * 5 + 100),
                        "high": float(np.random.randn() * 5 + 102),
                        "low": float(np.random.randn() * 5 + 98),
                        "vol": float(np.random.randint(1000, 5000)),
                    })
            return pl.DataFrame(rows).with_columns(
                pl.col(date_column).str.to_date()
            )

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        if p.suffix == ".parquet":
            return pl.read_parquet(p)
        if p.suffix == ".csv":
            return pl.read_csv(p)
        if p.suffix == ".json":
            return pl.read_json(p)
        raise ValueError(f"不支持的数据格式: {p.suffix}")


# 后续 M3-M6 PR 会添加：
# - Alpha101Command (M3)
# - Alpha158Command (M3)
# - Alpha360Command (M3)
# - AlphaGptCommand (M6)  ← added
# - AlphaCompareCommand (M7)


# ==============================================================================
# M6: Alpha-GPT 命令
# ==============================================================================


class AlphaGptCommand(Command):
    """quantnodes alpha-gpt - Alpha-GPT 5 智能体自动化因子挖掘

    用法:
        quantnodes alpha-gpt --objective "捕捉 A 股反转效应" --data data.parquet
        quantnodes alpha-gpt --objective "..." --iterations 5 --pool-size 10 --backtest
        quantnodes alpha-gpt --objective "..." --llm openai --model gpt-4o
        quantnodes alpha-gpt --objective "..." --quiet --output result.json

    详见 docs/quant_alpha/alpha_gpt_user_guide.md
    """
    name = "alpha-gpt"
    description = "Alpha-GPT 5 智能体自动化因子挖掘（QuantAlpha M6）"

    def add_arguments(self, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description=(
                "基于 nanobot Agent 体系的 5 智能体编排自动化因子挖掘。\n"
                "5 轮 iteration × 5 subagent spawn（idea-generator → formula-translator →\n"
                "evaluator → reflector → critic）。\n"
                "详见 docs/quant_alpha/alpha_gpt_architecture.md。"
            ),
        )
        # 必选
        parser.add_argument(
            "--objective", "-o",
            type=str, required=True,
            help="研究目标（如 '捕捉 A 股反转效应'）",
        )
        parser.add_argument(
            "--data", "-d",
            type=str, default=None,
            help="行情数据路径（Parquet/CSV），None=合成测试数据",
        )
        # 工作流
        parser.add_argument(
            "--iterations", "-n",
            type=int, default=5,
            help="迭代轮次（默认 5）",
        )
        parser.add_argument(
            "--pool-size", "-p",
            type=int, default=10,
            help="每轮想法/公式数量（默认 10）",
        )
        parser.add_argument(
            "--top-k", "-k",
            type=int, default=10,
            help="最终返回的 top-K 公式数量（默认 10）",
        )
        parser.add_argument(
            "--min-ir",
            type=float, default=0.5,
            help="IR 阈值（默认 0.5）",
        )
        parser.add_argument(
            "--max-mutual-ic",
            type=float, default=0.7,
            help="最大 mutual IC（默认 0.7）",
        )
        # LLM
        parser.add_argument(
            "--llm",
            type=str, default="deepseek",
            choices=["deepseek", "openai", "qwen", "azure", "mock"],
            help="LLM provider（默认 deepseek）",
        )
        parser.add_argument(
            "--model",
            type=str, default=None,
            help="模型名（如 'gpt-4o' / 'deepseek-chat'）",
        )
        parser.add_argument(
            "--temperature",
            type=float, default=0.7,
            help="采样温度（默认 0.7）",
        )
        # 数据
        parser.add_argument(
            "--date-column",
            type=str, default="date",
            help="日期列名（默认 date）",
        )
        parser.add_argument(
            "--code-column",
            type=str, default="code",
            help="股票代码列名（默认 code）",
        )
        parser.add_argument(
            "--forward-returns",
            type=str, default="1,5,20",
            help="前瞻期列表（逗号分隔，默认 '1,5,20'）",
        )
        # Trading 回测（可选）
        parser.add_argument(
            "--backtest",
            action="store_true",
            help="启用 Trading 回测（默认禁用，仅对 top-K 评估）",
        )
        parser.add_argument(
            "--top-k-backtest",
            type=int, default=10,
            help="跑回测的 top-K 数量（默认 10）",
        )
        parser.add_argument(
            "--initial-cash",
            type=float, default=1_000_000.0,
            help="初始资金（默认 1,000,000）",
        )
        parser.add_argument(
            "--commission",
            type=float, default=0.001,
            help="手续费率（默认 0.001）",
        )
        # 输出
        parser.add_argument(
            "--output", "-O",
            type=str, default="alpha_pool.json",
            help="结果保存路径（默认 alpha_pool.json）",
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="详细输出",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="安静模式（只输出 final pool）",
        )

    def run(self, args: argparse.Namespace) -> int:
        # 1. 解析 forward_returns
        try:
            forward_returns = [int(x.strip()) for x in args.forward_returns.split(",") if x.strip()]
        except ValueError:
            print(f"❌ --forward-returns 格式错误: {args.forward_returns}", file=sys.stderr)
            return 1

        # 2. 加载数据
        try:
            df = self._load_data(args.data, args.date_column, args.code_column)
        except Exception as e:
            print(f"❌ 数据加载失败: {e}", file=sys.stderr)
            return 1

        if not args.quiet:
            print(f"🎯 Alpha-GPT 自动化因子挖掘")
            print(f"📊 数据：{len(df)} 行 × {len(df.columns)} 列")
            print(f"🧠 LLM：{args.llm}{f' ({args.model})' if args.model else ''}")
            print(f"🔄 {args.iterations} 轮 × {args.pool_size} 候选 = {args.iterations * args.pool_size} 个公式")
            if args.backtest:
                print(f"💹 Trading 回测：启用（top-K={args.top_k_backtest}）")
            else:
                print(f"💹 Trading 回测：禁用")
            print()

        # 3. 配置 + 运行 workflow
        from QuantNodes.research.quant_alpha.workflow import (
            AlphaGptConfig, AlphaGptWorkflow,
        )

        config = AlphaGptConfig(
            objective=args.objective,
            iterations=args.iterations,
            pool_size=args.pool_size,
            top_k=args.top_k,
            min_ir_threshold=args.min_ir,
            max_mutual_ic_threshold=args.max_mutual_ic,
            forward_returns=forward_returns,
            date_column=args.date_column,
            code_column=args.code_column,
            llm_provider=args.llm,
            llm_model=args.model,
            temperature=args.temperature,
            enable_backtest=args.backtest,
            top_k_backtest=args.top_k_backtest,
        )

        llm_client = self._build_llm_client(args) if args.llm != "mock" else None

        workflow = AlphaGptWorkflow(
            config=config,
            data=df,
            llm_client=llm_client,
        )
        result = workflow.run()

        # 4. 输出结果
        if args.quiet:
            print(json.dumps(
                [f.to_dict() for f in result.final_pool],
                ensure_ascii=False, indent=2,
            ))
        else:
            self._print_human_readable(result, verbose=args.verbose)

        # 5. 保存到 JSON
        out_path = Path(args.output)
        out_data = {
            "metadata": {
                "objective": config.objective,
                "iterations": result.iterations_completed,
                "pool_size": config.pool_size,
                "data_rows": len(df),
                "llm_provider": config.llm_provider,
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": result.elapsed_seconds,
            },
            "final_pool": [f.to_dict() for f in result.final_pool],
            "summary": result.summary,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2, default=str)

        if not args.quiet:
            print(f"\n💾 结果已保存：{out_path}")

        return 0

    def _load_data(
        self,
        path: Optional[str],
        date_column: str,
        code_column: str,
    ) -> Any:
        """加载数据：路径 → Parquet/CSV；None → 合成"""
        import polars as pl
        import numpy as np

        if path is None:
            np.random.seed(42)
            dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
            rows = []
            for date in dates:
                for code in ["A", "B", "C", "D", "E"]:
                    close = float(np.random.randn() * 5 + 100)
                    rows.append({
                        date_column: date, code_column: code,
                        "close": close,
                        "open": close + np.random.randn() * 0.5,
                        "high": close + abs(np.random.randn()),
                        "low": close - abs(np.random.randn()),
                        "vol": float(np.random.randint(1000, 5000)),
                    })
            return pl.DataFrame(rows).with_columns(pl.col(date_column).str.to_date())

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        if p.suffix == ".parquet":
            return pl.read_parquet(p)
        if p.suffix == ".csv":
            df = pl.read_csv(p)
            if date_column in df.columns and df[date_column].dtype == pl.Utf8:
                df = df.with_columns(pl.col(date_column).str.to_date())
            return df
        raise ValueError(f"不支持的数据格式: {p.suffix}")

    def _build_llm_client(self, args: argparse.Namespace) -> Any:
        """根据 --llm 参数构造 LLM client

        复用 nanobot Agent 体系。
        若 nanobot 未安装或 API key 缺失，返回 None（用 mock）。
        """
        try:
            from QuantNodes.agent import NANOBOT_AVAILABLE, NanobotNotInstalled
            if not NANOBOT_AVAILABLE:
                print(f"⚠️  nanobot-ai 未安装，使用 mock LLM", file=sys.stderr)
                return None
            from QuantNodes.ai.llm.gateway import LLMGateway
            return LLMGateway(workspace=".agent")
        except NanobotNotInstalled:
            print(f"⚠️  nanobot-ai 未安装，使用 mock LLM", file=sys.stderr)
            return None

    def _print_human_readable(self, result: Any, verbose: bool = False) -> None:
        print()
        print(f"✅ 完成：{result.iterations_completed} 轮, "
              f"{result.total_formulas} 公式, "
              f"{len(result.final_pool)} 个 top-K")
        print(f"⏱️  耗时：{result.elapsed_seconds:.2f}s")
        print(f"📈 best_ir={result.summary.get('best_ir', 0):.3f}, "
              f"avg_ir={result.summary.get('avg_ir', 0):.3f}")
        print()

        if not result.final_pool:
            print("⚠️  final pool 为空（可能所有公式都失败了）")
            return

        print(f"🏆 Top {len(result.final_pool)} 公式：")
        for f in result.final_pool:
            print(f"  rank={f.rank:<3d}  IR={f.ir:>6.3f}  "
                  f"formula={f.formula!r}")
            if verbose and f.selection_reason:
                print(f"           reason: {f.selection_reason[:100]}")


class NanobotLLMWrapper:
    """把 nanobot Agent 包装成 workflow 期望的 client 接口

    现在内部委托 LLMGateway。
    """

    def __init__(self, args: argparse.Namespace):
        from QuantNodes.ai.llm.gateway import LLMGateway
        self._gateway = LLMGateway(workspace=".agent")
        self.temperature = args.temperature
        self.model = args.model

    def complete(self, agent_id: str, prompt: str) -> str:
        """同步调用 LLMGateway.complete()"""
        return self._gateway.complete(agent_id=agent_id, prompt=prompt)


# ==============================================================================
# 端到端流水线命令
# ==============================================================================


class AlphaPipelineCommand(Command):
    """quantnodes alpha-pipeline - 端到端因子挖掘流水线（多轮迭代版）

    用法:
        quantnodes alpha-pipeline --objective "捕捉 A 股反转效应" --data data.parquet
        quantnodes alpha-pipeline --objective "..." --max-rounds 5 --target-factors 10
        quantnodes alpha-pipeline --objective "..." --no-early-stopping --patience 3

    详见 docs/quant_alpha/pipeline_multi_round_design.md
    """
    name = "alpha-pipeline"
    description = "端到端因子挖掘流水线（Alpha-GPT → MCTS → 去重 → Wiki，支持多轮迭代）"

    def add_arguments(self, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            self.name,
            help=self.description,
            description=(
                "端到端因子挖掘流水线（多轮迭代版）：\n"
                "Round 1: Alpha-GPT → MCTS → 去重 → 反馈\n"
                "Round 2: Alpha-GPT(←反馈) → MCTS → 去重 → 反馈\n"
                "...\n"
                "Round N: 最终结果 → Wiki 持久化\n"
                "详见 docs/quant_alpha/pipeline_multi_round_design.md"
            ),
        )
        # 必选
        parser.add_argument(
            "--objective", "-o",
            type=str, required=True,
            help="研究目标（如 '捕捉 A 股反转效应'）",
        )
        parser.add_argument(
            "--data", "-d",
            type=str, default=None,
            help="行情数据路径（Parquet/CSV），None=合成测试数据",
        )
        # Wiki
        parser.add_argument(
            "--wiki-path",
            type=str, default="wiki/",
            help="Wiki 因子库路径（默认 wiki/）",
        )
        # 终止条件配置（新增）
        parser.add_argument(
            "--max-rounds",
            type=int, default=5,
            help="最大迭代轮次（默认 5）",
        )
        parser.add_argument(
            "--target-factors",
            type=int, default=10,
            help="目标因子数量（默认 10）",
        )
        parser.add_argument(
            "--min-improvement",
            type=float, default=0.01,
            help="最小 IR 提升阈值（默认 0.01）",
        )
        parser.add_argument(
            "--no-early-stopping",
            action="store_true",
            help="禁用早停机制",
        )
        parser.add_argument(
            "--patience",
            type=int, default=3,
            help="早停耐心值：连续 N 轮无改善则停止（默认 3）",
        )
        parser.add_argument(
            "--timeout",
            type=int, default=3600,
            help="总超时时间（秒，默认 3600）",
        )
        parser.add_argument(
            "--round-timeout",
            type=int, default=600,
            help="单轮超时时间（秒，默认 600）",
        )
        # Alpha-GPT 配置
        parser.add_argument(
            "--alphagpt-iterations",
            type=int, default=3,
            help="Alpha-GPT 迭代轮次（默认 3）",
        )
        parser.add_argument(
            "--alphagpt-pool-size",
            type=int, default=10,
            help="Alpha-GPT 每轮想法数量（默认 10）",
        )
        # MCTS 配置
        parser.add_argument(
            "--mcts-iterations",
            type=int, default=50,
            help="MCTS 迭代次数（默认 50）",
        )
        parser.add_argument(
            "--mcts-max-depth",
            type=int, default=5,
            help="MCTS 最大深度（默认 5）",
        )
        # 去重配置
        parser.add_argument(
            "--max-mutual-ic",
            type=float, default=0.7,
            help="最大 mutual IC 阈值（默认 0.7）",
        )
        # Alpha-GPT 过滤配置
        parser.add_argument(
            "--min-ir-threshold",
            type=float, default=0.1,
            help="最小 IR 阈值（默认 0.1）",
        )
        # 通用配置
        parser.add_argument(
            "--top-k", "-k",
            type=int, default=10,
            help="最终返回的 top-K 公式数量（默认 10）",
        )
        parser.add_argument(
            "--date-column",
            type=str, default="date",
            help="日期列名（默认 date）",
        )
        parser.add_argument(
            "--code-column",
            type=str, default="code",
            help="股票代码列名（默认 code）",
        )
        parser.add_argument(
            "--forward-returns",
            type=str, default="1,5,20",
            help="前瞻期列表（逗号分隔，默认 '1,5,20'）",
        )
        # LLM 配置
        parser.add_argument(
            "--llm",
            type=str, default="minimax",
            choices=["minimax", "deepseek", "openai", "qwen", "azure", "mock"],
            help="LLM provider（默认 minimax）",
        )
        parser.add_argument(
            "--model",
            type=str, default=None,
            help="模型名（如 'minimax-M3' / 'gpt-4o'）",
        )
        parser.add_argument(
            "--temperature",
            type=float, default=0.7,
            help="采样温度（默认 0.7）",
        )
        parser.add_argument(
            "--temperature-idea-gen",
            type=float, default=0.8,
            help="idea-generator 温度（默认 0.8）",
        )
        parser.add_argument(
            "--temperature-formula",
            type=float, default=0.4,
            help="formula-translator 温度（默认 0.4）",
        )
        parser.add_argument(
            "--temperature-reflector",
            type=float, default=0.6,
            help="reflector 温度（默认 0.6）",
        )
        parser.add_argument(
            "--temperature-critic",
            type=float, default=0.3,
            help="critic 温度（默认 0.3）",
        )
        # 输出
        parser.add_argument(
            "--output", "-O",
            type=str, default="pipeline_result.json",
            help="结果保存路径（默认 pipeline_result.json）",
        )
        parser.add_argument(
            "--output-dir",
            type=str, default="pipeline_output",
            help="详细结果输出目录（默认 pipeline_output）",
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="详细输出",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="安静模式（只输出摘要）",
        )

    def run(self, args: argparse.Namespace) -> int:
        from QuantNodes.research.quant_alpha.pipeline import (
            AlphaPipeline, PipelineConfig, TerminationConfig,
        )

        # 1. 解析 forward_returns
        try:
            forward_returns = tuple(
                int(x.strip()) for x in args.forward_returns.split(",") if x.strip()
            )
        except ValueError:
            print(f"❌ --forward-returns 格式错误: {args.forward_returns}", file=sys.stderr)
            return 1

        # 2. 加载数据
        data = self._load_data(args.data, args.date_column, args.code_column)
        if data is None:
            return 1

        if not args.quiet:
            print(f"🔬 端到端因子挖掘流水线（多轮迭代版）")
            print(f"📊 数据：{len(data)} 行 × {len(data.columns)} 列")
            print(f"🎯 目标：{args.objective}")
            print(f"🧠 LLM：{args.llm}{f' ({args.model})' if args.model else ''}")
            print(f"🔄 最大轮次: {args.max_rounds}, 目标因子: {args.target_factors}")
            print(f"🔄 Alpha-GPT: {args.alphagpt_iterations} 轮 × {args.alphagpt_pool_size} 候选")
            print(f"🔄 MCTS: {args.mcts_iterations} 次迭代")
            print(f"⏹️  早停: {'禁用' if args.no_early_stopping else f'启用 (patience={args.patience})'}")
            print(f"📚 Wiki：{args.wiki_path}")
            print()

        # 3. 配置 + 运行流水线
        termination = TerminationConfig(
            max_rounds=args.max_rounds,
            target_factors=args.target_factors,
            min_improvement=args.min_improvement,
            early_stopping=not args.no_early_stopping,
            patience=args.patience,
            timeout_seconds=args.timeout,
            round_timeout_seconds=args.round_timeout,
        )

        config = PipelineConfig(
            objective=args.objective,
            wiki_path=args.wiki_path,
            termination=termination,
            alphagpt_iterations=args.alphagpt_iterations,
            alphagpt_pool_size=args.alphagpt_pool_size,
            mcts_iterations=args.mcts_iterations,
            mcts_max_depth=args.mcts_max_depth,
            max_mutual_ic=args.max_mutual_ic,
            min_ir_threshold=args.min_ir_threshold,
            top_k=args.top_k,
            date_column=args.date_column,
            code_column=args.code_column,
            forward_returns=forward_returns,
            llm_provider=args.llm,
            llm_model=args.model,
            temperature=args.temperature,
            temperature_idea_gen=args.temperature_idea_gen,
            temperature_formula=args.temperature_formula,
            temperature_reflector=args.temperature_reflector,
            temperature_critic=args.temperature_critic,
            output_dir=args.output_dir,
        )

        pipeline = AlphaPipeline(config)
        result = pipeline.run(data)

        # 4. 保存结果
        self._save_result(result, args.output, args.verbose)

        # 5. 打印结果
        if not args.quiet:
            self._print_human_readable(result, args.verbose)

        return 0

    def _load_data(
        self, data_path: Optional[str], date_column: str, code_column: str
    ) -> Optional[Any]:
        """加载数据"""
        import polars as pl

        if data_path is None:
            print("⚠️  未指定数据路径，使用合成测试数据", file=sys.stderr)
            return self._generate_synthetic_data()

        path = Path(data_path)
        if not path.exists():
            print(f"❌ 数据文件不存在: {data_path}", file=sys.stderr)
            return None

        try:
            if path.suffix == ".parquet":
                df = pl.read_parquet(path)
            elif path.suffix == ".csv":
                df = pl.read_csv(path)
            elif path.suffix == ".json":
                df = pl.read_json(path)
            else:
                print(f"❌ 不支持的文件格式: {path.suffix}", file=sys.stderr)
                return None

            # 确保日期列存在
            if date_column not in df.columns:
                print(f"❌ 缺少日期列: {date_column}", file=sys.stderr)
                return None

            # 确保代码列存在
            if code_column not in df.columns:
                print(f"❌ 缺少代码列: {code_column}", file=sys.stderr)
                return None

            return df

        except Exception as e:
            print(f"❌ 数据加载失败: {e}", file=sys.stderr)
            return None

    def _generate_synthetic_data(self) -> Any:
        """生成合成测试数据"""
        import numpy as np
        import polars as pl

        np.random.seed(42)
        dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
        rows = []
        for date in dates:
            for code in ["A", "B", "C", "D", "E"]:
                close = float(np.random.randn() * 5 + 100)
                rows.append({
                    "date": date,
                    "code": code,
                    "close": close,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "vol": 1000.0,
                })
        return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())

    def _save_result(self, result: Any, output_path: str, verbose: bool) -> None:
        """保存结果"""
        try:
            output = {
                "summary": result.summary,
                "rounds": [
                    {
                        "round_num": r.round_num,
                        "elapsed_seconds": r.elapsed_seconds,
                        "feedback": r.feedback.to_dict() if r.feedback else None,
                        "final_factors": len(r.final_pool),
                    }
                    for r in result.rounds
                ],
                "final_pool": [
                    {
                        "formula_id": m.formula_id,
                        "ic_mean": m.ic_mean,
                        "ic_std": m.ic_std,
                        "ir": m.ir,
                        "rank_ic_mean": m.rank_ic_mean,
                        "overall_score": m.overall_score,
                    }
                    for m in result.final_pool
                ],
                "wiki_pages": result.wiki_pages,
                "elapsed_seconds": result.elapsed_seconds,
            }

            if verbose:
                output["alphagpt_summary"] = (
                    result.rounds[-1].alphagpt_result.summary
                    if result.rounds and result.rounds[-1].alphagpt_result
                    else None
                )
                output["mcts_stats"] = (
                    {
                        "formula_count": result.rounds[-1].mcts_result.formula_count,
                        "valid_count": result.rounds[-1].mcts_result.valid_count,
                        "rejected_count": result.rounds[-1].mcts_result.rejected_count,
                    }
                    if result.rounds and result.rounds[-1].mcts_result
                    else None
                )

            Path(output_path).write_text(
                json.dumps(output, ensure_ascii=False, indent=2)
            )
            print(f"💾 结果已保存到: {output_path}")

        except Exception as e:
            print(f"⚠️  结果保存失败: {e}", file=sys.stderr)

    def _print_human_readable(self, result: Any, verbose: bool = False) -> None:
        print()
        print(f"✅ 流水线完成（多轮迭代）")
        print(f"⏱️  耗时：{result.elapsed_seconds:.1f}s")
        print(f"📊 摘要：")
        print(f"  - 总轮次: {result.summary.get('total_rounds', 0)}")
        print(f"  - 最终因子: {result.summary.get('final_factors', 0)}")
        print(f"  - Wiki 页面: {result.summary.get('wiki_pages', 0)}")
        print(f"  - best_ir: {result.summary.get('best_ir', 0):.3f}")
        print(f"  - avg_ir: {result.summary.get('avg_ir', 0):.3f}")
        print()

        # 显示每轮统计
        if result.rounds:
            print(f"🔄 各轮统计：")
            for r in result.rounds:
                feedback = r.feedback
                if feedback:
                    print(f"  Round {r.round_num}: "
                          f"IR={feedback.best_ir:.3f}, "
                          f"有效因子={feedback.valid_count}, "
                          f"耗时={r.elapsed_seconds:.1f}s")
            print()

        if not result.final_pool:
            print("⚠️  final pool 为空（可能所有公式都失败了）")
            return

        print(f"🏆 Top {len(result.final_pool)} 因子：")
        for m in result.final_pool:
            print(f"  IR={m.ir:>6.3f}  IC={m.ic_mean:>6.4f}  formula_id={m.formula_id}")
            if verbose:
                print(f"           rank_ic={m.rank_ic_mean:.4f}  "
                      f"stability={m.stability_score:.4f}  "
                      f"overall={m.overall_score:.4f}")
