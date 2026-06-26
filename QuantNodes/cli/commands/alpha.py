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
