# coding=utf-8
"""
CLI 命令: alpha-mcts 等 QuantAlpha 子包相关命令

M2 PR 新增：
- `quantnodes alpha-mcts` —— MCTS 因子搜索

后续 M3-M6 PR 会陆续添加：
- `quantnodes alpha-101` (M3)
- `quantnodes alpha-158` (M3)
- `quantnodes alpha-360` (M3)
- `quantnodes alpha-gpt` (M5+)
- `quantnodes alpha-compare` (M7)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
# - AlphaGptCommand (M5+)
# - AlphaCompareCommand (M7)
