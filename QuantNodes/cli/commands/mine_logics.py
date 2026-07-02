# coding=utf-8
"""
CLI 命令: mine-logics — 并发批量逻辑挖掘 (v3.0.2)

Usage:
    quantnodes mine-logics --max-per-lib 5 --workers 2 --live
    quantnodes mine-logics --source-libs alpha101,alpha191 --max-per-lib 10
    quantnodes mine-logics --wiki-path wiki_auto --output-dir data/mine_runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from QuantNodes.cli.command import Command
from QuantNodes.research.quant_alpha.logic_mining.batch import (
    ThreadSafeMetrics,
    mine_logic_library_v2,
)
from QuantNodes.research.quant_alpha.logic_mining.report import MetricsReportBuilder


class MineLogicsCommand(Command):
    """quantnodes mine-logics - 并发批量逻辑挖掘

    并发挖掘 alpha101/alpha158/alpha191 的逻辑结构，
    生成 Logic pages 到 Wiki，输出 JSON + Markdown 报告。

    默认使用 NullLLMClient（离线模式）；加 --live 走真实 LLM。
    """
    name = "mine-logics"
    description = "并发批量逻辑挖掘（v3.0.2）"

    def add_arguments(self, subparsers: object) -> None:
        parser: argparse.ArgumentParser = subparsers.add_parser(
            self.name,
            help=self.description,
            description=(
                "基于 LogicMiningPipeline 三段式 Agent 的并发批量挖掘。\n"
                "支持幂等重跑（跳过已存在 Logic pages）。\n"
                "默认离线模式（NullLLMClient）；加 --live 走真实 LLM。"
            ),
        )
        parser.add_argument(
            "--source-libs",
            type=str,
            default="alpha101,alpha158,alpha191",
            help="逗号分隔的来源库列表（默认 alpha101,alpha158,alpha191）",
        )
        parser.add_argument(
            "--max-per-lib",
            type=int,
            default=10,
            help="每个来源库最多挖掘多少条公式（默认 10）",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="并发线程数（默认 4）",
        )
        parser.add_argument(
            "--wiki-path",
            type=str,
            default="wiki_auto",
            help="Wiki 根目录（默认 wiki_auto）",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="data/mine_runs",
            help="报告输出目录（默认 data/mine_runs）",
        )
        parser.add_argument(
            "--live",
            action="store_true",
            help="使用真实 LLM（默认离线 NullLLMClient）",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="启用严格模式（LLM/parse/structured 异常上抛）",
        )
        parser.add_argument(
            "--no-skip",
            action="store_true",
            help="不跳过已存在的 Logic pages（默认幂等跳过）",
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="安静模式（不打印进度）",
        )

    def run(self, args: argparse.Namespace) -> int:
        source_libs = [s.strip() for s in args.source_libs.split(",") if s.strip()]
        if not source_libs:
            print("Error: --source-libs is empty", file=sys.stderr)
            return 2

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 选择 LLM 客户端
        llm_client = None
        if args.live:
            try:
                from QuantNodes.ai.llm.gateway import get_llm_gateway
                llm_client = get_llm_gateway()
                print(f"[mine-logics] Live mode: LLM gateway initialized")
            except Exception as exc:
                print(f"Error: Failed to initialize LLM gateway: {exc}", file=sys.stderr)
                return 2
        else:
            print("[mine-logics] Offline mode: using NullLLMClient")

        # 构建 metrics + strict
        metrics = ThreadSafeMetrics()
        strict = None
        if args.strict:
            from QuantNodes.research.quant_alpha.logic_mining.metrics import StrictConfig
            strict = StrictConfig(call=True, parse=True, structured=True)

        # 进度回调
        def _on_progress(done: int, total: int, fid: str) -> None:
            if not args.quiet:
                print(f"  [{done}/{total}] {fid}", flush=True)

        print(f"[mine-logics] Source libs: {source_libs}")
        print(f"[mine-logics] max_per_lib={args.max_per_lib}, workers={args.workers}")
        print(f"[mine-logics] wiki_path={args.wiki_path}")
        print()

        t0 = time.perf_counter()
        try:
            batch = mine_logic_library_v2(
                source_libs=source_libs,
                llm_client=llm_client,
                max_per_lib=args.max_per_lib,
                workers=args.workers,
                metrics=metrics,
                strict=strict,
                wiki_path=args.wiki_path,
                skip_existing=not args.no_skip,
                on_progress=_on_progress,
            )
        except Exception as exc:
            print(f"Fatal error: {exc}", file=sys.stderr)
            return 2

        # 生成报告
        report = MetricsReportBuilder.from_batch(batch)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"metrics_{ts}.json"
        md_path = output_dir / f"metrics_{ts}.md"
        report.to_json(json_path)
        md_text = report.to_markdown()
        md_path.write_text(md_text, encoding="utf-8")

        # 打印摘要
        print()
        print(f"[mine-logics] Done in {batch.wall_clock_s:.2f}s")
        print(f"  Attempted: {len(batch.attempted_ids)}")
        print(f"  Mined:     {batch.n_mined}")
        print(f"  Skipped:   {batch.n_skipped}")
        print(f"  Failed:    {batch.n_failed}")
        if batch.warnings:
            for w in batch.warnings:
                print(f"  Warning:   {w}")
        print(f"  Reports:   {json_path} / {md_path}")

        # Wiki 写入
        if batch.pool and batch.n_mined > 0:
            try:
                from QuantNodes.research.wiki import WikiFactorProxy
                proxy = WikiFactorProxy(wiki_path=args.wiki_path)
                n_written = batch.pool.to_wiki(proxy)
                print(f"  Wiki:      {n_written} Logic pages written to {args.wiki_path}/")
                failed = batch.pool.failed_writes()
                if failed:
                    print(f"  Wiki failures: {len(failed)}")
            except Exception as exc:
                print(f"  Wiki write failed: {exc}", file=sys.stderr)

        # 退出码
        if batch.n_failed > 0:
            return 1  # 部分成功
        if batch.n_mined == 0 and batch.n_skipped == 0:
            return 2  # 空结果
        return 0  # 全部成功