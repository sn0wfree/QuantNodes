"""CLI for quantnodes-strategy-audit.

Commands:
  - scan: run Engine A static scan
  - precheck: run Engine A precheck for specific lessons
  - lesson: load a single lesson
  - lessons: list lessons
  - search: search lessons by keyword
  - validate cv: run CV% test
  - validate bootstrap: run bootstrap stability
  - gates: run 5-gates check
  - serve-mcp: start MCP server (stdio)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from quantnodes_strategy_audit.core.lesson import LessonLoader
from quantnodes_strategy_audit.core.report import Report
from quantnodes_strategy_audit.engines.static_engine import StaticEngine


def _find_pkg_paths() -> tuple[Path, Path]:
    """Find lessons/ and rules/ relative to package."""
    pkg_root = Path(__file__).parent.parent.parent
    lessons = pkg_root / "lessons"
    rules = pkg_root / "rules" / "simple_rules.yaml"
    return lessons, rules


def _make_engines():
    """Create LessonLoader and StaticEngine."""
    lessons_dir, rules_path = _find_pkg_paths()
    loader = LessonLoader(builtin_dir=lessons_dir)
    static = StaticEngine(rules_path=rules_path)
    return loader, static


@click.group()
@click.version_option()
def cli() -> None:
    """quantnodes-strategy-audit: 量化策略审计工具."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="CRITICAL > 0 则 fail (exit code 1)")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "sarif"]),
    default="text",
)
@click.option("--category", multiple=True)
@click.option("--severity", multiple=True, type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]))
@click.option("--lesson", multiple=True, help="按 L-NNN 过滤")
def scan(
    path: str,
    strict: bool,
    output: str | None,
    fmt: str,
    category: tuple[str, ...],
    severity: tuple[str, ...],
    lesson: tuple[str, ...],
) -> None:
    """扫描代码库, Engine A 静态检测."""
    target = Path(path).resolve()
    _, static = _make_engines()

    if target.is_file():
        warnings = static.scan_file(
            target,
            lesson_ids=list(lesson) if lesson else None,
            categories=list(category) if category else None,
            severities=list(severity) if severity else None,
        )
    else:
        warnings = static.scan_directory(
            target,
            lesson_ids=list(lesson) if lesson else None,
            categories=list(category) if category else None,
            severities=list(severity) if severity else None,
        )

    report = Report(warnings=warnings)
    _output_report(report, output, fmt)

    if strict and report.has_critical():
        click.echo(f"\n[STRICT FAIL] {report.summary}", err=True)
        sys.exit(1)
    sys.exit(0)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--lesson", multiple=True, required=True, help="L-NNN 列表")
@click.option("--output", "-o", type=click.Path())
def precheck(path: str, lesson: tuple[str, ...], output: str | None) -> None:
    """Engine A 预检 (按指定 lesson_ids)."""
    target = Path(path).resolve()
    _, static = _make_engines()

    if target.is_file():
        warnings = static.scan_file(target, lesson_ids=list(lesson))
    else:
        warnings = static.scan_directory(target, lesson_ids=list(lesson))

    report = Report(warnings=warnings)
    _output_report(report, output, "json")


@cli.command()
@click.argument("lesson_id")
def lesson(lesson_id: str) -> None:
    """加载单个教训 (markdown + check_prompt)."""
    loader, _ = _make_engines()
    less = loader.get(lesson_id)
    if less is None:
        click.echo(f"Lesson '{lesson_id}' not found", err=True)
        sys.exit(1)
    click.echo(f"# {less.id}: {less.title}\n")
    click.echo(f"**Severity**: {less.severity}")
    click.echo(f"**Category**: {less.category}")
    click.echo(f"**Auto-checkable**: {less.auto_checkable}\n")
    click.echo(f"## 一句话总结\n{less.one_sentence}\n")
    click.echo(f"## Check Prompt\n{less.check_prompt}\n")
    click.echo("---")
    click.echo(less.content_markdown)


@cli.command(name="lessons")
@click.option("--category")
@click.option("--severity", type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]))
@click.option("--auto-checkable", type=click.Choice(["static", "agent", "partial", "manual"]))
def list_lessons_cmd(
    category: str | None, severity: str | None, auto_checkable: str | None
) -> None:
    """列出所有教训."""
    loader, _ = _make_engines()
    lessons = loader.list_lessons(
        category=category, severity=severity, auto_checkable=auto_checkable
    )
    click.echo(f"Found {len(lessons)} lesson(s):\n")
    for lesson in lessons:
        click.echo(f"  [{lesson.severity}] {lesson.id}: {lesson.title}")
        click.echo(
            f"    Category: {lesson.category}, "
            f"auto_checkable: {lesson.auto_checkable}"
        )
        if lesson.one_sentence:
            click.echo(f"    {lesson.one_sentence[:120]}")
        click.echo("")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5)
def search(query: str, top_k: int) -> None:
    """搜索教训."""
    loader, _ = _make_engines()
    results = loader.search(query, top_k=top_k)
    click.echo(f"Query: {query}\n")
    for lesson, score in results:
        click.echo(f"  [{score:.2f}] {lesson.id}: {lesson.title}")


@cli.command()
@click.option("--strategy", required=True, help="回测函数路径 'module.path:callable'")
@click.option("--start-dates", multiple=True, required=True)
def validate_cv(strategy: str, start_dates: tuple[str, ...]) -> None:
    """CV% 起点依赖测试."""
    from quantnodes_strategy_audit.validators.cv_calculator import CVCalculator

    backtest_fn = _import_callable(strategy)
    calc = CVCalculator()
    result = calc.run(backtest_fn=backtest_fn, start_dates=list(start_dates))
    click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@cli.command()
@click.option("--strategy", required=True)
@click.option("--n-bootstrap", default=30)
@click.option("--block-size", default=63)
def validate_bootstrap(strategy: str, n_bootstrap: int, block_size: int) -> None:
    """块自助法稳定性测试."""
    from quantnodes_strategy_audit.validators.bootstrap_stability import BootstrapStability

    backtest_fn = _import_callable(strategy)
    validator = BootstrapStability(n_bootstrap=n_bootstrap, block_size=block_size)
    result = validator.run(backtest_fn=backtest_fn)
    click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@cli.command()
def gates() -> None:
    """5 道闸门集成检查 (需要自定义 check functions)."""
    from quantnodes_strategy_audit.validators.five_gates import FiveGates

    validator = FiveGates()
    result = validator.run()
    click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@cli.command(name="serve-mcp")
def serve_mcp() -> None:
    """以 MCP server 模式运行 (stdio transport), 供 agent 直接调用."""
    from quantnodes_strategy_audit.mcp_server import main as mcp_main

    mcp_main()


def _output_report(report: Report, output: str | None, fmt: str) -> None:
    if output:
        report.write(Path(output), fmt=fmt)
        click.echo(f"Report written to {output}", err=True)
        return
    if fmt == "json":
        click.echo(report.to_json())
    elif fmt == "sarif":
        click.echo(json.dumps(report.to_sarif(), ensure_ascii=False, indent=2))
    else:
        click.echo(report.to_text())


def _import_callable(module_str: str):
    import importlib.util

    if ":" not in module_str:
        raise click.BadParameter(f"Invalid '{module_str}', expected 'module.path:callable'")
    module_path, callable_name = module_str.rsplit(":", 1)

    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    spec = importlib.util.find_spec(module_path)
    if spec is None:
        raise click.BadParameter(f"Module not found: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, callable_name)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
