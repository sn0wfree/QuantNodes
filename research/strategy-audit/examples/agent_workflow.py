"""Example: Agent workflow using quantnodes-strategy-audit MCP tools.

This shows the complete audit flow as if from an Agent's perspective.
In production, the Agent would use the actual MCP SDK; this is a
demonstration using the Python API directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from quantnodes_strategy_audit import ContextEngine, LessonLoader, StaticEngine


def main():
    """Demonstrate the full audit workflow."""
    # Setup engines (typically done once in the MCP server)
    pkg_root = Path(__file__).resolve().parent.parent
    loader = LessonLoader(builtin_dir=pkg_root / "lessons")
    static = StaticEngine(rules_path=pkg_root / "rules" / "simple_rules.yaml")
    context_engine = ContextEngine(
        lesson_loader=loader, static_engine=static
    )

    # The file we want to audit
    target_file = Path("strategy/v7/data_loader.py")

    # ===== Step 1: List relevant lessons =====
    print("=" * 60)
    print("Step 1: List relevant lessons (lookahead / CRITICAL)")
    print("=" * 60)
    lessons_result = context_engine.list_lessons(
        category="lookahead", severity="CRITICAL"
    )
    for lesson in lessons_result["lessons"][:5]:
        print(f"  - {lesson['id']}: {lesson['title']}")
    print()

    # ===== Step 2: Engine A precheck =====
    print("=" * 60)
    print("Step 2: Engine A static precheck")
    print("=" * 60)
    if target_file.exists():
        precheck = context_engine.static_precheck(file=str(target_file))
        print(f"  Total warnings: {precheck['total_warnings']}")
        for lesson_id, evidence in precheck["by_lesson"].items():
            print(f"  Lesson {lesson_id}: {len(evidence)} violations")
            for ev in evidence[:2]:
                print(f"    Line {ev['line']}: {ev['snippet'][:60]}")
    else:
        print(f"  (file {target_file} not found, skipping)")
    print()

    # ===== Step 3: Load full lesson =====
    print("=" * 60)
    print("Step 3: Load full L-202 lesson")
    print("=" * 60)
    lesson = context_engine.get_lesson("L-202")
    print(f"  Title: {lesson['title']}")
    print(f"  Severity: {lesson['severity']}")
    print(f"  One sentence: {lesson['one_sentence'][:80]}")
    print(f"  Check prompt (first 200 chars):")
    print(f"  {lesson['check_prompt'][:200]}...")
    print()

    # ===== Step 4: Get code context =====
    if target_file.exists():
        print("=" * 60)
        print("Step 4: AST code context around focus lines")
        print("=" * 60)
        ctx = context_engine.get_code_context(
            file=str(target_file), focus_lines=[142, 143], depth=2
        )
        print(f"  Imports: {ctx['imports'][:3]}")
        if ctx["enclosing_function"]:
            print(f"  Enclosing function: {ctx['enclosing_function']['name']}")
            print(f"    Lines: {ctx['enclosing_function']['lines']}")
            print(f"    Docstring: {ctx['enclosing_function'].get('docstring', '')[:60]}")
        print()

    # ===== Step 5: Agent submits finding (simulated) =====
    print("=" * 60)
    print("Step 5: Agent submits finding (simulated)")
    print("=" * 60)
    result = context_engine.submit_finding({
        "file": str(target_file),
        "line": 142,
        "lesson_id": "L-202",
        "status": "VIOLATED",
        "severity": "CRITICAL",
        "evidence": {
            "snippet": "mean = X.mean()",
            "explanation": "In standardize function, full-sample mean used",
        },
        "fix_suggestion": "Use X.rolling(252).mean() or X.expanding().mean()",
        "confidence": 0.95,
    })
    print(f"  Finding ID: {result['finding_id']}")
    print(f"  Stored: {result['stored']}")
    print()

    # ===== Step 6: Summary =====
    print("=" * 60)
    print("Step 6: Summary")
    print("=" * 60)
    findings = context_engine.get_findings()
    print(f"  Total findings: {len(findings)}")
    for f in findings:
        print(f"  - {f.lesson_id} @ {f.file}:{f.line} [{f.status}, conf={f.confidence:.2f}]")


if __name__ == "__main__":
    main()