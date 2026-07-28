# quantnodes-strategy-audit

> 量化策略审计工具 — 双引擎 + 教训驱动 + Skill 化
> 源自 QuantNodes 项目 17 天研发周期的教训总结（48 条 L-NNN 教训 / 66 daily 教训）

## 架构

```
quantnodes-strategy-audit (Skill)
├─ Engine A: 静态规则引擎 (YAML 驱动, 快速, 0 token)
│   └─ rules/simple_rules.yaml (17 条规则)
├─ Engine B: 上下文提供器 (Agent 调用, 不调 LLM)
│   ├─ audit_get_lesson: 加载教训 markdown
│   ├─ audit_list_lessons: 列出教训
│   ├─ audit_get_code_context: AST 上下文
│   ├─ audit_search_lessons: 关键词搜索
│   ├─ audit_static_precheck: Engine A 预检
│   └─ audit_submit_finding: Agent 提交 finding
└─ Validators (运行时验证)
    ├─ CVCalculator: 起点依赖 CV% (L-203)
    ├─ BootstrapStability: 块自助法 (L-215)
    └─ FiveGates: 5 道闸门集成 (L-321)
```

## 安装

```bash
pip install quantnodes-strategy-audit

# 或开发模式
git clone https://github.com/sn0wfree/quantnodes-strategy-audit.git
cd quantnodes-strategy-audit
pip install -e ".[dev,mcp]"
```

## 快速上手

### CLI 模式 (人类 / CI)

```bash
# Engine A 静态扫描
quantnodes-audit scan /path/to/code --strict --output report.json

# 按 lesson 预检 (Engine A)
quantnodes-audit precheck /path/to/code --lesson L-202 --lesson L-213

# 查看教训
quantnodes-audit lesson L-202
quantnodes-audit lessons --severity CRITICAL
quantnodes-audit search "look-ahead"

# 运行时验证
quantnodes-audit validate cv --strategy v7_10:run --start-dates 2018-01-01 2020-01-01
quantnodes-audit validate bootstrap --strategy v7_10:run

# 启动 MCP server
quantnodes-audit serve-mcp
```

### Python API

```python
from quantnodes_strategy_audit import (
    LessonLoader, StaticEngine, ContextEngine,
    CVCalculator, Severity, Warning,
)

# 加载教训
loader = LessonLoader(builtin_dir="lessons")
lesson = loader.get("L-202")
print(lesson.check_prompt)

# Engine A 静态扫描
static = StaticEngine("rules/simple_rules.yaml")
warnings = static.scan_file(Path("strategy/v7/data_loader.py"))
for w in warnings:
    print(f"[{w.severity.value}] {w.detector} @ line {w.line}")

# Engine B 上下文提供器
from quantnodes_strategy_audit.core.code_context import CodeContextExtractor
extractor = CodeContextExtractor()
ctx = extractor.extract(Path("strategy/v7/data_loader.py"), focus_lines=[142], depth=2)
```

### MCP / Skill 模式 (Agent 调用)

```python
# Agent (LLM) 通过 MCP 协议调用工具
async with mcp_session() as session:
    # 1. 列出相关教训
    lessons = await session.call("audit_list_lessons", {"category": "lookahead"})

    # 2. Engine A 预检可疑位置
    precheck = await session.call("audit_static_precheck", {
        "file": "strategy/v7/data_loader.py",
        "lesson_ids": ["L-202", "L-213"],
    })

    # 3. 加载完整教训
    lesson = await session.call("audit_get_lesson", {"lesson_id": "L-202"})

    # 4. 获取代码上下文
    ctx = await session.call("audit_get_code_context", {
        "file": "strategy/v7/data_loader.py",
        "focus_lines": [142],
        "depth": 3,
    })

    # 5. Agent 自己判断 → 提交 finding
    await session.call("audit_submit_finding", {
        "file": "strategy/v7/data_loader.py",
        "line": 142,
        "lesson_id": "L-202",
        "status": "VIOLATED",
        "severity": "CRITICAL",
        "evidence": {"snippet": "mean = X.mean()"},
        "fix_suggestion": "use .rolling(252).mean()",
        "confidence": 0.95,
    })
```

## 48 教训清单

详见 `lessons/` 目录，按 ID + slug 组织：

| 类别 | 数量 | 列表 |
|---|---|---|
| methodology | 18 | L-101~L-134 |
| lookahead | 5 | L-201~L-205, L-223 |
| oos_validation | 4 | L-203, L-215, L-233, L-322 |
| data_quality | 5 | L-211, L-212, L-214, L-222 |
| nan_safe | 1 | L-213 |
| frequency | 1 | L-221 |
| engineering | 7 | L-231, L-232, L-241~L-243, L-323 |
| decision | 4 | L-301~L-306 |
| integration | 3 | L-321 |

## 17 条 Engine A 规则

详见 `rules/simple_rules.yaml`，按类别：

- **lookahead (8)**: shift_negative, full_sample_mean, standardscaler_default, minmaxscaler_default, train_test_split_leakage, shuffle_split_true, data_shift_negative_index, full_dataset_iteration, full_sample_method
- **nan_safe (4)**: bare_pct_change, fillna_zero, dropna_returns, interpolate_returns
- **standardize (2)**: full_sample_zscore_inline, full_sample_zscore_var
- **oos (1)**: fit_transform_on_full
- **evaluation (1)**: in_sample_eval

## 与 QuantNodes 主项目集成

```yaml
# QuantNodes/.gitmodules
[submodule "research/strategy-audit"]
    path = research/strategy-audit
    url = https://github.com/sn0wfree/quantnodes-strategy-audit.git
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# Lint
ruff check src/ tests/

# 类型检查
mypy src/
```

## 许可证

MIT

## 致谢

源自 QuantNodes 项目的 17 天教训梳理（2026-07-07 → 2026-07-28, 217 commit, 66 教训）。