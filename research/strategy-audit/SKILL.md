---
name: quantnodes-strategy-audit
description: "量化策略审计工具 — 双引擎 + 教训驱动. Engine A: 静态规则检测 (look-ahead / NaN-safe / 全样本标准化). Engine B: 上下文提供器 (Agent 调用, 不调 LLM). Use when user asks to audit a quantitative strategy, check for look-ahead bias, validate OOS results, or check NaN-safe computation. Triggers: '/audit', 'audit strategy', 'check look-ahead', 'CV% test', 'validate backtest', 'review strategy'."
trigger: /audit
---

# /audit — 量化策略审计工具 (Skill 化)

源自 QuantNodes 项目 17 天研发周期的教训总结（48 条 L-NNN 教训 / 17 条 Engine A 规则）。

## When to use this skill

Use this skill when:
- 用户要求审计量化策略代码（audit strategy / check for bugs）
- 用户提到 look-ahead bias / 未来函数 / 数据穿越
- 用户要求验证 OOS 结果（CV% / 起点依赖测试）
- 用户提到 NaN-safe / pct_change 安全计算
- 用户要求做 bootstrap 稳定性测试
- 用户提到 5 道闸门（5 gates / data/IC/factor/OOS/hardening gates）

Do NOT use when:
- 用户只是要 review 代码风格 → 用 ruff/mypy
- 用户要性能分析 → 用 cProfile/py-spy
- 用户要通用代码搜索 → 用 grep/ripgrep

## Architecture: 双引擎 + 教训驱动

```
quantnodes-strategy-audit (Skill)
├─ Engine A (StaticEngine, YAML 驱动)
│   └─ 17 条规则: .shift(-N) / .mean() / StandardScaler / .pct_change() ...
│
├─ Engine B (ContextEngine, 被动)
│   └─ 6 个 MCP tools: get_lesson / list_lessons / get_code_context / ...
│   └─ Agent 调我, 我不调 LLM
│
└─ Validators (CV% / Bootstrap / 5Gates)
```

## Usage

### Agent invocation (MCP tools via stdio)

When called as a skill, Agent calls these 6 tools:

```
audit_get_lesson(lesson_id)
  Load full lesson document (id, title, severity, check_prompt, content_markdown).
  NO file_path exposed.

audit_list_lessons(category, severity, auto_checkable)
  List lessons with optional filters.

audit_get_code_context(file, focus_lines, depth)
  AST-based code context (imports, enclosing_function, variables).

audit_search_lessons(query, top_k)
  Search lessons by keyword relevance.

audit_static_precheck(file, lesson_ids)
  Engine A quick scan, returns violations grouped by lesson.

audit_submit_finding(finding_data)
  Agent submits a finding from semantic judgment.
```

### CLI invocation (humans / CI)

```bash
# Engine A static scan
quantnodes-audit scan /path/to/code --strict --output report.json
quantnodes-audit scan /path/to/code --lesson L-202 --severity CRITICAL

# Engine A precheck by lesson
quantnodes-audit precheck /path/to/code --lesson L-202 --lesson L-213

# Lesson lookup
quantnodes-audit lesson L-202
quantnodes-audit lessons --severity CRITICAL --category lookahead
quantnodes-audit search "look-ahead"

# Runtime validators
quantnodes-audit validate cv --strategy v7_10:run --start-dates 2018-01-01 2020-01-01
quantnodes-audit validate bootstrap --strategy v7_10:run --n-bootstrap 30

# MCP server
quantnodes-audit serve-mcp
```

### Python API

```python
from quantnodes_strategy_audit import (
    LessonLoader, StaticEngine, ContextEngine,
    CVCalculator, BootstrapStability, FiveGates,
)

# Lesson loading
loader = LessonLoader(builtin_dir="lessons")
lesson = loader.get("L-202")  # full markdown + check_prompt

# Engine A static scan
static = StaticEngine("rules/simple_rules.yaml")
warnings = static.scan_file(Path("strategy/v7/data_loader.py"))

# Engine B context
extractor = CodeContextExtractor()
ctx = extractor.extract(file, focus_lines=[142])

# Validators
cv = CVCalculator()
result = cv.run(backtest_fn=run_strategy, start_dates=["2018-01-01", "2020-01-01"])
# status: PASS (<25%) / PROMISING (25-50%) / DEPRECATED (>50%)
```

## Built-in Lessons (48)

### methodology (18)
L-101 简单规则胜复杂 / L-102 截面 vs 时序 IC / L-103 因子去重 / L-104 Gram-Schmidt /
L-105 反转效应不存在 / L-106 Smart β 低 beta / L-107 A 股动量偏好海外 / L-108 树模型 R²≈0 /
L-109 Symmetry 正交失败 / L-110 DCC overlay / L-111 图谱/相关性距离 / L-121 逆波动率加权 /
L-122 信号+风控消融 / L-123 粗粒度组合 / L-124 Vol-parity / L-125 5大机制 Sharpe 天花板 /
L-131 HMM 距离先验 / L-132 LW vs IC² / L-133 二值 vs 连续 TF / L-134 expanding vs rolling

### lookahead (5)
L-201 OOS 4 步流程 ⭐⭐⭐ / L-202 全样本标准化 / L-204 X[t]→Y[t+1] 同期陷阱 /
L-205 Y 与因子重叠 / L-223 full_sample ADMM

### oos_validation (4)
L-203 CV% 阈值 / L-215 成块缺失 / L-233 walk_forward 框架 / L-322 4 步流程

### data_quality (5)
L-211 OHLCV 调整 / L-212 起跑日对齐 / L-214 动态资产池 / L-222 标准化方向 / L-213 NaN-safe

### engineering (7)
L-231 统一引擎 / L-232 YAML 配置 / L-241 plotly 内嵌 / L-242 业绩精简 /
L-243 HTML OOS bug / L-323 工程债

### decision (4)
L-301 架构≠业绩 / L-302 高 Ann≠高 Calmar / L-303 诚实归因 / L-304 CV% P0

### integration (3)
L-321 5 道闸门 ⭐⭐⭐ / L-305 消融 / L-306 看基础设施

## Built-in Static Rules (17)

详见 `rules/simple_rules.yaml`：
- lookahead.shift_negative / full_sample_mean / standardscaler_default / ...
- nan_safe.bare_pct_change / fillna_zero / dropna_returns / interpolate_returns
- standardize.full_sample_zscore_inline / full_sample_zscore_var
- oos.fit_transform_on_full / full_sample_method / ...

## Validators (3)

| Validator | Purpose | Lesson |
|---|---|---|
| CVCalculator | Start-date dependency CV% test | L-203 |
| BootstrapStability | Block bootstrap stability (block_size=63) | L-215 |
| FiveGates | 5-gates integrated check | L-321 |

## Output Formats

| Format | Use Case |
|---|---|
| `text` | Human readable terminal |
| `json` | CI / programmatic |
| `sarif` | GitHub Code Scanning |

## Plugin Extension

```python
from quantnodes_strategy_audit import BaseDetector, Severity, Warning
from quantnodes_strategy_audit.core.registry import DetectorRegistry

@DetectorRegistry.register
class MyDetector(BaseDetector):
    name = "custom.my_rule"
    category = "custom"
    severity = Severity.HIGH

    def scan_file(self, file):
        # custom detection logic
        yield Warning(...)
```

Or add YAML rules to `rules/simple_rules.yaml`.

## Source

QuantNodes `docs/research_history/05_LESSONS_LIBRARY.md` (48 主题教训 L-101~L-323) +
`docs/lessons/daily/` (66 daily 教训, 2026-07-07 → 2026-07-28).