# Agent 使用 quantnodes-strategy-audit 完整工作流

## 概述

quantnodes-strategy-audit 提供 6 个 MCP 工具供 Agent 调用。本文档展示一个完整的审计流程。

## 工作流示例：审查 v7.10 标准化函数

### Step 1: 列出相关教训

Agent 调用 `audit_list_lessons`：

```python
lessons = await mcp.call("audit_list_lessons", {
    "category": "lookahead",
    "severity": "CRITICAL",
})
# 返回 5+ 个相关教训: L-201, L-202, L-204, L-205, L-223
```

### Step 2: Engine A 预检可疑位置

```python
precheck = await mcp.call("audit_static_precheck", {
    "file": "v7/data_loader_v7_6.py",
    "lesson_ids": ["L-202", "L-213"],
})
# 返回:
# {
#   "total_warnings": 3,
#   "by_lesson": {
#     "L-202": [{"line": 142, "snippet": "mean = X.mean(axis=(0, 2))", ...}],
#     "L-213": [{"line": 165, "snippet": "returns = X.pct_change()", ...}],
#   },
#   "precheck_violated": ["L-202", "L-213"]
# }
```

### Step 3: 加载完整教训（含 check_prompt）

```python
lesson = await mcp.call("audit_get_lesson", {"lesson_id": "L-202"})
# 返回完整 markdown + check_prompt + content_markdown
```

### Step 4: 获取可疑位置的代码上下文

```python
context = await mcp.call("audit_get_code_context", {
    "file": "v7/data_loader_v7_6.py",
    "focus_lines": [142, 143],
    "depth": 3,
})
# 返回:
# {
#   "imports": ["import pandas as pd", "import numpy as np"],
#   "enclosing_function": {
#     "name": "standardize_v7_10",
#     "lines": "135-160",
#     "docstring": "标准化 v7.10 因子数据",
#     "args": ["X", "factor_names"],
#   },
#   "variables": [{"name": "mean", "line": 142, "snippet": "..."}],
#   "surrounding_lines": [...]
# }
```

### Step 5: Agent 自己做语义判断

Agent 现在有了：
- 完整教训 + check_prompt (Step 3)
- 代码上下文 (Step 4)
- Engine A 标记的位置 (Step 2)

Agent 用 LLM 判断：
- L-202: standardize_v7_10 在 fit 函数中用了全样本均值 → **VIOLATED**
- L-213: pct_change 没有 .where() 包装 → **VIOLATED**

### Step 6: 提交 finding

```python
await mcp.call("audit_submit_finding", {
    "file": "v7/data_loader_v7_6.py",
    "line": 142,
    "lesson_id": "L-202",
    "status": "VIOLATED",
    "severity": "CRITICAL",
    "evidence": {
        "snippet": "mean = X.mean(axis=(0, 2))",
        "explanation": "在 standardize 函数中用全样本均值, 含未来数据",
    },
    "fix_suggestion": "使用 X.rolling(252).mean() 或 X.expanding().mean()",
    "confidence": 0.95,
})

await mcp.call("audit_submit_finding", {
    "file": "v7/data_loader_v7_6.py",
    "line": 165,
    "lesson_id": "L-213",
    "status": "VIOLATED",
    "severity": "HIGH",
    "evidence": {"snippet": "returns = X.pct_change()"},
    "fix_suggestion": "添加 .where(nav.shift(1).notna() & nav.notna())",
    "confidence": 0.90,
})
```

### Step 7: 汇总报告

Agent 读取所有 submitted findings，生成结构化报告给用户。

## 关键设计原则

1. **Engine B 不调 LLM**：Agent 自己做判断
2. **Engine A 给候选**：标记可疑位置，避免 Agent 阅读全部代码
3. **教训驱动**：Agent 用 check_prompt 做语义理解
4. **Q3 = A 不暴露路径**：工具只返回内容

## 反馈循环

如果 Agent 发现新教训（不在 lessons/ 中），可以建议添加到 lessons/ 目录：

```python
# Agent 提交新教训提案
new_lesson = {
    "id": "L-NEW-1",
    "title": "新发现的 bug 模式",
    "severity": "HIGH",
    "category": "lookahead",
    "auto_checkable": "agent",
    "content_markdown": "...",
    "check_prompt": "...",
}
# 开发者审核后添加到 lessons/L-NEW-1-new-bug.md
```