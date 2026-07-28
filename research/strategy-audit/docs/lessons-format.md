# 教训 Markdown 编写规范

每个 L-NNN 教训是一个独立的 `.md` 文件，位于 `lessons/` 目录。

## 文件命名

```
L-{NNN}-{slug}.md
```

例如：`L-202-full-sample-std.md`（slug 用短横线分隔的英文短语）。

## 文件结构

```markdown
---
id: L-202
title: 教训标题
severity: CRITICAL  # 或 HIGH / MEDIUM / LOW
auto_checkable: agent  # 或 static / partial / manual
category: lookahead   # lookahead / nan_safe / oos_validation / ...
related_lessons: [L-201, L-223]
related_daily: [L-20260728-1]
source: 05_LESSONS_LIBRARY.md
---

# L-202: 教训标题

## 一句话总结
（一句话总结问题）

## 问题描述
（详细描述问题）

## 检测 prompt (给 Agent 的检查清单)
（关键：明确列出 Agent 应检查什么）

### 1. 找出可疑调用
（具体的模式识别清单）

### 2. 评估语义
（判断哪些是合法的，哪些是违规）

### 3. 判断违规
（明确的违规条件）

## 正确做法
（修复示例代码）

## 关联代码案例
（错误 vs 修复后对比）

## 历史教训来源
（首次发现时间、commit hash）
```

## auto_checkable 决定

| 值 | 含义 | 处理 |
|---|---|---|
| `static` | Engine A 静态规则可检测 | YAML 规则 |
| `agent` | 必须 Agent 语义理解 | 文档 + 教训 prompt |
| `partial` | 静态 + Agent 结合 | 静态给候选, Agent 确认 |
| `manual` | 仅供人参考 | 仅文档 |

## check_prompt 编写要点

1. **明确步骤**：列出 Agent 应遵循的检查顺序
2. **具体模式**：给出可疑代码的具体例子
3. **合法 vs 违规**：明确区分边界情况
4. **JSON 输出格式**：建议 Agent 返回的结构

## 编写建议

- 复制现有 `L-202.md` 作为模板
- severity 准确：CRITICAL = 数据穿越 / HIGH = 常见 bug / MEDIUM-LOW = 工程改进
- check_prompt 是核心：Agent 用它做语义判断
- 关联代码案例要真实可验证