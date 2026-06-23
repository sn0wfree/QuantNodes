---
name: quant-dream
description: 量化专属 Dream 钩子 — 因子/回测/策略洞察的自动沉淀与跨会话记忆。
---

# Quant Dream

在 HKUDS nanobot 通用 Dream（记忆整合）之上叠加的**量化专属**反思层。
挂在 nanobot 的 `AgentHook` 系统上，每个 session 结束后异步运行。

## 触发条件

每 N 轮对话后（或显式调用 `quant_dream_analyze` 工具）触发一次：
- session 中包含因子公式关键词 (`alpha`, `factor`, `IC`, `rank`)
- session 中包含回测关键词 (`backtest`, `sharpe`, `drawdown`)
- session 中包含策略关键词 (`strategy`, `pipeline`, `weight`)

## 分析内容

1. **因子洞察** — 提取 IC 表现好的因子，反思构造逻辑
2. **回测模式** — 记录过拟合/未来函数/手续费过高等问题模式
3. **策略启发** — 跨策略对比，识别共同成功因素
4. **风险事件** — 大回撤/极端行情下的策略表现
5. **代码模式** — LLM 生成的常见代码 bug（除零/未来函数/look-ahead bias）

## 输出

写入 `.agent/memory/topic-quant-dream.md`（Markdown 索引格式）：

```markdown
### 2026-06-23 - factor_insight
- momentum_20d 因子在 2020-2024 年化 18%, ICIR=1.2
  - 构造: close / close.shift(20) - 1
  - 适用: 中证 500 成分股, 调仓周期 5 日
  - 失效条件: 极端反转市（2015-06, 2020-03）

### 2026-06-23 - backtest_pattern
- 样本外夏普衰减 > 50% 提示过拟合
  - 案例: momentum_v3 样本内 2.1 → 样本外 0.8
  - 修正: 缩短调仓周期至 5 日, 增加中性化
```

## 工具集

| 工具 | 用途 |
|------|------|
| `wiki_write` | 写入 Wiki 页面（结构化） |
| `wiki_query` | 检索历史 Dream 洞察 |
| `quant_dream_analyze` | 显式触发分析 |

## 频率控制

- 默认: 每 10 轮对话触发一次（防过度反思）
- 显式调用: 立即触发
- 失败静默: 不打断主对话流
