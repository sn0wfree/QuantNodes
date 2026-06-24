---
name: strategy-design
description: 策略设计工作流 — 因子组合、权重优化、回测验证、参数敏感性。
---

# Strategy Design

驱动 QuantNodes 策略设计全流程：因子筛选 → 组合方式 → 回测验证 → 参数敏感性 → Wiki 沉淀。

## 工作流

1. **因子组合方案** — 基于 `factor-research` 输出，调用 `strategy_generate` 让 LLM 设计 2-3 种组合方案（等权/IC 加权/最大化 ICIR）
2. **Pipeline 验证** — 调用 `pipeline_validate` 验证整体 Pipeline 代码
3. **沙箱执行** — 调用 `sandbox_validate` 检查代码安全（无 import 危险模块）
4. **回测验证** — 调用 `backtest_run` 或 `config_backtest` 跑 3-5 年样本外回测
5. **参数敏感性** — 修改关键参数（调仓周期/持仓数/止损），观察绩效变化
6. **过拟合检测** — 样本内 vs 样本外对比；多空收益曲线对比；夏普衰减 < 30%
7. **沉淀 Wiki** — 调用 `wiki_write` 写入策略页面，附参数表 + 收益曲线 + 回撤分析

## 工具集

| 工具 | 用途 |
|------|------|
| `strategy_generate` | NL → Pipeline 代码 |
| `pipeline_validate` | Pipeline 语法/依赖验证 |
| `sandbox_validate` | 代码安全检查 |
| `backtest_run` | 快速回测（pipeline_code） |
| `config_backtest` | 配置驱动回测（YAML） |
| `wiki_write` | 写入策略 Wiki 页面 |

## 验收标准

- 年化收益 > 15%
- 最大回撤 < 20%
- 夏普比率 > 1.5
- 样本外夏普衰减 < 30%
- 调仓成本占比 < 30% 收益

## 反模式

- 不要在样本内做参数优化（过拟合）
- 不要忽视交易成本（手续费+滑点）
- 不要只看总收益不看回撤
