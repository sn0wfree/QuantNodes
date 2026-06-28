# LogicDrivenPipeline 端到端运行报告

> **日期**: 2026-06-28
> **版本**: v2.0
> **状态**: ✅ 完成

---

## 1. 概述

本次运行是 **多逻辑挖掘 (V1)** → **LogicDriven 端到端 (V2)** 的进阶版。

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 记录多逻辑挖掘经验 | ✅ |
| 2 | LogicDrivenPipeline 端到端脚本 | ✅ |
| 3 | 4 逻辑重跑（放宽约束） | ✅ |
| 4 | 增强 Wiki + 报告 + 提交 | ✅ |

---

## 2. V1 vs V2 对比

### 2.1 配置对比

| 项 | V1 | V2 (LogicDriven) |
|------|-----|------------------|
| 算子白名单 | 6 | 6-8 (放宽) |
| price_volume_divergence | 6 算子 | 8 算子 (+ts_mean, abs) |
| mean_reversion | 6 算子 | 6 算子（简化版） |
| Wiki 路径 | `wiki/` | `wiki/` (合并) |
| 输出路径 | `pipeline_output_mining/` | `pipeline_output_v2/` |

### 2.2 结果对比

| 逻辑 | V1 因子 | V1 best_ir | V2 因子 | V2 best_ir | 变化 |
|------|---------|------------|---------|------------|------|
| price_volume_divergence | 0 | 0.0000 | 0 | 0.0000 | - |
| mean_reversion | 0 | 0.0000 | 0 | 0.0000 | - |
| momentum | 10 | +0.0982 | 3 | +0.0526 | 因子数↓ IR↓ |
| volatility | 3 | -0.1208 | 3 | -0.1208 | 完全一致 |
| **总因子** | **13** | - | **6** | - | **-7** |

### 2.3 关键观察

1. **volatility 复现性极强**: V1 和 V2 完全相同的 3 个因子（IR 误差 < 0.001）
2. **momentum 不稳定**: 不同运行产生不同因子（LLM 输出波动）
3. **price_volume_divergence/mean_reversion 仍为 0**: 根因是 LLM critic 解析失败

---

## 3. LogicDrivenPipeline 端到端验证

### 3.1 架构

```
LogicDrivenPipeline
  ├─ AlphaLogicsWorkflow (外层循环)
  │   ├─ Logic Mining (alpha101/158 → 初始库)
  │   ├─ Round 1: H_1 → Alpha-GPT → evidence
  │   ├─ Round 2: H_2 = generator(H_1, evidence) → Alpha-GPT → evidence
  │   └─ Best H* 持久化
  └─ MCTS 增强 (基于 best H* 的 Gamma)
```

### 3.2 验证结果

| 测试 | 状态 | 备注 |
|------|------|------|
| AlphaLogicsWorkflow 初始化 | ✅ | mock 模式可工作 |
| 初始逻辑库构建 | ✅ | 4 个逻辑 (2 初始 + 2 外层生成) |
| 外层循环 | ✅ | 2 轮完成 |
| 证据聚合 | ✅ | 跨轮 IR/IC 记录 |
| Wiki 持久化 | ✅ | Logic 页面生成 |
| LLM 实际调用 | ⚠️ | 需真实 LLM 集成 |

**注意**: 当前测试使用 mock 模式（0.2s 完成）。生产环境需要真实 LLM 集成（预计 5-10 分钟/逻辑）。

---

## 4. 挖掘成果

### 4.1 最佳因子 (Top 5)

| 排名 | 公式 ID | 逻辑 | IR | IC |
|------|---------|------|-----|-----|
| 1 | FORMULA-1-2 | volatility | -0.1208 | -0.0144 |
| 2 | FORMULA-1-1 | volatility | -0.1133 | -0.0138 |
| 3 | FORMULA-1-3 | volatility | -0.1127 | -0.0138 |
| 4 | FORMULA-2-3 | momentum | +0.0982 | +0.0111 |
| 5 | FORMULA-1-5 | momentum | -0.0873 | -0.0104 |

### 4.2 Wiki 页面

8 个因子页面已持久化（V1 + V2 合并）：

| 页面 | 逻辑 | IR |
|------|------|-----|
| FORMULA-1-1 | volatility | -0.1133 |
| FORMULA-1-2 | volatility | -0.1208 |
| FORMULA-1-3 | volatility | -0.1127 |
| FORMULA-1-5 | momentum | -0.0873 |
| FORMULA-2-2 | momentum | -0.0541 |
| FORMULA-2-3 | momentum | +0.0982 |
| FORMULA-2-4 | momentum | -0.0527 |
| FORMULA-2-5 | momentum | -0.0610 |

---

## 5. 改进方向

### 5.1 短期改进

1. **修复 LLM critic 解析**: 当前 critic 经常失败，影响有效因子选择
2. **修复 price_volume_divergence**: 即使放宽算子白名单，仍为 0 因子
3. **统一两次运行结果**: momentum 在 V1/V2 产生不同因子

### 5.2 中期改进

1. **多轮迭代优化 IR**: 当前 1 轮，可扩展到 3-5 轮
2. **真实 LLM 一致性评分**: 启用 `_llm_judge_consistency` 路径
3. **并行化 4 逻辑**: 减少总耗时

### 5.3 长期改进

1. **PR-6 (Alpha191)**: 补全因子库
2. **跨数据集验证**: 在不同时间段/股票池验证因子稳健性
3. **实盘回测**: 集成到策略回测框架

---

## 6. 文件清单

| 文件 | 描述 |
|------|------|
| `tests/quant_alpha/run_logic_driven.py` | LogicDrivenPipeline 端到端脚本 |
| `pipeline_output_logic_driven/` | LogicDrivenPipeline 输出 |
| `pipeline_output_v2/` | V2 多逻辑挖掘输出 |
| `pipeline_output_v2/v2_summary.json` | V2 汇总数据 |
| `docs/quant_alpha/lessons_learned.md` | 经验教训 |
| `docs/quant_alpha/multi_logic_mining_report.md` | V1 报告 |

---

## 7. 结论

### 7.1 成就

- ✅ **6 个有效因子** 在 V2 中挖掘
- ✅ **volatility 复现性 100%**（V1 = V2 同样的 3 因子）
- ✅ **LogicDrivenPipeline 架构验证通过**
- ✅ **0 因子问题定位**（LLM critic 解析失败）

### 7.2 限制

- ⚠️ price_volume_divergence 和 mean_reversion 仍为 0 因子
- ⚠️ momentum 不稳定（V1=10 因子，V2=3 因子）
- ⚠️ LLM critic 解析失败频发

### 7.3 下一步

1. **修复 LLM 解析** (高优先级)
2. **多轮迭代** (中优先级)
3. **真实 LLM 一致性评分** (中优先级)

---

**报告生成**: 2026-06-28
**运行框架**: AlphaLogics v3.0.0 + LogicDrivenPipeline
**状态**: ✅ V2 完成