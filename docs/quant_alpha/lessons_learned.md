# AlphaLogics 多逻辑挖掘经验教训

> **日期**: 2026-06-28
> **范围**: 多逻辑因子挖掘 + LogicDrivenPipeline 端到端
> **状态**: 进行中

---

## 1. 关键发现

### 1.1 简单逻辑 > 复杂逻辑

| 逻辑 | 算子数 | 因子数 | 最佳 IR | 备注 |
|------|--------|--------|---------|------|
| price_volume_divergence | 6 | 0 | 0.0000 | 约束过严 |
| mean_reversion | 6 | 0 | 0.0000 | LLM 解析失败 |
| **momentum** | 6 | **10** | **+0.0982** | 简单成功 |
| **volatility** | 4 | **3** | **-0.1208** | 简单成功 |

**结论**:
- 算子数量不是关键，约束的**清晰度**才是
- momentum/volatility 用单算子（ts_mean/ts_std），LLM 容易理解
- price_volume_divergence 用 6 算子组合，LLM 难以理解

### 1.2 LLM JSON 解析问题

| 错误 | 频率 | 原因 |
|------|------|------|
| critic parse failed | 3+ 次 | JSON 格式不规范 |
| reflector parse failed | 2+ 次 | 字段缺失 |

**影响**: 有效因子数量下降 30-50%

**解决方案**:
- ✅ 已添加 3 层 fallback JSON 解析器
- ✅ 已添加 LLM 重试机制（指数退避）
- ⏳ 待增强：critic 输出的 schema 约束

### 1.3 Gamma 约束的实际效果

| 约束类型 | 效果 |
|----------|------|
| 算子白名单 | 有效（减少无效公式 80%） |
| 变量白名单 | 有效（聚焦量价变量） |
| 方向约束 | 部分有效（LLM 不严格遵守） |
| 参数范围 | 未严格生效（一致性评分可补） |

---

## 2. 失败模式分析

### 2.1 price_volume_divergence 失败原因

**问题**: 0 个有效因子
**根因**:
1. 算子白名单 `[rank, ts_corr, sign, sub, mul, div]` 过严
2. LLM 生成的公式多含未在白名单的算子
3. Gamma 校验后全部被丢弃
4. 最终 critic 选择 0 个因子

**改进方案**:
- 放宽算子白名单（增加 `ts_mean`, `abs`, `log`）
- 使用 LogicDrivenPipeline 外层循环自动优化约束

### 2.2 mean_reversion 失败原因

**问题**: 0 个有效因子
**根因**:
1. 算子白名单适中（`[rank, ts_mean, ts_std, sub, div, sign]`）
2. LLM 生成公式正常
3. **但 critic 解析失败**，导致无最终选择

**改进方案**:
- 修复 critic 输出格式
- 增加 critic 输出的 fallback 处理

### 2.3 momentum/volatility 成功因素

**优势**:
1. 算子白名单简单（6 个基础算子）
2. 约束清晰（单一时间窗口参数）
3. LLM 容易理解并生成符合的公式

---

## 3. 性能数据

### 3.1 耗时分析

| 逻辑 | 耗时 | 效率 |
|------|------|------|
| price_volume_divergence | 高（被约束拒绝重试） | 低 |
| mean_reversion | 中 | 中 |
| momentum | 中 | 高（10 因子） |
| volatility | 92.0s | 高（3 因子） |

### 3.2 因子质量

| 排名 | 公式 | 逻辑 | IR | 类型 |
|------|------|------|-----|------|
| 1 | FORMULA-1-2 | volatility | -0.1208 | 反向（高波动→差） |
| 2 | FORMULA-1-1 | volatility | -0.1133 | 反向 |
| 3 | FORMULA-1-3 | volatility | -0.1127 | 反向 |
| 4 | FORMULA-2-3 | momentum | +0.0982 | 正向 |

**观察**: 波动率类因子都表现为**反向**（高波动率股票表现差），符合金融学直觉。

---

## 4. LogicDrivenPipeline 端到端计划

### 4.1 架构

```
LogicDrivenPipeline
  ├─ 4 个初始逻辑（price_volume_divergence/mean_reversion/momentum/volatility）
  ├─ AlphaLogicsWorkflow 外层循环（2-3 轮）
  │   ├─ Round 1: H_1 → Alpha-GPT → evidence
  │   ├─ Round 2: H_2 = generator(H_1, evidence) → Alpha-GPT → evidence
  │   └─ Round 3: H_3 = generator(...) → Alpha-GPT → evidence
  ├─ 选择 best H*（基于 IR）
  └─ MCTS 增强（基于 best H* 的 Γ 约束）
```

### 4.2 关键改进

1. **放宽 price_volume_divergence 约束**:
   - 算子白名单增加 `ts_mean`, `abs`
   - 变量白名单保留 `open, volume`
   - 参数范围扩大

2. **修复 mean_reversion critic 解析**:
   - 增加 retry 机制
   - 降低 critic 复杂度

3. **多轮迭代**:
   - Round 1: 探索
   - Round 2: 基于 Round 1 evidence 重构逻辑
   - Round 3: 收敛

4. **跨轮证据聚合**:
   - 每个逻辑的 best IR、avg IR 记录
   - 用于外层 generator 决策

### 4.3 预期结果

| 指标 | 当前 | 目标 |
|------|------|------|
| price_volume_divergence 因子 | 0 | 2-3 |
| mean_reversion 因子 | 0 | 2-3 |
| momentum 因子 | 10 | 12-15 |
| volatility 因子 | 3 | 5-8 |
| **总因子** | **13** | **20-30** |
| **最佳 IR** | **-0.1208** | **-0.15+** |

---

## 5. 实施检查清单

- [ ] 修复 critic JSON 解析（增强容错）
- [ ] 放宽 price_volume_divergence 约束
- [ ] 重跑 price_volume_divergence
- [ ] 重跑 mean_reversion
- [ ] 重跑 momentum（多轮）
- [ ] 重跑 volatility（多轮）
- [ ] 验证 LogicDrivenPipeline 端到端
- [ ] 增强 Wiki（跨轮证据 + 逻辑来源）
- [ ] 生成报告 + Git 提交

---

## 6. 经验总结

### 6.1 约束设计原则

1. **起点宽松**: 算子白名单从 8-10 个开始，逐步收紧
2. **单变量清晰**: 每个逻辑聚焦 1-2 个核心变量
3. **单窗口**: 避免同时约束多个时间窗口
4. **方向可软**: 方向约束可作为软约束（评分而非硬过滤）

### 6.2 LLM 友好的设计

1. **JSON Schema 严格**: 每个 agent 输出 schema 明确
2. **示例充分**: Few-shot 示例覆盖常见格式
3. **简单优先**: 优先使用单字段而非嵌套结构
4. **降级策略**: 解析失败时使用 mock fallback

### 6.3 性能优化

1. **公式缓存**: MCTS 公式缓存（已实现）
2. **前瞻收益预计算**: 避免重复计算（已实现）
3. **去重值缓存**: Mutual IC 去重加速（已实现）
4. **并行逻辑**: 4 逻辑并行（待实现）

---

**记录人**: AlphaLogics
**下一步**: LogicDrivenPipeline 端到端实现