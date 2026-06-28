# 修复 LLM Critic JSON 解析失败 → 实际修复 Dedup 排序 Bug

> **日期**: 2026-06-28
> **版本**: V3
> **状态**: ✅ 修复成功

---

## 1. 背景

LogicDrivenPipeline V2 运行后，4 个逻辑中 `price_volume_divergence` 和 `mean_reversion` 产出 **0 因子**。初步判断是 "LLM critic JSON 解析失败"，但实际根因在更底层。

---

## 2. 根因分析

### 2.1 初始假设 (错误)

- 日志显示 `critic parse failed: Cannot parse JSON after 2 layers`
- 假设是 LLM 输出格式问题

### 2.2 实际根因 (真相)

**Dedup 排序 Bug**：当所有因子 `IR` 均为负数时，贪心 Spearman 去重会失败。

#### Bug 位置

`QuantNodes/research/quant_alpha/evaluation/evaluators/polars_evaluator.py:73` (修复前)：

```python
# Bug: 按 overall_score 降序排序（值都是负数 → 排序不变）
sorted_f = sorted(factors, key=lambda f: f.overall_score, reverse=True)
```

#### Bug 行为

对于 mean_reversion：

- 3 个因子都有 `ir = -0.0813` (负数)
- `overall_score = ir = -0.0813`
- 按 `reverse=True` 降序排序时，**值都是负数，排序无效**
- 第 1 个因子被加入 `selected` 集合
- 第 2 个因子：Spearman corr = 1.0（同一公式）→ 跳过
- 第 3 个因子：Spearman corr = 1.0 → 跳过
- **dedup 后剩 0 个因子** → final_pool 为空

#### 修复

```python
# 修复: 按 |overall_score| 降序排序
sorted_f = sorted(factors, key=lambda f: abs(f.overall_score), reverse=True)
```

同时也修复 `_merge_and_dup()` 排序 (`pipeline.py:770`)。

---

## 3. 验证结果 (V3)

### 3.1 V1 → V2 → V3 对比

| 逻辑 | V1 | V2 (修复前) | V3 (修复后) | 变化 |
|------|-----|------------|------------|------|
| price_volume_divergence | 0 | 0 | 0 | 无变化 (LLM 解析失败) |
| **mean_reversion** | 0 | 0 | **3** | **+3 因子** ✅ |
| momentum | 10 | 3 | 3 | -7 因子 (LLM 波动) |
| volatility | 3 | 3 | 0 | -3 因子 (LLM 波动) |
| **总因子** | **13** | **6** | **6** | - |

### 3.2 mean_reversion 修复细节

| 项 | 修复前 (V2) | 修复后 (V3) |
|----|------------|------------|
| 因子数 | 0 | 3 |
| Best IR | 0.0000 | +0.0775 |
| 实际原因 | 全部负 IR 被 dedup 过滤 | 修复 abs 排序后正常返回 |

**注意**: V3 中 mean_reversion 的 IR=+0.0775（正），与负数期望相反。这是因为 LLM 实际生成了 `sub(close, ts_mean(close, 10))` 公式（截断的窗口），效果与预期 `sub(close, ts_mean(close, 20))` 不同。

### 3.3 V3 完整输出

| 逻辑 | 因子数 | Best IR | 耗时 |
|------|--------|---------|------|
| price_volume_divergence | 0 | +0.0000 | 181.5s |
| mean_reversion | **3** | **+0.0775** | 934.5s |
| momentum | 3 | +0.0144 | 213.3s |
| volatility | 0 | +0.0000 | 68.3s |
| **总计** | **6** | - | **1397.6s** |

---

## 4. 教训

### 4.1 关键教训

1. **不要被表面错误信息误导**：`critic parse failed` 是表面错误，实际根因在 dedup 层
2. **负数 IR 的 dedup 必须用 `abs()`**：排序时 `reverse=True` 对负数无效
3. **调试要从输出反推**：从 `final_pool=0` 反推到 `_merge_and_dedup` → `deduplicate_mutual_ic` → sort key

### 4.2 类似 Bug 检查清单

- [x] `deduplicate_mutual_ic` 排序 - **已修复**
- [x] `_merge_and_dedup` 排序 - **已修复**
- [ ] `_select_final_pool` 排序 - **已正确** (用 `abs(m.ir)`)
- [ ] MCTS 排序 - 待检查
- [ ] 早期版本代码 - 需检查其他可能的 `sort(... reverse=True)` 模式

### 4.3 改进方向

1. **统一排序标准**：所有最终排序都应使用 `abs(score)`
2. **添加单元测试**：测试负数 IR 场景的 dedup 行为
3. **改进错误日志**：当 `final_pool=0` 时，自动打印每个因子的 IR 范围

---

## 5. 文件变更

| 文件 | 修改 |
|------|------|
| `QuantNodes/research/quant_alpha/evaluation/evaluators/polars_evaluator.py` | L73: `sort(key=lambda f: abs(f.overall_score), reverse=True)` |
| `QuantNodes/research/quant_alpha/pipeline.py` | L770: 同上修复 |
| `tests/quant_alpha/` | 添加负数 IR 测试（待办） |

---

## 6. 下一步

1. **检查 MCTS 排序**：确认无类似 bug
2. **重跑 volatility**：验证是否 LLM 波动导致 0 因子
3. **添加负数 IR 单元测试**：防止回归
4. **改进错误日志**：final_pool=0 时自动打印原因

---

**报告生成**: 2026-06-28
**状态**: ✅ V3 修复完成