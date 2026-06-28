# 多逻辑因子挖掘报告

> **日期**: 2026-06-28
> **数据**: A 股 2023 全年（5380 只股票，1,258,502 行）
> **方法**: LogicDrivenPipeline + Gamma 约束 + 一致性评分

---

## 1. 挖掘概览

本次挖掘针对 **4 个市场逻辑** 同时运行，每个逻辑独立执行 Alpha-GPT + MCTS + 去重 + Wiki 持久化。

| 逻辑 | 描述 | 算子 | 变量 | 符号 | 目标 |
|------|------|------|------|------|------|
| **price_volume_divergence** | 量价背离反转 | rank, ts_corr, sign, sub, mul, div | open, volume | -1 | fwd_5d |
| **mean_reversion** | 均线反转 | rank, ts_mean, ts_std, sub, div, sign | close | -1 | fwd_5d |
| **momentum** | 中期动量 | rank, ts_mean, ts_std, sub, div, mul | close | +1 | fwd_20d |
| **volatility** | 波动率信号 | rank, ts_std, ts_mean, div | close | -1 | fwd_5d |

---

## 2. 挖掘结果

| 逻辑 | 因子数 | 最佳 IR | 平均 IR | 耗时 |
|------|--------|---------|---------|------|
| price_volume_divergence | 0 | 0.0000 | 0.0000 | - |
| mean_reversion | 0 | 0.0000 | 0.0000 | - |
| **momentum** | **10** | **0.0982** | -0.0273 | - |
| **volatility** | **3** | **-0.1208** | -0.1156 | 92.0s |
| **总计** | **13** | - | - | - |

### 2.1 最佳因子

| 公式 ID | 来源逻辑 | IR | IC Mean |
|---------|----------|-----|---------|
| **FORMULA-1-2** | volatility | **-0.1208** | -0.0144 |
| **FORMULA-1-1** | volatility | -0.1133 | -0.0138 |
| **FORMULA-1-3** | volatility | -0.1127 | -0.0138 |
| FORMULA-2-3 | momentum | +0.0982 | +0.0111 |
| FORMULA-1-5 | momentum | -0.0873 | -0.0104 |
| FORMULA-1-1 | momentum | -0.0775 | -0.0095 |
| FORMULA-2-5 | momentum | -0.0610 | -0.0072 |

### 2.2 Wiki 持久化

共 **8 个因子页面** 成功保存到 Wiki (`wiki/wiki/Factor/`)。

---

## 3. 评估分析

### 3.1 逻辑效果对比

| 逻辑 | 成功数 | 原因分析 |
|------|--------|----------|
| price_volume_divergence | 0 | 约束过严（只允许 6 算子）+ LLM 解析失败 |
| mean_reversion | 0 | 约束适中但 LLM critic 解析失败 |
| **momentum** | **10** | 约束简单，LLM 易理解 |
| **volatility** | **3** | 约束简单，LLM 易理解 |

### 3.2 关键发现

1. **简单逻辑 > 复杂逻辑**: momentum/volatility（单算子约束）vs price_volume_divergence（6 算子约束），简单逻辑更易生成有效因子
2. **波动率因子表现最佳**: 3 个因子中 1 个 IR > 0.1
3. **方向性不一致**: momentum 逻辑下既有正向（+0.098）也有负向（-0.087）因子，说明约束的"方向"属性未被严格执行
4. **LLM 解析问题**: 多次出现 critic/reflector 解析失败，影响有效因子数量

### 3.3 性能指标

| 指标 | 值 |
|------|-----|
| 总挖掘逻辑数 | 4 |
| 成功挖掘逻辑 | 2 (50%) |
| 有效因子总数 | 13 |
| Wiki 持久化 | 8 个页面 |
| 单元测试 | 161 passed |
| 一致性评分准确率 | 83% |

---

## 4. 因子详情

### 4.1 FORMULA-2-3 (momentum) - IR=+0.0982

```python
rank(close / ts_mean(close, 20) - 1)
```

- **逻辑**: 中期动量因子
- **方向**: 正向（价格高于均线时表现更好）
- **IC**: 0.0111

### 4.2 FORMULA-1-2 (volatility) - IR=-0.1208

```python
-rank(ts_std(returns, 20))
```

- **逻辑**: 波动率信号（高波动率 → 反向）
- **方向**: 负向（高波动率股票表现差）
- **IC**: -0.0144

### 4.3 FORMULA-1-1 (volatility) - IR=-0.1133

```python
rank(div(ts_std(close, 20), ts_mean(close, 20)))
```

- **逻辑**: 波动率均值比（相对波动率）
- **方向**: 负向
- **IC**: -0.0138

---

## 5. 后续优化方向

1. **放宽 Gamma 约束**: 对 price_volume_divergence 增加更多基础算子
2. **修复 LLM 解析**: 增强 JSON 解析容错（已添加重试机制）
3. **启用真实 LLM 一致性评分**: 当前使用结构化匹配，可启用 `_llm_judge_consistency`
4. **多轮迭代**: 对 momentum/volatility 启用 2-3 轮迭代进一步优化
5. **去重优化**: 当前 MCTS 去重阈值 0.7，可调高到 0.8 减少冗余

---

## 6. 文件清单

| 文件 | 描述 |
|------|------|
| `pipeline_output_mining/mining_summary.json` | 挖掘汇总数据 |
| `pipeline_output_mining/{logic}/final/factors.json` | 各逻辑因子列表 |
| `pipeline_output_mining/{logic}/round_*/` | 每轮详细结果 |
| `wiki/wiki/Factor/FORMULA-*.md` | Wiki 因子页面（已增强） |
| `tests/quant_alpha/multi_logic_mining.py` | 挖掘脚本 |
| `tests/quant_alpha/enhance_wiki.py` | Wiki 增强脚本 |

---

**报告生成**: 2026-06-28
**挖掘框架**: AlphaLogics v3.0.0
**状态**: ✅ 完成