# V5 vs V4 对比报告

**版本对比**: V4 (baseline) vs V5 (Tier 1+2+4 思维链利用)
**日期**: 2026-06-28
**作者**: LLM Pipeline
**分支**: `feature/thinking-chain`

---

## 实验设置

| 维度 | V4 (baseline) | V5 (thinking-chain) |
|------|---------------|---------------------|
| **Tier 1: 思维链捕获** | ❌ 100% 丢弃 | ✅ 100% 持久化到 llm_raw/ |
| **Tier 2: 结构化推理** | ❌ 无 | ✅ 4 个 prompt 都加结构化指令 |
| **Tier 4: OpPrior** | ❌ 均匀采样 | ✅ alpha=0.7 指数衰减 + mix=0.5 |
| **其他** | 同 | 同 |

两个实验使用完全相同的：
- 4 逻辑（volatility / momentum / mean_reversion / price_volume_divergence）
- pool_size=3, iterations=1
- MCTS iterations=20, max_depth=4
- forward_returns=(1, 5, 20)
- LLM: MiniMax M3 via direct OpenAI API
- max_tokens=16384, timeout=300s

---

## 结果对比

### 总体

| 指标 | V4 | V5 | Δ |
|------|----|----|---|
| 总因子数 | 9 | 3 | **-67%** ⚠️ |
| 最佳 \|IR\| | 0.1208 | 0.1167 | -3% ≈ |
| 总耗时 | 522.2s | 554.7s | +6% ≈ |
| 4 逻辑全部成功 | 3/4 | 1/4 | **-50%** ⚠️ |

### 逐逻辑

| 逻辑 | V4 因子数 | V4 最佳 \|IR\| | V5 因子数 | V5 最佳 \|IR\| | 评价 |
|------|-----------|---------------|-----------|---------------|------|
| **momentum** | 3 | 0.0387 | 3 | **0.1167** | ✅ 3x 提升 |
| mean_reversion | 3 | 0.1133 | 0 | 0.0 | ❌ 退化 |
| volatility | 3 | 0.1208 | 0 | 0.0 | ❌ 退化 |
| price_volume_divergence | 0 | 0.0 | 0 | 0.0 | ➖ 同样失败 |

### 关键发现

1. **momentum 显著提升**：最佳 \|IR\| 从 0.0387 → 0.1167（3x 改善）
2. **3/4 逻辑回归**：mean_reversion / volatility 从 3 因子 → 0 因子
3. **总因子数下降**：9 → 3

### 回归根因分析

LLM 在新结构化 prompt 下，**将 HYPOTHESIS/MECHANISM/OPERATOR_RATIONALE 等字段塞进 JSON 的 `explanation` 字段**，导致：

```
HYPOTHESIS: ...
MECHANISM: ...
OPERATOR_RATIONALE: ...
PARAMETER_RATIONALE: ...
RISK: ...
```

每个 explanation 长度从 ~30 chars 膨胀到 ~500 chars，3 个 formulas × 500 chars = 1500 chars 的额外 token，触发 max_tokens 截断（16384 不够用），导致 JSON 解析失败。

**样例** (mean_reversion 公式翻译响应):
```json
{
  "formulas": [
    {
      "id": "FORMULA-1-1",
      "formula": "rank(sub(close, ts_mean(close, 20)))",
      "explanation": "20日价格反转因子。HYPOTHESIS: A股散户对短期涨跌过度反应...MECHANISM: sub(close, ts_mean(close,20)) 衡量...OPERATOR_RATIONALE: ..."
    },
    ...
  ]
}
```

JSON 在 FORMULA-1-3 之前被截断，LLM 继续输出 thinking + 重写 JSON 块，但 parser 无法恢复嵌套结构。

### Tier 1 价值（捕获）

**L 已持久化**：每个 LLM 调用的 `<think>` 块都保存到 `pipeline_output_v5/{logic}/llm_raw/{agent_id}_thinking_*.txt`，可用于：
- 人工调试
- 训练数据
- 后续提取更丰富信号

**样例 thinking 内容** (volatility, 7946 chars):
```
HYPOTHESIS: A 股市场波动率因子可从三个正交维度捕捉 alpha
MECHANISM: A 股散户主导（>60%）+ T+1 制度 + 涨跌停板使得高波动股票存在过度交易
OPERATOR_RATIONALE: 选用 ts_std 衡量已实现波动率、ts_slope 捕捉波动率时序趋势
PARAMETER_RATIONALE: 20 日对应月度反转学术经典窗口
RISK: 政策切换可能改变波动率-收益关系（如熔断、注册制改革）
SUGGESTED_OPS: ts_std, ts_slope, rank, neg, IndNeutralize
```

### Tier 2 价值（结构化推理）

**已生效**：
- IdeaRecord.hypothesis 在 3/4 逻辑中成功提取
- 例如 momentum 的 IDEA-1-1: "A股散户过度反应高波动股票，低波动组合长期跑赢"

**未生效**：
- 公式翻译阶段 LLM 把字段塞进 explanation → JSON 截断
- 反射器/评论家的 key_insights 未填充（thinking 没按结构化输出）

### Tier 4 价值（OpPrior）

**momentum 累积**：
```json
{
  "weights": {
    "ts_min": 0.36,
    "ts_std": 0.32,
    "rank": 0.25,
    "ts_mean": 0.25,
    "div": 0.24
  },
  "total_updates": 5
}
```

**跨逻辑累积**:
- 当前实现每个 logic 用自己的 output_dir，未跨逻辑共享
- 下次跑同 logic 会从历史 op_prior.json 加载

---

## 结论

### 思维链利用的潜力

✅ **momentum 3x IR 提升**：证明 hypothesis 提取 + OpPrior 引导能显著改善有效因子的质量
✅ **Tier 1 100% 持久化**：thinking 内容完全可追溯
✅ **IdeaRecord.hypothesis 提取成功**：structured 格式生效

### 思维链利用的当前问题

❌ **explanation 字段膨胀**：LLM 把结构化字段塞进 explanation 导致 JSON 截断
❌ **回归 mean_reversion/volatility**：需要在 prompt 中明确禁止
❌ **OpPrior 未跨逻辑共享**：当前每逻辑独立

### 后续迭代建议

#### 短期（修复 V5 回归）

1. **更新 formula-translator prompt**：
   ```
   CRITICAL: explanation field MUST be <100 chars. 
   Do NOT include HYPOTHESIS/MECHANISM/OPERATOR_RATIONALE etc. in explanation.
   Output structured reasoning ONLY in <think> block, NOT in JSON.
   ```

2. **更新 idea-generator prompt**：
   ```
   description field MUST be <200 chars. Keep it brief.
   Structured reasoning goes in <think> only.
   ```

3. **增加 max_tokens** to 32768（如果 API 支持）

#### 中期（强化 V5 优势）

1. **跨逻辑共享 OpPrior**：用 `output_dir = "pipeline_output_v5/"` 统一，4 逻辑累积先验
2. **强化 OpPrior 信号**：用 IdeaRecord.mentioned_ops（来自 Tier 2）作为额外信号
3. **MCTS LLM 通道用 hypothesis**：把 hypothesis 喂入 _llm_judge_consistency 做更准确的一致性评分

#### 长期

1. **多轮迭代中累积 hypothesis**：用 hypothesis 判断 idea 是否已被探索过
2. **基于 hypothesis 相似度的去重**：IdeaRecord.hypothesis 用于 OOS 检测
3. **prompt 长度动态调整**：根据 pool_size 决定 max_tokens

---

## 文件

- V4 脚本: `tests/quant_alpha/run_4_logic_v4.py`
- V5 脚本: `tests/quant_alpha/run_4_logic_v5.py`
- V4 baseline: `pipeline_output_v4/v4_summary.json`
- V5 results: `pipeline_output_v5/v5_summary.json`
- LLM raw responses: `pipeline_output_v5/{logic}/llm_raw/`
- OpPrior snapshots: `pipeline_output_v5/{logic}/op_prior.json`
