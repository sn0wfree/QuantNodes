# V8 暂停说明 (Paused for Testing)

## 暂停时间
2026-06-29

## 暂停原因
因子挖掘系统"老是出问题" (things keep breaking)，从 V1 到 V8 共发生以下主要故障：

### 历史 Bug 时间线
| 版本 | Bug | 影响 | 隐藏多久 |
|------|-----|------|---------|
| V4 | 无 thinking-chain 利用 | 仅 baseline | - |
| V5 | thinking-chain 集成导致 mean_reversion / volatility 失败 | 0 因子 → 0 因子, 退化 | 1 个版本 |
| V4-V7 | `vol` vs `volume` 列名不匹配, price_volume_divergence 一直 0 因子 | 1 个 logic 永远失败 | **3 个版本** |
| V6 | LLM 截断 JSON + thinking + 重写 JSON 模式被误判 | 解释字段膨胀 | 2 个版本 |
| V8 | `check_sign_hint` 宽松兜底, sign_constraint=-1 接受全正 IR 公式 | 3 个 intraday_reversal 因子 sign-mismatch | 1 个版本 |

### 根本问题
测试覆盖不足:
- 116 个测试 vs 162 个算子 + 50+ 业务方法
- 关键集成路径 (thinking-chain, multi-round pipeline) **0 测试**
- 关键 bug 路径 (sign_hint 宽松兜底) **0 测试**
- 端到端 smoke test (任何 logic 都应该跑出 ≥1 因子) **0 测试**

## 重启计划

**目标**: 测试 116 → 250+, 系统稳定性达到"改一个地方不会破坏其他地方"。

### 10 阶段计划 (test/expand-coverage-2x 分支)

| Phase | 目标 | 测试增量 | 累计 |
|-------|------|---------|------|
| 0 | 记录 V8 + 暂停标记 | 0 | 0 |
| 1 | 修 sign_hint bug + 红测试 | +6 | 6 |
| 2 | Operator 全面测试 (hypothesis) | +40 | 46 |
| 3 | LLM 失败模式 | +25 | 71 |
| 4 | Evaluator edge cases | +20 | 91 |
| 5 | MCTS 完整覆盖 | +30 | 121 |
| 6 | Pipeline + EarlyStopping | +18 | 139 |
| 7 | Workflow thinking-chain 集成 | +18 | 157 |
| 8 | Logic compiler 完整 | +14 | 171 |
| 9 | 新文件 集成 + 回归 | +50 | 221 |
| 10 | 修 e2e + CI | +5 | 226 |

### 重启 V9 条件

1. ✅ 所有 10 阶段完成
2. ✅ 116+226 = 342+ 测试全部 pass
3. ✅ 关键 bug 修复 (sign_hint, vol/volume 等)
4. ✅ 红→绿回归测试就位:
   - V4-V7 pvd=0 → smoke 测试
   - V5 thinking-chain regression → integration 测试
   - V6 4-layer defense → truncation 测试
   - V7 vol/volume → alias 测试
   - V8 sign-mismatch → sign_hint 测试
5. ✅ CI 集成 (pytest 5 min 内完成)
6. ✅ `docs/quant_alpha/TESTING.md` 文档完成

## 重启后方向 (V9+)

可选方向 (按优先级):
1. **多轮迭代** (max_rounds=2-3) — 用 feedback 改进 LLM 输出
2. **加更多 logic** (e.g. value, gap_reversal, turnover_quality)
3. **降低 IR 阈值** 挖 pvd 弱信号
4. **针对 best factor 做组合** (orthogonal pool)
5. **真实数据回测** (现有 data 只做 IC/IR, 没回测组合)

## 当前分支

- `master` - 稳定, 含 V8 结果
- `test/expand-coverage-2x` - 测试强化进行中

## 相关 Issues / 决策

- 用户决策 (2026-06-29):
  - 决策 1: 阶段顺序 0→1→2→...→10
  - 决策 2: 修 sign_hint bug 后接受 V8 intraday_reversal 失去 2 个正 IR 因子 (数据正确性优先)
  - 决策 3: 测试运行 < 5 分钟 (含 integration)
  - 决策 4: 使用 hypothesis 库做 property-based 测试
