# 因子挖掘实验结果索引 (V1-V8)

## 总览

| 版本 | 逻辑数 | 因子数 | 最佳 \|IR\| | 耗时 | 关键变更 |
|------|--------|--------|-------------|------|----------|
| V2 | 4 | 6 | - | - | LogicDrivenPipeline 端到端 (PR-1) |
| V3 | 4 | 6 | - | - | 多逻辑因子挖掘 (momentum + volatility) |
| V4 | 4 | 9 | 0.1208 | 522s | 4-logic E2E baseline (no thinking chain) |
| V5 | 4 | 3 | 0.1167 | 555s | 4-logic E2E with Tier 1+2+4 (thinking chain) |
| V6 | 4 | 9 | 0.1284 | 576s | 4-logic E2E with 4-layer defense (P0+P1+P2+P3) |
| V7 | 4 | 9 | 0.1208 | 622s | 4-logic E2E with volume alias fix (commit 8147a94) |
| **V8** | **6** | **15** | **0.1596** | **861s** | 6-logic E2E (4 old + 2 new) |

## V8 详细结果 (6 logic)

| Logic | 类型 | 因子数 | best \|IR\| | 耗时 | 状态 |
|-------|------|--------|-------------|------|------|
| price_volume_divergence | 旧 | 0 | 0.000 | 141s | 弱逻辑, A股价量关系不显著 |
| mean_reversion | 旧 | 3 | 0.0610 | 141s | - |
| momentum | 旧 | 3 | 0.0610 | 73s | - |
| volatility | 旧 | 3 | 0.1208 | 132s | - |
| **trend_breakout** | **新** | **3** | **0.1596** ⭐ | **108s** | 历史最高, 价量共振突破 |
| **intraday_reversal** | **新** | **3** | **0.1103** | **266s** | 全部正 IR (sign mismatch) |

## 关键里程碑

### 架构演进
- **V1-V3**: LogicDrivenPipeline, AlphaLogicsWorkflow
- **V4**: AlphaPipeline 端到端
- **V5**: Thinking-Chain (Tier 1 capture + Tier 2 structured + Tier 4 OpPrior)
- **V6**: 4-layer defense (P0 截断恢复 / P1 explanation 截断 / P2 idea_id 容错 / P3 字段重命名)
- **V7**: vol/volume alias 修复 (commit 8147a94)
- **V8**: +2 新 logic (trend_breakout / intraday_reversal)

### Bug 发现 & 修复
- **V4-V7 pvd 始终 0 因子** (vol/volume 列名不匹配) → 修于 V7 (commit 8147a94)
- **V5 thinking-chain 回归** (mean_reversion/volatility 失败) → 修于 V6 (4-layer defense)
- **V8 sign-mismatch** (intraday_reversal 3 因子全正 IR) → 计划在 Phase 1 修 (`check_sign_hint`)

## 当前状态 (V8 后)

- **暂停原因**: 系统"老是出问题", 决定先完善单元测试再继续挖掘
- **暂停时间**: 2026-06-29
- **下一步**: 执行 test/expand-coverage-2x 分支的 10 阶段测试强化计划
- **重启条件** (V9):
  1. Phase 1-10 全部完成 (测试 116 → 250+)
  2. 关键 bug 修复 (check_sign_hint 等)
  3. 红→绿回归测试就位 (V4-V8 已知 bug 全部防住)
  4. CI 集成完成

## 相关文档

- `v4_baseline_report.md` - V4 baseline
- `v5_vs_v5_comparison.md` - V5 thinking-chain 实验对比
- `v6_vs_v5_comparison.md` - V6 4-layer defense
- `v7_vs_v6_comparison.md` - V7 vol/volume 修复
- `v8_vs_v7_comparison.md` - V8 6-logic 拓展
- `thinking_chain_design.md` - Tier 1+2+4 设计
- `explanation_truncation_fix.md` - 4-layer defense 设计
- `v8_paused_for_testing.md` - 暂停说明

## 性能指标

| 指标 | V4 | V5 | V6 | V7 | V8 | 趋势 |
|------|----|----|----|----|----|------|
| 总因子 | 9 | 3 | 9 | 9 | 15 | +67% V4→V8 |
| 最佳 \|IR\| | 0.1208 | 0.1167 | 0.1284 | 0.1208 | 0.1596 | +32% V4→V8 |
| 0 因子 logic | 1 (pvd) | 2 (mr/vol) | 1 (pvd) | 1 (pvd) | 1 (pvd) | 改善 |
| 耗时/逻辑 | 130s | 139s | 144s | 156s | 144s | 稳定 |
| LLM 调用成功率 | - | 67% | 100% | 100% | 100% | 修复后稳 |
