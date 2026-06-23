# PR: M1 — QuantAlpha OperatorVocab (Issue: Factor Mining Enhancement)

> **Status**: Draft → Ready for review
> **Target branch**: `feat/nanobot-upgrade` → `main` (or `dev`)
> **M1 PR in 8-week QuantAlpha plan** ([`docs/quant_alpha/PROJECT_PLAN.md`](../../docs/quant_alpha/PROJECT_PLAN.md))
> **Date**: 2026-06-23

## Summary

新增 `QuantAlpha` 自动化因子挖掘子包。**M1 (OperatorVocab)** 是 6 条路线的基础设施，统一 162 个算子（vs 旧 12-lambda），修复 3 个 latent bug，开启方案 C 渐进合并的 Phase A。

## What's in this PR

### ✨ 新增（QuantAlpha 子包）

- **`QuantNodes/research/quant_alpha/`** 子包：
  - `operator_vocab/vocabulary.py` — `OperatorVocab` 主类
  - `operator_vocab/metadata.py` — 12 字段 `OperatorMetadata` schema
  - `operator_vocab/config.py` — `OperatorVocabConfig`
  - `operator_vocab/__init__.py` + `__init__.py` + `README.md` + `CHANGELOG.md`
- **5 个新算子**（Alpha 101 关键缺口）：
  - `signedpower(x, a)` = `sign(x) * abs(x) ** a`
  - `ts_decay_linear(x, d)` 别名
  - `IndNeutralize(x, ind_class)` 别名
  - `ts_skew(x, w)` 别名
  - `ts_kurt(x, w)` 别名
- **算子元数据 schema 扩展**：5 字段 → 12 字段（含 7 个 LLM 友好字段）

### 🐛 修复（3 个 latent bug）

`QuantNodes/research/factor_evaluator.py:202-215` 的 12-lambda namespace：

1. **`ts_corr` / `ts_cov` API 不存在** → 改用 L0 注册表 `rolling_corr` (Expr API)
2. **`rank` / `zscore` / `winsorize` 全局而非 per-date** → 默认 `cross_sectional=True` per-date over(date)
3. **异常被静默吞掉** → 完整错误抛出（不静默 `return None`）

### ⚠️ DeprecationWarning（Phase A）

4 个旧文件 import-time `DeprecationWarning`：
- `QuantNodes.research.factor_evaluator` → `quant_alpha.operator_vocab.OperatorVocab`
- `QuantNodes.research.factor_miner` → `quant_alpha` (M2+)
- `QuantNodes.research.mcts_search` → `quant_alpha.mcts.MCTSSearch` (M2 PR)
- `QuantNodes.research.auto_researcher` → `quant_alpha.AutoResearcher` (M5+ PR)

### 📚 文档

- `docs/quant_alpha/PROJECT_PLAN.md`（991 行，调研 + 规划）
- `docs/quant_alpha/migration.md`（旧 API → 新 API 完整映射）
- `CHANGELOG.md`（顶层 Unreleased 段）

## 关键数字

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 算子可用数 | 12 个 | **162 个**（+13.5×）|
| 元数据字段 | 5 个 | **12 个**（+7 字段 LLM 友好）|
| Latent bug | 3 个 | **0 个** |
| 新算子（Alpha 101 必需）| 0 个 | **5 个** |
| per-date 截面语义 | 全局（错误）| **正确**（per-date over(date)）|
| 异常处理 | 静默吞掉 | 完整抛出 |
| 旧 12-lambda 兼容 | N/A | `cross_sectional=False` 开关 |

## 测试

- **60 个新测试**（`tests/quant_alpha/test_operator_vocab.py`），覆盖：
  - `TestOperatorVocabBasics`（10 测试）：单例、stats、list、get、get_metadata
  - `TestNewOperators`（6 测试）：5 个新算子正确性
  - `TestPerDateOverFix`（5 测试）：per-date 修复验证
  - `TestAlpha101Formulas`（4 测试）：Alpha 101 #1/#6/#12/#101 公式
  - `TestErrorHandling`（5 测试）：错误抛出 + 长度/深度限制
  - `TestConvenienceFunctions`（5 测试）：模块级便捷函数
  - `TestMetadataInference`（8 测试）：12 字段自动推断
  - `TestLegacyCompatibility`（12 测试）：旧 12-lambda 兼容
  - `TestDeprecationWarnings`（5 测试）：4 旧文件 warning 触发
- **回归测试**：2777 passed（research + quant_alpha + factor_node + operators + core），**6 skipped**，**0 failed**

## Phase 时间表

| 阶段 | 状态 | 兼容性 |
|------|------|--------|
| **Phase A** (本 PR) | ✅ 旧 4 文件可用 + warning | 100% 向后兼容 |
| **Phase B** (M5+ 后) | 旧类变 thin wrapper | 100% 向后兼容 |
| **Phase C** (v3.0) | 旧实现归档 `_legacy_3c/` | 破坏性变更 |

## 相关链接

- 完整规划：[`docs/quant_alpha/PROJECT_PLAN.md`](../../docs/quant_alpha/PROJECT_PLAN.md)
- 迁移指南：[`docs/quant_alpha/migration.md`](../../docs/quant_alpha/migration.md)
- 子项目 README：[`QuantNodes/research/quant_alpha/README.md`](../../QuantNodes/research/quant_alpha/README.md)

## Checklist

- [x] OperatorVocab 主类实现（list / get / get_metadata / build_namespace / evaluate / stats）
- [x] 12 字段 OperatorMetadata schema（含 7 个 LLM 友好字段）
- [x] 5 个新算子注册到 L0 注册表
- [x] per-date over() 修复（rank / zscore / winsorize / IndNeutralize）
- [x] Expr 输入支持（用 `data.select(expr).to_series()` 物化）
- [x] 4 个旧文件 DeprecationWarning
- [x] migration.md（完整 API 映射 + 行为对比）
- [x] CHANGELOG.md（顶层 Unreleased 段）
- [x] 60 个新测试通过
- [x] 2777 个旧测试零失败

## Next Steps

- M2 PR（路线 7）：MCTS + 5 通道反馈 + 谱系追踪
- M3 PR（路线 1+2 借鉴）：alpha101/158 设计文档 + few-shot 示例
- M4 PR（路线 4）：PolarsAlphaCalculator 适配器
- M5+ PR（路线 6）：Alpha-GPT 完整工作流
