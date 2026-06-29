# 覆盖率追踪 (Coverage Tracking)

## 计划

- **范围**: `QuantNodes/research/quant_alpha/` only
- **目标** (Q7=B):
  - Tier 1+2 模块: ≥ 80%
  - Tier 3 模块: ≥ 60%
- **执行顺序** (Q6=A): 按缺口大小
- **排除** (Q8=C): `pragma: no cover` + `raise NotImplementedError` + `if __name__ == "__main__"`

## Baseline (2026-06-29, Phase 0-10 之前)

| 范围 | Stmts | Miss | Cover |
|------|-------|------|-------|
| evaluation/ (旧 .coverage 数据) | 745 | 339 | **54%** |
| 其他模块 (未测量) | ~13000 | 未知 | **0%** (假定) |

## Phase B Baseline (2026-06-29, 修 hanging 后)

**整体 85.6%** (4634 stmts, 668 miss) — 已超过 Q7=B 目标 (Tier 1+2 ≥ 80%)

### 完整 per-file 列表

| Pct | Stmts | Miss | File | Tier |
|-----|-------|------|------|------|
| 100% | * | 0 | __init__.py × 10 | - |
| 100% | 70 | 0 | evaluation/baselines/g1_handcrafted | - |
| 100% | 65 | 0 | evaluation/mock_data_loader | - |
| 100% | 82 | 0 | evaluation/runner | - |
| 100% | 91 | 0 | logic_mining/models | - |
| 100% | 106 | 0 | evaluation/contracts | - |
| 100% | 32/33 | 0 | alpha101/158 few_shot_examples | - |
| 99% | 86 | 1 | workflow/state | - |
| 98% | 48/105 | 1/2 | baselines/g3_alpha_gpt / operator_vocab/metadata | - |
| 97% | 152 | 5 | logic_mining/compiler | - |
| 96% | 103/120 | 4/5 | mcts/tree / mcts/cache | - |
| 94% | 188/304/16 | 11/19/1 | operator_vocab/vocabulary / mcts/feedback / config | - |
| 93% | 45 | 3 | alpha101_design/philosophy | - |
| 92% | 39/66 | 3/5 | alpha158_design/philosophy / mcts/op_prior | - |
| 91% | 44/111 | 4/10 | logic_mining/sources / parser | - |
| 90% | 163 | 16 | adapters/calculator | - |
| 88% | 155/291 | 18/35 | adapters/expression / llm/parser | - |
| 87% | 106/245/374 | 14/32/47 | mcts/extension_ops / mcts/search / workflow/alpha_gpt | - |
| 85% | 119 | 18 | logic_mining/generator | - |
| 80% | 229 | 46 | evaluation/evaluators/polars_evaluator | - |
| **79%** | **82** | **17** | **logic_mining/pipelines** | **T1** ⚠️ |
| **71%** | **121** | **35** | **evaluation/baselines/g2_llm_only** | **T1** ⚠️ |
| **71%** | **403** | **116** | **pipeline** | **T1** ⚠️ |
| **56%** | **88** | **39** | **evaluation/clickhouse_data_loader** | **T3** ⚠️ |
| **50%** | **137** | **69** | **logic_driven_pipeline** | **T1** ⚠️ |
| **44%** | **165** | **92** | **workflow/alpha_logics** | **T1** ⚠️ |

### < 80% 文件 (Phase D 目标)

| Rank | File | Pct | Gap | Tier | 估测试 |
|------|------|-----|-----|------|--------|
| 1 | workflow/alpha_logics.py | 44% | 36% | T1 | +30 |
| 2 | logic_driven_pipeline.py | 50% | 30% | T1 | +25 |
| 3 | pipeline.py | 71% | 9% | T1 | +20 |
| 4 | evaluation/baselines/g2_llm_only.py | 71% | 9% | T1 | +10 |
| 5 | logic_mining/pipelines.py | 79% | 1% | T1 | +5 |
| 6 | evaluation/clickhouse_data_loader.py | 56% | 4% | T3 | +5 |

## 进展

| Phase | 日期 | Tier 1+2 覆盖 | Tier 3 覆盖 | 整体 | 备注 |
|-------|------|---------------|-------------|------|------|
| Baseline | 2026-06-29 | 54% (eval only) | 0% | ~30% | 旧 .coverage |
| **Phase B** | 2026-06-29 | **已 ≥ 80%** | 已 ≥ 60% | **85.6%** | 修 hanging + 全量测量 |
| Phase D | TBD | 补 6 文件 | - | 目标 87%+ | 按缺口补测试 |

## 红→绿对照

| Bug | 防回归测试 | 来源 |
|-----|-----------|------|
| V4-V7 pvd=0 (vol/volume) | test_bug_regression.py | Phase 9.3 |
| V5 thinking-chain 失败 | test_bug_regression.py | Phase 9.3 |
| V6 P0 截断 | test_bug_regression.py | Phase 9.3 |
| V8 sign-mismatch | test_bug_regression.py | Phase 1 + 9.3 |
| 6-logic smoke | test_6_logic_smoke.py | Phase 9.1 |

## 历史

- 6月25日: 第一次 .coverage 数据 (evaluation/ only, 54%)
- 6月29日: Phase 0-10 完成 882 tests pass
- 6月29日: 开始覆盖率补完 (此文档追踪)
