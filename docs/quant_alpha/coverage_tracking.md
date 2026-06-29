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

## Phase D.1 (2026-06-29)

| 文件 | Before | After | Δ |
|------|--------|-------|---|
| workflow/alpha_logics.py | 44% | **98%** | **+54%** |
| 整体 | 85.6% | **88%** | +2.4% |

**+22 测试** in test_alpha_logics_coverage.py (覆盖 run() / _build_initial_library() / _run_inner_loop() / _build_summary() / 懒加载 helpers)

## 进展

| Phase | 日期 | 整体 | 备注 |
|-------|------|------|------|
| Baseline | 2026-06-29 | ~30% | 旧 .coverage (evaluation only) |
| **Phase B** | 2026-06-29 | **85.6%** | 修 hanging + 全量测量 |
| **Phase D.1** | 2026-06-29 | **88%** | alpha_logics 44→98% |
| Phase D.2+ | TBD | 目标 90%+ | 补 5 个剩余文件 |

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
