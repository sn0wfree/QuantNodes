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

## 进展

| Phase | 日期 | Tier 1+2 覆盖 | Tier 3 覆盖 | 整体 | 备注 |
|-------|------|---------------|-------------|------|------|
| Baseline | 2026-06-29 | 54% (eval only) | 0% | ~30% | 旧 .coverage |
| Phase A | 2026-06-29 | (待测) | (待测) | (待测) | 修 hanging + 测 baseline |

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
