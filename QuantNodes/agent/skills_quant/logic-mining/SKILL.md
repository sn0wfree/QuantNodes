---
name: logic-mining
description: 市场逻辑驱动的自动化因子挖掘 — Logic Mining + Γ 约束内层生成 + Wiki 沉淀。
---

# Logic Mining (v3.0.1)

驱动 QuantNodes Logic Mining 端到端工作流：选取 source → 3-stage LLM 抽取逻辑 → 编译 Γ 约束 → 内层 Alpha-GPT 受约束生成因子 → 外层重构 → 写入 Wiki。

## 工作流

1. **选 source 与定逻辑种子** — 调用 `sources.list_available_sources` 列数据源 (`alpha101` / `alpha158` / `alpha191`)，`sources.get_formulas_from_source(lib, max_count=N, only_volume_price=True)` 加载前 N 条公式
2. **3-stage 抽取逻辑** — 对每条公式:
   - 调用 `LogicMiningPipeline.run(formula, source_lib=lib)` 走 3 个 sub-agent (structure / semantics / abstraction)
   - 返回 `LogicAbstractionResult` 含 `structured_logic` (`WikiLogicStructured`)
3. **编译 Γ 约束** — 调用 `logic_mining.compile_to_constraint(logic.structured, source_logic=name)` 生成 `CompiledConstraint`
4. **注入 Alpha-GPT** — 在 `AlphaGptConfig.gamma = gamma` 透传；下游 `_step_formula_translator` 拒绝任何不符合 Γ 的公式
5. **内层循环** — 调用 `AlphaGptWorkflow(config=...).run()`，产出 inner_results 与 per-logic `LogicPerformanceEvidence`
6. **外层重构** — 调用 `MarketLogicGenerator.generate(library, current_logic, history, evidence, round_idx)` 与 `MarketLogicRefinementDirection.refine(...)` 进入下一轮 Algorithm 2 编排
7. **沉淀 Wiki** — 调用 `wiki_write` 写入 `Logic/` 页面，附 Γ + evidence

## 工具集

| 工具 | 用途 |
|------|------|
| `logic_mine` | `mine_logic_from_formula(formula, source_lib, llm_client)` 走 3-stage |
| `logic_compile` | `compile_to_constraint(logic, source_logic)` 生成 Γ |
| `logic_refine` | `MarketLogicRefinementDirection.refine(current_logic, history, evidence)` |
| `logic_validate` | `gamma.validate(formula_str)` 校验公式是否过 Γ |
| `logic_library_build` | `build_initial_logic_library(source_libs, llm_client, max_per_lib)` |
| `wiki_write` | 写入 Wiki Logic 页面 |

## 验收标准

- 初始逻辑库 ≥ 5 条 (从 alpha101 / alpha158 抽取)
- 每条逻辑最终 γ.validate(formula) 在示例上通过
- per-logic evidence 包含 `n_factors_explored` / `best_ir` / `best_ic`
- IR 上升 → 窗口收窄 20% (Phase 3 P0-3 修复)
- best_ic != best_ir (Phase 3 P0-2 修复)
- alpha191 source 不再静默 (Phase 3 P0-1 修复)
- silent fallback 通过 `pipeline.metrics` / `result.diagnostics` 暴露 (Phase 2)
- 可选 `strict=StrictConfig(...)` 把失败升级为异常

## 反模式

- 不要直接调用 inner AlphaGptWorkflow 而跳过 3-stage logic 抽取（失去 Logic Mining 意义）
- 不要忽略 `result.parse_error` / `result.diagnostics` 字段（silent fallback 信号）
- 不要用 `best_ic` 字段当 IR 读 (v3.0.0 时期旧 docstring 是错的, P0-2 已修)
- 不要在 IR 上升时反转 sign（这是 IR 下降的响应）
- 不要修改 alpha191 数据源逻辑，它是 OHLCV-only 守卫
