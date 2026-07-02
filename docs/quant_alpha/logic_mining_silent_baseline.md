# Logic Mining 静默容错基线报告 (v3.0.0 → v3.0.1)

**版本**: v1.0
**日期**: 2026-07-02
**作者**: Logic Mining Hardening Team
**状态**: Phase 1 (基线诊断) → Phase 2/3/4 (修复)
**分支**: `feat/logic-mining-harden`
**前置版本**: v3.0.0 (PyPI 发布于 2026-07-01)

---

## 1. 问题陈述

Logic Mining 子系统 (`QuantNodes/research/quant_alpha/logic_mining/` + 3 个 caller)
共存在 **24 个静默容错点**：异常被 `logger.warning` 吞掉，调用方仅得到默认值。
在真实 LLM 调用场景下，这些 silent fallback 会:
- 阻止开发者定位 prompt 故障 / 网络超时 / schema 漂移
- 让 mock 模式与真实模式无差别, 难以在 CI 捕获回归
- 给 E2E 调试提供极低信噪比

## 2. 静默容错点清单 (24 处)

按代码文件组织, 每个条目格式:

> **[ID]** `文件:行号`
> **当前代码**: <原始片段>
> **触发条件**: <何时触发>
> **降级行为**: <调用方得到什么>
> **修复目标**: <Phase 2-4 的处理>

### 2.1 `logic_mining/pipelines.py`

**[P-01]** `pipelines.py:67-72`  `_call_llm`
> **当前代码**:
> ```python
> try:
>     if hasattr(llm_client, "complete"):
>         return llm_client.complete(agent_id=agent_id, prompt=prompt)
>     return llm_client(prompt)
> except Exception as e:
>     logger.warning("LLM call failed for %s: %s, falling back to mock", agent_id, e)
>     return default_response
> ```
> **触发**: LLM 抛任意异常 (网络 / 限流 / JSON 截断)
> **降级**: 返回 mock 响应, 调用方察觉不到
> **修复**: `metrics.call_failures[agent_id] += 1`; strict=True 时 `raise LogicMiningStrictError`

**[P-02]** `pipelines.py:198-202`  Structure parse failure
> **当前代码**:
> ```python
> struct_result = parse_formula_structure(structure_resp)
> if not struct_result.ok:
>     logger.warning("FormulaStructureAgent parse failed: %s", struct_result.error)
>     result.formula_structure = {"operations": [], "window_length": 0,
>                                 "has_ranking": False, "has_normalization": False}
> ```
> **触发**: 3 层 JSON parser 全部失败
> **降级**: 默认 dict 注入下游 (下游误以为合法)
> **修复**: `result.parse_error = struct_result.error`; metrics 计数

**[P-03]** `pipelines.py:213-219`  Semantics parse failure
> **当前代码**:
> ```python
> sem_result = parse_financial_semantics(semantics_resp)
> if not sem_result.ok:
>     logger.warning("FinancialSemanticsMappingAgent parse failed: %s", sem_result.error)
>     result.financial_semantics = {
>         "price_role": "unknown", "volume_role": "unknown",
>         "time_pattern": "unknown", "behavior_interpretation": "unknown",
>     }
> ```
> **触发/降级/修复**: 同 P-02

**[P-04]** `pipelines.py:230-238`  Abstraction parse failure (注意: 整段)
> **当前代码**:
> ```python
> abs_result = parse_market_logic(abstract_resp)
> if not abs_result.ok:
>     logger.warning("MarketLogicAbstractionAgent parse failed: %s", abs_result.error)
>     return result  # 整段返回, structured_logic 仍为 None
>
> try:
>     result.structured_logic = _structured_from_dict(abs_result.data)
> except (KeyError, TypeError) as e:
>     logger.warning("Failed to build WikiLogicStructured: %s", e)
> ```
> **触发**: 三段式第 3 段 parse 失败, 或数据字段缺失
> **降级**: `result.structured_logic = None`, 后续 builder 把这类 result 跳过
> **修复**: `result.parse_error` / metrics 计数; strict=True 抛

**[P-05]** `pipelines.py:288-294`  `build_initial_logic_library` 整 formula try
> **当前代码**:
> ```python
> for f in formulas:
>     try:
>         result = pipeline.run(f["formula"], lib)
>         if result.structured_logic is not None:
>             results.append(result)
>     except Exception as e:
>         logger.warning("Failed to mine logic for %s: %s", f.get("id"), e)
> ```
> **触发**: 整个 pipeline.run() 抛任意异常
> **降级**: 公式被跳过, 不计入 metrics
> **修复**: metrics 捕获 + `result.parse_error`; strict=True 抛

### 2.2 `logic_mining/parser.py`

**[P-06]** `parser.py:67-70`  JSON 第 1 层
> **当前代码**:
> ```python
> try:
>     return ParseResult(ok=True, data=json.loads(raw), raw=raw)
> except json.JSONDecodeError:
>     pass
> ```
> **触发**: LLM 输出非严格 JSON
> **降级**: 静默进入下一层; 调用方看不到原始异常
> **修复**: `layer_reached = 1`, `last_error = str(e)`

**[P-07]** `parser.py:75-79`  JSON 第 2 层 (markdown fence)
> **触发/降级/修复**: 同 P-06

**[P-08]** `parser.py:82-87`  JSON 第 3 层 (任意 `{...}`)
> **触发/降级/修复**: 同 P-06

### 2.3 `logic_mining/generator.py`

**[P-09]** `generator.py:65-71`  `_call_llm` (与 P-01 镜像)
> 同 P-01, 只是 agent_id 不同

**[P-10]** `generator.py:211-216`  Refiner parse failure
> **当前代码**:
> ```python
> result = parse_json_response(raw)
> if not result.ok:
>     logger.warning("Refinement parse failed: %s, using mock", result.error)
>     data = json.loads(mock_response)
> else:
>     data = result.data
> ```
> **触发**: 同 P-02
> **降级**: 用 mock data 继续生成 (注意: 这里没有 structured = None 的退化!)

**[P-11]** `generator.py:212-216`  Generator parse failure (与 P-10 同模式)
> **当前代码**: 同上, 标志 message 不同
> **触发/降级/修复**: 同 P-10

**[P-12]** `generator.py:221-224`  `_structured_from_dict` 失败
> **当前代码**:
> ```python
> try:
>     structured = _structured_from_dict(data)
> except Exception as e:
>     logger.warning("Failed to build WikiLogicStructured: %s", e)
>     structured = None
> ```
> **触发**: 字段缺失/类型错
> **降级**: `WikiLogic.structured = None` 静默
> **修复**: metrics 计数; strict=True 抛

### 2.4 `logic_mining/compiler.py`

**[P-13]** `compiler.py:196-197`  `parse_op_args` value 解析失败
> **当前代码**:
> ```python
> for arg in args_str.split(','):
>     arg = arg.strip()
>     try:
>         nums.append(float(arg))
>     except ValueError:
>         pass  # 非数字 token 静默跳过
> ```
> **触发**: 公式参数中有非数字 token (常见: nested call)
> **降级**: 静默跳过, 调用方拿到的数字列表不完整
> **修复**: `logger.debug` 而非 silent pass; 必要 list 时返回 []

### 2.5 `workflow/alpha_logics.py`

**[P-14]** `alpha_logics.py:268-273`  `wiki.store_logic(h_best)` 失败
> **触发**: Wiki storage 抛错
> **降级**: 警告日志, h_best 不持久化

**[P-15]** `alpha_logics.py:286-291`  `wiki.store_logic(h_new)` 失败
> **触发/降级**: 同 P-14, 但针对新生成的 logic

**[P-16]** `alpha_logics.py:343-346`  `_build_initial_library` wiki 写失败
> **触发/降级**: 同 P-14

**[P-17]** `alpha_logics.py:382-392`  Inner loop workflow 失败
> **当前代码**:
> ```python
> try:
>     workflow = AlphaGptWorkflow(...)
>     alphagpt_result = workflow.run()
> except Exception as e:
>     logger.warning("Inner loop Alpha-GPT failed: %s", e)
>     alphagpt_result = None
> ```
> **触发**: AlphaGptWorkflow 整体崩溃
> **降级**: 外层继续, evidence 为 None
> **修复**: `diagnostics.inner_loop_failures += 1`; strict=True 抛

### 2.6 `logic_driven_pipeline.py`

**[P-18]** `logic_driven_pipeline.py:204-208`  best_logic 为空时静默退出
> **当前代码**:
> ```python
> if alphalogics_result.best_logic is None:
>     logger.warning("No best_logic found")
>     result.summary = {"error": "no_best_logic"}
>     return result
> ```
> **触发**: outer loop 未产出 best
> **降级**: 静默退出, 不冒泡

**[P-19]** `logic_driven_pipeline.py:215-230`  `best_logic.structured is None` 跳过 MCTS
> **当前代码**:
> ```python
> if best_logic.structured is None:
>     # 跳过 MCTS, 使用 raw inner_results
>     gamma = None
>     logic_driven_factors = [...]
> ```
> **触发**: 静默退出 MCTS, 不抛错
> **降级**: gamma 没注入, 用户察觉不到

**[P-20]** `logic_driven_pipeline.py:387-394`  `_persist_to_wiki` 单个 factor 失败
> **触发**: Wiki 写一个 factor 失败
> **降级**: log warning 跳过, 余下因子继续

## 3. P0 显式占位 & 简化 (3 处)

[P0-21] **alpha191 source 静默返回空** —
`sources.py:118-121` — `alpha191 not yet implemented, returning empty list`.
外部调用 `get_formulas_from_source("alpha191")` 静默返回 `[]`.
**修复**: 实现 12 条 OHLCV-only Alpha191 公式常量.

[P0-22] **`best_ic = best_ir` 代理** —
`workflow/alpha_logics.py:185` — `# 简化：暂用 IR 作为 IC proxy`. 下游若按 IC 解读会数据失真.
**修复**: `_compute_best_ic(alphagpt_result)` 取 `FactorMetrics.ic_mean.abs().max()`.

[P0-23] **IR 上升分支 `pass` 占位** —
`logic_mining/generator.py:271-274` — IR 上升时 `pass`, 仅在 IR 降时才反转 sign. 这是隐藏的不对称:
- IR 升 → "继续优化" (但什么都没做)
- IR 降 → 反转 sign

**修复**: IR 上升时窗口收窄 20% (基于现有 parameter_ranges 动), 保留 sign 不变.

## 4. 基线实测 (mock 模式, 0 LLM 调用)

| 测试 | 结果 |
|---|---|
| `pytest tests/quant_alpha/test_logic_mining.py -q` | 35 passed |
| `pytest tests/quant_alpha/test_logic_compiler.py -q` | 43 passed |
| `pytest tests/quant_alpha/test_logic_compiler_extended.py -q` | 31 passed |
| `pytest tests/quant_alpha/test_alpha_logics.py -q` | 17 passed |
| `pytest tests/quant_alpha/test_alpha_logics_coverage.py -q` | 22 passed |
| `pytest tests/quant_alpha/test_logic_driven_pipeline_coverage.py -q` | 22 passed |
| `pytest tests/quant_alpha/test_consistency_hook.py -q` | 17 passed |
| `pytest tests/quant_alpha/test_bug_regression.py -q` | 12 passed |
| `pytest tests/quant_alpha/test_6_logic_smoke.py -q` | 18 (3 parametrized × 6 logic) passed |
| **pytest 计数小计** | **217 passed (mock 模式)** |
| `python3.11 tests/quant_alpha/test_gamma_integration.py` | 非 pytest (4 个 print-based test 函数), 当前通过 |
| `python3.11 tests/quant_alpha/large_scale_e2e_test.py` | 需要 API key, mock 模式部分通过 |

### 4.1 已知 mock 触发的 silent fallback 路径

由于 mock 路径下 `llm_client=None`, [_call_llm] 总是在 try 块前 return mock,
下列 silent fallback 在 mock 模式下**不会触发**:
- P-01/P-09 (`_call_llm` 的 except 块)

以下路径会在 mock 模式被触发:
- P-02/P-03/P-04 (当 mock JSON 不合法时)
- P-10/P-11 (generator parse failure)
- P-12 (structured from dict 失败)
- P-13 (parse_op_args 非数字 token)
- P-14/P-15/P-16 (wiki store 失败 — 需要 mock wiki)
- P-17 (inner loop 失败 — 需要 mock workflow)

**Phase 2 测试覆盖目标**: 24/24 silent fallback 路径均能被显式测试触发并验证 metrics.

## 5. 设计-实现偏差 (1 处)

**[D-DOC-01]** 5 个 nanobot agent markdown 应在 `.agent/agents/` (`docs/32-…:328`), 当前缺失, 代码用内联 f-string 替代 (`pipelines.py::_build_*_prompt`).

**行为**: 不影响功能 (f-string 在 spawn context 替换). 偏差原因已被 design 文档解释 (避免 logic_mining 强依赖 nanobot).

**Phase 4 修复**: 在 `feat/logic-mining-harden` 第二批 PR (`feat(agent): nanobot subagent bindings`) 补 5 个 md + 1 个 SKILL.md.

## 6. 修复路线图

| 阶段 | 内容 | 总改动行 | 测试新增 |
|---|---|---|---|
| **Phase 1** | 本报告 | +350 | 0 |
| **Phase 2** | metrics.py + 24 silent fallback 接入 + `strict` 开关 | ~280 | ~250 |
| **Phase 3** | alpha191 + best_ic + IR 上升分支 | ~80 | ~80 |
| **Phase 4** | 5 个 .md + SKILL | ~750 | ~50 |
| **Phase 5** | CHANGELOG v3.0.1 + 版本 bump + 发布 | +120 | 0 |
| **小计** | — | ~1580 | ~380 |

## 7. 验收矩阵 (Phase 2/3/4 完成时)

- [x] 基线报告文件存在 (`docs/quant_alpha/logic_mining_silent_baseline.md`)
- [ ] `PipelineMetrics.to_dict()` 包含 6 类计数
- [ ] `strict=True` 时 24 silent fallback 升级为 `LogicMiningStrictError`
- [ ] `get_formulas_from_source("alpha191")` ≥ 3 条
- [ ] `_compute_best_ic` 与 IR 解耦
- [ ] `MarketLogicGenerator._mock_generate_response` IR 上升分支窗口收窄
- [ ] `.agent/agents/logic-mining-*.md` × 3 + `.agent/agents/market-logic-*.md` × 2 存在
- [ ] `.agent/skills_quant/logic-mining/SKILL.md` 存在并可 import
- [ ] 现有 217 个 quant_alpha 测试零回归
- [ ] 新增测试 ≥ 30 个 全过

## 8. 不在范围内 (out-of-scope)

- `extract_operators` / `parse_op_args` 的 regex 重写为真 AST (P3 大型重构)
- 引入真向量存储/RAG (不现实, 见设计 P3 备注)
- `alpha_gpt.py::except Exception` 也在内, 但不在 logic_mining 范畴, 后续单独 PR
- `mcts/search.py` 等 mcts 容错亦非本次范围

## 9. 引用

- 设计文档: `docs/32-市场逻辑驱动因子挖掘设计.md` (PR-3, PR-2)
- 报告: `docs/quant_alpha/dedup_fix_report.md` (V2 后 4 逻辑 pvd/mr 0 因子案例)
- 报告: `docs/quant_alpha/explanation_truncation_fix.md` (V5 截断案例, 4 层防御模式可参考)
- 测试基线: `docs/quant_alpha/TESTING.md`, `docs/quant_alpha/coverage_tracking.md`
