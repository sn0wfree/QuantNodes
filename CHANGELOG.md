# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.2] - 2026-07-02 — Automated Factor Mining CLI

**自动化因子挖掘闭环**: `quantnodes mine-logics` CLI + `FactorPool` 池抽象 + 离线报告生成器。
并发批处理 (ThreadPoolExecutor) + 幂等重跑 (跳过 wiki 已有 Logic pages) + 离线/真实双模式。

### Added

- **`quant_nodes/research/quant_alpha/factor_pool.py`** (NEW, ~320 lines)
  - `FactorEntry` — 因子池单条记录 (formula_id / formula / source_lib / ir / ic_mean / rank_ic / tags / structured / evidence)
  - `FactorPool` — 线程安全 in-mem 因子池
    - CRUD: `add` / `extend` / `remove` / `get` / `contains` / `clear`
    - 操作: `dedup(by=formula_id/formula/source_id)` / `select(top_n, by=ir)` / `filter(source_lib, min_ir, tags)`
    - Wiki 双向: `from_wiki(proxy)` / `to_wiki(proxy)` — 与 `WikiFactorProxy` 同步
    - 持久化: `save_json(path)` / `load_json(path)` — 离线 JSON 序列化
    - 统计: `summary()` → `{n_total, by_source_lib, ir_stats, n_with_wiki}`

- **`logic_mining/batch.py`** (NEW, ~320 lines)
  - `mine_logic_library_v2(source_libs, llm_client, max_per_lib, workers, wiki_path, ...)` — 并发批量挖掘入口
  - `LogicMiningBatchResult` — 结果汇总 (n_mined/n_skipped/n_failed/wall_clock_s/warnings)
  - `ThreadSafeMetrics` — `PipelineMetrics` 跨线程包装 (所有 `record_*` 加 `threading.Lock`)
  - 幂等性: 启动时 `from_wiki()` 预加载 → 跳过已存在 Logic pages
  - 进度回调: `on_progress(done, total, current_id)`
  - strict 模式: `LogicMiningStrictError` 上抛而非静默

- **`logic_mining/report.py`** (NEW, ~200 lines)
  - `MetricsReportBuilder` — 从 `LogicMiningBatchResult` 构建报告
  - `to_dict()` / `to_json(path)` / `to_markdown()` — 三种输出格式
  - 报告内容: Summary + Source breakdown + Agent stats + Failed IDs + Warnings

- **`cli/commands/mine_logics.py`** (NEW, ~220 lines)
  - `MineLogicsCommand` — `quantnodes mine-logics` 子命令
  - 7 个参数: `--source-libs` / `--max-per-lib` / `--workers` / `--wiki-path` / `--output-dir` / `--live` / `--strict`
  - 退出码: 0=全成功 / 1=部分失败 / 2=致命
  - 离线默认 (NullLLMClient), `--live` 显式开启真实 LLM

- **`docs/quant_alpha/automated_factor_mining.md`** — 使用文档 (架构 / 快速开始 / CLI 参数 / 幂等性 / 并发模型)
- **`examples/mine_logics_demo.py`** — 离线演示脚本

### Changed

- `logic_mining/__init__.py` — 导出 `mine_logic_library_v2` / `ThreadSafeMetrics` / `LogicMiningBatchResult` / `MetricsReportBuilder`
- `quant_alpha/__init__.py` — 导出 `FactorEntry` / `FactorPool`
- `cli/commands/__init__.py` — 注册 `MineLogicsCommand` 到 `COMMAND_REGISTRY`

### Tests

- `tests/quant_alpha/test_factor_pool.py` — 33 tests (CRUD / dedup / select / filter / Wiki mock / JSON / 并发)
- `tests/quant_alpha/test_mine_logic_batch.py` — 21 tests (ThreadSafeMetrics / BatchResult / idempotency / concurrency / strict)
- `tests/quant_alpha/test_metrics_report.py` — 15 tests (from_batch / to_dict / to_json / to_markdown)
- `tests/quant_alpha/test_mine_logics_cli.py` — 8 tests (args parsing / exit codes / file outputs)

### Migration Guide

`mine_logic_library_v2` 是新增 API，不影响现有 `build_initial_logic_library` (v1)。
新代码推荐使用 `mine_logic_library_v2` 获得并发 + 幂等 + 进度回调。

```python
# 旧 API (仍可用)
from QuantNodes.research.quant_alpha.logic_mining import build_initial_logic_library
results = build_initial_logic_library(source_libs=("alpha101",), llm_client=client)

# 新 API (v3.0.2)
from QuantNodes.research.quant_alpha.logic_mining import mine_logic_library_v2
batch = mine_logic_library_v2(source_libs=["alpha101"], llm_client=client, workers=4)
batch.pool.select(top_n=5)  # FactorPool
```

---

## [3.0.1] - 2026-07-02 — Logic Mining 健壮性补丁

Logic Mining (`QuantNodes/research/quant_alpha/logic_mining/`) 子系统 v3.0.0 发布后修复:
**24 个 silent fallback 点接入可观测性 + 3 个 P0 占位实装 + 5 个 nanobot subagent md 补位**。
基线报告: `docs/quant_alpha/logic_mining_silent_baseline.md`。

### Added

- **`logic_mining/metrics.py`** (NEW, 159 lines)
  - `PipelineMetrics` — 6 类计数器 (call_failures / parse_failures / parse_layer_reached
    / structured_failures / wiki_failures / inner_loop_failures)
  - `StrictConfig(call, parse, structured)` — 三挡 strict 开关 (默认全 False, 保持向后兼容)
  - `LogicMiningStrictError(kind, **context)` — strict 模式下升级为异常
  - `PipelineMetrics.to_dict()` / `total_failures()` 可序列化 + 求和
- **`AlphaLogicsDiagnostics` dataclass** (`workflow/alpha_logics.py`)
  - per-round wiki/inner failures 计数 (`by_round_wiki_failures` / `by_round_inner_failures`)
  - `strict_raised` + `strict_raised_messages` 暴露 strict 异常冒泡
- **`LogicAbstractionResult.parse_error` + `parse_layer` 字段**
  - 三段式任一阶段失败时, 调用方可读 `result.parse_error` + 最远触达 layer
- **`_compute_best_ic(alphagpt_result)` helper** (Phase 3 P0-2)
  - 取 `FactorMetrics.ic_mean.abs().max()`, 与 IR 真实解耦
- **`AlphaLogicsConfig.metrics` / `strict` 字段** (Optional, 透传到所有 sub-component)
- **`LogicDrivenPipelineConfig.metrics` / `strict` 字段** (同上)
- **5 个 nanobot subagent markdown** (`.agent/agents/`)
  - `logic-mining-structure.md` (Stage 1, 122 行)
  - `logic-mining-semantics.md` (Stage 2, 132 行)
  - `logic-mining-abstraction.md` (Stage 3, 156 行)
  - `market-logic-generator.md` (外层, 141 行)
  - `market-logic-refinement.md` (外层, 117 行)
- **`SKILL.md`** (`QuantNodes/agent/skills_quant/logic-mining/SKILL.md`)
  - YAML frontmatter (name=logic-mining)
  - 6 步工作流 + 验收标准 + 反模式

### Fixed

- **silent fallback observability** (24 处全接入 metrics)
  - `logic_mining/pipelines.py:62-93` `_call_llm` (P-01)
  - `logic_mining/pipelines.py:198-238` 3 个 stage parse failure (P-02/03/04)
  - `logic_mining/pipelines.py:289-294` `build_initial_logic_library` outer try (P-05)
  - `logic_mining/parser.py:67-89` 3 层 JSONDecodeError silent pass (P-06/07/08) →
    `ParseResult.layer_reached` / `last_error` / `layer_errors`
  - `logic_mining/generator.py:65-71` `_call_llm` (P-09)
  - `logic_mining/generator.py:212-216` parse failure (P-10/11)
  - `logic_mining/generator.py:222-224` `_structured_from_dict` 失败 (P-12)
  - `logic_mining/generator.py:274` IR-improving branch `pass` 占位 →
    真实实现: 窗口收窄 20% (Phase 3 P0-3)
  - `workflow/alpha_logics.py:272/290/346` wiki.store_logic 失败 (P-14/15/16) →
    `diagnostics.wiki_failures` 计数
  - `workflow/alpha_logics.py:390` inner loop 失败 (P-17) → `diagnostics.inner_loop_failures`
- **`alpha191` source 实现** (Phase 3 P0-1)
  - `logic_mining/sources.py:118-121` 占位 `not yet implemented, returning empty list` 替换
  - 新增 `ALPHA191_OHLCV_FORMULAS` 18 条 OHLCV-only 公式 (与 alpha101 范式重叠)
  - `get_formulas_from_source("alpha191")` 现在返回 ≥ 18 条; 不再静默
- **`best_ic` 解耦** (Phase 3 P0-2)
  - `workflow/alpha_logics.py:185` `best_ic=float(best_ir)  # 简化: 暂用 IR 作为 IC proxy`
    替换为 `_compute_best_ic(alphagpt_result)` — 取 `FactorMetrics.ic_mean.abs().max()`
  - 下游读取 `best_ic` 不再被 IR 误导

### Changed

- **`ParseResult` 字段扩展** (从 4 → 7 字段)
  ```python
  # Before (v3.0.0)
  @dataclass
  class ParseResult:
      ok: bool
      data: Optional[Dict[str, Any]] = None
      error: Optional[str] = None
      raw: str = ""

  # After (v3.0.1)
  @dataclass
  class ParseResult:
      ok: bool
      data: Optional[Dict[str, Any]] = None
      error: Optional[str] = None
      raw: str = ""
      layer_reached: int = 0       # 0=empty, 1=direct, 2=md_fence, 3=brace
      last_error: Optional[str] = None       # most recent JSONDecodeError
      layer_errors: Dict[int, str] = field(default_factory=dict)   # per-layer errors
  ```
  - 全为 Optional / 有 default, **backward-compatible**

- **`LogicAbstractionResult` 字段扩展** (新增 2 字段)
  ```python
  # Before (v3.0.0)
  structured_logic: Optional[WikiLogicStructured] = None

  # After (v3.0.1)
  structured_logic: Optional[WikiLogicStructured] = None
  parse_error: Optional[str] = None     # 失败原因, 全成功时 None
  parse_layer: int = 0                 # 最远触达 layer (1/2/3), 默认 0
  ```
  - `to_dict()` 自动包含新字段; old field 完整保留

- **`MarketLogicGenerator._mock_generate_response` 重写**
  ```python
  # Before (v3.0.0) — IR-improving branch 是空 pass
  if evidence and len(evidence) >= 2:
      if evidence[-1].best_ir > evidence[-2].best_ir:
          pass  # ← 占位
      else:
          sign = -sign if sign else 1

  # After (v3.0.1) — IR 升时窗口收窄 20%, 保留 sign
  if evidence and len(evidence) >= 2:
      cur, prev = evidence[-1], evidence[-2]
      if cur.best_ir > prev.best_ir and cur.n_factors_explored > 0:
          # 窗口收窄 20%, 保留 sign
          for op_key in list(param_ranges.keys()):
              lo, hi = param_ranges[op_key]
              if hi - lo > 1e-9:
                  span = hi - lo
                  param_ranges[op_key] = [lo + span * 0.2, hi - span * 0.2]
      else:
          sign = -sign if sign else 1
  ```

### Migration Guide (v3.0.0 → v3.0.1)

**100% backward-compatible**. 现有代码无需任何修改即可升级. 可选地, 推荐升级到 v3.0.1 的可观测性 API:

#### 1) 在生产代码中读取 silent fallback 计数

```python
from QuantNodes.research.quant_alpha.logic_mining import (
    PipelineMetrics, LogicMiningPipeline,
)

metrics = PipelineMetrics()
pipeline = LogicMiningPipeline(metrics=metrics, ...)
result = pipeline.run("-ts_corr(rank(open), rank(volume), 10)", "alpha101")

# 查看 silent fallback 触发次数
print(metrics.to_dict())
# {'call_failures': {}, 'parse_failures': {'logic-mining-structure': 0}, ...}

# 三段式 stage 1 失败原因 (Stage 1 之前从未暴露)
print(result.parse_error)  # None or error string
print(result.parse_layer)  # 0..3
```

#### 2) 把 silent fallback 升级为异常 (开发期调试)

```python
from QuantNodes.research.quant_alpha.logic_mining import (
    LogicMiningPipeline, LogicMiningStrictError, StrictConfig,
)

# 所有 silent fallback 升级为异常 — CI 调试用
strict = StrictConfig(call=True, parse=True, structured=True)
pipeline = LogicMiningPipeline(metrics=metrics, strict=strict, ...)
try:
    pipeline.run(bad_formula, source_lib="alpha101")
except LogicMiningStrictError as e:
    print(f"failure kind={e.kind}, ctx={e.context}")
    # kind='call' | 'parse' | 'structured'
```

#### 3) 读取 AlphaLogics 失败诊断

```python
from QuantNodes.research.quant_alpha.workflow.alpha_logics import AlphaLogicsWorkflow
from QuantNodes.research.quant_alpha.logic_mining import PipelineMetrics

metrics = PipelineMetrics()
config = AlphaLogicsConfig(metrics=metrics, ...)
wf = AlphaLogicsWorkflow(config, llm_client=...)
result = wf.run()

# 失败可观测
print(result.diagnostics.to_dict())
# {'wiki_failures': 0, 'inner_loop_failures': 0, 'strict_raised': 0, ...}

print(result.metrics.to_dict())
# 包含所有 PipelineMetrics 计数 (call / parse / structured failures)
```

#### 4) 真实 `best_ic` 替代 IR 代理 (下游解读 `LogicPerformanceEvidence.best_ic`)

```python
# Before (v3.0.0): best_ic 实际上等于 best_ir (语义错误)
ev = LogicPerformanceEvidence(best_ir=0.5, best_ic=0.5)  # 双倍, 误导

# After (v3.0.1): best_ic = max(|ic_mean|) across factors
ev = build_inner_evidence(logic_name, alphagpt_result, round_idx=1)
# best_ic 是真实 IC, 可独立解读
```

#### 5) alpha191 不再静默空

```python
# Before (v3.0.0): 静默返回 []
formulas = get_formulas_from_source("alpha191")   # []

# After (v3.0.1): 返回 18 条 OHLCV-only 公式
formulas = get_formulas_from_source("alpha191")   # 18 entries
```

### Tests

- **新增 5 个测试文件** (33 P0 + 39 Phase 2 + 13 P0 + 33 md/SKILL = 118 new test cases)
  - `tests/quant_alpha/test_pipeline_metrics.py` (39 tests)
  - `tests/quant_alpha/test_parse_result_layers.py` (10 tests)
  - `tests/quant_alpha/test_alphalogics_diagnostics.py` (10 tests)
  - `tests/quant_alpha/test_logic_mining_strict.py` (12 tests)
  - `tests/quant_alpha/test_p0_logic_mining_fixes.py` (13 tests)
  - `tests/quant_alpha/test_agent_md_files.py` (33 tests)
- **基线 1134 → 1218 passed** (+84 new), **零回归**
- 之前 `tests/quant_alpha/large_scale_e2e_test.py` 已知 fixture mismatch (3 ERROR) 保持不变
  (该文件在 master 上同样存在 fixture 错误)

### Known Limitations (Out of Scope)

- `extract_operators` / `parse_op_args` 在 `compiler.py` 仍用 regex+paren-counting,
  未升级为真 AST parser (P3 大型重构, 单独 PR)
- alpha191 只有 18 条 OHLCV-only 公式, 完整 191 条含财务类,
  后续若需扩展需引入财报数据源
- nanobot md 文件目前与原 Python `_build_*_prompt` (内联 f-string) **并存**,
  nanobot spawn 集成留待后续 PR

### Refs

- 设计文档: `docs/32-市场逻辑驱动因子挖掘设计.md` (PR-3 / PR-4 spec)
- 基线报告: `docs/quant_alpha/logic_mining_silent_baseline.md`
- V5 截断参考: `docs/quant_alpha/explanation_truncation_fix.md` (4 层防御模式)

---

## [2.10.0-mock.1] - 2026-06-25

Stage 1 mock Table 4 复现 — 论文 Alpha-GPT 框架 mock 端到端验证。

### Added

- **Stage 1 mock Table 4 复现** (`QuantNodes/research/quant_alpha/evaluation/`)
  (`scripts/reproduce_table4_mock.py`):
  - 接口契约 (`contracts.py`)：4 dataclass (`FactorSpec`/`FactorMetrics`/
    `Table4GroupResult`/`Table4Report`) + 4 ABC (`DataLoader`/`Evaluator`/
    `Baseline`/`Table4Runner`)
  - MockDataLoader：500 票 × 500 日 GBM 模拟数据 + 10 行业 + forward return
  - PolarsAlphaCalculatorEvaluator：包 alpha_evaluate tool（M5），Stage 1/2 共用
  - G1Handcrafted：从 OperatorVocab 动态生成 100 公式
  - G2LlmOnly：mock LLM 直接生成 50 公式（含 15% invalid）
  - G3AlphaGpt：包 AlphaGptWorkflow（M5）+ 兜底 mock
  - MockTable4Runner：串联 DataLoader + 3 Baseline + Evaluator
  - CLI `scripts/reproduce_table4_mock.py`：支持 `--quick` / `--full`
  - 测试 8 文件 114 PASSED（97% 覆盖 evaluation/ 子包）
  - 文档：`docs/quant_alpha/table4_reproduction.md` (380+ 行) +
    `docs/quant_alpha/stage2_data_requirements.md` (301 行)

### Notes

- Stage 1 mock 完成（8 commit pushed to origin）
- Stage 2 real 待用户准备 iFinD 数据 + MiniMax API key
- 预计 Stage 2: ~2.4d
- 复用 M1-M6 基础设施 85% (OperatorVocab/PolarsAlphaCalculator/alpha_evaluate/AlphaGptWorkflow)

## [2.10.0] - 2026-06-26

Stage 2 real Table 4 复现 + LLM Gateway 统一入口 + 工具调用扩展 + 向量化优化。

### Added

- **LLM Gateway 统一入口** (`QuantNodes/ai/llm/gateway.py`):
  - `LLMGateway` 类：统一所有 LLM 调用入口，委托 nanobot upstream
  - 4 种接口：`chat()` / `complete()` / `__call__()` / `run()`
  - `ToolCallResponse` 数据类：支持工具调用事件返回
  - `get_llm_gateway()` / `create_llm_gateway()` / `reset_llm_gateway()` 全局单例管理
  - 测试 54 passed（含 17 个工具调用测试）

- **OperatorLookupTool** (`QuantNodes/agent/tools/operator_lookup.py`):
  - 3 个 action：`list_operators` / `get_operator_info` / `validate_formula`
  - 让 agent 动态发现 OperatorVocab 的 162 个可用算子
  - 测试 20 passed

- **Stage 2 real Table 4** (`QuantNodes/research/quant_alpha/evaluation/`):
  - `ClickHouseDataLoader`：从 ClickHouse 加载全 A 股数据（6.62M rows, 5570 stocks）
  - `RealTable4Runner`：Stage 2 主入口
  - `scripts/reproduce_table4_real.py`：Stage 2 CLI
  - G2 改用 `agent.run()` + `operator_lookup` 生成公式（valid 率从 20% → 80%）
  - G3 注入 `LLMGateway` 到 `AlphaGptWorkflow`
  - 测试 18 passed

- **PolarsAlphaCalculatorEvaluator 重构** (`QuantNodes/research/quant_alpha/evaluation/evaluators/polars_evaluator.py`):
  - 使用 `OperatorVocab.evaluate()` 直接评估公式（替代 alpha_evaluate tool 的有限 parser）
  - 支持复杂表达式：`rank(ts_mean(close, 20) - ts_mean(close, 5))` 等
  - 向量化 IC 计算：polars `group_by` + `pl.corr()` 替代逐日循环
  - 1.2M rows × 2 公式：从超时 → 秒级完成

### Changed

- **9 个 LLM 消费模块默认使用 LLMGateway**：
  - `ai/strategy_gen.py` / `ai/optimizer.py` / `research/report_reproducer.py`
  - `research/quant_alpha/workflow/alpha_gpt.py`
  - `core/feedback/llm_judge.py` / `core/knowledge/lineage_compress.py` / `core/evolution/operators.py`
  - `agent/tools/strategy.py` / `cli/commands/alpha.py`

- **Agent.chat() 增强** (`QuantNodes/agent/nanobot_bridge.py`):
  - 新增 `tools` / `tool_choice` 参数
  - `_filter_tools()` 工具名列表过滤（未知工具名警告并跳过）
  - 修复 `AgentRunSpec.max_tool_result_chars` 参数传递

- **Loop 3 向量化优化** (`QuantNodes/research/factor_test/nodes/group_analyzer_node.py`):
  - `cycle_net.T.groupby(fg).mean().T` 替代内层 `for g` 循环
  - 1304 × 5 × 6 = ~39000 次 pandas 调用 → 1304 × 5 = ~6520 次（6x 提速）

- **零子进程架构** (`QuantNodes/cli/commands/serve.py`, `api/main.py`,
  `QuantNodes/agent/config_mapper.py`):
  - `cmd_serve` 从 `subprocess.Popen(uvicorn)` 改为 `uvicorn.Server.run()` 同进程运行
  - 新增 `--mcp` / `--mcp-port` / `--frontend` / `--frontend-port` 参数
  - daemon 模式改为双 fork（`os.fork()` + `os.setsid()`）
  - 测试 700 passed / 3 skipped

### Stage 2 验证结果

```
数据源: ClickHouse quote.stock_quote (2024-01-01 ~ 2024-06-30)
LLM: MiniMax (via LLMGateway → nanobot → MiniMax API)

Group            | N | Success | avg_IR  | best_IR
G1_Handcrafted   | 10|       1 | -0.0330 | -0.0330
G2_LlmOnly       |  5|       4 |  0.0067 |  0.0554
G3_AlphaGpt      |  5|       2 |  0.0376 |  0.1082

排名: G3 (0.0376) > G2 (0.0067) > G1 (-0.0330) ✅
```

### 测试基线

- 总计：2539 passed, 7 skipped
- LLM Gateway：54 passed
- OperatorLookupTool：20 passed
- Stage 2 real：18 passed
- Loop 3 向量化：73 passed

## [2.9.1] - 2026-06-24

Patch release — 2 个 bug 修复，无新增功能。

### Fixed

- **`list_composite_ops()` 顺序非确定性** (`QuantNodes/operators/composite_dag.py:278`):
  原实现 `list(set(polars_ops | pandas_ops))` 顺序依赖 `PYTHONHASHSEED`（默认 random）。
  改为 `sorted(...)` 保证跨进程确定性,消除 `tests/operators/test_facade.py::TestExistsKind::test_kind_composite`
  的潜在 flake（依赖 `list_composite_ops()[0]` 的具体名字）。
- **`TestExistsKind` 跨测污染** (`tests/operators/test_facade.py`):
  加 autouse fixture `_isolate_custom_registry` 在每个 test 前后清空
  `_CustomOperatorRegistry`,防止前序测试注册的 custom op 通过 `kind()` 优先级
  (`custom > builtin > composite`) 命中 `list_composite_ops()[0]` 而 fail。
- **`ConfigBacktestRunner._compute_statistics` 空 equity curve 抛 `argmax(empty)`
  ValueError** (`QuantNodes/backtest/config_runner.py:156`):
  重新应用 v2.9.0 修复（在 master HEAD 上被 commit 5295629 revert）。
  short-circuit `len(equity_curve) == 0` 在 `pct_change()` 之前。
- **`tests/cli/test_cli_command.py` CLI registry 期望值 14 → 15**:
  重新应用 v2.9.0 修复（master HEAD 上被 revert）。
  加 `alpha-gpt` 到 `EXPECTED_COMMANDS`,`test_all_14_commands_registered` →
  `test_all_15_commands_registered`。
- **MySQL/ClickHouse 集成测试无 server 跳过** (`tests/test_database_node.py`):
  新增 `_server_available(host, port)` 工具函数（`socket.create_connection` 1s 超时检测）。
  MySQL fixture (`mysql_node`) 和 ClickHouse fixture (`clickhouse_node`) 在 server 不可达时
  `pytest.skip`,避免 `OperationalError` 失败污染全测。CI 友好。
- **`SamplePoolFilterNode` 输出保留 `trade_dt` / `stklist` 轴**
  (`QuantNodes/research/factor_test/nodes/sample_pool_filter_node.py:103`):
  原 `pd.DataFrame(stock_sample)` 用默认 RangeIndex,导致下游 `TradabilityFilterNode`
  做乘法时 pandas 轴并集对齐, shape 从 (1305,50) 翻倍到 (2610,100)。
  显式赋予 `index=trade_dt.iloc[:, 0].values` 和 `columns=stklist.iloc[:, 0].values`,
  与 `DataLoader.add_index()` 约定一致。

### Testing

- 测试基线: **5303 passed, 27 skipped, 0 failed** (稳定)
- 单测隔离: `test_kind_composite` 在 `_isolate_custom_registry` autouse 保护下
  不再依赖 `_CustomOperatorRegistry` 状态
- 集成测试: MySQL/CH server 不可达时 skip,可达时正常 PASSED

---

## [2.9.0] - 2026-06-24
  - **WebSocket 端点**：`/api/alpha/alpha-gpt/stream/{sid}` 实时事件流（6 事件类型：round_started/round_completed/formulas_evaluated/final_pool_ready/done/error/heartbeat）
  - **事件总线**：`AlphaGptService.subscribe()` / `_emit()`，支持 buffer replay + 多订阅者
  - **跨线程事件注入**：`asyncio.run_coroutine_threadsafe` 把 sync workflow 的事件安全投递到主事件循环
  - **前端 Vue 页面**：`/alpha-gpt` 路由 + `AlphaGpt/index.vue`（Ant Design Vue 3 + 实时进度条 + IC 演化图 + 公式表）
  - **侧边栏入口**：`AppSidebar.vue` 加 `Alpha-GPT` 菜单项（仅 agent 启用时显示）
  - **Phase C 归档**：4 个旧模块（factor_evaluator/factor_miner/auto_researcher/mcts_search）移至 `QuantNodes/research/_legacy_3c/`
  - **向后兼容 shim**：`QuantNodes.research.factor_evaluator` 等通过 `_LegacyShim` 仍可导入但触发 DeprecationWarning
  - 6 个新 WebSocket 测试 + 1 个 Phase C 测试
- **Alpha-GPT CLI / API / v2.7.0 release**（M6 PR）：
  - CLI `quantnodes alpha-gpt`（`AlphaGptCommand`，18 个 argparse 参数）
  - API 5 端点（`/api/alpha/alpha-gpt/{generate,status/{sid},results/{sid},stop/{sid},list}`）
  - `AlphaGptService` 后台 session 管理（async + semaphore 并发限制）
  - `NanobotLLMWrapper`：把 `QuantNodes.agent.Agent` 包装成 workflow 期望的 client 接口
  - 14 个新测试（E2E 5 + CLI 5 + workflow 集成 4）
- **QuantAlpha 子包**（`QuantNodes/research/quant_alpha/`）：自动化因子挖掘引擎，参考 4 大因子库演进链（Alpha 101/158/360/AutoAlpha）
  - **OperatorVocab** (`quant_alpha.operator_vocab.OperatorVocab`)：统一算子查询/调用/元数据化接口（M1）
  - **5 个新算子**：`signedpower` / `ts_decay_linear` / `IndNeutralize` / `ts_skew` / `ts_kurt`（修复 Alpha 101 关键缺口，M1）
  - **算子元数据 schema 扩展**：从 5 字段到 12 字段（新增 7 个 LLM 友好字段，M1）
  - **per-date over() 修复**：rank/zscore/winsorize 默认 per-date 截面（修复 12-lambda namespace 的 BUG 2，M1）
  - **MCTS 子包**（`quant_alpha.mcts`，M2）：
    - `ExtensionOpPool`：从 OperatorVocab 动态生成 26 个扩展操作（vs 旧 7 硬编码）
    - `MCTSNode` / `MCTSTree`：完整谱系追踪（entry_id + parent_id + ancestors + lineage_depth）
    - 5 通道反馈框架（`MCTSFeedbackConfig` + 5 个 channel collector）：
      execution / shape / code / value / llm
    - `MCTSSearch`：UCB1 选择 + 5 通道反馈驱动 + 谱系持久化
  - **CLI 命令**：`quantnodes alpha-mcts`（M2）
  - **Alpha 101 设计借鉴**（`quant_alpha.alpha101_design`，M3）：
    - 8 条设计原则（P1-P8：数学即代码、动量反转、截面 rank、ts_argmax 提取极值位置、
      signedpower 保留符号、decay_linear 加权、三元条件、IndNeutralize 行业中性化）
    - 16 个核心算子（含经济意义 + Alpha 101 公式示例）
    - 8 个 A 股可移植性记录（4 个 Delay-0 不可移植 + 4 个可移植）
    - 10 个 few-shot 示例（覆盖 4 类：momentum / reversal / volume_price / intraday）
  - **Alpha 158/360 设计借鉴**（`quant_alpha.alpha158_design`，M3）：
    - 4 类特征模板：KBAR (9) / Price (20) / Volume (5) / Rolling (124) = 158 特征
    - Alpha 360 模板：6 字段 × 60 lookback = 360 特征
    - 10 个 few-shot 示例（覆盖 4 类）
  - **Alpha-GPT 5 智能体编排**（`agent.tools.alpha_*` + `quant_alpha.workflow`，M5）：
    - 2 个新工具：`alpha_evaluate`（包 M4 PolarsAlphaCalculator，批量 IC/IR/ic_decay）+ `alpha_backtest`（top-K 等权 Trading 回测，年化/Sharpe/MaxDD）
    - 5 个新 subagent：`alpha-gpt-{idea-generator, formula-translator, evaluator, reflector, critic}`（基于 `.agent/agents/alpha-gpt-*.md`）
    - `AlphaGptWorkflow` 协调器：5 轮主循环 + 多进程 spawn（复用 nanobot upstream）
    - JSON 三层降级解析器：`parse_{idea_generator, formula_translator, evaluator, reflector, critic}_output`
    - 算子白名单校验（32 算子）+ 简单 formula parser
    - 未来所有 RL/LLM 路线接入：**4 天**（vs 全栈 40+）
  - **AlphaGen RL 适配器**（`quant_alpha.adapters`，M4）：
    - 极简 Expression AST（11 算子 + Literal）：Feature / Ref / BinaryOp / UnaryOp / RollingOp
    - `BaseAlphaCalculator` ABC：7 个抽象方法（与 AlphaGen `AlphaCalculator` 接口兼容）
    - `PolarsAlphaCalculator`：参考实现，用 polars + OperatorVocab
    - 7 个方法完整实现：single_IC_ret / single_rIC_ret / single_all_ret /
      mutual_IC / pool_IC_ret / pool_rIC_ret / pool_all_ret
    - 公式缓存 + 自动按 (code, date) 排序
    - **未来所有 RL 路线接入只需 4 天**（vs 全栈复刻 40+）
  - **完整调研 + 规划文档**：`docs/quant_alpha/PROJECT_PLAN.md`（991 行）

### Changed

- **MCTS 操作池**：从 7 硬编码 → 26 动态生成（按 6 个 category：wrap/window/window_binary/unary/diff/ratio）

### Fixed

- `QuantNodes.research.factor_evaluator._compute_factor` 三大隐性 bug：
  1. `ts_corr` / `ts_cov` 用 `Series.rolling_corr`（Series 上不存在）→ 改用 L0 注册表 `rolling_corr` (Expr API)
  2. `rank` / `zscore` 全局计算而非 per-date 截面 → 默认 `cross_sectional=True` per-date over(date)
  3. 异常被 `except Exception: return None` 静默吞掉 → 完整错误抛出

### Deprecated

- `QuantNodes.research.factor_evaluator` → 迁移到 `quant_alpha.operator_vocab.OperatorVocab`
- `QuantNodes.research.factor_miner` → 迁移到 `quant_alpha.operator_vocab`（M2+）
- `QuantNodes.research.mcts_search` → 迁移到 `quant_alpha.mcts.MCTSSearch`（M2 PR）
- `QuantNodes.research.auto_researcher` → 迁移到 `quant_alpha.AutoResearcher`（M5+ PR）

### Migration

- 详见 [`docs/quant_alpha/migration.md`](docs/quant_alpha/migration.md)
- Phase A: 旧代码仍可用，DeprecationWarning 仅提示
- Phase B (v2.9+): 旧类变 thin wrapper
- Phase C (v3.0): 旧实现归档到 `_legacy_3c/`

## [3.0.0] - 2026-06-23

v3.0.0 — 上游 nanobot 迁移：从"复刻 nanobot 架构"升级为"直接消费 HKUDS/nanobot 0.2.1 上游"。

**关键变更**：
- 核心运行时从自写 `loop/runner/memory/...` 改为包装 `Nanobot.from_config()`
- `nanobot-ai` 改为 `[agent]` optional extra（量化库独立可用）
- 单进程架构：FastAPI uvicorn + nanobot gateway 共存于同一 Python 进程
- 完整 Phase 5 功能（5 个 stage，11 个 commit）：subagent / MCP / WebUI / 渠道 / Cron

### Stage 1 — Architecture (commit 2584462)

- **核心运行时迁移**：删除自写 `QuantNodes/agent/core/{loop,runner,memory,autocompact,context,hook,compaction}.py`、`bus/`、`session/`、`templates/agent/`、`config/{loader,executor}.py`、`cli/main.py`、`web/`。
- **编程式门面**：新增 `QuantNodes/agent/nanobot_bridge.py::Agent`，薄包装 `Nanobot.from_config(config_path, workspace=...)`，旧 `Agent(workspace, config)` 签名不变。
- **dream.py 保留**：从 `core/dream.py` 迁出为 `core/quant_dream.py`，实现 `QuantDreamHook(AgentHook)` 挂在上游 hook 系统；`core/dream.py` 保留为 re-export shim（DeprecationWarning）。
- **配置映射**：新增 `QuantNodes/agent/config_mapper.py`，把 `.env` 的 `QUANTNODES__LLM__*` 翻译为 `.agent/nanobot_config.json` 的 `providers` 块（含 slot 自动推断：OpenAI 兼容 → `custom`、Anthropic → `anthropic`、Ollama → `ollama`、Azure → `azure_openai`）。
- **Skills**：新增 `QuantNodes/agent/skills_quant/` 6 个 SKILL.md（factor-research / strategy-design / backtest-analyze / risk-management / quant-dream / config-driven），格式对齐上游 `nanobot/skills/*/SKILL.md`（YAML front-matter + 指令体）。
- **CLI/WebUI**：删除自写 `agent/cli/main.py` 和 `agent/web/`，改为 `python -m nanobot` + `python -m nanobot webui --port 18080`。
- **量化工具继承 nanobot Tool**：`QuantNodes/agent/tools/base.py::Tool` 改为继承 `nanobot.agent.tools.base.Tool`，15 个 quant 工具父类自动升级。`register_all_quant_tools(registry, workspace)` 把 14 个 tool 注入上游 `ToolRegistry`。

### Stage 2 — API 解耦 (commit 9e5999a)

- `api/services/agent_service.py`：改 import 为 `nanobot_bridge.Agent`，事件协议 `token/tool_call/tool_result/done` 向后兼容；新增 `reload_bot()`，保留 `reload_agent()` 别名。
- `api/services/wiki_service.py` (重写)：去除 `WikiTool`，直接用 `QuantNodes.research.wiki.WikiFactorProxy`。
- `api/services/stats_service.py` (重写)：同样去除 `WikiTool`。
- `api/services/dream_service.py`：改用 `core/quant_dream.QuantDreamHook` + `DreamEngine(workspace=...)`。
- `api/services/backtest_service.py`：注释说明仍用 `ConfigBacktestTool`，未来 TODO 用 `ConfigBacktestRunner` 替代。
- `api/routers/settings.py`：6 处 `reload_agent()` → `reload_bot()`。
- `api/routers/skill.py`：保持本地 `SkillRegistry`（上游 SKILL.md 解析器在 Stage 5.x 引入）。

### Stage 3 — Workspace 迁移 (commit 4b7560e)

- `.quant_agent/` → `.agent/`（HKUDS nanobot 上游默认约定）。
- 一次性迁移脚本 `scripts/migrate_workspace.py`：自动分割 v2 MEMORY.md 为 SOUL.md + memory/MEMORY.md；保留 sessions / settings.json；写 `.migration_manifest.json` 记录元数据。
- 16 处默认 workspace 改 `.agent/`：`QuantNodes/core/config.py`、`QuantNodes/cli/_helpers.py`、`QuantNodes/agent/tools/task.py`、`api/config.py`、`api/services/{agent,backtest,dream,settings,stats,wiki}_service.py`。
- `.gitignore` 加 `.agent/`（含 API key 等敏感配置）。

### Stage 4 — Tests + Docs (commit pending)

- 新增 `tests/agent/test_quant_dream_hook.py` (16 tests) — 覆盖 QuantDreamHook / DreamEngine shim / get_recent_dreams round-trip。
- 新增 `tests/agent/test_quant_tools.py` (10 tests) — 工具 schema / to_schema / register_all_quant_tools idempotent。
- 新增 `tests/agent/test_nanobot_integration.py` (10 tests) — 端到端 Agent(workspace) + 14 量化工具注册 + config_mapper 路由。
- 删除 11 个 broken-collect 测试文件（`test_{loop,runner,memory,memory_persistence,bus,session,chat,context,hook,autocompact,agent_loop_p1}.py`）—— 上游 nanobot 已覆盖这些功能。
- 更新 `docs/13-Agent架构设计.md`：新增"工作区约定"节。
- 更新 `AGENTS.md`：`.agent/` 路径说明 + 迁移脚本。

### Dependencies

- 增 `nanobot-ai>=0.2.1,<0.3.0`（alpha 期锁次版本号）。
- `pyproject.toml::requires-python` 升 `>=3.11`（upstream 最低要求）。
- 本地开发期 `pip install -e ~/Public/nanobot`（HKUDS/nanobot v0.2.1 源码克隆）。若 GitHub 不可达，可从 PyPI 安装 `nanobot-ai==0.2.1`。

### Breaking Changes

- 删 `QuantNodes/agent/core/{loop,runner,memory,compaction,autocompact,context,hook}.py` —— 直接 import 路径报错，必须改用 `Agent.run()` / `Nanobot.from_config()` facade。
- 删 `QuantNodes/agent/bus/`、`session/`、`config/{loader,executor}.py`、`templates/agent/`、`cli/main.py`、`web/` —— 同上。
- workspace 从 `.quant_agent/` 迁 `.agent/` —— 见 `scripts/migrate_workspace.py`。
- 旧 `agent/templates/agent/*.md` 改名为 `.agent/SOUL.md` —— 见 `scripts/migrate_workspace.py`。

### Baseline (Python 3.11, Stage 4)

- 非 agent 测试：4143 passed / 336 failed / 28 errors / 6 skipped
- tests/agent/：661 passed / 35 failed（pre-existing 网络测试 + TestAgentLoop session tests）

> **最终（Stage 6 完成后）**：非 agent 测试 5163 passed / 21 skipped / **0 failed / 0 errors**（顺序 + 并行均通过）；tests/agent 574 passed / 13 skipped。详见下方 *Stage 6 — 测试稳定化与依赖兼容*。

### Migration

详见 [`docs/14-上游nanobot升级指南.md`](docs/14-上游nanobot升级指南.md)。

### Stage 5.1 — Subagent 多 Agent 团队 (commit f7ac409)

- 启用 nanobot subagent 多 Agent 团队：在 nanobot 0.2.1 的 `spawn`/`read_file` 基础上，通过 `SOUL.md` + `.agent/agents/*.md` 实现角色化子智能体。
- 新增 `.agent/SOUL.md`：ResearchDirector 主体人格 + delegation matrix（"factor research" → factor-analyst，"backtest" → backtest-engineer，"risk" → risk-manager）。
- 新增 `.agent/agents/{factor-analyst,backtest-engineer,risk-manager}.md`：3 个专家子智能体的系统提示词。
- `.gitignore` 加 `!.agent/SOUL.md` / `!.agent/USER.md` / `!.agent/agents` / `!.agent/agents/*.md` re-include 规则（共享团队定义，忽略敏感 settings.json）。

### Stage 5.2 — MCP server (commit a37ef30)

- 新增 `QuantNodes/mcp_server/server.py` (~270 行)：FastMCP 3.4.2 + Pydantic per-tool 模型，暴露 9 个 MCP tools：
  - `call_backtest` / `call_config_backtest` / `call_factor` / `call_strategy` / `call_pipeline` / `call_sandbox` / `call_wiki`（7 个 quant 工具 dispatcher）
  - `list_quant_tools`（元数据 + JSON Schema 发现）
  - `data_query`（DuckDB SQL，v0）
- 设计：每 `call_*` 是 dispatcher（统一 `arguments` 形参），FastMCP 不支持 `**kwargs` 动态 schema。
- `QuantNodes/agent/config_mapper.py` 自动注入 `mcpServers.quant` 块到 `nanobot_config.json`。
- 新增 `pyproject.toml::[mcp]` optional-dependency。
- 新增 `tests/agent/test_mcp_server.py` (8 tests)：导入 / 注册 / 调用 / schema。

### Stage 5.3 — 单进程 WebUI 集成 + 可选依赖 (commit pending)

#### 🎯 核心变更

- **可选依赖**：``nanobot-ai`` 从强制依赖移到 `[agent]` extras。`pip install quantnodes` 即可获得纯量化库；`pip install 'quantnodes[agent]'` 启用完整 agent / WebUI / MCP。
- **单进程架构**：FastAPI lifespan 内手动拉起 nanobot 的 `AgentLoop` + `ChannelManager` + `CronService` 为 `asyncio.create_task`（不用 `asyncio.run`）。WebUI SPA + WebSocket 在 18080 端口（同一 Python 进程）。

#### 新增

- `api/services/nanobot_runtime.py` (~370 行)：单进程 lifespan 包装器
  - `NanobotRuntime` 类：手动连接 `MessageBus` / `SessionManager` / `CronService` / `AgentLoop.from_config()` / `ChannelManager(webui_static_dist=True)` / 注册 14 个 quant 工具
  - `init_runtime()` / `shutdown_runtime()` 进程级单例
  - 状态机：`uninitialized` → `starting` → `running` → `stopping` → `stopped` / `error` / `unavailable`
  - graceful 降级：未装 nanobot → `state=unavailable` + install hint
- `api/routers/agent.py` (~150 行)：6 个端点
  - `GET /api/agent/status`：200 + 运行时状态（永远返回 200，前端可读）
  - `GET /api/agent/health`：200/503 readiness probe
  - `POST /api/agent/restart`：销毁 + 重建（环境变量变更后用）
  - `POST /api/agent/chat/send`：非流式 chat
  - `GET /api/agent/sessions`：列 websocket sessions
  - `DELETE /api/agent/sessions/{key}`：删除 session
- `frontend/src/views/AgentChat.vue` (~110 行)：iframe + 状态机渲染
  - 加载中：spin
  - `unavailable`：`a-result` install 提示页
  - `starting`：spin + 提示
  - `error`：错误 + 重试按钮
  - `running`：`<iframe src="VITE_NANOBOT_GATEWAY_URL" sandbox="...">`
  - 5 秒 polling 监测状态
- `frontend/src/components/Layout/AppSidebar.vue`：`v-if="agentEnabled"` 控制 Agent Chat 入口（默认显示）
- `frontend/src/router/index.ts`：`/agent-chat` 路由
- `frontend/.env.development`：`VITE_AGENT_ENABLED=true` + `VITE_NANOBOT_GATEWAY_URL=http://127.0.0.1:18080/`
- `docs/15-可选依赖安装指南.md` (新)：三档安装说明 + 升级指南 + FAQ

#### 优雅降级（关键）

- `QuantNodes/agent/__init__.py`：PEP 562 `__getattr__` proxy + `NANOBOT_AVAILABLE` 标志
  - `from QuantNodes.agent import Agent` 永远成功（不抛 ImportError）
  - `Agent(...)` / `Agent.attr` 抛 `NanobotNotInstalled` 含 install hint
- `QuantNodes/agent/tools/base.py`：`Tool` 父类在未装 nanobot 时降级为最小 ABC
- `QuantNodes/agent/nanobot_bridge.py`：延后 `from nanobot import Nanobot` 到 `__init__`

#### 测试

- `tests/agent/test_optional_dependency.py` (8 tests)：未装 nanobot 时所有 import 路径、HTTP 端点
- `tests/agent/test_nanobot_runtime.py` (9 tests)：singleton、start/stop 单进程、状态机
- `tests/agent/test_webui_integration.py` (10 tests)：router / sidebar / iframe 路由 / env 契约 / pyproject extras

#### Breaking Changes

⚠️ **用户必须升级后显式装回 [agent] extra**：

```bash
pip install --upgrade quantnodes
pip install 'quantnodes[agent]'   # ← 这一步必做，否则失去 agent 功能
```

### Pending (Phase 5)

- [x] 5.1 Subagent 多 Agent 团队（main/factor-analyst/backtest-engineer/risk-manager）
- [x] 5.2 MCP server（quant 能力 stdio 暴露）
- [x] 5.3 单进程 WebUI + 可选依赖
- [x] 5.4 渠道接入（飞书 + WebSocket）
- [x] 5.5 Cron 调度（日终/周度/月度）

### Stage 5.5 — 量化专属 Cron 调度 (commit pending)

#### 新增

- **`QuantNodes/agent/cron_jobs.py`** (新, ~270 行) — 3 个 quant 系统任务的定义 + 注册逻辑：
  - `QuantCronJob` dataclass：纯 Python 数据类（无 nanobot 依赖）
  - 3 个默认任务：
    - `quant-daily-recap` — 工作日 16:30 因子 IC 重算 + 回测归档
    - `quant-weekly-review` — 周日 22:00 因子周报 + 风险归因
    - `quant-monthly-strategy-pool` — 每月 1日 02:00 Wiki 增量 + 策略池评审
  - `DEFAULT_TZ = "Asia/Shanghai"`
  - `build_quant_cron_jobs_from_env()` — 应用 env 覆盖并过滤 disabled
  - `register_quant_cron_jobs(cron_service)` — 调 `CronService.register_system_job()`
  - `NanobotNotInstalledForCron` 异常（ImportError 子类）
- **`api/services/nanobot_runtime.py`** — 在 `_build_components` 中调 `register_quant_cron_jobs(self._cron)`，try/except ImportError 容错
- **`api/routers/agent.py`** — 新增 2 个端点：
  - `GET /api/agent/cron` — 列出所有 cron jobs
  - `GET /api/agent/cron/{job_id}/run-now` — 立即触发某个任务
- **`.env.template`** — 新增 `QUANTNODES__CRON__<NAME>__<FIELD>` env 覆盖说明
- **`docs/13-Agent架构设计.md`** §13 新增：cron 三套任务 + 注册流程 + env 覆盖 + API 端点 + 失败模式

#### env 覆盖

```bash
# 禁用单个任务
QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED=false

# 修改 cron 表达式
QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__CRON_EXPR="0 20 * * 0"

# 自定义 prompt
QUANTNODES__CRON__QUANT_DAILY_RECAP__MESSAGE="自定义 prompt..."

# 关闭结果推送
QUANTNODES__CRON__QUANT_MONTHLY_STRATEGY_POOL__DELIVER=false

# 修改 channel
QUANTNODES__CRON__QUANT_DAILY_RECAP__CHANNEL=feishu
```

#### 测试（17 new tests, 14 pass + 3 skip）

- `tests/agent/test_cron_jobs.py` (17):
  - `test_three_default_jobs_exist` — 3 默认任务齐全
  - `test_default_jobs_have_distinct_schedules` — cron 表达式去重
  - `test_default_jobs_have_valid_cron_expressions` — 5 字段格式
  - `test_default_jobs_use_default_timezone_reference` — TZ 常量
  - `test_default_jobs_have_non_empty_messages` — prompt ≥ 50 字符
  - `test_default_jobs_are_enabled` — 默认 enabled + deliver
  - `test_default_jobs_have_descriptions`
  - `test_build_from_env_returns_defaults_when_no_env`
  - `test_env_can_disable_individual_job`
  - `test_env_can_disable_all_jobs`
  - `test_env_can_override_cron_expression`
  - `test_env_can_override_message`
  - `test_env_can_override_deliver_flag`
  - `test_env_truthy_values_accepted`
  - `test_register_quant_cron_jobs_with_mock` (skip if nanobot missing)
  - `test_register_is_idempotent_on_reregistration` (skip)
  - `test_register_respects_env_disable` (skip)

3 个注册相关测试需要 `nanobot.cron.types`，未装 nanobot-ai 时优雅跳过。

#### 用户启用

```bash
# 1. 装 [agent] extra
pip install 'quantnodes[agent]'

# 2. 重启 FastAPI，3 个任务自动注册（查看 /api/agent/cron 确认）

# 3. 默认行为：
#    - 工作日 16:30 自动跑日终复盘，结果送到 Feishu 群
#    - 周日 22:00 自动跑周度复盘
#    - 每月 1日 02:00 自动跑月度策略池评审

# 4. 手动触发（调试用）：
curl http://localhost:8000/api/agent/cron/quant-quant-daily-recap/run-now

# 5. 调整时间：
echo 'QUANTNODES__CRON__QUANT_DAILY_RECAP__CRON_EXPR="0 17 * * 1-5"' >> .env
```

### Stage 5.4 — 渠道接入 (commit pending)

#### 新增

- **`config_mapper.py`** — 新增 `channels` 块到生成的 nanobot 配置：
  - `_build_websocket_config()`：默认启用，端口 `8765`，SPA 由 `webui_static_dist=True` 提供
  - `_build_feishu_config()`：仅当 `FEISHU_APP_ID` + `FEISHU_APP_SECRET` 同时设置时启用
  - `channel_overrides` 参数：让 FastAPI runtime 注入 gateway host/port
- **`api/services/nanobot_runtime.py`** — `_build_components` 通过 `channel_overrides` 把 `NANOBOT_GATEWAY_HOST/PORT` 注入 websocket 块
- **`frontend/src/composables/useNanobotWebSocket.ts`** (新) — 完整的 nanobot wire protocol 客户端：
  - `fetchBootstrap()`：GET `/webui/bootstrap` 取短期 token
  - `WebSocket(wsUrl)`：携带 `?token=...&client_id=...`
  - 类型化的事件类型：`attached` / `user` / `message` / `tool_call` / `tool_result` / `tool_hint` / `tool_status` / `goal_status` / `error`
  - 发送：`{type: 'message', content, chat_id}` JSON envelope
  - 指数退避重连（默认 1.5×，最多无限重连）
  - `onUnmounted` 自动 disconnect
- **`.env.template`** — 新增 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_DOMAIN` / `FEISHU_GROUP_POLICY` / `FEISHU_REPLY_TO_MESSAGE` / `FEISHU_ALLOW_FROM` 等环境变量
- **`frontend/.env.development`** — `VITE_NANOBOT_BOOTSTRAP_PATH=/webui/bootstrap`
- **`docs/13-Agent架构设计.md`** §12 新增：渠道架构 + 配置注入 + wire 协议 + 飞书 channel + ChannelManager 生命周期 + 失败模式

#### 测试（12 new tests）

- `tests/agent/test_channel_config.py` (9) — channels 块配置：
  - websocket 默认启用 + 端口/host 默认值
  - feishu 缺失 env 时禁用
  - feishu 完整 env 时启用 + app_id/secret 正确
  - feishu 部分 env（仅 APP_ID）禁用
  - feishu 可选 knobs（domain / group_policy / reply / encrypt / verify / allow_from）
  - channel_overrides.propagate 到 websocket
  - channel_overrides 可强制禁用 websocket
  - mcpServers 块与 channels 共存
  - 顶层 keys 完整性
- `tests/agent/test_nanobot_websocket_protocol.py` (8) — 前端 wire 协议契约：
  - 文件存在
  - `useNanobotWebSocket` 是 named function
  - 调用 `/webui/bootstrap`
  - 发送 `type/content/chat_id` envelope
  - 引用全部 7 个核心事件类型
  - 指数退避
  - onUnmounted 自动 disconnect
  - env 包含 bootstrap path

#### 用户启用飞书

```bash
# 1. 在飞书开放平台创建应用，获取 APP_ID / APP_SECRET
# 2. 启用"机器人"能力 + 事件订阅 im.message.receive_v1
# 3. 装 SDK
pip install lark-oapi
# 4. 配置 .env
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
# 可选：
# FEISHU_GROUP_POLICY=open   # 接收所有群消息（默认 mention）
# 5. 重启 FastAPI，feishu channel 自动启动
```

### Stage 6 — 测试稳定化与依赖兼容 (commits c5a7e3c / 0ec6fe0 / 30a0352 / f30f9b8)

Python 3.11 + pandas 3.0 升级后，对全量测试套件做根因级修复。**最终全量
非 agent 测试：5163 passed / 21 skipped / 0 failed / 0 errors（顺序 + 并行
两种模式均通过）；tests/agent：574 passed / 13 skipped。**

#### v2.x 遗留测试清理与重写 (commits c5a7e3c / 0ec6fe0)

- 删除 13 个测试 v2.x 已删代码路径的文件（约 2700 行）。
- 重写 6 个测试文件以验证 v3.0.0 等价功能（共约 112 tests）：
  - `tests/agent/test_base.py` — `Tool` + `ToolExecutionResult`
  - `tests/agent/test_tools.py` — `ToolRegistry` + 14 quant 工具
  - `tests/agent/test_tools_parallel.py` — 并行执行
  - `tests/agent/test_tools_all.py` — Mock `AgentLoop`/`AgentRunner`（无 nanobot 时 skip）
  - `tests/agent/test_agent_service.py` — `AgentService` + `MockAgent`
  - `tests/agent/test_skills_phase4.py` — Bridge + `DreamEngine` shim
- 策略：需要上游 nanobot 的测试用 skip-when-missing 装饰器，**不删除**。

#### pandas 3.0 兼容 (commit 30a0352)

- `DataFrame.applymap` 在 pandas 3.0 移除 → 改用 `DataFrame.map`：
  - `research/factor_test/utils/date_utils.py`（`datenum_to_datetime` / `datetime_to_datenum`）
  - `research/factor_test/ifind_db/ifind_database.py`（行业 / ST / 停牌 / 涨跌停 4 个面板转换）
- `dtypes[0]` 在列名为混合类型（int `0` → str `'trade_dt'`）时抛 `KeyError` → 改用 `dtypes.iloc[0]`（`date_utils.py::valid_date`）。
- `DataFrame.values` 在单一 dtype 下变只读 → 测试改用 `DataFrame.where()` 而非原地 `.values[...]` 赋值。
- 字符串列推断为 `StringDtype(na_value=nan)`（非 `object`）→ 测试断言改为 `not is_numeric_dtype(...)`。

#### 可选依赖优雅降级 (commit 30a0352)

- `core/knowledge/retriever.py::TFIDFRetriever`：sklearn 缺失时回退到 `IdentityRetriever`（`RuntimeWarning` + `getattr` 委托）。
- `core/monitoring/dashboard.py` + `core/visualization/{gate_breakdown,lineage_dag,metric_distribution}.py`：plotly 缺失时 figure 函数返回 `None`，HTML 渲染输出友好安装提示而非崩溃。
- `agent/tools/base.py`：未装 `[agent]` extra 时，独立 `Tool.to_openai_schema()` 从 `name`/`description`/`parameters` 合成 OpenAI function schema（不再 raise），与 nanobot 的「方法调用」契约一致。
- 测试侧用 `pytest.importorskip("plotly")` 跳过纯 plotly 渲染测试（保留纯 Python 的 `TestLineageLayout`）；`pymysql` 缺失时跳过 MySQL 集成测试。

#### 系统级测试依赖（开发环境）

为让全部测试在本地通过，需安装以下（Python 3.11）：

```bash
pip install ta-lib tables plotly   # talib 需先装 TA-Lib C 库
```

- `ta-lib`：66 个 talib 算子测试（依赖 TA-Lib C 库）。
- `tables`（PyTables）：HDF5 读写测试。
- `plotly`：可视化 / dashboard / e2e HTML 报告测试（e2e 报告 >5KB 校验需真实 figure）。

#### 跨测试污染根因修复 (commit f30f9b8)

8 个「只在全量运行时失败、单独跑通过」的测试，根因为真实状态泄漏（非 flakiness）：

- **HOME 环境变量污染**（修 5 errors + 3 fails）：`tests/core/test_path_utils.py::test_expanduser` 用裸 `os.environ["HOME"] = <TemporaryDirectory>` 且未还原；临时目录在 context 退出后被删，残留悬空 `HOME`。后续基于 subprocess 的 e2e 测试（`data_prep` / `run_evolution_e2e`）继承坏 `HOME`，在 `~/.quantnodes` 写入时非零退出。→ 改用 `monkeypatch.setenv`（自动还原）。
- **composite registry 泄漏**（修 `test_op_names_match_polars`）：`test_composite_dag_pandas_engine.py` 经 `load_composites_from_yaml` 注册 YAML op 到全局 `_COMPOSITE_REGISTRY` 无清理；`test_composite_dag.py` 旧 fixture 仅删本模块注册的 op。→ 两文件均改为全量快照/还原 autouse fixture。
- **防御性加固**：e2e subprocess 超时 30s→180s / 60s→300s；两处缺失 `timeout` 的 `subprocess.run` 补 `timeout=60`。

### Stage 7 — CLI 完善 (lifecycle mgmt) (commits 40d80f3 / 86e0acd)

5 个新 lifecycle 子命令，对标 `llmwikify` 简洁接口风格。CLI 子命令总数从 13 升至 20。

| Command | 描述 | 模式 |
|---|---|---|
| `quantnodes serve` | 启动 FastAPI + nanobot gateway | 前台 / `--daemon` / `--frontend` / `--check-env` |
| `quantnodes stop` | 通过 `.quantnodes.pid` 停止 | 写 SIGTERM + 清理 pidfile |
| `quantnodes status` | 综合状态（pidfile + /api/agent/status） | JSON 输出 + exit code |
| `quantnodes logs` | `tail -f logs/quantnodes_serve.log` | `-f` 实时滚动 / 默认最后 200 行 |
| `quantnodes agent status` | `GET /api/agent/status` | HTTP 客户端（无需 CLI 装 nanobot-ai） |
| `quantnodes agent chat MSG` | `POST /api/agent/chat/send` | 单次问答，含 `--session` |
| `quantnodes agent restart` | `POST /api/agent/restart` | 重启 nanobot runtime |

新增 helpers（`QuantNodes/cli/_helpers.py`，约 80 行）：

- `load_env_file(env_path)` — dotenv 缺失/.env 缺失均 graceful；供 `api/main.py` 与 `quantnodes serve` 共用（DRY）。
- `write_pidfile / read_pidfile / remove_pidfile` — 项目根 `.quantnodes.pid`，单 int。
- `is_port_free(port)` — 启动前端口冲突检测，给出明确报错（避免 nanobot 内部 `OSError [Errno 98]` 才知端口占用）。
- `wait_for_health(api_url, timeout_s=30)` — 轮询 `/api/agent/status` 直到 `state=running`。
- `is_nanobot_installed / print_nanobot_install_hint` — `quantnodes init` 末尾给非阻塞友好提示。
- `DEFAULT_GATEWAY_PORT = 18090`（避开 gpustack 占用的 18080）。
- `DEFAULT_API_PORT` 由 8000 改为 19380（避免与系统服务冲突）。

`run` 命令兼容增强（`QuantNodes/cli/commands/run.py`）：

- 加 `--gateway-port` 参数，注入子进程 env `NANOBOT_GATEWAY_PORT`。
- `start_api_server` 新增 `gateway_port` 参数。

`init` 命令末尾提示（`QuantNodes/cli/commands/init.py`）：

- 新 `quantnodes init` 完成信息更新为新接口（`quantnodes serve --daemon` 等）。
- 末尾调用 `print_nanobot_install_hint()` 提示 `[agent]` extra。

`api/main.py` DRY 化：

- 8 行内联 `load_dotenv` 块替换为 `from QuantNodes.cli._helpers import load_env_file`。
- 行为不变，但 .env 缺失时多一行 warning log（便于 onboarding 诊断）。

测试覆盖（`tests/cli/test_serve.py` + `tests/cli/test_agent_cli.py`，共 40 tests）：

- argparse 解析（serve 7 flags + agent sub-subparsers）。
- 端口冲突预检（busy port → 退出 1 + 明确消息）。
- `--check-env` 模式（缺 API key → 退出 1）。
- daemon 模式（写 pidfile + 日志路径）。
- stop 三种场景（无 pidfile / stale pidfile / alive PID → SIGTERM）。
- status 调用 `/api/agent/status` + exit code 反映 state。
- logs 文件不存在 / `tail -n 200` fallback。
- `load_env_file` / `is_port_free` / `is_pid_alive` roundtrip。
- agent status/chat/restart 全路径（httpx mock + stderr 错误分流）。
- 兼容修复：`tests/cli/test_cli_command.py::EXPECTED_COMMANDS` 与 count 从 13 升至 20。

**最终全量测试基线**（v3.0.0 Stage 7 后）：

- `tests/cli/`：114 passed（74 已有 + 40 新增）/ 0 failed
- `tests/agent/`：577 passed / 3 skipped / 0 failed
- `tests/`（非 agent）：5163 passed / 21 skipped / 0 failed

详细使用文档：`docs/16-quantnodes-cli使用指南.md`（约 180 行）。

### Stage 8 — Workflows 多智能体框架 (commits 9b1d1bf → 4aa4bfc → ca27b1b)

8 个 stage 把 AlphaGpt 工作流拆分为 StepAgent 通用框架 + 可注册 Workflow。

#### 新增

- **`QuantNodes/research/quant_alpha/workflows/`** — 多智能体 workflow 框架：
  - `step_agent.py`：StepAgent 抽象（per-stage LLM call + validation + reflection）
  - `parsers.py`：薄封装 nanobot StepAgent parsers
  - `registry.py`：`WorkflowRegistry` — dict-based 名称 → workflow factory
  - `tool.py`：`WorkflowTool` — 注册到 nanobot 的入口
  - `implementations/alpha_gpt.py`：`AlphaGptWorkflow` 从内嵌类移植到 StepAgent 框架
  - `implementations/logic_mining.py`：`LogicMiningWorkflow` — 三段式 agent（propose → compile → critique）
- **`WorkflowTool`** — 注册到 `quantnodes.tools` entry point，`quantnodes run_workflow <name>` CLI
- **MCTS WorkflowSpec** (`4aa4bfc`)：共享 cache + IC/IR metric、per-stage metric tracking
- **MCTSCache** (`89ae464`)：跨 call 复用 MCTS search，避免重复 LLM 调用

#### 测试 (Stage 8 + earlier Workflow commits)

- `tests/quant_alpha/test_step_agent.py` (35 tests)
- `tests/quant_alpha/test_workflow_registry.py` (24 tests)
- `tests/quant_alpha/test_mcts_workflow.py` (29 tests)
- `tests/quant_alpha/test_alpha_gpt_workflow.py` (42 tests)

### Stage 9 — Alpha-GPT Pipeline 端到端

把 quant_alpha 论文 pipeline 从 v2.x 单体内嵌类拆分为可独立调用的 6 个 stage。

#### 新增

- **`LogicDrivenPipeline`** (`172f190`)：端到端因子挖掘 pipeline
  - V2 run：momentum + volatility 双逻辑
  - 多轮迭代：`TerminationConfig` + `EarlyStopping` + `RoundFeedback` (commits `8cc52b1`, `c42ce1c`, etc.)
  - `min_ir_threshold=0.1` 默认 (`404e0d6`)
- **OperatorVocab 扩展** (`82db35f`)：加入 `returns` namespace；`vol → volume` 列别名 (`8147a94`)
- **Parser 升级**：
  - 支持 `shift` / `Ref` 到 `ALLOWED_OPERATORS` 白名单 (`b9d36dd`)
  - 修复 `rank` / `zscore` / `winsorize` 参数顺序 (`efb79cf`)
  - 支持 `ts_corr` / `ts_cov` + `corr` / `cov` 到 `RollingOp` (`b4bd3e7`)
  - 扩展 ALLOWED_OPERATORS 支持更多算子 (`0ee93a3`, `22923ce`)
- **互信息去重** (`692fe73`)：避免冗余因子入池
- **Rank IC + 换手率 join 修复** (`9a3a93e`)
- **Evaluator 优化**：
  - precompute returns, dedup cache, tree index (`e96d6ed`)
  - financial constraints (volume / 价格区间)
- **LLM 增强**：
  - 重试机制 + 超时控制 (`0a66068`)
  - temperature wire-through (`e9c9ab6`, `33f7d79`)
  - per-stage temperature control

#### V7 / V8 实验

- **V7** (`219520b`)：volume alias fix + 4-logic baseline
- **V8** (`c2c5f5c`)：6 logics (4 old + 2 new)
- **V8 sign-mismatch fix** (`0764a6e`)：`compiler` 强制 strict `sign_hint` for `direction=-1`

#### 修复 (selected 13 commits)

- `f9cbab3`: dedup sort by `|overall_score|` (防 negative-IR 因子被丢弃)
- `8869c0e`: `FactorMetrics.formula` AttributeError
- `0a5430a`: end-to-end pipeline 工作（真实 LLM）
- `7f49e9e`: formula validation 改为非阻塞
- `b8e3d82`: bypass critic LLM for final pool selection
- `2eb2f08`, `5bdca50`, `07e1878`: prompt 优化减少 JSON parse 失败
- `a73b39e`: LLM 输出截断根因 + 3 层 fix
- `b77f09a`: max_tokens 8192 → 16384, timeout 120s → 300s
- `7826a28`: 4-layer defense against LLM JSON truncation
- `83bd9ac`: inline JSON schema + LLM thinking cleanup

#### 测试 (Phase A-E — 7 commits)

- `f0ffca7` (Phase B baseline) → `c1405ea` (Phase D.1 alpha_logics 44→98%) → `561a369` (D.2 logic_driven_pipeline 50→100%) → `95a8d9f` (D.3 pipeline 71→78%) → `9116067` (D.4 g2_llm_only 71→98%) → `537c5f9` (D.5 logic_mining 79→88%) → `8ea86e6` (D.6 clickhouse_data_loader 56→68%)
- **最终整体覆盖率**：90% on quant_alpha/ — 详见 `TESTING.md`

### Stage 10 — Plugin 系统 + EventBus + DataNode 重构

#### 新增 (commits `bc398c0`, `52142df`, `e3eb067`)

- **Plugin 发现机制** (`bc398c0`)：通过 setuptools `entry_points` 自动发现插件
  - `quantnodes.tools` group: 16 个 quant tool 入口
  - `quantnodes.operators` group: builtin OperatorVocab
  - 新增 `register_all_quant_tools()` / `discover_external_tools()`
  - 测试 16 个 (`test_plugin_discovery.py`)
- **EventBus** (`52142df`)：跨组件事件总线
  - 事件类型：`FactorMined` / `QualityGatePassed` / `QualityGateFailed`
  - demo wiring：`factor.mined` 事件触发 Wiki 更新
  - 测试 13 个 (`test_event_bus.py`)
- **methods/ 与 agent/tools/ 去重** (`e3eb067`)：删除 6 个 v2.x 重复工具（统一以 agent/tools/ 为准）

#### DataNode 重构 (commits `f6e4924`, `ef8428c`)

- **`ClickHouseDataLoader` → `ClickHouseNode`** 统一重构：
  - 从 `research/factor_db/` 移到 `database_node/`
  - 7 层文档（架构 / API / 性能 / 测试 / 迁移 / FAQ / changelog）
  - `factor_db` 模块标记 deprecated，10 年 grace period
  - 测试 25 个 (`test_clickhouse_node.py`)

#### 测试

- `tests/core/test_plugin_discovery.py` (16 tests)
- `tests/core/test_event_bus.py` (13 tests)
- `tests/database_node/test_clickhouse_node.py` (25 tests)

### Stage 11 — Test Coverage Expansion (commits a3c178f → bcfdc64)

5 轮 +1042 测试，目标 LIVE 模块 ≥ 80% 覆盖。

| Round | Commit | 新测试 | 目标模块 |
|---|---|---|---|
| 1 | a3c178f | 204 | serialization / factor / control / llm / strategy / tools |
| 2 | a753513 | 203 | feedback / knowledge / viz / mcts / types |
| 3 | 6c740bf | 157 | mcts search / db nodes / ops engine / registry / zoo |
| 4 | 1b4ff7e | 183 | quality_gate / knowledge / evolution / advanced db |
| 5 | bcfdc64 | 205 | feedback channels / trajectory pool / evolution loop / viz figures / mcp_server |
| extra | 7b0964a | 129 | plugin / events / tools |

**全量基线 (v3.0.0 Stage 11)**：
- `tests/`（非 agent）：4494 passed / 21 skipped / 25 pre-existing failures（`test_llm_gateway.py` 网络 / retry 测试，无关）
- `tests/agent/`：574 passed / 13 skipped
- 0 个新引入的失败

### P0 修复 — 循环依赖 + 包结构清理 (commit 60cfdee)

- **解决 P0 循环依赖**：`research.quant_alpha.evaluation` ↔ `agent.tools.alpha_evaluate` 的循环 import
- **删除空目录** (`72817bd`)：`QuantNodes/research/quant_alpha/_legacy_3c/` 等空 agent 子目录
- **版本统一** (`72817bd`)：所有 `__version__` 统一从 `quantnodes.__version__` 读取
- **3 层配置澄清** (`2bebe8c`)：core / api / agent 3-tier layering 文档化
- **9 个 pre-existing test 失败修复** (`8e330bc`)

### 发布基础 (本 release)

- **Stage 8 + 9 + 10 + 11 累计 110+ commits**
- **`pyproject.toml`** 同步更新：`[project.urls]` 补 `Documentation` / `Repository` / `Changelog`；`[tool.setuptools.package-data]` 把 `SKILL.md` / `*.yaml` 模板打进 sdist
- **`QuantNodes/agent/skills_quant/__init__.py`** 补齐 — 之前缺 `__init__.py` 导致 6 个 SKILL.md 不被识别为包
- **`.github/workflows/python-publish.yml`** 升级到 Python 3.11 + trusted publishing（OIDC）ready
- **`pip install quantnodes==3.0.0`** 可立即使用；`pip install 'quantnodes[agent]'` 启用 nanobot agent / WebUI / MCP / 飞书 / cron 全部能力

## [2.8.0] - 2026-06-22

Dual-Engine Composite — allows LLM to write pandas or polars code
interchangeably. Polars remains the default high-performance engine;
pandas is the LLM-friendly alternative for easier code generation.

### Added

- **`_engine.py`**: `Engine` enum (`POLARS`/`PANDAS`/`AUTO`) +
  `detect_engine(code)` scanning imports to auto-detect engine.
- **`composite_dag_pandas_ops.py`**: 20 pandas-mirror composite ops
  (same names as polars ops, registered with `engine="pandas"`).
  Covers: neutralization (3), normalization (3), rolling regression (3),
  volatility (4), pairs (2), winsorize (3), time-series (2).
- **`sandbox_pandas_bridge.py`**: `detect_and_inject_context()` for
  sandbox auto-detect + context injection (pandas/polars df injection).
- **`CompositeSpec.engine`** field (default `"polars"`): tracks which
  engine a composite spec uses.
- **`_COMPOSITE_REGISTRY_PANDAS`**: isolated pandas composite registry
  (parallel to `_COMPOSITE_REGISTRY` renamed to `_COMPOSITE_REGISTRY_POLARS`).
- **`_ALLOWED_FUNC_NAMES_PANDAS`**: strict YAML whitelist for pandas
  templates (40 names, mutually exclusive with polars whitelist).
- **4 factor prompts** updated with dual-engine examples:
  `ic_analysis.py`, `group_backtest.py`, `correlation.py`,
  `backtest/factor_based.py`.

### Changed

- `composite_dag.py`: `is_composite_op(name, engine="any")`,
  `get_composite_spec(name, engine="any")`,
  `list_composite_ops(category, engine="any")` — all accept optional
  `engine` kwarg. `engine="any"` = backward-compatible (union of both
  registries). `load_composites_from_yaml` requires `engine:` field in
  YAML entries (default: `"polars"`).
- `operators/__init__.py`: re-export pandas composite functions +
  `_COMPOSITE_REGISTRY_PANDAS`.
- `sandbox.py`: `__init__(default_engine="polars")` +
  `_detect_engine(code)` method + `validate_and_execute` context injection.

### Testing

Added **+66 tests** (4716 → ~4782 passed):

- `tests/test_composite_dag_pandas_ops.py` (+43, pandas mirror of 20 ops)
- `tests/test_composite_dag_engine.py` (+8, engine field + registry switching)
- `tests/test_sandbox_pandas_bridge.py` (+10, auto-detect + context injection)
- `tests/test_composite_dag_pandas_engine.py` (+5, YAML dual-whitelist)

### Documentation

- `docs/22-算子系统设计与规范.md` §18: Dual-Engine Composite (design rationale,
  architecture, 20 op mapping, YAML dual whitelist, risk assessment).
- `docs/25-LLM算子层升级设计.md` §12: PR-QN-4 continuation.

### Fixed
- **`DataLoader` parquet 分发修复 (孤儿方法接入)**
  (`QuantNodes/research/factor_test/utils/data_loader.py:72`)
  - `load_parquet` 自定义以来已定义，但 `load_factor` / `load_custom`
    的扩展名分发从未接入 `.parquet` 分支 → 传 `.parquet` 路径会落到
    else（`load_factor` → `load_custom` → `raise ValueError`），方法不可达。
  - `load_factor` 新增 `.parquet` → `load_parquet(factor_dir)` 分支。
  - `load_custom` 新增 `.parquet` 分支，沿用 csv 的 dir 尾斜杠语义
    （尾斜杠 → `dir + filename`，否则 `filename` 视为完整路径）。
  - 新增 3 个测试 (`tests/research/test_data_loader_edges.py`):
    `TestLoadFactor.test_parquet` + `TestLoadCustom.test_parquet_dir`
    / `test_parquet_fullpath`。
  - 行为变化：parquet 从"不可用 (抛 ValueError)"变为"可用"，其余分支
    (h5/csv/npy) bitwise 不变。
- **`GroupAnalyzerNode` 支持 bool / 离散 / 轻度 ties 因子**
  (`QuantNodes/research/factor_test/nodes/group_analyzer_node.py:55`)
  - 原 `_calc_group_return` 在 n_unique < n_groups 时（如 `pl.when(cond)
    .then(-1).otherwise(+1)` 产出 30×-1 + 20×+1，或 3 unique 的离散因子）
    调用 `pd.qcut(..., labels=range(1, group+1), duplicates='drop')` 抛
    `ValueError: Bin labels must be one fewer than the number of bin edges`。
  - 改为策略模式：`_classify_factor` 按 dtype + n_unique 判别 → 2 个纯
    函数 handler：
    - `_group_ranked` — 连续或轻度 ties 因子（含 alpha-004 场景：
      7 unique × 50 行有大量 ties 但 n_unique >= group），统一用
      `pd.qcut(series.rank(method='first'), ...)` 破 tie。对无 ties
      纯连续因子与原 `pd.qcut(series, ...)` 行为 bitwise 等价
      （rank 单调变换保序），零回归。
    - `_group_discrete` — bool/离散因子 (n_unique <= 2 或 bool/integer
      dtype 且 n_unique <= 10)，按 value 比例分配组段 + 内部 seeded
      shuffle（`seed=yyyymmdd % 2**31`）保证可复现。
  - `_classify_factor` 返回种类从 3 (`continuous`/`low_tie`/`discrete`)
    合并为 2 (`ranked`/`discrete`)，dispatch 简化为 2 分支。
  - 原 `_group_continuous` (修复 ties 抛错) + `_group_low_tie` (修复
    ties 抛错) 合并为单个 `_group_ranked`。
  - `_calc_group_return` 改写为 dispatch 调度，循环内只调一次
    `factor_data.loc[t_i].dropna()`。
  - 新增 `tests/test_group_analyzer_bool.py` 覆盖 2 分支 + dispatch，
    含 alpha-004 真实 ties 场景。

### Changed
- **`FactorNeutralizeNode` 改 Chain of Responsibility (Phase 2.1)**
  (`QuantNodes/research/factor_test/nodes/factor_neutralize_node.py:65`)
  - 原 `_neutralize` 72 行 3 个 if/elif 分支 (industry only / risk only
    / both) 几乎相同 (90% 重复)，仅 X 设计矩阵组装不同。
  - 抽出 `nodes/neutralizers.py`：
    - `Neutralizer` (ABC) / `IndustryNeutralizer` / `RiskNeutralizer`
    - `build_neutralizer_chain(if_industry, if_risk, industry, risk_data)`
      → `List[Neutralizer]`，自动过滤 `is_active() == False` 的环节
    - `apply_neutralizer_chain(factor_i, chain)` 统一"按日期循环 +
      merge + OLS + 写残差"流程
  - `_neutralize` 退化为 4 行：构造 chain + 委托
  - 新增中性化类型 (如 `StyleNeutralizer`) 只需新增一个 `Neutralizer`
    子类，`_execute` 无需修改
  - 顺带修复 2 个 latent bugs (原代码从未被测试覆盖):
    1. `pd.get_dummies` 产出 bool dtype，`sm.add_constant` 报
       "numpy boolean subtract" 错误。chain 实现显式 `.astype(float)`
    2. 原 branch 3 (risk only) `pd.concat` 组装 X 时未转置，产生
       `(n_risks, n_stocks)` 而非预期的 `(n_stocks, n_risks)`，
       导致后续 `merge(left_index=True, right_index=True)` 错位。
       chain 实现统一 X 形状 (index=股票代码, columns=factors)
  - 新增 `tests/research/factor_test/nodes/test_neutralizer_chain.py`
    覆盖 5 类 (ABC / Industry / Risk / Chain build / Chain apply /
    Backward compat / E2E)，37 个测试

- **`FactorPreprocessNode` 改 Strategy pattern (Phase 2.2)**
  (`QuantNodes/research/factor_test/nodes/factor_preprocess_node.py:85`)
  - 原 `_preprocess_vectorized` 102 行硬编码 3 类 if 链 (missing fill /
    de-extreme / normalise)，每类多个 method 分支。
  - 抽出 `nodes/preprocess_strategies.py`：
    - `MissingFillStrategy` (ABC) / `PassThroughMissing` / `IndustryAverageMissing`
    - `DeExtremeStrategy` (ABC) / `PassThroughExtreme` /
      `MedianAbsoluteDeviationExtreme` / `PercentileShrinkExtreme`
    - `NormStrategy` (ABC) / `PassThroughNorm` / `ZScoreNorm` / `RankToNormalNorm`
    - 工厂函数 `build_missing_strategy / build_extreme_strategy /
      build_norm_strategy / build_preprocess_strategies`
  - `_preprocess_vectorized` 退化为 3 行 dispatch:
    `result = missing_s.apply(result, industry=industry)`
    `result = extreme_s.apply(result, ...)`
    `result = norm_s.apply(result)`
  - 新增策略类型 (如 winsorize) 只需新增一个 Strategy 子类，
    `_preprocess_vectorized` 不变
  - 与原 if 链 bitwise 一致 (向后兼容) — 4 个 `TestBackwardCompat` 测试
    验证 zscore / median+pct_shrink / pct_shrink 路径与原公式输出
    decimal=10 一致
  - 新增 `tests/research/factor_test/nodes/test_preprocess_strategies.py`
    覆盖 7 类 (ABC / 3 strategy / Factory / E2E / BackwardCompat)，
    38 个测试

- **CLI 改 Command pattern + CommandRegistry (Phase 3.1)**
  (`QuantNodes/cli/command.py`, `QuantNodes/cli/__init__.py`)
  - 原 `cli/__init__.py:main` 34 行 if/elif ladder 派发到 13 个 `cmd_*`
    函数，且 `_build_parser` 90 行手写 13 个子命令的 argparse 构造。
  - 新增 `cli/command.py`：
    - `Command` (ABC)：`name` / `description` / `add_arguments(subparsers)`
      / `run(args) -> int` 四要素
    - `CommandRegistry`：`register` (重复同名 / 空 name 抛 ValueError，
      同实例幂等) / `get` / `all` / `names` / `clear` / `__len__` /
      `__contains__`
  - 13 个子命令各暴露一个 `Command` 子类 (`InitCommand` / `RunCommand` /
    `ChatCommand` / `EvolveCommand` / `Factor{Info,Best,Visual,Dashboard,
    DataFetch,RagEval,RagShow}Command` / `VersionCommand` / `HelpCommand`)，
    自身负责 `add_arguments` (复用 `_helpers.py` 的 add_* builders)。
  - `commands/__init__.py` 建模块级单例 `COMMAND_REGISTRY`，import 时按
    固定顺序注册全部 13 个 command。
  - `_build_parser` 退化为遍历 `COMMAND_REGISTRY.all()` 调
    `cmd.add_arguments(subparsers)`；`main()` 退化为
    `COMMAND_REGISTRY.get(args.command).run(args)` (缺省/未知回退 help)。
  - 新增子命令只需写 Command 子类 + 注册一行，无需改 main / _build_parser。
  - 向后兼容：所有 `cmd_*` 函数 + `start_api_server` /
    `start_frontend_server` / `_load_runner_from_config` 仍从
    `QuantNodes.cli` 原样 re-export，调用方式不变。
  - 新增 `tests/cli/test_cli_command.py` 覆盖 6 类 (Command ABC /
    Registry / 模块 registry 内容 / _build_parser / main dispatch /
    BackwardCompat)，25 个测试。

- **算子查询改 Facade pattern (Phase 3.2)**
  (`QuantNodes/operators/facade.py`)
  - QuantNodes 有 3 个并列算子注册表 (docs/26 §3.3)：L0 内置
    `_OPERATOR_REGISTRY` (factor_functions) / L1 复合 DAG
    `_COMPOSITE_REGISTRY` / L2 自定义 `_CustomOperatorRegistry`。调用方
    (agent executor / loader) 此前需分别 import `get_operator` /
    `is_composite_op`+`get_composite_spec` / `CustomOperator.get` 三套不同
    入口才能完成"按名查算子"。
  - 新增 `operators/facade.py`：
    - `OperatorFacade`：统一**只读**门面，方法 `resolve` (L0+L2 级联
      callable) / `info` / `get_composite` + `is_composite` (L1 隔离) /
      `exists` / `kind` (返回 custom/builtin/composite/None) /
      `list_all` (三层去重合并) / `documentation` / `composite_doc_for_llm`
    - 模块级单例 `operator_facade`，无状态缓存 → 运行时新注册的 custom /
      composite 算子实时可见
  - **只委托不改行为**：全部转调既有函数 (`get_operator` /
    `operator_info` / `get_composite_spec` / `list_operators` /
    `generate_documentation` 等)，与旧 API bitwise 一致；查询级联优先级
    (custom → builtin → composite) 与旧逻辑完全相同。
  - **只读边界**：注册仍走各自的 `@register_operator` /
    `@composite_operator` / `CustomOperator`，写路径不收敛，保持 L1 严格
    隔离语义 (见 composite_dag.py:137-146)。
  - `operators/__init__.py` 导出 `OperatorFacade` + `operator_facade`。
  - 新增 `tests/operators/test_facade.py` 覆盖 7 类 (import / resolve /
    composite / exists+kind / list_all / documentation / 自定义实时可见)，
    26 个测试。

### Added
- **`SingleFactorTestConfig` 流式构造器 (Phase 3.4 Builder)**
  (`QuantNodes/research/factor_test/config_builder.py`, 新增)
  - `SingleFactorTestConfigBuilder` 提供链式 setter: `.factor() /
    .dates() / .sample() / .preprocess() / .neutralize() / .tradable() /
    .ic() / .groups() / .longshort() / .score() / .risk_corr() /
    .output() / .feedback() / .quality_gate() / .evolution() /
    .data_path() / .load_keys()`, 终接 `.build()` 触发 pydantic 校验。
  - 默认值全部委托各 `*Setting` 的 pydantic 默认 (单一真值源, 不重复)。
  - 缺 `factor` (唯一必填) 时 `.build()` 抛 `ValueError`。
  - 不改 `SingleFactorTestConfig` / `config.py`; 现有直接构造方式不变,
    全新增量。
  - **真实样例**: `run_evolution_e2e.py::_build_config` 从 37 行嵌套
    构造改写为 35 行流式 builder (净增 6 行因为加注释, 但去掉 7 个
    `*Setting` import)。
  - 新增 `tests/research/factor_test/test_config_builder.py` (21 tests):
    链式返回值、最小/必填/缺必填报错、默认值来自 pydantic、setter 字段
    透传、pydantic ValidationError 触发、与等价直接构造 bitwise 一致。
- **顶层 `DataSource` ABC + DB 节点工厂 + 文件格式 Adapter (Phase 3.3)**
  - **`DataSource` 顶层抽象基类** (`QuantNodes/core/data_source.py`, 新增)
    - 最小化标记基类: 统一 `close()` 生命周期 + `__enter__/__exit__`
      上下文管理器 + "产出 pd.DataFrame" 语义约定。
    - 不强套 SQL 语义 (DB) 或面板矩阵语义 (文件), 两个数据接入子树
      各自保留专用接口, 避免泄漏抽象。
  - **DB 节点工厂** (`QuantNodes/database_node/factory.py`, 新增)
    - `create_db_node(source, **params)` 按字符串注册表创建 6 个后端
      (sqlite/duckdb/mysql/clickhouse/csv/parquet); 未知 source 抛
      `ValueError`。
    - `register_db_node(source, builder)` / `available_sources()` 供扩展。
    - `BaseDBNode` 改为继承 `DataSource`, 新增 `close()` 默认委托
      `disconnect()` (纯增量, 6 个子类无需改动)。
    - `database_node/__init__.py` 导出 `create_db_node` /
      `register_db_node` / `available_sources`。
    - `agent/tools/config_backtest.py` 的 `_build_db_node` /
      `_build_embedded_node` 保留方法名, 内部改为委托工厂 (conn.ini /
      path 解析仍留在调用方), 行为等价。
  - **文件格式 Adapter** (`QuantNodes/research/factor_test/utils/file_loaders.py`, 新增)
    - `FileFormatLoader(DataSource, ABC)` + 4 个适配器
      (`H5Loader/CSVLoader/NPYLoader/ParquetLoader`) + `build_file_loader(ext)`
      / `register_file_loader` / `available_extensions` 注册表。
    - `DataLoader` 的 `load_h5/csv/npy/parquet` + `load_factor` 分发改为
      委托 adapter; 公共方法签名/输出 bitwise 不变; H5 经 `store_getter`
      回调复用 `_h5_stores` 缓存 (保 Phase H3 优化)。
    - `load_custom` 的扩展名分支保持不变 (含 csv/parquet 尾斜杠语义)。
  - **新增测试** (+39):
    - `tests/core/test_data_source.py` (8): ABC 不可实例化 / close /
      上下文管理器 / 子类关系。
    - `tests/test_database_node_factory.py` (14): 6 后端构造 / 未知抛错 /
      register / available_sources / `isinstance(node, DataSource)`。
    - `tests/research/test_file_loaders.py` (17): 4 adapter 加载 / registry
      分发 / H5 store_getter 缓存复用 / `isinstance(loader, DataSource)`。
- **端到端集成测试 (Option D 巩固)**
  (`tests/research/factor_test/e2e/test_pipeline_bool_factor.py`)
  - 15 个 e2e 测试验证 Phase 1+2 重构后的 3 个节点
    (preprocess → neutralize → group_analyzer) 在 bool / 离散 /
    连续因子上的端到端行为
  - 6 个 test class: bool / low_tie / continuous / group_counts /
    output_keys / floor_mode
  - 验证场景: alpha-004 风格 30×-1 + 20×+1、7 unique ties、
    不同 group 数 (2/3/5/10)、4 种 preprocess 组合、
    industry neutralize、floor_mode='last'
- **设计模式审计文档 (Option D)**
  (`docs/26-设计模式重构与审计.md`)
  - 总结 Phase 1+2 已应用的 7 个 GoF 模式 (Null Object / Decorator /
    Builder / Visitor / Chain of Responsibility / Strategy)
  - 调研结论: Abstract Factory 在 QuantNodes 适用度有限 (无"一族
    互相依赖的产品族"场景), 改用 Facade + Simple Factory
  - Phase 3 路线: CLI Command pattern (推荐下一步), DataSource
    Factory + Adapter, Operator Facade
  - 模式选择决策树: 9 种场景 → 推荐模式, 供未来参考

---

## [2.7.0] - 2026-06-21

LLM Operator Layer Upgrade — implements 4 PRs from `docs/25-LLM算子层升级设计.md`
for llmwikify Loop v4 integration. Adds Composite DAG abstraction level
between primitive ops (L0) and business semantics (L3).

### Added

- **PR-QN-1**: `CodeSandbox` accepts instance-level `allowed_imports` /
  `blocked_imports` parameters. Class-level whitelist/blacklist
  extensible without monkey-patching. Default behavior unchanged.
  See `docs/24-核心功能框架设计.md` §15.
- **PR-QN-2**: `PipelineRunner` accepts `extra_phases` plugin mechanism
  (`__init__(specs=...)` / `from_dict(extra_phases=...)`). Downstream
  systems (e.g. llmwikify Loop v4) can inject custom stages after the
  standard 12 phases. `run()` now iterates `self._specs` instead of
  hardcoding `PIPELINE_SPEC`. See `docs/24-核心功能框架设计.md` §16.
- **PR-QN-3a**: Composite DAG core — `@composite_operator` decorator,
  `ParamSpec` / `CompositeSpec` dataclasses, `_COMPOSITE_REGISTRY`
  isolated registry (no pollution to main `_OPERATOR_REGISTRY`),
  `load_composites_from_yaml()` with AST parsing + function name
  whitelist (rejects bare `exec` risk), `get_composite_doc_for_llm()`
  producing LLM-friendly markdown. See `docs/22-算子系统设计与规范.md` §17.
- **PR-QN-3b**: 20 built-in composite ops covering quant research
  common algorithms:
  - Neutralization (3): `industry_neutralize`, `market_neutralize`,
    `subindustry_neutralize`
  - Cross-sectional normalization (3): `zscore_xs`, `rank_xs`, `scale_xs`
  - Rolling regression (3): `rolling_beta`, `rolling_ols_simplified`,
    `rolling_residual`
  - Volatility (4): `parkinson_vol`, `garman_klass_vol`,
    `yang_zhang_vol`, `realized_vol`
  - Pairs trading (2): `pair_zscore`, `pair_ratio`
  - Winsorize/outlier (3): `winsorize`, `mad_outlier`, `zscore_clip`
  - Composite time-series (2): `decay_linear_xs`, `momentum_accel`

  Polars 1.40+ API adaptation notes in `docs/22` §17.6.1 (corrections
  from original design spec: `Expr.group_by` → `.over()`, `rolling_corr`
  → OLS closed-form, `window=` → `window_size=`).

### Fixed

- **L1**: `resample_trade_date` accepts position aliases
  (`'beg'` / `'start'` / `'first'` → `'begin'`, `'last'` → `'end'`),
  no longer raises on natural shorthand.
- **L2**: `offset_date` explicit overflow check, fixing pandas iloc
  negative-index wrap-around silent bug. `n=-1` previously returned the
  last date silently; now raises `IndexError` with details, or clips
  to boundary when `if_modify=True`.
- **L3**: `FactorTestReport` raises `ValueError` for unknown `format`
  at runtime. Previously silently skipped (writing `'html'` produced
  no file and no error).

### Documentation

- New `docs/25-LLM算子层升级设计.md` (1338 lines, full PR-QN-1/2/3 design).
- Updated `docs/24-核心功能框架设计.md` §15 (PR-QN-1), §16 (PR-QN-2).
- Updated `docs/22-算子系统设计与规范.md` §17 (Composite DAG chapter).

### Testing

Added 6 test files, **+100 tests** (4608 → 4716 passed):

- `tests/test_sandbox_allowed_imports.py` (+17, PR-QN-1)
- `tests/test_pipeline_plugin.py` (+10, PR-QN-2)
- `tests/test_composite_dag.py` (+24, PR-QN-3a)
- `tests/test_composite_dag_ops.py` (+43, PR-QN-3b, includes 20-parametrize)
- `tests/research/factor_test/utils/test_date_utils_edge_cases.py`
  (+10, L1 alias + L2 overflow)
- `tests/research/factor_test/nodes/test_score_report.py` (+3, L3 format)

### Changed (Internal)

- `QuantNodes/operators/__init__.py`: 8 new composite re-exports.
- `QuantNodes/research/factor_test/pipeline_runner.py`: `__init__`
  accepts `specs`, `from_dict` accepts `extra_phases`, `run()` uses
  `self._specs`.

---

## Historical Versions

Earlier version history available via `git log`. This project adopted
structured CHANGELOG from May 2026; versions before 2.7.0 lack structured
entries.

[Unreleased]: https://github.com/sn0wfree/QuantNodes/compare/v2.8.0...HEAD
[2.8.0]: https://github.com/sn0wfree/QuantNodes/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/sn0wfree/QuantNodes/compare/v2.6.0...v2.7.0
