# QuantNodes 全面重构计划 (Tier 1-4)

> **作者**: Claude Code (opencode)
> **开始日期**: 2026-07-06
> **目标分支**: `dev/repro-merge-2026-07-04`
> **基线 tag**: `pre-m1-bugfixes` (commit `081bb21`)
> **预计工作量**: 1-2 周，分 4 个 Session

---

## 背景

`QuantNodes/research/` 包经过 2026-07-04 的大规模 vendor 迁移（commit `3c50b93`）后，存在三类问题：

1. **Latent bugs**: 4 个未触发的接口缺陷（如 `track_b_checkpoint.json` 缺产）
2. **死代码 / 孤儿**: ~2K LoC 未引用代码（`_legacy_3c/`、pipeline stubs、`alpha101_design`/`alpha158_design`）
3. **架构分裂**: Paper → Factor → Strategy 三层只在第一层贯通，Strategy 层与生产编排器完全脱钩

本计划分 4 个 Milestone / 6 个 PR 逐次清理和重构。

---

## 整体架构（当前）

```
┌──────────────────────────────────────────────────────────────────┐
│                    Paper Reproduction Layer                       │
│  scripts/run_101_alphas_v2.py:976 → PaperStage                   │
│  └─ paper_understanding/llm_extraction/orchestrator.py:129        │
│     run_one_paper(paper_id, source_path, output_root)             │
│     ├─ Stage 0: stage0_ingest → parsed.md                        │
│     ├─ Stage 1.1: section_detector → sections                    │
│     ├─ Stage 1.2: planner → plan.json                            │
│     ├─ Stage 1.3: track_a → track_a.json                         │
│     ├─ Stage 1.4: track_b → track_b_pass1.json + pass2.json      │
│     │              ⚠ 不产出 track_b_checkpoint.json              │
│     └─ preview.md + draft factor.yaml                            │
└──────────────────────────────────────────────────────────────────┘
                              ↓ signal (Signal)
┌──────────────────────────────────────────────────────────────────┐
│                         Factor Layer                              │
│  scripts/run_101_alphas_v2.py:642 → FactorStage.run_one_factor   │
│  ├─ load_formula_brief(idx) ← track_b_checkpoint.json           │
│  ├─ llm_code_react(name, formula_brief, df, llm)                 │
│  ├─ QuantNodesBacktest.run(code, h5_path, signal)                │
│  ├─ ResultFactory.success(...) → FactorResult                    │
│  └─ Sinks (3 路并行):                                            │
│     ├─ SingleJsonSink  → output_dir/single_factor_<NNN>.json     │
│     ├─ YamlDuckdbSink  → factors/.../factor.yaml + factor.duckdb │
│     │                    ⚠ factor_wide=None, duckdb 只存 metrics │
│     └─ RecordStage.record (累积)                                 │
│  └─ BatchSummarySink.write_batch → multi_alpha_*.{json,md}        │
└──────────────────────────────────────────────────────────────────┘
                              ↓ WikiFactor + backtest metrics
┌──────────────────────────────────────────────────────────────────┐
│                       Strategy Layer (DISCONNECTED!)              │
│  scripts/reproduce_table4_*.py → AlphaPipeline.run              │
│  ├─ AlphaGptWorkflow.run() (5 agents × N rounds)                 │
│  └─ WikiFactorProxy.store_factor(formula) → wiki/factor/*.md     │
│  ⚠ 与 run_101_alphas_v2.py 完全脱钩，无 shared sink/registry       │
│  ⚠ persist/strategy_library.py 不存在                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                  Cross-Cutting (横切关注点)                       │
│  telemetry.py: Counter + recent-events (无 consumer)               │
│  wiki.py: WikiFactorProxy (1155 LoC, 含自循环)                   │
│  common/llm/: StreamableLLMClient + 9 个 llm helper 模块          │
│  common/config.py: 配置加载（双路径 llmwikify/.quantnodes）       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Milestone 总览

| Milestone | 内容 | 工作量 | 风险 | PR | 状态 |
|---|---|---|---|---|---|
| **M1** | Tier 1: 6 个 latent bug 快速修复 | 0.5 天 | 低 | PR1 | ✅ 完成 |
| **M2** | Tier 2: 死代码清理（~3000 LoC） | 0.5 天 | 低 | PR2 | ✅ 完成 |
| **Phase B** | H5 key contract bug fix | 0.1 天 | 中 | (合并到 PR5) | ✅ 完成 |
| **M3.2** | LLM 配置统一 (5-tier 优先级) | 0.3 天 | 中 | PR5 | ✅ 完成 |
| **M3 主** | Tier 3.1: backtest 双包合并 | 1 天 | **高** | PR4 | ✅ 完成 |
| **M3.3** | 新建 strategy_library.py | 0.3 天 | 中 | PR5 | ✅ 完成 |
| M3 前置 | Tier 3.5: WikiFactor 类型统一 | 0.5 天 | 低 | PR3 | ✅ 完成 (PR6.5) |
| **M3.4** | run_101 接入 strategy sink | 0.5 天 | 中 | PR5 | ✅ 完成 |
| M3 后置 | 删 backtest_pkg/ shim | 0.5 天 | **高** | PR5 | ✅ 完成 |
| **M4.1 (PR6)** | SignalV2 — TradeSignal 重命名 + cross-layer bridge | 1 天 | 中 | PR6 | ✅ 完成 |
| M4.2 | 配置统一 + wiki.py 拆分 + sink async | 2 天 | **高** | PR6 | ⏳ Future |

每个 Milestone 完成后打 tag (`post-m1-bugfixes` / `post-m2-deadcode` / 等)，验证通过后继续。

---

## Milestone 1: Tier 1 快速修复 (PR1)

**目标**: 修复 4 个 latent bug + 2 个清理项，零行为变化。

### M1.1 — 修 `track_b_checkpoint.json` 缺产 bug

**文件**: `QuantNodes/research/paper_understanding/llm_extraction/orchestrator.py`

**问题**: Stage 1.4 写 `track_b_pass1.json` + `track_b_pass2.json`，但 `run_101_alphas_v2.py:282-284` 断言 `track_b_checkpoint.json` 存在。

**修复**: 在 `track_b_pass1.json` 写入后，同步写一份 `track_b_checkpoint.json`（同 schema）。

```python
# orchestrator.py 在 write_track_b_pass1 末尾加:
ckpt_path = output_root / paper_id / "track_b_checkpoint.json"
shutil.copy(pass1_path, ckpt_path)
```

### M1.2 — 修 YamlDuckdbSink 缺 factor_wide

**文件**: `QuantNodes/research/sink/yaml_duckdb.py:59-127`

**问题**: `write_one(result)` 永远传 `factor_wide=None`，DuckDB 没时序数据，无法重做回测。

**修复**: 在 sink 入口补 `wide_from_long` pivot。

### M1.3 — 删 `preload_market_data` 重复

**文件**: `scripts/research/run_101_alphas_v2.py:179-196`

**问题**: 与 `pipeline/data_loader.py:57 load_and_build_df` 重复。

**修复**: 删 18 行，调用方改用 `load_and_build_df`。

### M1.4 — 删 wiki.py 自循环

**文件**: `QuantNodes/research/wiki.py:235-237`

**修复**: 删 3 行 `from QuantNodes.research.wiki import WikiFactorProxy`。

### M1.5 — 删空目录

```bash
rmdir QuantNodes/research/llm_extraction/ 2>/dev/null || true
```

### M1.6 — print() → logger

**文件**: `QuantNodes/research/pipeline/persist.py:140-141`

**修复**: `print(...)` → `logger.info(...)`。

### M1 验证

```bash
# 1) 全量测试不应退化
.venv-mig/bin/python -m pytest tests/research/ tests/agent/ \
  --tb=short --timeout=120 -q 2>&1 | tee /tmp/m1_tests.log

# 2) 1 alpha smoke (mock-free，真 LLM)
.venv-mig/bin/python scripts/research/run_101_alphas_v2.py \
  --start 1 --end 1 --no-delay \
  --track-b /home/ll/llmwikify/quant/factors/101_alphas/data/track_b_checkpoint.json \
  --output-dir /tmp/m1_alpha \
  --factors-dir /tmp/m1_alpha/factors 2>&1 | tee /tmp/m1_alpha.log

# 3) Tag M1 完成
git tag post-m1-bugfixes
```

---

## Milestone 2: Tier 2 死代码清理 (PR2)

**目标**: 删除约 3000 行未引用代码，零行为变化（只删除用户不可见的死代码）。

### M2.1 — 删 `_legacy_3c/` (1490 LoC) + shim

- 删 `QuantNodes/research/_legacy_3c/` (4 文件)
- 改 `QuantNodes/research/__init__.py:33-119` 删除 shim
- `__all__` 移除 9 个 legacy symbol

### M2.2 — 删 pipeline 全部 stub

- `pipeline/cli/{__init__,__main__}.py`
- `pipeline/runner.py`
- `pipeline/workspace.py`
- `pipeline/stages/{backtest,persist_factor,paper_understanding,base}.py`
- 保留 `pipeline/stages/codegen.py:23 llm_code_oneshot`（真实现）

### M2.3 — 迁 `alpha101_design/` + `alpha158_design/` 到 test_fixtures/

- 新建 `QuantNodes/research/test_fixtures/alpha_design/`
- 移 2 个 design 子包 → test_fixtures/alpha_design/
- 加 `__init__.py` + `README.md`（说明：design philosophy + few-shot examples 备用）
- 从 `quant_alpha/__init__.py` 移除 re-export

### M2.4 — 删 `_make_factor_dir_name`

**文件**: `scripts/research/run_101_alphas_v2.py:213-216` (4 行)

### M2.5 — 删 async `generate_factor_code`

**文件**: `QuantNodes/research/codegen/agent/codegen_pipeline.py:156-?`（async 版本，仅 sync 被用）

### M2.6 — 删 `compile_to_code_react` (deprecated)

**文件**: `QuantNodes/research/codegen/react_engine.py:138-?`（约 300 行）

1. grep 所有 `compile_to_code_react` caller
2. 改 caller 用 `from QuantNodes.research.codegen.react_runner import llm_code_react`
3. 删函数

### M2 验证

```bash
# 1) 全量测试
.venv-mig/bin/python -m pytest tests/research/ tests/agent/ \
  --tb=short --timeout=120 -q 2>&1 | tee /tmp/m2_tests.log

# 2) 5 alpha smoke (mock-free 真 LLM)
.venv-mig/bin/python scripts/research/run_101_alphas_v2.py \
  --start 1 --end 5 --no-delay \
  --track-b /home/ll/llmwikify/quant/factors/101_alphas/data/track_b_checkpoint.json \
  --output-dir /tmp/m2_alpha \
  --factors-dir /tmp/m2_alpha/factors 2>&1 | tee /tmp/m2_alpha.log

# 3) Tag M2 完成
git tag post-m2-deadcode
```

---

## Milestone 3: Tier 3 核心整合 (PR3-5)

### M3 前置 (PR3 / PR6.5) — WikiFactor 类型统一 ✅

**Commit `22302ef` · Tag `post-wikifactor-v2-merge`**

**问题**：仓库内有 2 个 `WikiFactor` dataclass：
- `QuantNodes.research.wiki.WikiFactor` (21 字段, **生产代码 10+ 文件使用**)
- `QuantNodes.research.paper_understanding.schemas.WikiFactor` (6 字段, **仅 4 个测试文件 + 2 manifest 字符串**, 0 生产 caller)

**方案 — 字段合并**：
- WikiFactor 扩展到 23 字段 (21 原有 + `factor_params: Dict[str, Any]` + `status: str = "draft"`)
- `schemas.WikiFactor` 和 `schemas.WikiStrategy` 全部删除
- `schemas.py` 仅保留 `BacktestResult` + `FactorBacktestResult` (生产用)
- 4 个 production caller 主动填充新字段（避免默认值的语义混淆）

**关键技术决策**：
- ✅ **不用 discriminated union** (两个 `WikiFactor` 不是 super/sub set, 而是字段互补)
- ✅ **生产端主动填充** `factor_params` / `status`（而不是用 dataclass 默认值）
- ✅ **`_render_factor_markdown` / `_parse_factor_from_page` 双向兼容**：旧 markdown 文件缺新字段时回退默认值
- ✅ **0 上游 llmwikify blocker** (grep `/home/ll/llmwikify/src` 验证 upstream 无 WikiFactor 类)

**Migration**（11 文件, +223 / -95 = +128 LoC）：
- `wiki.py`: 加 2 字段 + module docstring + `_render_factor_markdown` (2 行) + `_parse_factor_from_page` (新字段解析)
- `schemas.py`: 删 `WikiFactor` (15 LoC) + `WikiStrategy` (23 LoC) + V2 注解 docstring
- 4 production caller 主动填充：`report_reproducer._store_verified_factor` / `quant_alpha.pipeline._to_wiki_factor` / `logic_driven_pipeline._persist_to_wiki` / `agent.tools.wiki.WikiTool._store_factor`
- 测试迁移：删 `TestWikiFactor` (2) + `TestWikiStrategy` (1) + `test_wiki_factor_with_status` + `test_wiki_strategy_with_factor_refs` (2)，新增 `TestWikiFactorV2` (7 tests) + 2 个 V2 测试 in test_repro_integration.py
- Manifest 更新：`test_module_inventory.py` exports 列表 + `test_imports.py` line 162

**Verified**：
- pytest baseline: **3146 passed (+4) / 17 failed / 14 errors** (17F 全是 pre-existing)
- 1-alpha smoke: 1/1, 20.9s, IC=-0.0330
- WikiFactor round-trip via markdown: `factor_params` + `status` 完整保留

### M3 主 (PR4) — backtest 双包合并

**目标**: 删 `backtest_pkg/`，全部迁到 `backtest/`。

1. 列出所有 `backtest_pkg.*` caller
2. 收敛 metrics 字段从 30+ 到 6 核心 (`ic_mean, icir, rank_ic_mean, win_rate, annual_return, max_drawdown`)
3. 给每个 caller 写 migration shim
4. 删 `backtest_pkg/` (4500 LoC)

### M3 后 (PR5) — LLM 配置 + strategy_library + 端到端

**M3.2 — 统一 LLM 配置**:
- 删 `common/llm_factory.py`
- 全部走 `common/llm/client.py`
- `~/.quantnodes/llm.json` 优先 + `~/.llmwikify/llmwikify.json` fallback

**M3.3 — 创建 `persist/strategy_library.py`**:
- 模仿 `factor_library.py`
- `write_strategy_yaml` / `read_strategy_yaml` / `update_index`
- 单元测试 + 集成测试

**M3.4 — 接入 strategy 层** (已细化，详见下方):

**a. CLI flag**:
```python
parser.add_argument(
    "--strategy-mode",
    choices=["off", "per_alpha", "after_batch"],
    default="off",
    help="Strategy layer integration mode (default: off = backward compat)",
)
parser.add_argument(
    "--strategies-dir",
    type=Path,
    default=None,
    help="Output base dir (default: PROJECT_ROOT/quant/strategies)",
)
```

**b. RunConfig 新增**:
```python
strategy_mode: str = "off"
strategies_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "quant" / "strategies")
```

**c. alpha→signal_type 启发式**（branch + fallback `factor_rank`）:
| 关键词 (lower) | signal_type |
|---|---|
| `rsi` | `rsi` |
| `ma` | `ma_cross` |
| `volatility` / `vol` | `volatility` |
| `momentum` / `mom` | `momentum` |
| `factor` / `rank` | `factor_rank` |
| (其他 101 alpha) | `factor_rank` (fallback) |

**d. strategy_name 命名**:
```
"101_alphas_minimal_NNN_factor_rank"
例: "101_alphas_minimal_005_factor_rank"
```

**e. signal_params 派生**（从 backtest metrics）:
```python
{
    "ic_mean": fr.backtest.get("ic_mean"),
    "icir": fr.backtest.get("icir"),
    "rank_period": 20,                       # 写死 default
    "factor_col": fr.signal.name,             # 实际因子名（for factor_rank）
    "factor_direction": config.factor_direction,
}
```

**f. 三种 mode 行为**:
| mode | 触发时机 | 写入 |
|---|---|---|
| `off` | 无 | 完全无操作（向后兼容） |
| `per_alpha` | 每个 factor backtest 成功后（`run_one_factor` L717 之后） | 单 factor strategy YAML + DuckDB |
| `after_batch` | 所有 factor 跑完后（`run` L584 `_write_summary` 之前） | 聚合 composite strategy YAML（weights 均分） |

**g. 5-alpha smoke × 2 mode**:
- `--strategy-mode per_alpha --start 1 --end 5 --no-delay`
- `--strategy-mode after_batch --start 1 --end 5 --no-delay`
- 检查 `quant/strategies/101_alphas_minimal_001_factor_rank/strategy.yaml` 等存在
- 检查 `quant/strategies/101_alphas_minimal_composite/strategy.yaml` 存在
- **不** 跑 101 alpha（M3.4 PR 风险未明）

**h. 风险与缓解**:
| 风险 | 缓解 |
|---|---|
| CLI flag 改动破坏老用户 | default=off 字节相同 |
| `after_batch` 在 0 个 success 时写出空文件 | 早 return，仅 count > 0 才写 |
| factor_name 含中文导致策略名 sanitize 后冲突 | 复用 `_SAFE_KEY_RE` 模式 |
| YamlDuckdbSink + strategy_library 并发写 race | 串行执行（workers ≤ 3 已有 lock），DuckDB 单独库无 race |

---

## Milestone 4: Tier 4 架构重构 (PR6)

### M4.1 — 配置文件统一

- 所有 `~/.llmwikify/*` 路径 → `~/.quantnodes/*`
- 加迁移脚本 `~/.quantnodes/migrate_from_llmwikify.py`

### M4.2 — SignalV2 重设计 ✅ (PR6 完成)

**Commit `0523b1e` · Tag `post-signal-v2-trade-signal-rename`**

**设计原则**：**Layer-appropriate single canonical dataclass + 显式 cross-layer bridge**
（NOT discriminated union — 两个 Signal 字段零重叠，union 会成 Optional[] 反模式）

**3 个 canonical 类型**：
| Layer | 类型 | 位置 |
|---|---|---|
| Paper | `Signal` (id, name, formula_brief, metadata) | `signal_source/base.py` |
| Trade | `TradeSignal` (code, signal_type, strength, price, date) — alias: `Signal` | `backtest/strategy_node.py` |
| Classifier | `SignalType` (enum) | `paper_understanding/contracts.py` |

**Cross-layer bridge** — `QuantNodes/research/signal_source/bridge.py`:
- `classify_paper_signal(signal: Signal) -> SignalType` — heuristic (rsi > volatility > momentum > ma_cross > factor_rank default)
- `classify_name(name: str) -> SignalType` — plain string variant
- `signal_type_to_strategy_class(sig_type: SignalType) -> type[StrategyNode] | None`

**变更**（9 文件，+450 LoC）：
- 1 新文件 `bridge.py` (130 行)
- 1 新测试文件 `test_signal_bridge.py` (220 行, 46 tests)
- 4 production 文件改名: `Signal` → `TradeSignal` (backtest/__init__ + config_strategy + strategy_node + research/backtest/strategies)
- 1 脚本换用 bridge: `run_101_alphas_v2.py` (_alpha_to_signal_type 现在 delegate to bridge)
- 2 测试文件 (test_strategy_node + test_strategy_library YAML string)

**向后兼容**：`Signal = TradeSignal` 1 行 alias 让所有现有 `from ...strategy_node import Signal` 继续工作。

**验证**：
- pytest: 3142P / 17F / 14E (+46 new tests, baseline 一致)
- 1-alpha smoke: 1/1 (15.2s)
- 5-alpha smoke (per_alpha): 5/5 (78.2s, 5 strategies + 5 DuckDB)

### M4.3 — 拆 wiki.py (1155 LoC) ⏸

- `wiki/{factor,logic,strategy,proxy}_wiki.py`
- 风险：上游 llmwikify 已有 WikiFactor 6-field 类型事故先例，迁前需 ping upstream

### M4.4 — Sink 异步化

- 3 sink 改 async
- 支持流式写

---

## 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| Tier 3.1 删 backtest_pkg/ 破坏 api/routers | **高** | 3.1b 列出所有 caller，3.1c 加 shim 兼容，3.1d 才删 |
| Tier 3.2 改 LLM 配置路径破坏 .env 配置 | 中 | 先看 .env 内容，加双路径 fallback |
| Tier 3.4 接 strategy 后端到端跑挂 | 中 | 灰度：先在 1 alpha 上跑通，再 5，再 101 |
| Tier 4.2 SignalV2 改动 20+ 文件 | **高** | ✅ 已完成 (PR6, commit `0523b1e`) — 9 文件 +450 LoC |
| 101 alpha 在某 milestone 后 pass rate 下降 > 5% | 中 | 立即回滚到上一个 milestone tag |

---

## Baseline 参考

| Tag | Commit | 含义 |
|---|---|---|
| `pre-repro-merge-2026-07-04` | (commit 3c50b93 之前) | vendor 迁移前 |
| `pre-m1-bugfixes` | `081bb21` | vendor + 测试补全，未做迁移清理 |
| `post-A+B+C+E` | `e301d45` | 上轮 codegen 清理完成 |
| `post-m1-bugfixes` | (TBD) | M1 完成 |
| `post-m2-deadcode` | (TBD) | M2 完成 |
| `post-m3-integration` | (TBD) | M3 完成 |
| `post-m4-architecture` | (TBD) | M4 完成 |

---

## Session 进度

### Session 1 (2026-07-06) — 完成 ✅

- [x] Git 前置（备份 branch `backup/pre-refactor-session1-20260706` + tag `pre-m1-bugfixes`）
- [x] 提交上轮 A+B+C+E 改动 → `e301d45`
- [x] 写本文档 → `d0fe809`
- [x] Baseline 测试 → `3062 passed / 16 failed / 14 errors` (commit `ad48f49`)
- [x] Baseline 1 alpha smoke → codegen 成功 (13s, real LLM), backtest 失败 (pre-existing)
- [x] **M1 完成** → commit `426553b`, tag `post-m1-bugfixes`
  - M1.1 track_b_checkpoint.json 缺产 fix
  - M1.2 YamlDuckdbSink factor_wide fix
  - M1.3 preload_market_data 重复 fix
  - M1.6 print → logger
  - M0 (preflight) scripts/research/run_101_alphas_v2.py 缺 import fix
  - 跳过 M1.4 (wiki.py 自循环是误判), M1.5 (空目录已清理)
- [x] **M2 完成** → commit `ae49a84`, tag `post-m2-deadcode`
  - M2.1 删 _legacy_3c/ + shim (~1500 LoC)
  - M2.2 删 pipeline stubs (~700 LoC)
  - M2.3 迁 alpha101_design + alpha158_design 到 test_fixtures/
  - M2.4 删 _make_factor_dir_name
  - 跳过 M2.5 (async 函数是 sync 包装的实现), M2.6 (compile_to_code_react 延后)

### Session 2 (2026-07-06) — 完成 ✅

- [x] Git 前置（备份 branch `backup/pre-refactor-session2-20260706`）
- [x] **Phase B: H5 key bug fix** → commit `91fd466`, tag `post-h5-fix`
  - 修 `backtest/quantnodes.py:60-62` `_default_resolver`：对 `signal.name` 应用与 writer 相同的正则净化
  - 根因：writer 用 `alpha_005` 作 H5 key，reader 用 `005` → KeyError
  - 验证：5/5 alpha success（此前 0/5），IC=0.0152, ICIR=0.1012, WinRate=52.5%
- [x] **M3.2: LLM 配置统一** → commit `ef82d7f`, tag `post-m3.2-llm-config`
  - `client.py`: CONFIG_PATHS 优先级 (Tier 1 新 → Tier 2 legacy) + QUANTNODES__LLM__* env override (Tier 3)
  - `scripts/migrate_llm_config.py`: 迁移脚本 (--dry-run, --force)
  - `tests/research/test_llm_config_paths.py`: 20 个新测试
  - 验证：配置从 `~/.quantnodes/llm.json` 加载成功，1/1 alpha success
- [x] M3 前置（删 6-field WikiFactor）→ **推到 Tier 4**
  - 理由：上游 llmwikify 已有"删 WikiFactor 导致事故"的先例；6-field 字段名作为 wiki frontmatter 约定仍活跃；与 wiki.md 拆分 + contracts.py Pydantic 协调一起做更安全

### Session 3 (2026-07-06) — 完成 ✅

- [x] **M3 主: backtest/ vs backtest_pkg/ 物理合并** → commit `9a41c4f`, tag `post-m3-main-backtest-merge`
  - `git mv` 8 模块从 `backtest_pkg/` → `backtest/` (git 保留 history)
  - `backtest/__init__.py` 加 legacy re-exports (3→36 symbols)
  - `backtest_pkg/<submodule>.py` 改 PEP 562 `__getattr__` shim（含 `_` 开头的 private symbols）
  - `backtest_pkg/__init__.py` 用 `importlib.import_module` 规避 re-export shadowing
  - 3 个生产 caller 改新路径
  - 测试：3 个 manifest test (test_imports, test_module_inventory, test_e2e_smoke) + 3 个 patch path test (l5_orchestrator / l5_reflection / l4_hypothesis_sync)
  - 验证：3065 → 3036 passed 首跑（manifest 错位），修后 3036 passed / 17F / 14E（与 M3.2 baseline 完全一致）
- [x] **M3.3: persist/strategy_library.py 新建** → commit `e657607`, tag `post-m3.3-strategy-library`
  - 4-layer strategy YAML 持久化层（mirror factor_library.py）
  - 公开 API 9 个：list_strategies / read_strategy_yaml / write_strategy_yaml / list_strategies_by_signal_type / update_index / get_strategy_node_from_yaml / save_backtest_duckdb / read_backtest_duckdb / strategy_dir
  - DuckDB schema 简化（无 `factor_values` 表，因 strategies 不存 per-stock 因子值）
  - 关键技术细节：
    - pandas 把 DuckDB NULL DOUBLE 读成 NaN — read 加 `_clean()` helper 标准化为 None
    - `exec()` 自定义 StrategyNode 子类时，`dir(module)` 会扫到 base `StrategyNode` — 加 `if attr is StrategyNode: continue` + `issubclass` 检查
  - 测试：tests/research/test_strategy_library.py (29 tests, 7 类)
  - 验证：3065 passed (+29 新测试) / 17F / 14E，alpha smoke 1/1 success

### Session 3+ (2026-07-06) — 完成 ✅ (Session 3 全部关闭)

- [x] **M3.4: strategy-mode flag 接入 run_101_alphas_v2** → commit `788a16b`, tag `post-m3.4-strategy-sink`
  - 新增 `--strategy-mode {off,per_alpha,after_batch}` flag (default `off`)
  - alpha→signal_type 启发式 (rsi/volatility/momentum/ma_cross → fallback `factor_rank`)
  - 5-alpha smoke × 2 mode：5/5 success in 83.3s (per_alpha) + 89.6s (after_batch)
- [x] **M3 后置: 删除 `backtest_pkg/` shim** → commit `30beb7b`, tag `post-m3-postaction-shim-removed`
  - 269 LoC shim 净删除（9 文件 PEP 562 forwarder 全删）
  - 16 test 文件 ~70 行 `backtest_pkg.X` → `backtest.X` sed-replace
  - 4 docstring 注释清理 (research/__init__, backtest/__init__, test_imports, test_e2e_smoke, test_module_inventory, test_equity)
  - **Bugfix**：test_backtest.py 用了 shim 制造的 `b` 是模块而非函数的语义，改用 `importlib.import_module()` 显式获取模块（5 tests in TestRunBacktestSignature/Structure）
  - 验证：3096 passed / 17F / 14E (与 M3.4 baseline 完全一致) + 1-alpha + 5-alpha smoke 双通

### Session 4 (2026-07-07) — SignalV2 PR6 完成 ✅

- [x] **M4.1 (PR6): SignalV2 — TradeSignal 重命名 + cross-layer bridge** → commit `0523b1e`, tag `post-signal-v2-trade-signal-rename`
  - **3 个 canonical Signal-like 类型** (Layer-appropriate single canonical dataclass):
    - `Signal` (paper, 不变) — `signal_source/base.py`
    - `TradeSignal` (trade, 新名) — `backtest/strategy_node.py`（旧 `Signal` 现在是 alias）
    - `SignalType` (enum, 不变) — `paper_understanding/contracts.py`
  - **Cross-layer bridge** — `research/signal_source/bridge.py` (130 行):
    - `classify_paper_signal(signal)` → `SignalType`
    - `classify_name(name)` → `SignalType`
    - `signal_type_to_strategy_class(sig_type)` → `type[StrategyNode] | None`
  - **修改**（9 文件，+450 LoC）：
    - 4 production 文件 rename: backtest/__init__ + config_strategy + strategy_node + research/backtest/strategies
    - 1 脚本换用 bridge: run_101_alphas_v2.py (_alpha_to_signal_type delegate)
    - 2 测试文件: test_strategy_node + test_strategy_library YAML
    - 1 新测试文件 test_signal_bridge.py (46 tests)
  - **验证**：3142P (+46) / 17F / 14E (baseline 一致) + 1-alpha 1/1 (15.2s) + 5-alpha per_alpha 5/5 (78.2s)

### Session 4+ (2026-07-07) — M3 前置 WikiFactor V2 完成 ✅

- [x] **M3 前置 (PR6.5): WikiFactor V2 — 字段合并 + schemas 清理** → commit `22302ef`, tag `post-wikifactor-v2-merge`
  - WikiFactor 扩展到 23 字段 (21 原有 + `factor_params` + `status`)
  - `schemas.WikiFactor` (6 字段) + `schemas.WikiStrategy` (8 字段) 全部删除
  - `schemas.py` 仅保留 `BacktestResult` + `FactorBacktestResult` (生产用)
  - 4 production caller 主动填充新字段（避免默认值的语义混淆）
  - 11 文件, +223 / -95 = +128 LoC
  - **验证**：3146P (+4) / 17F / 14E (baseline 一致) + 1-alpha 1/1 (20.9s)

### Session 5+ (待办)

- [x] M4.3: wiki.py 拆分 (1218 LoC → `wiki/{enums,factor,logic,strategy,reproduction,errors,init_factor_wiki,proxy}.py`) → `7f1bc04`, tag `post-m4.3-wiki-split`
- [ ] M4.2: 配置统一 (`~/.llmwikify/*` → `~/.quantnodes/*`)
- [ ] M4.4: Sink 异步化 (3 sink 改 async + 流式写)

---

## Session 1-2 结果汇总

### 提交

| Commit | 内容 | 文件数 | LoC 变化 |
|---|---|---|---|
| `e301d45` (上轮) | A+B+C+E codegen 迁移清理 | 13 | +144/-85 |
| `d0fe809` | REFACTOR_PLAN.md 文档 | 1 | +336 |
| `ad48f49` | Baseline 记录 | 1 | +19/-2 |
| `426553b` | **M1** Tier-1 快速修复 | 8 | +81/-28 |
| `ae49a84` | **M2** Tier-2 死代码清理 | 30 | +99/-2562 |
| `91fd466` | **Phase B** H5 key bug fix | 1 | +30/-9 |
| `ef82d7f` | **M3.2** LLM 配置统一 | 4 | +479/-51 |
| `9a41c4f` | **M3 主** backtest_pkg/ → backtest/ 物理合并 | 14 | +3299/-41 |
| `e657607` | **M3.3** strategy_library.py 新建 | 4 | +975/-0 |
| `5f43ce4` | Session 3 docs 总结 | 2 | +260/-0 |
| `788a16b` | **M3.4** strategy-mode 接入 | 3 | +228/-3 |
| `30beb7b` | **M3 后置** 删 backtest_pkg/ shim | 32 | +85/-357 |
| `0523b1e` | **M4.1 PR6** SignalV2 (TradeSignal + bridge) | 9 | +450/-47 |
| `22302ef` | **M3 前置 PR6.5** WikiFactor V2 (字段合并) | 11 | +223/-95 |

### 累计 Session 1-3+ LoC 净变化

| 阶段 | 新增 | 删除 | 净 |
|---|---|---|---|
| Session 1 (M1+M2) | +1188 | -2737 | **-1549** |
| Session 2 (Phase B+M3.2) | +509 | -60 | **+449** |
| Session 3 (M3 主+M3.3) | +5827 | -4464 | **+1363** |
| Session 3+ (M3.4+M3 后置) | +573 | -360 | **+213** |
| Session 4 (M4.1 PR6 SignalV2) | +450 | -47 | **+403** |
| Session 4+ (M3 前置 PR6.5 WikiFactor V2) | +223 | -95 | **+128** |
| **总计 (Session 1-4+)** | **+8770** | **-7763** | **+1007** |

注：Session 2/3/3+/4 新增主要为：tests (60+ new + 46 SignalV2 new) + 新文件 (scripts/migrate_llm_config.py, persist/strategy_library.py, test_strategy_sink.py, signal_source/bridge.py) + legacy re-exports (36 symbols) + manifest test updates。

### Tags

- `pre-m1-bugfixes` → `081bb21` (vendor + 测试补全，未做迁移清理)
- `post-A+B+C+E` → `e301d45` (上轮 codegen 清理完成)
- `post-m1-bugfixes` → `426553b` (M1 完成)
- `post-m2-deadcode` → `ae49a84` (M2 完成)
- `post-h5-fix` → `91fd466` (Phase B H5 fix)
- `post-m3.2-llm-config` → `ef82d7f` (M3.2 LLM config)
- `post-m3-main-backtest-merge` → `9a41c4f` (M3 主: backtest_pkg → backtest 物理合并)
- `post-m3.3-strategy-library` → `e657607` (M3.3: strategy_library.py 新建)
- `post-session3-wrapup` → `5f43ce4` (Session 3 docs 总结)
- `post-m3.4-strategy-sink` → `788a16b` (M3.4: strategy-mode 接入 run_101)
- `post-m3-postaction-shim-removed` → `30beb7b` (M3 后置: 删 backtest_pkg/ shim)
- `post-signal-v2-trade-signal-rename` → `0523b1e` (M4.1 PR6: SignalV2 TradeSignal + bridge)
- `post-wikifactor-v2-merge` → `22302ef` (M3 前置 PR6.5: WikiFactor V2 字段合并)

### Backup branches

- `backup/pre-refactor-session1-20260706` → `081bb21`
- `backup/pre-refactor-session2-20260706` → `91fd466` (H5 fix 后)

---

## 测试基线 (2026-07-06)

参考 `/tmp/baseline_tests.log` / `/tmp/m1_tests.log` / `/tmp/m2_tests.log` / `/tmp/m33_tests.log`：

### Baseline (commit `d0fe809`)

**排除的 4 个 pre-existing broken 测试文件**（vendor 迁移遗留，不在本次重构范围）：

| 文件 | 原因 |
|---|---|
| `tests/research/test_endpoint.py` | `from llmwikify.interfaces.server.http.reproduction import ...` — 模块在已装的 llmwikify 0.38.0 中不存在 |
| `tests/research/test_routes.py` | 同上 |
| `tests/research/test_paper_api.py` | 同上 (通过 `paper_client` fixture) |
| `tests/research/test_reproduction_api.py` | 同上 (通过 `repro_client` fixture in conftest.py) |

**Baseline 结果**:
- `tests/research/` + `tests/agent/` (排除 4 broken file)
- **3062 passed**
- **16 failed** (pre-existing：pandas 3.0 `'M'`/`'ME'` 等)
- **14 errors** (pre-existing：test_quant.py 缺 `factor_client` fixture)
- **67 skipped**

### Milestone 累计测试结果

| Milestone | passed | failed | errors | 关键说明 |
|---|---|---|---|---|
| Baseline (`d0fe809`) | 3062 | 16 | 14 | vendor 迁移 + 测试补全 |
| M1 (`426553b`) | ≥ 3062 | 16 | 14 | 0 回归 |
| M2 (`ae49a84`) | ≥ 3062 | 16 | 14 | 0 回归 |
| Phase B (`91fd466`) | 3014 | 17 | 14 | +1 fail（其他贡献者 WIP ValidationTool） |
| M3.2 (`ef82d7f`) | 3036 | 17 | 14 | +20 new tests（M3.2 LLM config 路径） |
| M3 主 (`9a41c4f`) | 3036 | 17 | 14 | manifest 修后 = baseline +0 |
| M3.3 (`e657607`) | 3065 | 17 | 14 | +29 new tests（strategy_library） |
| M3.4 (`788a16b`) | 3096 | 17 | 14 | +31 new tests（strategy sink） |
| M3 后置 (`30beb7b`) | 3096 | 17 | 14 | 完全 baseline 一致 + bugfix test_backtest.py |
| M4.1 PR6 (`0523b1e`) | 3142 | 17 | 14 | +46 new tests（signal_bridge.py） |
| M3 前置 PR6.5 (`22302ef`) | 3146 | 17 | 14 | +7 new V2 tests - 5 removed schemas tests |

**Failure 来源分布（17 failed, 全部 pre-existing）**：

| 数量 | 来源 | 类别 |
|---|---|---|
| 7 | `tests/research/test_factor_backtest_cross_section.py` | pandas 3.0 `'M'` → `'ME'` 升级 |
| 5 | `tests/research/test_quant.py::TestQuantInitCommand` | vendor 迁移遗留 |
| 2 | `tests/research/test_no_uncovered_smoke.py` | subprocess pytest 版本 |
| 1 | `tests/agent/test_factor_tool_extended.py::test_concurrency_safe` | 其他贡献者 WIP |
| 2 | `tests/agent/test_tool_consistency.py::TestToolMetadata::test_all_tools_have_{concurrency_safe,read_only}_property` | 其他贡献者 WIP (`ValidationTool`) |

**14 errors**：全部 `tests/research/test_quant.py` 缺 `factor_client` fixture（vendor 迁移遗留，pre-existing）。

### 101 alpha smoke 基线

| 阶段 | 结果 |
|---|---|
| Baseline (`d0fe809`) | codegen 成功 (13s) + backtest 失败（pre-existing） |
| M1 | 1 alpha 跑通 |
| M2 | 5 alpha ≥ 1 success |
| **Phase B 关键修复** | 5/5 alpha success（之前 0/5），alpha-005 IC=0.0152, ICIR=0.1012, WinRate=52.5% |
| M3.2 | 1/1 alpha success, config 加载自 `~/.quantnodes/llm.json` |
| M3 主 | 1/1 alpha success, 17.6s |
| M3.3 | 1/1 alpha success, 16.6s |

### Architecture 当前状态

```
                 ┌─────────────────────────────────────────────┐
                 │          Paper Reproduction Layer           │
                 │  scripts/run_101_alphas_v2.py (FactorStage) │
                 └─────────────────────┬───────────────────────┘
                                       │
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │            Factor Layer (sinks)             │
                 │  YamlDuckdbSink (factors/{name}/factor.yaml)│
                 │  SingleJsonSink (single_factor_NNN.json)    │
                 │  BatchSummarySink (multi_alpha_*.json)      │
                 └─────────────────────┬───────────────────────┘
                                       │  ↑ M3.4 将插入 strategy sink
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │       Strategy Layer (NEARLY DISCONNECTED)   │
                 │  persist/strategy_library.py ← M3.3 已建 ✅  │
                 │  backtest/strategies.py  SIGNAL_NODE_REGISTRY│
                 │  run_101_alphas_v2.py ← M3.4 待接入 ⚠       │
                 └─────────────────────────────────────────────┘
```

**M3 闭环路径**：M3.3 (库) ✅ → M3.4 (CLI 接入) ⏳ → M3 后置 (删 shim) ⏳