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

| Milestone | 内容 | 工作量 | 风险 | PR |
|---|---|---|---|---|
| **M1** | Tier 1: 6 个 latent bug 快速修复 | 0.5 天 | 低 | PR1 |
| **M2** | Tier 2: 死代码清理（~3000 LoC） | 0.5 天 | 低 | PR2 |
| M3 前置 | Tier 3.5: WikiFactor 类型统一 | 0.5 天 | 低 | PR3 |
| M3 主 | Tier 3.1: backtest 双包合并 | 1 天 | **高** | PR4 |
| M3 后 | Tier 3.2 + 3.3 + 3.4: LLM 配置 + strategy_library + 端到端集成 | 1 天 | 中 | PR5 |
| M4 | Tier 4: 配置统一 + SignalV2 + wiki 拆分 + sink async | 3 天 | **高** | PR6 |

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

### M3 前置 (PR3) — WikiFactor 类型统一

- 删 `paper_understanding/schemas.py:86` 的 6 字段 `WikiFactor`
- 全部走 `wiki.py:50` 的 23 字段 `WikiFactor`
- 修所有 caller

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

**M3.4 — 接入 strategy 层**:
- `run_101_alphas_v2.py` 加 `--strategy-mode` flag
- 跑完所有 factor 后自动调 `AlphaPipeline` 生成策略
- 写入 `strategy.yaml`

---

## Milestone 4: Tier 4 架构重构 (PR6)

### M4.1 — 配置文件统一

- 所有 `~/.llmwikify/*` 路径 → `~/.quantnodes/*`
- 加迁移脚本 `~/.quantnodes/migrate_from_llmwikify.py`

### M4.2 — SignalV2 重设计

- 单一 dataclass 贯穿三层
- 改 `signal_source/`、`codegen/`、`backtest/`、`sink/`、`strategy/` 全部接口

### M4.3 — 拆 wiki.py (1155 LoC)

- `wiki/{factor,logic,strategy,proxy}_wiki.py`

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
| Tier 4.2 SignalV2 改动 20+ 文件 | **高** | 必须前面所有 milestone 通过 + 测试 baseline 已稳定 |
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

### Session 1 (2026-07-06) — 进行中

- [x] Git 前置（备份 branch + tag）
- [x] 提交上轮 A+B+C+E 改动 → `e301d45`
- [ ] 写本文档
- [ ] Baseline 测试
- [ ] M1 完成
- [ ] M2 完成

### Session 2+ (待办)

- [ ] M3 前置 + M3 主
- [ ] M3 后
- [ ] M4 全部

---

## 测试基线 (2026-07-06)

参考 `/tmp/baseline_tests.log` / `/tmp/m1_tests.log` / `/tmp/m2_tests.log`：

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

| Suite | Baseline | M1 期望 | M2 期望 |
|---|---|---|---|
| `tests/research/` + `tests/agent/` (排除 4 broken file) | 3062 passed | ≥ 3062 | ≥ 3062 |
| `tests/research/test_loop_v4_pr1_to_pr7.py` | 46 passed + 2 skipped | 同 | 同 |
| 101 alpha smoke (1/5) | (待跑 baseline) | 1 alpha 跑通 | 5 alpha ≥ 1 success |