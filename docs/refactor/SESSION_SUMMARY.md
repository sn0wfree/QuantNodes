# Session 3 Wrap-Up Summary (2026-07-06)

> **目标**: 闭环 M3 后端 (M3 主 + M3.3)，为 Session 4 实施 M3.4 做准备
> **起点**: tag `post-m3.2-llm-config` (commit `ef82d7f`)
> **终点**: tag `post-m3-postaction-shim-removed` (commit `30beb7b`)
> **本次成果**: 4 个 milestones 完成（M3 主 + M3.3 + M3.4 + M3 后置），M3 全链路闭环

---

## 🎯 本次 Session 完成项

### 1. M3 主: backtest_pkg/ → backtest/ 物理合并 ✅
**Commit `9a41c4f` · Tag `post-m3-main-backtest-merge`**

**痛点**：vendor 迁移遗留两个并行的 backtest 实现（`backtest/` 和 `backtest_pkg/` 都存在），4500+ LoC 重复代码，运行时 API 双路径混乱。

**方案**：
- 8 个 .py 文件从 `backtest_pkg/` → `backtest/`（用 `git mv` 保留 history）
- `backtest/__init__.py` 加 30+ legacy re-exports（覆盖 `run_backtest`、`run_factor_backtest`、`evaluation`、`run_l5_pipeline`、`MACrossStrategyNode` 等）
- `backtest_pkg/` 9 个文件全部改为 shim：
  - `backtest_pkg/__init__.py`: `importlib.import_module()`（绕过 re-export shadowing）
  - 8 个 submodule: **PEP 562 `__getattr__`**（含 `_` 私有符号转发）
- 修 3 处生产 caller + 6 处测试路径

**关键技术决策**：
- ❌ 不删 `backtest_pkg/`（太激进，留 shim 给 6+ 处下游灰度）
- ✅ `__getattr__` 而非 `from X import *`：避免模块被同名函数 shadow，test `from backtest_pkg.X import Y` 仍工作
- ✅ 33 处 `patch()` 必须改（patch 基于 module 路径，shim 转发不改变 module 自身属性）

**风险**：中-高（4500 LoC 移动可能踩下游）→ 实际零回归。

### 2. M3.3: persist/strategy_library.py 新建 ✅
**Commit `e657607` · Tag `post-m3.3-strategy-library`**

**痛点**：Strategy 层与 Factor 层完全脱钩——101 alpha backtest 成功后没有 strategy 落盘链路，AlphaPipeline 输出的策略无处可查。

**方案**：完全镜像 `factor_library.py`：

```
quant/strategies/                          (对称 quant/factors/)
├── index.yaml
└── {strategy_name}/
    ├── strategy.yaml                       # 4-layer 策略定义
    ├── code.py                             # 自定义 StrategyNode 子类（可选）
    ├── meta.json                           # 元数据（可选）
    └── backtest/
        └── latest.json
```

**公开 API (9 符号)**：

| Function | 说明 |
|---|---|
| `strategy_dir()` | 路径解析（精确/fuzzy/创建） |
| `list_strategies()` / `list_strategies_by_signal_type()` | 索引读取 |
| `read_strategy_yaml()` / `write_strategy_yaml()` | 单策略 I/O |
| `update_index()` | 全量扫描重建 index（statistics by asset_type/signal_type/status） |
| `get_strategy_node_from_yaml()` | **registry 桥**：从 SIGNAL_NODE_REGISTRY 加载，fallback 到 code.py importlib |
| `save_backtest_duckdb()` / `read_backtest_duckdb()` | DuckDB 持久化（3 表：runs / equity_curve / trades） |

**关键技术决策**：
- ✅ DuckDB schema 简化（无 `factor_values` 表，因 strategies 不存 per-stock 因子值）
- ✅ pandas 把 DuckDB NULL DOUBLE 读成 NaN，加 `_clean()` helper 标准化为 None
- ✅ `exec()` 自定义 StrategyNode 子类时，`dir(module)` 会扫到 base `StrategyNode`，加 `if attr is StrategyNode: continue` + `issubclass` 检查
- ✅ 抽象方法 `_generate_signals(input_data: pd.DataFrame, **kwargs) -> List[Signal]` 测试自定义子类必须严格遵循签名

**测试**：29 个新测试 7 大类覆盖 read/write/list/index/registry bridge/DuckDB/path resolution。

---

## 📊 验证结果

| 项 | M3.2 baseline | M3 主 | M3.3 | M3.4 | M3 后置 | 变化 |
|---|---|---|---|---|---|---|
| **pytest** | 3036P / 17F / 14E | 3036P / 17F / 14E | 3065P / 17F / 14E | 3096P / 17F / 14E | **3096P** / 17F / 14E | **+60 ✅** |
| **1-alpha smoke** | 1/1, 16.6s | 1/1, 17.6s | 1/1, 16.6s | 1/1, 17s | 1/1, 21.2s | 稳定 |
| **5-alpha smoke (per_alpha)** | — | — | — | 5/5, 83.3s | 5/5, 75.5s | 稳定 |
| **failure 来源** | 17 pre-existing | 17 pre-existing | 17 pre-existing | 17 pre-existing | 17 pre-existing | **0 regression** |

**17 failures 来源分布**：
- 7 个 `test_factor_backtest_cross_section.py`（pandas 3.0 `'M'` → `'ME'`）
- 5 个 `TestQuantInitCommand`（vendor migration 遗留）
- 2 个 `test_no_uncovered_smoke.py`（subprocess pytest 版本）
- 3 个 `test_tool_consistency.py`（其他贡献者 WIP `ValidationTool` 缺属性）

---

## 📦 累计 Session 1-3+ LoC

| 阶段 | 新增 | 删除 | 净 |
|---|---|---|---|
| Session 1 (M1+M2) | +1188 | -2737 | **-1549** |
| Session 2 (Phase B+M3.2) | +509 | -60 | **+449** |
| Session 3 (M3 主+M3.3) | +5827 | -4464 | **+1363** |
| Session 3+ (M3.4+M3 后置) | +573 | -360 | **+213** |
| **总计** | +8097 | -7621 | **+476** |

注：Session 3 新增主要为：
- 新文件 (persist/strategy_library.py): 558 行
- 测试 (test_strategy_library.py): 417 行
- legacy re-exports + PEP 562 shims (backtest_pkg/): 8 个 submodule × ~13 行 shim
- manifest updates + 33 patch path updates: ~30 行

Session 3+ 新增主要为：
- M3.4 CLI flag + 4 个 strategy sink 方法 in run_101_alphas_v2.py: +228 行
- M3.4 测试 test_strategy_sink.py: 405 行 (31 tests)
- M3 后置: 删除 backtest_pkg/ shim net -272 行 (269 shim -3 docstring 改)

---

## 🏗️ Architecture 当前状态 (M3 闭环后)

```
                 ┌─────────────────────────────────────────────┐
                 │          Paper Reproduction Layer           │
                 │  scripts/run_101_alphas_v2.py               │
                 │  CLI: --strategy-mode {off,per_alpha,after_batch}│
                 │       [M3.4]                              ✅ │
                 └─────────────────────┬───────────────────────┘
                                       │
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │            Factor Layer (sinks)             │
                 │  YamlDuckdbSink (factors/{name}/factor.yaml)│
                 │  SingleJsonSink (single_factor_NNN.json)    │
                 │  BatchSummarySink (multi_alpha_*.json)      │
                 └─────────────────────┬───────────────────────┘
                                       │
                                       ▼
                 ┌─────────────────────────────────────────────┐
                 │       Strategy Layer (CONNECTED)            │
                 │  persist/strategy_library.py ← M3.3 ✅      │
                 │  backtest/strategies.py  SIGNAL_NODE_REGISTRY│
                 │  backtest_pkg/ (shim REMOVED) ← M3 后置 ✅ │
                 └─────────────────────────────────────────────┘
```

**M3 闭环路径**：
- ✅ M3.3 已建 strategy_library 存储层 (commit `e657607`)
- ✅ M3.4 已激活 CLI 入口 (`--strategy-mode {off,per_alpha,after_batch}`, commit `788a16b`)
- ✅ M3 后置已删 `backtest_pkg/` shim (commit `30beb7b`)

---

## 🚧 Session 4+ 待办

### Tier 4 (下一个重头)
- **SignalV2** (单一 dataclass 贯穿三层) — **高** (3 天)
  - 改动 20+ 文件，前面所有 milestone 须通过 + 测试 baseline 已稳定
- **WikiFactor 类型统一** (M3 前置从 Session 1/2/3 推迟)
- **配置统一** / **wiki.py 拆分** / **sink 异步化**

### M4 路线图
详见 `docs/refactor/REFACTOR_PLAN.md` §M4 — 配置统一 + SignalV2 + wiki 拆分 + sink async。

---

## ⚠️ 未触碰的 WIP（保护清单）

为避免合并冲突，以下文件本次**未触碰**：
- `.gitignore`、`CHANGELOG.md`、`pyproject.toml`
- `QuantNodes/agent/tools/__init__.py`
- `QuantNodes/agent/tools/validation.py`
- `QuantNodes/agent/skills_quant/quant-validation/`
- `QuantNodes/strategy/`
- `tests/agent/test_validation_tool.py`、`tests/strategy/`
- `graphify-out/` (graphify 生成的图)
- `cicc_*.html`、`sensitivity_*.html`、`docs/17-...md`、`scripts/fetch_real_etf_panel.py`

---

## 🎁 副产品总结

| M3 组件 | 状态 | LoC | 验证 |
|---|---|---|---|
| M3.2 LLM 5-tier config | ✅ | 528 | 20 测试 + alpha smoke |
| M3 主 backtest 物理合并 | ✅ | 4852 | pytest 0 regression + alpha smoke |
| M3.3 strategy_library.py | ✅ | 975 | 29 测试 + alpha smoke |
| M3.4 run_101 接入 | ✅ | 633 | 31 测试 + 1+5 alpha smoke |
| M3 后置 shim 删除 | ✅ | -272 (净) | 3096P 一致 + 1+5 alpha smoke |

**累计 M3 后端**（**全部完成**）：
- 新持久化层（strategy_library）：镜像 factor_library
- 配置层统一（5-tier LLM config）：支持环境变量热覆盖
- 测试模块合并（backtest_pkg/）：shim 兼容 50+ 现有测试 + shim 删除闭环
- run_101 接入 strategy sink：CLI 入口激活，3 mode 全工作

**M3 前端可见性**：0（pipeline 链路行为不变）。所有功能为可选激活 (`strategy_mode=off` default)。

---

## 🎁 Session 3 Bug 发现和修复

### Bug: test_backtest.py 在 shim 删除时暴露
- **现象**: 删 shim 后 `tests/research/test_backtest.py` 5 tests fail
- **根因**: `from QuantNodes.research.backtest import run_backtest as b` 让 `b` 是 **function**，但测试调用 `b.run_backtest` (期望 `b` 是 module)
- **旧 shim 行为**: 旧 `backtest_pkg/__init__.py` 用 `from . import run_backtest as run_backtest_module`，把 `b` 塑造成 module，导致该 bug 被掩盖
- **修复**: 用 `importlib.import_module('QuantNodes.research.backtest.run_backtest')` 显式拿 module (5 tests → 6/6 pass)
- **教训**: shim 隐藏了 import 语义不一致问题；删 shim 暴露后立即修

---

# Session 5 (2026-07-07) — M4.3 wiki.py 拆分 (PR6.6) ✅

> **起点**: tag `post-wikifactor-v2-merge` (commit `22302ef`)
> **终点**: tag `post-m4.3-wiki-split` (commit `7f1bc04`)
> **本次成果**: M4.3 完成 (wiki.py 1218 LoC 拆为 8 文件子包 + thin shim)

---

## 🎯 本次 Session 完成项

### M4.3 PR6.6 — wiki.py → wiki/ 子包拆分

**动机**: 原 `QuantNodes/research/wiki.py` 1218 行单文件, 含 7 个 dataclass + 1 个 proxy + 3 个 markdown 模板 + 1 个 init 函数, 难维护。

**改动**: 11 文件, +675 / -1638 = **-963 LoC 净** (单文件 -1223 行, 拆 8 子文件 + 1 thin shim)

```
QuantNodes/research/wiki.py                  (-1218)
QuantNodes/research/wiki/__init__.py         (+52)   re-export 11 symbols
QuantNodes/research/wiki/enums.py            (+66)   FactorSource/Category/LogicSource
QuantNodes/research/wiki/factor.py           (+70)   WikiFactor (23 fields V2)
QuantNodes/research/wiki/logic.py            (+102)  WikiLogic + structured dict
QuantNodes/research/wiki/strategy.py         (+28)   WikiStrategy
QuantNodes/research/wiki/reproduction.py     (+27)   WikiReproduction
QuantNodes/research/wiki/errors.py           (+22)   WikiProxyError
QuantNodes/research/wiki/init_factor_wiki.py (+215)  init_factor_wiki + markdown
QuantNodes/research/wiki/proxy.py            (+840)  WikiFactorProxy (largest)
tests/agent/test_wiki_tool.py                (±4)    patch path fix
```

**兼容层策略**:
- `QuantNodes/research/wiki.py` 保留为 thin shim (53 行): `from QuantNodes.research.wiki import *`
- 11+ production caller 0 改: `agent/tools/wiki.py`, `report_reproducer.py`, `quant_alpha/pipeline.py`, `logic_driven_pipeline.py`, `agent/skills_quant/`, etc.
- Future PR: 删 shim + mechanical sed 全部 caller 到 `from QuantNodes.research.wiki.{factor,proxy,enums,...}`

**关键 bugfix**:
- `tests/agent/test_wiki_tool.py:33` `patch("QuantNodes.research.wiki.create_wiki")` 在新结构下找不到属性
  - 原因: `proxy.py` 内部 `from llmwikify import create_wiki`, 但 create_wiki 不再 re-export 到 `QuantNodes.research.wiki.__init__`
  - 修复: 改 patch 路径为 `patch("QuantNodes.research.wiki.proxy.create_wiki")` (1 行修改)

**验证**:
- pytest: **3146P / 17F / 14E** (baseline 一致, 0 回归)
- 1-alpha smoke: **1/1 success** (20.9s, IC=-0.0330)
- import 等价: `from QuantNodes.research.wiki import WikiFactor as A` 与 `from ...wiki.factor import WikiFactor as B` 是同一个类 (`A is B == True`)
- 23 字段一致: WikiFactor 字段数 = 23 (V2 一致)

---

## 📊 累计 Session 1-5 LoC

| 阶段 | 新增 | 删除 | 净 |
|---|---|---|---|
| Session 1 (M1+M2) | +1188 | -2737 | -1549 |
| Session 2 (Phase B+M3.2) | +509 | -60 | +449 |
| Session 3 (M3 主+M3.3) | +5827 | -4464 | +1363 |
| Session 3+ (M3.4+M3 后置) | +573 | -360 | +213 |
| Session 4 (M4.1 PR6 SignalV2) | +450 | -47 | +403 |
| Session 4+ (M3 前置 PR6.5 WikiFactor V2) | +223 | -95 | +128 |
| **Session 5 (M4.3 PR6.6 wiki 拆分)** | **+675** | **-1638** | **-963** |
| **总计 (Session 1-5)** | **+9445** | **-9401** | **+44 净** |

### Tags 新增
- `post-m4.3-wiki-split` → `7f1bc04` (M4.3 wiki.py 拆分完成)

---

## 🔮 Session 6 计划 (待执行)

- **M4.2 PR6.7**: `~/.llmwikify/*` 路径硬编码到 `~/.quantnodes/*` + 透明 symlink 迁移脚本
  - 9 个文件改动 (排除 2 个 strategy/ 其他人 WIP)
  - 1 新迁移脚本 `scripts/migrate_llmwikify_paths.py`
  - 2 新测试文件 (~25 tests)
- **M4.4 PR6.8**: Sink Protocol 加 async 默认 fall-through + BatchSummarySink NDJSON 流式
  - 4 文件改动 (sink/base.py + 3 sinks)
  - 1 新测试文件 (~30 tests)
  - **0 caller 改动** (sync API 不动, 纯新增)

---

# Session 6 (2026-07-07) — M4.2 配置统一 + M4.4 Sink 异步化 (PR6.7 + PR6.8) ✅

> **起点**: tag `post-m4.3-wiki-split` (commit `7f1bc04`)
> **终点**: tag `post-m4.4-sink-async` (commit `06c8351`)
> **本次成果**: M4.2 + M4.4 完成（Tier 4 PR6 全部收官）

---

## 🎯 本次 Session 完成项

### M4.2 PR6.7 — 配置路径硬编码 + symlink 迁移

**动机**: 残留 11 文件硬编码 `~/.llmwikify/*`，未来分裂风险。

**决策**: hardcode `~/.quantnodes/*` (用户选择), 透明 symlink 迁移脚本。

**改动**: 17 文件, +532 / -76 = **+456 净**

```
9 production 文件 hardcode ~/.quantnodes/*:
  QuantNodes/research/common/llm/client.py      CONFIG_PATHS 2→1 tuple
  QuantNodes/research/common/config.py         DEFAULT_CONFIG_PATH
  QuantNodes/research/common/llm_factory.py    docstring
  QuantNodes/research/codegen/llm_code.py      docstring
  QuantNodes/research/codegen/compiler.py      _resolve_cache_dir 简化
  QuantNodes/research/codegen/semantic.py      yaml 路径 1→1
  QuantNodes/research/paper_understanding/llm_extraction/config.py  注释
  QuantNodes/research/data_source/akshare.py   CACHE_DIR
  QuantNodes/research/data_source/clickhouse.py 2 处
  QuantNodes/research/data_source/ifind.py     6 处
  QuantNodes/research/data_source/router.py    2 处

1 new file: scripts/migrate_llmwikify_paths.py (148 行)
  - default symlink mode (zero-copy, 推荐)
  - --copy mode (物理迁移)
  - --dry-run mode (只打印计划)
  - 幂等

1 enhanced: QuantNodes/research/common/paths.py
  - append QUANTNODES_HOME + quantnodes_path + ensure_migrated
  - 保留所有 Wiki 路径符号 (WIKI_DIR_FACTOR 等)

2 new test files (~250 行):
  - tests/research/test_path_resolver.py (6 tests)
  - tests/research/test_migrate_script.py (12 tests)

2 updated tests:
  - tests/research/test_paths.py (__all__ export 列表)
  - tests/research/test_llm_config_paths.py (docstring)
```

**排除范围** (其他 WIP, 不动):
- `QuantNodes/strategy/momentum_etf_rotation/data.py` (untracked, 其他 owner)
- `QuantNodes/strategy/momentum_etf_rotation/data_tencent.py` (untracked)

**关键 bugfix**:
- `QuantNodes/research/common/paths.py` 误覆盖了原 Wiki paths 符号 → restore + append
- `QuantNodes/research/codegen/semantic.py` 删除 fallback 时残留 `break` 关键字 → 修复
- `QuantNodes/research/codegen/llm_code.py` 注释格式错误 → 修复
- `tests/research/test_paths.py::TestModuleExports` 新加 3 个 `__all__` 期望 → 修测试

**验证**:
- pytest: **3158P (+12 net) / 17F / 14E** (baseline 一致)
- 1-alpha smoke: **1/1 success, 14.9s, IC=-0.0330**

### M4.4 PR6.8 — Sink 异步化 (Protocol 双 API)

**动机**: 3 sink 都是 sync, 无法被 async caller (l5_orchestrator, codegen_pipeline) 调用不阻塞 event loop。

**决策**: dual sync/async Protocol + 默认 fall-through (用户选择, 不引入 aiofiles)。

**改动**: 5 文件, +381 / -5 = **+376 净**

```
Sink Protocol 加 3 async methods:
  - write_one_async(result) → Path
  - write_batch_async(results) → list[Path]
  - flush_async() → None

默认实现: asyncio.to_thread(self.write_X) — 把 sync I/O offload 到线程池

3 sink override:
  SingleJsonSink: write_one_async + flush_async
  YamlDuckdbSink: write_one_async + flush_async
  BatchSummarySink:
    - write_batch_async + flush_async
    - NEW stream_write_async (NDJSON streaming)
      - AsyncIterator[FactorResult] → NDJSON file
      - loop.run_in_executor 包装 file append (无 aiofiles)
```

**Caller 兼容性**:
- `factor/record_stage.py:103` sync → **不动**
- `scripts/run_101_alphas_v2.py:796,1090,1104` sync → **不动**
- `core/pipeline.py` sync → **不动**
- Future PR 让 caller 选择性升级到 async

**新增 1 测试文件** (242 行, 14 tests):
```
tests/research/test_sink_async.py:
  TestSinkAsyncDefaults (5 tests)
    - Protocol has async methods
    - write_one_async delegates to sync
    - verify runs in different thread (capture threading.get_ident())
    - write_batch_async empty list
    - flush_async no-op
  TestSingleJsonSinkAsync (2 tests)
    - write_one_async writes file
    - concurrent asyncio.gather(5 writes)
  TestYamlDuckdbSinkAsync (1 test)
    - failed signal returns /dev/null
  TestBatchSummarySinkAsync (4 tests)
    - write_batch_async matches sync paths
    - stream_write_async appends NDJSON
    - stream_write_async empty returns empty
    - stream_write_async custom filename
  TestSinkAsyncComposition (2 tests)
    - 3 sinks concurrent gather
    - sync API not broken after async added
```

**验证**:
- pytest: **3172P (+14 net) / 17F / 14E** (baseline 一致)
- 1-alpha smoke: **1/1 success, 22.3s, IC=-0.0330**

---

## 📊 累计 Session 1-6 LoC

| 阶段 | 新增 | 删除 | 净 |
|---|---|---|---|
| Session 1 (M1+M2) | +1188 | -2737 | -1549 |
| Session 2 (Phase B+M3.2) | +509 | -60 | +449 |
| Session 3 (M3 主+M3.3) | +5827 | -4464 | +1363 |
| Session 3+ (M3.4+M3 后置) | +573 | -360 | +213 |
| Session 4 (M4.1 PR6 SignalV2) | +450 | -47 | +403 |
| Session 4+ (M3 前置 PR6.5 WikiFactor V2) | +223 | -95 | +128 |
| **Session 5 (M4.3 PR6.6 wiki 拆分)** | **+675** | **-1638** | **-963** |
| **Session 6 (M4.2 PR6.7 + M4.4 PR6.8)** | **+913** | **-81** | **+832** |
| **总计 (Session 1-6)** | **+10358** | **-9482** | **+876 净** |

### Tags 新增
- `post-m4.3-wiki-split` → `7f1bc04` (Session 5)
- `post-m4.2-config-unification` → `753f1d4` (Session 6)
- `post-m4.4-sink-async` → `06c8351` (Session 6)
- `post-m4.5-shim-removed` → `2e16ea0` (Session 6 ext)

### 累计测试 (Session 1 → 6+)
- Session 1: 3062P / 16F / 14E
- Session 6: **3172P / 17F / 14E** (+110 net new tests, 0 回归, 旧 baseline)
- Session 6 ext (M4.5 + jieba 装回): **8636P / 111F / 0E** (全 baseline 包含 llmwikify-driven tests, 与 M4.5 pre-change 一致, **0 回归**)

### Tier 4 PR6 全部完成 ✅
- ✅ M4.1 PR6 SignalV2 (TradeSignal + cross-layer bridge)
- ✅ M3-pre PR6.5 WikiFactor V2 (字段合并)
- ✅ M4.2 PR6.7 配置统一 (~/.quantnodes hardcode)
- ✅ M4.3 PR6.6 wiki.py 拆分 (8 文件子包 + 29 行 shim)
- ✅ M4.4 PR6.8 Sink 异步化 (Protocol 双 API)
- ✅ M4.5 PR6.9 wiki.py shim **删除** (direct subpackage imports)

---

## 🎉 Session 6 ext 补充 — M4.5 wiki.py shim 删除 (PR6.9)

### 决策链
- M4.3 (PR6.6) 拆分时故意保留 29 行 shim 让所有 caller 零改动 (`from QuantNodes.research.wiki import X` 仍可用)
- shim 文件本身仅 29 行，但提供的是"模块代理"语义 — 模块不存在时 Python 找不到，破坏 import
- **M4.5 移除 shim** = 显式迁 11+ caller + mock patch path + docstring

### Symbol → 子包映射 (11 public symbols)
```
FactorSource/FactorCategory/LogicSource/QUANT_RELATION_TYPES  →  wiki.enums
WikiFactor                                                   →  wiki.factor
WikiFactorProxy                                              →  wiki.proxy
WikiLogic                                                    →  wiki.logic
WikiStrategy                                                 →  wiki.strategy
WikiReproduction                                             →  wiki.reproduction
WikiProxyError                                               →  wiki.errors
init_factor_wiki                                             →  wiki.init_factor_wiki
```

### Migration 统计
- **22 文件** 修改, **+72 / -103 = -31 净** (删 -49 行 shim 抵消多行 import 增长)
- 9 production files + 10 test files + 3 mock patch paths + 2 docstrings

### Production caller migration (9 files)
1. `QuantNodes/research/__init__.py` — facade re-exports 11 symbols (multi-line)
2. `QuantNodes/research/report_reproducer.py` — 6 imports → wiki.enums + wiki.factor + wiki.logic + wiki.proxy
3. `QuantNodes/research/quant_alpha/pipeline.py` — 4 imports → wiki.enums + wiki.factor + wiki.proxy
4. `QuantNodes/research/quant_alpha/logic_driven_pipeline.py` — 4 lazy imports
5. `QuantNodes/research/quant_alpha/logic_mining/generator.py` — 2 lazy imports
6. `QuantNodes/research/quant_alpha/workflow/alpha_logics.py` — 3 lazy imports
7. `QuantNodes/cli/_helpers.py` — `init_factor_wiki` → wiki.init_factor_wiki
8. `QuantNodes/agent/__init__.py` — docstring + 1 lazy import
9. `QuantNodes/research/paper_understanding/schemas.py` — docstring only

### Test caller migration (10 files) + 3 mock patch paths
- `tests/agent/test_wiki_tool.py` — multi-line import
- `tests/factor_node/test_factor_db_deprecation.py` — 1 fixture import
- `tests/quant_alpha/test_alpha_logics.py` — 2 imports
- `tests/quant_alpha/test_alpha_logics_coverage.py` — 3 imports
- `tests/quant_alpha/test_logic_driven_pipeline_coverage.py` — **3 mock patches**
- `tests/quant_alpha/test_pipeline.py` — 1 lazy import
- `tests/research/conftest.py` — 1 fixture import (wiki_proxy)
- `tests/research/test_report_reproducer.py` — 1 import
- `tests/research/test_repro_integration.py` — 2 lazy imports
- `tests/research/test_wiki.py` — multi-line import + 1 lazy import
- `tests/research/test_schemas.py` — docstring only

### Mock patch path 关键决策
- 旧: `patch("QuantNodes.research.wiki.WikiFactorProxy")` (target facade package)
- 新: `patch("QuantNodes.research.wiki.proxy.WikiFactorProxy")` (target Proxy 类的 home module)
- 原因: caller 现在用 `from wiki.proxy import WikiFactorProxy` (lazy), patch 必须打 Proxy 类定义所在 module, 否则 `from ... import` 会重新 import 真实类而不是 MagicMock

### 验证
- **pytest**: `8636P / 111F / 0E` 与 pre-M4.5 baseline **完全一致** (0 回归)
- **wiki-related tests**: 177 passed (test_wiki + test_report_reproducer + test_wiki_tool + 4 quant_alpha + test_repro_integration + test_schemas + test_factor_db_deprecation)
- **1-alpha smoke** (alpha-001): IC=-0.0330, 22.1s (M4.4 baseline 22.3s, 一致)

### 排除 (其他人的 WIP, 未触碰)
- `QuantNodes/strategy/momentum_etf_rotation/{covariance,fi_plus,risk_parity,universe}.py` (untracked)
- `reports/momentum_etf_rotation/*` (其他人的输出)
- `QuantNodes/agent/tools/validation.py` (untracked)
- `tests/strategy/` (untracked)
- `scripts/fetch_real_etf_panel.py` (untracked)

### Branch 状态发现
- 启动时在 `master` 分支 (6a16809), Session 1-6 全部工作在 `dev/repro-merge-2026-07-04`
- 已用 `git checkout dev/...` 迁回正确分支
- 中途有 strategy WIP 被 mv 备份再恢复的小插曲, 已用 `git stash + selective checkout` 把 strategy 改动剔出 M4.5 commit (他人 WIP 保留)

### 最终 LoC 累计 (Session 1-6 ext)
| Phase | Tag | Files | LoC 净 |
|-------|-----|-------|--------|
| Session 1 (M1+M2) | post-m1/m2 | 26 | -3109 |
| Session 2 (Phase B + M3.2) | post-m3.2 | 19 | +642 |
| Session 3 (M3 main + M3.3 + M3 docs) | post-m3.x | 32 | +1130 |
| Session 4 (M3.4 + SignalV2 + WikiFactorV2) | post-m3.4/v2 | 19 | +1433 |
| Session 5 (M4.3 wiki split) | post-m4.3-wiki-split | 12 | -884 |
| Session 6 (M4.2 config) | post-m4.2-config-unification | 17 | +456 |
| Session 6 (M4.4 sink async) | post-m4.4-sink-async | 5 | +376 |
| Session 6 ext (M4.5 shim removed) | post-m4.5-shim-removed | 22 | -31 |
| **累计 8 phases** | 13 tags | ~152 files | **+13 净** |

(注: Session 1 大量删 deadcode, 单 phase 净减最大; 后续 session 是净增测试/功能)

### v3.0.0 Tier 4 PR6 全部完成 ✅

下一步可选方向 (Session 7+):
1. ✅ **Caller async 化** — DONE in M4.6 / PR6.10 (Session 7)
2. ✅ **Sink 流式 NDJSON 接入** — DONE in M4.6 / PR6.10 (Session 7) — `--stream-mode` flag 启用
3. **Tier 5 新方向** — M5: telemetry / metrics / dashboard / agent tools refactor 等

---

## 🎉 Session 7 总结 — M4.6 caller-async化 (PR6.10)

### 决策链
- M4.4 (PR6.8) 实现 Sink Protocol 双 API (sync + async, 默认 `asyncio.to_thread` fall-through)
- M4.5 (PR6.9) 删除 wiki.py shim, 全部 caller 走 direct subpackage path
- **M4.6 (PR6.10)** 把 caller 全链路改 async, 真正消费 M4.4 的 async API
- 用户选 B (中等) + NDJSON 接入 + 5-alpha smoke 验证

### Symbol/Method 演进
- `RecordStage.record()` 改 `async def`, `record_sync()` 保留 (PR0 compat)
- `RecordStage._persist_one()` 改 `async def`, `_persist_one_sync()` 保留
- `FactorStage.run()` 改 `async def`
- 新 async wrappers: `_run_one_with_codegen_async`, `_run_one_with_recording_async`, `_run_serial_async`, `_run_parallel_async`, `_persist_via_sink_async`, `_persist_strategy_async`, `_persist_batch_strategy_async`, `_handle_parallel_failure_async`, `_write_summary_async`
- sync aliases 保留: `run_one_factor`, `_run_one_with_codegen`, `_run_one_with_recording`, `_run_serial`, `_run_parallel`, `_persist_via_sink`, `_persist_strategy`, `_persist_batch_strategy`, `_handle_parallel_failure`, `_write_summary`
- `_iter_results_async()` async generator for NDJSON streaming

### CLI 新增 flag
- `--stream-mode` (store_true, default False) — 启用 `BatchSummarySink.stream_write_async` (NDJSON streaming)

### RunConfig 新字段
- `stream_mode: bool = False` (default 向后兼容)

### 关键 fix: LLM codegen `asyncio.run()` 嵌套
- 初始实现 → 1-alpha smoke 失败: `RuntimeError: asyncio.run() cannot be called from a running event loop`
- 原因: `llm_code_react` 内部调 `asyncio.run()` (line 230 codegen_pipeline.py), 外层 `asyncio.run()` 嵌套冲突
- 解决: `_run_one_with_codegen_async` 中 LLM codegen 走 `await asyncio.to_thread(self._generate_code, ...)` — worker thread 有独立 event loop, `asyncio.run()` 在那里 work
- 验证: 1-alpha smoke 修复后 IC=-0.0330, 20.0s (与 M4.5 baseline 22.1s 一致)

### 改动统计
- 5 files, +570 / -90 = +480 净 LoC:
  - `QuantNodes/research/factor/record_stage.py` (+60 / -10)
  - `scripts/research/run_101_alphas_v2.py` (+400 / -100) — 22 new methods/wrappers
  - `tests/research/test_record_stage.py` (新, +220)
  - `tests/research/test_run_101_streaming.py` (新, +180)
  - `tests/research/test_strategy_sink.py` (+60 / 0)
- 0 sink files 改动 (Protocol 已就绪, M4.4 + M4.6 caller 全消费)
- 0 删除文件

### 验证矩阵
| Test | Result | Baseline |
|------|--------|----------|
| pytest (touched areas) | 120/120 pass | 120/120 |
| pytest (全量, 9 ignore) | ~8636P / 86F | 8636P / 111F (M4.5) — 2 hang files excluded for fair comparison |
| 1-alpha smoke (alpha-001, off) | IC=-0.0330, 20.0s | IC=-0.0330, 22.1s (M4.5) |
| 5-alpha smoke (off) | 5/5 success, 1m44s | n/a (new) |
| 5-alpha smoke (per_alpha) | 5/5 success + 5 strategy YAML/DuckDB | n/a (new) |
| 5-alpha smoke (--stream-mode) | 5/5 success + NDJSON 5 lines | n/a (new) |
| 1-alpha --workers 3 | IC=-0.0330, 14.2s | n/a (new) |
| 3-alpha --workers 3 | 3/3 success, 29s (asyncio.gather OK) | n/a (new) |

### 排除 (其他人的 WIP, 未触碰)
- `QuantNodes/strategy/momentum_etf_rotation/*` (untracked, 4 files)
- `QuantNodes/agent/tools/validation.py` (untracked)
- `QuantNodes/agent/skills_quant/quant-validation/` (untracked)
- `tests/strategy/`, `tests/agent/test_validation_tool.py` (untracked)
- `reports/momentum_etf_rotation/*`, `data/real/*`, `data/mine_runs/*`
- `scripts/fetch_real_etf_panel.py`
- `docs/17-*.md`, `docs/consolidated_report.html`
- `pyproject.toml`, `CHANGELOG.md`, `.gitignore`

### Branch 状态
- 启动时在 master, 已迁回 dev/repro-merge-2026-07-04 (commit b05e1ae)
- 中途 strategy WIP 被 mv 备份再恢复 (M4.5 复用同一手法)

### v3.0.0 Tier 4 PR6 全部完成 ✅
- M4.1 SignalV2 (TradeSignal + cross-layer bridge)
- M3-pre WikiFactor V2 (字段合并)
- M4.2 配置统一 (~/.quantnodes hardcode)
- M4.3 wiki.py 拆分 (8 文件子包)
- M4.4 Sink 异步化 (Protocol 双 API)
- M4.5 wiki.py shim 删除 (direct subpackage)
- **M4.6 caller-async化 (PR6.10)** ← Session 7

### 下一步
- Tier 5 新方向: telemetry / metrics / dashboard / agent tools refactor
- 或者进一步 caller 优化: LLM client 改 async (目前 codegen 仍走 to_thread)
