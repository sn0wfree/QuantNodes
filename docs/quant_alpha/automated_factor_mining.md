# Automated Factor Mining (v3.0.2)

## 概述

v3.0.2 引入了 `mine-logics` CLI 命令和配套的 Python API，提供自动化因子挖掘的完整闭环：

- **并发批处理** (`mine_logic_library_v2`): ThreadPoolExecutor 并发 + 幂等跳过
- **因子池** (`FactorPool`): in-mem + Wiki 双向同步 + JSON 离线持久化
- **报告生成** (`MetricsReportBuilder`): JSON + Markdown 离线报告
- **CLI** (`quantnodes mine-logics`): 7 个参数 + 双模式 + 3 退出码

## 架构

```
quantnodes mine-logics --source-libs=alpha101,alpha158,alpha191 --live
                        ↓
    batch.mine_logic_library_v2 (ThreadPoolExecutor)
        ├─ WikiFactorProxy.list_logics() → 幂等跳过
        ├─ LogicMiningPipeline.run(formula, lib) × N 线程
        ├─ ThreadSafeMetrics (Lock 保护)
        └─ FactorPool.add()
            ↓
    FactorPool.to_wiki(WikiFactorProxy)
        ↓
    MetricsReportBuilder → JSON + Markdown
```

## 快速开始

### 离线模式 (默认，无 LLM 调用)

```bash
quantnodes mine-logics --max-per-lib 5 --workers 2 --wiki-path wiki_auto
```

### 真实 LLM 模式

```bash
quantnodes mine-logics --live --max-per-lib 10 --workers 4 --strict
```

### 仅 alpha101 + alpha191

```bash
quantnodes mine-logics --source-libs alpha101,alpha191 --max-per-lib 10
```

## Python API

```python
from QuantNodes.research.quant_alpha.logic_mining.batch import (
    mine_logic_library_v2, ThreadSafeMetrics,
)
from QuantNodes.research.quant_alpha.logic_mining.report import MetricsReportBuilder
from QuantNodes.research.quant_alpha.factor_pool import FactorPool

# 1. 运行批量挖掘
batch = mine_logic_library_v2(
    source_libs=["alpha101", "alpha191"],
    max_per_lib=10,
    workers=4,
    wiki_path="wiki_auto",
    skip_existing=True,
)

# 2. 生成报告
report = MetricsReportBuilder.from_batch(batch)
report.to_json("data/mine_runs/metrics.json")
report.to_markdown()  # → str

# 3. 池操作
pool = batch.pool
pool.select(top_n=5, by="ir")
pool.filter(source_lib="alpha101", min_ir=0.5)
pool.save_json("data/mine_runs/pool.json")
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source-libs` | `alpha101,alpha158,alpha191` | 逗号分隔的来源库 |
| `--max-per-lib` | `10` | 每个库最多挖掘多少条 |
| `--workers` | `4` | 并发线程数 |
| `--wiki-path` | `wiki_auto` | Wiki 根目录 |
| `--output-dir` | `data/mine_runs` | 报告输出目录 |
| `--live` | `False` | 真实 LLM 模式 |
| `--strict` | `False` | 严格模式 (异常上抛) |
| `--no-skip` | `False` | 不跳过已存在的 Logic pages |
| `--quiet` | `False` | 安静模式 |

## 退出码

| Code | 含义 |
|---|---|
| 0 | 全部成功 (0 失败) |
| 1 | 部分成功 (有失败) |
| 2 | 致命 (参数错误 / 空结果) |

## 数据源

| Source Lib | 公式数 | 说明 |
|---|---|---|
| alpha101 | 15 | WorldQuant 101 (5 个通过 volume_price 过滤) |
| alpha158 | 8 | Qlib Alpha158 (7 个模板) |
| alpha191 | 18 | Alpha191 OHLCV-only 子集 |

注意: `_is_volume_price` 会排除含 `"pe"` 子串的公式 (误过滤含 "open" 的公式)。

## 幂等性

默认 `skip_existing=True`:
1. 启动时 `WikiFactorProxy.list_logics()` 读取现有 Logic pages
2. 对每个 (source_lib, formula_id) 检查 `pool.contains(fid)`
3. 已存在则跳过，不调用 LLM
4. 二次运行 → `n_skipped > 0`

## 并发模型

- `ThreadPoolExecutor(max_workers=workers)`
- `ThreadSafeMetrics`: 所有 `record_*` 方法加 `threading.Lock`
- `FactorPool.add()`: 内部 `_lock` 保护
- Wiki 写入: 完成后批量 `pool.to_wiki()` (非并发)

## 报告

每次运行产出两个文件:
- `data/mine_runs/metrics_{ts}.json`: 结构化 JSON
- `data/mine_runs/metrics_{ts}.md`: Markdown 表格

报告包含:
- Summary (attempted/mined/skipped/failed/success_rate)
- Source Library Breakdown (per-lib mined/attempted)
- Agent Statistics (per-agent call/parse/structured failures)
- Failed Formulas (formula_id + error)
- Warnings (alpha158 template-only, etc.)
