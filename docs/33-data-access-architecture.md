# 数据访问架构 (v3.0)

> **TL;DR**: QuantNodes 有 **7 个数据访问层**，每层职责不同，**不应强行合并**。本文档解释各层用途、何时使用哪层、以及历史设计决策。

## 1. 为什么需要分层

数据访问是 QuantNodes 最复杂的领域之一。不同的数据源（SQL DB / HDF5 文件 / polars 流式 / 知识库）有不同的语义：

- **SQL DB**：声明式查询、事务、CRUD
- **HDF5/Parquet 文件**：面板矩阵语义、按键索引
- **Polars 流式**：Stage 2 Table 4 复现的延迟敏感场景
- **知识库**：因子元数据、论文笔记、跨引用

强行用单一 ABC 会泄漏抽象（pandas vs polars, SQL vs 文件路径, 流式 vs 全量）。`core/data_source.py:1-19` 的设计注释明确反对合并：

> *"QuantNodes 有两个独立的数据接入子树...两者接口差异大 (文件是面板矩阵语义, 数据库是 SQL 语义), 强行合并会泄漏抽象。"*

## 2. 7 层架构总览

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 7: WikiFactorProxy (L7)                                       │
│   → research/wiki.py                                                │
│   → 知识库: 因子元数据 / IC 指标 / 论文笔记 (markdown)             │
├────────────────────────────────────────────────────────────────────┤
│ Layer 6: contracts.DataLoader (L6)                                  │
│   → research/quant_alpha/evaluation/contracts.py                   │
│   → Table 4 复现数据: polars-first, Stage 1 mock / Stage 2 CH    │
├────────────────────────────────────────────────────────────────────┤
│ Layer 5: factor_test DataLoader (L5)                                │
│   → research/factor_test/utils/data_loader.py                      │
│   → iFinD H5 测试数据 facade                                        │
├────────────────────────────────────────────────────────────────────┤
│ Layer 4: FileFormatLoader (L4)                                      │
│   → research/factor_test/utils/file_loaders.py                     │
│   → 文件格式适配器 (H5/CSV/NPY/Parquet) — registry pattern         │
├────────────────────────────────────────────────────────────────────┤
│ Layer 3: cache_node (L3)                                            │
│   → cache_node/                                                     │
│   → 透明 Parquet 缓存装饰器 — 包装 L1 节点                          │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2: factor_db (L2)  ⚠️ DEPRECATED                              │
│   → factor_node/factor_db.py                                        │
│   → 历史占位: 大部分方法返回 0/None, 无生产使用                    │
├────────────────────────────────────────────────────────────────────┤
│ Layer 1: BaseDBNode (L1)                                            │
│   → database_node/                                                  │
│   → SQL CRUD: SQLite/DuckDB/MySQL/CH/CSV/Parquet                   │
│   → (含生产可用的 ClickHouseNode + CHBase)                         │
└────────────────────────────────────────────────────────────────────┘
```

## 3. 各层详情

### L1: `database_node/` — SQL CRUD 基础层

| 属性 | 值 |
|------|-----|
| 文件 | `QuantNodes/database_node/` (9 files, ~970 LOC) |
| ABC | `BaseDBNode(DataSource, ABC)` |
| 返回 | `pd.DataFrame` via `query(sql, params)` |
| 写 | ✅ `execute()`, `insert_df()` |
| 上下文 | `connect/disconnect`, `__enter__/__exit__` |
| 生产调用 | 2 (`agent/tools/config_backtest.py`) |
| 测试调用 | 4 |
| **何时使用** | 需要 SQL 查询 / 写操作的场景。生产首选 L1。 |

**关键实现**: `ClickHouseNode` 继承 `BaseDBNode`，使用 `CHBase` (11 种格式支持) 作为底层 HTTP 客户端。**生产可用**。

### L2: `factor_node/factor_db.py` — DEPRECATED ⚠️

| 属性 | 值 |
|------|-----|
| 文件 | `QuantNodes/factor_node/factor_db.py` (403 LOC) |
| ABC | `FactorDB` / `WritableFactorDB` |
| 返回 | `FactorTable` 对象 |
| 写 | 接口存在但无实现 |
| 生产调用 | 2 (但都是 re-export, 实际未调用业务方法) |
| **何时使用** | **不使用**。详见 [Deprecated L2](#deprecated-l2-factornodefactor_dbpy)。 |

### L3: `cache_node/` — 缓存装饰器

| 属性 | 值 |
|------|-----|
| 文件 | `QuantNodes/cache_node/` (3 files, ~460 LOC) |
| ABC | `MarketDataCacheNode(BaseNode)` |
| 返回 | `pd.DataFrame` via `_execute(dict)` |
| 写 | ✅ (写 Parquet 缓存) |
| **何时使用** | 包装一个 L1 节点, 增加透明缓存。**正交**于 L1 — 装饰器模式, 不是替代品。 |

**架构关系**: `L3 → wraps → L1`。L3 接收一个 L1 节点作为参数, 在 `_execute` 中检查缓存。

### L4: `FileFormatLoader` — 文件格式适配器

| 属性 | 值 |
|------|-----|
| 文件 | `research/factor_test/utils/file_loaders.py` (150 LOC) |
| ABC | `FileFormatLoader(DataSource, ABC)` |
| 返回 | `pd.DataFrame` via `load(path, *, key, store_getter)` |
| 写 | 无 (只读) |
| **何时使用** | iFinD 风格的 H5/CSV/NPY/Parquet 文件, 按 key 索引。 |

### L5: `factor_test/data_loader.py` — iFinD facade

| 属性 | 值 |
|------|-----|
| 文件 | `research/factor_test/utils/data_loader.py` (141 LOC) |
| ABC | `DataLoader` (concrete, **不是** ABC) |
| 返回 | `pd.DataFrame` via `load_h5/csv/npy/parquet` |
| **何时使用** | factor_test e2e 测试, 需要加载 iFinD 风格测试数据。 |

**与 L4 关系**: L5 是 facade (高层 API), L4 是 registry (可扩展适配器)。L5 内部调用 L4。

### L6: `contracts.DataLoader` — Table 4 polars-first

| 属性 | 值 |
|------|-----|
| 文件 | `research/quant_alpha/evaluation/contracts.py` (375 LOC) |
| ABC | `DataLoader(abc.ABC)` |
| 返回 | `pl.DataFrame` via `load()` (**polars**, 不是 pandas) |
| 写 | 无 |
| 实现 | `MockDataLoader`, `ClickHouseDataLoader` |
| 生产调用 | 0 (经 `runner.py` 间接使用) |
| 测试调用 | 5 |
| **何时使用** | Stage 2 Table 4 复现 (polars 性能关键路径)。 |

**与 L1 关系**: L6 故意不继承 `BaseDBNode`, 因为返回 polars 而非 pandas。**唯一交叉**: `ClickHouseDataLoader._query_clickhouse` 用了 raw `http.client`, 应该改用 L1 的 `ClickHouseNode.query()` (TODO, P2.12c.3)。

### L7: `WikiFactorProxy` — 知识库

| 属性 | 值 |
|------|-----|
| 文件 | `research/wiki.py` (1,155 LOC) |
| ABC | `WikiFactorProxy` (concrete) |
| 返回 | `WikiFactor` / `WikiLogic` dataclasses |
| 写 | ✅ (markdown 文件 + 关系索引) |
| 生产调用 | 9 |
| 测试调用 | 8 |
| **何时使用** | 因子元数据 / IC 指标 / 论文笔记 / 跨引用存储。**不是市场数据**, 是**知识管理**。 |

**与 L1 名称冲突**: L1 也有 `factor_db.py` (L2), L7 是 `WikiFactorProxy`。两者完全不同:
- L2: OHLCV + factors in SQL/pandas
- L7: 因子 metadata in markdown (IC, IR, 论文笔记)

## 4. 如何选择层

```
需要 SQL 写操作?                → L1 BaseDBNode
需要透明缓存 + L1?              → L3 cache_node 包装 L1
需要 H5/CSV/NPY 文件?            → L4 FileFormatLoader
需要 iFinD facade?               → L5 DataLoader (封装 L4)
需要 polars + Table 4?           → L6 contracts.DataLoader
需要存储因子元数据/Wiki?         → L7 WikiFactorProxy
需要 ClickHouse (polars)?        → L6 ClickHouseDataLoader (TODO: 委托 L1)
```

## 5. Deprecated L2: `factor_node/factor_db.py`

L2 历史上为因子表的存储系统预留, 但**从未完整实现**:
- 所有 `FactorDB` 抽象方法都返回 `0` 或 `None`
- 没有具体的生产子类
- 仅被 `factor_node/__init__.py` re-export, 无业务调用

**处理方案**:
- ✅ **保留** L2 文件 (不影响运行, 保持向后兼容)
- ✅ **文档化** 标 DEPRECATED (本文档)
- 🔜 下个大版本 (v4.0) 删除 L2

## 6. 已知重复与 TODO

### ClickHouse HTTP 重复

`ClickHouseDataLoader._query_clickhouse` (`research/quant_alpha/evaluation/clickhouse_data_loader.py:112-145`) 用了 raw `http.client.HTTPConnection`, **绕过了** 自己 docstring 提到的 `database_node/clickhouse_node.py` 的 `CHBase` 客户端。

**TODO (P2.12c.3)**: 改为委托 `ClickHouseNode.query()`, 通过 `pl.from_pandas()` 转换 DataFrame。

**风险**: LOW — `ClickHouseNode` 是生产可用客户端, 测试已覆盖 (`tests/quant_alpha/test_table4_real.py:138`)。

## 7. 历史决策

- **2026-06-23**: v3.0 重构 (HKUDS nanobot 迁移) 时, `factor_node/factor_db.py` 未实现
- **2026-03 (Phase 3.3)**: `factor_test/utils/file_loaders.py` 从 `data_loader.py` 提取, 采用 registry pattern
- **2026-02 (Phase 2.5)**: `core/data_source.py` 文档化"两个独立数据接入子树"的设计决策
- **v2.x 时期**: L7 WikiFactorProxy 与 L2 FactorDB 命名相似导致混淆

## 8. 相关文件

| 文件 | 角色 |
|------|------|
| `QuantNodes/core/data_source.py` | `DataSource` ABC (L1+L4 父类) |
| `QuantNodes/database_node/__init__.py` | L1 factory + 注册 |
| `QuantNodes/database_node/base.py` | `BaseDBNode` ABC |
| `QuantNodes/database_node/clickhouse_node.py` | `ClickHouseNode` (生产用 CH 客户端) |
| `QuantNodes/factor_node/factor_db.py` | L2 (deprecated) |
| `QuantNodes/cache_node/base.py` | L3 cache 装饰器 |
| `QuantNodes/research/factor_test/utils/file_loaders.py` | L4 file format registry |
| `QuantNodes/research/quant_alpha/evaluation/contracts.py` | L6 `DataLoader` ABC |
| `QuantNodes/research/wiki.py` | L7 `WikiFactorProxy` |