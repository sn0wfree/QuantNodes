# 25-Config-Driven 回测数据加载设计

> 本文档详细记录功能1（配置驱动回测）的数据加载架构设计，包括 database_node 统一接入、列名映射、改动范围等。

---

## 一、设计目标

**一句话**: 用户只需写好 YAML + conn.ini，系统自动从任意数据源加载数据，执行回测，输出结果。

**约束**: 数据加载必须全部通过 `database_node`（`BaseDBNode` 子类），不由 `config_backtest.py` 自行实现 IO。

---

## 二、database_node 后端参数映射

| source | Node 类 | 构造参数 | 参数来源 | SQL 支持 |
|--------|---------|---------|---------|---------|
| `clickhouse` | `ClickHouseNode` | `host`, `port`, `user`, `passwd`, `database`, `interface`, `pool_size` | conn.ini | ✅ 完整 SQL |
| `mysql` | `MySQLNode` | `host`, `port`, `user`, `passwd`, `db`, `charset`, `pool_size` | conn.ini | ✅ 完整 SQL |
| `sqlite` | `SQLiteNode` | `database` (路径或 `:memory:`) | YAML `path` | ✅ 完整 SQL |
| `duckdb` | `DuckDBNode` | `database` (路径或 `:memory:`), `read_only` | YAML `path` | ✅ 完整 SQL |
| `csv` | `CSVNode` | `filepath`, `encoding`, `sep` | YAML `path` | ⚠️ 仅 WHERE 过滤 |
| `parquet` | `ParquetNode` | `filepath` | YAML `path` | ⚠️ 仅 WHERE 过滤 |

**所有 Node 统一接口**: `BaseDBNode.query(sql) -> pd.DataFrame`

---

## 三、YAML 配置格式

### 3.1 数据库类 (clickhouse / mysql)

```yaml
data:
  source: "clickhouse"
  conn_ini: "conn.ini"
  conn_section: "ClickHouse"
  table: "quote.cn_stock"
  query_filter: "WHERE trade_date >= '2022-01-01'"
  columns: [ts_code, trade_date, open, high, low, close, vol]
  date_column: "trade_date"
  code_column: "ts_code"
  column_mapping:
    ts_code: "code"
    trade_date: "date"
    vol: "volume"
```

### 3.2 文件类 (csv / parquet)

```yaml
data:
  source: "csv"
  path: "data/stock_data.csv"
  columns: [date, code, open, high, low, close, volume]
  date_column: "date"
  code_column: "code"
```

### 3.3 嵌入式数据库 (sqlite / duckdb)

```yaml
data:
  source: "duckdb"
  path: "data/market.duckdb"
  table: "daily_kline"
  date_column: "date"
  code_column: "code"
  column_mapping:
    trade_date: "date"
```

### 3.4 conn.ini 格式

```ini
[ClickHouse]
host = 0.0.0.0
port = 8123
user = data
passwd = 123456
db = quote

[MySQL]
host = localhost
port = 3306
user = root
passwd = password
db = market
```

---

## 四、数据加载流程

```
YAML data 配置
  │
  ▼
ConfigBacktestTool._load_data()
  │
  ├─ data_path 参数 (向后兼容)
  │    └── _read_data_file(path) → pl.scan_csv/parquet
  │
  ├─ config.data.source == "csv" / "parquet"
  │    └── CSVNode/ParquetNode(path).query() → pd.DataFrame
  │
  ├─ config.data.source == "clickhouse" / "mysql"
  │    ├── IniConfigNode(conn_ini, section).execute() → dict
  │    ├── ClickHouseNode/MySQLNode(**conn_params)
  │    ├── 构建 SQL: SELECT {columns} FROM {table} {query_filter}
  │    └── node.query(sql) → pd.DataFrame
  │
  ├─ config.data.source == "sqlite" / "duckdb"
  │    ├── SQLiteNode/DuckDBNode(database=path)
  │    └── node.query(sql) → pd.DataFrame
  │
  ▼
列名映射: df.rename(columns=column_mapping)
  │
  ▼
DateTime → Date 类型转换 (如需)
  │
  ▼
pl.from_pandas(df) → pl.LazyFrame
```

---

## 五、列名映射策略

**核心原则**: 在数据加载层统一列名，下游组件（executor, runner, strategy, broker）全部使用标准列名。

**标准列名**: `date`, `code`, `open`, `high`, `low`, `close`, `volume`

| 数据源原始列名 | 标准列名 | 映射方式 |
|---------------|---------|---------|
| `ts_code` | `code` | `column_mapping` |
| `trade_date` | `date` | `column_mapping` |
| `vol` | `volume` | `column_mapping` |
| `stock_code` | `code` | `column_mapping` |
| `Code` / `code` | `code` | 自动 (不需要 mapping) |

**DateTime 处理**: ClickHouse/MySQL 返回 DateTime 类型，需要在数据加载层转为 Date：
```python
if df["date"].dtype == pl.Datetime:
    df = df.with_columns(pl.col("date").cast(pl.Date))
```

**下游兼容**: `_normalize_columns()` 保持 `code→Code`, `close→Close`, `open→Open` 的映射不变，确保 broker/strategy 节点正常工作。

---

## 六、SQL 构建逻辑

```python
def _build_query(data_config, universe_codes=None):
    """从配置构建 SQL 查询"""
    cols = data_config.columns or ["*"]
    cols_str = ", ".join(cols)
    table = data_config.table

    sql = f"SELECT {cols_str} FROM {table}"

    # WHERE 条件拼接
    where_parts = []

    # Universe 过滤 (下推到数据库)
    if universe_codes:
        code_col = data_config.code_column
        codes_str = ", ".join(f"'{c}'" for c in universe_codes)
        where_parts.append(f"{code_col} IN ({codes_str})")

    # 用户自定义过滤
    if data_config.query_filter:
        where_parts.append(data_config.query_filter.lstrip("WHERE "))

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    # 排序 (因子计算需要)
    date_col = data_config.date_column
    code_col = data_config.code_column
    sql += f" ORDER BY {code_col}, {date_col}"

    return sql
```

---

## 七、改动范围

### 7.1 types.py — 扩展 DataConfig

```python
@dataclass
class DataConfig:
    """数据源配置"""
    source: str = "csv"
    path: str = ""
    table: str = ""                                    # 新增: 数据库表名
    conn_ini: str = "conn.ini"                         # 新增: INI 配置文件路径
    conn_section: str = "ClickHouse"                   # 新增: INI section 名
    columns: List[str] = field(default_factory=list)   # 新增: 指定查询列
    date_column: str = "date"
    code_column: str = "code"
    column_mapping: Dict[str, str] = field(default_factory=dict)  # 新增: 列名映射
    query_filter: str = ""                             # 新增: 额外 SQL WHERE 条件
```

改动量: ~6 行新增字段

### 7.2 loader.py — 解析新字段

`_parse()` 方法中 `data` 配置解析部分，新增:
```python
table=d.get("table", ""),
conn_ini=d.get("conn_ini", "conn.ini"),
conn_section=d.get("conn_section", "ClickHouse"),
column_mapping=d.get("column_mapping", {}),
query_filter=d.get("query_filter", ""),
```

改动量: ~6 行

### 7.3 config_backtest.py — 核心改动

新增 `_load_from_db()` 方法，统一处理所有 `database_node` 后端:

```python
def _load_data(self, config, data_path=None):
    import polars as pl

    # 向后兼容: data_path 参数
    if data_path:
        return self._read_data_file(data_path)

    if config.data is None:
        raise ValueError("No data config provided")

    source = config.data.source

    # 文件类: 通过 database_node 统一加载
    if source in ("csv", "parquet", "sqlite", "duckdb"):
        return self._load_from_db(config.data)

    # 数据库类: 通过 database_node 统一加载
    if source in ("clickhouse", "mysql"):
        return self._load_from_db(config.data)

    # 向后兼容: 旧的 path 模式
    if config.data.path:
        return self._read_data_file(config.data.path)

    raise ValueError(f"Unsupported data source: {source}")


def _load_from_db(self, data_config) -> pl.LazyFrame:
    """通过 database_node 统一加载数据"""
    import polars as pl

    # 1. 根据 source 选择 Node 类
    NodeClass = self._get_node_class(data_config.source)

    # 2. 获取构造参数
    node_kwargs = self._get_node_params(data_config)

    # 3. 构建 SQL
    sql = self._build_query(data_config)

    # 4. 通过 database_node 执行查询
    with NodeClass(**node_kwargs) as node:
        df = node.query(sql)

    # 5. 列名映射
    if data_config.column_mapping:
        df = df.rename(columns=data_config.column_mapping)

    # 6. 转 Polars LazyFrame
    return pl.from_pandas(df).lazy()


def _get_node_class(self, source: str):
    """根据 source 返回 Node 类"""
    from QuantNodes.database_node import (
        ClickHouseNode, MySQLNode, SQLiteNode,
        DuckDBNode, CSVNode, ParquetNode
    )
    NODE_MAP = {
        "clickhouse": ClickHouseNode,
        "mysql": MySQLNode,
        "sqlite": SQLiteNode,
        "duckdb": DuckDBNode,
        "csv": CSVNode,
        "parquet": ParquetNode,
    }
    if source not in NODE_MAP:
        raise ValueError(f"Unsupported source: {source}")
    return NODE_MAP[source]


def _get_node_params(self, data_config) -> dict:
    """获取 Node 构造参数"""
    source = data_config.source

    if source in ("clickhouse", "mysql"):
        # 从 conn.ini 读取连接参数
        from QuantNodes.conf_node.ini_config import IniConfigNode
        ini = IniConfigNode(data_config.conn_ini, section=data_config.conn_section)
        conn = ini.execute()
        if source == "clickhouse":
            return {
                "host": conn.get("host", "localhost"),
                "port": int(conn.get("port", 8123)),
                "user": conn.get("user", "default"),
                "passwd": conn.get("passwd", ""),
                "database": conn.get("db", "default"),
            }
        else:  # mysql
            return {
                "host": conn.get("host", "localhost"),
                "port": int(conn.get("port", 3306)),
                "user": conn.get("user", "root"),
                "passwd": conn.get("passwd", ""),
                "db": conn.get("db", "default"),
            }

    elif source in ("sqlite", "duckdb"):
        return {"database": data_config.path}

    elif source == "csv":
        return {"filepath": data_config.path}

    elif source == "parquet":
        return {"filepath": data_config.path}

    raise ValueError(f"Unsupported source: {source}")


def _build_query(self, data_config) -> str:
    """从配置构建 SQL 查询"""
    cols = data_config.columns or ["*"]
    cols_str = ", ".join(cols)
    table = data_config.table

    # 文件类节点不需要 FROM 子句
    if data_config.source in ("csv", "parquet"):
        return None

    sql = f"SELECT {cols_str} FROM {table}"

    if data_config.query_filter:
        where = data_config.query_filter.strip()
        if not where.upper().startswith("WHERE"):
            where = f"WHERE {where}"
        sql += f" {where}"

    return sql
```

改动量: ~80 行

### 7.4 executor.py — 信号输出修复

行 437: `data.select("date", "code", "signal")` → 用 `config.data.code_column` 替换硬编码

改动量: ~2 行

### 7.5 runner.py — groupby 修复

行 213, 291: `groupby("code")` → `groupby("Code")` (与 `_normalize_columns` 产出一致)

改动量: ~2 行

### 7.6 conn.ini — 新建

```ini
[ClickHouse]
host = 0.0.0.0
port = 8123
user = data
passwd = 123456
db = quote
```

---

## 八、改动汇总

| # | 文件 | 改动 | 行数 | 类型 |
|---|------|------|------|------|
| 1 | `types.py` | `DataConfig` 新增 6 个字段 | ~6 | 新增字段 |
| 2 | `loader.py` | `_parse()` 解析新字段 | ~6 | 新增解析 |
| 3 | `config_backtest.py` | `_load_data()` 分发 + `_load_from_db()` + `_get_node_class()` + `_get_node_params()` + `_build_query()` | ~80 | 核心改动 |
| 4 | `executor.py` | 信号输出用 `code_column` | ~2 | bug 修复 |
| 5 | `runner.py` | `groupby("Code")` 统一 | ~2 | bug 修复 |
| 6 | `conn.ini` | ClickHouse 连接配置 | 新文件 | 新增文件 |

**总改动量**: ~96 行 + 1 个新文件，涉及 **5 个现有文件**。

---

## 九、测试验证

### 测试步骤

1. 创建 `conn.ini` 配置 ClickHouse 连接
2. 编写 YAML 配置（source: clickhouse, table: quote.cn_stock）
3. 运行 `ConfigBacktestTool.execute(config_path="test.yaml")`
4. 验证输出文件存在且数据正确

### 验证点

| 验证项 | 预期 |
|--------|------|
| ClickHouse 数据加载 | LazyFrame 非空，行数 > 0 |
| 列名映射 | ts_code→code, trade_date→date, vol→volume |
| DateTime→Date 转换 | date 列为 Date 类型 |
| 因子计算 | 因子值非空 |
| 信号生成 | signal 列存在，值为 1/-1/0 |
| 回测执行 | 交易记录非空 |
| 统计指标 | sharpe, max_drawdown, win_rate 存在 |
| 输出文件 | parquet/json 文件存在 |

---

*文档版本: v1.0*
*创建日期: 2026-04-30*
*状态: 设计完成，待实施*
