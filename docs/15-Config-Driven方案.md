# Agent 配置文件驱动方案

> 合并自: 15-Agent配置文件驱动方案.md + 16-Polars配置驱动迁移方案.md + 21-Config-driven回测实现计划.md  
> 状态: 设计阶段  

---

## 一、设计背景与目标

### 1.1 问题陈述

当前 Agent 的工作流程中，Agent 需要编写大量逻辑代码来完成策略研发。

**痛点**：Agent 逻辑负担重，生成代码质量不稳定，难以保证测试通过。

### 1.2 设计目标

| 目标 | 说明 |
|------|------|
| **配置即策略** | Agent 编写 YAML 配置代替编写代码 |
| **自动闭环** | 配置 → 代码生成 → 验证 → 回测 自动执行 |
| **Agent 兜底** | 不可配置部分 Agent 补充自定义算子 |
| **测试自运行** | 配置中声明测试，Agent 不需要手动运行 |

---

## 二、整体架构

### 2.1 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                │
│  1. 编写 strategy_config.yaml                              │
│     ↓                                                      │
│  2. ConfigLoader 解析配置                                  │
│     ↓                                                      │
│  3. OperatorRegistry 检查覆盖度                          │
│     ├─ 全部可配置 → 4a                                     │
│     └─ 有无法表达 → 4b                                      │
│     ↓                                                      │
│  4a. FactorExecutor.run(config)                           │
│      → 自动生成代码 → 沙箱验证 → 运行测试                  │
│      ↓                                                     │
│  4b. 返回 [unresolved] → Agent 编写自定义算子             │
│      → 注册到 Registry → 重新检查                            │
│      ↓                                                     │
│  5. 合并执行 + 返回结果                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
agent/config/
├── __init__.py
├── loader.py                    # YAML配置解析器
├── executor.py                  # 配置执行器
├── types.py                     # 类型定义
└── templates/                   # YAML 策略模板
    ├── empty.yaml
    ├── momentum.yaml
    └── mean_reversion.yaml
```

---

## 三、配置文件格式

### 3.1 完整示例

```yaml
# strategy_config.yaml
version: "1.0"
name: "momentum_alpha_v1"
description: "动量因子策略"

# 1. 数据源定义
data:
  source: "csv"                    # csv/database/api
  path: "data/stock_data.csv"
  columns:
    - date
    - code
    - open
    - high
    - low
    - close
    - volume
  date_column: "date"
  code_column: "code"

# 2. 因子定义
factors:
  - name: momentum_20d
    type: expression
    formula: "close / close.shift(20) - 1"
    description: "20日动量因子"

  - name: turnover
    type: expression
    formula: "volume / volume.rolling(20).mean()"
    description: "20日平均换手率"

# 3. 因子运算 (可配置部分)
operations:
  # 时间序列运算
  - type: time_series
    name: momentum_20d_ma
    category: ts_mean              # 算子名称
    inputs: [momentum_20d]        # 引用因子
    params:
      window: 20
      min_periods: 10
  
  # 截面运算
  - type: section
    name: momentum_rank
    category: rank
    inputs: [momentum_20d_ma]
    params:
      method: "dense"            # dense/ordinal/percent

# 4. 因子组合
composite:
  - name: alpha_factor
    formula: "momentum_rank * 0.6 + (1 - turnover) * 0.4"
    normalize: true              # z-score标准化
    winsorize:
      lower: 0.01
      upper: 0.01

# 5. 回测配置
backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  universe: "A_stock"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001
  
  signals:
    buy_threshold: 0.05
    sell_threshold: -0.03
  positions:
    max_positions: 10
    rebalance_freq: "weekly"

# 6. 验证配置
validation:
  run_tests: true
  test_files:
    - "tests/test_factor*.py"
  
  metrics:
    ic_threshold: 0.02
    max_correlation: 0.7
    min_samples: 100
  
  # 自定义算子 (Agent额外实现)
  custom_operators:
    - "my_custom_operator.py"

# 7. 输出配置
output:
  format: "parquet"
  path: "outputs/result.parquet"
  save_signals: true
  save_positions: true
  save_equity_curve: true
```

### 3.2 简化版 (最小配置)

```yaml
# minimal_config.yaml
name: "simple_momentum"

factors:
  - name: returns
    formula: "close.pct_change(20)"

operations:
  - type: section
    category: rank
    inputs: [returns]

backtest:
  start_date: "2023-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
```

---

## 四、算子注册表

### 4.1 可配置的算子清单

#### 时间序列算子 (ts_*)

| 算子 | 参数 | 说明 |
|------|------|------|
| `ts_mean` | `window`, `min_periods`, `center` | 滚动均值 |
| `ts_sum` | `window`, `min_periods`, `center` | 滚动求和 |
| `ts_std` | `window`, `min_periods`, `center` | 滚动标准差 |
| `ts_max` | `window`, `min_periods`, `center` | 滚动最大值 |
| `ts_min` | `window`, `min_periods`, `center` | 滚动最小值 |
| `ts_median` | `window`, `min_periods` | 滚动中位数 |
| `ts_corr` | `window`, `min_periods`, `pair` | 滚动相关系数 |
| `ts_cov` | `window`, `min_periods`, `pair` | 滚动协方差 |
| `ts_rank` | `window` | 滚动排名 |
| `ts_delta` | `periods` | 差分 |
| `ts_pct_change` | `periods` | 百分比变化 |
| `ts_lag` | `periods` | 滞后 |

#### 截面算子 (section_*)

| 算子 | 参数 | 说明 |
|------|------|------|
| `rank` | `method` | 截面排名 |
| `scale` | `method`, `scale` | 归一化 |
| `winsorize` | `lower`, `upper` | 去极值 |
| `neutralize` | `method` | 行业中性 |
| `zscore` | | Z-score标准化 |

#### 算术算子

| 算子 | 参数 | 说明 |
|------|------|------|
| `add` | `scalar`, `factor` | 加法 |
| `sub` | `scalar`, `factor` | 减法 |
| `mul` | `scalar`, `factor` | 乘法 |
| `div` | `scalar`, `factor` | 除法 |
| `pow` | `exponent` | 幂运算 |
| `log` | | 对数 |
| `abs` | | 绝对值 |

### 4.2 自定义算子接口

```python
# custom_operators.py
def custom_momentum_zscore(
    data: pd.DataFrame,
    factor_name: str,
    window: int = 20,
    zscore_window: int = 60,
    **kwargs
) -> pd.DataFrame:
    """自定义动量z-score算子"""
    factor = data[factor_name]
    momentum = factor / factor.shift(window) - 1
    result = (momentum - momentum.rolling(zscore_window).mean()) / \
            momentum.rolling(zscore_window).std()
    return result.to_frame(factor_name)
```

---

## 五、核心模块设计

### 5.1 ConfigLoader (`loader.py`)

```python
class ConfigLoader:
    """YAML配置解析器"""
    
    def __init__(self, registry: "OperatorRegistry"):
        self.registry = registry
    
    def load(self, path: str) -> StrategyConfig:
        """加载YAML配置文件"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data)
    
    def check_coverage(self, config: StrategyConfig) -> CoverageReport:
        """检查配置覆盖度，返回不可配置的部分"""
        unresolved = []
        
        for op in config.operations:
            if op.category not in self.registry:
                unresolved.append(op)
        
        return CoverageReport(
            covered=[...],
            unresolved=unresolved
        )
```

### 5.2 OperatorRegistry (`registry.py`)

```python
class OperatorRegistry:
    """算子注册表 - 复用 factor_functions.py"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        from QuantNodes.factor_node.factor_functions import _OPERATOR_REGISTRY
        self._operators = _OPERATOR_REGISTRY
    
    def get(self, name: str) -> OperatorMetadata:
        """获取算子"""
        return self._operators.get(name)
    
    def list_by_category(self, category: str) -> List[str]:
        """按类别列出算子"""
        return [
            op.name for op in self._operators.values()
            if op.category == category
        ]
    
    def register_custom(self, module_path: str):
        """注册自定义算子"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name in dir(module):
            if name.startswith("custom_"):
                func = getattr(module, name)
                self.register(name, "custom", func, {})
```

### 5.3 FactorExecutor (`executor.py`)

```python
class FactorExecutor:
    """配置执行器：配置 → 代码生成 → 执行"""
    
    def __init__(self, registry: OperatorRegistry, sandbox: "CodeSandbox"):
        self.registry = registry
        self.sandbox = sandbox
    
    def run(self, config: StrategyConfig) -> ExecutionResult:
        """执行完整流程"""
        results = {}
        
        # 1. 生成因子代码
        code = self._generate_code(config)
        
        # 2. 沙箱验证
        validation_result = self.sandbox.validate(code)
        if not validation_result.is_safe:
            raise ExecutionError(f"Code not safe: {validation_result.errors}")
        
        # 3. 执行因子计算
        results["factors"] = self._compute_factors(config, code)
        
        # 4. 执行运算
        results["operations"] = self._compute_operations(config)
        
        # 5. 计算组合因子
        results["composite"] = self._compute_composite(config)
        
        # 6. 运行回测
        if config.backtest:
            results["backtest"] = self._run_backtest(config)
        
        # 7. 运行验证测试
        if config.validation.get("run_tests"):
            results["tests"] = self._run_tests(config)
        
        return ExecutionResult(**results)
```

---

## 六、Config-Driven 回测实现

### 6.1 架构决策

#### 原始方案（已废弃）

```
YAML Config → ConfigLoader → ConfigCodeGenerator (regex 转换) → BacktestTool (sandbox)
```

**废弃原因**：
- regex 替换不可靠
- 算子覆盖不全 (仅 ~46/317 个)
- 跨截面语义错误

#### 当前方案：直接 Polars 桥接

```
YAML Config → ConfigLoader → ConfigExecutor (Polars 因子计算) → ConfigBacktestRunner → backtest/ 引擎 (Pandas)
```

**优势**：
- 单点实现 — 因子逻辑只在 Polars 中实现一次
- 317 个算子全覆盖（含 174 个 TA-Lib 指标）
- 语义正确

### 6.2 数据流架构

```
StrategyConfig + pl.LazyFrame
        │
        ▼
ConfigExecutor.run_backtest(config, data)
        │  result.data = LazyFrame (所有因子列 + signal 列)
        │  result.backtest = {signals, config, thresholds}
        ▼
┌─ ConfigBacktestRunner.run() ────────────────────────────────┐
│                                                              │
│  1. df = result.data.collect().to_pandas()                   │
│  2. df = _normalize_columns(df)  # 统一 Code/close 大小写   │
│  3. ConfigStrategyNode._generate_signals(df) → OrdersResult  │
│  4. PositionLimitRiskNode.execute(orders) → RiskResult       │
│  5. ExecutionBrokerNode.execute((orders, df)) → TradeResult  │
│  6. _compute_statistics(trade_result) → BacktestResult       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 关键类接口

#### ConfigStrategyNode

```python
class ConfigStrategyNode(StrategyNode):
    def __init__(self, signal_col: str = "signal", **kwargs): ...
    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]: ...
```

输入: DataFrame 包含 `code`, `date`, `close`, `signal` 列（小写）  
输出: `List[Signal]`

#### ConfigBacktestRunner

```python
class ConfigBacktestRunner:
    def run(self, config: StrategyConfig, data: pl.LazyFrame) -> BacktestResult: ...
```

输入: `StrategyConfig` + `pl.LazyFrame`  
输出: `BacktestResult`

### 6.4 文件变更清单

| 操作 | 文件 | 说明 | 行数 |
|------|------|------|------|
| 新建 | `backtest/config_strategy.py` | ConfigStrategyNode | ~50 |
| 新建 | `backtest/config_runner.py` | ConfigBacktestRunner | ~150 |
| 修改 | `agent/config/executor.py` | run_backtest() | - |
| 修改 | `agent/tools/config_backtest.py` | 重写 | - |
| 删除 | `agent/config/generator.py` | 废弃 | - |

---

## 七、Agent 工作流程集成

### 7.1 完整流程

```
1. Agent 分析用户需求
   → "帮我生成一个20日动量因子策略，回测2023年"

2. Agent 编写配置文件
   → strategy_config.yaml

3. ConfigLoader 加载并检查
   → CoverageReport(
       covered: ["ts_mean", "rank", ...],
       unresolved: []
     )

4. FactorExecutor 自动执行
   → 生成代码 → 验证 → 回测 → 验证测试

5. 返回结果
   → IC: 0.05, 回测收益: 15%, Sharpe: 1.2
```

### 7.2 需要 Agent 兜底的流程

```
1. Agent 编写配置文件
   → 包含自定义算子 "custom_momentum_zscore"

2. ConfigLoader 检查覆盖度
   → unresolved: ["custom_momentum_zscore"]

3. Agent 编写自定义算子
   → custom_operators.py

4. 重新执行 → 成功
```

---

## 八、配置文件模板

### 动量因子配置

```yaml
version: "1.0"
name: "momentum_factor"
description: "经典动量因子"

factors:
  - name: returns_20d
    formula: "close / close.shift(20) - 1"
  
  - name: returns_60d
    formula: "close / close.shift(60) - 1"

operations:
  - type: time_series
    name: returns_20d_ma
    category: ts_mean
    inputs: [returns_20d]
    params:
      window: 20
  
  - type: section
    name: momentum_rank
    category: rank
    inputs: [returns_20d_ma]

composite:
  - name: alpha
    formula: "momentum_rank"
    normalize: true

backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
```

---

## 九、错误处理

| 错误类型 | 说明 | 处理方式 |
|----------|------|----------|
| `ConfigParseError` | YAML格式错误 | Agent 修复配置 |
| `OperatorNotFound` | 算子不存在 | Agent 实现自定义算子 |
| `CodeValidationError` | 代码不安全 | Agent 修复代码 |
| `ExecutionError` | 执行失败 | Agent 调试 |
| `BacktestError` | 回测失败 | Agent 调整配置 |
| `TestFailure` | 测试失败 | Agent 修复代码/配置 |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 表达式解析复杂度 | 高 | 分阶段实现，从简单开始 |
| Polars API 变化 | 中 | 版本固定 |
| 现有代码破坏 | 高 | 完全保留，不修改 |
| 向后兼容 | 中 | 保留旧路径，渐进迁移 |

---

## 十一、数据加载设计

### 11.1 设计目标

**一句话**: 用户只需写好 YAML + conn.ini，系统自动从任意数据源加载数据，执行回测，输出结果。

**约束**: 数据加载必须全部通过 `database_node`（`BaseDBNode` 子类），不由 `config_backtest.py` 自行实现 IO。

### 11.2 BaseDBNode 基类

```python
class BaseDBNode(QuantNodesObject):
    """数据库节点基类 - 统一接口"""

    def connect(self) -> Any:
        """建立数据库连接"""
        raise NotImplementedError

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """执行 SQL 查询，返回 DataFrame"""
        raise NotImplementedError

    def execute(self, sql: str, params: tuple = None) -> int:
        """执行 DDL/DML 语句，返回影响行数"""
        raise NotImplementedError

    def insert_df(self, df: pd.DataFrame, table: str, if_exists: str = 'append') -> int:
        """插入 DataFrame 到数据库"""
        raise NotImplementedError

    def disconnect(self) -> None:
        """关闭数据库连接"""
        raise NotImplementedError

    def health_check(self) -> bool:
        """健康检查"""
        raise NotImplementedError
```

### 11.3 各 Node 实现概览

| 节点 | 功能 | 数据源 | 连接池 | WHERE 过滤 |
|------|------|--------|---------|-----------|
| `SQLiteNode` | 查询/插入 | `.db` / `:memory:` | ❌ | ✅ 完整 SQL |
| `DuckDBNode` | 查询/插入/分析 | `.duckdb` / `:memory:` | ❌ | ✅ 完整 SQL |
| `MySQLNode` | 查询/插入/DDL | 远程服务器 | ✅ 可配置 | ✅ 完整 SQL |
| `ClickHouseNode` | 查询/插入/DDL | 远程服务器 | ✅ 可配置 | ✅ 完整 SQL |
| `CSVNode` | 读取/过滤 | `.csv` | ❌ | ⚠️ 仅 WHERE |
| `ParquetNode` | 读取/过滤 | `.parquet` | ❌ | ⚠️ 仅 WHERE |

### 11.4 YAML 数据配置格式

#### 数据库类 (clickhouse / mysql)

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

#### 文件类 (csv / parquet)

```yaml
data:
  source: "csv"
  path: "data/stock_data.csv"
  columns: [date, code, open, high, low, close, volume]
  date_column: "date"
  code_column: "code"
```

#### 嵌入式数据库 (sqlite / duckdb)

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

#### conn.ini 格式

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

### 11.5 列名映射策略

**核心原则**: 在数据加载层统一列名，下游组件全部使用标准列名。

**标准列名**: `date`, `code`, `open`, `high`, `low`, `close`, `volume`

| 数据源原始列名 | 标准列名 | 映射方式 |
|---------------|---------|---------|
| `ts_code` | `code` | `column_mapping` |
| `trade_date` | `date` | `column_mapping` |
| `vol` | `volume` | `column_mapping` |
| `stock_code` | `code` | `column_mapping` |
| `Code` / `code` | `code` | 自动 |

**DateTime 处理**: ClickHouse/MySQL 返回 DateTime 类型，需要转为 Date：
```python
if df["date"].dtype == pl.Datetime:
    df = df.with_columns(pl.col("date").cast(pl.Date))
```

### 11.6 SQL 构建逻辑

```python
def _build_query(data_config, universe_codes=None):
    """从配置构建 SQL 查询"""
    cols = data_config.columns or ["*"]
    cols_str = ", ".join(cols)
    table = data_config.table

    sql = f"SELECT {cols_str} FROM {table}"

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

    # 排序
    date_col = data_config.date_column
    code_col = data_config.code_column
    sql += f" ORDER BY {code_col}, {date_col}"

    return sql
```

### 11.7 数据加载流程

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
