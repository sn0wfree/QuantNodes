# QuantNodes

> 量化研究节点架构 - 基于 Pipeline 组合原语的统一量化分析平台

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目简介

QuantNodes 是一个面向量化研究的节点架构平台，通过统一的 **BaseNode + Pipeline** 模式，实现因子计算、回测分析、数据库查询的无缝集成。

### 核心特性

- **统一节点架构**: 万物皆 Node，Pipeline 是唯一组合原语
- **97+ 内置算子**: 涵盖时间序列、截面运算、多截面聚合等
- **多数据库支持**: ClickHouse、DuckDB、MySQL、SQLite、CSV、Parquet
- **AI 原生设计**: 内置策略生成器和 Pipeline 优化器
- **零 QuantStudio 依赖**: 完全自主实现，代码清晰可控

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Meta-Programming AI                                │
│  StrategyGenerator │ PipelineOptimizer │ CodeSandbox          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Pipeline 组合原语                                    │
│  Pipeline │ Parallel │ Join │ IfNode │ MapNode │ WhileNode   │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 处理节点                                            │
│  DatabaseNode │ FactorNode │ BacktestNode │ UINode │ ConfigNode │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
QuantNodes/
├── QuantNodes/                    # 主包
│   ├── core/                      # 核心架构
│   │   ├── node.py                # BaseNode, Pipeline
│   │   ├── control.py             # IfNode, MapNode, WhileNode
│   │   ├── expression.py          # Expression, LambdaExpression
│   │   ├── serialization.py       # 序列化支持
│   │   ├── data_preprocessing.py  # DataPreprocessingFun
│   │   └── pandas_utils.py        # Pandas 工具函数
│   │
│   ├── factor_node/               # 因子引擎
│   │   ├── factor.py             # Factor, DerivativeFactor
│   │   ├── factor_functions.py   # 143+ 算子 + 注册表 (Polars)
│   │   └── factor_operation.py    # Point/Time/Section/PanelOperation
│   │
│   ├── database_node/             # 数据库节点
│   │   ├── clickhouse_node.py     # ClickHouse 引擎
│   │   ├── duckdb_node.py         # DuckDB 引擎
│   │   ├── mysql_node.py          # MySQL 引擎
│   │   └── csv_node.py            # CSV 读取
│   │
│   ├── symbolic/                  # 符号计算引擎
│   │   ├── expression.py          # SQL 表达式 AST
│   │   ├── compiler.py            # SQL 编译器
│   │   └── dialect.py             # ClickHouse/DuckDB/MySQL 方言
│   │
│   ├── backtest/                 # 回测引擎
│   │   ├── backtest_node.py       # BacktestNode
│   │   ├── strategy_node.py       # StrategyNode
│   │   └── broker_node.py         # BrokerNode
│   │
│   ├── operator_node/             # SQL 构建节点
│   ├── ai/                        # AI 元节点
│   ├── conf_node/                 # 配置节点
│   ├── ui_node/                   # UI 数据准备节点
│   └── app/                       # Streamlit 应用
│
├── tests/                        # 测试套件 (413 tests)
├── examples/                     # 示例代码
└── docs/                        # 设计文档
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 基本使用

```python
from QuantNodes.core.node import BaseNode, Pipeline
from QuantNodes.factor_node.factor_functions import get_operator, list_operators

# 列出所有算子
print(list_operators())  # ['abs', 'add', 'aggr_count', ...]

# 获取算子
rolling_mean = get_operator('rolling_mean', 'time')

# 构建 Pipeline
pipeline = Pipeline(
    source_node,
    transform_node,
    output_node
)

# 执行
result = pipeline.execute()
```

### 因子计算示例

```python
from QuantNodes.factor_node.factor_functions import rolling_mean, rolling_std, zscore

# 滚动均值
result = rolling_mean(factor_data, window=20)

# 滚动标准差
std = rolling_std(factor_data, window=20)

# Z-Score 标准化
zscore = zscore(factor_data)
```

## 算子注册表 API

```python
from QuantNodes.factor_node.factor_functions import (
    list_operators,    # 列出所有算子
    get_operator,      # 获取算子函数
    operator_info,     # 获取算子元信息
    generate_documentation,  # 生成文档
)

# 按类别列出算子
point_ops = list_operators('point')      # 单点运算
time_ops = list_operators('time')       # 时间序列运算
section_ops = list_operators('section') # 截面运算
multi_ops = list_operators('multi_section')  # 多截面运算

# 获取算子详情
info = operator_info('rolling_mean', 'time')
print(info)
# {'name': 'rolling_mean', 'category': 'time', 'doc': '...', 'parameters': [...]}

# 生成 Markdown 文档
docs = generate_documentation('markdown')
```

## 算子分类

| 类别 | 数量 | 示例 |
|------|------|------|
| **Point (单点)** | 31 | `abs`, `log`, `sign`, `isnull` |
| **Time (时间序列)** | 42 | `rolling_mean`, `rolling_std`, `ewm`, `diff` |
| **Section (截面)** | 8 | `standardizeZScore`, `winsorize`, `fillNaNByVal` |
| **Multi-Section (多截面)** | 12 | `aggregate`, `disaggregate`, `aggr_sum` |

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_factor_functions.py -v
pytest tests/agent/test_config_executor.py -v

# 查看覆盖率
pytest tests/ --cov=QuantNodes --cov-report=html
```

## 设计文档

- [架构设计文档](docs/04-架构设计文档.md)
- [统一架构最终方案](docs/06-统一架构最终方案.md)
- [执行清单](docs/07-执行清单.md)
- [移除 QuantStudio 依赖重构方案](docs/09-移除QuantStudio依赖重构方案.md)

## 变更日志

### v0.2.0 (2026-04-28)

- ✅ 实现 DataPreprocessingFun 类，完全移除 QuantStudio 依赖
- ✅ 统一 PointOperation/TimeOperation 空数据处理
- ✅ 添加参数边界验证 (winsorize, half_life)
- ✅ 改进 rolling_regress 错误处理
- ✅ 修复 rolling_change_rate 除法顺序问题
- ✅ 添加 pandas_utils 工具模块
- ✅ 添加 Streamlit UI 应用骨架

### v0.1.0 (2026-04-27)

- ✅ 完成 9/9 阶段重构
- ✅ 97+ 因子算子实现
- ✅ 完整测试套件 (413 tests)
- ✅ 多数据库支持

## 许可证

MIT License
