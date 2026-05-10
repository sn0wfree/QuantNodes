# QuantNodes

> 量化研究节点架构 - 基于 Pipeline 组合原语的统一量化分析平台

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目简介

QuantNodes 是一个面向量化研究的节点架构平台，通过统一的 **BaseNode + Pipeline** 模式，实现因子计算、回测分析、数据库查询的无缝集成。

### 核心特性

- **统一节点架构**: 万物皆 Node，Pipeline 是唯一组合原语
- **317+ 内置算子**: 涵盖时间序列、截面运算、多截面聚合等，装饰器注册表模式
- **多数据库支持**: ClickHouse、DuckDB、MySQL、SQLite、CSV、Parquet
- **AI 原生设计**: 内置策略生成器和 Pipeline 优化器
- **Config-Driven 回测**: YAML 配置文件驱动回测，支持算子扩展
- **零 QuantStudio 依赖**: 完全自主实现，代码清晰可控
- **Agent 智能体**: 15 个内置工具，支持文件操作、代码搜索、Git、Web 搜索等
- **真流式输出**: Token 级别实时流式响应，工具调用过程可视化
- **Markdown 渲染**: Agent 回复支持代码高亮、表格、列表等富文本格式
- **Session 管理**: 多会话切换、历史记录、新建/删除对话

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
│   ├── agent/                     # Agent 智能体系统
│   │   ├── core/                  # Agent 核心（loop, runner, context）
│   │   ├── tools/                 # 15 个内置工具
│   │   │   ├── file_ops.py        # 文件读/写/编辑
│   │   │   ├── code_search.py     # 代码搜索（grep/find）
│   │   │   ├── git_ops.py         # Git 操作
│   │   │   ├── web_fetch.py       # 网页抓取
│   │   │   ├── web_search.py      # 网络搜索
│   │   │   ├── task.py            # 任务管理
│   │   │   └── ...                # 更多工具
│   │   └── providers/             # LLM Provider
│   │
│   ├── cli/                       # 命令行工具
│   │   ├── __init__.py            # CLI 主入口
│   │   └── enhanced.py            # Rich Agent 交互终端
│   │
│   ├── core/                      # 核心架构
│   │   ├── node.py                # BaseNode, Pipeline
│   │   ├── control.py             # IfNode, MapNode, WhileNode
│   │   ├── expression.py          # Expression, LambdaExpression
│   │   ├── serialization.py       # 序列化支持
│   │   ├── data_preprocessing.py  # DataPreprocessingFun
│   │   └── pandas_utils.py        # Pandas 工具函数
│   │
│   ├── factor_node/               # 因子引擎
│   │   ├── factor.py              # Factor, DerivativeFactor
│   │   ├── factor_functions.py    # 317+ 算子 + 注册表 (Polars)
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
├── tests/                        # 测试套件
├── examples/                     # 示例代码
└── docs/                        # 设计文档
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 初始化项目

```bash
# 初始化配置文件（创建 .env 和 conn.ini）
quantnodes init
# 或使用 Python 模块方式
python -m QuantNodes init
```

### 启动服务

```bash
# 启动全部服务（前端 + API）
quantnodes run

# 启动参数
quantnodes run --host 0.0.0.0 --port 8080 --api-port 8000 --daemon

# 仅启动 API
quantnodes run --api-only

# 仅启动前端
quantnodes run --frontend-only
```

### Agent 对话

```bash
# 交互模式
quantnodes chat

# 单次提问
quantnodes chat "帮我生成一个动量因子"

# 指定工作目录
quantnodes chat --workspace ./my_project
```

### Agent 工具

| 工具 | 功能 | read_only |
|------|------|-----------|
| `file_ops` | 文件读/写/编辑/列表/glob | ❌ |
| `code_search` | grep/ find_files/ search_code | ✅ |
| `git_ops` | status/ diff/ log/ commit | ❌ |
| `web_fetch` | 网页抓取 | ✅ |
| `web_search` | DuckDuckGo 搜索 | ✅ |
| `task` | 任务管理（创建/更新/列表） | ❌ |
| `pipeline` | Pipeline 验证 | ✅ |
| `factor` | 因子分析 | ✅ |
| `backtest` | 回测执行 | ❌ |
| `config_backtest` | YAML 配置回测 | ❌ |
| `strategy` | 策略生成 | ❌ |
| `wiki` | Wiki 知识库 | ❌ |
| `sandbox` | 代码沙箱验证 | ✅ |
| `echo` | 回声测试 | ✅ |

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
| **Point (单点)** | 46 | `abs`, `log`, `sign`, `isnull` |
| **Time (时间序列)** | 65 | `rolling_mean`, `rolling_std`, `ewm`, `diff` |
| **Section (截面)** | 17 | `standardizeZScore`, `winsorize`, `fillNaNByVal` |
| **Multi-Section (多截面)** | 15 | `aggregate`, `disaggregate`, `aggr_sum` |
| **TA-Lib** | 174 | `talib_rsi`, `talib_sma`, `talib_macd_line` |
| **Custom (用户自定义)** | ∞ | `@CustomOperator.point()`, Builder API |

## 自定义算子 API

```python
from QuantNodes.operators import CustomOperator

# 装饰器风格（直接注册）
@CustomOperator.point("my_double")
def my_double(f, multiplier=2.0):
    return f * multiplier

# Builder 链式风格
my_ewm_30 = (CustomOperator.time("my_ewm_30")
    .param("span", int, 30, "窗口大小")
    .execute(lambda s, span: s.ewm_mean(span=span))
    .register())

# 模板工厂（基于内置算子创建）
from QuantNodes.operators import CustomOperator
my_ewm_30 = CustomOperator.time_from("my_ewm_30", "ewm_mean", span=30)

# 级联查询：自定义算子优先，内置算子 fallback
from QuantNodes.factor_node.factor_functions import get_operator
op = get_operator("my_double")  # 先查自定义，再查内置
```

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

- [架构设计](docs/04-架构设计.md)
- [执行清单](docs/07-执行清单.md)
- [算子系统设计与规范](docs/22-算子系统设计与规范.md)
- [核心功能框架设计](docs/24-核心功能框架设计.md)
- [大型项目开发测试规范](docs/大型项目开发测试规范.md)
- [操作手册](docs/QuantNodes-操作手册.md)
- [Feature 3A - WikiFactorProxy](docs/Feature3A-实施计划.md)
- [Feature 3B - 研报复现](docs/Feature3B-实施计划.md)
- [Feature 3C - AutoResearch 自动因子挖掘](docs/Feature3C-实施计划.md)
- [Feature 3D - 用户友好自定义算子 API](docs/Feature3D-用户友好自定义算子API设计.md)

## 变更日志

### v0.5.0 (2026-05-10)

- ✅ Agent 智能体系统：15 个内置工具，支持文件操作、代码搜索、Git、Web 搜索等
- ✅ 新增 `web_fetch` 工具：网页抓取，支持 text/html 格式
- ✅ 新增 `web_search` 工具：DuckDuckGo 网络搜索
- ✅ 新增 `task` 工具：任务管理（创建/更新/列表），JSON 持久化
- ✅ 新增 `quantnodes chat` 命令：Rich Agent 交互终端
- ✅ CLI 重构为包结构（`cli/`）
- ✅ 修复 `compile_expression` 方言 import Bug
- ✅ 修复 Agent `api_base` URL 双重拼接 Bug
- ✅ 修复 `should_execute_tools` 逻辑 Bug
- ✅ 2698+ 测试用例

### v0.4.1 (2026-05-09)

- ✅ CLI 优化：依赖安装自动化（Python 依赖通过 pyproject.toml，前端依赖首次 import 时自动安装）
- ✅ CLI 优化：init 流程简化（只初始化 Wiki 和创建配置文件）
- ✅ llmwikify Wiki 集成：初始化时自动创建 raw/, wiki/, wiki.md, index.md, log.md
- ✅ 目录结构优化：Wiki 放在项目根目录，与 data/ 平级

### v0.4.0 (2026-05-08)

- ✅ Feature 3A - WikiFactorProxy：因子库读写接口，支持 llmwikify 解析 PDF
- ✅ Feature 3B - 研报复现：从研报提取因子公式，验证有效性，存入 Wiki 因子库
- ✅ Feature 3C - AutoResearch 自动因子挖掘：模板枚举 + MCTS 搜索自动生成验证因子
- ✅ Feature 3D - 用户友好自定义算子 API：装饰器风格 + Builder 链式 + 模板工厂 + 级联查询
- ✅ 317+ 内置算子（point: 46, time: 65, section: 17, multi_section: 15, talib: 174）
- ✅ 2574+ 测试用例，覆盖率 ~80%
- ✅ 统一前端到 Vue 3 + Ant Design Vue 4.x

### v0.3.0 (2026-04-30)

- ✅ 实现 config-driven 回测的算子扩展机制（通用 fallback + custom_operators 加载）
- ✅ 迁移 6 个独有算子到 factor_functions（mad, weighted_aggr_mean, fill_null_by_strategy 等）
- ✅ 删除 factor_nodes.py，统一算子系统（-1274 行）
- ✅ ts_corr 协方差公式修复 + group_winsorize 实现
- ✅ 算子构建规范文档

### v0.2.0 (2026-04-28)

- ✅ 实现 DataPreprocessingFun 类，完全移除 QuantStudio 依赖
- ✅ 统一 PointOperation/TimeOperation 空数据处理
- ✅ 添加参数边界验证 (winsorize, half_life)
- ✅ 改进 rolling_regress 错误处理
- ✅ 修复 rolling_change_rate 除法顺序问题
- ✅ 添加 pandas_utils 工具模块
- ✅ 添加 Vue 3 前端应用骨架 (后统一为 Vue)

### v0.1.0 (2026-04-27)

- ✅ 完成 9/9 阶段重构
- ✅ 97+ 因子算子实现
- ✅ 完整测试套件
- ✅ 多数据库支持

## 致谢

本项目的因子计算系统设计深受 **QuantStudio** 启发。

### 开源社区

- [Polars](https://pola.rs/) - 高性能 DataFrame 库
- [Vue 3](https://vuejs.org/) - 渐进式前端框架
- [Ant Design Vue](https://antdv.com/) - 企业级 UI 组件库
- [DuckDB](https://duckdb.org/) - 嵌入式分析型数据库
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架

## 许可证

MIT License
