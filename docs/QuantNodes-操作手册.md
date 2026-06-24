# QuantNodes 操作手册

> AI-native 量化研究框架 - 完整的项目核心内容梳理与操作指南

---

## 1. 项目概述

### 1.1 项目定位

**QuantNodes** 是一个面向量化研究的节点架构平台，通过统一的 **BaseNode + Pipeline** 模式，实现因子计算、回测分析、数据库查询的无缝集成。核心设计理念是"万物皆为节点，Pipeline 是唯一的组合原语"。

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| **统一节点架构** | 所有功能模块继承自 BaseNode，通过 Pipeline 串联 |
| **317+ 内置算子** | 涵盖时间序列、截面运算、多截面聚合、TA-Lib 技术指标 |
| **多数据库支持** | ClickHouse、DuckDB、MySQL、SQLite、CSV、Parquet |
| **AI 原生设计** | 内置策略生成器和 Pipeline 优化器，支持 LLM 集成 |
| **Config-Driven** | YAML 配置文件驱动回测，支持算子扩展 |
| **Vue 3 前端** | Ant Design Vue 4.x + TypeScript + Vite |

### 1.3 版本信息

- **当前版本**: v3.0.0 (2026-06-23)
- **Python 版本**: 3.11+（v3.0.0 起，上游 nanobot 要求）
- **许可证**: MIT

---

## 2. 技术栈

| 层级 | 技术 | 版本/说明 |
|------|------|-----------|
| **前端框架** | Vue 3 | 组合式 API |
| **前端 UI** | Ant Design Vue | 4.x |
| **前端语言** | TypeScript | 严格类型 |
| **前端构建** | Vite | 开发服务器 |
| **后端框架** | FastAPI | ASGI |
| **后端服务器** | Uvicorn | ASGI 服务器 |
| **Python** | Python | 3.11+ |
| **数据处理** | Polars | 高性能 DataFrame |
| **数据处理** | Pandas | 兼容支持（3.0：`applymap` 移除→`.map`、`df.values` 单 dtype 只读） |
| **数据处理** | NumPy | 数值计算 |
| **数据库** | ClickHouse | 列式数据库 |
| **数据库** | DuckDB | 嵌入式分析数据库 |
| **数据库** | MySQL | 关系型数据库 |
| **数据库** | SQLite | 轻量级数据库 |
| **测试框架** | pytest | 单元/集成测试 |
| **测试扩展** | pytest-asyncio | 异步测试 |
| **测试覆盖** | pytest-cov | 覆盖率报告 |
| **代码检查** | ruff | Python linter |
| **类型检查** | mypy | 类型标注检查 |

---

## 3. 目录结构

```
QuantNodes/
├── QuantNodes/                    # 主包 (Python)
│   ├── __init__.py               # 包入口
│   │
│   ├── core/                     # 核心架构
│   │   ├── __init__.py
│   │   ├── node.py               # BaseNode, Pipeline 实现
│   │   ├── control.py            # 控制流节点 (IfNode, MapNode, WhileNode)
│   │   ├── expression.py          # 表达式系统 (Expression, LambdaExpression)
│   │   ├── serialization.py      # 序列化/反序列化支持
│   │   ├── data_preprocessing.py # 数据预处理功能
│   │   └── pandas_utils.py       # Pandas 工具函数
│   │
│   ├── factor_node/              # 因子引擎模块
│   │   ├── __init__.py
│   │   ├── factor.py             # Factor, DataFactor, Factorize 类
│   │   ├── factor_functions/     # 算子实现
│   │   │   ├── __init__.py       # 算子注册表 API
│   │   │   ├── _helpers.py       # 辅助函数与注册表
│   │   │   ├── _helpers_debug.py # 调试辅助
│   │   │   ├── time_ops.py       # 时间序列算子
│   │   │   ├── section_ops.py    # 截面算子
│   │   │   ├── math_ops.py       # 数学算子
│   │   │   ├── composite_ops.py  # 组合算子
│   │   │   └── talib_ops.py      # TA-Lib 技术指标
│   │   ├── factor_operation.py   # 运算操作类
│   │   ├── factor_db.py          # 因子数据库
│   │   ├── factor_table.py       # 因子表实现
│   │   ├── factor_functions_v2.py # 算子 v2 (实验)
│   │   └── _deprecated.py        # 已废弃代码
│   │
│   ├── database_node/            # 数据库节点模块
│   │   ├── __init__.py
│   │   ├── base.py               # BaseDBNode 抽象基类
│   │   ├── clickhouse_node.py    # ClickHouse 实现
│   │   ├── duckdb_node.py        # DuckDB 实现
│   │   ├── mysql_node.py         # MySQL 实现
│   │   ├── sqlite_node.py        # SQLite 实现
│   │   ├── csv_node.py           # CSV 文件读取
│   │   └── parquet_node.py      # Parquet 文件读取
│   │
│   ├── backtest/                 # 回测引擎模块
│   │   ├── __init__.py
│   │   ├── backtest_node.py      # 回测主节点
│   │   ├── strategy_node.py     # 策略节点
│   │   ├── broker_node.py       # 券商/经纪商节点
│   │   └── recorder_node.py     # 记录器节点
│   │
│   ├── symbolic/                 # 符号计算引擎
│   │   ├── __init__.py
│   │   ├── expression.py        # SQL 表达式 AST
│   │   ├── compiler.py          # SQL 编译器
│   │   ├── optimizer.py         # 查询优化器
│   │   ├── dialect.py           # 数据库方言
│   │   └── utils.py             # 工具函数
│   │
│   ├── operators/               # 自定义算子模块
│   │   ├── __init__.py
│   │   ├── registry.py          # 自定义算子注册表
│   │   ├── proxy.py             # 算子代理/级联查询
│   │   ├── builder.py           # Builder 模式实现
│   │   └── custom.py            # CustomOperator 类
│   │
│   ├── agent/                   # Agent 系统模块
│   │   ├── __init__.py
│   │   ├── config/              # 配置加载
│   │   │   ├── __init__.py
│   │   │   ├── loader.py        # 配置加载器
│   │   │   ├── types.py         # 配置类型定义
│   │   │   └── executor.py      # 配置执行器
│   │   ├── core/                # 核心逻辑
│   │   │   ├── __init__.py
│   │   │   ├── loop.py          # Agent 循环
│   │   │   ├── memory.py        # 内存管理
│   │   │   ├── autocompact.py   # 自动压缩
│   │   │   └── events.py       # 事件系统
│   │   ├── bus/                 # 消息总线
│   │   │   ├── __init__.py
│   │   │   └── events.py       # 事件定义
│   │   └── templates/           # 提示词模板
│   │       ├── agent/
│   │       │   ├── system_prompt.md
│   │       │   ├── identity.md
│   │       │   └── tools_description.md
│   │
│   ├── research/                # 研究模块
│   │   ├── __init__.py
│   │   ├── wiki.py              # Wiki 知识库
│   │   └── README.md
│   │
│   ├── monitor/                 # 监控模块
│   │   ├── __init__.py
│   │   ├── scheduler/            # 调度器
│   │   │   ├── __init__.py
│   │   │   ├── runner.py        # 运行器
│   │   │   └── scheduler.py     # 调度器
│   │   └── storage/             # 存储
│   │
│   ├── cache_node/             # 缓存节点
│   │   ├── __init__.py
│   │   └── base.py             # 缓存基类
│   │
│   ├── ui_node/                 # UI 数据准备节点
│   │   ├── __init__.py
│   │   └── display_node.py     # 显示节点
│   │
│   ├── operator_node/           # SQL 构建节点
│   │   ├── __init__.py
│   │   └── sql_builder.py      # SQL 构建器
│   │
│   ├── ai/                      # AI 模块
│   │   ├── __init__.py
│   │   ├── prompts/             # 提示词模板
│   │   │   ├── __init__.py
│   │   │   └── templates.py    # 模板定义
│   │   └── generator/           # 生成器
│   │
│   ├── conf_node/               # 配置节点
│   │   ├── __init__.py
│   │   └── config_node.py      # 配置节点
│   │
│   ├── deprecated/              # 已废弃模块
│   │   ├── __init__.py
│   │   └── factor_node/        # 旧因子节点
│   │
│   └── test/                    # 内部测试
│
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── main.ts              # 入口文件
│   │   ├── App.vue              # 根组件
│   │   ├── components/          # 通用组件
│   │   │   ├── NodeGraph.vue    # 节点图
│   │   │   ├── PipelineEditor.vue # 管道编辑器
│   │   │   └── ...
│   │   ├── views/               # 页面视图
│   │   │   ├── HomeView.vue     # 首页
│   │   │   ├── FactorView.vue   # 因子页面
│   │   │   ├── BacktestView.vue # 回测页面
│   │   │   └── ...
│   │   ├── stores/              # Pinia 状态管理
│   │   ├── api/                # API 调用封装
│   │   ├── router/             # Vue Router 配置
│   │   ├── types/              # TypeScript 类型
│   │   └── utils/              # 工具函数
│   ├── public/                 # 静态资源
│   ├── index.html              # HTML 入口
│   ├── package.json            # NPM 依赖
│   ├── tsconfig.json           # TS 配置
│   ├── vite.config.ts          # Vite 配置
│   └── .env.development        # 环境变量
│
├── api/                         # FastAPI 后端
│   ├── __init__.py
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置
│   ├── deps.py                 # 依赖注入
│   ├── requirements.txt         # Python 依赖
│   ├── routers/                # API 路由
│   │   ├── __init__.py
│   │   ├── agent.py            # Agent 路由
│   │   ├── backtest.py         # 回测路由
│   │   ├── factor.py           # 因子路由
│   │   ├── wiki.py             # Wiki 路由
│   │   ├── strategy.py         # 策略路由
│   │   ├── dream.py            # 研究路由
│   │   ├── settings.py         # 设置路由
│   │   ├── stats.py            # 统计路由
│   │   └── skill.py            # 技能路由
│   ├── schemas/                # Pydantic 模型
│   ├── services/               # 服务层
│   │   ├── wiki_service.py
│   │   └── ...
│   └── websocket/             # WebSocket
│
├── tests/                       # 测试套件
│   ├── conftest.py             # pytest 配置
│   ├── test_*.py               # 各类测试
│   └── coverage/               # 覆盖率报告
│
├── examples/                    # 示例代码
│   └── 01_quick_start.py       # 快速开始示例
│
├── docs/                        # 设计文档
│   ├── QuantNodes-操作手册.md   # 本文档
│   ├── 04-架构设计.md
│   ├── 22-算子系统设计与规范.md
│   └── ...
│
├── docker/                      # Docker 配置
├── docker-compose.yml           # 容器编排
├── pyproject.toml              # 项目配置
├── Makefile                    # 构建命令
├── conn.ini                     # 数据库连接配置
├── conn.ini.template           # 配置模板
├── README.md                   # 项目 README
└── AGENTS.md                   # Agent 说明
```

---

## 4. 核心概念详解

### 4.1 BaseNode - 节点基类

所有功能模块都继承自 `BaseNode`，它定义了节点的统一接口：

```python
from QuantNodes.core.node import BaseNode
from typing import Any, Optional

class MyNode(BaseNode):
    """自定义节点示例"""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        super().__init__(name)
        self.config = config or {}
    
    def execute(self, data: Any = None) -> Any:
        """执行节点逻辑
        
        Args:
            data: 上游节点传入的数据
            
        Returns:
            处理后的数据，传递给下游节点
        """
        # 业务逻辑
        result = self.process(data)
        
        # 触发下游节点（如果使用 Pipeline）
        if self._next_node:
            return self._next_node.execute(result)
        
        return result
    
    def process(self, data: Any) -> Any:
        """具体的处理逻辑，子类实现"""
        return data
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        return True
    
    def reset(self) -> None:
        """重置节点状态"""
        pass
```

**BaseNode 核心属性**：

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | str | 节点名称 |
| `_next_node` | Optional[BaseNode] | 下一个节点（Pipeline 链式） |
| `_prev_node` | Optional[BaseNode] | 上一个节点 |
| `_config` | dict | 节点配置 |

**BaseNode 核心方法**：

| 方法 | 描述 |
|------|------|
| `execute(data)` | 执行节点逻辑 |
| `reset()` | 重置状态 |
| `validate_config()` | 验证配置 |
| `__rshift__(other)` | 实现 `>>` 运算符 |

### 4.2 Pipeline - 管道

Pipeline 是唯一的组合原语，使用 `>>` 运算符串联节点：

```python
from QuantNodes.core.node import BaseNode, Pipeline

# 方式1: 使用 >> 运算符
pipeline = node_a >> node_b >> node_c >> node_d
result = pipeline.execute(input_data)

# 方式2: 使用 Pipeline 类
pipeline = Pipeline(node_a, node_b, node_c)
result = pipeline.execute(input_data)

# 方式3: 动态构建
pipeline = node_a
pipeline = pipeline >> node_b
pipeline = pipeline >> node_c
```

### 4.3 控制流节点

除了数据处理节点，还有控制流节点：

```python
from QuantNodes.core.control import IfNode, MapNode, WhileNode

# 条件分支
if_node = IfNode(
    condition=lambda x: x["value"] > 0,
    true_node=positive_node,
    false_node=negative_node
)

# 映射节点
map_node = MapNode(
    func=lambda x: x * 2,
    source=source_node
)

# 循环节点
while_node = WhileNode(
    condition=lambda x: x["count"] < 10,
    body=loop_body_node,
    initial={"count": 0}
)
```

### 4.4 因子算子注册表

317+ 内置算子，通过注册表统一管理：

```python
from QuantNodes.factor_node.factor_functions import (
    list_operators,       # 列出所有算子
    get_operator,         # 获取算子函数
    operator_info,        # 获取算子详情
    generate_documentation, # 生成文档
    register_operator,    # 注册新算子
    OperatorCategory,     # 算子类别枚举
)

# 列出所有算子
all_ops = list_operators()
print(f"共 {len(all_ops)} 个算子")

# 按类别列出
point_ops = list_operators('point')      # 单点运算
time_ops = list_operators('time')       # 时间序列
section_ops = list_operators('section') # 截面运算
multi_ops = list_operators('multi_section')  # 多截面
talib_ops = list_operators('talib')     # TA-Lib

# 获取算子函数
rolling_mean = get_operator('rolling_mean')
result = rolling_mean(factor_data, window=20)

# 获取算子详情
info = operator_info('rolling_mean', 'time')
print(info)
# {'name': 'rolling_mean', 'category': 'time', 'doc': '...', 'parameters': [...]}

# 生成 Markdown 文档
docs = generate_documentation('markdown')
```

---

## 5. 快速开始

### 5.1 环境准备

```bash
# 1. 克隆项目
git clone <repo-url>
cd QuantNodes

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装项目依赖
pip install -e .
# Python 依赖会自动安装
# 首次 import 时前端依赖会自动安装

# 可选：agent / MCP 能力（v3.0.0 起 nanobot-ai 为可选依赖）
pip install -e '.[agent]'   # + nanobot agent / WebUI / 多 channel
pip install -e '.[all]'     # agent + mcp 一键装齐

# 4. 复制数据库连接配置（可选，已有模板）
cp conn.ini.template conn.ini
# 编辑 conn.ini 填入实际数据库连接信息
```

### 5.2 启动服务

```bash
# 方式1: 使用 CLI（推荐，首次需先运行 init）
quantnodes run                    # 启动全部服务
quantnodes run --api-only         # 仅启动 API
quantnodes run --frontend-only    # 仅启动前端
quantnodes run --daemon           # 后台运行
quantnodes run --port 8080        # 指定前端端口

# 方式2: 使用 Makefile
make dev                    # 同时启动前端和 API
make dev-api                # 启动 API 服务
make dev-frontend           # 启动前端服务

# 方式3: 手动启动
# 终端1: 启动后端
python -m uvicorn api.main:app --reload --port 8000

# 终端2: 启动前端
cd frontend && npm run dev
```

### 5.2.1 初始化项目（首次使用）

```bash
# 初始化配置文件（.env 和 conn.ini）
quantnodes init

# 查看帮助
quantnodes help

# 查看版本
quantnodes version
```

### 5.3 访问服务

- **前端**: http://localhost:5173
- **API 文档**: http://localhost:8000/docs
- **API 根路径**: http://localhost:8000

### 5.4 运行测试

> 全量测试需先装系统级/可选依赖：`pip install ta-lib tables plotly`（`ta-lib` 需先装 TA-Lib C 库）。缺失时相关测试会优雅降级或跳过。基线（Python 3.11 + pandas 3.0）：非 agent `5163 passed / 21 skipped / 0 failed`，`tests/agent` `574 passed / 13 skipped`。详见 [可选依赖安装指南](15-可选依赖安装指南.md)。

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_factor_functions.py -v
pytest tests/test_database_node.py -v

# 运行特定测试
pytest tests/test_factor_functions.py::test_rolling_mean -v

# 查看覆盖率
pytest tests/ --cov=QuantNodes --cov-report=html
# 报告生成在 htmlcov/index.html
```

---

## 6. 算子系统详解

### 6.1 算子分类详细说明

| 类别 | 数量 | 文件 | 功能描述 |
|------|------|------|----------|
| **Point** | 46 | `math_ops.py` | 单点运算，如 abs, log, sign, isnull |
| **Time** | 65 | `time_ops.py` | 时间序列运算，如 rolling_mean, ewm, diff |
| **Section** | 17 | `section_ops.py` | 截面运算，如 standardizeZScore, winsorize |
| **Multi-Section** | 15 | `composite_ops.py` | 多截面聚合，如 aggregate, disaggregate |
| **TA-Lib** | 174 | `talib_ops.py` | 技术指标，如 RSI, MACD, Bollinger |

### 6.2 算子使用示例

```python
import polars as pl
from QuantNodes.factor_node.factor_functions import (
    rolling_mean, rolling_std, zscore, 
    group_zscore, winsorize
)

# 假设 df 是 Polars DataFrame
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "code": ["000001", "000001", "000001"],
    "close": [100.0, 102.0, 101.5],
    "volume": [1000, 1100, 950]
})

# 1. 滚动均值 (时间序列)
result = rolling_mean(df["close"], window=5)

# 2. 滚动标准差
std = rolling_std(df["close"], window=20)

# 3. Z-Score 标准化
z = zscore(df["close"])

# 4. 分组 Z-Score（按 code 分组）
group_z = group_zscore(df, "close", "code")

# 5. 缩尾处理
winsored = winsorize(df["close"], limits=(0.05, 0.05))
```

### 6.3 自定义算子

```python
from QuantNodes.operators import CustomOperator
from typing import Union
from polars import Expr

# 方式1: 装饰器风格（直接注册）
@CustomOperator.point("my_factor")
def my_factor(f: Union[Expr, str], multiplier: float = 2.0) -> Expr:
    """自定义因子：乘以系数
    
    Args:
        f: 表达式或列名
        multiplier: 乘数
        
    Returns:
        转换后的表达式
    """
    from QuantNodes.factor_node.factor_functions._helpers import _ensure_expr
    e = _ensure_expr(f)
    return e * multiplier

# 使用自定义算子
result = my_factor(df["close"], multiplier=3.0)

# 方式2: Builder 链式风格
my_ewm = (
    CustomOperator.time("my_ewm_30")
    .param("span", int, 30, "指数移动平均窗口")
    .param("min_periods", int, 1, "最小观测数")
    .execute(lambda s, span, min_periods: s.ewm_mean(span=span, min_periods=min_periods))
    .register()
)

# 使用
result = my_ewm(df["close"], span=30)

# 方式3: 模板工厂（基于内置算子创建）
my_ewm = CustomOperator.time_from(
    "my_custom_ewm",    # 新算子名
    "ewm_mean",         # 基础算子
    span=30             # 固定参数
)

# 使用
result = my_custom_ewm(df["close"])

# 方式4: 级联查询（先查自定义，再查内置）
from QuantNodes.factor_node.factor_functions import get_operator
op = get_operator("my_factor")  # 返回自定义算子
op = get_operator("rolling_mean")  # 返回内置算子
```

### 6.4 算子注册装饰器

在因子函数模块中直接注册：

```python
from QuantNodes.factor_node.factor_functions._helpers import register_operator, OperatorCategory

@register_operator("my_operator", OperatorCategory.TIME)
def my_operator(
    f,  # Polars Expr 或列名
    window: int = 20,  # 窗口大小
    min_periods: int = None  # 最小观测数
):
    """算子描述
    
    Args:
        f: 输入表达式
        window: 窗口大小
        min_periods: 最小观测数
        
    Returns:
        Polars Expr
    """
    from ._helpers import _ensure_expr
    e = _ensure_expr(f)
    # 实现逻辑
    return e.rolling_mean(window)
```

---

## 7. 数据库节点详解

### 7.1 BaseDBNode 抽象基类

所有数据库节点继承自 `BaseDBNode`：

```python
from QuantNodes.database_node.base import BaseDBNode
import pandas as pd
from typing import Optional, List

class MyDBNode(BaseDBNode):
    """自定义数据库节点"""
    
    def connect(self):
        """建立连接"""
        # 实现连接逻辑
        self._connection = ...
        return self
    
    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """执行查询"""
        # 实现查询逻辑
        return pd.DataFrame()
    
    def execute(self, sql: str, params: Optional[tuple] = None) -> int:
        """执行 DDL/DML"""
        # 实现执行逻辑
        return 0
    
    def disconnect(self):
        """关闭连接"""
        self._connection = None
    
    def health_check(self) -> bool:
        """健康检查"""
        return True
    
    def show_tables(self) -> List[str]:
        """列出所有表"""
        return []
```

### 7.2 各数据库节点使用

```python
# ===== ClickHouse =====
from QuantNodes.database_node import ClickHouseNode

ch = ClickHouseNode(
    host="localhost",
    port=8123,
    user="default",
    passwd="",
    database="default",
    interface="http"  # 或 "native"
)
ch.connect()
df = ch.query("SELECT * FROM stock_data LIMIT 100")
ch.disconnect()

# ===== DuckDB =====
from QuantNodes.database_node import DuckDBNode

db = DuckDBNode(
    path=":memory:"  # 或 "data.duckdb"
)
db.connect()
df = db.query("SELECT * FROM parquet_scan('data.parquet')")
db.disconnect()

# ===== MySQL =====
from QuantNodes.database_node import MySQLNode

mysql = MySQLNode(
    host="localhost",
    port=3306,
    user="root",
    password="",
    database="test"
)
mysql.connect()
df = mysql.query("SELECT * FROM users LIMIT 10")
mysql.disconnect()

# ===== SQLite =====
from QuantNodes.database_node import SQLiteNode

sqlite = SQLiteNode(
    path="data.db"
)
sqlite.connect()
df = sqlite.query("SELECT * FROM trades")
sqlite.disconnect()

# ===== CSV =====
from QuantNodes.database_node import CSVNode

csv_node = CSVNode(
    path="/path/to/data.csv",
    sep=",",
    encoding="utf-8",
    sql="SELECT * FROM data WHERE date > '2024-01-01'"  # 可选 SQL 预处理
)
csv_node.connect()
df = csv_node.query()
csv_node.disconnect()

# ===== Parquet =====
from QuantNodes.database_node import ParquetNode

pq = ParquetNode(
    path="/path/to/data.parquet",
    sql="SELECT * FROM parquet_scan('data.parquet') WHERE code = '000001'"
)
pq.connect()
df = pq.query()
pq.disconnect()
```

### 7.3 节点通用方法

| 方法 | 描述 | 返回类型 |
|------|------|----------|
| `connect()` | 建立连接 | self |
| `query(sql)` | 执行查询 | pd.DataFrame |
| `execute(sql)` | 执行 DDL/DML | int (影响行数) |
| `disconnect()` | 关闭连接 | None |
| `health_check()` | 健康检查 | bool |
| `show_tables()` | 列出表 | List[str] |
| `show_databases()` | 列出数据库 | List[str] |

---

## 8. 回测系统

### 8.1 回测组件

```python
from QuantNodes.backtest import BacktestNode, StrategyNode, BrokerNode

# 创建回测节点
backtest = BacktestNode(
    initial_capital=1000000,  # 初始资金
    commission=0.0003,        # 手续费率
    slippage=0.0001,          # 滑点
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 创建策略节点
strategy = StrategyNode(
    name="ma_cross",
    params={"fast_ma": 5, "slow_ma": 20}
)

# 创建券商节点
broker = BrokerNode(
    name="simulated",
    initial_cash=1000000
)

# 构建回测 Pipeline
pipeline = data_source >> strategy >> broker >> backtest

# 执行回测
result = pipeline.execute()
```

### 8.2 回测结果

```python
# 回测结果包含
result = {
    "summary": {
        "total_return": 0.15,        # 总收益率
        "annual_return": 0.18,       # 年化收益率
        "sharpe_ratio": 1.5,         # 夏普比率
        "max_drawdown": 0.08,        # 最大回撤
        "win_rate": 0.55,            # 胜率
    },
    "trades": [...],                 # 交易记录
    "positions": [...],              # 持仓记录
    "equity_curve": [...],           # 权益曲线
}
```

---

## 9. Agent 系统

### 9.1 Agent 架构

```
Agent System
├── 配置层 (config/)
│   ├── loader.py      - YAML 配置加载
│   ├── types.py      - 配置类型定义
│   └── executor.py   - 配置执行器
├── 核心层 (core/)
│   ├── loop.py       - 主循环
│   ├── memory.py     - 记忆管理
│   └── autocompact.py - 自动压缩
├── 消息总线 (bus/)
│   └── events.py     - 事件定义
└── 提示词模板 (templates/)
    ├── system_prompt.md
    ├── identity.md
    └── tools_description.md
```

### 9.2 使用 Agent

```python
from QuantNodes.agent import AgentLoop, MemoryStore

# 初始化 Agent
agent = AgentLoop(
    config_path="agent_config.yaml",
    memory=MemoryStore(max_history=1000)
)

# 运行 Agent
result = agent.run(user_input="帮我分析最近一周的收益情况")

# 查看记忆
history = agent.memory.get_history()
```

---

## 10. 前端详解

### 10.1 前端技术栈

| 技术 | 用途 |
|------|------|
| Vue 3 | 框架 (组合式 API) |
| Ant Design Vue 4.x | UI 组件库 |
| TypeScript | 类型安全 |
| Vite | 开发服务器/构建 |
| Pinia | 状态管理 |
| Vue Router | 路由管理 |
| Axios | HTTP 请求 |

### 10.2 前端目录结构

```
frontend/src/
├── main.ts              # 入口
├── App.vue              # 根组件
├── api/                 # API 调用
│   ├── index.ts         # axios 实例
│   ├── factor.ts        # 因子 API
│   ├── backtest.ts      # 回测 API
│   └── ...
├── components/          # 通用组件
│   ├── NodeGraph.vue    # 节点图可视化
│   ├── PipelineEditor.vue # 管道编辑器
│   ├── DataTable.vue    # 数据表格
│   └── ...
├── views/               # 页面
│   ├── HomeView.vue     # 首页
│   ├── FactorView.vue   # 因子研究
│   ├── BacktestView.vue # 回测
│   ├── DataView.vue     # 数据管理
│   └── ...
├── stores/              # Pinia stores
│   ├── useFactorStore.ts
│   ├── useBacktestStore.ts
│   └── ...
├── router/
│   └── index.ts         # 路由配置
├── types/               # TypeScript 类型
│   └── ...
└── utils/
    └── ...              # 工具函数
```

### 10.3 前端开发命令

```bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build

# 类型检查
npx vue-tsc --noEmit

# 代码检查
npm run lint

# 预览生产版本
npm run preview
```

---

## 11. API 接口详解

### 11.1 主要路由

| 路由 | 方法 | 描述 |
|------|------|------|
| `/api/agent/execute` | POST | 执行 Agent 命令 |
| `/api/backtest/run` | POST | 运行回测 |
| `/api/backtest/result/{id}` | GET | 获取回测结果 |
| `/api/factor/query` | POST | 查询因子 |
| `/api/factor/list` | GET | 列出因子 |
| `/api/wiki/search` | GET | 搜索 Wiki |
| `/api/wiki/write` | POST | 写入 Wiki |
| `/api/strategy/list` | GET | 列出策略 |
| `/api/strategy/save` | POST | 保存策略 |

### 11.2 API 使用示例

```python
import requests

# 查询因子
response = requests.post(
    "http://localhost:8000/api/factor/query",
    json={
        "factor_name": "close",
        "codes": ["000001", "000002"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
)
data = response.json()

# 运行回测
response = requests.post(
    "http://localhost:8000/api/backtest/run",
    json={
        "strategy": "ma_cross",
        "params": {"fast_ma": 5, "slow_ma": 20},
        "capital": 1000000
    }
)
task_id = response.json()["task_id"]
```

---

## 12. 配置与运行

### 12.1 环境变量

**前端** (`frontend/.env.development`):
```env
VITE_API_BASE=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### 12.2 数据库连接配置

**文件**: `conn.ini`
```ini
[clickhouse]
host = localhost
port = 8123
user = default
password = 
database = default

[duckdb]
path = ./data.duckdb

[mysql]
host = localhost
port = 3306
user = root
password = 
database = quantnodes

[sqlite]
path = ./data.db
```

### 12.3 Make 命令

```bash
make help              # 显示帮助
make dev               # 启动前后端开发服务器
make dev-api           # 启动 API
make dev-frontend      # 启动前端
make build             # 构建前端
make docker-up         # 启动 Docker
make docker-down       # 停止 Docker
make docker-build      # 构建 Docker 镜像
make clean             # 清理
make test              # 运行测试
make lint              # 代码检查
```

---

## 13. 代码质量

### 13.1 代码检查

```bash
# ruff 检查（所有问题）
python3 -m ruff check .

# ruff 检查（统计）
python3 -m ruff check --statistics .

# ruff 自动修复
python3 -m ruff check --fix .

# ruff 仅检查特定类型
python3 -m ruff check --select F401,F841 .

# mypy 类型检查
python3 -m mypy QuantNodes/

# Python 语法检查
python3 -m py_compile QuantNodes/
```

### 13.2 编译验证

```bash
# Python 编译检查
python3 -m py_compile QuantNodes/

# Vue 类型检查（前端）
cd frontend && npx vue-tsc --noEmit
```

---

## 14. 测试

### 14.1 测试命令

> 全量测试需 Python 3.11+ 并装齐 `pip install ta-lib tables plotly`（`ta-lib` 需先装 TA-Lib C 库）。基线：非 agent `5163 passed / 21 skipped / 0 failed`，`tests/agent` `574 passed / 13 skipped`（顺序 + 并行均通过）。详见 [可选依赖安装指南](15-可选依赖安装指南.md)。

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定文件
pytest tests/test_factor_functions.py -v

# 运行特定测试
pytest tests/test_factor_functions.py::test_rolling_mean -v

# 运行带覆盖率
pytest tests/ --cov=QuantNodes --cov-report=html

# 运行异步测试
pytest tests/ -v --asyncio-mode=auto
```

### 14.2 测试配置

**pyproject.toml** 中的 pytest 配置：
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --strict-markers --ignore=QuantNodes"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
]
```

---

## 15. 项目状态

### 15.1 完成度

| 模块 | 状态 | 说明 |
|------|------|------|
| 核心架构 | ✅ 完成 | BaseNode, Pipeline, Control |
| 因子引擎 | ✅ 完成 | 317+ 算子 |
| 数据库节点 | ✅ 完成 | 6 种数据库 |
| 回测引擎 | ✅ 完成 | 完整回测流程 |
| Agent 系统 | ✅ 完成 | 上游 nanobot 0.2.1（可选依赖） |
| 前端 | ✅ 完成 | Vue 3 + Ant Design |
| API | ✅ 完成 | FastAPI |
| 测试 | ✅ 完成 | 非 agent 5163 + agent 574，0 failed |

### 15.2 已知问题

1. `_fix_v2.py` - 存在语法错误（废弃文件，建议删除）
2. `deprecated/` 目录 - 包含旧代码
3. 部分 ruff 警告 - 主要是测试文件和 deprecated 文件

---

## 16. 常见问题

### Q1: 如何添加新的因子算子？

在 `QuantNodes/factor_node/factor_functions/` 对应模块中添加函数，使用 `@register_operator` 装饰器注册。

### Q2: 如何连接新的数据库？

继承 `BaseDBNode`，实现 `connect()`, `query()`, `execute()`, `disconnect()` 方法。

### Q3: 如何运行特定测试？

```bash
pytest tests/test_factor_functions.py::test_rolling_mean -v
```

### Q4: 前端启动失败？

前端依赖在首次 import QuantNodes 时自动安装。如需手动安装，请运行：
```bash
cd frontend && npm install
```

### Q5: API 返回 404？

检查后端是否启动，端口是否正确（默认 8000）

### Q6: 数据库连接失败？

检查 `conn.ini` 配置是否正确，确保数据库服务已启动

---

## 17. 相关文档

| 文档 | 路径 | 描述 |
|------|------|------|
| README | `README.md` | 项目概述 |
| 架构设计 | `docs/04-架构设计.md` | 系统架构 |
| 算子规范 | `docs/22-算子系统设计与规范.md` | 算子设计规范 |
| 开发测试规范 | `docs/大型项目开发测试规范.md` | 测试规范 |
| 本文档 | `docs/QuantNodes-操作手册.md` | 操作指南 |