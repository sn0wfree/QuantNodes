# QuantNodes

> 量化研究节点架构 - 基于 Pipeline 组合原语的统一量化分析平台

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目简介

QuantNodes 是一个面向量化研究的节点架构平台，通过统一的 **BaseNode + Pipeline** 模式，实现因子计算、回测分析、数据库查询的无缝集成。

### 核心特性

- **统一节点架构**: 万物皆 Node，Pipeline 是唯一组合原语
- **317+ 内置算子**: 涵盖时间序列、截面运算、多截面聚合等，装饰器注册表模式
- **多数据库支持**: ClickHouse、DuckDB、MySQL、SQLite、CSV、Parquet
- **Config-Driven 回测**: YAML 配置文件驱动回测，支持算子扩展
- **可选 LLM Agent**（v3.0.0）: 直接消费上游 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 0.2.1，作为 `[agent]` 可选依赖——装则获得 WebUI / WebSocket / cron / subagent / 多 channel，不装则纯量化库完全可用
- **MCP server**（可选）: 把量化能力暴露为 MCP tools（stdio + HTTP）
- **方法库 + 提示词库**: 纯 Python 方法 + 内置策略提示词，亦可经 REST API 供外部 Agent 调用
- **自动化因子挖掘（v2.9.0, QuantAlpha）**: 5 智能体编排（idea-generator → formula-translator → evaluator → reflector → critic），基于 nanobot spawn + OperatorVocab 162 算子 + PolarsAlphaCalculator（AlphaGen 兼容 7 方法）。CLI `quantnodes alpha-gpt` + REST API 5 端点 + WebSocket 实时流 + Vue 3 前端可视化。详见 [`docs/quant_alpha/alpha_gpt_user_guide.md`](docs/quant_alpha/alpha_gpt_user_guide.md)
- **Table 4 复现 v2.10.0-mock.1（QuantAlpha Stage 1 mock）**: 论文 Table 4 端到端 mock 复现。3 baseline (G1 手工 / G2 LLM-Only / G3 Alpha-GPT) + 统一接口契约 + 端到端 mock pipeline。11 文件 + 8 测试（114 PASSED，97% 覆盖 evaluation/ 子包）。CLI `python3.11 scripts/reproduce_table4_mock.py --quick`。详见 [`docs/quant_alpha/table4_reproduction.md`](docs/quant_alpha/table4_reproduction.md) + [`docs/quant_alpha/stage2_data_requirements.md`](docs/quant_alpha/stage2_data_requirements.md)

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
├── archive/                          # 统一归档目录
│   ├── QuantNodes/                   # QuantNodes 包归档（v2.x 旧代码）
│   │   └── test/                     # 测试数据归档
│   ├── frontend/                    # 前端归档
│   │   └── src/archive/             # 前端 Chat UI 归档
│   ├── api/                         # API 归档
│   │   └── archive/                 # API 归档
│   └── docs/                        # 文档归档
│       └── archived/                # 历史文档
├── QuantNodes/                      # 主包
│   ├── agent/                       # Agent 系统（v3.0.0: 上游 nanobot + 量化工具）
│   │   ├── nanobot_bridge.py        # Nanobot.from_config 包装门面
│   │   ├── config_mapper.py         # .env → nanobot_config.json
│   │   ├── tools/                   # 14 个量化工具（继承 nanobot Tool）
│   │   ├── skills_quant/            # 6 个量化 SKILL.md
│   │   └── cron_jobs.py             # 量化专属 cron 调度
│   ├── mcp_server/                  # MCP server（9 个量化 tools）
│   ├── monitor/                     # 监控 agent 工具（策略监控/调度/版本）
│   ├── methods/                     # 方法库（外部 Agent API）
│   ├── prompts/                     # 提示词库
│   ├── factor_node/                 # 因子引擎（317+ 算子）
│   ├── core/                        # 核心架构（BaseNode / Pipeline / 回测）
│   ├── database_node/               # 数据库节点（ClickHouse/DuckDB/MySQL/SQLite/CSV/Parquet）
│   └── ...
├── api/                             # FastAPI 服务器
│   ├── routers/                     # API 路由
│   └── services/                    # 服务层（含 nanobot_runtime.py 单进程包装器）
├── frontend/src/                    # Vue 3 + Ant Design 前端
│   └── views/
├── tests/                           # 测试套件（5163+ 非 agent / 574 agent）
├── examples/                        # 示例代码
├── docs/                           # 设计文档
├── .agent/                          # nanobot workspace（gitignore，含 API key）
├── output/                          # factor_test pipeline 节点产物 (运行时)
└── outputs/                         # 回测引擎结果 (运行时)
```

## 快速开始

### 安装

```bash
pip install -e .
```

可选附加组件:

```bash
pip install -e '.[agent]'          # LLM Agent（上游 nanobot）/ WebUI / 多 channel
pip install -e '.[mcp]'            # MCP server（把量化能力暴露为 MCP tools）
pip install -e '.[all]'            # agent + mcp 一键装齐
pip install -e '.[streamlit-ui]'   # 启用 Streamlit Agent 调试 UI
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
# ── v3.0.0 推荐方式（lifecycle 接口）──
quantnodes serve                     # 前台启动，Ctrl+C 停止
quantnodes serve --check-env        # 启动前校验 API key
quantnodes serve --gateway-port 18090  # 自定义 nanobot gateway 端口
quantnodes serve --frontend         # 同时启动 Vite dev server
quantnodes serve --daemon           # 后台运行，写 .quantnodes.pid
quantnodes stop                     # 停止后台 serve
quantnodes status                   # 综合健康检查
quantnodes logs -f                  # 实时查看日志

# ── 旧接口（兼容保留）──
quantnodes run
quantnodes run --host 0.0.0.0 --port 19380 --gateway-port 18090 --daemon
quantnodes run --api-only
quantnodes run --frontend-only
```

详细使用文档：[`docs/16-quantnodes-cli使用指南.md`](docs/16-quantnodes-cli使用指南.md)

### 外部 Agent API

QuantNodes 通过 REST API 为外部 Agent 提供方法调用和提示词。

v3.0.0 新增内置 agent 端点（需 `[agent]` extra，否则返回 503）：

```bash
# 查询 agent 状态
curl http://localhost:8000/api/agent/status

# 发送 chat 消息
curl -X POST http://localhost:8000/api/agent/chat/send \
  -H "Content-Type: application/json" \
  -d '{"message": "分析最近的动量因子表现"}'
```

#### 获取策略提示词

```bash
curl -X GET "http://localhost:8000/api/prompts/strategy/momentum" \
  -H "X-API-Key: qn_live_xxxxxxxxxxxxxxxxxxxxxxxx"
```

#### 验证代码安全

```bash
curl -X POST "http://localhost:8000/api/code/validate" \
  -H "X-API-Key: qn_live_xxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"code": "import pandas as pd\\nprint(pd.DataFrame())"}'
```

#### 运行回测

```bash
curl -X POST "http://localhost:8000/api/backtest/run" \
  -H "X-API-Key: qn_live_xxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"pipeline_code": "...", "start_date": "2020-01-01", "end_date": "2024-12-31"}'
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

> 要求 Python 3.11+。全量测试需先装系统级/可选依赖：`pip install ta-lib tables plotly`（`ta-lib` 需先装 TA-Lib C 库）。缺失时相关测试会优雅降级或跳过。基线：非 agent `5163 passed / 21 skipped / 0 failed`，`tests/agent` `574 passed / 13 skipped`。详见 [可选依赖安装指南](docs/15-可选依赖安装指南.md)。

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_factor_functions.py -v
pytest tests/test_backtest_tool.py -v

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
- [v2.6.0 实际架构基线](docs/Architecture-v2.6.md)
- [单因子回测节点化整合设计](docs/SingleFactorBacktest-Integration-Design.md)
- [演化框架设计 (FactorFeedback + TrajectoryPool + QualityGate)](docs/Evolution-Framework.md)
  - [FactorFeedback 详细规格](docs/FactorFeedback.md)
  - [TrajectoryPool 详细规格](docs/TrajectoryPool.md)
  - [QualityGate 详细规格](docs/QualityGate.md)

## 变更日志

> 完整变更记录见 [CHANGELOG.md](CHANGELOG.md)。下面仅列里程碑版本。

### v3.0.0 (2026-06-23)

- ✅ **上游 nanobot 迁移**: Agent 核心从自写 runtime 改为直接消费 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 0.2.1（`Nanobot.from_config()` 门面）
- ✅ **nanobot-ai 改为可选依赖**: `pip install 'quantnodes[agent]'`；纯量化库可独立使用
- ✅ **单进程架构**: FastAPI uvicorn + nanobot gateway 共存于同一进程（`api/services/nanobot_runtime.py`）
- ✅ **Phase 5 功能**: subagent 多 Agent 团队 / MCP server（9 tools）/ 单进程 WebUI / WebSocket + 飞书 channel / 量化专属 cron 调度
- ✅ **workspace 迁移**: `.quant_agent/` → `.agent/`（HKUDS nanobot 约定，附迁移脚本）
- ⚙️ **Python 3.11+**（上游最低要求）；**pandas 3.0 兼容**（`applymap`→`.map` 等）
- ✅ **测试稳定化**: 非 agent `5163 passed / 21 skipped / 0 failed`、`tests/agent` `574 passed / 13 skipped`（顺序 + 并行均通过）；可选依赖优雅降级
- 💥 **Breaking**: 删除自写 `agent/core/{loop,runner,memory,...}`，改用 `Agent.run()` / `Nanobot.from_config()` facade

### v2.6.0 (2026-05-14)

- ✅ **架构重构**: 从"带 UI 的 Agent 系统"转向"外部 Agent 方法库 + 提示词库"
- ✅ **移除 Chat UI**: 前端不再有交互式 Chat
- ✅ **移除 Agent LLM**: QuantNodes 不再包含 LLM
- ✅ **新增 methods/**: 纯方法实现，外部 Agent 可通过 API 调用
- ✅ **新增 prompts/**: 完整策略提示词库，含参考代码
- ✅ **API 重构**: 新增 prompts、code 路由，移除 chat 路由
- ✅ **API Key 认证**: 支持 X-API-Key 和 Authorization Bearer 认证
- ✅ **前端展示页面**: Dashboard、Portfolios、Status 视图

### v2.5.0 (2026-05-10)

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

### v2.4.1 (2026-05-09)

- ✅ CLI 优化：依赖安装自动化（Python 依赖通过 pyproject.toml，前端依赖首次 import 时自动安装）
- ✅ CLI 优化：init 流程简化（只初始化 Wiki 和创建配置文件）
- ✅ llmwikify Wiki 集成：初始化时自动创建 raw/, wiki/, wiki.md, index.md, log.md
- ✅ 目录结构优化：Wiki 放在项目根目录，与 data/ 平级

### v2.4.0 (2026-05-08)

- ✅ Feature 3A - WikiFactorProxy：因子库读写接口，支持 llmwikify 解析 PDF
- ✅ Feature 3B - 研报复现：从研报提取因子公式，验证有效性，存入 Wiki 因子库
- ✅ Feature 3C - AutoResearch 自动因子挖掘：模板枚举 + MCTS 搜索自动生成验证因子
- ✅ Feature 3D - 用户友好自定义算子 API：装饰器风格 + Builder 链式 + 模板工厂 + 级联查询
- ✅ 317+ 内置算子（point: 46, time: 65, section: 17, multi_section: 15, talib: 174）
- ✅ 2574+ 测试用例，覆盖率 ~80%
- ✅ 统一前端到 Vue 3 + Ant Design Vue 4.x

### v2.3.0 (2026-04-30)

- ✅ 实现 config-driven 回测的算子扩展机制（通用 fallback + custom_operators 加载）
- ✅ 迁移 6 个独有算子到 factor_functions（mad, weighted_aggr_mean, fill_null_by_strategy 等）
- ✅ 删除 factor_nodes.py，统一算子系统（-1274 行）
- ✅ ts_corr 协方差公式修复 + group_winsorize 实现
- ✅ 算子构建规范文档

### v2.2.0 (2026-04-28)

- ✅ 实现 DataPreprocessingFun 类，完全移除 QuantStudio 依赖
- ✅ 统一 PointOperation/TimeOperation 空数据处理
- ✅ 添加参数边界验证 (winsorize, half_life)
- ✅ 改进 rolling_regress 错误处理
- ✅ 修复 rolling_change_rate 除法顺序问题
- ✅ 添加 pandas_utils 工具模块
- ✅ 添加 Vue 3 前端应用骨架 (后统一为 Vue)

### v2.1.0 (2026-04-27)

- ✅ 完成 9/9 阶段重构
- ✅ 97+ 因子算子实现
- ✅ 完整测试套件
- ✅ 多数据库支持

## 致谢

- [HKUDS/nanobot](https://github.com/HKUDS/nanobot) — Agent 核心运行时（v3.0.0 直接消费上游 0.2.1）
- [QuantaAlpha](https://arxiv.org/abs/2602.07085) — 演化实验框架设计（CoSTEER 反馈 / TrajectoryPool / QualityGate）
- [QuantStudio](https://github.com/quantopian/quantstudio) — 因子计算系统设计灵感
- [TA-Lib](https://ta-lib.org/) — 174 个技术指标算子（C 库 + Python wrapper）

## 许可证

MIT License
