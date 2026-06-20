# QuantNodes 架构改造文档

> 当前状态：已完成
> 创建日期：2025-05-14
> 完成日期：2025-05-14
> 版本：1.2（完成版）

---

## 1. 改造目标

将 QuantNodes 从"带 UI 的 Agent 系统"重构为"外部 Agent 的方法库 + 提示词库 + API 服务器"。

### 核心变化

| 变化 | 说明 |
|------|------|
| **移除 Chat UI** | 前端不再有交互式 Chat |
| **移除 Agent LLM** | QuantNodes 不再调用 LLM |
| **外部 Agent** | opencode/openclaw 等通过 API 调用 QuantNodes |
| **QuantNodes 定位** | 纯执行层 + 提示词提供方 |

---

## 2. 架构决策

### 2.1 决策汇总

| 决策 | 选择 |
|------|------|
| QuantNodes 定位 | 方法库 + 提示词库 + API 服务器（无 LLM） |
| 外部 Agent | 自己带 LLM，通过 API 交互 |
| UI | 纯展示 Dashboard，无 Chat 交互 |
| 策略生成 | 混合方案（简单=提示词，复杂=生成端点） |
| 方法迁移 | sandbox, backtest, pipeline, factor 等迁移到 `methods/` |
| 提示词 | 含参考代码的完整提示词 |
| 认证 | API Key |
| API Key 存储 | `settings.json` 或环境变量 |
| 数据库处理 | Chat 相关表标记为 archive，不删除数据 |

### 2.2 策略生成方案

**混合方案：**

| 场景 | 方案 | 说明 |
|------|------|------|
| 简单策略 | 提示词方案 | 外部 Agent 获取提示词 → 用自己 LLM 生成代码 → QuantNodes 执行 |
| 复杂策略 | 生成端点 | 提供 `/api/strategy/generate` 端点（作为可选的高级功能） |

### 2.3 API Key 认证设计

```python
# API Key 格式
# - 长度：32 字符
# - 格式：qn_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# - 存储：settings.json 或环境变量 QUANTNODES_API_KEY

# 认证方式
# - Header: Authorization: Bearer <api_key>
# - 或 Header: X-API-Key: <api_key>

# 速率限制（初始）
# - 无认证：100 次/小时
# - 有认证：1000 次/小时
```

---

## 3. 实施计划

### Phase 依赖关系

```
Phase 1 ──┬── 依赖：无
          └── 被依赖：Phase 5（需要先清理 UI）

Phase 2 ──┬── 依赖：无
          └── 被依赖：Phase 4（API 调用 methods）

Phase 3 ──┬── 依赖：无（可并行）
          └── 被依赖：Phase 4.3（prompts 端点需要先有提示词）

Phase 4 ──┬── 依赖：Phase 2, 3
          └── 被依赖：Phase 5（前端调用 API）

Phase 5 ──┬── 依赖：Phase 1, 4
          └── 被依赖：Phase 6

Phase 6 ──┬── 依赖：Phase 1-5 全部完成
          └── 被依赖：无
```

---

### Phase 1：代码归档（Chat/UI）

| Step | Task | Source → Dest |
|------|------|---------------|
| 1.1 | 创建归档目录 | `archive/frontend/src/archive/chat/`, `archive/QuantNodes/agent/` |
| 1.2 | 归档 Chat 视图 | `frontend/src/views/AgentChat/` → `archive/frontend/src/archive/chat/AgentChat/` |
| 1.3 | 归档 Chat 组件 | `frontend/src/components/Chat/` → `archive/frontend/src/archive/chat/Chat/` |
| 1.4 | 归档 agent store & API | `frontend/src/stores/agent.ts`, `frontend/src/api/agent.ts` → `archive/frontend/src/archive/` |
| 1.5 | 归档 `QuantNodes/agent/` | `core/loop.py`, `session/`, `providers/`, `skills/` → `archive/QuantNodes/agent/` |
| 1.6 | 更新 import 路径 | 全局搜索 `from ../AgentChat` 等引用，标记为 archive |
| 1.7 | 前端 router 清理 | `router/index.ts` 移除 `/chat`, `/agent` 路由 |

**归档目录结构：**
```
archive/
├── frontend/src/archive/
│   ├── chat/
│   │   ├── AgentChat/           # views/AgentChat/
│   │   ├── Chat/                # components/Chat/
│   │   └── agent/               # stores/agent.ts, api/agent.ts
│   └── ...
├── QuantNodes/agent/
│   ├── core/
│   │   └── loop.py
│   ├── session/
│   ├── providers/
│   ├── skills/
│   └── tools/               # 未迁移的工具
└── ...
```

**前端 router 调整：**
```typescript
// 移除路由
- /chat/:sessionId
- /agent

// 保留路由
- /                       Dashboard
- /strategies             Strategies
- /backtests             Backtests
- /portfolios            Portfolios
- /factors               Factors
- /status                Status
- /build                 Build 模式（如需要）
- /plan                  Plan 模式（如需要）
```

---

### Phase 2：方法迁移（Agent → Methods）

| Step | Task | Source → Dest |
|------|------|---------------|
| 2.1 | 创建 `QuantNodes/methods/` | - |
| 2.2 | 迁移 `backtest.py` | `agent/tools/backtest.py` → `methods/backtest.py` |
| 2.3 | 迁移 `sandbox.py` | `agent/tools/sandbox.py` → `methods/sandbox.py` |
| 2.4 | 迁移 `pipeline.py` | `agent/tools/pipeline.py` → `methods/pipeline.py` |
| 2.5 | 迁移 `factor.py` | `agent/tools/factor.py` → `methods/factor.py` |
| 2.6 | 迁移辅助工具 | `wiki.py`, `file_ops.py`, `code_search.py`, `git_ops.py` |
| 2.7 | 创建 `methods/__init__.py` | 统一导出 |

**目标目录结构：**
```
QuantNodes/methods/
├── __init__.py           # from .backtest import run_backtest; ...
├── backtest.py           # run_backtest(config) → BacktestResult
├── sandbox.py            # validate_code(code) → ValidationResult
├── pipeline.py           # execute_pipeline(config) → PipelineResult
├── factor.py             # analyze_factor(data, method) → FactorResult
├── wiki.py               # query_wiki(topic) → WikiResult
├── file_ops.py           # 文件操作
├── code_search.py        # 代码搜索
└── git_ops.py            # Git 操作
```

**方法签名示例：**
```python
# methods/backtest.py
def run_backtest(
    code: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000.0,
    **kwargs
) → BacktestResult:
    """执行回测"""
    pass

# methods/sandbox.py
def validate_code(code: str) → ValidationResult:
    """验证代码安全性"""
    pass

def execute_code(code: str, **kwargs) → ExecutionResult:
    """执行代码（沙盒环境）"""
    pass
```

---

### Phase 3：提示词库建设（Prompts）

| Step | Task | Content |
|------|------|---------|
| 3.1 | 创建 `QuantNodes/prompts/` | 目录结构 + VERSION 文件 |
| 3.2 | 策略提示词 | momentum, mean_reversion, trend_following, pairs_trading, market_neutral |
| 3.3 | 回测提示词 | standard, factor_based |
| 3.4 | 因子提示词 | ic_analysis, group_backtest, correlation |
| 3.5 | 版本管理 | 每个提示词包含 version, created_at, updated_at |

**VERSION 文件：**
```
PROMpts_VERSION = "1.0.0"
LAST_UPDATED = "2025-05-14"
```

**提示词内容结构：**
```python
# prompts/strategy/momentum.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "momentum"
    description: str = "动量策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于动量策略..."

    @property
    def required_params(self) -> List[str]:
        return ["symbol", "window", "threshold"]

    @property
    def output_format(self) -> str:
        return "python_code"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 500,
            "allowed_imports": ["numpy", "pandas", "talib"],
            "forbidden_patterns": ["os.", "subprocess", "eval"]
        }

    @property
    def example_code(self) -> str:
        return '''import numpy as np
import pandas as pd

def momentum_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """动量策略示例"""
    returns = data['close'].pct_change()
    momentum = returns.rolling(window=window).sum()
    signal = (momentum > threshold).astype(int)
    return signal
'''
```

**目标目录结构：**
```
QuantNodes/prompts/
├── __init__.py              # 统一导出
├── VERSION                  # 版本信息
├── strategy/
│   ├── __init__.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── trend_following.py
│   ├── pairs_trading.py
│   └── market_neutral.py
├── backtest/
│   ├── __init__.py
│   ├── standard.py
│   └── factor_based.py
└── factor/
    ├── __init__.py
    ├── ic_analysis.py
    ├── group_backtest.py
    └── correlation.py
```

---

### Phase 4：API 重构

| Step | Task | Change | 依赖 |
|------|------|--------|------|
| 4.1 | 移除 Chat 路由 | `agent.router` (chat, ws 等) | Phase 1 |
| 4.2 | 保留方法端点 | backtest, factor, pipeline, wiki, stats | 无 |
| 4.3 | 新增 prompts 端点 | `/api/prompts/strategy/{type}` 等 | Phase 3 |
| 4.4 | 新增 strategy_generate 端点 | `/api/strategy/generate` (可选高级功能) | Phase 2, 3 |
| 4.5 | 新增 code 端点 | `/api/code/execute`, `/api/code/validate` | Phase 2 |
| 4.6 | 实现 API Key 认证 | `deps.py` → `verify_api_key()` | 无 |
| 4.7 | 数据库迁移 | 标记 chat 相关表为 archive | Phase 1 |
| 4.8 | WebSocket 处理 | 移除 `/api/ws/chat`，保留其他 WS（如有） | Phase 1 |

**API Key 认证实现（deps.py）：**
```python
# api/deps.py
from fastapi import Header, HTTPException, Security
from typing import Optional

API_KEYS = {
    "qn_live_xxxxxxxxxxxxxxxxxxxxxxxx": {"name": "opencode", "rate_limit": 1000},
    "qn_live_yyyyyyyyyyyyyyyyyyyyyyyyy": {"name": "openclaw", "rate_limit": 1000},
}

async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
) -> dict:
    key = x_api_key or (authorization.replace("Bearer ", "") if authorization else None)
    if not key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return API_KEYS[key]
```

**保留端点：**
| 端点 | 用途 |
|------|------|
| `POST /api/backtest/run` | 执行回测 |
| `GET /api/backtest/history` | 回测历史 |
| `GET /api/backtest/templates` | 回测模板 |
| `POST /api/factor/analyze` | 因子分析 |
| `POST /api/pipeline/run` | Pipeline 执行 |
| `GET /api/strategies` | 策略列表 |
| `GET /api/strategies/{id}` | 策略详情 |
| `GET /api/stats/performance` | 绩效统计 |
| `GET /api/settings` | 设置 |
| `GET /api/health` | 健康检查 |

**新增端点：**
| 端点 | 用途 |
|------|------|
| `GET /api/prompts/strategy/{type}` | 获取策略提示词 |
| `GET /api/prompts/backtest/{type}` | 获取回测提示词 |
| `GET /api/prompts/factor/{type}` | 获取因子提示词 |
| `POST /api/strategy/generate` | 生成策略（可选高级功能，需 LLM 配置） |
| `POST /api/code/validate` | 验证代码安全 |
| `POST /api/code/execute` | 执行代码 |
| `POST /api/strategies` | 保存外部生成的策略 |

**移除端点：**
| 端点 | 原因 |
|------|------|
| `POST /api/chat` | Chat 已移除 |
| `GET /api/chat/history/{session_id}` | Chat 已移除 |
| `DELETE /api/chat/history/{session_id}` | Chat 已移除 |
| `GET /api/chat/sessions` | Chat 已移除 |
| `POST /api/chat/sessions` | Chat 已移除 |
| `WebSocket /api/ws/chat` | Chat 已移除 |

**API Router 文件更新：**
```
api/routers/
├── __init__.py
├── backtest.py        # 保留
├── factor.py         # 保留
├── pipeline.py       # 保留
├── wiki.py           # 保留
├── stats.py          # 保留
├── settings.py      # 保留
├── prompts.py        # 新增
├── strategy.py       # 新增（strategy CRUD）
├── strategy_generate.py  # 新增（generate 端点，可选）
├── code.py          # 新增（validate/execute）
└── deps.py          # 更新（API Key 认证）
```

---

### Phase 5：前端展示页面

| Step | Task | 路由/组件 |
|------|------|----------|
| 5.1 | 创建 Dashboard | `/` - 总览（策略数量、回测状态、绩效） |
| 5.2 | 创建 Strategies | `/strategies` - 策略列表 + 状态 + 代码预览 |
| 5.3 | 创建 Backtests | `/backtests` - 回测结果 + 图表 |
| 5.4 | 创建 Portfolios | `/portfolios` - 组合概览 |
| 5.5 | 创建 Factors | `/factors` - 因子分析面板 |
| 5.6 | 创建 Status | `/status` - 系统状态 |
| 5.7 | 更新 router | 路由配置 + 导航栏更新 |

**展示页面路由：**
```
/                   → Dashboard        总览
/strategies         → Strategies       策略列表
/backtests          → Backtests        回测结果
/portfolios          → Portfolios       组合概览
/factors             → Factors          因子分析
/status             → Status           系统状态
```

**前端目录结构：**
```
frontend/src/
├── views/                    # 展示页面
│   ├── Dashboard/
│   │   ├── index.vue
│   │   └── components/
│   ├── Strategies/
│   ├── Backtests/
│   ├── Portfolios/
│   ├── Factors/
│   └── Status/
├── archive/                   # 归档的 Chat 组件
│   └── chat/
│       ├── AgentChat/
│       └── Chat/
├── components/               # 基础组件（保留）
│   ├── common/
│   └── charts/
├── api/                      # API 客户端
│   ├── backtest.ts
│   ├── strategy.ts
│   └── prompts.ts
├── stores/                   # 状态管理
│   └── app.ts
└── router/                   # 路由配置
    └── index.ts
```

---

### Phase 6：清理与文档

| Step | Task | 依赖 |
|------|------|------|
| 6.1 | 清理无用的导入和依赖 | Phase 1-5 |
| 6.2 | 删除废弃的 LLM 配置 | Phase 4 |
| 6.3 | 更新 `README.md`（新架构说明） | Phase 1-5 |
| 6.4 | 更新 API 文档 | Phase 4 |
| 6.5 | 验证所有端点工作正常 | Phase 4 |
| 6.6 | 验证前端展示页面正常 | Phase 5 |

---

## 4. 最终目录结构

```
QuantNodes/
├── methods/                  # 纯方法（从 agent/tools 迁移）
│   ├── __init__.py
│   ├── backtest.py
│   ├── sandbox.py
│   ├── pipeline.py
│   ├── factor.py
│   ├── wiki.py
│   ├── file_ops.py
│   ├── code_search.py
│   └── git_ops.py
├── prompts/                   # 提示词库（新增）
│   ├── __init__.py
│   ├── VERSION
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── momentum.py
│   │   ├── mean_reversion.py
│   │   ├── trend_following.py
│   │   ├── pairs_trading.py
│   │   └── market_neutral.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── standard.py
│   │   └── factor_based.py
│   └── factor/
│       ├── __init__.py
│       ├── ic_analysis.py
│       ├── group_backtest.py
│       └── correlation.py
├── api/                       # API 服务器
│   ├── main.py
│   ├── config.py
│   ├── deps.py               # API Key 认证
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── backtest.py
│   │   ├── factor.py
│   │   ├── pipeline.py
│   │   ├── wiki.py
│   │   ├── stats.py
│   │   ├── settings.py
│   │   ├── prompts.py        # 新增
│   │   ├── strategy.py      # 新增（strategy CRUD）
│   │   ├── strategy_generate.py  # 新增（可选 generate）
│   │   └── code.py          # 新增（validate/execute）
│   └── services/
│       ├── backtest_service.py
│       ├── factor_service.py
│       ├── pipeline_service.py
│       └── wiki_service.py
├── archive/                           # 统一归档目录
│   ├── QuantNodes/                # QuantNodes 包归档
│   │   └── agent/                  # Agent 系统归档
│   │       ├── core/
│   │       ├── session/
│   │       ├── providers/
│   │       └── skills/
│   ├── frontend/                   # 前端归档
│   │   └── src/archive/            # 前端 Chat UI
│   ├── api/                       # API 归档
│   │   └── archive/               # API Agent 端点
│   └── docs/                      # 文档归档
│       └── archived/              # 历史文档
├── QuantNodes/                    # QuantNodes 主包
│   ├── methods/                   # 纯方法（外部 Agent API）
│   ├── prompts/                   # 提示词库
│   ├── factor_node/               # 因子引擎（317+算子）
│   ├── backtest/                  # 回测引擎
│   ├── core/                      # 核心架构（BaseNode, Pipeline）
│   ├── database_node/             # 多数据库支持
│   └── ...
├── api/                           # API 服务器
│   ├── routers/
│   └── services/
├── frontend/src/                   # 前端源码
│   ├── views/                     # 展示页面
│   ├── components/                # 基础组件
│   └── ...
└── tests/                         # 测试套件
```

---

## 5. 外部 Agent 使用流程

### 5.1 简单策略（提示词方案）

```python
import requests

# 1. 获取提示词
prompts = requests.get("http://quantnodes:8000/api/prompts/strategy/momentum")
prompt_data = prompts.json()

# 2. 用自己的 LLM 生成代码
prompt = prompt_data["prompt"]
params = {
    "symbol": "BTC",
    "window": 20,
    "threshold": 0.05
}
code = my_llm.generate(prompt, params=params)

# 3. 验证代码安全
validation = requests.post(
    "http://quantnodes:8000/api/code/validate",
    json={"code": code},
    headers={"X-API-Key": "qn_live_xxxxxxxxxxxxxxxxxxxxxxxx"}
)
if not validation.json()["is_safe"]:
    print(f"Validation failed: {validation.json()['errors']}")
    return

# 4. 运行回测
result = requests.post(
    "http://quantnodes:8000/api/backtest/run",
    json={
        "code": code,
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 1000000
    },
    headers={"X-API-Key": "qn_live_xxxxxxxxxxxxxxxxxxxxxxxx"}
)
```

### 5.2 复杂策略（生成端点）

```python
import requests

# 1. 调用生成端点（可选高级功能）
generated = requests.post(
    "http://quantnodes:8000/api/strategy/generate",
    json={
        "type": "market_neutral",
        "params": {
            "symbols": ["BTC", "ETH"],
            "lookback": 60,
            "z_score_threshold": 2.0
        }
    },
    headers={"X-API-Key": "qn_live_xxxxxxxxxxxxxxxxxxxxxxxx"}
)
code = generated.json()["code"]

# 2. 验证并执行
validation = requests.post(
    "http://quantnodes:8000/api/code/validate",
    json={"code": code}
)
result = requests.post(
    "http://quantnodes:8000/api/backtest/run",
    json={"code": code, "start_date": "2020-01-01", "end_date": "2024-12-31"}
)
```

---

## 6. 待确认问题

| 问题 | 状态 | 说明 |
|------|------|------|
| Phase 1-6 执行顺序是否正确？ | 已确认 | 依赖关系已明确 |
| API Key 生成方式？ | 已确认 | 手动配置在 settings.json |
| `/api/strategy/generate` 使用哪个 LLM？ | 已确认 | 可选功能，使用外部 LLM 配置 |
| 现有策略代码如何处理？ | 已确认 | 迁移到 prompts/example_code 作为参考 |
| Chat 数据库表是否删除？ | 已确认 | 标记为 archive，不删除数据 |
| 是否需要保留 WebSocket？ | 待确认 | 目前计划移除所有 WS |

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 外部 Agent 生成的代码质量不可控 | 高 | 高 | 提供完整的参考代码提示词 + 代码验证端点 |
| 提示词需要频繁更新 | 中 | 中 | 提示词版本化管理 + VERSION 文件 |
| API 性能瓶颈 | 中 | 中 | 异步任务队列（future） |
| 外部 Agent 需要适配新 API | 高 | 中 | 提供详细的 API 文档 |
| 数据库迁移丢失数据 | 低 | 高 | 只标记为 archive，不删除 |

---

## 8. 验证清单

### Phase 1 完成标准
- [ ] `archive/frontend/src/archive/chat/` 存在且包含所有 Chat 相关代码
- [ ] `frontend/src/views/` 中无 AgentChat 相关代码
- [ ] `frontend/src/components/` 中无 Chat 相关组件
- [ ] `router/index.ts` 已更新，无 `/chat` 路由

### Phase 2 完成标准
- [ ] `QuantNodes/methods/` 存在且包含所有迁移的方法
- [ ] `QuantNodes/methods/__init__.py` 可正常导入所有方法
- [ ] API 端点可正常调用 methods

### Phase 3 完成标准
- [ ] `QuantNodes/prompts/` 存在且包含所有提示词
- [ ] 每个提示词包含 version, prompt, required_params, validation_rules, example_code
- [ ] `GET /api/prompts/strategy/momentum` 返回正确的 JSON

### Phase 4 完成标准
- [ ] Chat 相关端点全部移除
- [ ] API Key 认证正常工作
- [ ] `/api/prompts/` 端点正常返回数据
- [ ] `/api/code/validate` 端点正常工作
- [ ] 数据库 chat 相关表已标记为 archive

### Phase 5 完成标准
- [ ] 前端 `/` 路由显示 Dashboard
- [ ] 前端 `/strategies` 路由显示策略列表
- [ ] 前端 `/backtests` 路由显示回测结果
- [ ] 导航栏正常工作

### Phase 6 完成标准
- [ ] 无废弃导入警告
- [ ] README.md 已更新
- [ ] API 文档已更新