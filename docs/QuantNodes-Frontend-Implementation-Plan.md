# QuantNodes 前端实施计划

> 版本：v1.0  
> 日期：2026-05-07  
> 状态：规划中

---

## 1. 概述

### 1.1 目标

为 QuantNodes 项目构建现代化 Web 前端，提供：
- Agent 对话界面 (WebSocket 流式响应)
- Dashboard 数据概览
- Wiki 知识库浏览/管理
- 回测中心
- 因子分析可视化
- 策略编辑器

### 1.2 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| **前端框架** | Vue 3 | 3.4+ |
| **UI 框架** | Ant Design Vue | 4.x |
| **状态管理** | Pinia | 2.x |
| **路由** | Vue Router | 4.x |
| **图表** | ECharts | 5.x |
| **代码编辑器** | Monaco Editor | latest |
| **构建工具** | Vite | 5.x |
| **后端 API** | FastAPI | 0.110+ |
| **WebSocket** | FastAPI WebSocket | built-in |
| **运行时** | Uvicorn | latest |
| **容器化** | Docker Compose | v2 |
| **Web 服务器** | Nginx | 1.25+ |

### 1.3 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      Nginx (Port 80)                        │
│                    Static Files + Proxy                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌────────────────────────────┐  │
│  │   Vue 3 Frontend    │    │    FastAPI Backend API     │  │
│  │   (Port 3000)       │◄──►│    (Port 8000)             │  │
│  │                     │    │                            │  │
│  │  - Dashboard        │    │  - REST Endpoints          │  │
│  │  - Agent Chat       │    │  - WebSocket /ws/chat      │  │
│  │  - Wiki Browser     │    │  - Agent Integration       │  │
│  │  - Backtest Center  │    │  - Wiki CRUD               │  │
│  │  - Factor Analysis  │    │  - Backtest Execution      │  │
│  │  - Strategy Editor  │    │  - Factor Analysis         │  │
│  └─────────────────────┘    └────────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   QuantNodes Core (Python)                   │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │  Agent   │ │   Wiki   │ │  Skills  │ │     Dream     │  │
│  │  System  │ │  System  │ │  System  │ │    System     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Tools Layer                       │   │
│  │  EchoTool │ SandboxTool │ FactorTool │ BacktestTool  │   │
│  │  WikiTool │ StrategyTool│ PipelineTool│ ConfigBacktest│   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
QuantNodes/
├── frontend/                          # Vue 3 前端项目
│   ├── public/                        # 静态资源
│   ├── src/
│   │   ├── api/                       # API 请求模块
│   │   │   ├── index.ts               # Axios 实例配置
│   │   │   ├── agent.ts               # Agent/Chat API
│   │   │   ├── wiki.ts                # Wiki CRUD API
│   │   │   ├── backtest.ts            # 回测 API
│   │   │   ├── factor.ts              # 因子分析 API
│   │   │   ├── skill.ts               # 技能管理 API
│   │   │   └── dream.ts               # Dream 洞察 API
│   │   ├── assets/                    # 图片、图标、样式
│   │   ├── components/                # 共享组件
│   │   │   ├── Layout/                # 布局组件
│   │   │   │   ├── AppLayout.vue      # 主布局
│   │   │   │   ├── AppHeader.vue      # 顶部导航
│   │   │   │   ├── AppSidebar.vue     # 侧边栏
│   │   │   │   └── AppFooter.vue      # 底部
│   │   │   ├── Charts/                # 图表组件
│   │   │   │   ├── EquityCurve.vue    # 权益曲线
│   │   │   │   ├── IcChart.vue        # IC 分析图
│   │   │   │   └── MetricCard.vue     # 指标卡片
│   │   │   ├── Chat/                  # 聊天组件
│   │   │   │   ├── ChatMessage.vue    # 消息气泡
│   │   │   │   ├── ChatInput.vue      # 输入框
│   │   │   │   └── ToolCallCard.vue   # 工具调用展示
│   │   │   └── Editor/                # 编辑器组件
│   │   │       └── MonacoEditor.vue   # Monaco Editor 封装
│   │   ├── composables/               # 组合式函数
│   │   │   ├── useWebSocket.ts        # WebSocket 管理
│   │   │   ├── useAgent.ts            # Agent 交互
│   │   │   └── useTheme.ts            # 主题切换
│   │   ├── router/                    # 路由配置
│   │   │   └── index.ts
│   │   ├── stores/                    # Pinia 状态管理
│   │   │   ├── agent.ts               # Agent 状态
│   │   │   ├── wiki.ts                # Wiki 状态
│   │   │   └── app.ts                 # 全局状态
│   │   ├── views/                     # 页面组件
│   │   │   ├── Dashboard/             # 仪表盘
│   │   │   │   └── index.vue
│   │   │   ├── AgentChat/             # Agent 对话
│   │   │   │   └── index.vue
│   │   │   ├── Wiki/                  # Wiki 知识库
│   │   │   │   ├── FactorList.vue     # 因子列表
│   │   │   │   ├── StrategyList.vue   # 策略列表
│   │   │   │   └── FactorDetail.vue   # 因子详情
│   │   │   ├── Backtest/              # 回测中心
│   │   │   │   ├── ConfigEditor.vue   # YAML 配置编辑
│   │   │   │   └── ResultView.vue     # 回测结果
│   │   │   ├── FactorAnalysis/        # 因子分析
│   │   │   │   └── index.vue
│   │   │   ├── Strategy/              # 策略编辑器
│   │   │   │   └── Editor.vue
│   │   │   ├── Dream/                 # Dream 洞察
│   │   │   │   └── index.vue
│   │   │   └── Settings/              # 设置
│   │   │       └── index.vue
│   │   ├── styles/                    # 全局样式
│   │   │   ├── variables.less         # Ant Design 变量
│   │   │   └── global.less
│   │   ├── utils/                     # 工具函数
│   │   ├── App.vue                    # 根组件
│   │   └── main.ts                    # 入口文件
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── .env.development
│
├── api/                               # FastAPI 后端
│   ├── main.py                        # FastAPI 应用入口
│   ├── config.py                      # 配置管理
│   ├── deps.py                        # 依赖注入
│   ├── routers/                       # 路由模块
│   │   ├── __init__.py
│   │   ├── agent.py                   # Agent/Chat 路由
│   │   ├── wiki.py                    # Wiki CRUD 路由
│   │   ├── backtest.py                # 回测路由
│   │   ├── factor.py                  # 因子分析路由
│   │   ├── skill.py                   # 技能管理路由
│   │   ├── dream.py                   # Dream 路由
│   │   └── stats.py                   # 统计数据路由
│   ├── schemas/                       # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── agent.py                   # Agent 请求/响应模型
│   │   ├── wiki.py                    # Wiki 数据模型
│   │   ├── backtest.py                # 回测数据模型
│   │   └── common.py                  # 通用模型
│   ├── services/                      # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── agent_service.py           # Agent 服务
│   │   ├── wiki_service.py            # Wiki 服务
│   │   └── backtest_service.py        # 回测服务
│   └── websocket/                     # WebSocket 处理
│       ├── __init__.py
│       └── chat.py                    # Agent 对话 WebSocket
│
├── docker/                            # Docker 配置
│   ├── nginx/
│   │   └── nginx.conf                 # Nginx 配置
│   ├── Dockerfile.frontend            # 前端构建镜像
│   └── Dockerfile.api                 # API 镜像
│
├── docker-compose.yml                 # Docker Compose 编排
└── Makefile                           # 开发命令快捷方式
```

---

## 3. 后端 API 设计

### 3.1 API 端点总览

| 模块 | 方法 | 端点 | 描述 |
|------|------|------|------|
| **Stats** | GET | `/api/stats` | 获取统计数据 |
| **Agent** | POST | `/api/chat` | 发送消息 (同步) |
| **Agent** | WS | `/ws/chat` | Agent 对话 (流式) |
| **Wiki** | GET | `/api/wiki/factors` | 因子列表 |
| **Wiki** | GET | `/api/wiki/factors/{name}` | 因子详情 |
| **Wiki** | POST | `/api/wiki/factors` | 创建因子 |
| **Wiki** | PUT | `/api/wiki/factors/{name}` | 更新因子 |
| **Wiki** | DELETE | `/api/wiki/factors/{name}` | 删除因子 |
| **Wiki** | GET | `/api/wiki/strategies` | 策略列表 |
| **Wiki** | GET | `/api/wiki/strategies/{name}` | 策略详情 |
| **Wiki** | POST | `/api/wiki/strategies` | 创建策略 |
| **Wiki** | GET | `/api/wiki/search` | 搜索 Wiki |
| **Backtest** | POST | `/api/backtest/run` | 运行回测 |
| **Backtest** | GET | `/api/backtest/{id}` | 获取回测结果 |
| **Factor** | POST | `/api/factor/analyze` | 因子分析 |
| **Skill** | GET | `/api/skills` | 技能列表 |
| **Skill** | POST | `/api/skills/{name}/execute` | 执行技能 |
| **Dream** | GET | `/api/dreams` | 洞察列表 |
| **Dream** | GET | `/api/dreams/stats` | 洞察统计 |

### 3.2 WebSocket 消息格式

#### 客户端 → 服务端
```json
{
  "type": "message",
  "content": "生成一个动量因子",
  "session_id": "optional-session-id"
}
```

#### 服务端 → 客户端 (流式)
```json
{
  "type": "chunk",
  "content": "正在分析",
  "message_id": "msg-123"
}
```

```json
{
  "type": "tool_call",
  "tool_name": "factor_tool",
  "arguments": {"expression": "..."},
  "message_id": "msg-123"
}
```

```json
{
  "type": "tool_result",
  "tool_name": "factor_tool",
  "result": {"status": "success", ...},
  "message_id": "msg-123"
}
```

```json
{
  "type": "done",
  "message_id": "msg-123",
  "usage": {"prompt_tokens": 100, "completion_tokens": 50}
}
```

### 3.3 数据模型

```python
# api/schemas/agent.py
from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message_id: str
    content: str
    tools_used: List[str] = []
    usage: dict = {}

class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: Optional[dict] = None

# api/schemas/wiki.py
class FactorInfo(BaseModel):
    name: str
    formula: str
    source: str
    category: str
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    icir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    tags: List[str] = []

class StrategyInfo(BaseModel):
    name: str
    description: str
    category: str
    tags: List[str] = []
    strategy_yaml: Optional[str] = None
    backtest_result: Optional[dict] = None

# api/schemas/backtest.py
class BacktestRequest(BaseModel):
    config_yaml: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestResult(BaseModel):
    status: str
    summary: dict
    config_info: dict
    warnings: List[str] = []
```

---

## 4. 前端页面设计

### 4.1 Dashboard 仪表盘

**布局：**
```
┌─────────────────────────────────────────────────────────┐
│  Header: QuantNodes Logo │ Agent Chat │ Settings        │
├─────────┬───────────────────────────────────────────────┤
│ Sidebar │  Dashboard Content                            │
│         │                                               │
│ □ Home  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│ □ Chat  │  │ Factors │ │Strategies│ │ Backtests│        │
│ □ Wiki  │  │   128   │ │    45   │ │    23   │         │
│ □ Back  │  └─────────┘ └─────────┘ └─────────┘         │
│ □ Factor│                                               │
│ □ Dream │  ┌─────────────────────────────────────────┐  │
│ □ Skills│  │  Recent Activity / Dream Insights       │  │
│         │  │  - momentum_20d: IC=0.05, ICIR=1.2     │  │
│         │  │  - dual_ma strategy backtested ✓       │  │
│         │  └─────────────────────────────────────────┘  │
│         │                                               │
│         │  ┌─────────────────────────────────────────┐  │
│         │  │  Quick Actions                          │  │
│         │  │  [New Factor] [New Strategy] [Run Test] │  │
│         │  └─────────────────────────────────────────┘  │
└─────────┴───────────────────────────────────────────────┘
```

**数据来源：**
- `GET /api/stats` → 因子数、策略数、回测数
- `GET /api/dreams?limit=5` → 最近洞察
- `GET /api/wiki/factors?sort=updated&limit=5` → 最近更新的因子

### 4.2 Agent Chat 对话界面

**布局：**
```
┌─────────────────────────────────────────────────────────┐
│  Header                                                 │
├─────────┬───────────────────────────────────────────────┤
│ Sidebar │  Chat Area                                    │
│         │                                               │
│         │  ┌─────────────────────────────────────────┐  │
│         │  │  User: 生成一个动量因子                   │  │
│         │  └─────────────────────────────────────────┘  │
│         │                                               │
│         │  ┌─────────────────────────────────────────┐  │
│         │  │  Agent: 正在分析您的需求...              │  │
│         │  │                                         │  │
│         │  │  ┌─────────────────────────────────┐   │  │
│         │  │  │ 🔧 Tool Call: factor_tool        │   │  │
│         │  │  │ Args: {expression: "close/..."}  │   │  │
│         │  │  │ Result: ✅ Success               │   │  │
│         │  │  └─────────────────────────────────┘   │  │
│         │  │                                         │  │
│         │  │  已生成动量因子，IC 均值为 0.05，        │  │
│         │  │  ICIR 为 1.2，表现良好。                │  │
│         │  └─────────────────────────────────────────┘  │
│         │                                               │
│         │  ┌─────────────────────────────────────────┐  │
│         │  │  [Input Box...]           [Send] [Stop] │  │
│         │  └─────────────────────────────────────────┘  │
└─────────┴───────────────────────────────────────────────┘
```

**WebSocket 流程：**
1. 用户输入消息
2. 前端通过 WebSocket 发送 `{type: "message", content: "..."}`
3. 服务端流式返回 `{type: "chunk", content: "..."}`
4. 工具调用时返回 `{type: "tool_call", ...}` 和 `{type: "tool_result", ...}`
5. 完成时返回 `{type: "done", ...}`

### 4.3 Wiki 知识库

**因子列表页：**
```
┌─────────────────────────────────────────────────────────┐
│  Wiki > Factors                                         │
├─────────────────────────────────────────────────────────┤
│  [Search...] [Category: All ▼] [Sort: Updated ▼]      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │ momentum_20d        │ momentum  │ IC: 0.05      │   │
│  │ 20日动量因子         │ ICIR: 1.2 │ Updated: 2h   │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ bollinger_breakout  │ technical │ IC: 0.03      │   │
│  │ 布林带突破因子       │ ICIR: 0.8 │ Updated: 1d   │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ volume_price        │ composite │ IC: 0.04      │   │
│  │ 量价因子            │ ICIR: 1.0 │ Updated: 3d   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  [Create New Factor]                                   │
└─────────────────────────────────────────────────────────┘
```

### 4.4 回测中心

**布局：**
```
┌─────────────────────────────────────────────────────────┐
│  Backtest Center                                        │
├─────────────────────────┬───────────────────────────────┤
│  Config Editor          │  Results                      │
│                         │                               │
│  ┌─────────────────┐   │  ┌─────────────────────────┐  │
│  │ name: test      │   │  │  Performance Summary    │  │
│  │ factors:        │   │  │  ─────────────────────  │  │
│  │   - momentum    │   │  │  Total Return: +12.5%   │  │
│  │ operations:     │   │  │  Sharpe Ratio: 1.8      │  │
│  │   - rank        │   │  │  Max Drawdown: -8.2%    │  │
│  │ backtest:       │   │  │  Win Rate: 65%          │  │
│  │   start: ...    │   │  └─────────────────────────┘  │
│  │   end: ...      │   │                               │
│  │   cash: 1000000 │   │  ┌─────────────────────────┐  │
│  └─────────────────┘   │  │  Equity Curve Chart     │  │
│                         │  │  [ECharts Line]         │  │
│  [Run Backtest]         │  └─────────────────────────┘  │
│  [Save Config]          │                               │
└─────────────────────────┴───────────────────────────────┘
```

---

## 5. 实施阶段

### Phase 1: 基础设施 (2-3 天)

**目标：** 搭建前后端骨架

| 任务 | 产出文件 | 优先级 |
|------|----------|--------|
| 创建 Vue 3 项目 | `frontend/` | P0 |
| 配置 Vite + Ant Design Vue | `vite.config.ts`, `package.json` | P0 |
| 创建基础布局 | `AppLayout.vue`, `AppHeader.vue`, `AppSidebar.vue` | P0 |
| 创建 FastAPI 应用 | `api/main.py`, `api/config.py` | P0 |
| 配置路由 | `frontend/src/router/index.ts` | P0 |
| 配置 Pinia | `frontend/src/stores/` | P0 |
| Docker Compose 配置 | `docker-compose.yml` | P1 |

### Phase 2: 核心页面 (3-5 天)

**目标：** 实现 Dashboard + Agent Chat

| 任务 | 产出文件 | 优先级 |
|------|----------|--------|
| Dashboard 页 | `views/Dashboard/index.vue` | P0 |
| 统计卡片组件 | `components/Charts/MetricCard.vue` | P0 |
| Agent Chat 页 | `views/AgentChat/index.vue` | P0 |
| WebSocket 管理 | `composables/useWebSocket.ts` | P0 |
| 聊天消息组件 | `components/Chat/ChatMessage.vue` | P0 |
| 聊天输入组件 | `components/Chat/ChatInput.vue` | P0 |
| 工具调用展示 | `components/Chat/ToolCallCard.vue` | P0 |
| Agent API 路由 | `api/routers/agent.py` | P0 |
| WebSocket 处理 | `api/websocket/chat.py` | P0 |
| 统计 API 路由 | `api/routers/stats.py` | P1 |

### Phase 3: Wiki 知识库 (2-3 天)

**目标：** 因子/策略浏览和管理

| 任务 | 产出文件 | 优先级 |
|------|----------|--------|
| 因子列表页 | `views/Wiki/FactorList.vue` | P0 |
| 策略列表页 | `views/Wiki/StrategyList.vue` | P0 |
| 因子详情页 | `views/Wiki/FactorDetail.vue` | P1 |
| Wiki API 路由 | `api/routers/wiki.py` | P0 |
| Wiki 服务层 | `api/services/wiki_service.py` | P0 |

### Phase 4: 量化功能 (3-5 天)

**目标：** 回测 + 因子分析

| 任务 | 产出文件 | 优先级 |
|------|----------|--------|
| 回测配置编辑器 | `views/Backtest/ConfigEditor.vue` | P0 |
| 回测结果展示 | `views/Backtest/ResultView.vue` | P0 |
| 权益曲线图表 | `components/Charts/EquityCurve.vue` | P0 |
| 因子分析页 | `views/FactorAnalysis/index.vue` | P0 |
| IC 分析图表 | `components/Charts/IcChart.vue` | P0 |
| 回测 API 路由 | `api/routers/backtest.py` | P0 |
| 因子分析 API | `api/routers/factor.py` | P0 |

### Phase 5: 高级功能 (2-3 天)

**目标：** Dream + Skills + 策略编辑器

| 任务 | 产出文件 | 优先级 |
|------|----------|--------|
| Dream 洞察页 | `views/Dream/index.vue` | P1 |
| 策略编辑器 | `views/Strategy/Editor.vue` | P1 |
| Monaco Editor 封装 | `components/Editor/MonacoEditor.vue` | P1 |
| Skill 管理页 | `views/Skills/index.vue` | P2 |
| 设置页 | `views/Settings/index.vue` | P2 |

---

## 6. 关键实现细节

### 6.1 WebSocket 连接管理

```typescript
// composables/useWebSocket.ts
import { ref, onUnmounted } from 'vue'

export function useWebSocket(url: string) {
  const isConnected = ref(false)
  const messages = ref<any[]>([])
  let ws: WebSocket | null = null

  function connect() {
    ws = new WebSocket(url)
    
    ws.onopen = () => {
      isConnected.value = true
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      messages.value.push(data)
    }
    
    ws.onclose = () => {
      isConnected.value = false
    }
  }

  function send(message: object) {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message))
    }
  }

  function disconnect() {
    ws?.close()
  }

  onUnmounted(() => {
    disconnect()
  })

  return { isConnected, messages, connect, send, disconnect }
}
```

### 6.2 FastAPI Agent 路由

```python
# api/routers/agent.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.agent_service import AgentService

router = APIRouter(prefix="/api", tags=["agent"])

@router.post("/chat")
async def chat(message: ChatMessage):
    """同步 Agent 对话"""
    service = AgentService()
    result = await service.run(message.content, message.session_id)
    return ChatResponse(
        message_id=result.message_id,
        content=result.final_content,
        tools_used=result.tools_used,
        usage=result.usage
    )

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket Agent 对话 (流式)"""
    await websocket.accept()
    service = AgentService()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            async for event in service.stream(data["content"], data.get("session_id")):
                await websocket.send_json(event)
                
    except WebSocketDisconnect:
        pass
```

### 6.3 Agent 服务层

```python
# api/services/agent_service.py
import asyncio
from typing import AsyncGenerator
from QuantNodes.agent import Agent

class AgentService:
    def __init__(self):
        self.agent = Agent(workspace="./workspace")
    
    async def run(self, content: str, session_id: str = None) -> AgentRunResult:
        """同步执行 Agent"""
        return await self.agent.run(content)
    
    async def stream(self, content: str, session_id: str = None) -> AsyncGenerator[dict, None]:
        """流式执行 Agent"""
        # 通过 MessageBus 实现流式输出
        async for event in self.agent.stream(content):
            yield event
```

### 6.4 Monaco Editor 封装

```vue
<!-- components/Editor/MonacoEditor.vue -->
<template>
  <div ref="editorContainer" class="monaco-editor" />
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import * as monaco from 'monaco-editor'

const props = defineProps<{
  modelValue: string
  language?: string
  readOnly?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const editorContainer = ref<HTMLElement>()
let editor: monaco.editor.IStandaloneCodeEditor

onMounted(() => {
  editor = monaco.editor.create(editorContainer.value!, {
    value: props.modelValue,
    language: props.language || 'python',
    readOnly: props.readOnly,
    minimap: { enabled: false },
    fontSize: 14,
    theme: 'vs-dark',
    automaticLayout: true,
  })

  editor.onDidChangeModelContent(() => {
    emit('update:modelValue', editor.getValue())
  })
})

watch(() => props.modelValue, (newVal) => {
  if (editor && editor.getValue() !== newVal) {
    editor.setValue(newVal)
  }
})
</script>
```

---

## 7. Docker 部署

### 7.1 docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - api
    networks:
      - quantnodes-net

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
      - workspace:/app/workspace
    environment:
      - PYTHONPATH=/app
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - quantnodes-net

  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - api
    networks:
      - quantnodes-net

volumes:
  workspace:

networks:
  quantnodes-net:
```

### 7.2 Nginx 配置

```nginx
# docker/nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:80;
    }
    
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            proxy_pass http://frontend;
        }

        # API Proxy
        location /api/ {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # WebSocket Proxy
        location /ws/ {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
}
```

### 7.3 Dockerfile.frontend

```dockerfile
# docker/Dockerfile.frontend
FROM node:20-alpine AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM nginx:1.25-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx/frontend.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### 7.4 Dockerfile.api

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8. 开发环境配置

### 8.1 package.json

```json
{
  "name": "quantnodes-frontend",
  "version": "0.1.0",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext .vue,.js,.jsx,.cjs,.mjs,.ts,.tsx,.cts,.mts --fix"
  },
  "dependencies": {
    "vue": "^3.4.0",
    "vue-router": "^4.3.0",
    "pinia": "^2.1.0",
    "ant-design-vue": "^4.2.0",
    "@ant-design/icons-vue": "^7.0.0",
    "axios": "^1.7.0",
    "echarts": "^5.5.0",
    "vue-echarts": "^7.0.0",
    "monaco-editor": "^0.47.0",
    "dayjs": "^1.11.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.4.0",
    "typescript": "^5.4.0",
    "vue-tsc": "^2.0.0",
    "less": "^4.2.0",
    "eslint": "^8.57.0",
    "@typescript-eslint/parser": "^7.0.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0"
  }
}
```

### 8.2 requirements.txt (Backend)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
websockets>=12.0
python-dotenv>=1.0.0
```

---

## 9. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| WebSocket 连接不稳定 | Agent 对话中断 | 实现自动重连机制 |
| Monaco Editor 包体积大 | 首屏加载慢 | 按需加载 + CDN |
| Agent 异步执行超时 | 前端等待过长 | 超时提示 + 后台执行 |
| ECharts 图表渲染慢 | 数据量大时卡顿 | 虚拟滚动 + 数据采样 |

---

## 10. 测试计划

### 10.1 单元测试

- 组件测试: Vue Test Utils
- API 测试: pytest + httpx
- WebSocket 测试: pytest-websockets

### 10.2 E2E 测试

- Playwright 端到端测试
- 覆盖核心流程: Chat → Tool Call → Result

### 10.3 性能测试

- Lighthouse 性能评分 > 90
- 首屏加载 < 2s (4G 网络)

---

## 附录 A: 参考资源

- [Ant Design Vue 4.x 文档](https://www.antdv.com/)
- [Vue 3 官方文档](https://vuejs.org/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Monaco Editor 文档](https://microsoft.github.io/monaco-editor/)
- [ECharts 文档](https://echarts.apache.org/)
- [QuantDinger Vue 前端](https://github.com/brokermr810/QuantDinger-Vue) (参考设计)
