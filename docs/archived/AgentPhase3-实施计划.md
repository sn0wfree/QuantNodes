# Agent Phase 3 实施计划：Wiki 工具集成

> 文档版本: v1.0  
> 创建日期: 2026-05-07  
> 状态: 待实施

---

## 一、背景

### 1.1 目标

Agent Phase 3 是 Agent 系统与 llmwikify Wiki 知识库的集成层。目标是让 Agent 能够通过标准工具接口（ToolRegistry）操作 Wiki 知识库，实现策略/因子/研报知识的自动沉淀。

### 1.2 现有能力

| 组件 | 状态 | 说明 |
|------|------|------|
| llmwikify MCP Server | ✅ 已实现 | 20+ Wiki 工具，`llmwikify mcp` 启动 |
| llmwikify Python API | ✅ 已实现 | `Wiki` 类直接调用 |
| WikiFactorProxy | ✅ 已实现 | Factor/Llogic CRUD，支持 3A/3B/3C |
| Agent ToolRegistry | ✅ 已实现 | 工具注册表框架 |
| Agent Phase 1-2 | ✅ 已完成 | 核心框架 + 工具集 |

### 1.3 连接方式选择

经讨论，采用**直接 Python API** 方式集成：

- **优点**：无网络开销、实现简单、调试方便
- **缺点**：Agent 与 Wiki 紧耦合（同一进程）
- **结论**：适合当前阶段，后续可平滑迁移到 MCP 协议

---

## 二、架构设计

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Core                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ AgentLoop   │  │ ToolRegistry │  │ ContextBuilder   │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ tool call
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent Tools                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │Backtest  │  │ Factor   │  │ Pipeline │  │   Wiki    │  │
│  │Tool      │  │ Tool     │  │ Tool     │  │   Tool    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ direct API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Research Layer                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │WikiFactorProxy│  │AutoResearcher│  │ReportReproducer  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  llmwikify Wiki                                              │
│  Wiki (Python API) → 文件系统 (wiki/) → SQLite FTS5         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流（简化版）

```
用户: "帮我研究一个动量因子，存入因子库"
    ↓
Agent: 理解意图，调用 WikiTool.store_factor()
    ↓
WikiTool: 验证参数，调用 WikiFactorProxy
    ↓
WikiFactorProxy: 渲染 Markdown，调用 wiki.write_page()
    ↓
llmwikify Wiki: 写入 wiki/Factor/momentum_20d.md，更新索引
```

### 2.3 场景一：因子存储（store_factor）

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 0: 用户交互层                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
用户: "研究双均线动量因子，IC>0.03就存入因子库"
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Agent 理解层                                                │
│                                                                          │
│ AgentLoop: 解析用户意图                                               │
│   → 识别需要创建因子                          │
│   → 调用 WikiTool.store_factor()                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
ToolRegistry.lookup("wiki") → WikiTool
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2: WikiTool 参数层                                             │
│                                                                          │
│ WikiTool.store_factor(                                              │
│     name="dual_ma_momentum",                                        │
│     formula="ma(close,20)/ma(close,60)-1",                          │
│     source="auto_research",                                         │
│     category="momentum",                                             │
│     ic_mean=0.042,                                                   │
│     tags=["dual_ma", "momentum"],                                   │
│ )                                                                   │
│                                                                          │
│ 参数验证:                                                              │
│   - name: 非空、合法文件名                                            │
│   - formula: 非空                                                    │
│   - ic_mean: 数值范围检查                                            │
│   - category: 枚举值验证                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
WikiTool.proxy.store_factor(factor)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: WikiFactorProxy 转换层                                      │
│                                                                          │
│ WikiFactorProxy.store_factor(WikiFactor):                             │
│                                                                          │
│   1. 构建页面名:                                                     │
│      page_name = "Factor/dual_ma_momentum"                          │
│                                                                          │
│   2. 渲染 Markdown (WikiFactorProxy._render_factor_markdown):        │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ ---                                                          │    │
│      │ type: Factor                                                │    │
│      │ name: dual_ma_momentum                                      │    │
│      │ formula: "ma(close,20)/ma(close,60)-1"                     │    │
│      │ source: auto_research                                        │    │
│      │ category: momentum                                           │    │
│      │ tags: [dual_ma, momentum]                                   │    │
│      │ ic_mean: 0.042                                              │    │
│      │ ...                                                         │    │
│      │ ---                                                          │    │
│      │ ## 单因子表现                                                │    │
│      │ | IC Mean | 0.042 |                                         │    │
│      └─────────────────────────────────────────────────────────┘    │
│                                                                          │
│   3. 调用 wiki.write_page(page_name, content)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
wiki.write_page("Factor/dual_ma_momentum", markdown_content)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4: llmwikify Wiki 持久层                                       │
│                                                                          │
│ 4.1 文件写入:                                                        │
│    wiki/Factor/dual_ma_momentum.md                                   │
│                                                                          │
│ 4.2 SQLite 索引更新:                                                │
│    - 提取文本内容 (名称、标签、公式)                                   │
│    - 更新 FTS5 全文搜索索引                                            │
│    - 更新关系图谱 (如指定了 related_factors)                           │
│                                                                          │
│ 4.3 返回:                                                            │
│    page_name = "Factor/dual_ma_momentum"                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 场景二：因子查询（search_factors）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 用户: "搜索所有动量因子，IC大于0.03"                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
AgentLoop → WikiTool.search_factors(query="momentum IC>0.03", limit=20)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ WikiTool 内部处理                                                    │
│                                                                          │
│ # 方式A: 全文搜索                                                    │
│ results = wiki_tool.proxy.wiki.search("momentum", limit=20)           │
│                                                                          │
│ # 方式B: 结构化查询                                                  │
│ results = wiki_tool.proxy.list_factors(                              │
│     category=FactorCategory.MOMENTUM,                                 │
│     source=None,                                                      │
│     tags=None,                                                        │
│     limit=20,                                                         │
│ )                                                                    │
│                                                                          │
│ 返回: List[WikiFactor]                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ llmwikify Wiki 查询层                                                │
│                                                                          │
│ # SQLite FTS5 查询                                                   │
│ SELECT * FROM pages                                                  │
│ WHERE type='Factor'                                                  │
│   AND content MATCH 'momentum'                                        │
│ ORDER BY rank                                                        │
│ LIMIT 20                                                             │
│                                                                          │
│ # 或遍历文件:                                                        │
│ wiki/Factor/*.md → _parse_factor_from_page() → WikiFactor           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.5 场景三：策略存储（store_strategy）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 用户: "保存双均线策略配置到Wiki"                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
WikiTool.store_strategy(
    name="dual_ma_cross",
    strategy_yaml="""
        name: dual_ma_cross
        factors:
          - ma_20: rolling_mean(close, 20)
          - ma_60: rolling_mean(close, 60)
        signals:
          - type: cross
            fast: ma_20
            slow: ma_60
        backtest:
          start_date: 2020-01-01
    """,
    description="经典双均线策略",
    tags=["趋势跟踪", "双均线"],
)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ WikiFactorProxy.store_strategy(WikiStrategy)                          │
│                                                                          │
│ 1. page_name = "Strategy/dual_ma_cross"                              │
│                                                                          │
│ 2. 渲染 Markdown:                                                    │
│    ┌─────────────────────────────────────────────────────────┐    │
│    │ ---                                                          │    │
│    │ type: Strategy                                              │    │
│    │ name: dual_ma_cross                                        │    │
│    │ description: 经典双均线策略                                 │    │
│    │ tags: [趋势跟踪, 双均线]                                    │    │
│    │ created_at: 2026-05-07                                     │    │
│    │ ---                                                          │    │
│    │ ## 策略配置                                                 │    │
│    │ ```yaml                                                     │    │
│    │ name: dual_ma_cross                                        │    │
│    │ factors:                                                    │    │
│    │   - ma_20: rolling_mean(close, 20)                        │    │
│    │   ...                                                       │    │
│    │ ```                                                         │    │
│    └─────────────────────────────────────────────────────────┘    │
│                                                                          │
│ 3. wiki.write_page("Strategy/dual_ma_cross", content)                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.6 场景四：知识关系建立（add_relation）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 用户: "建立策略dual_ma_cross使用因子dual_ma_momentum的关系"          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
WikiTool.add_relation(
    source_name="Strategy/dual_ma_cross",
    target_name="Factor/dual_ma_momentum",
    relation="uses",
)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ WikiFactorProxy.add_relation()                                        │
│                                                                          │
│ 调用 wiki.write_relations([                                          │
│     {                                                               │
│         "source": "Strategy/dual_ma_cross",                          │
│         "target": "Factor/dual_ma_momentum",                         │
│         "relation": "uses",                                          │
│         "confidence": "EXTRACTED"                                    │
│     }                                                               │
│ ])                                                                  │
│                                                                          │
│ llmwikify 写入 SQLite relations 表:                                   │
│ ┌──────────────┬────────────────────────────┬──────────┬───────────┐  │
│ │ source      │ target                    │ relation │ confidence│  │
│ ├──────────────┼────────────────────────────┼──────────┼───────────┤  │
│ │Strategy/... │ Factor/dual_ma_momentum   │ uses     │ EXTRACTED │  │
│ └──────────────┴────────────────────────────┴──────────┴───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.7 完整调用链路图

```
用户自然语言
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ AgentLoop (agent/core/loop.py)                                   │
│                                                                  │
│ 1. 解析用户消息 → 意图识别                                        │
│ 2. 选择工具: ToolRegistry.lookup("wiki")                         │
│ 3. 构建 ToolCall: {name: "wiki", args: {...}}                   │
│ 4. 执行工具调用                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ WikiTool (agent/tools/wiki.py) — 新增                             │
│                                                                  │
│ 1. 参数验证 (BaseTool.execute 前置)                              │
│ 2. 调用 WikiFactorProxy 对应方法                                  │
│ 3. 转换返回值为 Agent 友好的格式                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ WikiFactorProxy (research/wiki.py)                                 │
│                                                                  │
│ 1. 数据模型验证 (WikiFactor/WikiLogic/WikiStrategy)              │
│ 2. Markdown 渲染 (_render_*_markdown)                            │
│ 3. 调用 wiki.write_page() / wiki.read_page() / wiki.search()    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ llmwikify Wiki (llmwikify/core/wiki.py)                          │
│                                                                  │
│ 1. 文件系统操作 (wiki/*.md)                                      │
│ 2. SQLite 索引更新 (FTS5, relations)                            │
│ 3. 返回操作结果                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、文件结构

### 3.1 新增文件

```
agent/
├── tools/
│   ├── wiki.py              # NEW: Wiki 工具封装 (~200行)
│   └── __init__.py         # UPDATE: 导出 WikiTool
```

### 3.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `research/wiki.py` | 新增 Strategy/Reproduction 类型支持 (~150行) |
| `agent/tools/__init__.py` | 导出 WikiTool |
| `agent/tools/registry.py` | WikiTool 注册到工具表 |

---

## 四、WikiTool 详细设计

### 4.1 类结构

```python
class WikiTool(BaseTool):
    """Wiki 知识库操作工具"""
    
    name = "wiki"
    description = "QuantNodes Wiki 知识库 - 因子/逻辑/策略的存取与查询"
    read_only = False  # WikiTool 包含写操作
    
    def __init__(self, wiki_path: str):
        self.proxy = WikiFactorProxy(wiki_path)
        self._logger = logging.getLogger(f"tools.{self.name}")
```

### 4.2 参数 Schema（Agent 上下文构建用）

每个工具方法需要提供完整的 JSON Schema，供 Agent 理解参数含义：

#### store_factor Schema

```python
STORE_FACTOR_SCHEMA = {
    "name": "store_factor",
    "description": "将一个验证通过的因子存储到 Wiki 知识库",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "因子名称（唯一标识，如 dual_ma_momentum）",
                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
            },
            "formula": {
                "type": "string",
                "description": "因子公式（如 ts_mean(close, 20) / ts_mean(close, 60) - 1）",
            },
            "source": {
                "type": "string",
                "enum": ["research_report", "auto_research", "manual", "derived", "imported"],
                "description": "因子来源",
            },
            "category": {
                "type": "string",
                "enum": ["momentum", "value", "quality", "volatility", "size", "growth", "other"],
                "description": "因子分类",
            },
            "ic_mean": {
                "type": "number",
                "description": "IC 均值（越大越好，通常 > 0.02）",
            },
            "ic_std": {"type": "number", "description": "IC 标准差"},
            "icir": {"type": "number", "description": "IC IR = ic_mean / ic_std"},
            "rank_ic_mean": {"type": "number", "description": "Rank IC 均值"},
            "turnover": {"type": "number", "description": "换手率（越小越稳定）"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "标签列表（如 [momentum, dual_ma]）",
            },
            "description": {"type": "string", "description": "因子描述"},
            "strategy_yaml": {"type": "string", "description": "关联的策略 YAML 配置"},
        },
        "required": ["name", "formula", "source", "category"],
    },
}
```

#### search_factors Schema

```python
SEARCH_FACTORS_SCHEMA = {
    "name": "search_factors",
    "description": "全文搜索 Wiki 中的因子",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（如 momentum、IC、dual_ma）",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "返回结果数量上限",
            },
        },
        "required": ["query"],
    },
}
```

### 4.3 工具方法

#### 4.3.1 因子操作

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `store_factor` | name, formula, source, category, ic_mean, tags, ... | page_name | 存储因子 |
| `get_factor` | name | WikiFactor | 获取因子详情 |
| `search_factors` | query, limit | List[WikiFactor] | 搜索因子 |
| `list_factors` | source, category, tags, limit | List[WikiFactor] | 列举因子 |

#### 4.3.2 逻辑操作

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `store_logic` | name, content, source, extracted_formula | page_name | 存储研报逻辑 |
| `get_logic` | name | WikiLogic | 获取逻辑详情 |
| `search_logics` | query, limit | List[WikiLogic] | 搜索逻辑 |

#### 4.3.3 策略操作（新增）

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `store_strategy` | name, strategy_yaml, description, tags | page_name | 存储策略配置 |
| `get_strategy` | name | Dict | 获取策略详情 |
| `list_strategies` | category, tags, limit | List[Dict] | 列举策略 |

#### 4.3.4 关系操作

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `add_relation` | source_name, target_name, relation | bool | 添加关系 |
| `get_neighbors` | name, direction | List[Dict] | 获取关联节点 |

#### 4.3.5 基础设施

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `ping` | - | bool | 检查 Wiki 可用性 |
| `status` | - | Dict | Wiki 状态统计 |
| `search` | query, limit, backend | List[Dict] | 全文搜索 |

### 4.4 错误处理设计

```python
class WikiToolError(Exception):
    """WikiTool 基础异常"""
    pass

class WikiNotFoundError(WikiToolError):
    """Wiki 页面不存在"""
    pass

class WikiValidationError(WikiToolError):
    """参数校验失败"""
    pass

class WikiConnectionError(WikiToolError):
    """Wiki 连接失败"""
    pass

# 错误码规范
WIKI_ERROR_CODES = {
    "WIKI_NOT_FOUND": 404,
    "WIKI_VALIDATION_ERROR": 400,
    "WIKI_DUPLICATE": 409,
    "WIKI_CONNECTION_ERROR": 503,
    "WIKI_INTERNAL_ERROR": 500,
}
```

### 4.5 边界情况处理

| 场景 | 处理方式 |
|------|----------|
| 因子名已存在 | 返回 409 Conflict，或根据 `overwrite` 参数覆盖 |
| Wiki 目录不存在 | 自动创建 `.wiki/` 目录结构 |
| Wiki 锁文件存在 | 等待锁释放（超时 30s） |
| 页面解析失败 | 返回原始 Markdown，附加解析警告 |
| 搜索无结果 | 返回空列表，不抛异常 |
| 关系类型非法 | 抛 WikiValidationError，提示合法关系类型 |

### 4.6 日志与监控

```python
# 日志规范
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 关键日志点
logger.info(f"[WikiTool] store_factor: name={name}, page={page_name}")
logger.warning(f"[WikiTool] Factor already exists, overwriting: {name}")
logger.error(f"[WikiTool] Wiki connection failed: {e}")

# 监控指标（可接入 Prometheus）
METRICS = {
    "wiki_store_total": "因子存储总次数",
    "wiki_store_errors": "因子存储失败次数",
    "wiki_search_total": "搜索总次数",
    "wiki_search_latency_ms": "搜索延迟(ms)",
    "wiki_page_size_bytes": "页面大小(字节)",
}
```

### 4.7 性能考虑

| 操作 | 预期延迟 | 优化策略 |
|------|----------|----------|
| `store_*` | < 100ms | 批量写入缓冲 |
| `get_*` | < 50ms | 文件缓存 |
| `search` | < 200ms | SQLite FTS5 索引 |
| `list_*` | < 100ms | 目录遍历缓存 |

**缓存策略**：
- WikiTool 实例级别缓存（TTL=5min）
- 文件读取使用 mmap
- search 结果缓存（按 query hash）

---

## 五、WikiFactorProxy 扩展

### 5.1 新增页面类型

```python
class PageType(Enum):
    FACTOR = "Factor"
    LOGIC = "Logic"
    STRATEGY = "Strategy"        # 新增
    REPRODUCTION = "Reproduction"  # 新增
```

### 5.2 WikiStrategy 数据类

```python
@dataclass
class WikiStrategy:
    name: str
    strategy_yaml: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    backtest_result: Optional[Dict] = None
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
```

### 5.3 WikiReproduction 数据类

```python
@dataclass
class WikiReproduction:
    report_title: str
    pdf_path: str
    verified_count: int = 0
    failed_count: int = 0
    report_markdown: str = ""
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
```

### 5.4 扩展方法签名

```python
# 新增方法 - WikiFactorProxy

def store_strategy(self, strategy: WikiStrategy) -> str:
    """存储策略到 Wiki"""
    page_name = f"Strategy/{strategy.name}"
    content = self._render_strategy_markdown(strategy)
    self.wiki.write_page(page_name, content)
    strategy.wiki_page_name = page_name
    return page_name

def get_strategy(self, name: str) -> Optional[WikiStrategy]:
    """获取策略"""
    page_name = f"Strategy/{name}"
    page_file = self.wiki.wiki_dir / "Strategy" / f"{name}.md"
    if not page_file.exists():
        return None
    try:
        page_data = self.wiki.read_page(page_name)
    except Exception:
        return None
    return self._parse_strategy_from_page(page_name, page_data)

def list_strategies(
    self,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
) -> List[WikiStrategy]:
    """列举策略"""
    # 类似 list_factors 实现

def _render_strategy_markdown(self, strategy: WikiStrategy) -> str:
    """渲染策略 Markdown"""
    lines = ["---"]
    lines.append(f"type: {PageType.STRATEGY.value}")
    lines.append(f"name: {strategy.name}")
    lines.append(f"category: {strategy.category}")
    lines.append("tags: [" + ", ".join(strategy.tags) + "]")
    lines.append(f"created_at: {strategy.created_at or datetime.now().isoformat()}")
    lines.append("---")
    lines.append(f"## {strategy.name}")
    lines.append(strategy.description)
    lines.append("")
    lines.append("## 策略配置")
    lines.append("```yaml")
    lines.append(strategy.strategy_yaml)
    lines.append("```")
    if strategy.backtest_result:
        lines.append("")
        lines.append("## 回测结果")
        lines.append(f"```json")
        lines.append(json.dumps(strategy.backtest_result, indent=2))
        lines.append("```")
    return "\n".join(lines)
```

---

## 六、工具注册集成

### 6.1 registry.py 修改

```python
from QuantNodes.agent.tools.wiki import WikiTool

def get_default_tools() -> List[BaseTool]:
    return [
        # ... existing tools ...
        WikiTool(wiki_path=config.get("wiki_path")),
    ]

def get_tool_schemas() -> List[Dict]:
    """返回所有工具的 JSON Schema，供 Agent 上下文构建使用"""
    return [
        *BASE_TOOL_SCHEMAS,
        WikiTool.get_schema(),  # WikiTool 参数 Schema
    ]
```

### 6.2 配置文件扩展

```yaml
# quantnodes.yaml 或环境变量
wiki_path: "/path/to/.quantresearch"

# 可选配置
wiki:
  cache_ttl_seconds: 300      # 缓存 TTL
  max_page_size_kb: 1024     # 单页面大小限制
  search_limit_default: 10    # 默认搜索结果数
  auto_init: true             # Wiki 不存在时自动初始化
```

### 6.3 配置加载优先级

```
环境变量 QUANTNODES_WIKI_PATH
    ↓
config.yaml wiki_path
    ↓
命令行 --wiki-path
    ↓
默认值 ~/.quantresearch/wiki
```

---

## 七、实施步骤详解

### 7.1 Step 1: WikiStrategy + WikiReproduction 数据类

**目标**: 在 `research/wiki.py` 中新增两个数据类

```python
# research/wiki.py 新增

@dataclass
class WikiStrategy:
    name: str
    strategy_yaml: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    backtest_result: Optional[Dict] = None
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WikiReproduction:
    report_title: str
    pdf_path: str
    verified_count: int = 0
    failed_count: int = 0
    report_markdown: str = ""
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**验收标准**:
- [ ] `from QuantNodes.research.wiki import WikiStrategy, WikiReproduction` 正常导入
- [ ] 数据类可正常实例化
- [ ] 类型检查通过 mypy

### 7.2 Step 2: WikiFactorProxy 策略方法

**目标**: 实现 `store_strategy`, `get_strategy`, `list_strategies`

```python
# research/wiki.py 新增方法

def store_strategy(self, strategy: WikiStrategy) -> str:
    page_name = f"Strategy/{strategy.name}"
    content = self._render_strategy_markdown(strategy)
    self.wiki.write_page(page_name, content)
    return page_name

def get_strategy(self, name: str) -> Optional[WikiStrategy]:
    page_name = f"Strategy/{name}"
    page_file = self.wiki.wiki_dir / "Strategy" / f"{name}.md"
    if not page_file.exists():
        return None
    try:
        page_data = self.wiki.read_page(page_name)
        return self._parse_strategy_from_page(page_name, page_data)
    except Exception:
        return None

def list_strategies(
    self,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 50,
) -> List[WikiStrategy]:
    # 遍历 Strategy/ 目录，解析并过滤
    ...
```

### 7.3 Step 3: WikiFactorProxy 复现方法

**目标**: 实现 `store_reproduction`, `get_reproduction`

```python
def store_reproduction(self, reproduction: WikiReproduction) -> str:
    page_name = f"Reproduction/{reproduction.report_title}"
    content = self._render_reproduction_markdown(reproduction)
    self.wiki.write_page(page_name, content)
    return page_name

def get_reproduction(self, report_title: str) -> Optional[WikiReproduction]:
    page_name = f"Reproduction/{report_title}"
    ...
```

### 7.4 Step 4-7: WikiTool 实现

**WikiTool 类骨架**:

```python
# agent/tools/wiki.py

from QuantNodes.agent.tools.base import BaseTool
from QuantNodes.research.wiki import (
    WikiFactorProxy,
    WikiFactor,
    WikiLogic,
    WikiStrategy,
    FactorSource,
    FactorCategory,
)
from typing import Any, Dict, List, Optional

class WikiTool(BaseTool):
    name = "wiki"
    description = "QuantNodes Wiki 知识库 - 因子/逻辑/策略的存取与查询"
    read_only = False

    SCHEMAS = {
        "store_factor": STORE_FACTOR_SCHEMA,
        "search_factors": SEARCH_FACTORS_SCHEMA,
        # ...
    }

    def __init__(self, wiki_path: str, **kwargs):
        super().__init__(**kwargs)
        self.proxy = WikiFactorProxy(wiki_path)

    @classmethod
    def get_schema(cls, method: str) -> Dict[str, Any]:
        """返回指定方法的 JSON Schema"""
        return cls.SCHEMAS.get(method, {})

    async def execute(
        self,
        action: str,
        **kwargs
    ) -> Any:
        """统一执行入口，根据 action 分发到具体方法"""
        action_map = {
            "store_factor": self._store_factor,
            "get_factor": self._get_factor,
            "search_factors": self._search_factors,
            "list_factors": self._list_factors,
            "store_logic": self._store_logic,
            "get_logic": self._get_logic,
            "search_logics": self._search_logics,
            "store_strategy": self._store_strategy,
            "get_strategy": self._get_strategy,
            "list_strategies": self._list_strategies,
            "add_relation": self._add_relation,
            "get_neighbors": self._get_neighbors,
            "ping": self._ping,
            "status": self._status,
            "search": self._search,
        }
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        return await action_map[action](**kwargs)

    async def _store_factor(self, **kwargs) -> str:
        """存储因子"""
        factor = WikiFactor(**kwargs)
        return self.proxy.store_factor(factor)

    async def _search_factors(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索因子"""
        factors = self.proxy.search_factors(query, limit=limit)
        return [self._factor_to_dict(f) for f in factors]
```

### 7.5 Step 8-9: 工具注册

**修改 `agent/tools/__init__.py`**:

```python
from QuantNodes.agent.tools.wiki import WikiTool

__all__ = [
    # ... existing exports
    "WikiTool",
]
```

**修改 `agent/tools/registry.py`**:

```python
def get_default_tools(config: Optional[Dict] = None) -> List[BaseTool]:
    config = config or {}
    wiki_path = config.get("wiki_path", os.path.expanduser("~/.quantresearch/wiki"))
    
    return [
        BacktestTool(),
        FactorTool(),
        PipelineTool(),
        SandboxTool(),
        StrategyTool(),
        ConfigBacktestTool(),
        WikiTool(wiki_path=wiki_path),  # 新增
    ]
```

### 7.6 Step 10-11: 测试

**单元测试 `tests/agent/test_wiki_tool.py`**:

```python
import pytest
from unittest.mock import MagicMock, patch
from QuantNodes.agent.tools.wiki import WikiTool
from QuantNodes.research.wiki import WikiFactor, FactorSource, FactorCategory

@pytest.fixture
def wiki_tool(mock_wiki):
    with patch("QuantNodes.research.wiki.create_wiki", return_value=mock_wiki):
        yield WikiTool(wiki_path="/tmp/test_wiki")

@pytest.fixture
def mock_wiki():
    wiki = MagicMock()
    wiki.write_page = MagicMock(return_value="Factor/test")
    wiki.read_page = MagicMock(return_value={"content": MOCK_FACTOR_CONTENT})
    wiki.search = MagicMock(return_value=[{"page_name": "Factor/test"}])
    return wiki

class TestWikiToolStoreFactor:
    def test_store_factor_success(self, wiki_tool):
        result = wiki_tool.execute_sync(
            action="store_factor",
            name="test_factor",
            formula="close / open - 1",
            source="manual",
            category="momentum",
        )
        assert result == "Factor/test_factor"

    def test_store_factor_missing_required(self, wiki_tool):
        with pytest.raises(TypeError):
            wiki_tool.execute_sync(action="store_factor", name="test")

    def test_store_factor_invalid_category(self, wiki_tool):
        with pytest.raises(ValueError):
            wiki_tool.execute_sync(
                action="store_factor",
                name="test",
                formula="close",
                source="manual",
                category="invalid_category",
            )
```

**集成测试 `tests/research/test_wiki_proxy.py`**:

```python
import pytest
import tempfile
import shutil
from QuantNodes.research.wiki import WikiFactorProxy, WikiFactor

class TestWikiProxyIntegration:
    @pytest.fixture
    def temp_wiki(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_store_and_get_factor(self, temp_wiki):
        proxy = WikiFactorProxy(temp_wiki)
        factor = WikiFactor(
            name="test_factor",
            formula="close / open - 1",
            source=FactorSource.MANUAL,
            category=FactorCategory.MOMENTUM,
        )
        page_name = proxy.store_factor(factor)
        assert page_name == "Factor/test_factor"

        retrieved = proxy.get_factor("test_factor")
        assert retrieved is not None
        assert retrieved.name == "test_factor"
```

---

## 八、测试策略

### 8.1 单元测试

- Mock `llmwikify.Wiki`
- 测试 WikiTool 每个方法
- 测试参数验证、错误处理

### 8.2 集成测试

- 使用临时目录创建真实 Wiki
- 测试完整数据流：store → get → search → delete

### 8.3 Mock 策略

```python
@pytest.fixture
def mock_wiki(mocker):
    wiki = mocker.MagicMock()
    wiki.write_page = mocker.MagicMock(return_value="Factor/test")
    wiki.read_page = mocker.MagicMock(return_value={"content": "..."})
    wiki.search = mocker.MagicMock(return_value=[])
    return wiki
```

---

## 九、与现有模块的关系

### 9.1 复用关系

| 模块 | 复用方式 | 位置 |
|------|----------|------|
| WikiFactorProxy | 组合（has-a） | `WikiTool.proxy` |
| WikiFactor/WikiLogic | 直接使用 | `WikiTool` 方法参数/返回值 |
| BaseTool | 继承（is-a） | `WikiTool(BaseTool)` |
| ToolRegistry | 注册 | `registry.py` |

### 9.2 3A/3B/3C 集成

```
Feature 3A (WikiFactorProxy) ← 核心依赖
        ↑
Feature 3B (ReportReproducer) → 直接调用 proxy.store_factor/logic()
Feature 3C (AutoResearcher)   → 直接调用 proxy.store_factor()
        ↑
Agent Phase 3 (WikiTool)      → 封装 proxy 为 Agent 工具
```

---

## 十、预期成果

### 10.1 完成后状态

- Agent 可通过自然语言调用 Wiki 工具
- 策略/因子/研报知识自动沉淀到 Wiki
- 完整的工具注册 + 工具描述 + 参数 schema

### 10.2 使用示例

```python
# Agent 内部
result = await agent.chat(
    "研究一个双均线动量因子，IC 大于 0.03 就存入因子库"
)

# Agent 内部调用链
# 1. WikiTool.store_factor(
#      name="dual_ma_momentum",
#      formula="ma_20 / ma_60 - 1",
#      ic_mean=0.042,
#      source="auto_research",
#      category="momentum",
#    )
# 2. WikiFactorProxy → wiki.write_page("Factor/dual_ma_momentum", ...)
```

---

## 十一、后续扩展点

### 11.1 MCP 协议迁移

当需要 Agent 与 Wiki 解耦时，可替换为：

```python
# agent/tools/wiki_mcp.py
class WikiMCPTool(BaseTool):
    """通过 MCP 协议调用远程 Wiki"""
    
    def __init__(self, mcp_server_url: str):
        self.client = MCPClient(mcp_server_url)
```

### 11.2 技能系统集成（Phase 4）

WikiTool 可被技能系统进一步封装：

```
Skills/strategy_design/momentum.py
    ↓
调用 WikiTool.store_strategy()
```

---

**文档版本**: v1.1  
**最后更新**: 2026-05-07
