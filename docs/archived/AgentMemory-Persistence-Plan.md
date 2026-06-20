# Agent Memory Persistence 设计文档

> **版本**: v2.5.0
> **状态**: 已实施
> **创建日期**: 2026-05-10
> **最后更新**: 2026-05-10（合并所有确认决策）

---

## 1. 现状分析

### 1.1 当前架构

```
AgentLoop / AgentService
    ├── SessionManager          ← 磁盘持久化 (.quant_agent/sessions/*.json)
    ├── MemoryStore             ← memory.md (长期记忆) + history.jsonl (审计日志)
    ├── DreamStore              ← dreams.jsonl (洞察记录)
    └── DreamEngine             ← 异常洞察生成 + Skill 分发
```

### 1.2 已识别问题

| # | 问题 | 严重度 | 位置 |
|---|------|--------|------|
| 1 | **Session 双存储** — `AgentService._sessions` (内存 dict) 与 `SessionManager` (磁盘) 并存，API 的 `list_sessions/get_history/delete_session` 只操作内存 dict，重启丢失 | CRITICAL | `api/services/agent_service.py:20,64-83,163-198` |
| 2 | **history.jsonl 只写不读** — `append_history()` 写入但无任何消费者，纯审计日志 | MODERATE | `QuantNodes/agent/core/memory.py:129-133` |
| 3 | **memory.md 无自动丰富** — 仅通过 `dream_store.inject_to_memory()` 更新，Agent 无法从对话中自动学习 | MODERATE | `QuantNodes/agent/core/memory.py:117-122` |
| 4 | **memory.md 无结构化索引** — 单个大文件，无分类，无加载控制 | LOW | `QuantNodes/agent/templates/agent/system_prompt.md` |
| 5 | **Dream 与 Agent 对话脱节** — `analyze_factor/analyze_strategy` 需要外部调用，Agent 自身对话中的洞察不被捕获 | MODERATE | `QuantNodes/agent/core/dream.py` |
| 6 | **上下文截断丢失信息** — `truncate_history` 丢弃旧消息、`microcompact` 截断工具输出，有价值洞察随截断永久丢失 | MODERATE | `autocompact.py`, `runner.py:68-69,178-179,318` |

---

## 2. 设计目标

1. **单一 Session 存储源** — 消除 `AgentService._sessions`，全部委托给 `SessionManager`
2. **对话历史可检索** — `history.jsonl` 增强为可查询的对话摘要存储
3. **Agent 自主记忆** — 借鉴 Claude Code `MEMORY.md` 模式，Agent 在对话中决定记什么
4. **Dream 与对话闭环** — Agent 对话中的洞察自动进入 Dream 系统
5. **渐进式加载** — System prompt 按需注入相关记忆，而非全量注入
6. **截断前保留洞察** — 上下文压缩时提取有价值信息，防止随截断永久丢失

---

## 3. Phase A: 统一 Session 存储

### 3.1 目标

消除 `AgentService._sessions` 双存储问题，所有 Session 操作委托给 `SessionManager`。

### 3.2 架构变更

```
API Layer (agent_service.py)
    └── SessionManager (唯一存储)
         └── .quant_agent/sessions/*.json (磁盘)

Agent Loop (loop.py)
    └── SessionManager (同一实例)
```

### 3.3 具体改动

#### `api/services/agent_service.py`

**删除**:
```python
# 删除这行 (line 20)
self._sessions: dict[str, list] = {}
```

**修改 `_get_agent()`**: 注入共享 `SessionManager` 实例

```python
def _get_agent(self, config: dict = None) -> Agent:
    if self._agent is None:
        if config is None:
            config = self._load_settings_config()
        self._agent = Agent(workspace=self.workspace, config=config)
    return self._agent
```

**新增 property**:
```python
@property
def session_manager(self) -> SessionManager:
    """获取 Agent 内部的 SessionManager"""
    agent = self._get_agent()
    return agent.loop.session_manager
```

**修改 `send_message()`**:
```python
async def send_message(self, content, session_id="default", config=None):
    agent = self._get_agent(config)
    session = self.session_manager.get_session(session_id)
    session.add_message("user", content)
    response = await agent.run(content, session_id=session_id)
    self.session_manager.save_session(session)
    # ... return dict
```

**修改 `stream_message()`** — 同理，用 SessionManager 替代 `_sessions` 操作。

**修改 `list_sessions()`**:
```python
def list_sessions(self) -> list[dict]:
    sessions = []
    for sid in self.session_manager.list_sessions():
        session = self.session_manager.get_session(sid)
        first_msg = session.messages[0] if session.messages else None
        last_msg = session.messages[-1] if session.messages else None
        sessions.append({
            "session_id": sid,
            "message_count": len(session.messages),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "first_message": first_msg.get("content", "")[:100] if first_msg else "",
            "last_message": last_msg.get("content", "")[:100] if last_msg else "",
        })
    return sessions
```

**修改 `get_history()`**:
```python
def get_history(self, session_id: str) -> list:
    session = self.session_manager.get_session(session_id)
    return [{"role": m["role"], "content": m["content"]} for m in session.messages]
```

**修改 `delete_session()`**:
```python
def delete_session(self, session_id: str) -> bool:
    return self.session_manager.delete_session(session_id)
```

**修改 `clear_history()`**:
```python
def clear_history(self, session_id: str) -> None:
    self.session_manager.delete_session(session_id)
```

### 3.4 SessionManager 增强

#### `QuantNodes/agent/session/manager.py`

新增 `get_session_info()` 方法，供 API 层 `list_sessions` 使用：

```python
def get_session_info(self, session_id: str) -> dict | None:
    """获取会话元数据（不加载全部消息）"""
    session_file = self.workspace / f"{session_id}.json"
    if not session_file.exists():
        return None
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "session_id": session_id,
        "message_count": len(data.get("messages", [])),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }
```

新增 `list_sessions_with_info()` 方法：
```python
def list_sessions_with_info(self) -> List[dict]:
    """列出所有会话及其元数据"""
    result = []
    for sid in self.list_sessions():
        info = self.get_session_info(sid)
        if info:
            result.append(info)
    return sorted(result, key=lambda x: x.get("updated_at", ""), reverse=True)
```

### 3.5 API 路由适配

`api/routers/chat.py` 中已有调用 `agent_service.list_sessions()` / `get_history()` 等方法，由于我们修改的是 `AgentService` 内部实现（接口不变），路由层无需改动。

### 3.6 文件变更清单

| 文件 | 操作 |
|------|------|
| `api/services/agent_service.py` | 修改 — 删除 `_sessions`，委托 SessionManager |
| `QuantNodes/agent/session/manager.py` | 修改 — 新增 `get_session_info()`, `list_sessions_with_info()` |

---

## 4. Phase B: history.jsonl 增强

### 4.1 目标

将 `history.jsonl` 从审计日志升级为可查询的对话摘要存储。

### 4.2 数据格式

每条记录扩展为：

```json
{
  "session_key": "default",
  "user": "帮我分析一下沪深300的IC分布",
  "assistant": "根据分析结果，沪深300的IC均值为0.032，ICIR为0.67...",
  "tools_used": ["factor", "wiki"],
  "insights": ["IC均值0.032，ICIR 0.67，因子有预测能力"],
  "timestamp": "2026-05-10T15:30:00"
}
```

### 4.3 具体改动

#### `QuantNodes/agent/core/memory.py`

**修改 `append_history()`**:

```python
def append_history(
    self,
    entry: Dict[str, Any],
    tools_used: List[str] = None,
    insights: List[str] = None,
) -> None:
    """追加对话摘要到历史"""
    entry["timestamp"] = datetime.now().isoformat()
    if tools_used:
        entry["tools_used"] = tools_used
    if insights:
        entry["insights"] = insights
    with open(self._history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**新增 `get_recent_history()`**:

```python
def get_recent_history(
    self, limit: int = 50, session_key: str = None
) -> List[Dict[str, Any]]:
    """读取最近的历史摘要"""
    if not self._history_file.exists():
        return []
    entries = []
    with open(self._history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if session_key and data.get("session_key") != session_key:
                    continue
                entries.append(data)
            except json.JSONDecodeError:
                continue
    return entries[-limit:]
```

**新增 `search_history()`**:

```python
def search_history(
    self, query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """按关键词搜索历史摘要"""
    if not self._history_file.exists():
        return []
    results = []
    with open(self._history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                user_text = data.get("user", "")
                assistant_text = data.get("assistant", "")
                if query in user_text or query in assistant_text:
                    results.append(data)
            except json.JSONDecodeError:
                continue
    return results[-limit:]
```

#### `QuantNodes/agent/core/loop.py`

**修改 `_process_message()` 和 `chat()` 中的 `append_history` 调用**:

```python
# 之前
self.memory.append_history({
    "session_key": msg.session_key,
    "user": msg.content[:200],
    "assistant": (result.final_content or "")[:200],
})

# 之后
self.memory.append_history(
    {
        "session_key": msg.session_key,
        "user": msg.content[:500],
        "assistant": (result.final_content or "")[:500],
    },
    tools_used=[tc.get("name", "") for tc in result.tool_calls] if hasattr(result, 'tool_calls') else [],
)
```

### 4.4 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/memory.py` | 修改 — 增强 `append_history`，新增 `get_recent_history`, `search_history` |
| `QuantNodes/agent/core/loop.py` | 修改 — 更新 `append_history` 调用参数 |

---

## 5. Phase C: Memory 自动丰富 (Claude Code 风格)

### 5.1 目标

借鉴 Claude Code `MEMORY.md` 模式：Agent 自身在对话中决定记什么，无需额外 LLM 调用。采用 `MEMORY.md` 索引 + 主题文件按需加载。

### 5.2 设计理念

Claude Code 的记忆系统核心思想：
- **MEMORY.md** 是索引文件（≤200行/25KB），列出所有主题及其一句话摘要
- **topic-*.md** 是详细内容文件，按需加载
- **Agent 自主写入** — 在对话中直接通过 `file_ops` 工具更新记忆文件
- **System prompt 注入索引** — 每次对话开始时注入 MEMORY.md 索引，Agent 按需读取详细文件

### 5.3 业界调研摘要

| 框架 | 写入触发 | 存储格式 | 加载策略 | 记忆验证 |
|------|---------|---------|---------|---------|
| **Claude Code** | 自动(模型判断) + 用户手动 | Markdown 索引 + topic 文件 | 索引每次加载(200行上限) + topic 按需读取 | 无(模型自主判断) |
| **Cursor** | 纯用户手动 | Markdown rules 文件 | 按配置模式注入 | 无 |
| **ChatGPT** | 自动 + 用户明确 | 服务端存储 | 系统自动注入 | 隐私偏向(避免敏感信息) |
| **LangChain** | 自动checkpoint + 开发者编码 | JSON in DB | checkpoint全量 + store按需搜索 | `@after_model` 中间件 |
| **AutoGPT** | 自动(Memory Creator Agent) | SQLite/JSON | 每轮注入 | LLM-based 过滤器 |

**结论**: 没有系统有真正的「验证」机制。Claude Code 模式（Agent 自主判断 + 索引上限控制 + 用户可通过 file_ops 审查/删除）已足够。**暂不实现 memory_guard**，如未来 Agent 写入垃圾记忆再加过滤。

### 5.4 目录结构

```
.quant_agent/memory/
├── memory.md              ← 废弃（直接删除内容，迁移到 topic 文件）
├── MEMORY.md              ← 新增：记忆索引（Agent 维护）
├── topic-factor.md        ← 因子相关记忆
├── topic-strategy.md      ← 策略相关记忆
├── topic-backtest.md      ← 回测相关记忆
├── topic-market.md        ← 市场观察
├── topic-user.md          ← 用户偏好
├── topic-project.md       ← 项目结构
├── history.jsonl          ← 增强后的对话摘要
└── dream-insights.md      ← Dream 洞察主题文件（Phase D）
```

### 5.5 MEMORY.md 格式

```markdown
# Memory Index

## Factors
- 因子IC分析经验：沪深300因子IC均值0.03+为有效，ICIR>0.5为优秀 (topic-factor.md:ic-analysis)

## Strategies
- 多因子策略回测：年化>15%且最大回撤<20%为合格线 (topic-strategy.md:performance-benchmarks)

## Market
- 市场状态观察：最近一次检查是2026-05-10，A股震荡偏强 (topic-market.md:recent-state)

## User Preferences
- 偏好使用QMT作为回测引擎 (topic-user.md:tools)

## Project
- 项目结构：QuantNodes v2.5.0，Agent系统15个工具 (topic-project.md:architecture)
```

### 5.6 具体改动

#### `QuantNodes/agent/templates/agent/system_prompt.md`

在末尾添加记忆系统指引：

```markdown
## 记忆系统

你有一个持久化的记忆系统，位于 `.quant_agent/memory/` 目录。

### 记忆规则
1. **MEMORY.md** 是记忆索引，列出所有主题及其一句话摘要
2. **topic-{name}.md** 是详细内容文件
3. 在对话中如果学到了值得记住的信息，使用 `file_ops` 工具写入对应主题文件
4. 写入详细文件后，同步更新 MEMORY.md 索引
5. MEMORY.md 保持简洁（≤200行），每条索引一行
6. 每次对话开始时你会看到记忆索引，可按需用 `file_ops` 读取详细文件

### 记忆分类
- `topic-factor.md` — 因子分析经验、IC/IR数据
- `topic-strategy.md` — 策略性能、回测结果
- `topic-backtest.md` — 回测配置、参数经验
- `topic-market.md` — 市场观察、行情记录
- `topic-user.md` — 用户偏好、工作习惯
- `topic-project.md` — 项目结构、代码架构
- `topic-general.md` — 其他知识

### 何时记忆
- 用户明确要求记住某事
- 发现了重要的因子/策略规律
- 回测结果有参考价值
- 用户表达了偏好或习惯
- 项目结构发生变化
```

#### `QuantNodes/agent/core/memory.py`

**新增 `MemoryManager` 类**:

```python
class MemoryManager:
    """Claude Code 风格的记忆管理器"""

    INDEX_FILE = "MEMORY.md"
    MAX_INDEX_LINES = 200
    TOPIC_PREFIX = "topic-"

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace) / "memory"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._index_file = self.workspace / self.INDEX_FILE

    def read_index(self) -> str:
        """读取记忆索引"""
        if self._index_file.exists():
            return self._index_file.read_text(encoding="utf-8")
        return "# Memory Index\n"

    def write_index(self, content: str) -> None:
        """写入记忆索引（Agent 通过 file_ops 调用）"""
        self._index_file.write_text(content, encoding="utf-8")

    def read_topic(self, topic: str) -> str:
        """读取主题文件"""
        topic_file = self.workspace / f"{self.TOPIC_PREFIX}{topic}.md"
        if topic_file.exists():
            return topic_file.read_text(encoding="utf-8")
        return ""

    def write_topic(self, topic: str, content: str) -> None:
        """写入主题文件"""
        topic_file = self.workspace / f"{self.TOPIC_PREFIX}{topic}.md"
        topic_file.write_text(content, encoding="utf-8")

    def list_topics(self) -> List[str]:
        """列出所有主题"""
        topics = []
        for f in self.workspace.glob(f"{self.TOPIC_PREFIX}*.md"):
            topic = f.stem.removeprefix(self.TOPIC_PREFIX)
            topics.append(topic)
        return sorted(topics)

    def get_memory_context(self) -> str:
        """获取注入 System prompt 的记忆上下文（仅索引）"""
        index = self.read_index()
        if index.strip() == "# Memory Index":
            return ""
        return f"## 记忆索引\n\n{index}\n\n如需查看详细内容，使用 file_ops 读取对应 topic 文件。"
```

#### `QuantNodes/agent/core/loop.py`

**修改记忆注入逻辑**:

```python
# 之前
memory_ctx = self.memory.get_memory_context()

# 之后 — 使用 MemoryManager
memory_ctx = self.memory_manager.get_memory_context()
```

**在 `__init__` 中初始化 MemoryManager**:

```python
from .memory import MemoryStore, MemoryManager

# __init__ 中
self.memory = MemoryStore(self.workspace)
self.memory_manager = MemoryManager(self.workspace)
```

### 5.7 迁移策略

1. Phase C 实施时，直接删除 `memory.md` 内容，迁移到对应的 topic 文件
2. 生成 `MEMORY.md` 索引

### 5.8 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/memory.py` | 修改 — 新增 `MemoryManager` 类 |
| `QuantNodes/agent/core/loop.py` | 修改 — 初始化 `MemoryManager`，注入索引上下文 |
| `QuantNodes/agent/templates/agent/system_prompt.md` | 修改 — 添加记忆系统指引 |

---

## 6. Phase D: Dream 集成

### 6.1 目标

Agent 对话中的洞察自动进入 Dream 系统，形成「对话 → 洞察 → 记忆」闭环。

### 6.2 设计决策（已确认）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| DreamHook 触发位置 | **AgentLoop runner 完成后**（非 `after_iteration`） | `after_iteration` 仅在工具执行后触发，无法捕获最终文本响应；AgentLoop 层可直接访问 `session.messages` 计算轮数 |
| 激活门槛 | **可配置 `min_rounds_before_activate`**（默认 5） | Agent 需要足够对话历史才能做出有价值的洞察判断 |
| 分析粒度 | **整体分析**（一轮对话一个 Dream 摘要） | 逐轮分析产生过多低质量 Dream |
| 新洞察检查 | **分析前先检查是否包含新洞察**（`should_analyze_conversation()`） | 避免对无信息量的对话生成 Dream |
| 记忆验证 | **暂不实现 memory_guard** | Claude Code 模式（Agent 自主判断 + 索引上限控制）已足够 |

### 6.3 架构

```
Agent 对话
    │
    ▼
AgentLoop._process_message() / chat() / chat_stream()
    │ runner 完成后
    ├── 检查 round_count >= min_rounds_before_activate
    ├── should_analyze_conversation() 关键词检查
    ▼
DreamEngine.analyze_conversation()
    │
    ▼
DreamStore.save_dream() → dreams.jsonl
    │
    ▼
dream_insights.md (主题文件，通过 MemoryManager 写入)
    │
    ▼
MEMORY.md 索引更新
```

### 6.4 DreamConfig 新增字段

```python
@dataclass
class DreamConfig:
    # ... 现有字段 ...
    min_rounds_before_activate: int = 5   # 新增：激活前最少对话轮数
    analysis_keywords: List[str] = field(default_factory=lambda: [
        "IC", "ICIR", "因子", "factor", "回测", "策略", "收益", "夏普",
        "回撤", "胜率", "年化", "记住", "以后", "每次", "偏好",
    ])  # 新增：对话洞察检测关键词
```

### 6.5 具体改动

#### `QuantNodes/agent/core/dream.py`

**新增 `analyze_conversation()`**:

```python
async def analyze_conversation(
    self,
    user_message: str,
    assistant_response: str,
    tools_used: List[str] = None,
) -> Optional[Dream]:
    """分析对话并生成洞察 Dream（仅在包含新洞察时生成）"""
    insights = []
    confidence = 0.6

    # 检测因子分析相关
    if any(kw in user_message for kw in ["IC", "ICIR", "因子", "factor"]):
        if any(kw in assistant_response for kw in ["IC均值", "ICIR", "因子有效", "因子无效"]):
            insights.append("对话涉及因子分析")
            confidence += 0.1

    # 检测策略回测相关
    if any(kw in user_message for kw in ["回测", "策略", "收益", "夏普"]):
        if any(kw in assistant_response for kw in ["年化", "夏普", "回撤", "胜率"]):
            insights.append("对话涉及策略回测")
            confidence += 0.1

    # 检测用户偏好
    if any(kw in user_message for kw in ["记住", "以后", "每次", "偏好"]):
        insights.append("用户表达了偏好或要求记忆")
        confidence += 0.15

    if not insights:
        return None  # 无新洞察，不生成 Dream

    confidence = min(max(confidence, 0.3), 1.0)
    return await self.generate_dream(
        dream_type="conversation_insight",
        content=f"对话摘要: {user_message[:100]}",
        source="conversation",
        insights=insights,
        confidence=confidence,
        tags=["conversation", "auto"],
    )
```

**新增 `should_analyze_conversation()`**:

```python
def should_analyze_conversation(
    self, user_message: str, assistant_response: str
) -> bool:
    """快速检查对话是否可能包含值得提取的洞察"""
    combined = user_message + assistant_response
    return any(kw in combined for kw in self.config.analysis_keywords)
```

#### `QuantNodes/agent/core/loop.py`

**不在 Hook 中触发 Dream，改为在 AgentLoop 层直接调用**:

```python
async def _process_message(self, msg: InboundMessage) -> None:
    session = self.session_manager.get_session(msg.session_key)

    # ... 现有逻辑：构建消息、执行 runner ...

    result = await self.runner.run(spec)

    # Phase D: 对话洞察分析（仅在积累足够后触发）
    round_count = len([m for m in session.messages if m.get("role") == "user"])
    if round_count >= self.dream_engine.config.min_rounds_before_activate:
        if self.dream_engine.should_analyze_conversation(msg.content, result.final_content or ""):
            dream = await self.dream_engine.analyze_conversation(
                user_message=msg.content,
                assistant_response=result.final_content or "",
                tools_used=result.tools_used,
            )
            if dream and dream.confidence >= self.dream_engine.config.min_confidence:
                await self._update_dream_topic(dream)

    # ... 现有逻辑：保存 Session、append_history ...
```

**新增 `_update_dream_topic()` 辅助方法**:

```python
async def _update_dream_topic(self, dream) -> None:
    """将 Dream 洞察写入 dream-insights.md 主题文件"""
    try:
        existing = self.memory_manager.read_topic("dream-insights")
        timestamp = dream.timestamp[:10]
        new_entry = f"\n### {timestamp} - {dream.type}\n"
        new_entry += f"- {dream.content}\n"
        for insight in dream.insights:
            new_entry += f"  - {insight}\n"

        updated = existing + new_entry
        self.memory_manager.write_topic("dream-insights", updated)

        # 更新 MEMORY.md 索引
        index = self.memory_manager.read_index()
        if "topic-dream-insights.md" not in index:
            index += "\n## Insights\n- Dream洞察记录 (topic-dream-insights.md)\n"
            self.memory_manager.write_index(index)
    except Exception:
        pass
```

`chat()` 和 `chat_stream()` 中同理添加相同逻辑。

### 6.6 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/dream.py` | 修改 — 新增 `analyze_conversation()`, `should_analyze_conversation()`，`DreamConfig` 新增字段 |
| `QuantNodes/agent/core/loop.py` | 修改 — 三个方法中添加 Dream 分析逻辑，新增 `_update_dream_topic()` |

---

## 7. Phase E: AgentLoop 集成

### 7.1 目标

将 Phase A-D 的所有变更集成到 AgentLoop 主循环中，确保端到端工作。

### 7.2 集成点

#### 7.2.1 AgentLoop `__init__` 初始化顺序

```python
class AgentLoop:
    def __init__(self, ...):
        # ... 现有初始化 ...

        # Phase A: SessionManager（已有）
        self.session_manager = session_manager or SessionManager(self.workspace)

        # Phase B: MemoryStore（已有，增强 append_history）
        self.memory = MemoryStore(self.workspace)

        # Phase C: MemoryManager（新增）
        self.memory_manager = MemoryManager(self.workspace)

        # Phase D: DreamEngine（新增）
        dream_store = DreamStore(self.workspace)
        self.dream_engine = DreamEngine(dream_store)

        # Phase F: 截断分析队列（新增）
        self._pending_dream_analysis: List[Dict] = []
        self._compaction_counter: int = 0
```

#### 7.2.2 `_process_message()` / `chat()` / `chat_stream()` 统一改造

三个方法遵循相同模式（以 `_process_message` 为例）：

```python
async def _process_message(self, msg: InboundMessage) -> None:
    session = self.session_manager.get_session(msg.session_key)

    # 构建消息
    history = [m for m in session.messages if m.get("role") in ("user", "assistant", "tool")]

    # Phase F: 截断前捕获被丢弃的消息
    history, dropped = truncate_history(history, max_messages=20)
    if dropped:
        self._pending_dream_analysis.extend(dropped)

    messages = self.context_builder.build_messages(
        history=history, current_message=msg.content,
        media=msg.media, channel=msg.channel, chat_id=msg.chat_id,
    )

    # Phase C: 注入记忆索引
    memory_ctx = self.memory_manager.get_memory_context()
    if memory_ctx and messages and messages[0].get("role") == "system":
        messages[0]["content"] += f"\n\n{memory_ctx}"

    # Phase B: 注入最近对话摘要
    recent = self.memory.get_recent_history(limit=5, session_key=msg.session_key)
    if recent:
        summary_lines = [f"- {r['user'][:80]} → {(r['assistant'] or '')[:80]}" for r in recent[-3:]]
        summary = "\n".join(summary_lines)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n## 最近对话\n{summary}"

    # 执行
    spec = AgentRunSpec(
        initial_messages=messages, tools=self.tool_registry,
        model=self.model, max_iterations=self.max_iterations,
    )
    result = await self.runner.run(spec)

    # Phase F: runner 完成后处理待分析截断消息
    if self._pending_dream_analysis:
        await self._process_compaction_dreams(msg.session_key)

    # Phase D: 对话洞察分析（仅在积累足够后触发）
    round_count = len([m for m in session.messages if m.get("role") == "user"])
    if round_count >= self.dream_engine.config.min_rounds_before_activate:
        if self.dream_engine.should_analyze_conversation(msg.content, result.final_content or ""):
            dream = await self.dream_engine.analyze_conversation(
                user_message=msg.content,
                assistant_response=result.final_content or "",
                tools_used=result.tools_used,
            )
            if dream and dream.confidence >= self.dream_engine.config.min_confidence:
                await self._update_dream_topic(dream)

    # 保存 Session
    session.add_message("user", msg.content)
    if result.final_content:
        session.add_message("assistant", result.final_content)
    self.session_manager.save_session(session)

    # Phase B: 保存对话摘要
    self.memory.append_history(
        {"session_key": msg.session_key, "user": msg.content[:500], "assistant": (result.final_content or "")[:500]},
        tools_used=[tc.get("name", "") for tc in (result.tool_calls or [])],
    )

    # 发送响应
    response = OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result.final_content or "")
    await self.bus.publish_outbound(response)
```

#### 7.2.3 Agent 入口 `__init__.py` 同步更新

`QuantNodes/agent/__init__.py` 中的 `Agent` 类需要同步 `loop` 的 `memory_manager` 和 `dream_engine` 引用，以便 API 层访问。

### 7.3 完整数据流

```
用户消息
    │
    ▼
AgentLoop._process_message() / chat() / chat_stream()
    │
    ├── 1. SessionManager.get_session()              ← 读取历史
    ├── 2. truncate_history() → (保留, 丢弃)          ← Phase F: 捕获丢弃消息
    ├── 3. build_messages()                           ← 构建 prompt
    ├── 4. memory_manager.get_memory_context()        ← Phase C: 注入索引
    ├── 5. memory.get_recent_history()                ← Phase B: 注入近期摘要
    │
    ▼
AgentRunner.run() / run_stream()
    │
    ├── LLM 调用
    ├── 工具执行（truncate_text 截断工具输出）
    │
    ▼
AgentLoop 后处理
    │
    ├── _process_compaction_dreams()                  ← Phase F: 截断消息 Dream 分析
    ├── DreamEngine.analyze_conversation()            ← Phase D: 对话洞察分析
    ├── session.add_message() × 2
    ├── session_manager.save_session()                ← Phase A: 磁盘持久化
    ├── memory.append_history()                       ← Phase B: 对话摘要
    │
    ▼
Agent 通过 file_ops 工具写入记忆（Phase C）
    │
    ├── file_ops("write", ".quant_agent/memory/topic-xxx.md")
    └── file_ops("edit", ".quant_agent/memory/MEMORY.md")
```

### 7.4 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/loop.py` | 修改 — 初始化所有组件，统一改造三个方法 |
| `QuantNodes/agent/__init__.py` | 修改 — 同步 memory_manager 引用 |

---

## 8. Phase F: Compaction-Dream 集成

### 8.1 目标

上下文压缩（`truncate_history`）丢弃旧消息时，提取有价值洞察生成 Dream，防止信息永久丢失。

### 8.2 设计决策（已确认）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 实现方案 | **方案 C: 返回值修改** | 最简单，不改 Hook 接口，truncate_history 返回 `(保留, 丢弃)` 元组 |
| Dream 触发层级 | **仅 AgentLoop 层**（`max=20`） | AgentRunner 内部截断（`max=30`）的消息来自同一次 runner 调用，会在 runner 结束后通过 Session 保存，下次 AgentLoop 调用时如果再被截断才会进入 Dream 分析 |
| 分析粒度 | **整体分析** | 合并所有被丢弃消息为一个 Dream 摘要 |
| 频率控制 | **每 N 次截断才分析一次**（`compaction_dream_interval`，默认 5） | 截断可能每次调用都发生，需限制 Dream 提取频率 |

### 8.3 工具输出截断分析

对 `microcompact` 和 `truncate_text` 截断的工具输出进行价值评估：

| 工具 | 截断风险 | Dream 价值 | 截断部分包含的新信息 |
|------|---------|-----------|-------------------|
| **`factor`** | 极高 | **极高** | IC 时序数据尾部（最近日期的 IC 值），趋势/regime 信息 |
| **`web_fetch`** | 几乎必然 | **极高** | 文章主体内容（只看到前100词） |
| **`wiki`** (list/search) | 高 | 高 | 因子目录完整性、知识图谱结构 |
| `file_ops` (read_file) | 极高 | 中高 | 代码文件尾部结构 |
| `code_search` | 极高 | 中 | 搜索结果尾部匹配 |
| `git_ops` (git_diff) | 极高 | 中 | 多文件变更的尾部 |
| 其他 | 低-中 | 低 | — |

**结论**: `factor` 和 `web_fetch` 的截断输出值得做 Dream 分析，但工具级 Dream 提取需要 tool-specific extractors，复杂度较高。Phase F 优先实现**消息级截断 Dream**（truncate_history 丢弃的旧消息），工具级截断作为后续优化。

### 8.4 具体改动

#### `QuantNodes/agent/core/autocompact.py`

**修改 `truncate_history()` 返回值**:

```python
def truncate_history(
    messages: List[Dict[str, Any]],
    max_messages: int = 20,
    keep_system: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """裁剪历史消息，返回 (保留的消息, 被丢弃的消息)"""
    if len(messages) <= max_messages:
        return messages, []

    system_msgs = []
    other_msgs = []

    for msg in messages:
        if keep_system and msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            other_msgs.append(msg)

    dropped = []
    if len(other_msgs) > max_messages:
        dropped = other_msgs[:-max_messages]
        other_msgs = other_msgs[-max_messages:]

    return system_msgs + other_msgs, dropped
```

#### `QuantNodes/agent/core/runner.py`

**忽略 dropped 返回值**（AgentRunner 内部不参与 Dream 分析）:

```python
# runner.py:68 和 runner.py:178
messages, _ = truncate_history(messages, max_messages=30)
```

#### `QuantNodes/agent/core/loop.py`

**新增截断 Dream 处理逻辑**:

```python
async def _process_compaction_dreams(self, session_key: str) -> None:
    """分析被截断的消息，提取洞察生成 Dream"""
    if not self._pending_dream_analysis:
        return

    # 频率控制：每 N 次截断才分析一次
    self._compaction_counter += 1
    if self._compaction_counter < self.dream_engine.config.compaction_dream_interval:
        self._pending_dream_analysis.clear()
        return
    self._compaction_counter = 0

    # 整体分析：将所有被丢弃的消息合并为一个摘要
    dropped_text = "\n".join([
        f"[{m.get('role', '?')}]: {m.get('content', '')[:200]}"
        for m in self._pending_dream_analysis[-10:]  # 最多取最后10条
    ])

    # 快速检查是否包含可提取的洞察
    if not self.dream_engine.should_analyze_conversation(dropped_text, ""):
        self._pending_dream_analysis.clear()
        return

    # 生成 Dream
    dream = await self.dream_engine.generate_dream(
        dream_type="compaction_insight",
        content=f"被截断的对话历史摘要 ({len(self._pending_dream_analysis)} 条消息)",
        source="compaction",
        insights=[f"截断消息中检测到关键词，已提取摘要"],
        confidence=0.7,
        tags=["compaction", "auto"],
    )

    if dream and dream.confidence >= self.dream_engine.config.min_confidence:
        await self._update_dream_topic(dream)

    self._pending_dream_analysis.clear()
```

#### `QuantNodes/agent/core/dream.py`

**`DreamConfig` 新增字段**:

```python
compaction_dream_interval: int = 5  # 每 N 次截断才分析一次
```

### 8.5 向后兼容

`truncate_history` 签名变更影响所有调用方：

| 调用方 | 改动 |
|--------|------|
| `loop.py:97,151,201` | 改为 `history, dropped = truncate_history(...)` |
| `runner.py:68,178` | 改为 `messages, _ = truncate_history(...)` |
| 所有测试文件 | 同步适配新签名 |

### 8.6 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/autocompact.py` | 修改 — `truncate_history` 返回 `(保留, 丢弃)` 元组 |
| `QuantNodes/agent/core/runner.py` | 修改 — 适配新签名（忽略 dropped） |
| `QuantNodes/agent/core/loop.py` | 修改 — 三个方法中集成截断 Dream 分析 |
| `QuantNodes/agent/core/dream.py` | 修改 — `DreamConfig` 新增 `compaction_dream_interval` |

### 8.7 后续优化（不在本次实施范围）

- **工具级截断 Dream**: 针对 `factor` (IC 时序) 和 `web_fetch` (文章内容) 的 tool-specific extractors
- **截断内容直接注入 Dream**: 将被截断的工具输出作为 Dream 内容，而非仅记录「检测到关键词」

---

## 9. 全量 Dream 触发点总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dream 触发点（共 3 个）                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Phase F: AgentLoop 对话截断                                  │
│     truncate_history() 丢弃旧消息                                 │
│     → dropped 消息存入 _pending_dream_analysis                   │
│     → runner 完成后由 _process_compaction_dreams() 处理           │
│     → 频率控制：每 compaction_dream_interval 次才分析              │
│     → 整体分析：合并为一个 Dream 摘要                              │
│                                                                 │
│  2. Phase D: AgentLoop 对话分析                                  │
│     runner 完成后                                               │
│     → round_count >= min_rounds_before_activate                  │
│     → should_analyze_conversation() 关键词检查                    │
│     → analyze_conversation() 生成 Dream                          │
│                                                                 │
│  3. Phase C: Agent 主动写入                                      │
│     Agent 通过 file_ops 工具写入 MEMORY.md + topic 文件           │
│     → 无需 Dream 系统，直接持久化                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 全量文件变更汇总

| Phase | 文件 | 操作类型 |
|-------|------|----------|
| A | `api/services/agent_service.py` | 修改 |
| A | `QuantNodes/agent/session/manager.py` | 修改 |
| B | `QuantNodes/agent/core/memory.py` | 修改 |
| B | `QuantNodes/agent/core/loop.py` | 修改 |
| C | `QuantNodes/agent/core/memory.py` | 修改 |
| C | `QuantNodes/agent/core/loop.py` | 修改 |
| C | `QuantNodes/agent/templates/agent/system_prompt.md` | 修改 |
| D | `QuantNodes/agent/core/dream.py` | 修改 |
| D | `QuantNodes/agent/core/loop.py` | 修改 |
| F | `QuantNodes/agent/core/autocompact.py` | 修改 |
| F | `QuantNodes/agent/core/runner.py` | 修改 |
| F | `QuantNodes/agent/core/loop.py` | 修改 |
| E | `QuantNodes/agent/__init__.py` | 修改 |
| — | `tests/agent/test_memory_persistence.py` | 新增 |

**涉及文件**: 9 个现有文件修改 + 1 个新增测试文件

---

## 11. 测试计划

### 11.1 Phase A 测试

- `test_session_unification.py` — 验证 `AgentService` 不再有 `_sessions` 字段
- `test_list_sessions_reads_disk.py` — 验证重启后 Session 仍可查询
- `test_delete_session_persists.py` — 验证删除操作持久化

### 11.2 Phase B 测试

- `test_history_append_with_tools.py` — 验证 `tools_used` 字段写入
- `test_get_recent_history.py` — 验证读取和过滤
- `test_search_history.py` — 验证关键词搜索

### 11.3 Phase C 测试

- `test_memory_manager_index.py` — 验证索引读写
- `test_memory_manager_topic.py` — 验证主题文件读写
- `test_memory_context_injection.py` — 验证 System prompt 注入
- `test_memory_migration.py` — 验证 memory.md 迁移

### 11.4 Phase D 测试

- `test_analyze_conversation.py` — 验证对话分析生成 Dream
- `test_should_analyze_conversation.py` — 验证关键词过滤
- `test_dream_rounds_threshold.py` — 验证 min_rounds_before_activate 门槛
- `test_dream_topic_update.py` — 验证 dream-insights.md 更新
- `test_dream_confidence_filter.py` — 验证低置信度过滤

### 11.5 Phase E 测试

- `test_end_to_end_memory_flow.py` — 端到端：消息 → Session + History + Dream + Memory
- `test_agent_loop_init_memory.py` — 验证初始化包含所有组件

### 11.6 Phase F 测试

- `test_truncate_history_returns_dropped.py` — 验证返回值包含丢弃消息
- `test_compaction_dream_frequency.py` — 验证频率控制
- `test_compaction_dream_analysis.py` — 验证截断消息 Dream 分析
- `test_compaction_dream_no_insight.py` — 验证无洞察时不生成 Dream
- `test_runner_ignores_dropped.py` — 验证 AgentRunner 忽略 dropped 返回值

---

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Agent 过度记忆，MEMORY.md 膨胀 | 中 | 低 | MAX_INDEX_LINES=200 限制，Agent 自行管理 |
| Dream 分析耗时影响响应 | 低 | 中 | 异步执行，不阻塞主响应；频率控制 |
| history.jsonl 无限增长 | 高 | 低 | 定期裁剪，保留最近 10000 条 |
| memory.md 迁移数据丢失 | 低 | 中 | 直接迁移到 topic 文件 |
| SessionManager 并发写入冲突 | 低 | 高 | 已有 `_cache` + 文件级写入 |
| truncate_history 签名变更影响范围 | 中 | 低 | 所有调用方同步适配，测试覆盖 |
| 截断 Dream 频率过高产生噪音 | 中 | 低 | compaction_dream_interval 频率控制 |

---

## 13. 实施顺序

```
Phase A (Session 统一)          ← 先做，消除数据不一致
    ↓
Phase B (history 增强)          ← 简单，为 Phase C 打基础
    ↓
Phase C (Memory 自动丰富)       ← 核心特性，Claude Code 风格
    ↓
Phase D (Dream 集成)            ← 依赖 Phase C 的 MemoryManager
    ↓
Phase F (Compaction-Dream)      ← 依赖 Phase D 的 DreamEngine
    ↓
Phase E (AgentLoop 集成)        ← 端到端验证，合并所有变更
```

每个 Phase 独立可测试、可提交，不破坏现有功能。
