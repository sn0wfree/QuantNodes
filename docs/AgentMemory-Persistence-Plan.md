# Agent Memory Persistence 设计文档

> **版本**: v2.5.0
> **状态**: 待实施
> **创建日期**: 2026-05-10

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

---

## 2. 设计目标

1. **单一 Session 存储源** — 消除 `AgentService._sessions`，全部委托给 `SessionManager`
2. **对话历史可检索** — `history.jsonl` 增强为可查询的对话摘要存储
3. **Agent 自主记忆** — 借鉴 Claude Code `MEMORY.md` 模式，Agent 在对话中决定记什么
4. **Dream 与对话闭环** — Agent 对话中的洞察自动进入 Dream 系统
5. **渐进式加载** — System prompt 按需注入相关记忆，而非全量注入

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
        # 确保 agent 的 session_manager 可被 API 层访问
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
                # 在 user 和 assistant 内容中搜索
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

### 5.3 目录结构

```
.quant_agent/memory/
├── memory.md              ← 废弃（渐进迁移后删除）
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

### 5.4 MEMORY.md 格式

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

### 5.5 具体改动

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

**修改 `_process_message()` 和 `chat()` 中的记忆注入逻辑**:

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

### 5.6 迁移策略

1. Phase C 实施时，检测是否存在 `memory.md`
2. 如果存在，将内容迁移到对应的 topic 文件
3. 生成 `MEMORY.md` 索引
4. 保留 `memory.md` 作为备份，标记为废弃

### 5.7 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/memory.py` | 修改 — 新增 `MemoryManager` 类 |
| `QuantNodes/agent/core/loop.py` | 修改 — 初始化 `MemoryManager`，注入索引上下文 |
| `QuantNodes/agent/templates/agent/system_prompt.md` | 修改 — 添加记忆系统指引 |

---

## 6. Phase D: Dream 集成

### 6.1 目标

Agent 对话中的洞察自动进入 Dream 系统，形成「对话 → 洞察 → 记忆」闭环。

### 6.2 架构

```
Agent 对话
    │
    ▼
DreamHook (after_iteration)
    │ 分析对话内容
    ▼
DreamEngine.generate_dream()
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

### 6.3 具体改动

#### `QuantNodes/agent/core/dream.py`

**新增 `analyze_conversation()`**:

```python
async def analyze_conversation(
    self,
    user_message: str,
    assistant_response: str,
    tools_used: List[str] = None,
) -> Optional[Dream]:
    """分析对话并生成洞察 Dream"""
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
        return None

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

#### 新增 `QuantNodes/agent/core/hook.py`

**新增 `DreamHook`**:

```python
class DreamHook(AgentHook):
    """对话结束时自动分析并生成 Dream"""

    def __init__(self, dream_engine: "DreamEngine"):
        self.dream_engine = dream_engine
        self._current_user_msg = None

    async def before_iteration(self, context: AgentHookContext) -> None:
        """捕获用户消息"""
        if context.messages:
            last_user = None
            for msg in reversed(context.messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            self._current_user_msg = last_user

    async def after_iteration(self, context: AgentHookContext) -> None:
        """对话结束后分析并生成 Dream"""
        if not context.final_content or not self._current_user_msg:
            return

        tools_used = [
            tc.get("name", "") for tc in context.tool_calls
        ] if context.tool_calls else []

        dream = await self.dream_engine.analyze_conversation(
            user_message=self._current_user_msg,
            assistant_response=context.final_content,
            tools_used=tools_used,
        )

        if dream and dream.confidence >= self.dream_engine.config.min_confidence:
            # 将洞察写入 dream-insights.md
            await self._update_dream_topic(dream)

    async def _update_dream_topic(self, dream) -> None:
        """更新 dream-insights.md 主题文件"""
        try:
            from .memory import MemoryManager
            mm = MemoryManager(self.dream_engine.dream_store.workspace.parent)
            existing = mm.read_topic("dream-insights")
            timestamp = dream.timestamp[:10]
            new_entry = f"\n### {timestamp} - {dream.type}\n"
            new_entry += f"- {dream.content}\n"
            for insight in dream.insights:
                new_entry += f"  - {insight}\n"

            updated = existing + new_entry
            mm.write_topic("dream-insights", updated)

            # 更新 MEMORY.md 索引
            index = mm.read_index()
            if "topic-dream-insights.md" not in index:
                index += "\n## Insights\n- Dream洞察记录 (topic-dream-insights.md)\n"
                mm.write_index(index)
        except Exception:
            pass
```

#### `QuantNodes/agent/core/loop.py`

**注册 DreamHook**:

```python
# 在 __init__ 或初始化方法中
from .dream import DreamEngine
from .memory import DreamStore
from .hook import DreamHook

dream_store = DreamStore(self.workspace)
dream_engine = DreamEngine(dream_store)
dream_hook = DreamHook(dream_engine)

if isinstance(self.hook, CompositeHook):
    self.hook.add_hook(dream_hook)
```

### 6.4 文件变更清单

| 文件 | 操作 |
|------|------|
| `QuantNodes/agent/core/dream.py` | 修改 — 新增 `analyze_conversation()` |
| `QuantNodes/agent/core/hook.py` | 修改 — 新增 `DreamHook` 类 |
| `QuantNodes/agent/core/loop.py` | 修改 — 注册 `DreamHook` |

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

        # Phase D: DreamEngine + DreamHook（新增）
        dream_store = DreamStore(self.workspace)
        self.dream_engine = DreamEngine(dream_store)
        dream_hook = DreamHook(self.dream_engine)
        if isinstance(self.hook, CompositeHook):
            self.hook.add_hook(dream_hook)
```

#### 7.2.2 `_process_message()` / `chat()` / `chat_stream()` 统一改造

三个方法遵循相同模式：

```python
async def _process_message(self, msg: InboundMessage) -> None:
    session = self.session_manager.get_session(msg.session_key)

    # 构建消息
    history = [m for m in session.messages if m.get("role") in ("user", "assistant", "tool")]
    history = truncate_history(history, max_messages=20)

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
    ├── 1. SessionManager.get_session()     ← 读取历史
    ├── 2. truncate_history()                ← 裁剪到20条
    ├── 3. build_messages()                  ← 构建 prompt
    ├── 4. memory_manager.get_memory_context()  ← Phase C: 注入索引
    ├── 5. memory.get_recent_history()       ← Phase B: 注入近期摘要
    │
    ▼
AgentRunner.run() / run_stream()
    │
    ├── LLM 调用
    ├── 工具执行
    ├── DreamHook.after_iteration()         ← Phase D: 分析对话
    │   └── DreamEngine.analyze_conversation()
    │       └── DreamStore.save_dream()
    │       └── MemoryManager.write_topic("dream-insights")
    │
    ▼
AgentLoop 后处理
    │
    ├── session.add_message() × 2
    ├── session_manager.save_session()      ← Phase A: 磁盘持久化
    ├── memory.append_history()             ← Phase B: 对话摘要
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
| `QuantNodes/agent/core/loop.py` | 修改 — 初始化 MemoryManager + DreamHook，统一改造三个方法 |
| `QuantNodes/agent/__init__.py` | 修改 — 同步 memory_manager 引用 |

---

## 8. 全量文件变更汇总

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
| D | `QuantNodes/agent/core/hook.py` | 修改 |
| D | `QuantNodes/agent/core/loop.py` | 修改 |
| E | `QuantNodes/agent/core/loop.py` | 修改 |
| E | `QuantNodes/agent/__init__.py` | 修改 |
| — | `tests/agent/test_memory_persistence.py` | 新增 |

**涉及文件**: 8个现有文件修改 + 1个新增测试文件

---

## 9. 测试计划

### 9.1 Phase A 测试

- `test_session_unification.py` — 验证 `AgentService` 不再有 `_sessions` 字段
- `test_list_sessions_reads_disk.py` — 验证重启后 Session 仍可查询
- `test_delete_session_persists.py` — 验证删除操作持久化

### 9.2 Phase B 测试

- `test_history_append_with_tools.py` — 验证 `tools_used` 字段写入
- `test_get_recent_history.py` — 验证读取和过滤
- `test_search_history.py` — 验证关键词搜索

### 9.3 Phase C 测试

- `test_memory_manager_index.py` — 验证索引读写
- `test_memory_manager_topic.py` — 验证主题文件读写
- `test_memory_context_injection.py` — 验证 System prompt 注入
- `test_memory_migration.py` — 验证 memory.md 迁移

### 9.4 Phase D 测试

- `test_analyze_conversation.py` — 验证对话分析生成 Dream
- `test_dream_hook_after_iteration.py` — 验证 Hook 触发
- `test_dream_topic_update.py` — 验证 dream-insights.md 更新
- `test_dream_confidence_filter.py` — 验证低置信度过滤

### 9.5 Phase E 测试

- `test_end_to_end_memory_flow.py` — 端到端：消息 → Session + History + Dream + Memory
- `test_agent_loop_init_memory.py` — 验证初始化包含所有组件

---

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Agent 过度记忆，MEMORY.md 膨胀 | 中 | 低 | MAX_INDEX_LINES=200 限制，Agent 自行管理 |
| DreamHook 分析耗时影响响应 | 低 | 中 | 异步执行，不阻塞主响应 |
| history.jsonl 无限增长 | 高 | 低 | 定期裁剪，保留最近 10000 条 |
| memory.md 迁移数据丢失 | 低 | 中 | 保留原文件备份 |
| SessionManager 并发写入冲突 | 低 | 高 | 已有 `_cache` + 文件级写入 |

---

## 11. 实施顺序

```
Phase A (Session 统一)     ← 先做，消除数据不一致
    ↓
Phase B (history 增强)     ← 简单，为 Phase C 打基础
    ↓
Phase C (Memory 自动丰富)  ← 核心特性，Claude Code 风格
    ↓
Phase D (Dream 集成)       ← 依赖 Phase C 的 MemoryManager
    ↓
Phase E (AgentLoop 集成)   ← 端到端验证
```

每个 Phase 独立可测试、可提交，不破坏现有功能。
