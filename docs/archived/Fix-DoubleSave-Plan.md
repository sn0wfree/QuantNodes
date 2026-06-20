# Fix: AgentService 消息双重保存

> **版本**: v1.0
> **状态**: 待实施
> **创建日期**: 2026-05-11
> **关联**: Agent Memory Persistence Plan Phase A 遗留问题

---

## 1. 问题概述

Phase A 统一 Session 存储后，`AgentService._sessions` 被移除，API 层委托给 `SessionManager`。但 `AgentService.send_message()` 和 `stream_message()` 保留了自己写入 Session 消息的逻辑，而 `AgentLoop.chat()` / `chat_stream()` 内部也有相同的消息写入逻辑。由于两者共享同一个 `SessionManager` 缓存对象，导致每轮对话消息被写入 2 次。

### 1.1 受影响路径

| 路径 | 入口 | 是否双重保存 |
|------|------|-------------|
| Bus 路径 | `AgentService` → MessageBus → `AgentLoop._process_message()` | ❌ 无（AgentLoop 独立写入）|
| API 非流式 | `AgentService.send_message()` → `Agent.run()` → `AgentLoop.chat()` | ✅ 双重保存 |
| API 流式 | `AgentService.stream_message()` → `Agent.chat()` → `AgentLoop.chat_stream()` | ✅ 双重保存 |

### 1.2 具体表现

1. **LLM 上下文 user 消息重复** — AgentService 在调用 `agent.run()` 前已将 user 消息写入 session，AgentLoop.chat() 读取 history 时包含该消息，再作为 `current_message` 传给 LLM，导致同一条 user 消息在 context 中出现 2 次
2. **Session 文件膨胀 2x** — 每轮 4 条消息（user×2 + assistant×2），预期 2 条
3. **有效对话轮数减半** — `max_messages=20` 截断实际只容纳 10 轮对话
4. **错误路径孤儿消息** — `agent.run()` 抛异常时，AgentService 已写入的 user 消息留在内存 cache 中，后续调用 history 包含该孤儿消息
5. **API 返回数据膨胀** — `get_history` 返回 4 条/轮而非 2 条

### 1.3 不受影响的系统

- `history.jsonl` — AgentLoop.chat() 写入内容使用原始 `message[:500]`，与重复写入无关
- `_inject_memory_context` — 从 history.jsonl 读取，不读 session
- Dream 截断分析 — 虽然 dropped 消息可能包含重复数据，但影响轻微

---

## 2. 修复原则

1. **AgentLoop 是消息写入的唯一责任人** — 与 Bus 路径（`_process_message`）保持一致
2. **AgentService 是薄代理层** — 只负责调用 Agent + 返回结果，不操作 Session 内容
3. **保持 API 接口不变** — 对前端透明，不改变请求/响应格式
4. **同时修复错误路径** — 异常情况下不再产生孤儿消息

---

## 3. 具体改动

### 3.1 `api/services/agent_service.py` — send_message()

**移除**：line 69-70（用户消息写入）和 line 77（助手消息写入）

```python
# 修改前
async def send_message(self, content, session_id="default", config=None) -> dict:
    agent = self._get_agent(config)

    session = self.session_manager.get_session(session_id)
    session.add_message("user", content)              # ← 删除

    try:
        response = await agent.run(content, session_id=session_id)

        session.add_message("assistant", response)    # ← 删除
        self.session_manager.save_session(session)    # ← 删除

        return { ... }
    except Exception as e:
        return { ... }

# 修改后
async def send_message(self, content, session_id="default", config=None) -> dict:
    agent = self._get_agent(config)

    try:
        response = await agent.run(content, session_id=session_id)

        return {
            "message_id": f"msg-{uuid.uuid4().hex[:12]}",
            "content": response,
            "tools_used": [],
            "usage": {},
        }
    except Exception as e:
        return {
            "message_id": "msg-error",
            "content": f"Error: {str(e)}",
            "tools_used": [],
            "usage": {},
            "error": str(e),
        }
```

### 3.2 `api/services/agent_service.py` — stream_message()

**移除**：line 106-107（用户消息写入）和 line 138-139（助手消息写入 + save）

```python
# 修改前
async def stream_message(self, content, session_id="default", config=None):
    agent = self._get_agent(config)
    message_id = f"msg-{uuid.uuid4().hex[:12]}"

    session = self.session_manager.get_session(session_id)
    session.add_message("user", content)              # ← 删除

    try:
        full_content = ""
        tools_used = []
        async for event in agent.chat(content, session_id=session_id):
            ...
        session.add_message("assistant", full_content)  # ← 删除
        self.session_manager.save_session(session)       # ← 删除
    except Exception as e:
        yield { ... }

# 修改后
async def stream_message(self, content, session_id="default", config=None):
    agent = self._get_agent(config)
    message_id = f"msg-{uuid.uuid4().hex[:12]}"

    try:
        full_content = ""
        tools_used = []
        async for event in agent.chat(content, session_id=session_id):
            event["message_id"] = message_id

            if event["type"] == "token":
                full_content += event.get("content", "")
                yield event
            elif event["type"] == "tool_call":
                tools_used.append(event.get("name", ""))
                yield event
            elif event["type"] == "tool_result":
                yield event
            elif event["type"] == "done":
                final = event.get("content", "")
                if final:
                    full_content = final
                yield {
                    "type": "done",
                    "message_id": message_id,
                    "content": full_content,
                    "tools_used": list(set(tools_used)),
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            elif event["type"] == "error":
                yield event

    except Exception as e:
        yield {
            "type": "error",
            "content": str(e),
            "message_id": message_id,
        }
```

### 3.3 `api/services/agent_service.py` — get_history() / list_sessions()

无需修改。这两个方法读取 `SessionManager` 中的数据，修改后数据正确，读取自然正确。

### 3.4 `QuantNodes/agent/core/loop.py` — 无需修改

`AgentLoop.chat()` 和 `AgentLoop.chat_stream()` 中的消息写入逻辑保持不变，它们是唯一的消息写入点。

---

## 4. 影响分析

### 4.1 Bus 路径 — 无影响

`_process_message()` 不经过 `AgentService`，不调用 `Agent.run()`/`Agent.chat()`。独立路径，不受影响。

### 4.2 API 非流式路径 — 修复

修改前：session 每轮 4 条消息，LLM context 中 user 消息重复
修改后：session 每轮 2 条消息，与 Bus 路径行为一致

### 4.3 API 流式路径 — 修复

修改前：session 每轮 4 条消息 + 错误路径孤儿消息
修改后：session 每轮 2 条消息 + 错误路径由 AgentLoop.chat_stream() 内部处理

### 4.4 错误路径 — 改善

修改前：异常时 AgentService 已写入 user 消息，留在内存 cache
修改后：异常时无额外写入。AgentLoop.chat() 内部不写入（因为异常发生在 runner.run()），但 session 不会有孤儿数据

### 4.5 下游系统

| 系统 | 影响 |
|------|------|
| LLM Token 消耗 | 降低（user 消息不再重复） |
| Session 文件大小 | 降低 50% |
| 截断容量 | 恢复正常（max_messages=20 = 20 轮） |
| get_history API | 返回正确消息数 |
| history.jsonl | 无变化（本来就正确） |
| Dream 分析 | dropped 消息不再包含重复数据 |

---

## 5. 测试计划

### 5.1 新增测试

在 `tests/agent/` 中新增 `test_agent_service.py`：

```python
class TestAgentServiceDoubleSave:
    """验证 AgentService 不再双重保存消息"""

    async def test_send_message_session_message_count(self):
        """send_message 后 session 应有 2 条消息（非 4 条）"""
        ...

    async def test_send_message_llm_receives_single_user_message(self):
        """LLM 收到的 messages 中 user 消息不重复"""
        ...

    async def test_send_message_error_no_orphan(self):
        """agent.run() 异常时 session 不产生孤儿 user 消息"""
        ...

    async def test_stream_message_session_message_count(self):
        """stream_message 后 session 应有 2 条消息"""
        ...

    async def test_stream_message_error_no_save(self):
        """stream_message 异常时 session 不写入"""
        ...

    async def test_multiple_rounds_correct_count(self):
        """多轮对话后消息数 = 轮数 × 2"""
        ...
```

### 5.2 回归测试

现有测试不受影响：
- `test_loop.py` — 直接测 AgentLoop.chat()，不经过 AgentService ✅
- `test_chat.py` — 同上 ✅
- `test_memory_persistence.py` — Phase A-E 测试 ✅
- `test_session.py` — SessionManager 单元测试 ✅

### 5.3 验证命令

```bash
python3 -m pytest tests/agent/ -v --tb=short
```

---

## 6. 提交计划

单次提交：

```
fix(agent): remove double message save in AgentService

AgentService.send_message() and stream_message() were duplicating
message writes to Session, since AgentLoop.chat()/chat_stream()
already handle session persistence. This caused:
- User messages appearing twice in LLM context
- Session files growing 2x
- Half the effective conversation capacity
- Orphaned user messages on error paths
```

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 前端依赖双倍消息数 | 低 | 低 | API 接口不变，返回格式不变 |
| 其他代码读取 AgentService 保存的消息 | 极低 | 低 | AgentService 无其他消费者 |
| AgentLoop.chat() 在异常时不保存 session | 中 | 低 | 异常前无消息写入，无孤儿数据 |
| Session cache 中残留旧数据 | 极低 | 无 | cache 是内存对象，重启即清 |

---

## 8. 总结

| 项目 | 说明 |
|------|------|
| 改动文件 | 1 个（`api/services/agent_service.py`）|
| 改动行数 | ~10 行删除 |
| 新增测试 | 6 个测试用例 |
| 风险等级 | 低 |
| 预计耗时 | 15 分钟 |
