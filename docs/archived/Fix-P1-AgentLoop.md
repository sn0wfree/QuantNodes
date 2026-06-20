# P1: AgentLoop 三项修复

> **版本**: v1.0
> **状态**: 待实施
> **创建日期**: 2026-05-12
> **关联**: 架构审计 P1 级发现

---

## 1. P1-1: Dream 未注入 System Prompt

### 1.1 问题

`DreamStore.get_injection_content(config)` 已实现（memory.py:81-93），能将高置信度 Dream 洞察格式化为注入文本。但 `AgentLoop._inject_memory_context()` 从未调用它，导致 LLM 永远看不到 Dream 洞察。

当前 `_inject_memory_context()` 注入了：
- Phase C: MemoryManager 索引（topic 文件列表）
- Phase B: 最近 3 条对话摘要

缺失的：
- Phase D: Dream 洞察（高置信度的对话/因子/策略分析结果）

### 1.2 修复方案

在 `_inject_memory_context()` 末尾增加 Dream 注入：

```python
# Phase D: 注入 Dream 洞察
dream_ctx = self.dream_store.get_injection_content(self.dream_engine.config)
if dream_ctx:
    messages[0]["content"] += f"\n\n{dream_ctx}"
```

**改动位置**: `QuantNodes/agent/core/loop.py` — `_inject_memory_context()` 方法末尾
**改动行数**: +3 行

### 1.3 注意事项

- `self.dream_store` 需要在 AgentLoop.__init__ 中保存为实例变量（当前只传给了 DreamEngine，未保存引用）
- 需要在 `__init__` 中添加 `self.dream_store = dream_store`

---

## 2. P1-2: `_pending_dream_analysis` 并发共享

### 2.1 问题

`_pending_dream_analysis` 是 `AgentLoop` 的实例变量（line 66），类型为 `List[Dict]`。所有使用同一个 AgentLoop 实例的 session 共享这个列表。

当两个 session 并发处理时：
1. Session A 截断消息 → 追加到 `_pending_dream_analysis`
2. Session B 截断消息 → 追加到同一个列表
3. Session A 的 `_process_compaction_dreams()` 分析时，会包含 Session B 的消息

结果：Session A 的 Dream 洞察可能基于 Session B 的对话内容，产生错误归因。

### 2.2 修复方案

将 `_pending_dream_analysis` 从 `List[Dict]` 改为 `Dict[str, List[Dict]]`，按 session_key 隔离：

```python
# __init__:
self._pending_dream_analysis: Dict[str, List[Dict]] = {}

# 截断时（_process_message 和 chat/chat_stream）:
if dropped:
    if session_key not in self._pending_dream_analysis:
        self._pending_dream_analysis[session_key] = []
    self._pending_dream_analysis[session_key].extend(dropped)

# 分析时（_process_dream_analysis）:
if session_key in self._pending_dream_analysis and self._pending_dream_analysis[session_key]:
    await self._process_compaction_dreams(session_key)
```

**改动位置**: `QuantNodes/agent/core/loop.py`
- `__init__`: 类型变更
- `_process_compaction_dreams`: 接收 session_key 参数
- `_process_dream_analysis`: 传入 session_key
- 3 个调用点（`_process_message`, `chat`, `chat_stream`）: 改为按 session_key 存储

**改动行数**: ~15 行变更

### 2.3 影响范围

| 路径 | 影响 |
|------|------|
| `_process_message` (Bus) | 按 session_key 存储 dropped 消息 |
| `chat` (API 非流式) | 按 session_key 存储 dropped 消息 |
| `chat_stream` (API 流式) | 按 session_key 存储 dropped 消息 |
| `_process_compaction_dreams` | 接收 session_key，只处理该 session 的 dropped 消息 |
| Dream 分析频率控制 | `_compaction_counter` 改为 per-session Dict |

---

## 3. P1-3: `_process_message` 异常无响应

### 3.1 问题

`_process_message()` 中 `await self.runner.run(spec)` 可能抛出异常（LLM 调用失败、provider 错误等）。异常时：
- 消息已从 bus 消费
- 无 OutboundMessage 发出
- 用户永远收不到响应

当前代码（loop.py:238）：
```python
result = await self.runner.run(spec)  # 如果这里抛异常？
# ... 后续代码全部跳过
# ... 包括 bus.publish_outbound(response)
```

### 3.2 修复方案

用 try/except 包裹 runner.run() 及后续逻辑，异常时发送错误 OutboundMessage：

```python
async def _process_message(self, msg: InboundMessage) -> None:
    """处理单条消息"""
    try:
        session = self.session_manager.get_session(msg.session_key)
        # ... 现有逻辑 ...
        result = await self.runner.run(spec)
        # ... 现有逻辑 ...
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(
            "Error processing message: %s", e, exc_info=True
        )
        response = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"处理消息时发生错误: {str(e)}",
        )
        await self.bus.publish_outbound(response)
```

**改动位置**: `QuantNodes/agent/core/loop.py` — `_process_message()` 方法
**改动行数**: +8 行（try/except 包裹 + 错误响应）

### 3.3 注意事项

- 异常时不写入 session（避免孤儿消息）
- 异常时不写入 history.jsonl（避免错误数据污染历史）
- 错误消息格式：`"处理消息时发生错误: {error}"`
- 仍然记录日志（error 级别 + exc_info）

---

## 4. 测试计划

### 4.1 新增测试文件

`tests/agent/test_agent_loop_p1.py`

| 测试 | 验证内容 |
|------|---------|
| **P1-1 Dream 注入** | |
| `test_inject_memory_includes_dream_insights` | _inject_memory_context 注入了 DreamStore 中的高置信度 Dream |
| `test_inject_memory_no_dream_when_empty` | DreamStore 为空时不注入 |
| `test_inject_memory_dream_respects_confidence_threshold` | 低置信度 Dream 不注入 |
| **P1-2 并发隔离** | |
| `test_pending_dream_analysis_per_session` | 不同 session 的 dropped 消息隔离存储 |
| `test_compaction_dreams_only_processes_session` | _process_compaction_dreams 只处理指定 session |
| **P1-3 错误恢复** | |
| `test_process_message_error_sends_outbound` | runner.run 抛异常时仍发送 OutboundMessage |
| `test_process_message_error_no_session_write` | 异常时不写入 session |
| `test_process_message_error_content` | 错误 OutboundMessage 包含异常信息 |

### 4.2 回归测试

现有测试不受影响：
- `test_loop.py` — AgentLoop.chat() 直接测试，不涉及 _process_message
- `test_agent_service.py` — AgentService 层测试
- `test_memory_persistence.py` — MemoryStore/DreamStore 单元测试
- `test_skills_phase4.py` — SkillRegistry/SkillToolBridge 测试

---

## 5. 提交计划

单次提交：

```
fix(agent): three P1 fixes — dream injection, per-session dream queue, error recovery

- Inject DreamStore insights into system prompt via _inject_memory_context()
- Change _pending_dream_analysis from shared List to per-session Dict
- Wrap _process_message() in try/except, send error OutboundMessage on failure
```

---

## 6. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Dream 注入增加 system prompt 长度 | 低 | 低 | get_injection_content() 有 max_dreams_per_day=10 限制 |
| per-session dict 内存增长 | 极低 | 无 | session 处理完即清除，且 bus 路径有 concurrency_gate |
| 错误响应暴露内部异常信息 | 低 | 低 | 仅暴露 str(e)，不暴露堆栈 |
| _compaction_counter per-session 增加内存 | 极低 | 无 | Dict[str,int]，每个 session 一条 |
