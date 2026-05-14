# LiteLLM SDK 集成方案

## 概述

本文档描述了 QuantNodes Agent 系统如何集成 LiteLLM SDK 来解决 LLM API 调用中的可靠性问题，包括速率限制、重试逻辑和连接池管理。

## 背景

### 问题描述

在测试 Agent Chat 功能时，发现以下问题：
- OpenRouter 免费账号有严格的速率限制（每天 50 次请求）
- 高频请求会触发 500 Internal Server Error
- 现有代码使用 `requests` 库，每次请求新建 TCP 连接，无连接复用
- 重试逻辑简单，不区分 429（限流）和 500（服务端错误）

### 解决方案

集成 LiteLLM Python SDK，提供以下能力：
- 内置指数退避重试（区分 429/500）
- httpx 连接池（连接复用）
- 可配置的速率限制（Token Bucket）
- 多模型路由和 Fallback 支持

## 架构设计

### 集成模式

采用 **SDK 模式**（而非 Proxy 模式），原因：
- 与现有 Provider 接口兼容，直接替换内部实现
- 部署简单，无需额外服务进程
- 性能更好（少一跳网络延迟）
- 速率限制更容易在调用层实现

### 组件关系

```
agent/providers/quantnodes.py
    ├── QuantNodesLLMProvider (主 Provider)
    │       ├── LiteLLM SDK (acompletion)
    │       ├── AsyncTokenBucket (速率限制)
    │       └── Legacy LLMClientBase (Fallback)
    │
    └── providers/rate_limiter.py
            └── AsyncTokenBucket (令牌桶实现)
```

### 请求流程

1. 请求进入 `QuantNodesLLMProvider.chat()`
2. 通过 `AsyncTokenBucket.acquire()` 进行速率限制
3. 调用 LiteLLM `acompletion()`（内置重试 + 连接池）
4. 成功：返回 `LLMResponse`
5. 失败：降级到 `Legacy LLMClientBase`

## 依赖

### 安装

```bash
pip install litellm>=1.70.0
```

或在 `pyproject.toml` 中添加：
```toml
dependencies = [
    "litellm>=1.70.0,<2.0.0",
]
```

## 模块说明

### rate_limiter.py

**路径**: `QuantNodes/agent/providers/rate_limiter.py`

#### TokenBucket (同步版本)

```python
class TokenBucket:
    def __init__(self, requests_per_second: float = 0.5, burst: int = 1):
        """
        Args:
            requests_per_second: 每秒允许的请求数（免费账号建议 0.5）
            burst: 突发容量
        """
```

#### AsyncTokenBucket (异步版本)

用于 async 环境中的速率限制：
```python
class AsyncTokenBucket:
    async def acquire(self):
        """获取令牌，必要时等待"""
```

### quantnodes.py 修改

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_litellm` | bool | `True` | 是否启用 LiteLLM SDK |
| `rate_limit_rps` | float | `0.5` | 每秒请求数（2秒1次） |
| `max_retries` | int | `3` | LiteLLM 内置重试次数 |
| `timeout` | float | `60` | 请求超时（秒） |

#### Fallback 机制

当 LiteLLM SDK 调用失败时，自动降级到原有的 `LLMClientBase`：

```python
async def _fallback_to_legacy(self, messages, tools, model, max_tokens, temperature):
    """LiteLLM 失败时降级到原有 LLMClientBase"""
```

## 配置示例

### settings.json

```json
{
  "agent": {
    "provider": "litellm",
    "use_litellm": true,
    "rate_limit_rps": 0.5,
    "api_key": "sk-or-v1-...",
    "api_base": "https://openrouter.ai/api/v1",
    "model": "openrouter/free",
    "max_tokens": 102400,
    "llm_timeout": 60,
    "llm_max_retries": 3
  }
}
```

### 环境变量

```bash
# LiteLLM 配置
LITELLM_API_KEY=sk-or-v1-...
LITELLM_BASE_URL=https://openrouter.ai/api/v1
LITELLM_MODEL=openrouter/free

# 速率限制
LITELLM_RATE_LIMIT_RPS=0.5
```

## 速率限制策略

### 免费账号策略

- **请求间隔**: 2 秒（0.5 请求/秒）
- **突发容量**: 1
- **降级处理**: 触发限流时等待后重试

### 付费账号策略

```json
{
  "rate_limit_rps": 5  // 每秒5次请求
}
```

### 实现原理

Token Bucket 算法：
1. 桶以 `rate` 速度补充令牌
2. 每个请求消耗 1 个令牌
3. 无令牌时请求阻塞等待

## 重试策略

### LiteLLM 内置重试

| 错误类型 | 行为 |
|---------|------|
| 429 (Rate Limit) | 指数退避，等待后重试 |
| 500 (Server Error) | 指数退避，等待后重试 |
| 401 (Auth Error) | 不重试，直接失败 |

### 重试参数

```python
acompletion(
    max_retries=3,
    timeout=60,
)
```

## Fallback 降级流程

```
LiteLLM 调用
    │
    ├── 成功 ──→ 返回 LLMResponse
    │
    └── 失败 (限流/服务端错误)
            │
            ├── 检查 legacy_client 是否可用
            │       │
            │       ├── 可用 ──→ 降级到 LLMClientBase
            │       │
            │       └── 不可用 ──→ 抛出异常
            │
            └── 返回降级结果或抛出异常
```

## 测试计划

### 单元测试

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/agent/test_chat.py` | 验证 chat 功能正常 |
| `tests/agent/test_agent_service.py` | 验证 fallback 机制 |

### 回归测试

```bash
# 运行全部 agent 测试
pytest tests/agent/ -v

# 预期结果: 783 passed
```

### 速率限制测试

```python
# 验证请求间隔
import time

bucket = AsyncTokenBucket(requests_per_second=0.5)
start = time.time()
for i in range(3):
    await bucket.acquire()
elapsed = time.time() - start
assert elapsed >= 4.0  # 3 requests * 2s = 6s theoretical minimum
```

## 与旧代码的兼容性

### 保留的组件

| 组件 | 用途 |
|------|------|
| `LLMClientBase` | Fallback 降级使用 |
| `ProviderRegistry` | 多模型路由配置 |
| `MessageBus` | 消息总线（无变化） |
| `AgentLoop` | 主循环（无变化） |

### 弃用的部分

| 组件 | 弃用原因 |
|------|---------|
| 直接使用 `requests.post()` | 无连接复用，无重试 |

## 故障排查

### 问题: 所有请求返回 500

**可能原因**:
1. OpenRouter 免费额度用完
2. 请求频率超过限制
3. API Key 无效

**排查步骤**:
```bash
# 检查 API Key 有效性
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

### 问题: 速率限制过于严格

**解决方案**: 调整 `rate_limit_rps`

```json
{
  "agent": {
    "rate_limit_rps": 1.0
  }
}
```

### 问题: LiteLLM 调用超时

**解决方案**: 增加 timeout

```json
{
  "agent": {
    "llm_timeout": 120
  }
}
```

## 未来规划

### Phase 2 (可选)

- 添加 Circuit Breaker 模式
- 支持多模型自动切换
- 添加请求缓存

### Phase 3 (可选)

- 支持 LiteLLM Proxy 模式（多团队共享）
- 添加用量监控 Dashboard
- 支持自定义 Provider 路由策略

## 参考资料

- [LiteLLM 官方文档](https://docs.litellm.ai/docs/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Token Bucket 算法](https://en.wikipedia.org/wiki/Token_bucket)
- [OpenRouter API 文档](https://openrouter.ai/docs/)