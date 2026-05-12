# Multi-Provider LLM Support

## 背景

QuantNodes 当前通过 `OpenAIClient` 调用 LLM API，只支持单一 provider 配置（`api_key` + `api_base`）。随着使用场景扩展，需要支持同时配置多个 LLM 提供商（DeepSeek、阿里百炼、硅基流动、OpenRouter 等），实现：

- 按 model 名自动路由到正确的 provider
- 同 model 多 provider 时按优先级选择
- 主 provider 失败时自动 fallback 到备选
- 每条消息可指定不同 model（已有的 per-message model override）

## 业界调研

| 方案 | 模式 | Provider数 | 适用性 |
|---|---|---|---|
| LiteLLM | 独立库，model字符串路由 | 100+ | 重依赖，不推荐 |
| LangChain | 继承BaseChatModel | 50+ | 重依赖，不推荐 |
| OpenCode (Go) | 泛型baseProvider + switch | 8个 | 轻量，可参考 |
| **纯OpenAI兼容** | 统一client + base_url | 任意 | **推荐** |

**关键发现**: 国内主流 provider（DeepSeek、DashScope、SiliconFlow、智谱、月之暗面）全部支持 OpenAI 兼容 API，使用标准 `Bearer <token>` 认证。差异仅在于：
- base URL 不同
- 部分 provider 需要额外 header（如 OpenRouter 的 `X-OpenRouter-Title`）
- 推理模型的特殊参数（`extra_body`）

因此**无需引入外部依赖**，在现有 `OpenAIClient` 上扩展即可。

## 架构设计

### 核心组件

```
settings.json
    ↓
ProviderRegistry.from_settings()
    ↓
┌─────────────────────────────────┐
│  ProviderRegistry               │
│  ├── ProviderConfig (deepseek)  │
│  ├── ProviderConfig (dashscope) │
│  └── ProviderConfig (openrouter)│
└─────────────┬───────────────────┘
              │ resolve(model)
              ↓
┌─────────────────────────────────┐
│  QuantNodesLLMProvider          │
│  ├── registry → 动态创建client  │
│  ├── default_model              │
│  └── fallback_providers         │
└─────────────────────────────────┘
              │ chat(model="deepseek-chat")
              ↓
    ProviderRegistry.resolve("deepseek-chat")
    → ProviderConfig(name="deepseek", api_base="https://api.deepseek.com/v1")
    → OpenAIClient(base_url="...", extra_headers={})
    → HTTP POST /chat/completions
```

### 配置格式

```json
{
  "agent": {
    "provider": "dashscope",
    "model": "deepseek-v4-pro",
    "api_key": "sk-xxx",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "providers": {
      "deepseek": {
        "api_key": "sk-xxx",
        "api_base": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "priority": 1
      },
      "dashscope": {
        "api_key": "sk-xxx",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["deepseek-v4-pro", "qwen3.6-plus", "qwen3.6-flash"],
        "priority": 1
      },
      "openrouter": {
        "api_key": "sk-or-v1-xxx",
        "api_base": "https://openrouter.ai/api/v1",
        "models": ["baidu/cobuddy:free", "google/gemma-3-27b-it:free"],
        "extra_headers": {"X-OpenRouter-Title": "QuantNodes"},
        "priority": 3
      }
    },
    "fallback_providers": ["deepseek", "dashscope"],
    "llm_timeout": 60,
    "llm_max_retries": 3,
    "max_tokens": 102400
  }
}
```

**向后兼容**: `providers` 字段为空或不存在时，用顶层 `api_key` + `api_base` 创建单一 client，行为与现在完全一致。

### 路由策略

1. **model=None** → 返回默认 provider（`agent.provider` 指定的）
2. **model 在默认 provider 的 models 中匹配** → 优先返回默认 provider
3. **model 在其他 provider 中匹配** → 按 `priority` 排序（数字小优先），返回最优
4. **model 在所有 provider 中都找不到** → 返回默认 provider，使用默认 model 兜底
5. **主 provider 调用失败** → 按 `fallback_providers` 顺序尝试备选

### Provider 差异处理

| 差异点 | 处理方式 |
|---|---|
| Base URL | 每个 ProviderConfig 独立配置 |
| 额外 Header | `extra_headers` dict 传递给 OpenAIClient |
| 认证格式 | 统一 Bearer token（所有国内 provider 兼容） |
| 流式 SSE 格式 | 统一 OpenAI 格式（已验证所有国内 provider 一致） |
| 推理模型参数 | 通过 `kwargs` / `extra_body` 传递（不在此方案范围内） |

### 支持的 Provider 预设

| Provider | Base URL | 需要额外 Header | 免费额度 |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | 否 | 有试用额度 |
| DashScope (阿里百炼) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 否 | 注册送额度 |
| SiliconFlow (硅基流动) | `https://api.siliconflow.cn/v1` | 否 | 有免费模型 |
| OpenRouter | `https://openrouter.ai/api/v1` | `X-OpenRouter-Title` | 免费模型 |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4/` | 否 | 注册送额度 |
| 月之暗面 | `https://api.moonshot.cn/v1` | 否 | 注册送额度 |
| Ollama (本地) | `http://localhost:11434/v1` | 否 | 免费 |

## 实施计划

### Phase 1: 后端 Provider 注册表 + 路由

#### Step 1: OpenAIClient 支持 extra_headers

**文件**: `QuantNodes/ai/llm/openai.py`

- `__init__` 新增 `extra_headers: Optional[Dict[str, str]] = None`
- `_get_headers()` 中 `headers.update(self.extra_headers)`
- 删除硬编码的 OpenRouter headers（移到配置层）

#### Step 2: 创建 ProviderRegistry

**新文件**: `QuantNodes/agent/providers/registry.py`

- `ProviderConfig` 数据类：name, api_key, api_base, models, extra_headers, priority, timeout, max_retries
- `ProviderRegistry` 类：
  - `from_settings(agent_config)` — 从配置加载
  - `resolve(model)` — 路由到最优 provider
  - `get_client(config)` — 创建 OpenAIClient
  - `list_providers()` / `get_models_map()` — 查询

#### Step 3: QuantNodesLLMProvider 动态路由

**文件**: `QuantNodes/agent/providers/quantnodes.py`

- 持有 `ProviderRegistry` 替代单一 `LLMClientBase`
- `_get_client_for_model(model)` — 根据 model 动态选择 client
- `chat()` / `chat_stream()` 中使用动态 client
- Fallback 逻辑：主 provider 失败时尝试 `fallback_providers`

#### Step 4: Agent._create_provider() 使用注册表

**文件**: `QuantNodes/agent/__init__.py`

- 初始化 `ProviderRegistry.from_settings(config)`
- 传 registry 给 `QuantNodesLLMProvider`

**文件**: `QuantNodes/core/config.py`

- `LLMConfig` 新增 `providers: dict = {}`
- `load_from_settings()` 解析 `agent.providers`

#### Step 5: 测试

**新文件**: `tests/agent/test_provider_registry.py`

- 向后兼容测试（无 providers 字段）
- 多 provider 加载 + model 路由
- Priority 排序
- 默认 provider 优先
- Fallback 逻辑
- extra_headers 传递

### Phase 2: 前端 + API（Phase 1 验证通过后）

- Settings API 新增 Provider CRUD
- 前端 Provider 管理 tab
- 模型选择器按 provider 分组

## Commit 计划

| # | Commit | 内容 |
|---|---|---|
| 1 | `docs: multi-provider LLM support design` | 设计文档 |
| 2 | `feat: OpenAIClient support extra_headers` | Step 1 |
| 3 | `feat: ProviderRegistry for multi-provider routing` | Step 2 + 3 + 4 |
| 4 | `test: ProviderRegistry unit tests` | Step 5 |
