# QuantNodes 待执行任务

## Phase 1: Agent Status 修复 ✅ 已完成

- [x] Agent 状态端点 `/agent/status`
- [x] WebSocket 走 Vite proxy
- [x] Agent lifespan 预热
- [x] 路由冲突修复
- [x] 端口配置统一

## Phase 2: Agent opencode 化 ✅ 已完成

### Tier 1: 核心工具 ✅

- [x] `file_ops` - 读/写/编辑文件、列出目录、glob 模式匹配
- [x] `code_search` - grep 内容搜索、按模式找文件、带上下文代码搜索
- [x] `git_ops` - git 操作
- [x] `wiki` - Wiki 知识库工具（之前遗漏，已注册）

### Tier 2: 增强能力 ✅

- [x] `web_fetch` - 网页抓取工具（httpx + BeautifulSoup）
- [x] `web_search` - DuckDuckGo 网络搜索
- [x] `task` - 任务管理（JSON 持久化）

### Tier 3: 集成 ✅

- [x] `system_prompt.md` - 更新系统提示
- [x] `__init__.py` - 注册所有工具到 Agent
- [x] `tools_description.md` - 工具详细参数文档
- [x] `cli/enhanced.py` - Rich Agent 交互终端（流式输出、Markdown 渲染）

### Tier 4: 测试与文档 ✅

- [x] Agent 工具单元测试（64 个测试）
- [x] CLI 集成测试
- [x] README 更新（v2.5.0）

### Bug 修复 ✅

- [x] 修复 `api_base` URL 双重拼接
- [x] 修复 `should_execute_tools` 逻辑
- [x] 统一 `tool_call` 格式
- [x] 修复 `compile_expression` 方言 import

---

## Phase 4: Skill Bridging ✅ 已完成

- [x] SkillRegistry 线程安全（RLock）
- [x] SkillToolBridge: Skill→Tool 适配器（7 个 skill 注册为 Agent 工具）
- [x] DreamEngine: dispatch_skills() + push_to_agent()
- [x] Skill API Router: 真实 CRUD 连接 SkillRegistry
- [x] 29 个新测试

---

## Phase 5: Agent Web UI Enhancement ✅ 已完成

### 后端真流式输出 ✅

- [x] `QuantNodesLLMProvider.chat_stream()` - SSE 流式 + tool_call 检测
- [x] `AgentRunner.run_stream()` - yield token/tool_call/tool_result/done 事件
- [x] `AgentLoop.chat_stream()` - 新 async generator
- [x] `Agent.chat()` - yield 事件流（token, tool_call, tool_result, done, error）
- [x] WebSocket 协议扩展 - 新增 tool_call/tool_result 消息类型

### 前端 Markdown 渲染 ✅

- [x] 安装 `markdown-it` + `highlight.js`
- [x] `MarkdownRenderer.vue` - GitHub 风格 + 代码高亮
- [x] `ChatMessage.vue` - assistant 用 Markdown 渲染，复制按钮

### Tool 可视化 ✅

- [x] `useAgent.ts` - 处理 tool 事件，`currentToolCalls` 实时状态
- [x] `ToolCallCard.vue` - 可折叠参数/结果，支持字符串结果
- [x] AgentChat - 流式中实时显示 ToolCallCard

### Session 管理 ✅

- [x] 后端: `GET/POST/DELETE /api/chat/sessions`
- [x] 前端: session store (load/create/switch/delete)
- [x] AgentChat: session 下拉菜单，切换/新建对话

### 样式优化 ✅

- [x] 连接状态指示器（绿点）
- [x] Session 下拉菜单
- [x] 流式 typing 动画

---

## 当前状态

- 全量测试: **2727 passed** ✅
- Git: 所有变更已提交
- Agent 工具: **15 个**（echo, sandbox, pipeline, strategy, backtest, factor, config_backtest, wiki, file_ops, code_search, git_ops, web_fetch, web_search, task）
- Agent Web UI: 真流式 + Markdown + Tool 可视化 + Session 管理
