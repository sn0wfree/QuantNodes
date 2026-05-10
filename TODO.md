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
- [x] README 更新（v0.5.0）

### Bug 修复 ✅

- [x] 修复 `api_base` URL 双重拼接
- [x] 修复 `should_execute_tools` 逻辑
- [x] 统一 `tool_call` 格式
- [x] 修复 `compile_expression` 方言 import

---

## 当前状态

- 全量测试: **2698 passed** ✅
- Git: 所有变更已提交
- Agent 工具: **15 个**（echo, sandbox, pipeline, strategy, backtest, factor, config_backtest, wiki, file_ops, code_search, git_ops, web_fetch, web_search, task）
