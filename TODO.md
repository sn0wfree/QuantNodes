# QuantNodes 待执行任务

## Phase 1: Agent Status 修复 ✅ 已完成

- [x] Agent 状态端点 `/agent/status`
- [x] WebSocket 走 Vite proxy
- [x] Agent lifespan 预热
- [x] 路由冲突修复
- [x] 端口配置统一

## 待提交 (Git)

### Phase 1 遗留未提交
| 文件 | 修改内容 |
|------|----------|
| `.env.template` | `QUANTNODES__LLM__API_KEY` 命名同步 |
| `QuantNodes/cli.py` | `init_llmwikify_wiki(force=args.force)` |
| `api/main.py` | lifespan 预热 Agent |
| `api/routers/agent.py` | `/agent/status` 端点 |
| `frontend/src/composables/useAgent.ts` | WebSocket 走 proxy |
| `frontend/vite.config.ts` | 默认端口 5173 |

### Phase 2 本次会话新增/修改
| 文件 | 修改内容 |
|------|----------|
| `QuantNodes/agent/__init__.py` | 注册 file_ops/code_search/git_ops/wiki + api_base normalize |
| `QuantNodes/agent/tools/__init__.py` | export 新工具 |
| `QuantNodes/agent/tools/file_ops.py` | 新建 - 文件操作工具 |
| `QuantNodes/agent/tools/code_search.py` | 新建 - 代码搜索工具 |
| `QuantNodes/agent/tools/git_ops.py` | 新建 - Git 操作工具 |
| `QuantNodes/agent/providers/base.py` | should_execute_tools 逻辑修复 |
| `QuantNodes/agent/templates/agent/system_prompt.md` | 新增工具说明 + tool_call 格式 |
| `QuantNodes/agent/templates/agent/tools_description.md` | 新增详细参数文档 |
| `QuantNodes/symbolic/compiler.py` | 修复 compile_expression 方言 import |
| `tests/agent/test_tools_all.py` | 新建 - 41 个 Agent 工具测试 |
| `tests/agent/test_providers_quantnodes.py` | 修正 should_execute_tools 测试 |

---

## Phase 2: Agent opencode 化

### Tier 1: 核心工具 ✅ 已完成

- [x] `file_ops` - 读/写/编辑文件、列出目录、glob 模式匹配
  - 文件: `QuantNodes/agent/tools/file_ops.py`
  - 功能: `read_file`, `write_file`, `edit_file`, `list_files`, `glob_files`
- [x] `code_search` - grep 内容搜索、按模式找文件、带上下文代码搜索
  - 文件: `QuantNodes/agent/tools/code_search.py`
  - 功能: `grep`, `find_files`, `search_code`
- [x] `git_ops` - git 操作
  - 文件: `QuantNodes/agent/tools/git_ops.py`
  - 功能: `git_status`, `git_diff`, `git_commit`, `git_log`
- [x] `wiki` - Wiki 知识库工具
  - 已注册到 Agent（之前遗漏）

### Bug 修复 ✅ 已完成

- [x] 修复 `api_base` URL 双重拼接（`/chat/completions` 被拼接两次）
- [x] 修复 `should_execute_tools` 逻辑（`finish_reason="stop"` 不应执行工具）
- [x] 统一 system_prompt 中的 ````tool_call``` 格式
- [x] 修复 `compile_expression` 遗漏 DuckDB/MySQL 方言 import

### 测试 ✅ 已完成

- [x] 41 个单元测试全部通过（ToolRegistry, PipelineTool, FactorTool, BacktestTool, ConfigBacktestTool, AgentRunner, AgentLoop）
- [x] MySQL 集成测试通过（Docker mysql:5.7 容器）
- [x] 全量测试 2671 passed

### Tier 2: 增强能力

- [ ] `web` - 网络工具
  - 文件: `QuantNodes/agent/tools/web.py`
  - 功能: `fetch_url`, `search_web`
- [ ] `task` - 任务管理
  - 文件: `QuantNodes/agent/tools/task.py`
  - 功能: `create_task`, `update_task`, `list_tasks`

### Tier 3: 集成

- [x] `system_prompt.md` - 更新系统提示（添加 file_ops/code_search/git_ops/wiki 工具指令）
- [x] `__init__.py` - 注册新工具到 Agent（file_ops, code_search, git_ops, wiki）
- [x] `tools_description.md` - 工具详细参数文档
- [ ] `cli/enhanced.py` - Rich 终端 UI
  - 流式输出、Markdown 渲染、命令历史、Tab 补全

### Tier 4: 测试与文档

- [ ] web/task 工具单元测试
- [ ] CLI 集成测试
- [ ] README 更新

---

## 执行顺序

1. ~~Git 提交当前变更~~ → 待提交
2. ~~Tier 1: 核心工具 (file_ops, code_search, git_ops)~~ ✅
3. Tier 2: 增强能力 (web, task)
4. ~~Tier 3: 集成 (CLI, prompts, registration)~~ ✅ prompts + registration 已完成
5. ~~Tier 4: 测试与文档~~ ✅ Agent 工具测试已完成，待补充 web/task 测试
