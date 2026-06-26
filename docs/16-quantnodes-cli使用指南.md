# QuantNodes CLI 使用指南 (v3.0.0+)

v3.0.0 Stage 7 新增 `quantnodes serve / stop / status / logs` + `agent` 子命令组，
提供类似 `llmwikify` 的简洁 lifecycle 接口。CLI 子命令总数从 13 升至 20。

---

## 1. 快速启动

### 1.1 安装

```bash
pip install -e .                      # 纯量化库（Wiki / Factor / Backtest）
pip install -e '.[agent]'             # + nanobot agent / WebUI / MCP
pip install -e '.[all]'               # 一键装齐
```

### 1.2 初始化项目

```bash
quantnodes init                       # 交互式向导：生成 .env + wiki/ + 目录结构
quantnodes init --force               # 强制重新初始化
```

初始化完成后 `.env` 写入 `QUANTNODES__LLM__API_KEY` / `BASE_URL` / `MODEL`。

### 1.3 启动服务

```bash
quantnodes serve                      # 前台启动，Ctrl+C 停止
quantnodes serve --check-env          # 启动前校验 API key
quantnodes serve --gateway-port 18090 # 设置 nanobot gateway 端口（默认 18090）
quantnodes serve --frontend           # 同时启动 Vite dev server（开发模式）
quantnodes serve --mcp                # 同时启动 MCP server（供外部客户端）
quantnodes serve --daemon             # 后台运行，写 .quantnodes.pid
```

启动成功后终端显示：

```
⏳ 等待后端就绪 (timeout 30s)...

✓ QuantNodes 后端已就绪
  API:       http://127.0.0.1:19380
  WebUI:     http://127.0.0.1:18090/
  日志:      tail -f logs/quantnodes_serve.log

按 Ctrl+C 停止
```

---

## 2. 子命令参考

### `quantnodes serve` — 启动后端

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--host` | 绑定主机 | `127.0.0.1` |
| `--port` | FastAPI 端口 | `19380` |
| `--gateway-port` | nanobot WebSocket + WebUI 端口 | `18090` |
| `--frontend` | 同时启动 Vite dev server（开发模式） | 不启动 |
| `--frontend-port` | 前端端口 | `5173` |
| `--mcp` | 同时启动 MCP server（供 Claude Desktop / Cursor 等） | 不启动 |
| `--mcp-port` | MCP server HTTP 端口 | `8765` |
| `--daemon` | 后台运行，写 `.quantnodes.pid` | 前台 |
| `--check-env` | 启动前校验 `QUANTNODES__LLM__API_KEY` | 不校验 |

### `quantnodes stop` — 停止后台 serve

```bash
quantnodes stop
# ✓ 已发送 SIGTERM 到 PID 12345
# pidfile 已清理: .quantnodes.pid
```

通过 `.quantnodes.pid` 查找 PID 并发送 SIGTERM。stale pidfile 会自动清理。

### `quantnodes status` — 综合状态

```bash
quantnodes status
quantnodes status --api-url http://127.0.0.1:19380
```

输出 JSON（含 pidfile 状态 + `/api/agent/status` 完整内容）：
- exit code 0：`state=running`
- exit code 1：pidfile/stale/不可达

### `quantnodes logs` — 查看日志

```bash
quantnodes logs         # 打印最后 200 行后退出
quantnodes logs -f      # 实时滚动（tail -F）
```

日志路径：`logs/quantnodes_serve.log`

### `quantnodes run` — 旧接口（兼容保留）

```bash
quantnodes run                              # 全部服务（需手动 kill）
quantnodes run --host 0.0.0.0 --port 8080 --api-port 8000 --daemon
quantnodes run --api-only
quantnodes run --frontend-only
quantnodes run --gateway-port 18090        # v3.0.0 新增
```

> **v3.0.0 变更**：`DEFAULT_API_PORT` 由 8000 改为 19380（避免与系统服务冲突）；
> 新增 `--gateway-port` 并注入子进程 env `NANOBOT_GATEWAY_PORT`。

### `quantnodes chat` — CLI 直连对话（需 [agent] extra）

```bash
quantnodes chat                          # 进入交互模式
quantnodes chat "一句话回答动量因子"       # 单次问答
quantnodes chat --workspace /path/to/.agent
```

> 需要 `pip install 'quantnodes[agent]'`（CLI 进程内直接调用 `Agent.run()`）。
> 流式 Markdown 渲染（rich）。

### `quantnodes agent status | chat | restart` — HTTP 客户端

```bash
quantnodes agent status                          # GET /api/agent/status
quantnodes agent chat "动量因子的核心思想"         # POST /api/agent/chat/send
quantnodes agent chat "test" --session my-sess    # 指定 session_id
quantnodes agent restart                          # POST /api/agent/restart
```

> 不需要 CLI 装 `nanobot-ai`，仅需后端服务在跑。
> 连接失败时提示 "quantnodes serve"。

---

## 3. 端口规划

v3.0.0 调整了默认端口，避免与系统服务（gpustack 18080 / MySQL 3306 等）冲突：

| 端口 | 服务 | 可通过参数覆盖 |
|---|---|---|
| **5173** | Vite dev server（前端，仅开发模式） | `--frontend-port` |
| **19380** | FastAPI REST API（`/api/*`） | `--port`（serve）/ `--api-port`（run） |
| **18090** | nanobot WebSocket + WebUI + HTTP API（`/gateway/*`） | `--gateway-port` |
| 18080 | ⚠️ gpustack 默认占用（不可用） | — |

> 如果 18090 仍不可用，用 `--gateway-port 18100` 换一个。
> 确认方式：`ss -tlnp | grep 18090`。

### 前端路由与代理

Vite dev server (5173) 通过 `/vite.config.ts` 的 proxy 转发请求：

- `/api/*` → FastAPI (19380)：量化数据 API（wiki/factor/backtest/stats 等）
- `/gateway/*` → nanobot (18090)：session/settings/mcp/workspace（去掉 `前缀后转发`）

AgentChat.vue 的 WebSocket 直连 gateway (18090)（通过 `VITE_NANOBOT_GATEWAY_URL` 注入）。
前端不直接暴露 gateway 端口到浏览器，通过 Vite proxy 统一走 5173。

---

## 4. 后台 + pidfile

`--daemon` 模式写入项目根 `.quantnodes.pid`（单 int）。

```bash
quantnodes serve --daemon     # 写 pidfile + 返回 0
quantnodes status             # 读 pidfile + 调 /api/agent/status
quantnodes stop               # SIGTERM + 清理 pidfile
```

- `status` 检测 stale pidfile（进程已不存在）并给出提示
- `stop` 在进程不存在时仍清理 pidfile 并返回 0
- `status` 输出 JSON 中包含 `pidfile.pid` / `pidfile.alive`

---

## 5. Agent Chat 双路径对比

| 维度 | `quantnodes chat` | `quantnodes agent chat` | 浏览器 WebUI |
|---|---|---|---|
| 依赖 | 需 `[agent]` extra | 仅需后端在跑 | 仅需后端在跑 |
| 通信 | CLI 进程内 `Agent.run()` | HTTP POST | 原生 WebSocket |
| 流式 | 终端 rich 渲染 | 单次完整响应 | 实时 streaming + markdown |
| session | `--workspace` | `--session` | 侧边栏 session 列表 |
| settings | — | — | ⚙ 面板（model/temperature/max_tokens） |
| 适用 | 交互调试 / 脚本 | 服务化 / HTTP 调用 | 日常使用（推荐） |

> **推荐**：日常使用浏览器 `http://localhost:5173/agent-chat`（原生 WebSocket + session 管理 + settings 面板）。
> 快速验证用 `quantnodes chat`，自动化脚本用 `quantnodes agent chat`。

---

## 6. 常见问题

### 端口 18080 被占用

gpustack 或其他服务默认占用 18080。解决：

```bash
quantnodes serve --gateway-port 18090   # 推荐
```

### 端口 19380 已被占用

```bash
quantnodes serve --port 9380            # 换一个端口
ss -tlnp | grep 19380                   # 查看占用进程
```

### .env 不存在

```bash
quantnodes init                         # 交互式生成 .env + wiki/ + 目录结构
```

### nanobot-ai 未装

`quantnodes init` 末尾会显示提示：
```
ℹ 未检测到 nanobot-ai（量化 agent 可选依赖）
  安装后可启用 Agent Chat / WebUI / MCP / 飞书:
    pip install 'quantnodes[agent]'
  未安装时量化工具库完全可用
```

未安装时：
- `quantnodes serve` 正常（但 Agent 功能不可用）
- `quantnodes agent status` 返回 `available: false` + install hint
- Wiki / Factor / Backtest / Strategy 等量化工具库完全可用

### LLM 调用超时 / 404

检查 `.env`：
```bash
cat .env | grep LLM                     # 确认 API_KEY / BASE_URL / MODEL
quantnodes serve --check-env            # 启动前自动校验
```

---

## 7. 与 llmwikify 的对比

| | llmwikify | quantnodes serve |
|---|---|---|
| 风格 | 单入口 + 语义化子命令 | 同 |
| 子命令 | wiki / analyze-source / synthesize ... | serve / stop / status / logs / agent / ... |
| registry | `COMMAND_REGISTRY` + `Command` ABC | 同（复用相同模式） |
| 后台模式 | — | `--daemon` + `.quantnodes.pid` |
| 端口管理 | — | 端口冲突预检 + `--gateway-port` |
| 健康检查 | — | `wait_for_health()` 轮询 `/api/agent/status` |
| 可选依赖提示 | — | `is_nanobot_installed()` + 非阻塞 warning |

---

## 8. 完整命令列表 (v3.0.0, 共 20 个)

```
quantnodes serve          # 启动后端（推荐）
quantnodes stop           # 停止后台 serve
quantnodes status         # 综合健康检查
quantnodes logs           # 查看日志

quantnodes run            # 启动服务（旧接口，兼容保留）
quantnodes chat           # CLI 直连对话（需 [agent]）
quantnodes agent status   # HTTP 调用 /api/agent/status
quantnodes agent chat     # HTTP 调用 /api/agent/chat/send
quantnodes agent restart  # HTTP 调用 /api/agent/restart

quantnodes init           # 初始化项目
quantnodes evolve         # 进化实验（v2.x）
quantnodes alpha-mcts     # Alpha MCTS 自动因子搜索
quantnodes alpha-gpt      # Alpha-GPT 5 智能体编排
quantnodes factor-info    # 因子元数据查询
quantnodes factor-best    # 因子排名
quantnodes factor-visual  # 因子可视化
quantnodes factor-rag-show
quantnodes factor-rag-eval
quantnodes factor-data-fetch
quantnodes factor-dashboard
quantnodes version
quantnodes help
```
