## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Only `GRAPH_REPORT.md` and `.graphify_root` are tracked in VCS. The `graph.json` (~21MB) and `manifest.json` (~144KB) are gitignored and **regenerated locally** by `graphify update .` (AST-only, no API cost). After fresh clone, run `graphify update .` once before reading the graph.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## nanobot 上游依赖

> v3.0.0+ Agent 核心直接消费 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 上游（PyPI 包名 `nanobot-ai`）。

依赖声明（**可选依赖，从 v3.0.0 Stage 5.3 起**）：`nanobot-ai>=0.2.1,<0.3.0`（alpha 期锁次版本号，避免 API 破坏）。

```bash
# 三档安装
pip install quantnodes            # 纯量化库
pip install 'quantnodes[agent]'   # + nanobot agent / WebUI / MCP
pip install 'quantnodes[all]'     # 一键装齐 agent + mcp
```

本地开发期从 `/tmp/nanobot` 装：
```bash
pip install -e /tmp/nanobot
```

关键路径（v3.0.0）：
- `QuantNodes/agent/__init__.py` — `NANOBOT_AVAILABLE` 标志 + PEP 562 proxy + `NanobotNotInstalled` 异常
- `QuantNodes/agent/nanobot_bridge.py` — `Agent` 门面（包装 `Nanobot.from_config`）
- `QuantNodes/agent/config_mapper.py` — `.env` → `.agent/nanobot_config.json`（含 channels / mcpServers）
- `QuantNodes/agent/core/quant_dream.py` — 量化专属 Dream 钩子（保留自 v2.x）
- `QuantNodes/agent/cron_jobs.py` — 3 个 quant 系统任务（daily-recap / weekly-review / monthly-strategy-pool）
- `QuantNodes/agent/tools/*.py` — 14 个量化工具（父类 `nanobot.agent.tools.base.Tool`）
- `QuantNodes/agent/skills_quant/*.md` — 6 个 SKILL.md（factor / strategy / backtest / risk / dream / config）
- `api/services/nanobot_runtime.py` — 单进程 lifespan 包装器（FastAPI + nanobot 共存）
- `api/routers/agent.py` — `/api/agent/{status,health,restart,chat/send,sessions,cron,...}`
- `frontend/src/views/AgentChat.vue` — iframe + 状态机
- `frontend/src/composables/useNanobotWebSocket.ts` — wire protocol client
- `QuantNodes/mcp_server/server.py` — FastMCP 9 tools（stdio + HTTP）
- `.agent/` — workspace 根（HKUDS nanobot 约定，从 v2.x 的 `.quant_agent/` 迁移）
  - 迁移脚本：`scripts/migrate_workspace.py`
  - 在 `.gitignore` 中（含 API key）
- `.agent/nanobot_config.json` — 主配置（由 `agent/config_mapper.py` 从 `.env` 生成）
- `.agent/SOUL.md` + `.agent/agents/*.md` — 多 Agent 团队（main + factor-analyst / backtest-engineer / risk-manager）

升级指南见 [`docs/14-上游nanobot升级指南.md`](docs/14-上游nanobot升级指南.md)。
可选依赖 + 单进程集成指南见 [`docs/15-可选依赖安装指南.md`](docs/15-可选依赖安装指南.md)。
