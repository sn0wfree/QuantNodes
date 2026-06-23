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

依赖声明：`nanobot-ai>=0.2.1,<0.3.0`（alpha 期锁次版本号，避免 API 破坏）。

本地开发期从 `/tmp/nanobot` 装：
```bash
pip install -e /tmp/nanobot
```

关键路径：
- `QuantNodes/agent/nanobot_bridge.py` — `Agent` 门面（包装 `Nanobot.from_config`）
- `QuantNodes/agent/core/quant_dream.py` — 量化专属 Dream 钩子（保留自 v2.x，向后兼容）
- `QuantNodes/agent/tools/*.py` — 15 个量化工具（父类已改为 `nanobot.agent.tools.base.Tool`）
- `.agent/` — workspace 根（上游 nanobot 默认约定，迁移自 `.quant_agent/`）
- `.agent/nanobot_config.json` — 主配置（由 `agent/config_mapper.py` 从 `.env` 生成）

升级指南见 [`docs/14-上游nanobot升级指南.md`](docs/14-上游nanobot升级指南.md)。
