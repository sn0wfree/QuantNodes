# 上游 nanobot 升级指南

> 配合 `docs/13-Agent架构设计.md` v3.0.0 使用
> 上游: [HKUDS/nanobot](https://github.com/HKUDS/nanobot) (PyPI: `nanobot-ai`)
> 锁定版本: `>=0.2.1,<0.3.0` (alpha 期锁次版本号，避免 API 大改)

---

## 一、为什么升级

| 维度 | v2.x（自写核心） | v3.0.0（上游 nanobot） |
|------|------------------|----------------------|
| 核心运行时 | 自写 1867 行 (loop/runner/memory/dream) | 上游 `nanobot.agent.AgentLoop` ~1700 行 |
| 维护成本 | 全自维护 | 跟 upstream 同步，bug 修复秒级 |
| Subagent | 缺失（设计有未实现） | 上游 `SubagentManager` |
| MCP 桥 | 缺失 | 上游 `tools/mcp.py` 全 transport |
| WebUI | 自写 1305 行 | 上游 `webui/` |
| 渠道 | 无 | 12+ 渠道（feishu/telegram/discord/...） |
| Cron | 无 | 上游 `CronService` |
| Dream | 自写 255 行 + 1161 行 memory | 上游 `Dream + Consolidator` |
| 量化定制 | — | 通过 `QuantDreamHook` 挂在上游 hook 系统 |

---

## 二、安装与版本管理

### 2.1 本地源码安装（开发期）

```bash
# ~/Public/nanobot 是 HKUDS/nanobot 的本地克隆
cd ~/Public/nanobot
git fetch origin
git checkout v0.2.1   # 或特定 commit hash

# 安装到 QuantNodes 环境
pip install -e ~/Public/nanobot
```

### 2.2 PyPI 安装（生产期）

```bash
pip install 'nanobot-ai>=0.2.1,<0.3.0'
```

### 2.3 pyproject.toml 锁定

```toml
[project]
dependencies = [
    # ... 其他依赖
    "nanobot-ai>=0.2.1,<0.3.0",
]
```

### 2.4 版本检查

```bash
python -c "import nanobot; print(nanobot.__version__)"
# 期望: 0.2.1
```

---

## 三、架构变更点

### 3.1 删除的模块（自写 → 上游替代）

| 旧路径 | 新位置（上游） | 替代原因 |
|--------|---------------|---------|
| `agent/core/loop.py` (519行) | `nanobot/agent/loop.py` (1724行) | 上游更完整 |
| `agent/core/runner.py` (367行) | `nanobot/agent/runner.py` (1348行) | 上游更完整 |
| `agent/core/memory.py` (253行) | `nanobot/agent/memory.py` (1161行) | 上游含 Dream/Consolidator |
| `agent/core/dream.py` (255行) | 见 §3.2 | 量化专属，保留为 hook |
| `agent/core/{autocompact,context,hook,compaction}.py` | 上游同名 | 完全替代 |
| `agent/bus/` (events+queue) | `nanobot/bus/` | 同上 |
| `agent/session/` (manager) | `nanobot/session/manager.py` | 同上 |
| `agent/templates/agent/` | `.agent/SOUL.md` | 上游约定 |
| `agent/config/{loader,executor}.py` | `nanobot/config/*` | 上游 schema 更完善 |
| `agent/cli/main.py` | `python -m nanobot` | 上游 CLI 更完整 |
| `agent/web/` (WebUI) | `python -m nanobot webui --port 18080` | 上游 WebUI |

### 3.2 保留的量化模块

| 路径 | 用途 | 变更 |
|------|------|------|
| `agent/core/quant_dream.py` | 量化专属 Dream 钩子 | 从 `dream.py` 迁出，实现 `AgentHook` |
| `agent/core/dream.py` | 向后兼容 shim | re-export quant_dream |
| `agent/tools/*.py` (15个) | 量化工具 | 改父类为 `nanobot.agent.tools.base.Tool` |
| `agent/providers/quantnodes.py` | Provider 工厂 | 输出 `nanobot.providers` 配置 |
| `agent/skills_quant/` (NEW) | 6 个 SKILL.md | 新建 |
| `mcp_server/server.py` (NEW) | MCP server | FastMCP 暴露 8 个 tool |

### 3.3 新增的桥接层

| 路径 | 用途 | 行数 |
|------|------|------|
| `agent/nanobot_bridge.py` | `Agent` 门面包装 `Nanobot.from_config` | ~100 |
| `agent/config_mapper.py` | `.env` → `.agent/nanobot_config.json` | ~120 |
| `agent/cron_jobs.py` | 周期任务定义（日终/周度/月度） | ~80 |

### 3.4 workspace 迁移 `.quant_agent/` → `.agent/`

```bash
# 一次性迁移（v3.0.0 阶段 3 跑）
python scripts/migrate_workspace.py \
    --src .quant_agent \
    --dst .agent \
    --backup-keep-days 7
```

迁移映射：
- `sessions/*.json` → `.agent/sessions/*.json`
- `memory/MEMORY.md` → `.agent/SOUL.md`（个性化）+ `.agent/memory/MEMORY.md`（事实库）
- `memory/history.jsonl` → `.agent/memory/history.jsonl`
- `topic-dream-insights.md` → `.agent/memory/topic-dream-insights.md`
- `skills/*.py` → `.agent/skills/<name>/SKILL.md`（用 `migrate_skills.py` 转）

---

## 四、API 兼容性矩阵

### 4.1 编程式 API（向后兼容）

```python
# v2.x（保留工作）
from QuantNodes.agent import Agent
agent = Agent(workspace=".agent", config={...})
result = await agent.run("hello", session_id="default")

# v3.0.0 内部（Nanobot facade）
from nanobot import Nanobot
bot = Nanobot.from_config(".agent/nanobot_config.json", workspace=".agent")
result = await bot.run("hello", session_key="default")
```

`Agent` 类现在薄包装 `Nanobot`，签名不变。`agent.loop.session_manager` 仍可访问（沿用上游 API）。

### 4.2 REST API（api/services/*）

| 服务 | 变更 |
|------|------|
| `api/services/agent_service.py` | import `nanobot_bridge.Agent`，事件协议 token/tool_call/done 兼容 |
| `api/services/backtest_service.py` | 不再 `from QuantNodes.agent.tools.config_backtest`，改调 `backtest.config_runner.ConfigBacktestRunner` |
| `api/services/wiki_service.py` | 不再经 agent，直接调 `QuantNodes.research.wiki` |
| `api/services/stats_service.py` | 同 wiki |
| `api/services/dream_service.py` | 改用 `agent.core.quant_dream.QuantDreamHook` |
| `api/routers/skill.py` | 用上游 skill 解析器 |
| `api/routers/settings.py` | `reload_agent()` → `reload_bot()` |

---

## 五、跟 upstream 同步策略

### 5.1 季度同步流程

```bash
# 1. 看 upstream 变更
cd ~/Public/nanobot
git fetch origin
git log --oneline main..origin/main | head -20

# 2. 看 nanobot_config.json / .agent/ 兼容性影响
git diff origin/main -- nanobot/config/schema.py | head -50

# 3. 检查本地扩展点
rg "from nanobot" QuantNodes/agent/

# 4. 升级测试
pip install -e ~/Public/nanobot
pytest tests/agent/ -x

# 5. 跑核心行数自检（应 <5000；仅完整 GitHub clone 含此脚本，PyPI sdist 不含）
bash ~/Public/nanobot/core_agent_lines.sh

# 6. 若 diff 大，分 PR（branch: chore/sync-nanobot-0.x.y）
git checkout -b chore/sync-nanobot-0.2.2
# ... 测试通过后合入 feat/nanobot-upgrade 或 main
```

### 5.2 兼容性检查清单

每次同步 upstream 后必跑：

- [ ] `nanobot.config.schema.Config` 字段是否变更？
- [ ] `nanobot.agent.tools.base.Tool` 接口签名是否变更？
- [ ] `nanobot.agent.AgentLoop.from_config` 入口是否变更？
- [ ] `nanobot.providers.registry` provider dialect 列表是否新增/删除？
- [ ] `Nanobot.from_config` 返回类型是否变更？
- [ ] `.agent/SOUL.md` / `MEMORY.md` / `USER.md` 格式是否变更？
- [ ] `nanobot.skills` SKILL.md front-matter 字段是否变更？
- [ ] `nanobot.cron` CronSchedule 类型是否变更？
- [ ] `nanobot.channels` channel 接口是否变更？

### 5.3 锁版本策略

| 阶段 | 版本约束 | 原因 |
|------|---------|------|
| alpha 期（当前） | `>=0.2.1,<0.3.0` | 上游 API 可能变，锁次版本号 |
| 首次稳定版后 | `>=0.3,<1.0` | 跨次版本号跟随 |
| 上游稳定后 | `>=1.0,<2.0` | 跟随主版本 |

升级时先在 `feat/nanobot-upgrade` 分支试跑，确认无 breaking change 后合入 main。

---

## 六、回滚预案

若 v3.0.0 出现严重问题，回滚路径：

```bash
# 1. 切回 master
git checkout master

# 2. 临时装旧版（如果 PyPI 还能下载）
pip install 'nanobot-ai<0.2.1'  # 无效：旧版未发布到 PyPI

# 3. 实际回滚：保留 .quant_agent/ 备份
mv .quant_agent/.backup .quant_agent
rm -rf .agent/

# 4. 还原 agent/ 目录（git）
git checkout master -- QuantNodes/agent/

# 5. 还原 docs/
git checkout master -- docs/13-Agent架构设计.md docs/14-上游nanobot升级指南.md
```

回滚后上游 nanobot 仍可作为可选增强（不强依赖）。

---

## 七、相关文档

- [docs/13-Agent架构设计.md](13-Agent架构设计.md) — 主架构
- [docs/Architecture-v2.6.md](Architecture-v2.6.md) — v2.x 实际架构基线（历史）
- [AGENTS.md](../AGENTS.md) — 仓库级 agent 操作指南
- [CHANGELOG.md](../CHANGELOG.md) — 版本变更日志

---

**文档版本**: v1.0
**最后更新**: 2026-06-23
**配套代码**: `feat/nanobot-upgrade` 分支