## 核心原则
### 0. Human Edit Only
本文件只允许人类进行修改编辑

### 1. Think Before Coding（编码前思考）
不假设。多解就列出。困惑就停下问。资深工程师会嫌复杂就重写。
**When in doubt, ask。** 停下来，说出困惑，问清楚再动手。

### 2. Simplicity First（简洁优先）
200 行能 50 行就重写。无推测功能、无单次抽象、无用不到的"灵活性"。

### 3. Surgical Changes（精准修改）
只动该动的。匹配现有风格。每一行 diff 可追溯到用户请求。
顺手重构 / 改格式 / 删预存死代码 = 禁止。
发现无关死代码 → **提，不删**。发现无关未跟踪文件 → **不动，除非用户请求**。

### 4. Goal-Driven Execution（目标驱动）
"修 bug" → "写复现测试，让它通过"。多步任务先列 [步骤] → [verify]。

### 5. Context First（上下文优先）
动手前 Read 完整实现、grep 同模式用法、graphify 查现成测试。
不假设 helper 是空的。

### 6. Verify-Then-Proceed（改完即验）
每 edit 后立即 `ruff check <file>`。改完一处就跑受影响的 `test_<name>.py`。
不攒到最后才发现 fixture 错。

### 7. Loop Until Done（循环到目标）
目标明确就循环到通过（ruff 干净 + pytest 全过 + 用户状态达成）。
不"差不多"就停。

### 8. Memory Hygiene（记忆清洁）
中文 commit `<type>(<scope>): 说明`。archive 不删（保留 rename）。
stash 遗留改动 commit 前主动提醒。

### 9. Safety First（破坏性操作红线）

**`docker` / `pkill` / `rm -rf` / `git push --force` 永远指定目标，禁止模糊匹配。**

| 禁止 | 必须替代为 | 原因 |
|------|-----------|------|
| `docker rm -f $(docker ps -aq)` | `docker rm -f <container_id>` | `ps -aq` 包含**所有**容器（含其他服务），`-f` 强删 = 误杀 |
| `pkill -f <pattern>` | `kill <pid>` 或 `pkill -x <exact_name>` | `-f` 匹配整条命令行，可能误杀同名进程 |
| `rm -rf $path`（变量未引号） | 先 `ls "$path"` 确认范围，加 `set -e` 早停 | 变量未引号 = 灾难 |
| `git push --force` | 永远先 `git status` + `git log --oneline` | 误覆盖他人提交 |

**debug 卡住进程的顺序**：
1. `docker logs <id>` / `docker exec <id> ps aux` 看具体卡哪
2. `docker stop <id>`（指定单个，给 10s 优雅停）
3. 实在不行才 `docker kill <id>` + `docker rm <id>`（指定单个）
4. **绝不**用 `$(docker ps -aq)` 或 `pkill -f` 一锅端

## Before Acting Checklist（操作前清单）
**每次修改文件/删除/提交前，必须过这 7 步：**

0. （**docker/systemd/全局进程**）先 `docker ps` / `ps aux` 列出**当前所有**在跑的容器/进程，操作前报告「会动哪些」给用户
0. （**破坏性批量命令**）`rm -f <list>` 类必须先 dry-run（`docker ps -q --filter name=...` 列出）再执行
1. 这个操作需要做吗？（YAGNI — 不需要就别做）
2. 这是用户明确请求的吗？（未请求 → 不动）
3. 会影响其他文件吗？（grep 改动范围）
4. 改完后 ruff / test 会过吗？（验证）
5. 用户确认了吗？（commit/删除/大改动必须确认）