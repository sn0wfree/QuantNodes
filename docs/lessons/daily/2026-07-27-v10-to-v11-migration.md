# 2026-07-27 — v10→v11 迁移 + 重复类清理 + Lint 大扫除

> **本日 commit 数**：13 个
> **主题**：5 层架构 v10→v11 迁移 + 重复类清理 + 350+ lint 错误修复
> **阶段**：架构重构期

---

## 今日 commits（按主题分组）

### Group 1: v10→v11 迁移（5 commit）
- `bc74414` — docs(research_history): 添加 V0->V10 行业轮动策略研发全周期复盘（10 份文档）
- `a86dcd3` — WIP: 阶段 1 修复 + 阶段 1 整合 (v10→v11 迁移前快照)
- `a67cb01` — refactor(v10/v11): 5 层架构从 v10 迁移到 v11, v10 保留 4 策略主体
- `c4ff551` — refactor(core): Phase 1-4 项目结构整理
- `c51c473` — refactor(v5/v6): 合并 v5+v5_1, 合并 v6+v6_1+v6_2

### Group 2: 重复类清理（3 commit）
- `b1c22e3` — refactor(config): 消除 common/ 与 core/ 的重复类定义
- `a65a761` — fix(tests): 修复重构后的 broken imports + 补缺失 BacktestConfig
- `d934fd3` — fix(tests): 修复 v4/hmmlearn/validation 7 个测试失败

### Group 3: Import 路径 + 止损（2 commit）
- `9a262f3` — fix(core): strategy_versions.py v7 import 路径 .v7 → ..v7
- `9f5cc69` — fix(v7): 止损回测支持合成数据 + v7.7 标记 DEPRECATED

### Group 4: Lint 大扫除（3 commit）
- `f212605` — fix(v2/v4/v9): 修复 10 个 pre-existing F821 lint 错误
- `3908484` — fix(scripts/common/core/v5): 修复 2 处 broken import + 清理 10 个重构残留 unused imports
- `7745298` — **style: ruff --fix 清理 219 个格式化问题 (F401/W293/W292/F541/W291)**

---

## 当日教训

### L-20260727-1: 迁移前必须有快照（WIP commit） [HIGH]

**问题**：`a86dcd3` WIP: 阶段 1 修复 + 阶段 1 整合 (v10→v11 迁移前快照)。

**教训**：
1. **重大迁移前**：必须有 WIP 快照 commit
2. **失败可回退**：快照保证可回滚
3. **审计可追溯**：git history 保留完整记录

**正确做法**：
```bash
# 重大迁移前
git checkout -b migration/wip
git commit -am "WIP: snapshot before migration"
# 实施迁移
# 失败: git reset --hard HEAD~1  # 回到快照
```

**关联**：[05_LESSONS_LIBRARY §L-323](../research_history/05_LESSONS_LIBRARY.md) 工程债的"识别 + 修复 + 预防"

---

### L-20260727-2: 重复类合并必须找所有 caller [HIGH]

**问题**：`b1c22e3` 消除 common/ 与 core/ 的重复类定义。

**教训**：
1. **重复类（duplicate class）**：是典型技术债
2. **合并前**：grep 所有 import + 测试
3. **保留 BackwardCompat 别名**：避免破坏外部依赖

**正确做法**：
```python
# 合并前: 两个地方定义 BacktestConfig
# common/backtest_config.py
class BacktestConfig: ...

# core/backtest_config.py
class BacktestConfig: ...  # 重复

# 合并后: 保留一处 + 另一处做 shim
# core/backtest_config.py  # 权威定义
# common/backtest_config.py
from ..core.backtest_config import BacktestConfig  # 兼容
```

---

### L-20260727-3: Lint 大扫除收益极高（219 错误 → 0） [HIGH]

**问题**：`7745298` ruff --fix 清理 219 个格式化问题。

**教训**：
1. **lint 不是 nice-to-have**：是 P1 任务
2. **F401（未用导入）**：419 → 190（清理 229）
3. **批量修复**：ruff --fix 安全规则（F401/W293/W292/F541/W291）
4. **手动修复**：F811/F841/E70x 等需要人工判断

**正确做法**：
```bash
# 1. 安全自动修复
ruff check --select F401,W293,W292,F541,W291 --fix

# 2. 手动修复
ruff check --select F811,F841,E70x,E402,E721,E712,E741

# 3. CI gate
ruff check  # 必须 0 错误
```

**关联**：[05_LESSONS_LIBRARY §L-323](../research_history/05_LESSONS_LIBRARY.md) 工程债

---

### L-20260727-4: Import 路径修复必须用绝对路径或明确相对 [MEDIUM]

**问题**：`9a262f3` strategy_versions.py v7 import 路径 `.v7` → `..v7`。

**教训**：
1. **相对 import**（`.v7` vs `..v7`）：在包结构变化时容易出错
2. **推荐**：用绝对 import `from package.module import X`
3. **pytest 友好**：绝对 import 减少路径问题

**正确做法**：
```python
# 错误: 相对 import（包结构敏感）
from .v7 import strategy  # ❌
from ..v7 import strategy  # ❌ 容易错

# 正确: 绝对 import
from momentum_etf_rotation.v7 import strategy  # ✅
```

---

## 第二天的防范清单（07-28）

1. **v7 全系列审计**：从 v7.0 到 v7.14 全部审查
2. **v5.1 无未来函数确认**：扩展审计
3. **v7.10 标记 DEPRECATED**：全样本标准化未来函数
4. **v7.3 数据管道修复**：核心 bug