# 2026-07-18 — V7.7 树模型失败 + V7.10 Stage 32 硬化

> **本日 commit 数**：4 个
> **主题**：V7.7 Phase 1 树模型 ML 路线失败 + V7.10 硬化（P0 任务清单）
> **阶段**：V7 关键修正期
> **关键发现**：修复 look-ahead 后，树模型 R² ≈ 0（ML 路线失败）

---

## 今日 commits

| hash | 类型 | 描述 |
|------|------|------|
| `bbcaf86` | **feat(v7.10)**: Stage 32 硬化 - stop_loss + CV% + v6.2 DEPRECATED + pandas 兼容 |
| `37d345c` | fix: P0-P2 修复 - v6.2 DEPRECATED 标注 + docstring 修正 + 死导入清理 |
| `415c169` | chore: gitignore 数据文件 + 新增 v7.7 源码/文档 |
| `a5de7f3` | **feat(v7.7)**: Phase 1 模型筛选完成 - 树模型显著优于线性模型 |

---

## 当日教训

### L-20260718-1: 树模型在 ETF 横截面上的预测力约等于 0 [CRITICAL]

**问题**：`a5de7f3` V7.7 Phase 1 树模型 PyCaret 25 模型：

| 模型 | 修复前 R² | 修复 look-ahead 后 |
|------|---------|-------------------|
| 树模型（LightGBM/RF/CatBoost）| 0.40 | **≈ 0** |
| 线性（Lasso/Ridge/Huber/GBR）| 0.30 | **≈ 0** |
| MLP | 0.20 | **< 0** |

**根因**：
1. **核心问题在因子（IC 噪声），不在模型**
2. 39 个因子对下一周截面收益 R² ≈ 0 是固有的
3. 修复 look-ahead 之前的"高 R²"是数据穿越

**正确做法**：
```python
# 不要尝试: 纯 ML 模型（树/MLP）做横截面预测
model = lgb.LGBMRegressor()
model.fit(X_train, Y_train)  # Y 含 look-ahead → R² 0.40
# 修复后:
model.fit(X_train, Y_train_real)  # R² ≈ 0

# 改用: 宏观择时 + 半衰期短因子
# 未来路线 (L-108)
```

**应用**：
1. **不要再尝试纯 ML 模型**（特别是树/MLP）
2. **核心问题在因子（IC 噪声），不在模型**
3. **未来要走 "宏观择时 + 半衰期短因子" 路线**

**关联**：[05_LESSONS_LIBRARY §L-108](../research_history/05_LESSONS_LIBRARY.md) 树模型在 ETF 横截面上的预测力约等于 0

---

### L-20260718-2: V7.10 Stage 32 硬化 = 5 道闸门 [HIGH]

**问题**：`bbcaf86` V7.10 Stage 32 硬化（stop_loss + CV% + v6.2 DEPRECATED + pandas 兼容）。

**5 道闸门**：

| 闸门 | 内容 | 阈值 |
|------|------|------|
| 1. 数据闸门 | OHLCV 前复权 / 动态资产池 / 缺数据审计 | 50% 阈值、min_assets=10 |
| 2. IC 闸门 | 单因子 IC / 滚动 IC / 去重 / 阈值过滤 | \|IC\|>0.05 |
| 3. 因子闸门 | 宏观时序 vs PV 截面 / ADMM 正交化慎用 | 区分 + 残差化 |
| 4. OOS 闸门 | single → 5-fold → walk-forward → expanding + CV% | **CV% < 25%** |
| 5. 硬化闸门 | 起点依赖测试 / 死代码清理 / 文档 / 工厂函数 | 见各条 |

**应用**：
1. **任何策略上 P0**：必须经过 5 道闸门
2. **每道闸门**：有明确阈值和测试方法
3. **缺一不可**

**关联**：[05_LESSONS_LIBRARY §L-321](../research_history/05_LESSONS_LIBRARY.md) 5 道闸门

---

### L-20260718-3: 死导入清理是 P2 任务但必做 [LOW]

**问题**：`37d345c` P0-P2 修复包含死导入清理。

**教训**：
1. **死导入**：未使用的 import 语句
2. **常见原因**：重构后遗留 / 测试代码引入 / 实验代码忘记删
3. **危害**：维护负担 + 命名空间污染 + 测试假阳性

**正确做法**：
```bash
# 自动检测
ruff check --select F401

# 自动修复
ruff check --select F401 --fix
```

---

## 第二天的防范清单（07-19）

1. **V7.10 过拟合验证**：4 步 OOS 流程第一步
2. **V7.7 树模型修复 look-ahead**：重做 Phase 1
3. **诚实归因**：失败就降级 DEPRECATED