# 2026-07-28 — v7 全系列审计 + v7.3 数据管道修复

> **本日 commit 数**：6 个
> **主题**：v7 全系列审计 + v7.10 标记 DEPRECATED + v7.3 数据管道重构 + 10 个教训文档
> **阶段**：审计 + 数据正确性修复期
> **关键发现**：v7.10 全样本标准化是未来函数；v7.3 数据管道 4 个 bug

---

## 今日 commits

| hash | 类型 | 描述 |
|------|------|------|
| `847d6ec` | fix(v7): standardize_v7_10() 消除全样本标准化未来函数 |
| `9a07ca3` | **docs(v7): v7.10 标记 DEPRECATED — 全样本标准化未来函数确认** |
| `61291a1` | fix(v7): CV% 测试脚本修正 — 每起点独立标准化 + common/metrics |
| `087252f` | test(v7): v7.10 test2 Rolling Window Z-score 标准化实验 |
| `8363019` | **audit(v7/v5): v7 全系列审计 + v5_1 无未来函数确认** |
| `6507d0b` | **fix(v7.3): 数据管道重构 + v7.3.2 β预筛选 + 10个教训文档** |

---

## 当日教训

### L-20260728-1: 全样本标准化是未来函数红线 [CRITICAL]

**问题**：`847d6ec` + `9a07ca3` 修复 standardize_v7_10() 全样本标准化未来函数：
- **DEPRECATED v7.10**：全样本标准化导致 look-ahead
- 修复：每起点独立标准化（rolling-window）

**根因**：
```python
# 错误: 全样本标准化（包含未来数据）
def standardize_v7_10_full(X):
    """用全样本均值和标准差标准化"""
    return (X - X.mean()) / X.std()  # ❌ 用未来数据

# 正确: rolling-window 或 expanding-window
def standardize_v7_10_rolling(X, window=252):
    """只用过去 window 数据"""
    rolling_mean = X.rolling(window).mean()
    rolling_std = X.rolling(window).std()
    return (X - rolling_mean) / rolling_std  # ✅
```

**关联**：[05_LESSONS_LIBRARY §L-202](../research_history/05_LESSONS_LIBRARY.md) full-sample 含前视偏差反致过拟合

---

### L-20260728-2: v7.3 数据管道 4 个 bug [CRITICAL]

**问题**：`6507d0b` v7.3 数据管道重构（详见 `docs/lessons/001-010.md`）：

| Bug | 严重度 | 修复 |
|---|---|---|
| `resample("W").last().pct_change()` on 收益数据 | CRITICAL | 输入必须是价格 |
| expanded_panel 混合 simple + log 收益 | HIGH | 统一为 simple return |
| NAV `(1+log_return).cumprod()` | HIGH | 用 `exp(cumsum)` 或 simple return |
| `>= curr_date` 1 日前视 | MODERATE | 改为 `>` |

**教训**：
1. **数据层只返回价格**，收益计算在策略层
2. **统一收益类型**：避免 simple + log 混用
3. **每改一处必跑测试**：避免回归

**关联**：见 `docs/lessons/002-resample-on-returns.md`, `003-mixed-return-types.md`, `004-nav-calculation-wrong.md`, `006-1day-lookahead.md`

---

### L-20260728-3: v7 全系列审计是 P0 任务 [CRITICAL]

**问题**：`8363019` v7 全系列审计 + v5_1 无未来函数确认。

**结论**：
- **v7.3**：clean（数据管道修复后）
- **v7.7**：DEPRECATED（树模型 R² ≈ 0）
- **v7.10**：DEPRECATED（全样本标准化未来函数）
- **v7.11/12/13/14**：实验性，未投产
- **v5.1**：clean（无未来函数）

**教训**：
1. **全系列审计**：每季度必须做一次
2. **DEPRECATED 标记**：失败策略要明示
3. **诚实归因**：不在生产但有研究价值的保留为研究版本

**正确做法**：
```python
# 策略状态分类
strategies_status = {
    'v1.0': 'production',
    'v3': 'deprecated',      # 1/N 等权失败
    'v4': 'production',
    'v5.1': 'production',
    'v6': 'production',
    'v6.1': 'deprecated',    # 过拟合
    'v6.2': 'research',      # CV% 56.9% FAIL
    'v7.3': 'production',    # 数据管道修复后
    'v7.7': 'deprecated',    # 树模型 R² ≈ 0
    'v7.10': 'deprecated',   # 全样本标准化未来函数
    'v7.11-14': 'research',  # 实验性
    'v9macro': 'production',
    'v10': 'production',     # 4 策略 Vol-parity
}
```

---

### L-20260728-4: v7.10 4 步 OOS 必须再做一次 [HIGH]

**问题**：`087252f` v7.10 test2 Rolling Window Z-score 标准化实验。

**教训**：
1. **即使已经做 4 步**：滚动窗口 Z-score 还需额外测试
2. **CV% 测试脚本**：每起点独立标准化（不能共享）
3. **common/metrics**：统一指标计算

---

### L-20260728-5: 数据探索中的指标选择错误 [HIGH]

**问题**（详见 `docs/lessons/001-data-exploration-mistake.md`）：
- 用 `notna().all()` 判断"可用" → 错误（要求零 NaN）
- 正确用 `notna().any()`（有数据）+ NaN 比例判断质量

**应用**：
1. **数据探索三板斧**：缺失值统计 → 抽样验证 → 结论自查
2. **禁止**：用 `notna().all()` 判断"可用"
3. **每次结论后**：抽查 3-5 个具体数据点

**关联**：`docs/lessons/001-data-exploration-mistake.md`

---

## 第二天的防范清单（07-29+）

1. **v7.3.2 β 预筛选**：评估冗余性（L-007 Lasso 稀疏性）
2. **10 个教训文档**：与 05_LESSONS_LIBRARY 整合
3. **v11 迁移收尾**：所有策略接入统一引擎

---

## 跨日总结

**22 天教训密度**（按主题）：
- **Look-ahead bias**：8 次出现（07-15, 07-19, 07-20, 07-27, 07-28）
- **NaN-safe**：5 次（07-13, 07-15, 07-16, 07-20）
- **OOS 4 步**：4 次（07-19, 07-20, 07-28）
- **诚实归因/DEPRECATED**：6 次（07-09, 07-10, 07-14, 07-18, 07-19, 07-28）
- **简单规则胜复杂**：3 次（07-09, 07-14, 07-17）

**最高频教训**：
1. **Look-ahead bias**（最常见陷阱）
2. **诚实归因**（最难做到）
3. **NaN-safe 计算**（最易遗漏）

详见 `summary.md`（待生成）。