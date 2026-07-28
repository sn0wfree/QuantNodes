# 2026-07-20 — V7.10 4 步 OOS 完成 + 统一回测引擎 + YAML + V7.11/12/14

> **本日 commit 数**：23 个（**第二次高峰**）
> **主题**：V7.10 步骤 2-4 完成 + 统一引擎 + YAML + V7.11/12/13/14 + Stage 33 规划
> **阶段**：V7 收尾 + 工程化期

---

## 今日 commits（按主题分组）

### Group 1: V7.10 4 步 OOS 收尾（4 commit）
- `eaa6c9b` — **fix(v7.10): 修复 TV-PR 验证 off-by-one bug - Calmar 0.671→0.486 (-28%)**
- `ead005c` — **fix(v7.10): 用 expanding-window 彻底消除 look-ahead - 正确验证 TV-PR**
- `be2cfe2` — fix(html): 更新 v7.10 指标为 expanding-window OOS 结果
- `cad8654` — chore: Stage 32 收尾 - expanding-window 验证 + HTML 更新 + 数据文件

### Group 2: Stage 33 规划（1 commit）
- `fd61982` — docs: Stage 33 规划 - 新因子挖掘 + HMM Regime + 跨资产信号 + 代码清理

### Group 3: V7.10 工厂函数 + 测试（2 commit）
- `e4e116a` — test: 补 stop_loss (11 tests) + v7.10 工厂函数 (8 tests)
- `252fabe` — chore: A3 代码清理 - 归档研究脚本 + 标记死代码 + 移除未使用导入

### Group 4: V7.11/12/13/14 因子工程（5 commit）
- `be4ad1e` — feat(v7.11): 新增10因子 + IC 测试
- `5c30172` — **fix(v7.11): 区分截面/时序因子 + 修正 IC 计算**
- `2ac6b33` — feat(v7.12): DCC 6 维时序特征 + regime overlay
- `1305d64` — fix(tvpr): K > N 条件修复 + v7.13 图谱距离因子测试
- `ef96daa` — feat(v7.14): 相关性距离因子测试

### Group 5: 统一回测引擎 + YAML（5 commit）
- `53d6e5c` — **feat: 统一回测引擎（消除 v1-v7 8 个文件重复）**
- `6ad3f88` — fix: NaN-safe 日收益计算 + daily_returns 可选参数
- `4f094b4` — fix: 统一引擎 inline NAV 计算 + 正确 metrics 函数
- `c18e691` — feat: v3/v4/v6/v7 适配器 + v7 OOS 修正
- `5f613f4` — refactor: 最简策略引擎 — BaseStrategy + StrategyEngine
- `c9eb84c` — feat: YAML 配置驱动回测 — run_from_yaml()
- `1b0db20` — fix: v3 import + v6 factor panel injection + YAML config runner

### Group 6: 共享模块 + NaN 过滤（5 commit）
- `680f6bc` — fix: v7 NaN weight 过滤 — std() 返回 NaN 时跳过资产
- `b336326` — refactor: 提取 R&D 共享模块（消除 24 个文件重复）
- `baab074` — refactor: rd_utils.py 提升到 momentum_etf_rotation 层级
- `40e2d52` — refactor(v7.14): 动态资产池 + 修复 fetch import 路径

---

## 当日教训（关键技术债务清理日）

### L-20260720-1: V7.10 off-by-one bug 是步骤 2 必须修的 [CRITICAL]

**问题**：`eaa6c9b` 修复 off-by-one bug：
- 步骤 2 修复后：Calmar 0.671 → **0.486**（**-28%, bug 不只是过拟合**）

**根因**：
- 索引/标签/执行周期错位 1 个时间单位
- 常见原因：周频数据用日频索引访问、起始日边界处理
- OOS 验证必须做 off-by-one 检查

**正确做法**：
```python
# 验证: 用 last-out-of-sample 测试
for shift in [-1, 0, 1]:
    test_oos(shift_days=shift)
# 如果不同 shift 给出不同 Calmar → 有 off-by-one
```

**关联**：[05_LESSONS_LIBRARY §L-201](../research_history/05_LESSONS_LIBRARY.md) OOS 验证 4 步标准化流程

---

### L-20260720-2: expanding-window 彻底消除 look-ahead [CRITICAL]

**问题**：`ead005c` V7.10 用 expanding-window 彻底消除 look-ahead → 真实 OOS：

| 步骤 | Calmar | 解释 |
|------|--------|------|
| 初始显示 OOS | 0.671 | 看起来太好了 ⚠️ |
| (1) 过拟合验证 | 0.241 | -64%, 严重过拟合 |
| (2) off-by-one 修复 | 0.486 | -28%, bug 不只是过拟合 |
| **(3) expanding OOS (2022+)** | **0.662** | **真实 OOS** |
| (4) expanding OOS (2023+) | 1.121 | 单段稳定性 |
| **final 真实 OOS（全期）** | **0.466** ⭐ | 这是最终真相 |

**教训**：
1. **full_sample 含前视偏差反致过拟合**：Sharpe 1.11 → expanding 1.57（+41%）
2. **`method="expanding"` 默认 OOS 无前视**
3. **`method="admm"` DEPRECATED**

**正确做法**：
```python
# 错误: full_sample ADMM 平滑的 β[t]
# β[t] 用 [0, T] 全量估计 → 包含 t 之后数据 → look-ahead
def full_sample_tvpr(X, Y):
    beta[t] = estimate_admm(X[0:T], Y[0:T])  # ❌

# 正确: expanding-window
def expanding_tvpr(X, Y):
    for t in range(window, T):
        beta[t] = estimate_admm(X[0:t], Y[0:t])  # ✅
```

**关联**：[05_LESSONS_LIBRARY §L-202](../research_history/05_LESSONS_LIBRARY.md) full-sample 含前视偏差反致过拟合 / §L-223 全 sample ADMM 平滑的 β[t] 天然包含未来数据

---

### L-20260720-3: DCC regime overlay 是危机预警最稳定信号 [MEDIUM]

**问题**：`2ac6b33` V7.12 DCC 6 维时序特征 + regime overlay。

**结论**：`dcc_zscore_mean > 1.5` 触发 crisis 防御（reduce_factor=0.5, cooldown=4 周）。

**为什么比单资产 bear 信号稳定**：
1. 反映"整个市场相关性结构突变"
2. 比"HS300 < MA200"更有结构性意义
3. 多次实证 2022 H1 bear_combo 71 天识别

**正确做法**：
```python
# DCC overlay: 应用到所有 TV-PR 系策略
if dcc_zscore_mean > 1.5:
    weights *= reduce_factor  # 0.5
    cooldown = 4  # 周
```

**关联**：[05_LESSONS_LIBRARY §L-110](../research_history/05_LESSONS_LIBRARY.md) DCC regime overlay 是危机预警最稳定信号

---

### L-20260720-4: 统一回测引擎消除 8 个文件重复 [HIGH]

**问题**：`53d6e5c` 统一回测引擎（消除 v1-v7 8 个文件重复）。

**核心抽象**：
```python
class BaseStrategy(ABC):
    @abstractmethod
    def compute_weights(self, date, price_panel, nav_history) -> dict[str, float]:
        ...

    def on_risk_check(self, weights, current_nav, regime) -> dict[str, float]:
        """可选回调: 策略内部风险控制"""
        return weights
```

**复用价值**：
1. v1-v10 所有策略接入点
2. 风控优先级：策略回调 > 引擎配置（VT→TF→SL）

**关联**：[05_LESSONS_LIBRARY §L-231](../research_history/05_LESSONS_LIBRARY.md) 统一回测引擎消除 8 个文件重复

---

### L-20260720-5: YAML 配置驱动让"策略配置"与"策略实现"解耦 [MEDIUM]

**问题**：`c9eb84c` YAML 配置驱动 `run_from_yaml()`（6 个 YAML 模板）。

**应用价值**：
1. 用户零代码改动即可切换策略
2. v7.10.yaml 配 `lambda_tv=0.15, lambda_l1=0.05, method=expanding` 直接对应 `V7_6Config`

**关联**：[05_LESSONS_LIBRARY §L-232](../research_history/05_LESSONS_LIBRARY.md) YAML 配置驱动

---

### L-20260720-6: NaN-safe 日收益计算是 P0 [HIGH]

**问题**：`6ad3f88` NaN-safe 日收益计算 + daily_returns 可选参数。

**正确做法**：
```python
# pandas 默认: pct_change 会把 NaN 视为 0 → 伪零收益或跨缺口收益
returns = nav.pct_change()  # ❌ 默认会填 0

# NaN-safe:
returns = nav.pct_change().where(nav.shift(1).notna() & nav.notna())
# 跨缺口收益不应计算
```

**应用**：
1. 所有 `pct_change` 都应 NaN-safe
2. 长期停牌视为 NaN（不是 0）

**关联**：[05_LESSONS_LIBRARY §L-213](../research_history/05_LESSONS_LIBRARY.md) NaN-safe pct_change

---

### L-20260720-7: 动态资产池（min_assets=10）作为最简洁工程 [MEDIUM]

**问题**：`40e2d52` V7.14 动态资产池 + 修复 fetch import 路径。

**实现**：
```python
def _get_valid_assets(returns, t, window, min_assets=10):
    """返回 t 时有足够历史的资产子集"""
    history = returns.iloc[t-window:t+1]
    valid = history.notna().sum() > window * 0.7
    if valid.sum() < min_assets:
        # 放宽到 0.5
        valid = history.notna().sum() > window * 0.5
    return valid[valid].index.tolist()
```

**应用**：
1. 任何 miss-data 敏感场景都应"动态资产池"
2. 比"复杂 imputation"简洁

**关联**：[05_LESSONS_LIBRARY §L-214](../research_history/05_LESSONS_LIBRARY.md) 动态资产池作为最简洁工程

---

### L-20260720-8: K > N 条件是 TV-PR 数值稳定性红线 [HIGH]

**问题**：`1305d64` TV-PR K > N 条件修复 + v7.13 图谱距离因子测试。

**教训**：
1. **TV-PR 需要 K（因子数） ≤ N（资产数）**
2. **K > N 时**：线性系统欠定，β 估计不稳定
3. **修复**：先做因子去重 / PCA 降维 / 增加资产池

**正确做法**：
```python
def tvpr_estimate(X, Y, K, N):
    if K > N:
        # 因子太多，需要降维
        X = pca_reduce(X, n_components=N-1)
        K = N - 1
    return estimate_admm(X, Y)
```

---

## 第二天的防范清单（07-21）

1. **StrategyResearch 子模块**：工具复用设计
2. **7 天后（07-24）**：V10 起点
3. **继续 4 步 OOS 流程**：任何新策略必须经此