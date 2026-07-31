# ETF 动量轮动策略开发进程总结

> **日期**: 2026-07-28
> **资产**: 43-56 ETF (A 股宽基/行业/海外/商品/债券)
> **数据**: 2018-01 ~ 2026-05 (8.4 年)
> **OOS**: 2022-01 ~ 2026-05 (4.4 年)
> **分支**: `refactor/v10-to-v11-migration` (13 commits)

---

## 一、演进路线图

```
v0 (baseline)
  └-> v1/v2 (CICC动量 + VT/TF/Cost)        <- v1.0 locked
       └-> v3 (SubStrategy 多策略组合)       <- X 失败
       └-> v4 (IC 因子择时 + 风格轮动)       <- ! 实验
       └-> v5 (11 月频量价因子)              <- 生产
            └-> v5_1 (逆波动率加权)          <- 生产
                 └-> v6 (v5.1 选股 + v1.0 风控)  <- 生产
                      └-> v6_1 (IC-IR 加权)     <- ! 实验
                      └-> v6_2 (正交化)         <- X 过拟合
       └-> v7 (TV-PR 时变 beta)              <- 生产
            └-> v8 (Jump Model + ML)        <- ! 实验
       └-> v9 (银河/中信因子 + 动态仓位)     <- ! 实验
       └-> v10 (4 策略 Vol-parity)          <- 生产 (最终)
            └-> v11 (5 层架构 + ACT 升级)    <- 生产
```

---

## 二、逐版本核心进展

> 指标口径: sqrt(252) + 日历日年化 (2026-07-27 审计统一)
> 数据来源: `VERSION_TRACKING.md` + `docs/77-v0_v10_codebase_audit.md`

| 版本 | Stage | 核心机制 | 全期 Sharpe | OOS Sharpe | OOS MaxDD | 状态 |
|------|-------|---------|------------|-----------|----------|------|
| **v0.0** | 8 | 90d 动量 baseline | 1.223 | 1.366 | -7.01% | 基线 |
| **v1.0** | 12A | hybrid 动量 + VT + Cost, 80/20 固收+ | 1.122 | **1.459** | **-1.94%** | 生产 |
| **v3** | 16A | SubStrategy 抽象基类 + 多策略组合 | 0.781 | 1.035 | -11.87% | 失败 |
| **v4** | 17 | IC 驱动因子择时 + 风格轮动 + HMM | 0.248 | 0.259 | -45.0% | 实验 |
| **v5** | 22 | 11 月频量价因子 (等权) | 0.839 | 0.513 | -19.41% | 生产 |
| **v5.1** | 22+ | 逆波动率加权 (+20.7% Calmar) | 0.948 | 0.623 | -18.59% | 生产 |
| **v7.10** | 30 | TV-PR 时变 beta + 17 宏观 + 19 量价 | 0.903 | 1.017 | -15.42% | 生产 |
| **v8** | 31 | Jump Model 牛熊检测 + per-asset sigmoid | - | 0.871 (5bp) | -18.14% | 实验 |
| **v9** | 32 | 银河熵权 + 动态仓位 + 中信 4 策略 | 0.834 | 0.924 | -14.5% | 实验 |
| **v10** | 33 | 4 策略 Vol-parity 组合 | 1.216 | **1.434** | **-7.20%** | 生产 |
| **v10-DynD** | 33+ | 信号加权动态权重 | 1.350 | **1.464** | **-4.77%** | 生产 |
| **v11** | - | 5 层架构 + yang_zhang_vol + kelly + drawdown | - | 1.131 | - | 生产 |

注: v8 的 0.871 为 per-asset 5bp 成本方案 (`docs/63-final_summary.md`)；v8 method_b (有未来函数) Sharpe 1.045 不可实盘。

---

## 三、关键里程碑

### 3.1 v1.0 locked - 建立框架 (Stage 12A)

- 4 步组合管理: 去重 -> 剔高相关 -> 强制分散 -> 逆波动加权 + 止损补位
- 80/20 固收+动量，MaxDD 仅 **-1.94%**
- OOS Sharpe **1.459** (全策略最高之一)
- hybrid 动量 (price + slope_r2) + VolTargeting + CostModel 是最有效组合

### 3.2 v3 失败教训 - 架构 != 业绩 (Stage 16A)

- SubStrategy 抽象基类设计完善 (v4-v7 全部继承)
- 但 1/N 等权多策略组合 Calmar 0.504 < v2 0.892
- **教训**: "架构先进 != 业绩进步"

### 3.3 v7 TV-PR 突破 - 时变 beta (Stage 30)

- TV-PR 时变 beta 估计 (Cui 2025)，expanding-window 真实 OOS
- 17 宏观 + 19 量价因子，CV% 16.6% PASS
- v7.10 硬化: stop_loss + CV% 验证 + v6.2 DEPRECATED
- OOS Sharpe 1.017，OOS AnnRet +20.87%

### 3.4 v8 Jump Model 修复 - per-asset (Stage 31)

- 从聚合 composite signal 改为 per-asset 处理 (关键洞察)
- sigmoid 阈值 0.50 将触发率从 86% 降至 5%
- 换手率从 47x 降至 15x，Sharpe +14%，MaxDD -34%
- **教训**: Jump Model 应 per-asset，聚合 signal 导致高触发率

### 3.5 v9 动态仓位发现 - #1 alpha 源 (Stage 32)

- Brinson 归因验证: `pos=(0.7-0.5*z).clip(0.2,1.0)` 贡献 **71% alpha**
- 银河因子体系 (17 因子熵权法)
- 中信 4 策略 (多因子/行业轮动/大类资产/里昂全天候)

### 3.6 v10 Vol-parity 组合 - 最终生产 (Stage 33)

- 4 策略波动率平价: v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%
- DualMom 与其他策略相关性 **-0.005** = 真正独立 alpha 源
- 5 个动态权重方案 (A 市场状态 / B 波动率 / C 回撤 / D 信号加权 / E 混合)
- 方案 D (信号加权) OOS Calmar **1.806**，MaxDD **-4.77%**

### 3.7 v11 5 层架构迁移 - ACT 升级

- 从 v10 复制 5 层架构 (macro/factor/industry/style/portfolio/position/risk)
- ACT-1: yang_zhang_vol 替换 realized_vol
- ACT-2: kelly_audit 审计
- ACT-3: drawdown_controller 回撤控制
- OOS Sharpe 1.131

---

## 四、全面审计与重构 (2026-07-27 ~ 07-28)

### 4.1 审计发现 (9 个 bug)

| # | 严重度 | 文件 | 问题 | 影响 |
|---|--------|------|------|------|
| 1 | CRITICAL | v10 `dynamic_weight_schemes.py` | `resample('D').ffill()` 零收益稀释 | Sharpe 虚增 ~23% |
| 2 | CRITICAL | v4 `multi_strategy_v4.py` | 硬编码 252/n 对周频年化 | Sharpe 虚增 ~2.2x |
| 3 | CRITICAL | `scripts/v4/v4_full_backtest.py` | freq="W" 用在日频数据 | N_Years=39.6 (实际 8.2) |
| 4 | HIGH | `scripts/v9/v9_factor_galaxy.py` | 日频数据默认 freq='W' | Sharpe 虚增 ~2.2x |
| 5 | HIGH | `combo/unified_v1v5_compare.py` | v0 lookback 144->90 错配 | v0 NAV 非规范配置 |
| 6 | MODERATE | v7 `macro_substrategy_v7_6.py` | daily_nav 格式不匹配 | 脚本运行时崩溃 |
| 7 | MODERATE | v4 factor 策略 | warmup 期权重归零 | Sharpe=-3.322 |
| 8 | LOW | v9 `factor_galaxy.py` | return 后不可达代码 | 无运行时影响 |
| 9 | LOW | v10 `rrg_rotation.py` | 缺少 rebalance guard | 代码质量问题 |

所有 CRITICAL + HIGH (P0-P1) 已修复。

### 4.2 修复前后对比

| 策略 | 修复前 Sharpe | 修复后 Sharpe | 变化 | 原因 |
|------|-------------|-------------|------|------|
| v10 Vol-parity | 1.930 | 1.216 | -0.714 | resample('D') 零收益稀释 |
| v10 DynD | 1.849 | 1.350 | -0.499 | 同上 |
| v0.0 baseline | 0.827 | 1.223 | +0.396 | lookback 144->90 |

### 4.3 重构工作 (13 commits)

分支 `refactor/v10-to-v11-migration`，从 `bc74414` 到 `7745298`：

| # | Commit | 内容 |
|---|--------|------|
| 1 | `bc74414` | docs: V0->V10 全周期复盘 (10 份文档) |
| 2 | `a86dcd3` | WIP: 阶段 1 修复 + 整合 (迁移前快照) |
| 3 | `a67cb01` | refactor: v10 5 层架构迁移到 v11 |
| 4 | `c4ff551` | refactor: core/ Phase 1-4 项目结构整理 |
| 5 | `c51c473` | refactor: v5+v5_1 合并, v6+v6_1+v6_2 合并 |
| 6 | `b1c22e3` | refactor: common/ 与 core/ 重复类消除 |
| 7 | `a65a761` | fix: 重构后 broken imports + BacktestConfig |
| 8 | `d934fd3` | fix: v4/hmmlearn/validation 7 个测试失败 |
| 9 | `9a262f3` | fix: strategy_versions.py v7 import 路径 |
| 10 | `9f5cc69` | fix: v7 止损回测 + v7.7 DEPRECATED |
| 11 | `f212605` | fix: v2/v4/v9 共 10 个 F821 lint 错误 |
| 12 | `3908484` | fix: 2 处 broken import + 10 个 unused imports |
| 13 | `7745298` | style: ruff --fix 清理 219 个格式化问题 |

### 4.4 重构成果

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| **指标计算** | 6 种不同实现散落各版本 | 统一为 `common/metrics.py` `compute_metrics()` |
| **项目结构** | 9 个根目录 stub + v5/v5_1, v6/v6_1/v6_2 分散 | core/ 提取, common/ 统一, 版本合并 |
| **重复类** | common/ 与 core/ 各有一套 VolTargeting/TrendFilter/CostConfig | 统一到 `common/backtest_config.py` |
| **测试** | 多处 broken import + 失败 | 350 passed, 11 skipped, 0 failed |
| **ruff 错误** | 415 errors | 190 errors (F401/W293/W292/F541/W291/F821 全部清零) |

剩余 190 个 ruff 错误均为需逐文件手动改代码逻辑的 pre-existing 项 (E501 行太长 120, E701 多语句 29, F841 未用变量 17 等)。F822 仍有 1 个 pre-existing (v7/adapters.py `weights_to_daily_shares`)。

---

## 五、核心经验教训

| # | 教训 | 来源 |
|---|------|------|
| 1 | **动态仓位是 #1 alpha 源** (71% 贡献, Brinson 验证) | v9 |
| 2 | **Jump Model 应 per-asset 处理**，聚合 signal 导致高触发率 | v8 |
| 3 | **TV-PR 时变 beta 是最有效因子择时方法** (CV% 16.6% PASS) | v7 |
| 4 | **Vol-parity 多策略组合 > 任何单策略** (低相关性叠加) | v10 |
| 5 | **`resample('D').ffill()` 是陷阱** - 稀释收益虚增 Sharpe 23% | v10 审计 |
| 6 | **`freq='W'` 默认值是陷阱** - 周频数据用 252/n 年化虚高 Sharpe ~2x | v4 审计 |
| 7 | **v3 教训: 架构 != 业绩**，1/N 等权不如单策略 | v3 |
| 8 | **v6_2 教训: "PROMISING" 必须经过 CV% 测试**才能生产 | v6_2 |
| 9 | **归因必须凭数据, 不能靠记忆** | v8 修复历程 |
| 10 | **指标计算必须统一口径**，6 种实现是 bug 根源 | 全审计 |

---

## 六、产出文件索引

### 核心代码

| 路径 | 说明 |
|------|------|
| `common/metrics.py` | 统一指标计算 `compute_metrics()` + `detect_freq()` |
| `common/backtest_config.py` | 统一 VolTargeting, TrendFilter, CostConfig |
| `common/strategy_engine.py` | BaseStrategy + StrategyEngine |
| `core/momentum.py` | 动量信号核心 |
| `core/portfolio.py` | 组合管理核心 |
| `core/backtest.py` | 回测引擎 |
| `core/strategy_versions.py` | 版本注册与调度 |
| `v10/dynamic_weight_schemes.py` | v10 4 策略 Vol-parity + 5 动态权重方案 |
| `v11/v11_strategy.py` | v11 5 层架构策略 |
| `v7/tvpr_estimator.py` | TV-PR 时变 beta 估计器 |

### 文档

| 路径 | 说明 |
|------|------|
| `docs/77-v0_v10_codebase_audit.md` | 全面审计报告 (9 bugs + 修复前后对比) |
| `docs/78-refactoring-plan.md` | 重构方案 |
| `docs/54-v1_v9_strategy_summary.md` | v1-v9 因子/措施汇总 |
| `docs/63-final_summary.md` | v0-v10 综合汇总 (2026-07-24) |
| `docs/75-v10_results.md` | v10 最终结果 |
| `docs/58-v11_mega_design.md` | v11 统一大策略设计 |
| `VERSION_TRACKING.md` | 版本跟踪 (性能/状态/依赖) |
| `docs/79-strategy_development_summary.md` | 本文档 |

---

*文档版本: 1.0*
*日期: 2026-07-28*
