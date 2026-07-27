# docs/research_history/ — 行业轮动策略研发全周期复盘

> **覆盖范围**：momentum_etf_rotation 策略从 V0 → V10 的完整研发链路
> **时间跨度**：2026-07-07 ~ 2026-07-24（18 天，971 commit）
> **核心结论**：4 策略 Vol-parity（v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%）以 OOS Sharpe **1.991** 成为生产首选

---

## 文档目录

| 文档 | 主题 | 状态 | 目标读者 |
|------|------|------|----------|
| [00_TIMELINE.md](./00_TIMELINE.md) | V0→V10 全时间轴 + 一图概览 + 关键决策 | ✅ 完成 | 所有人（入口）|
| [01_DATA_FOUNDATION.md](./01_DATA_FOUNDATION.md) | 数据准备全链路（OHLCV/Proxy/因子库/Pipeline） | ✅ 完成 | 数据工程师、策略研发 |
| [02_V0_V3.md](./02_V0_V3.md) | V0–V3 策略期（动量 → slope_r² → 多策略架构） | ✅ 完成 | 策略研发 + 历史阅读 |
| [03_V4_V6.md](./03_V4_V6.md) | V4–V6 策略期（Smart β → 量价 → IC 加权 → 正交化） | ✅ 完成 | 策略研发 + 因子研究 |
| [04_V7_V10.md](./04_V7_V10.md) | V7–V10 策略期（TV-PR → 树模型失败 → 引擎收敛 → Vol-parity） | ✅ 完成 | 策略研发 + 工程师 |
| [05_LESSONS_LIBRARY.md](./05_LESSONS_LIBRARY.md) | 教训库（方法论 / 工程 / 流程 三大类，共 60+ 条） | ✅ 完成 | 策略研发 + 流程设计 |
| [06_RESEARCH_SOP.md](./06_RESEARCH_SOP.md) | 研发流程 SOP（数据 → 因子 → 选股 → 加权 → 风控 → OOS → 硬化） | ✅ 完成 | 新策略研发（流程模板）|
| [07_INVENTORY.md](./07_INVENTORY.md) | 关键资产索引（脚本 / 代码 / 报告 / 数据 / HTML） | ✅ 完成 | 资料检索 |
| [08_FUTURE_WORK.md](./08_FUTURE_WORK.md) | STAGE33 规划 + 余下机会 | ✅ 完成 | 后续研发 |
| [**09_QUICK_START.md**](./09_QUICK_START.md) | **🆕 10 条金科玉律 + 8 工序 + 5 闸门 + 4 步 OOS 快速参考** | ✅ 完成 | **新人/自动研究必读** ⭐ |

---

## 快速导航（按"我是谁"）

### 我是新接手者，今天就要开始一个策略
1. **[09_QUICK_START.md](./09_QUICK_START.md)** ⭐⭐⭐ ← **从这里开始**（10 条金科玉律 + 决策树 + 提交清单）
2. 然后看 [06_RESEARCH_SOP.md](./06_RESEARCH_SOP.md) 完整 SOP
3. 任何具体细节查 [07_INVENTORY.md](./07_INVENTORY.md)

### 我是策略新人，想快速理解整体研发历程
1. **[00_TIMELINE.md](./00_TIMELINE.md)** ← 从这里开始（一图概览 + 时间轴）
2. **[05_LESSONS_LIBRARY.md](./05_LESSONS_LIBRARY.md)**（快速浏览"方法论类"）
3. **[04_V7_V10.md §D.6](./04_V7_V10.md)**（看 v10 落地与工具沉淀）

### 我想复用工具 / 写新策略
1. **[06_RESEARCH_SOP.md](./06_RESEARCH_SOP.md)**（按 SOP 流程一步步推进）
2. **[07_INVENTORY.md](./07_INVENTORY.md)**（找现成组件）
3. **[04_V7_V10.md §D.6](./04_V7_V10.md)**（统一引擎 + YAML + walk_forward 的复用方法）
4. **[08_FUTURE_WORK.md](./08_FUTURE_WORK.md)**（看 STAGE33 计划）

### 我想理解"为什么 v6.2 被 DEPRECATED / 为什么 v7.7 树模型失败"
- **[03_V4_V6.md §C.5](./03_V4_V6.md)**（V4-V6 教训）
- **[04_V7_V10.md §D.5](./04_V7_V10.md)**（V7-V10 教训）
- **[05_LESSONS_LIBRARY.md](./05_LESSONS_LIBRARY.md)**（跨阶段教训库）

### 我想理解数据链路 / 因子构建
- **[01_DATA_FOUNDATION.md](./01_DATA_FOUNDATION.md)**（数据准备全链路 + 因子库清单）

### 我想做代码 Review / 安全审计
- **[05_LESSONS_LIBRARY.md §5.2](./05_LESSONS_LIBRARY.md)**（工程类教训：look-ahead / NaN / 频率对齐）

---

## 核心方法论资产

### 单策略最佳：v1.0 locked（V2 重构期）

```
Stage 12A hybrid (price + slope×R²) 50/50
  + VolTargeting (target_vol=0.15, scale∈[0.3,1.5])
  + Cost Model (commission 5bp + slippage 10bp)
  = OOS Calmar 1.79 / Sharpe 1.51 / MaxDD -1.94% ⭐
```

- 代码：`QuantNodes/strategy/momentum_etf_rotation/v2/portfolio_v2.py` + `momentum_v2.py`
- 文档：`reports/momentum_etf_rotation/v2/stage12a_report.md`
- 角色：**极致防御**

### 单策略激进：v7.10 TV-PR（Stage 32 硬化期）

```
X[T,N,K] 面板（12 宏观 + 24 量价 = 36 因子）
  + TV-PR 时变 β 估计（expanding-window, ADMM）
  + 标准化（宏观时序 Z-score + PV 截面 Z-score + Winsorize）
  + Stop Loss (-15% DD, cooldown 5 周)
  + 起点依赖测试 CV% < 25% (16.6% PASS)
  = OOS Sharpe 1.40 / Calmar 2.18 / DD -16.5%
```

- 代码：`QuantNodes/strategy/momentum_etf_rotation/v7/tvpr_estimator.py` + `macro_substrategy_v7_6.py`
- 文档：`reports/momentum_etf_rotation/STAGE32_PLAN.md` + `v7_10_*.md`
- 角色：**激进 alpha**

### 跨资产防御：DualMom（v10 第 4 策略）

```
Antonacci GEM 模型：
  - A 股 (510300) / 美股 (513100) / 黄金 (518880) / 国债 (511260)
  - 4 大类资产
  - 12 个月动量 rank, MOM > 0 持有，< 0 退到国债
  - 预热期 1/4 等权 + 3/4 国债
  = Sharpe 1.276
```

- 代码：`QuantNodes/strategy/momentum_etf_rotation/v10/dual_momentum.py`
- 角色：**跨资产防御**

### 宏观择时：v9macro（v10 第 2 策略）

```
5 宏观因子（增长/通胀/汇率/利率/信用）
  + 熵权法（信息熵越小权重越大）
  + 银河方案（Dynamic Position: z_score→position）
  + Jump Model（regime-aware 仓位调整）
  = Sharpe 0.962
```

- 代码：`QuantNodes/strategy/momentum_etf_rotation/v9/macro_layer.py` + `factor_galaxy.py`
- 角色：**宏观择时**

### 组合：4 策略 Vol-parity（v10 生产首选）

```
基础权重 {v1.0:0.74, v7.10:0.09, v9macro:0.12, DualMom:0.05}
target_vol = 0.08
每子策略 ≈ target_vol / 子策略波动率
= OOS Sharpe 1.991 ⭐
```

- 代码：`QuantNodes/strategy/momentum_etf_rotation/v10/portfolio_layer.py`
- 文档：`reports/momentum_etf_rotation/combo/STRATEGY_ITERATION_RECORD.html`
- 角色：**生产首选**

---

## 关键工具沉淀

| 工具 | 路径 | 复用价值 |
|------|------|---------|
| `BaseStrategy + StrategyEngine` | `QuantNodes/strategy/momentum_etf_rotation/common/strategy_engine.py` | v1-v10 所有策略接入点 |
| `run_from_yaml()` | `QuantNodes/strategy/momentum_etf_rotation/common/config_runner.py` | 6 个 YAML 模板（v1.0/v2/v3/v4/v6/v7.10）|
| `walk_forward.py` | `QuantNodes/strategy/momentum_etf_rotation/common/walk_forward.py` | 990 行通用 OOS 框架（策略无关 + NO LOOKAHEAD）|
| `ic_utils.py` | `QuantNodes/strategy/momentum_etf_rotation/common/ic_utils.py` | 截面 vs 时序 IC 统一计算 |
| `risk_parity.py` | `QuantNodes/strategy/momentum_etf_rotation/common/risk_parity.py` | `solve_risk_parity()` + `solve_max_diversification()` |
| `covariance.py` | `QuantNodes/strategy/momentum_etf_rotation/common/covariance.py` | 4 方法（Ledoit-Wolf / EWMA / 样本 / 对角）|
| `extended_metrics.py` | `QuantNodes/strategy/momentum_etf_rotation/common/extended_metrics.py` | 17 指标 |
| `data_loader_v7_6.py` | `QuantNodes/strategy/momentum_etf_rotation/v7/data_loader_v7_6.py` | TV-PR 面板构造 |
| `nav_curves_html.py` | `QuantNodes/strategy/momentum_etf_rotation/combo/nav_curves_html.py` | 9 策略精简版 HTML 生成 |
| `strategy-research/` | `research/strategy-research/` | 7 个工具复用 + 41 个新算子 |

---

## 关键文档资产

### 设计文档（docs/）

| 文档 | 行数 | 主题 |
|------|------|------|
| `38-v7_3_macro_only.md` | 819 | v7_macro_baseline 锁定声明 |
| `44-StrategyResearch设计文档.md` | 612 | 通用策略自动研究框架 |
| `45-StrategyResearch工具复用设计文档.md` | 452 | 7 工具复用 + 41 算子 |
| `49-v9_cycle_timing.md` | 457 | 宏观周期择时 |
| `39-v7_6_tvpr.md` | 254 | TV-PR 数学公式 + 实验设计 |
| `40-v7_6_sensitivity.md` | 330 | 10 阶段敏感性测试设计 |
| `41-v7_6_factor_ic_and_enhancement.md` | 411 | 因子 IC 评估框架 |
| `43-v7_7_lgbm.md` | 368 | PyCaret 25 模型对比 |
| `46-v8_ml_design.md` | 410 | v8 ML 因子择时 5 方向 |
| `57-v10_final_design.md` | 288 | 用户确认版 5 层架构 |
| `16-TV-PR迭代记录v7_6到v7_9.md` | 249 | TV-PR 全期迭代记录 |

### 阶段报告（reports/）

| 文档 | 行数 | 主题 |
|------|------|------|
| `STAGE_SUMMARY.md` | 236 | Stage 12A 之前总览 |
| `STAGE32_PLAN.md` | 121 | V7.10 硬化 5 P0 任务 |
| `STAGE33_PLAN.md` | 145 | 新因子 + HMM + 跨资产 + 代码清理 |
| `STAGE17_22_INDEX.md` | 380+ | Stage 17→18→19→22 完整研究链 |
| `UNIFIED_V1V5_REPORT.md` | 224 | V1-V5 演进对比 |
| `v4/STAGE17_RESEARCH_INDEX.md` | 219 | 4 份研究汇总 |
| `v4/STAGE17_VALIDATION.md` | 173 | 6 模式对比 |
| `v4/STAGE19_LW_INTEGRATION.md` | 264 | LW 不显著优于 IC² |
| `v5/STAGE22_V5_REPORT.md` | 410 | 11 量价因子 |
| `v6_2/STAGE29_PROMOTION.md` | — | v6.2 PROMISING→研究版本 |
| `v7_6_sensitivity_report.md` | — | 5 个 🔴 红色 Phase |
| `v7_10_oos_validation.md` | 118 | V7.10 OOS 详细 |
| `v7_10_cv_test.md` | 58 | 3 起点 CV% PASS |
| `v9/strategy_factor_analysis.md` | 184 | 9 策略核心机制 + 10 因子 |

---

## 研发链路 8 阶段（详见 06_RESEARCH_SOP.md）

```
数据 → IC → 因子 → 选股 → 加权 → 风控 → 组合 → OOS → 硬化
 │    │     │      │      │      │      │      │     │
 [1]  [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]
 │    │     │      │      │      │      │      │     │
OHLCV IC   宏观   动量   等权   VT     Vol-  5-fold Stage
 Proxy 滚动  vs   + 反转   逆波   TF    parity  walk- 32
 池  IC    PV    + 行业   动率   SL     4策略 forward 硬化
 缺  自相关 截面  ABC    IC    DCC   组合   expan- CV%
 数  阈值   时序  子策   IC-IR  regime         ding  死代码
 据  |IC|>  对数  略单  正交化 overlay         wind   清理
 扰  0.05   变换  测                              工厂
 动                                            函数

5 道闸门：
1. 数据闸门（OHLCV 前复权 / 动态资产池 / 缺数据审计）
2. IC 闸门（单因子 IC / 滚动 IC / 去重 / 阈值过滤）
3. 因子闸门（宏观时序 vs PV 截面 / ADMM 正交化慎用）
4. OOS 闸门（single → 5-fold → walk-forward → expanding + CV% < 25%）
5. 硬化闸门（起点依赖测试 / 死代码清理 / 文档 / 工厂函数）
```

---

## 写作纪律

本目录文档均依据 Subagent A/B/C/D 调研报告 + `git log --all` + 现实代码 / 报告 / 设计文档交叉验证。

**所有 commit hash 精确到 7 位短 hash**。
**所有数字指标均来自实际文件中的 `.md` / `.csv` / `.json` / `.parquet` 报告**。

---

## 致未来

> "复盘的真正价值不是记住所有细节，而是让下次决策时少踩一些坑。"
> —— Subagent B 报告末尾语

下一步：**[06_RESEARCH_SOP.md](./06_RESEARCH_SOP.md)** 是直接可复用的研发流程模板；
**[08_FUTURE_WORK.md](./08_FUTURE_WORK.md)** 是 `fd61982` STAGE33 规划的具体落地清单。
