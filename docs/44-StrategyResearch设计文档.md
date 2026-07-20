# StrategyResearch — 通用策略自动研究框架设计文档

> **版本**: v1.0
> **日期**: 2026-07-20
> **状态**: 设计完成，待实施

---

## 一、项目概述

### 1.1 目标

构建 QuantNodes 的顶级组件 `StrategyResearch` — 一个通用的策略自动研究框架。

核心理念: **Karpathy autoresearch 极简 + 多 Agent 增强 + 因子研发流水线**。

### 1.2 设计原则

- **Karpathy 极简**: 框架提供工具和循环指引，不调 LLM。外部 Agent 读 prompt 后自主决策。
- **Skill/Harness 模式**: 不是 Pipeline 编排器，而是 Agent 的 "研究实验室"。
- **因子全生命周期**: 假设 → 因子发现 → 因子验证 → 策略集成 → 回测 → 反思。
- **自适应策略**: 因子少时优先外部搜索+LLM，充足时本地算子挖掘。
- **DuckDB 统一存储**: 行情数据、因子数据、验证缓存、回测结果统一管理。
- **复用优先**: 最大化复用现有组件 (12 个零改动，5 个微调)。

### 1.3 与现有系统的关系

```
QuantNodes 现有系统
├── research/quant_alpha/     — 因子挖掘 (AlphaGPT/MCTS/AlphaLogics)
├── agent/tools/              — Agent 工具 (alpha_evaluate/factor/backtest)
├── agent/workflows/          — Workflow 引擎 (StepAgent/WorkflowSpec)
├── strategy/momentum_etf_rotation/ — ETF 轮动策略 (v1-v7.10)
└── research/strategy_research/     — 新增: 通用策略研究框架
```

`StrategyResearch` 是一个 **上层编排**，复用现有组件，不修改它们。

---

## 二、架构设计

### 2.1 顶层结构

```
research/strategy_research/
│
├── data.duckdb                    # 全局共享 DuckDB
│
├── core/                          # 框架核心 (纯工具，无 LLM)
│   ├── __init__.py
│   ├── engine.py                  # 实验循环引擎
│   ├── db.py                      # DuckDB 连接 + 读写工具
│   ├── factor_discover.py         # 本地因子发现
│   ├── factor_validate.py         # 因子验证 (IC/IR + 6维 + 缓存)
│   ├── factor_search.py           # 外部因子搜索
│   ├── factor_integrate.py        # 因子集成 + 面板重建
│   ├── backtest.py                # 回测执行
│   ├── pareto.py                  # Pareto 追踪器
│   ├── parallel.py                # 并行执行
│   ├── git.py                     # Git 操作
│   └── discovery.py               # 策略目录发现
│
├── strategies/                    # 每策略一个目录
│   └── etf_rotation/
│       ├── prepare.py             # 数据加载 + 评估 (固定)
│       ├── strategy.py            # PARAMS + FACTOR_EXPRS + FACTOR_WEIGHT_METHOD
│       └── results.tsv            # 实验记录
│
├── prompts/                       # Prompt 集中存储
│   ├── base/                      # 通用角色 (跨策略复用)
│   │   ├── researcher.md          # 研究员: 评估状态 + 选择策略 + 提假设
│   │   ├── factor_analyst.md      # 因子分析员: 发现 + 验证 (3条路径)
│   │   ├── strategist.md          # 策略师: 集成因子 + 优化参数 + 移除因子
│   │   ├── critic.md              # 评论员: 评估 + 风控 + 抗过拟合
│   │   └── coordinator.md         # 协调员: 编排循环 + 停止条件
│   │
│   └── etf_rotation/              # 策略专属
│       └── program.md             # 完整实验指引
│
└── tests/
    ├── test_engine.py
    ├── test_db.py
    ├── test_factor_validate.py
    ├── test_factor_search.py
    ├── test_factor_integrate.py
    ├── test_pareto.py
    └── strategies/
        └── test_etf_rotation.py
```

### 2.2 组件职责

| 组件 | 职责 | 复用来源 |
|------|------|---------|
| engine.py | 实验循环 (git commit/reset + 记录) | VersionManager + GitOpsTool |
| db.py | DuckDB 连接 + 读写 | DuckDBNode + factor_library |
| factor_discover.py | 本地因子发现 | AlphaGPT + MCTS |
| factor_validate.py | IC/IR + 6维评分 + 缓存 | alpha_evaluate + polars_evaluator |
| factor_search.py | 外部因子搜索 + 提取 | web_search + web_fetch + report_reproducer |
| factor_integrate.py | 因子集成 + 面板重建 | enhanced_factors + data_loader_v7_6 |
| backtest.py | 回测执行 | v7.6 backtest |
| pareto.py | Pareto 前沿追踪 | 新建 |
| parallel.py | 并行回测 | parallel_evaluate |
| git.py | git commit/reset/head | GitOpsTool + VersionManager |
| discovery.py | 策略目录发现 | strategy_dir + list_strategies |

---

## 三、五阶段流水线

### 3.1 流水线概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  StrategyResearch 5-Stage Pipeline                       │
│                                                                         │
│  Stage 1: HYPOTHESIS                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Researcher 提出假设                                         │        │
│  │  输入: 因子池状态 + 历史实验 + 上轮反馈                        │        │
│  │  输出: {action, hypothesis, skip_factor_discovery}           │        │
│  │  skip_factor_discovery=true 时直接跳到 Stage 3               │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          ↓                                              │
│  Stage 2: FACTOR RESEARCH (发现 + 验证，合并)                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Factor Analyst 发现并验证因子                                │        │
│  │  Step 1: 生成候选因子 (路径A/B/C)                             │        │
│  │  Step 2: 逐个 IC/IR 验证 (IC > 0.03, IR > 0.5) + 缓存       │        │
│  │  Step 3: 6 维评分 (Return/Stability/Diversification/...)     │        │
│  │  Step 4: Mutual IC 去重 (|corr| < 0.7)                       │        │
│  │  Step 5: IC 衰减检查 (IC_5d >= 30% * IC_1d)                  │        │
│  │  输出: 通过验证的因子列表                                     │        │
│  │  ⏭️ skip_factor_discovery=true 时跳过此阶段                   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          ↓                                              │
│  Stage 3: STRATEGY INTEGRATION                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Strategist 集成因子 / 优化参数                               │        │
│  │  因子集成: 新因子加入 FACTOR_EXPRS + 重建面板                  │        │
│  │  参数优化: 修改 PARAMS (条件触发)                              │        │
│  │  因子移除: 移除低 IR 因子 (Critic 建议时)                      │        │
│  │  输出: 更新后的 strategy.py                                    │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          ↓                                              │
│  Stage 4: BACKTEST                                                      │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  执行策略回测                                                 │        │
│  │  python strategy.py > run.log                                │        │
│  │  输出: calmar, sharpe, max_dd, ann_return, turnover          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          ↓                                              │
│  Stage 5: REFLECTION                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Critic 评估结果                                              │        │
│  │  - 风控阈值检查 (MaxDD/Calmar/Sharpe/权重/换手/因子数)        │        │
│  │  - 4 项抗过拟合检验 (起点/偏移/扰动/消融)                     │        │
│  │  - 判断: keep (git commit) / discard (git reset)             │        │
│  │  输出: {verdict, direction, suggestions}                     │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          ↓                                              │
│  回到 Stage 1                                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage 1: HYPOTHESIS

**角色**: Researcher

**职责**:
- 分析当前因子池状态 (因子数、覆盖维度)
- 选择行动策略 (自适应)
- 提出研究假设

**自适应策略选择**:

| 条件 | 行动 | 原因 |
|------|------|------|
| 因子数 < 20 或 覆盖 < 60% | search_external | 优先外部搜索+LLM，快速补充 |
| 因子数 >= 20 且 覆盖 >= 60% | discover_local | 本地算子挖掘，精细探索 |
| 因子充足但参数不优 | optimize_param | 直接优化参数 |
| 因子过多 (>30) | remove_factor | 精简因子池 (少见) |

**覆盖维度定义** (6 类):

| 维度 | 示例因子 | 检查方式 |
|------|---------|---------|
| 动量 | ts_return(close, N) | 因子名/category 包含 momentum |
| 反转 | short_term_reversal | 包含 reversal |
| 波动率 | realized_vol, atr | 包含 volatility |
| 流动性 | amihud, turnover | 包含 liquidity |
| 量价 | price_volume_corr | 包含 volume_price |
| 宏观 | 宏观增长因子 | 包含 macro |

覆盖度 = 有因子的维度数 / 6

**输出格式**:
```json
{
  "action": "search_external | discover_local | optimize_param | remove_factor",
  "skip_factor_discovery": false,
  "discovery_reason": "因子数 12 < 20, 覆盖 3/6, 缺少波动率/流动性/宏观, 优先外部搜索",
  "hypothesis": "缺少尾部风险度量因子，搜索 realized skewness 相关研究",
  "factor_direction": "波动率类因子",
  "search_query": "realized skewness ETF tail risk factor",
  "search_sources": ["arxiv", "sscn"],
  "params_to_try": null,
  "factor_to_remove": null,
  "expected": "预期效果"
}
```

**skip_factor_discovery 逻辑**:
- 当 `action=optimize_param` 时，`skip_factor_discovery=true`，直接跳到 Stage 3
- 当 `action=remove_factor` 时，`skip_factor_discovery=true`，直接跳到 Stage 3
- 当 `action=search_external` 或 `discover_local` 时，`skip_factor_discovery=false`

### 3.3 Stage 2: FACTOR RESEARCH

**角色**: Factor Analyst

**职责**: 通过合适路径发现因子并验证

**⏭️ 跳过条件**: 当 `skip_factor_discovery=true` 时，直接跳到 Stage 3

**三条发现路径**:

| 路径 | 触发条件 | 方法 | 复用组件 |
|------|---------|------|---------|
| A: 本地算子挖掘 | 因子充足，精细探索 | MCTS 285 算子组合 | AlphaGPT + MCTS |
| B: 外部知识搜索 | 因子不足，快速补充 | web_search + 提取 | web_fetch + report_reproducer |
| C: LLM 直接建议 | 需要方向指引 | LLM 分析当前状态 | — |

**验证流程 (先单后批)**:

```
Step 0: LLM 分析当前状态 → 建议搜索方向
Step 1: 生成候选因子 (按路径 A/B/C)
Step 2: 逐个 IC/IR 验证 (带缓存)
  ├── 检查 validation_cache 表 (DuckDB)
  ├── 未缓存 → 执行验证 → 写入缓存
  └── 已缓存 → 直接用缓存结果
  通过条件: IC > 0.03, IR > 0.5
Step 3: 6 维评分
  Return (0.30) + Stability (0.20) + Diversification (0.20)
  + (1-Turnover) (0.15) + Monotonicity (0.10) + Coverage (0.05)
Step 4: 批量 Mutual IC 去重
  |Spearman corr| < 0.7 与已有因子
Step 5: IC 衰减检查
  IC_5d >= 30% * IC_1d
```

**IC 判断标准**:

| 指标 | Good | Fair | Poor |
|------|------|------|------|
| IC_mean | > 0.05 | 0.02-0.05 | < 0.02 |
| ICIR | > 0.5 | 0.3-0.5 | < 0.3 |
| IC_positive_ratio | > 60% | 50-60% | < 50% |
| t_stat (panel IC) | > 2.0 | 1.5-2.0 | < 1.5 |

**6 维评分详细**:

| 维度 | 指标 | 权重 | Pass 标准 |
|------|------|------|----------|
| Return | IC, ICIR, Rank IC | 0.30 | \|IC\| > 0.03, ICIR > 0.5 |
| Stability | Rolling IC mean/std | 0.20 | Stability > 0.6 |
| Diversification | 因子间 Spearman 相关 | 0.20 | 平均相关 < 0.7 |
| Turnover | 排名变化率 | 0.15 | < 0.5 |
| Monotonicity | 5 分位收益单调性 | 0.10 | > 0.7 |
| Coverage | 非空比例 | 0.05 | > 0.8 |

**输出格式**:
```json
{
  "path_used": "external",
  "candidates": [
    {
      "factor_name": "realized_skew_60d",
      "factor_code": "ts_skew(ts_return(close, 1), 60)",
      "category": "volatility",
      "source": "external:Ang2006",
      "ic_mean": 0.052,
      "ir": 0.85,
      "overall_score": 0.72,
      "passed": true,
      "notes": "IC稳定, 与现有因子低相关"
    }
  ],
  "rejected": [
    {
      "factor_name": "bad_factor",
      "reason": "IC 0.018 < 0.03"
    }
  ],
  "recommendation": "建议集成 realized_skew_60d"
}
```

### 3.4 Stage 3: STRATEGY INTEGRATION

**角色**: Strategist

**职责**: 将验证通过的因子集成到策略中，或优化参数，或移除因子

**操作类型 1: 因子集成 (action=search_external 或 discover_local)**

```
Step 1: 单独验证 (先单)
  - 每个因子单独加入 FACTOR_EXPRS
  - 执行回测验证
  - Calmar 改善 → 保留
  - Calmar 不变 → 标记 (可能与其他因子协同)
  - Calmar 退化 → 丢弃

Step 2: 批量集成 (后批)
  - 所有通过单独验证的因子一起加入
  - 选择权重方式: equal / inv_vol / ic_ir / risk_parity

Step 3: 面板重建
  - 根据因子类型写入对应 DuckDB 表:
    - market_ts → factor_market_timeseries
    - asset_ts → factor_asset_timeseries
    - cross_section → factor_cross_section
  - 更新 factor_registry 元数据

Step 4: 条件触发参数优化
  - 如果新增因子数 >= 3 → 自动优化 PARAMS
  - 如果权重方式变化 → 自动优化 PARAMS
```

**操作类型 2: 参数优化 (action=optimize_param)**

```
Step 1: 修改 PARAMS
  - lambda_tv / lambda_l1 / top_n / max_weight / vol_window / stop_loss_threshold
  - 根据 Researcher 建议的 params_to_try

Step 2: 执行回测验证
  - Calmar 改善 → keep
  - Calmar 不变或退化 → discard
```

**操作类型 3: 因子移除 (action=remove_factor，少见)**

```
Step 1: 识别低 IR 因子
  - IR < 0.3 的因子
  - 或 Critic 建议移除的因子

Step 2: 移除后回测验证
  - Calmar 不变或改善 → 确认移除
  - Calmar 退化 → 恢复
```

**输出**: 更新后的 strategy.py

### 3.5 Stage 4: BACKTEST

**执行**: `python strategy.py > run.log 2>&1`

**输出**: calmar, sharpe, max_dd, ann_return, ann_vol, sortino, turnover

### 3.6 Stage 5: REFLECTION

**角色**: Critic

**风控阈值**:

| 指标 | 阈值 | 说明 |
|------|------|------|
| MaxDD | <= -15% | 最大回撤上限 |
| Calmar | >= 0.5 | 收益/回撤比下限 |
| Sharpe | >= 0.3 | 风险调整收益下限 |
| 单 ETF 权重 | <= 25% | 集中度上限 |
| 年化换手 | <= 600% | 成本控制 |
| 因子数 | <= 30 | 避免维度爆炸 |

**抗过拟合检验 — 4 项检验**:

| # | 检验 | Pass 标准 | 实现 |
|---|------|----------|------|
| 1 | 起点依赖 | 3 起点 CV% < 25% | 从多个起点 (2019/2020/2022/2023) 运行回测，计算 Calmar CV |
| 2 | 调仓日偏移 | ±5 日 Calmar 稳定 | 偏移 -5/-3/0/+3/+5 交易日，Calmar CV <= 15% |
| 3 | 参数扰动 | ±10% 退化 < 20% | 扰动 lookback/corr_threshold/a_share_cap，所有 Calmar > 0.4 |
| 4 | 消融实验 | 每关一项退化 >= 5% | 逐个关闭规则，检查每个消融退化 Calmar >= 5% |

**ValidationConfig 数据结构**:

```python
@dataclass
class ValidationConfig:
    start_points: tuple[str, ...] = ("2019-01-01", "2020-06-01", "2022-01-01", "2023-06-01")
    perturb_lookbacks: tuple[int, ...] = (80, 100, 120)
    perturb_corr_thresholds: tuple[float, ...] = (0.85, 0.90, 0.95)
    perturb_a_share_caps: tuple[int, ...] = (2, 3, 4)
    rebalance_offsets: tuple[int, ...] = (-5, -3, 0, 3, 5)
    calmar_cv_threshold: float = 0.25        # 起点依赖 CV 阈值
    calmar_floor: float = 0.4                 # 参数扰动 Calmar 下限
    ablation_drop_threshold: float = 0.05     # 消融退化阈值
    min_calmar_threshold: float = 0.4
```

**5 维风险评级框架**:

| 维度 | 检验 | Red 标准 |
|------|------|---------|
| A. 参数敏感性 | Phase 1 | lambda_tv 0.01→0.05, OOS Calmar 退化 > 50% |
| B. 起点稳定性 | 现有数据 | CV% > 35% |
| C. 时间衰减 | Phase 2 | 2024-2026 vs 2022-2024 Calmar 比值 < 0.5 |
| D. Bootstrap 稳定性 | Phase 3 | Calmar std/mean > 50% |
| E. 数据鲁棒性 | Phase 4 | 20% X mask Calmar 退化 > 50% |

**风险评级**:

| 级别 | 行为 | 结论 |
|------|------|------|
| **Green (低)** | 单参数偏离 50%, CV% 退化 < 30%; 多段 hold-out Calmar 接近 | 真实可靠，可锁定 |
| **Yellow (中)** | 30-50% 退化; 部分起点退化 | 需缩小 lambda 范围重新测试 |
| **Red (高)** | > 50% 退化; 起点 2022 Calmar < 0; bootstrap std/mean > 50% | 严重过拟合，需重新设计 |

**判断**:
- calmar 改善 + 风险评级 Green → keep (git commit)
- calmar 不变或退化 → discard (git reset)
- calmar 改善但风险评级 Yellow/Red → discard + 警告

**输出**:
```json
{
  "verdict": "keep",
  "analysis": "新因子 realized_skew_60d 提升 Calmar 从 0.662 到 0.710",
  "risk_rating": "Green",
  "pattern": "波动率类因子在当前市场环境下有效",
  "direction": "exploit",
  "suggestions": ["继续探索波动率类因子", "尝试 kurt_60d"],
  "risk_warnings": ["因子数已到 38，接近上限"]
}
```

### 3.7 Coordinator 编排逻辑

**角色**: Coordinator

**职责**: 编排 Researcher → Factor Analyst → Strategist → Critic 循环

**循环流程**:
```
1. 调用 Researcher → 获取 action + hypothesis
2. 如果 skip_factor_discovery=false:
   调用 Factor Analyst → 获取验证通过的因子
3. 调用 Strategist → 更新 strategy.py
4. 执行 Backtest → 获取指标
5. 调用 Critic → 获取 verdict + suggestions
6. 记录到 results.tsv + DuckDB
7. 回到 Step 1
```

**停止条件**:
- 连续 5 轮无 Calmar 改善
- 达到最大轮数 (默认 50 轮)
- 用户中断 (Ctrl+C)
- 风控阈值被触发 (如 MaxDD > -15%)

**结果记录**:
- `results.tsv`: 每轮一行，记录 commit/round/action/metrics/status
- `DuckDB.backtest_results`: 详细指标 + 净值曲线
- `DuckDB.backtest_nav`: 净值曲线时间序列

---

## 四、DuckDB 存储规范

### 4.1 三类数据区分

| 类型 | 说明 | 示例 | 计算方式 |
|------|------|------|---------|
| 行情数据 | 原始市场数据，不可变 | OHLCV, 宏观指标 | 外部加载 |
| 时序因子 | 基于时间窗口的滚动计算 | ts_return, ts_std, ts_corr | 每个资产独立计算 |
| 截面因子 | 基于截面排名/标准化 | rank, zscore, scale | 每个日期跨资产计算 |

时序因子进一步分为:
- **市场级时序因子**: 全市场一个值 (如 VIX, DXY, 宏观指标)
- **资产级时序因子**: 每个资产各自独立计算 (如 个股动量, 个股波动率)

### 4.2 表结构

#### 行情数据表

```sql
-- 日频 OHLCV
CREATE TABLE market_ohlcv (
    date DATE NOT NULL,
    code VARCHAR NOT NULL,           -- "510300"
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    vwap DOUBLE,
    PRIMARY KEY (date, code)
);

-- 日频宏观指标 (市场级，每日期一行)
CREATE TABLE market_macro_daily (
    date DATE NOT NULL PRIMARY KEY,
    dxy DOUBLE,
    vix DOUBLE,
    real_rate DOUBLE,
    credit_spread DOUBLE,
    term_spread DOUBLE
);

-- 周频宏观因子 (市场级，每日期一行)
CREATE TABLE market_macro_weekly (
    date DATE NOT NULL PRIMARY KEY,
    宏观增长因子 DOUBLE,
    宏观通胀因子_生活端 DOUBLE,
    宏观通胀因子_生产端 DOUBLE,
    无风险收益率 DOUBLE,
    信用利差因子 DOUBLE,
    期限利差因子_债 DOUBLE,
    期限利差因子_股 DOUBLE,
    宏观汇率因子 DOUBLE
);
```

#### 因子数据表

```sql
-- 1. 市场级时序因子 (无资产维度，全市场一个值)
CREATE TABLE factor_market_timeseries (
    date DATE NOT NULL,
    factor_name VARCHAR NOT NULL,    -- "vix", "dxy", "market_return_20d"
    factor_code VARCHAR NOT NULL,    -- "vix", "ts_return(close, 20)"
    value DOUBLE,
    PRIMARY KEY (date, factor_name)
);

-- 2. 资产级时序因子 (per-asset，每个资产各自独立计算)
CREATE TABLE factor_asset_timeseries (
    date DATE NOT NULL,
    asset_code VARCHAR NOT NULL,     -- "510300", "159740"
    factor_name VARCHAR NOT NULL,    -- "momentum_20d", "realized_vol_20d"
    factor_code VARCHAR NOT NULL,    -- "ts_return(close, 20)"
    value DOUBLE,
    PRIMARY KEY (date, asset_code, factor_name)
);

-- 3. 截面因子 (per-date 跨资产计算)
CREATE TABLE factor_cross_section (
    date DATE NOT NULL,
    asset_code VARCHAR NOT NULL,     -- "510300", "159740"
    factor_name VARCHAR NOT NULL,    -- "rank_momentum_20d"
    factor_code VARCHAR NOT NULL,    -- "rank(ts_return(close, 20))"
    value DOUBLE,
    PRIMARY KEY (date, asset_code, factor_name)
);

-- 合并视图 (回测时使用)
CREATE VIEW factor_panel AS
    SELECT date, asset_code, factor_name, factor_code, value, 'market_ts' as factor_type
    FROM factor_market_timeseries
    CROSS JOIN (SELECT DISTINCT code as asset_code FROM market_ohlcv) assets
    UNION ALL
    SELECT date, asset_code, factor_name, factor_code, value, 'asset_ts' as factor_type
    FROM factor_asset_timeseries
    UNION ALL
    SELECT date, asset_code, factor_name, factor_code, value, 'cross_section' as factor_type
    FROM factor_cross_section;
```

**注意**: 市场级时序因子通过 `CROSS JOIN` 广播到所有资产。

#### 元数据表

```sql
-- 因子注册表
CREATE TABLE factor_registry (
    factor_name VARCHAR NOT NULL,    -- "momentum_20d" (唯一 ID)
    factor_code VARCHAR NOT NULL,    -- "ts_return(close, 20)" (表达式)
    factor_type VARCHAR NOT NULL,    -- "market_ts" / "asset_ts" / "cross_section"
    category VARCHAR,                -- "momentum" / "volatility" / "macro" / ...
    source VARCHAR,                  -- "v7_baseline" / "external" / "local" / "llm"
    window INTEGER,
    params JSON,
    strategy_name VARCHAR NOT NULL,  -- "etf_rotation"
    added_at TIMESTAMP,
    added_by VARCHAR,                -- "round_5" / "initial"
    PRIMARY KEY (strategy_name, factor_name)
);

-- 验证缓存
CREATE TABLE validation_cache (
    factor_name VARCHAR NOT NULL,
    factor_code VARCHAR NOT NULL,
    strategy_name VARCHAR NOT NULL,
    ic_mean DOUBLE,
    ic_std DOUBLE,
    ir DOUBLE,
    ic_decay_1d DOUBLE,
    ic_decay_5d DOUBLE,
    ic_decay_20d DOUBLE,
    rank_ic_mean DOUBLE,
    stability_score DOUBLE,
    diversification_score DOUBLE,
    turnover DOUBLE,
    monotonicity_score DOUBLE,
    coverage DOUBLE,
    overall_score DOUBLE,
    is_valid BOOLEAN,
    fail_reasons VARCHAR,
    validated_at TIMESTAMP,
    source VARCHAR,
    data_fingerprint VARCHAR,
    PRIMARY KEY (strategy_name, factor_name)
);

-- 数据指纹
CREATE TABLE data_fingerprint (
    table_name VARCHAR NOT NULL,
    strategy_name VARCHAR NOT NULL,
    fingerprint VARCHAR,
    updated_at TIMESTAMP,
    row_count INTEGER,
    PRIMARY KEY (table_name, strategy_name)
);
```

#### 回测输出表

```sql
-- 回测指标
CREATE TABLE backtest_results (
    strategy_name VARCHAR NOT NULL,
    round INTEGER NOT NULL,
    commit_hash VARCHAR,
    action VARCHAR,
    calmar DOUBLE,
    sharpe DOUBLE,
    max_dd DOUBLE,
    ann_return DOUBLE,
    ann_vol DOUBLE,
    sortino DOUBLE,
    turnover DOUBLE,
    factors_added INTEGER,
    factors_removed INTEGER,
    params_changed INTEGER,
    status VARCHAR,
    description VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (strategy_name, round)
);

-- 净值曲线
CREATE TABLE backtest_nav (
    strategy_name VARCHAR NOT NULL,
    round INTEGER NOT NULL,
    date DATE NOT NULL,
    nav DOUBLE,
    PRIMARY KEY (strategy_name, round, date)
);
```

### 4.3 字段命名规范

| 字段 | 含义 | 示例 |
|------|------|------|
| factor_name | 因子唯一 ID (英文) | `momentum_20d`, `vix`, `rank_momentum` |
| factor_code | 因子表达式 | `ts_return(close, 20)`, `vix`, `rank(ts_return(close, 20))` |
| factor_type | 因子类型 | `market_ts`, `asset_ts`, `cross_section` |
| asset_code | 资产代码 | `510300`, `159740` |
| strategy_name | 策略名称 | `etf_rotation` |
| category | 因子类别 | `momentum`, `volatility`, `macro` |
| source | 因子来源 | `v7_baseline`, `external`, `local`, `llm` |

### 4.4 因子分类逻辑

```python
TS_OPERATORS = {
    "ts_mean", "ts_std", "ts_corr", "ts_return", "ts_delta", "ts_rank",
    "ts_max", "ts_min", "ts_sum", "ts_cov", "ts_argmax", "ts_argmin",
    "delay", "delta", "ewm_mean", "ewm_std"
}

CS_OPERATORS = {
    "rank", "zscore", "scale", "winsorize", "normalize",
    "group_norm", "cross_sectional_mean", "cross_sectional_rank",
    "cross_sectional_std", "cross_sectional_zscore", "neutralize"
}

MACRO_KEYWORDS = {
    "dxy", "vix", "real_rate", "credit_spread", "term_spread",
    "宏观", "通胀", "增长", "汇率", "收益率"
}

def classify_factor(factor_name: str, factor_code: str) -> str:
    """判断因子类型: market_ts / asset_ts / cross_section

    规则:
    1. 因子名或表达式包含宏观关键词 → market_ts
    2. 只含 ts_ 算子 → asset_ts
    3. 含截面算子 → cross_section
    4. 混合 → 以最外层算子为准
    """
    for kw in MACRO_KEYWORDS:
        if kw in factor_name or kw in factor_code:
            return "market_ts"

    ops_found = set()
    for op in TS_OPERATORS | CS_OPERATORS:
        if op + "(" in factor_code:
            ops_found.add(op)

    ts_found = ops_found & TS_OPERATORS
    cs_found = ops_found & CS_OPERATORS

    if ts_found and not cs_found:
        return "asset_ts"
    elif cs_found and not ts_found:
        return "cross_section"
    elif ts_found and cs_found:
        last_ts = max(factor_code.rfind(op) for op in ts_found)
        last_cs = max(factor_code.rfind(op) for op in cs_found)
        return "asset_ts" if last_ts > last_cs else "cross_section"
    else:
        return "asset_ts"
```

### 4.5 缓存策略

| 数据类型 | 缓存方式 | 失效条件 |
|---------|---------|---------|
| 行情数据 | DuckDB 表 (持久化) | 手动更新 |
| 时序因子 | DuckDB 表 (持久化) | factor_registry 变化 |
| 截面因子 | DuckDB 表 (持久化) | factor_registry 变化 |
| 验证缓存 | DuckDB 表 (持久化) | data_fingerprint 变化 |
| 回测输出 | DuckDB 表 (持久化) | 永不失效 (历史记录) |

### 4.6 VIEW vs TABLE 性能对比

需要对比 `factor_panel` 作为 VIEW 和 TABLE 的查询性能:

- **VIEW**: 实时合并，不占额外空间，查询可能稍慢
- **TABLE**: 预计算，查询快，但需要同步重建

测试查询:
1. 单因子查询 (最常用)
2. 多因子查询
3. 全因子面板
4. 因子相关性计算

决策: 先用 VIEW，如性能不可接受再改 TABLE。

---

## 五、策略配置文件

### 5.1 strategy.py — Agent 可改的文件

```python
"""
ETF 轮动 v7.10 策略配置。
Agent 可以修改: PARAMS, FACTOR_EXPRS, FACTOR_WEIGHT_METHOD
"""

# ============================================================
# 策略参数 (Agent 可改)
# ============================================================
PARAMS = {
    "lambda_tv": 0.15,
    "lambda_l1": 0.05,
    "top_n": 10,
    "max_weight": 0.25,
    "vol_window": 26,
    "stop_loss_threshold": -0.10,
}

# ============================================================
# 新增因子表达式 (Agent 可改)
# ============================================================
FACTOR_EXPRS = [
    # 示例:
    # {"factor_name": "momentum_20d", "factor_code": "ts_return(close, 20)", "category": "momentum"},
    # {"factor_name": "realized_vol_20d", "factor_code": "ts_std(ts_return(close, 1), 20)", "category": "volatility"},
]

# ============================================================
# 因子权重方式 (Agent 可改)
# ============================================================
FACTOR_WEIGHT_METHOD = "inv_vol"  # "equal" | "inv_vol" | "ic_ir" | "risk_parity"

# ============================================================
# 以下不改
# ============================================================
if __name__ == "__main__":
    from prepare import evaluate
    metrics = evaluate(PARAMS, FACTOR_EXPRS, FACTOR_WEIGHT_METHOD)
    print("---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
```

### 5.2 results.tsv — 实验记录

```
commit  round  action           calmar  sharpe  max_dd  ann_return  turnover  factors_added  factors_removed  params_changed  status  description
a1b2c3d 1      optimize_param   0.662   0.779   -0.12   0.085       0.32      0              0                1               keep    baseline
b2c3d4e 2      search_external  0.710   0.820   -0.11   0.092       0.35      2              0                0               keep    +realized_vol_20d +skew_60d
c3d4e5f 3      discover_local   0.695   0.810   -0.12   0.088       0.38      1              0                0               discard +momentum_60d (IC低)
```

---

## 六、Prompt 设计

### 6.1 Prompt 结构

```
prompts/
├── base/                          # 通用角色 (跨策略复用)
│   ├── researcher.md              # 研究员
│   ├── factor_analyst.md          # 因子分析员
│   ├── strategist.md              # 策略师
│   ├── critic.md                  # 评论员
│   └── coordinator.md             # 协调员
│
└── etf_rotation/                  # 策略专属
    └── program.md                 # 完整实验指引
```

### 6.2 Prompt 复用来源

| 新文件 | 复用来源 | 复用内容 |
|--------|---------|---------|
| researcher.md | alpha-gpt-idea-generator + reflector + mcts-reflector | 假设生成 + 反思模式 + 搜索方向 |
| factor_analyst.md | factor-analyst + evaluator + formula-translator | 因子发现 + IC验证 + 代码修改规则 |
| strategist.md | backtest-engineer + strategy-design | 回测流程 + 因子组合 + 参数敏感性 |
| critic.md | alpha-gpt-critic + risk-manager + quant-validation | 评分公式 + 风控阈值 + 4项检验 |
| coordinator.md | SOUL.md + cron_jobs | 编排模式 + 任务描述 |
| program.md | backtest-engineer 阈值 + risk-manager 风控 | 策略专属知识 |

### 6.3 各角色 Prompt 要点

#### researcher.md
- 评估因子池状态 (因子数、覆盖维度)
- 自适应策略选择 (外部搜索 vs 本地挖掘)
- 输出: action, hypothesis, factor_direction, search_query

#### factor_analyst.md
- 三条路径: 本地算子 / 外部搜索 / LLM建议
- 先单后批验证: 逐个 IC/IR → 批量去重
- 缓存: 已验证因子直接用缓存结果
- 输出: candidates, rejected, recommendation

#### strategist.md
- 因子加入: 先单后批
- 因子移除: Critic 建议时 (少见)
- 参数优化: 条件触发 (因子数>=3 或权重变化)
- 输出: action, new_factors, weight_method, param_changes

#### critic.md
- 风控阈值: MaxDD<=-15%, Calmar>=0.5, Sharpe>=0.3
- 抗过拟合: 4项检验 (起点/偏移/扰动/消融)
- 搜索方向: exploit/explore/diversify
- 输出: verdict, analysis, direction, suggestions

#### coordinator.md
- 编排 Researcher→Factor Analyst→Strategist→Critic 循环
- 停止条件: 连续5轮无改善 / 最大轮数 / 用户中断
- 记录: results.tsv + DuckDB

#### program.md (ETF轮动专属)
- 策略概述: v7.10 TV-PR, 17宏观+19量价
- 参数含义表
- 因子表达式语法
- 市场知识: A股T+1, 散户主导

---

## 七、复用分析

### 7.1 零改动复用 (12 个)

| 组件 | 文件 | 用途 |
|------|------|------|
| generate_run_id() | research/common/run_id.py | Run ID 生成 |
| StrategyRunRepository | monitor/storage/repository.py | 实验记录 (SQLite) |
| save_backtest_duckdb() | research/persist/strategy_library.py | 结果持久化 |
| ConfigBacktestRunner._save_dataframe() | backtest/config_runner.py | 输出保存 |
| extended_metrics() | common/extended_metrics.py | 17 指标计算 |
| parallel_evaluate() | core/parallel/worker.py | 并行执行 |
| strategy_dir() + list_strategies() | research/persist/strategy_library.py | 策略发现 |
| load_etf_nav_panel() | common/data.py | 数据加载 |
| StrategyConfig | agent/config/types.py | 参数 schema |
| read_strategy_yaml() | research/persist/strategy_library.py | YAML 读 |
| write_strategy_yaml() | research/persist/strategy_library.py | YAML 写 |
| VerifyConfig.weights | evaluation/contracts.py | 加权评分配置 |

### 7.2 微调复用 (5 个)

| 组件 | 文件 | 改动 |
|------|------|------|
| VersionManager._git_commit() | monitor/version/version_manager.py | 提取为独立同步函数 |
| GitOpsTool._run_git() | agent/tools/git_ops.py | 提取为独立同步函数 |
| RunnerSnapshot | core/parallel/worker_process.py | 适配参数快照 |
| PolarsAlphaCalculatorEvaluator.verify() | evaluation/evaluators/polars_evaluator.py | 提取维度打分函数 |
| validate_parameter_perturbation() | common/validation.py | 接受通用网格 |

### 7.3 新建

| 组件 | 说明 |
|------|------|
| ParetoTracker | Pareto 前沿追踪 |
| 6 个 Prompt 文件 | 组合自现有 prompt |
| db.py | DuckDB 工具函数 |
| factor_search.py | 外部因子搜索封装 |

### 7.4 代码量估算

| 类别 | 新增行数 | 复用行数 |
|------|---------|---------|
| 核心框架 | ~640 | ~410 |
| 策略目录 | ~130 | ~160 |
| Prompt 文件 | ~470 | (组合自 ~1195 行) |
| 测试 | ~460 | — |
| 总计 | ~1700 | ~570 |

---

## 八、迭代方案

### Phase 1: 基础闭环 (2-3 天)

```
core/engine.py + git.py + discovery.py + db.py
core/factor_validate.py (IC/IR 验证 + 缓存)
core/backtest.py
strategies/etf_rotation/prepare.py + strategy.py
prompts/base/researcher.md + critic.md + coordinator.md
prompts/etf_rotation/program.md
tests/test_engine.py + test_db.py + test_factor_validate.py
```

产出: "假设 → 参数修改 → 回测 → keep/discard" 闭环

### Phase 2: 因子流水线 (3-4 天)

```
core/factor_discover.py (本地发现)
core/factor_search.py (外部搜索)
core/factor_integrate.py (集成 + 面板重建)
prompts/base/factor_analyst.md + strategist.md
tests/test_factor_search.py + test_factor_integrate.py
```

产出: 完整 "发现 → 验证 → 集成 → 回测" 流水线

### Phase 3: Pareto 多目标 (2-3 天)

```
core/pareto.py
tests/test_pareto.py
```

产出: 多目标 Pareto 前沿追踪

### Phase 4: 并行 + 多策略 (2 天)

```
core/parallel.py
core/discovery.py 增强
strategies/new_strategy/ 模板
```

产出: 并行回测 + 多策略扩展

### 总时间线

```
Phase 1 (2-3 天): 基础闭环
Phase 2 (3-4 天): 因子流水线
Phase 3 (2-3 天): Pareto 多目标
Phase 4 (2 天):   并行+扩展
总计: 9-12 天
```

---

## 九、关键设计决策

| 问题 | 决策 |
|------|------|
| LLM 调用在哪？ | 不在框架内。外部 Agent 读 prompt 后自主决策 |
| 框架是什么？ | Skill/Harness — 提供工具和循环指引 |
| 因子发现策略？ | 自适应 — 因子少时外部搜索，充足时本地挖掘 |
| 验证缓存？ | DuckDB validation_cache 表 — 不重复验证已测试因子 |
| 因子加入粒度？ | 先单后批 — 先单独验证 IC，再批量集成 |
| 因子移除？ | 支持但少见 — Critic 建议时 Strategist 执行 |
| 参数重新优化？ | 条件触发 — 因子数 >= 3 或权重变化时 |
| Stage 数量？ | 5 个 — 合并发现+验证 |
| 数据存储？ | DuckDB 全局共享 — 行情/因子/缓存/回测统一管理 |
| 因子分类？ | 三类 — market_ts / asset_ts / cross_section |
| 因子面板？ | VIEW — 实时合并，自动同步 |

---

## 十、Karpathy autoresearch 对比

| 维度 | Karpathy (LLM训练) | StrategyResearch |
|------|---------------------|-----------------|
| 可改文件 | train.py (代码) | strategy.py (参数+因子) |
| 评估时间 | 5 分钟 (固定) | 几秒~几分钟 (回测) |
| 指标 | val_bpb (越低越好) | calmar (越高越好) |
| 发现路径 | 无 (只改代码) | 3 条路径 (本地/外部/LLM) |
| 验证方式 | 无 (直接跑) | IC/IR + 6维 + 去重 |
| 缓存 | 无 | DuckDB 验证缓存 |
| 多目标 | 无 | Pareto 前沿 |
| 并行 | 无 | ThreadPoolExecutor |

核心理念相同:
- 框架提供工具，Agent 提供智能
- 修改 → 执行 → 检查 → keep/discard → 重复
- 永不停止，直到用户中断
