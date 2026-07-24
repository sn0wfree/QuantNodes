# StrategyResearch — 通用策略自动研究框架设计文档

> **版本**: v2.0
> **日期**: 2026-07-20
> **状态**: 设计完成，待实施

---

## 一、项目概述

### 1.1 目标

构建 QuantNodes 的 `StrategyResearch` — 一个通用的策略自动研究框架。

核心理念: **Karpathy autoresearch 极简 + 多 Agent 增强 + 因子研发流水线**。

支持的策略类型: ETF 轮动、股票选股、因子择时、行业轮动。

### 1.2 设计原则

- **Karpathy 极简**: 框架提供工具和循环指引，不调 LLM。外部 Agent 读 prompt 后自主决策。
- **Skill/Harness 模式**: 不是 Pipeline 编排器，而是 Agent 的 "研究实验室"。
- **通用性**: 框架与具体策略解耦，通过目标函数接口适配不同策略。
- **自适应**: 因子少时优先外部搜索+LLM，充足时本地算子挖掘。
- **DuckDB 统一存储**: 行情数据、因子数据、验证缓存、回测结果统一管理。
- **实验可复现**: 每次实验保存快照，可随时复现。

### 1.3 与现有系统的关系

```
QuantNodes 现有系统
├── research/quant_alpha/     — 因子挖掘 (AlphaGPT/MCTS/AlphaLogics)
├── agent/tools/              — Agent 工具 (alpha_evaluate/factor/backtest)
├── strategy/momentum_etf_rotation/ — ETF 轮动策略 (v1-v7.10)
└── research/strategy_research/     — 新增: 通用策略研究框架
```

`StrategyResearch` 是一个 **上层编排**，复用现有组件，不修改它们。

---

## 二、架构设计

### 2.1 工作区结构

通过 CLI 命令初始化：

```bash
quantnodes research init /path/to/workspace
```

交互式创建：

```
$ quantnodes research init /path/to/workspace

策略名称: etf_rotation
策略类型 (rotation/selection/timing/industry): rotation
目标函数 [calmar]:

✓ 创建目录结构
✓ 生成提示词
✓ 生成策略模板
✓ 初始化 DuckDB
✓ 初始化 git
```

生成的工作区：

```
/path/to/workspace/
├── README.md                    # Agent 入口 (含 subagent 说明)
├── config.yaml                  # 工作区配置
├── data.duckdb                  # 共享数据库
├── .git/
├── .prompts/                    # Subagent 提示词 (隐藏目录)
│   ├── researcher.md            # 评估 + 决策
│   ├── factor_analyst.md        # 发现 + 验证
│   ├── strategist.md            # 集成 + 优化
│   └── critic.md                # 评估 + 风控
└── strategies/
    └── etf_rotation/
        ├── program.md           # 策略知识 + subagent 调用指引
        ├── prepare.py           # 数据加载 + 目标函数
        ├── strategy.py          # Agent 修改的文件
        └── runs/                # 实验记录
            ├── run_0001/
            │   ├── strategy.py  # 快照 (可复现)
            │   ├── run.log      # 回测输出
            │   └── metrics.json # 关键指标
            └── results.tsv      # 汇总表
```

### 2.2 文件职责

| 文件 | 位置 | 职责 | 谁读 |
|------|------|------|------|
| README.md | 根目录 | 工作区说明 + subagent 列表 | 主 Agent |
| .prompts/*.md | 隐藏目录 | 角色定义 + 行为模式 | Subagent |
| program.md | 策略目录 | 策略知识 + 循环指引 + subagent 调用 | 主 Agent |
| prepare.py | 策略目录 | 数据加载 + 目标函数 | 框架调用 |
| strategy.py | 策略目录 | 策略配置 (PARAMS + FACTOR_EXPRS) | Agent 修改 |
| runs/ | 策略目录 | 实验记录 (快照 + 日志 + 指标) | Agent 读取 |

### 2.3 Subagent 分工

| Subagent | 文件 | 职责 | 何时 spawn |
|----------|------|------|-----------|
| Researcher | .prompts/researcher.md | 评估因子池，决策行动 | 主 Agent 直接执行 |
| Factor Analyst | .prompts/factor_analyst.md | 发现 + 验证因子 | 需要因子发现时 |
| Strategist | .prompts/strategist.md | 集成因子到策略 | 主 Agent 直接执行 |
| Critic | .prompts/critic.md | 评估结果 + 风控 | 每次回测后 |

**主 Agent** 读 README.md + program.md，知道做什么。
**Subagent** 读 .prompts/*.md，知道怎么做。

---

## 三、目标函数接口

### 3.1 prepare.py 接口

每个策略的 `prepare.py` 必须实现：

```python
# 目标函数名称
GOAL_METRIC = "calmar"

# 目标方向
GOAL_DIRECTION = "maximize"  # "maximize" 或 "minimize"

def load_data():
    """加载策略数据。返回 dict。"""
    raise NotImplementedError

def evaluate(params, factor_exprs, factor_weight_method, data):
    """评估策略表现。
    
    Returns:
        dict: {GOAL_METRIC: value, "sharpe": ..., "max_dd": ..., ...}
    """
    raise NotImplementedError
```

### 3.2 目标函数示例

| 策略类型 | GOAL_METRIC | GOAL_DIRECTION |
|---------|-------------|----------------|
| ETF 轮动 | calmar | maximize |
| 股票选股 | sharpe | maximize |
| 因子择时 | accuracy | maximize |
| 行业轮动 | sector_adjusted_return | maximize |

---

## 四、五阶段流水线

### 4.1 流水线概览

```
┌─────────────────────────────────────────────────────────────┐
│                  StrategyResearch 5-Stage Pipeline           │
│                                                             │
│  Stage 1: HYPOTHESIS                                        │
│  读取状态 → 决策行动                                         │
│                          ↓                                  │
│  Stage 2: FACTOR RESEARCH                                   │
│  发现 + 验证因子 (skip 优化/移除时)                           │
│                          ↓                                  │
│  Stage 3: STRATEGY INTEGRATION                              │
│  集成因子 / 优化参数 / 移除因子                               │
│                          ↓                                  │
│  Stage 4: BACKTEST                                          │
│  运行策略回测                                                │
│                          ↓                                  │
│  Stage 5: EVALUATE                                          │
│  评估结果 → keep / discard                                   │
│                          ↓                                  │
│  回到 Stage 1                                                │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 1: HYPOTHESIS

**输入**: strategy.py + results.tsv + run.log
**输出**: action + hypothesis

```
1. 读 strategy.py → 当前因子池和参数
2. 读 results.tsv → 历史实验
3. 读 run.log → 上轮指标
4. 决策:
   - 因子不足 → search_external
   - 因子充足 → discover_local
   - 参数不优 → optimize_param
   - 因子过多 → remove_factor
```

### 4.3 Stage 2: FACTOR RESEARCH

**输入**: action
**输出**: 通过验证的因子列表

```
(skip 优化/移除时)

Step 1: 生成候选因子 (外部搜索/本地挖掘/LLM 建议)
Step 2: IC/IR 验证 (IC > 0.03, IR > 0.5)
Step 3: 6 维评分
Step 4: Mutual IC 去重 (|corr| < 0.7)
Step 5: IC 衰减检查
```

**结果存储**: validation.log + DuckDB validation_cache

### 4.4 Stage 3: STRATEGY INTEGRATION

**输入**: 验证通过的因子
**输出**: 更新后的 strategy.py

```
因子集成: 先单后批 (单独验证 → 批量加入)
参数优化: 修改 PARAMS
因子移除: 移除低 IR 因子
```

### 4.5 Stage 4: BACKTEST

**输入**: strategy.py
**输出**: run.log + metrics.json

```
运行: python strategy.py > run.log 2>&1
提取: grep "^calmar:\|^sharpe:" run.log
保存: runs/run_XXXX/ (strategy.py + run.log + metrics.json)
```

### 4.6 Stage 5: EVALUATE

**输入**: metrics.json
**输出**: keep / discard

```
比较目标函数 → 改善则 keep，否则 discard
检查风控阈值 → 触发则 discard
git commit (keep) 或 git reset (discard)
追加到 results.tsv
```

---

## 五、实验记录结构

### 5.1 runs/run_XXXX/

每次实验自动创建：

```
runs/run_0001/
├── strategy.py      # 快照 (可复现)
├── run.log          # 回测输出
└── metrics.json     # 关键指标
```

### 5.2 metrics.json

```json
{
  "run": "run_0001",
  "commit": "a1b2c3d",
  "action": "optimize_param",
  "goal_metric": 0.710,
  "sharpe": 0.820,
  "max_dd": -0.11,
  "ann_return": 0.092,
  "turnover": 0.35,
  "factors_added": 0,
  "status": "keep",
  "description": "baseline",
  "timestamp": "2026-07-20T10:30:00"
}
```

### 5.3 results.tsv

```tsv
run	commit	action	calmar	sharpe	max_dd	ann_return	turnover	factors_added	status	description
run_0001	a1b2c3d	optimize_param	0.710	0.820	-0.11	0.092	0.35	0	keep	baseline
run_0002	b2c3d4e	search_external	0.750	0.850	-0.10	0.095	0.38	2	keep	+realized_skew
```

### 5.4 复现实验

```bash
# 方法 1: 手动
cd strategies/etf_rotation/
cp runs/run_0002/strategy.py strategy.py
python strategy.py

# 方法 2: CLI
quantnodes research reproduce /path/to/workspace etf_rotation run_0002
```

---

## 六、DuckDB 存储规范

### 6.1 表结构

```sql
-- 因子注册表
CREATE TABLE factor_registry (
    factor_name VARCHAR NOT NULL,
    factor_code VARCHAR NOT NULL,
    factor_type VARCHAR NOT NULL,    -- market_ts / asset_ts / cross_section
    category VARCHAR,
    source VARCHAR,
    strategy_name VARCHAR NOT NULL,
    added_at TIMESTAMP,
    PRIMARY KEY (strategy_name, factor_name)
);

-- 验证缓存
CREATE TABLE validation_cache (
    factor_name VARCHAR NOT NULL,
    factor_code VARCHAR NOT NULL,
    strategy_name VARCHAR NOT NULL,
    ic_mean DOUBLE,
    ir DOUBLE,
    overall_score DOUBLE,
    is_valid BOOLEAN,
    validated_at TIMESTAMP,
    PRIMARY KEY (strategy_name, factor_name)
);

-- 回测结果
CREATE TABLE backtest_results (
    strategy_name VARCHAR NOT NULL,
    run VARCHAR NOT NULL,
    commit_hash VARCHAR,
    action VARCHAR,
    goal_metric DOUBLE,
    calmar DOUBLE,
    sharpe DOUBLE,
    max_dd DOUBLE,
    status VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY (strategy_name, run)
);
```

### 6.2 因子分类

| 类型 | 说明 | 示例 |
|------|------|------|
| market_ts | 市场级时序 | vix, dxy, 宏观指标 |
| asset_ts | 资产级时序 | 个股动量, 个股波动率 |
| cross_section | 截面因子 | rank, zscore |

---

## 七、文件模板内容

### 7.1 README.md

```markdown
# Research Workspace

## 快速开始
1. 读取 `strategies/{strategy}/program.md` 了解策略
2. 读取 `strategies/{strategy}/strategy.py` 了解当前配置
3. 读取 `strategies/{strategy}/runs/results.tsv` 了解历史
4. 开始实验循环

## Subagent
| Subagent | 文件 | 用途 |
|----------|------|------|
| Researcher | `.prompts/researcher.md` | 评估因子池，决策行动 |
| Factor Analyst | `.prompts/factor_analyst.md` | 发现并验证因子 |
| Strategist | `.prompts/strategist.md` | 集成因子到策略 |
| Critic | `.prompts/critic.md` | 评估结果，风控检查 |

何时 spawn：
- 需要因子发现/验证 → spawn Factor Analyst
- 需要评估结果 → spawn Critic
- 其他 → 主 Agent 直接执行

## 实验循环
LOOP FOREVER:
1. 读取当前状态
2. 决策下一步行动
3. 执行
4. 保存到 runs/run_XXXX/
5. git commit 或 reset

## 复现实验
进入 `runs/run_XXXX/`，用 `strategy.py` 替换当前配置，运行即可。
```

### 7.2 .prompts/researcher.md

```markdown
# Role: Researcher

你是量化策略研究员。基于历史实验结果，提出研究假设。

## 输入
- strategy.py 中的因子池和参数
- results.tsv 中的历史实验
- 上一轮 Critic 反馈

## 自适应策略
| 条件 | 行动 |
|------|------|
| 因子不足或覆盖低 | search_external |
| 因子充足 | discover_local |
| 参数不优 | optimize_param |
| 因子过多 | remove_factor |

## 输出
```json
{
  "action": "search_external | discover_local | optimize_param | remove_factor",
  "hypothesis": "一句话描述假设",
  "reason": "决策依据"
}
```
```

### 7.3 .prompts/factor_analyst.md

```markdown
# Role: Factor Analyst

你是因子研究专家。发现并验证因子。

## 三条路径
| 路径 | 触发条件 | 方法 |
|------|---------|------|
| A: 本地算子 | 因子充足 | MCTS 搜索 |
| B: 外部搜索 | 因子不足 | web_search + 提取 |
| C: LLM 建议 | 需要方向 | 分析当前状态 |

## 验证流程
1. IC/IR 验证 (IC > 0.03, IR > 0.5)
2. 6 维评分
3. Mutual IC 去重 (|corr| < 0.7)
4. IC 衰减检查

## 输出
```json
{
  "candidates": [{"factor_name": "...", "ic_mean": 0.05, "passed": true}],
  "rejected": [{"factor_name": "...", "reason": "..."}]
}
```
```

### 7.4 .prompts/strategist.md

```markdown
# Role: Strategist

你是策略集成专家。将因子集成到策略中。

## 因子集成流程
1. 单独验证: 每个因子单独加入，回测
2. 批量集成: 通过验证的因子一起加入
3. 面板重建: 写入 DuckDB

## 输出
更新 strategy.py (PARAMS / FACTOR_EXPRS / FACTOR_WEIGHT_METHOD)
```

### 7.5 .prompts/critic.md

```markdown
# Role: Critic

你是策略评估专家。评估回测结果，控制风险。

## 风控阈值
| 指标 | 阈值 |
|------|------|
| MaxDD | <= -15% |
| Calmar | >= 0.5 |
| Sharpe | >= 0.3 |

## 抗过拟合检验
1. 起点依赖: CV% < 25%
2. 调仓日偏移: ±5 日稳定
3. 参数扰动: ±10% 退化 < 20%
4. 消融实验: 每关一项退化 >= 5%

## 输出
```json
{
  "verdict": "keep | discard",
  "analysis": "分析原因",
  "direction": "exploit | explore | diversify",
  "suggestions": ["建议1", "建议2"]
}
```
```

### 7.6 strategies/{name}/program.md

```markdown
# {Strategy Name} Research

## 策略概述
- 类型: {rotation/selection/timing/industry}
- 目标函数: {calmar/sharpe/...}
- 基线指标: {基线值}

## 策略知识
(填写策略专属知识)

## 实验循环
LOOP FOREVER:
1. 读 strategy.py → 当前状态
2. 读 runs/results.tsv → 历史
3. 读 runs/run_XXXX/run.log → 上轮结果
4. 决策 → 执行 → 保存

## Subagent 调用
| 何时 | spawn 谁 |
|------|---------|
| 需要因子发现/验证 | Factor Analyst |
| 需要评估结果 | Critic |
| 其他 | 主 Agent 直接执行 |

## 停止条件
- 连续 N 轮无改善
- 最大 M 轮
- 用户中断

## NEVER STOP
一旦开始，不要停下来问用户。持续运行直到被手动中断。
```

### 7.7 strategies/{name}/prepare.py

```python
"""
{Strategy Name} 数据加载和评估。
"""

GOAL_METRIC = "calmar"
GOAL_DIRECTION = "maximize"

def load_data():
    """加载策略数据。"""
    raise NotImplementedError

def evaluate(params, factor_exprs, factor_weight_method, data):
    """评估策略表现。
    Returns: {GOAL_METRIC: value, "sharpe": ..., ...}
    """
    raise NotImplementedError
```

### 7.8 strategies/{name}/strategy.py

```python
"""
{Strategy Name} 策略配置。
Agent 可以修改: PARAMS, FACTOR_EXPRS, FACTOR_WEIGHT_METHOD
"""

PARAMS = {}
FACTOR_EXPRS = []
FACTOR_WEIGHT_METHOD = "inv_vol"

if __name__ == "__main__":
    import importlib
    prepare = importlib.import_module(".prepare", package=__package__)
    data = prepare.load_data()
    metrics = prepare.evaluate(PARAMS, FACTOR_EXPRS, FACTOR_WEIGHT_METHOD, data)
    print("---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
```

---

## 八、实施步骤

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | CLI 命令框架 | `quantnodes/cli/research.py` |
| 2 | 初始化逻辑 (交互式) | `quantnodes/cli/research.py` |
| 3 | 模板文件 | `quantnodes/templates/research/` |
| 4 | DuckDB 初始化 | `core/db.py` |
| 5 | 测试 | `tests/test_research_init.py` |

---

## 九、关键设计决策

| 问题 | 决策 |
|------|------|
| 框架是什么？ | Skill/Harness — 提供工具和循环指引 |
| LLM 调用在哪？ | 不在框架内。外部 Agent 读 prompt 后自主决策 |
| 框架通用性？ | 通过目标函数接口适配不同策略 |
| Subagent 数量？ | 4 个 (Researcher/Factor Analyst/Strategist/Critic) |
| 主 Agent 角色？ | Coordinator — 读 program.md 后自主循环 |
| 状态如何传递？ | 文件 (strategy.py + results.tsv + run.log) |
| 实验如何记录？ | runs/run_XXXX/ (快照 + 日志 + 指标) |
| 如何复现实验？ | 用快照 strategy.py 替换当前配置，运行即可 |
| 初始化方式？ | CLI 命令 `quantnodes research init <path>` |
| prompts 位置？ | .prompts/ (隐藏目录) |
