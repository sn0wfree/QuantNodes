# Agent 策略构建操作手册

> QuantNodes AI 原生量化研究平台 - 小白入门指南  
> 版本：v1.0  
> 更新日期：2026-05-08

---

## 目录

1. [快速开始](#1-快速开始)
2. [界面认识](#2-界面认识)
3. [Agent 对话基础](#3-agent-对话基础)
4. [策略生成案例](#4-策略生成案例)
5. [回测执行与结果](#5-回测执行与结果)
6. [因子分析](#6-因子分析)
7. [知识库保存](#7-知识库保存)
8. [常见问题](#8-常见问题)
9. [最佳实践](#9-最佳实践)
10. [附录：Prompt 模板](#10-附录prompt-模板)

---

## 1. 快速开始

### 1.1 环境启动

本项目分为前端（Vue 3）和后端（FastAPI）两部分，需要分别启动：

```bash
# 方式1: 使用 CLI（推荐，首次需先运行 init）
quantnodes init              # 首次初始化
quantnodes run               # 启动全部服务

# 方式2: 手动启动
# 终端 1: 启动后端服务
cd /path/to/QuantNodes
python -m uvicorn api.main:app --reload --port 8000

# 终端 2: 启动前端服务
cd /path/to/QuantNodes/frontend
npm run dev
```

### 1.2 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| **前端界面** | http://localhost:5173 | 主要操作界面 |
| **API 文档** | http://localhost:8000/docs | 后端 API 列表 |
| **Agent 服务** | 通过前端界面访问 | 对话式策略生成 |

### 1.3 首次配置

在使用 Agent 之前，需要配置 LLM API：

1. 打开前端界面 http://localhost:5173
2. 进入 **Settings** 页面（设置）
3. 配置 OpenAI API Key

```python
# 也可以通过环境变量配置
import os
os.environ["OPENAI_API_KEY"] = "sk-xxx..."
```

---

## 2. 界面认识

### 2.1 首页仪表盘 (Dashboard)

> **[截图位置：Dashboard 页面]**
> 
> 首次登录后看到的首页，展示：
> - 快捷入口卡片
> - 最近策略
> - 回测统计

### 2.2 导航菜单

> **[截图位置：侧边导航栏]**
> 
> 左侧导航包含以下功能：

| 菜单 | 功能说明 |
|------|----------|
| **Dashboard** | 首页仪表盘 |
| **Chat** | Agent 对话界面 |
| **Backtest** | 回测中心 |
| **Factor Analysis** | 因子分析 |
| **Wiki** | 知识库（因子/策略） |
| **Dream** | 研究洞察 |
| **Settings** | 系统设置 |

### 2.3 核心功能入口

```
┌─────────────────────────────────────────┐
│  QuantNodes                              │
├─────────────────────────────────────────┤
│  📊 Dashboard          (首页)            │
│  💬 Chat               (Agent 对话)      │
│  📈 Backtest           (回测中心)        │
│  🔬 Factor Analysis    (因子分析)        │
│  📚 Wiki                (知识库)         │
│    ├─ Factors          (因子库)          │
│    └─ Strategies       (策略库)          │
│  💭 Dream               (研究洞察)        │
│  ⚙️ Settings            (设置)            │
└─────────────────────────────────────────┘
```

---

## 3. Agent 对话基础

### 3.1 打开 Agent 对话

1. 点击左侧导航 **Chat** 或 **Agent Chat**
2. 进入对话界面

> **[截图位置：Agent Chat 页面]**
> 
> 页面包含：
> - 对话消息区域（上方）
> - 输入框（下方）

### 3.2 发送消息

在底部输入框中输入你的需求，点击发送按钮或按 Enter 键。

```plaintext
[输入框] 请在此输入您的策略描述...
[发送按钮]
```

### 3.3 理解 Agent 回复

Agent 会返回：
1. **文字说明** - 解释策略思路
2. **代码块** - 可执行的 Python 代码
3. **回测结果**（如果请求执行回测）

---

## 4. 策略生成案例

### 4.1 案例一：MA 均线交叉策略

#### 4.1.1 对话输入

```
帮我生成一个 5/20 日均线交叉策略
```

#### 4.1.2 Agent 预期回复

> **[截图位置：Agent 返回 MA 策略代码]**

```
我为你生成一个经典的 MA(5,20) 均线交叉策略：

【策略思路】
- 当 5 日均线上穿 20 日均线时买入（多头信号）
- 当 5 日均线下穿 20 日均线时卖出（空头信号）

【策略代码】
```python
import pandas as pd
import numpy as np
from QuantNodes.factor_node.factor_functions import rolling_mean

# 策略参数
short_window = 5
long_window = 20

# 计算均线
df['ma5'] = rolling_mean(df['close'], window=short_window)
df['ma20'] = rolling_mean(df['close'], window=long_window)

# 生成信号
df['signal'] = (df['ma5'] > df['ma20']).astype(int)
df['signal'] = df['signal'].shift(1)  # 信号滞后一期

# 计算收益
df['return'] = df['close'].pct_change()
df['strategy_return'] = df['return'] * df['signal']

# 绩效指标
total_return = (1 + df['strategy_return']).prod() - 1
annual_return = (1 + total_return) ** (252/len(df)) - 1
sharpe = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
print(f"总收益率: {total_return:.2%}")
print(f"年化收益率: {annual_return:.2%}")
print(f"夏普比率: {sharpe:.2f}")
```

请告诉我是否需要我执行这个回测？
```

#### 4.1.3 代码解读

| 代码部分 | 说明 |
|----------|------|
| `rolling_mean(df['close'], window=5)` | 计算 5 日移动平均 |
| `(df['ma5'] > df['ma20']).astype(int)` | 金叉买入信号 = 1 |
| `df['signal'].shift(1)` | 信号滞后一期，避免未来函数 |
| `df['return'] * df['signal']` | 策略收益 = 市场收益 × 仓位 |

---

### 4.2 案例二：动量因子策略

#### 4.2.1 对话输入

```
生成一个基于过去 20 日收益率的动量策略，买入收益率最高的前 10 只股票
```

#### 4.2.2 Agent 预期回复

> **[截图位置：Agent 返回动量策略代码]**

```
我为你生成一个动量因子选股策略：

【策略思路】
- 计算过去 20 日收益率作为动量因子
- 选取动量因子值最高的前 10 只股票
- 等权配置，每月调仓

【策略代码】
```python
import pandas as pd
import numpy as np
from QuantNodes.factor_node.factor_functions import rolling_mean, ts_rank

# 策略参数
lookback_period = 20
top_n = 10

# 假设 df 包含多只股票数据，列: date, code, close

# 方法1：使用 rolling 手动计算
df['momentum'] = df.groupby('code')['close'].transform(
    lambda x: x.pct_change(lookback_period)
)

# 方法2：使用 ts_rank 算子（推荐）
df['momentum_rank'] = ts_rank(df['close'], window=lookback_period)

# 每月末选取 Top N
def select_stocks(group):
    return group.nlargest(top_n, 'momentum')

# 获取每月调仓信号
monthly_rebalance = df.groupby('date').apply(select_stocks)
monthly_rebalance['signal'] = 1  # 持有信号

# 合并信号到原始数据
df = df.merge(
    monthly_rebalance[['signal']], 
    on=['date', 'code'], 
    how='left'
).fillna(0)

# 计算收益
df['return'] = df.groupby('code')['close'].pct_change()
df['strategy_return'] = df['return'] * df['signal']

print(f"动量策略回测完成")
```

需要我执行这个回测吗？请提供数据文件路径。
```

---

### 4.3 案例三：多因子组合策略

#### 4.3.1 对话输入

```
创建一个动量 + 波动率的等权组合因子，其中波动率使用 20 日 rolling std 计算
```

#### 4.3.2 Agent 预期回复

> **[截图位置：Agent 返回多因子策略代码]**

```
我为你生成一个双因子组合策略：

【策略思路】
- 因子1：20 日动量 (momentum)
- 因子2：20 日波动率 (volatility) - 取倒数，高波动惩罚
- 等权组合两个因子

【策略代码】
```python
import pandas as pd
import numpy as np
from QuantNodes.factor_node.factor_functions import (
    rolling_mean, rolling_std, zscore, rank
)

# 计算动量因子
df['momentum'] = df.groupby('code')['close'].transform(
    lambda x: x.pct_change(20)
)

# 计算波动率因子
df['volatility'] = df.groupby('code')['close'].transform(
    lambda x: rolling_std(x, window=20)
)

# 标准化处理
df['momentum_zscore'] = zscore(df['momentum'])
df['volatility_zscore'] = zscore(df['volatility'])

# 波动率取倒数并标准化（低波动更好）
df['inv_vol_zscore'] = zscore(-df['volatility_zscore'])

# 等权组合
df['composite_factor'] = (df['momentum_zscore'] + df['inv_vol_zscore']) / 2

# 选取因子值最高的股票
top_stocks = df.groupby('date').apply(
    lambda x: x.nlargest(10, 'composite_factor')
)

print(f"组合因子策略已完成")
print(f"因子相关性: {df['momentum'].corr(df['volatility']):.3f}")
```

需要我执行回测吗？
```

---

### 4.4 案例四：完整回测流程

#### 4.4.1 对话输入

```
用刚才的 MA 交叉策略，回测 2024 年的数据，初始资金 100 万
```

#### 4.4.2 Agent 预期回复

> **[截图位置：回测执行结果]**

```
正在执行回测...

【回测配置】
- 策略：MA(5,20) 交叉策略
- 时间：2024-01-01 至 2024-12-31
- 初始资金：1,000,000 元
- 手续费：0.03%

【回测结果】
┌─────────────────────────────────────┐
│  📊 绩效指标                         │
├─────────────────────────────────────┤
│  总收益率：     +23.56%              │
│  年化收益率：   +23.56%               │
│  夏普比率：     1.82                  │
│  最大回撤：     -12.34%               │
│  胜率：         58.23%                │
│  交易次数：     48 次                 │
│  最终资金：     1,235,600 元          │
└─────────────────────────────────────┘

【收益曲线】
[图表位置：权益曲线图]

是否需要我保存这个策略到知识库？
```

---

## 5. 回测执行与结果

### 5.1 回测结果页面

> **[截图位置：回测结果页面 - ResultView]**

回测结果页面包含：

#### 5.1.1 关键指标卡片

| 指标 | 含义 | 示例值 | 判断标准 |
|------|------|--------|----------|
| **Total Return** | 总收益率 | +23.56% | 越高越好 |
| **Annual Return** | 年化收益率 | +23.56% | > 10% 为佳 |
| **Sharpe Ratio** | 夏普比率 | 1.82 | > 1.0 为佳 |
| **Max Drawdown** | 最大回撤 | -12.34% | < 20% 为佳 |
| **Win Rate** | 胜率 | 58.23% | > 50% 为佳 |
| **Trade Count** | 交易次数 | 48 | 适中为宜 |

#### 5.1.2 收益曲线图

> **[截图位置：权益曲线图]**
> 
> 展示策略资金随时间的变化

#### 5.1.3 交易记录

> **[截图位置：交易记录表格]**
> 
> 列出每笔交易的日期、买卖方向、收益率等

### 5.2 指标解读

| 指标 | 说明 | 好的标准 |
|------|------|----------|
| **总收益率** | 策略累计收益 | > 20% |
| **年化收益率** | 每年平均收益 | > 10% |
| **夏普比率** | 风险调整后收益 | > 1.0 |
| **最大回撤** | 历史最大亏损 | < 20% |
| **胜率** | 盈利交易占比 | > 50% |

---

## 6. 因子分析

### 6.1 IC 分析

#### 6.1.1 对话输入

```
分析刚才因子的 IC 表现
```

#### 6.1.2 Agent 预期回复

> **[截图位置：IC 分析结果页面]**

```
【因子 IC 分析】

IC (Information Coefficient) = 因子值与下期收益的相关系数

┌─────────────────────────────────────────┐
│  IC 统计                               │
├─────────────────────────────────────────┤
│  IC 均值：       0.0523                 │
│  IC 标准差：     0.1287                 │
│  ICIR：          0.41                   │
│  IC > 0 占比：   58.3%                  │
│  Rank IC 均值：  0.0612                 │
│  Rank ICIR：     0.48                   │
└─────────────────────────────────────────┘

【解读】
- IC 均值 0.05 > 0，说明因子有一定预测能力
- ICIR 0.41 偏低，因子稳定性有待提升
- IC > 0 占比 58.3%，超过一半时间有效

需要进一步分析吗？
```

### 6.2 IC 指标说明

| 指标 | 含义 | 判断标准 |
|------|------|----------|
| **IC 均值** | 因子与收益的平均相关性 | > 0.03 为有效 |
| **IC 标准差** | IC 的波动程度 | 越小越稳定 |
| **ICIR** | IC 均值/IC 标准差 | > 0.5 为佳 |
| **IC > 0 占比** | IC 为正的时间比例 | > 50% |

### 6.3 相关性分析

#### 6.3.1 对话输入

```
分析动量因子和波动率因子的相关性
```

#### 6.3.2 Agent 预期回复

```
【因子相关性分析】

相关系数矩阵：
           momentum  volatility
momentum     1.000     -0.234
volatility  -0.234      1.000

【结论】
- 动量与波动率相关系数 -0.23，呈负相关
- 低相关性说明两个因子具有互补性
- 可以考虑等权组合
```

---

## 7. 知识库保存

### 7.1 保存策略到 Wiki

#### 7.1.1 对话输入

```
保存这个策略到知识库
```

#### 7.1.2 Agent 预期回复

```
已保存策略到知识库！

【策略信息】
- 名称：MA_cross_5_20
- 类型：趋势跟踪
- 创建时间：2024-01-15

> **[截图位置：Wiki 策略列表页面]**
> 
> 可以在 Wiki > Strategies 中查看
```

### 7.2 查看保存的策略

> **[截图位置：Wiki 策略详情页面]**

在 Wiki > Strategies 页面可以看到：
- 策略名称
- 策略描述
- 策略代码
- 回测结果
- 创建时间

### 7.3 保存因子

> **[截图位置：Wiki 因子详情页面]**

在 Wiki > Factors 页面可以查看：
- 因子名称
- 因子描述
- 因子计算代码
- IC 分析结果

---

## 8. 常见问题

### 8.1 Agent 不响应

| 可能原因 | 解决方案 |
|----------|----------|
| API Key 未配置 | 检查 Settings 页面配置 |
| API Key 错误 | 确认 Key 是否有效 |
| 网络问题 | 检查网络连接 |

### 8.2 代码执行失败

```
Prompt 示例：
刚才的代码报错了，错误信息是：NameError: name 'df' is not defined
```

Agent 会帮你修复代码。

### 8.3 回测结果为 0

| 可能原因 | 解决方案 |
|----------|----------|
| 数据文件路径错误 | 检查数据文件是否存在 |
| 日期范围无数据 | 检查 start_date/end_date |
| 信号始终为 0 | 检查因子计算逻辑 |

### 8.4 回测结果异常

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 收益极高 ( > 100%) | 未来函数 | 检查是否使用了未来数据 |
| 收益为负 | 策略逻辑错误 | 检查买卖逻辑是否正确 |
| 交易次数过多 | 调仓频率过高 | 调整信号生成逻辑 |

---

## 9. 最佳实践

### 9.1 从简单开始

1. **先测试环境** - 用最简单的 MA 策略验证环境正常
2. **再增加复杂度** - 熟悉后再尝试多因子策略

### 9.2 逐步迭代

```
第1步：让 Agent 生成基础版本
    ↓
第2步：检查回测结果是否合理
    ↓
第3步：如有问题，描述问题让 Agent 调整
    ↓
第4步：满意后保存到知识库
```

### 9.3 验证为王

- 始终检查回测结果的合理性
- 警惕"太好不真实"的结果
- 了解策略的假设和限制

### 9.4 记录研究

- 每次生成策略后保存到 Wiki
- 记录策略思路和参数
- 保存回测结果便于对比

---

## 10. 附录：Prompt 模板

### 10.1 策略生成类

| 场景 | Prompt 模板 |
|------|-------------|
| 均线策略 | `生成一个 {short}/{long} 日均线交叉策略` |
| 动量策略 | `生成一个基于过去 {period} 日收益率的动量策略，买入前 {N} 只` |
| 因子策略 | `使用 {operator} 算子创建 {factor_name} 因子` |
| 多因子 | `创建 {factor1} + {factor2} 的等权组合因子` |
| 配对策略 | `生成一个 {stock1} 和 {stock2} 的配对交易策略` |

### 10.2 回测类

| 场景 | Prompt 模板 |
|------|-------------|
| 执行回测 | `用刚才的策略，回测 {start_date} 到 {end_date} 的数据` |
| 修改参数 | `把初始资金改为 {amount}，手续费改为 {commission}` |
| 对比回测 | `用相同参数回测 2023 年的数据，对比一下` |

### 10.3 分析类

| 场景 | Prompt 模板 |
|------|-------------|
| IC 分析 | `分析这个因子的 IC 表现` |
| 相关性分析 | `分析 {factor1} 和 {factor2} 的相关性` |
| 绩效归因 | `分析策略收益来源` |

### 10.4 保存类

| 场景 | Prompt 模板 |
|------|-------------|
| 保存策略 | `保存这个策略到知识库，名称叫 {name}` |
| 保存因子 | `保存这个因子到知识库，名称叫 {name}` |
| 查看列表 | `列出我保存的所有策略` |

---

## 附录：快速参考

### 环境启动

```bash
# 后端
python -m uvicorn api.main:app --reload --port 8000

# 前端
cd frontend && npm run dev
```

### 访问地址

- 前端：http://localhost:5173
- API 文档：http://localhost:8000/docs

### 核心页面

| 功能 | 路径 |
|------|------|
| Agent 对话 | /chat |
| 回测中心 | /backtest |
| 因子分析 | /factor-analysis |
| 策略库 | /wiki/strategies |
| 因子库 | /wiki/factors |

---

**文档版本**：v1.0  
**更新时间**：2026-05-08  
**适用版本**：QuantNodes v2.4.0+