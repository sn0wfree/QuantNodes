# QuantNodes Quick Start

> 5 分钟快速入门指南  
> 版本：v1.0  
> 更新：2026-05-08

---

## 一、安装 (2 分钟)

### 1.1 克隆项目

```bash
git clone https://github.com/sn0wfree/QuantNodes.git
cd QuantNodes
```

### 1.2 安装依赖

```bash
# Python 依赖
pip install -e .

# 前端依赖
cd frontend && npm install && cd ..
```

---

## 二、启动服务 (1 分钟)

```bash
# 终端 1: 启动后端
python -m uvicorn api.main:app --reload --port 8000

# 终端 2: 启动前端
cd frontend && npm run dev
```

访问 **http://localhost:5173**

---

## 三、快速开始

### 方式一：通过 Agent 对话生成策略 (推荐)

1. 打开浏览器访问 http://localhost:5173
2. 进入 **Chat** 页面
3. 输入：
   ```
   帮我生成一个 5/20 日均线交叉策略
   ```

### 方式二：Python 代码示例

```python
# 1. 导入模块
from QuantNodes.factor_node.factor_functions import rolling_mean, zscore
import pandas as pd
import numpy as np

# 2. 模拟数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)

# 3. 计算因子
ma5 = rolling_mean(prices, window=5)
ma20 = rolling_mean(prices, window=20)

# 4. 生成信号
signal = (ma5 > ma20).astype(int)

# 5. 计算收益
returns = prices.pct_change()
strategy_returns = returns * signal.shift(1)

# 6. 查看结果
sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
print(f"夏普比率: {sharpe:.2f}")
```

---

## 四、核心概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **BaseNode** | 最小计算单元 | `class MyNode(BaseNode)` |
| **Pipeline** | 串联节点 | `node_a >> node_b` |
| **Factor Functions** | 317+ 内置算子 | `rolling_mean()`, `zscore()` |

---

## 五、常用算子

```python
# 时间序列
rolling_mean(df, window=20)    # 滚动均值
rolling_std(df, window=20)     # 滚动标准差
ewm_mean(df, span=12)          # 指数移动平均
ts_rank(df, window=20)         # 时间序列排名

# 截面运算
zscore(df)                     # Z-Score 标准化
winsorize(df, limits=(0.05, 0.05))  # 缩尾处理
rank(df)                       # 排序

# 数学运算
abs(df), log(df), sqrt(df)     # 数学函数
where(cond, then, else_)       # 条件运算
```

---

## 六、下一步

| 目标 | 资源 |
|------|------|
| **Agent 使用** | [Agent-策略构建操作手册.md](./Agent-策略构建操作手册.md) |
| **完整功能** | [QuantNodes-操作手册.md](./QuantNodes-操作手册.md) |
| **代码示例** | `examples/01_quick_start.py` |

---

**开始你的量化研究之旅！** 🚀