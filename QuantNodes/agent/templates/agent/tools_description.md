# QuantNodes Agent 工具说明

## 工具总览

| 工具名 | 功能 | 安全限制 | 并发安全 |
|--------|------|----------|----------|
| sandbox | 代码安全验证 | 无 | 是 |
| pipeline | Pipeline构建验证 | 无 | 是 |
| strategy | 策略生成 | 无 | 是 |
| backtest | 回测执行 | 否 | 否 |
| factor | 因子分析 | 是 | 是 |
| echo | 测试工具 | 是 | 是 |

---

## 详细说明

### sandbox - 沙箱工具

验证Python代码的安全性。

**参数**:
- `code` (string, 必需): 待验证的Python代码
- `allow_warnings` (boolean, 可选): 是否允许警告，默认True
- `max_code_length` (integer, 可选): 最大代码长度，默认10000

**返回**:
- `is_safe` (boolean): 代码是否安全
- `errors` (list): 错误列表
- `warnings` (list): 警告列表

**安全检查**:
- 禁止 `os.system`, `subprocess` 等危险调用
- 禁止 `eval()`, `exec()` 等动态执行
- 代码长度限制

---

### pipeline - Pipeline构建验证

提取代码中的Node并验证语法。

**参数**:
- `code` (string, 必需): 包含Node定义的Python代码

**返回**:
- `is_valid` (boolean): 代码是否有效
- `nodes` (list): 提取到的Node类型列表
- `errors` (list): 错误列表

**支持的因子计算方式**:
- factor_functions 模块 (ff.rolling_mean, ff.zscore, ff.rank 等)
- Polars 表达式

---

### strategy - 策略生成

根据描述生成量化策略代码。

**参数**:
- `description` (string, 必需): 策略描述
- `validate` (boolean, 可选): 是否验证生成的代码，默认True

**返回**:
- `message` (string): 生成结果消息
- `is_valid` (boolean): 代码是否有效

---

### backtest - 回测执行

执行Strategy→Risk→Broker回测流程。

**参数**:
- `pipeline_code` (string, 必需): 策略Pipeline代码
- `start_date` (string, 可选): 回测开始日期 YYYY-MM-DD
- `end_date` (string, 可选): 回测结束日期 YYYY-MM-DD
- `initial_cash` (number, 可选): 初始资金，默认100000
- `commission` (number, 可选): 手续费率，默认0.001

**返回**:
- `status` (string): 执行状态 success/error
- `summary` (dict): 回测结果摘要
- `config` (dict): 回测配置
- `nodes` (dict): 使用的Node信息

**代码要求**:
代码中需创建以下变量：
- `strategy`: StrategyNode实例
- `quote_data`: 行情数据DataFrame（可选）
- `broker`: SimulatedBrokerNode实例（可选，有默认值）

---

### factor - 因子分析

对因子进行IC分析、相关性分析等。

**参数**:
- `factor_code` (string, 必需): 因子的Python代码
- `analysis_type` (string, 必需): 分析类型 ic/correlation/both
- `start_date` (string, 可选): 分析开始日期
- `end_date` (string, 可选): 分析结束日期

**返回**:
- `status` (string): 执行状态
- `analysis.ic` (dict): IC分析结果
  - `ic_mean`: IC均值
  - `ic_std`: IC标准差
  - `icir`: IC信息比率
  - `rank_ic_mean`: 秩IC均值
- `analysis.correlation` (dict): 相关性分析结果

**代码要求**:
代码中需创建 `result` 变量，为Polars DataFrame，包含：
- `date`: 日期列（可选，用于时序分析）
- `code`: 股票代码列（可选）
- `factor_value`: 因子值列（必需）
- `forward_return`: 前瞻收益率列（必需）

**示例**:
```python
import polars as pl

result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
```

---

### echo - 测试工具

测试工具，返回输入的消息内容。

**参数**:
- `message` (string, 必需): 待回显的消息

**返回**:
- 输入的消息内容
