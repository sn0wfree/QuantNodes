# QuantNodes Agent 工具说明

## 工具总览

| 工具名 | 功能 | 安全限制 | 并发安全 |
|--------|------|----------|----------|
| file_ops | 文件读写编辑 | 路径限制 | 否 |
| code_search | 代码搜索 | 只读 | 是 |
| git_ops | Git 操作 | 路径限制 | 否 |
| sandbox | 代码安全验证 | 无 | 是 |
| pipeline | Pipeline构建验证 | 无 | 是 |
| strategy | 策略生成 | 无 | 是 |
| backtest | 回测执行 | 否 | 否 |
| factor | 因子分析 | 是 | 是 |
| wiki | Wiki知识库 | 路径限制 | 否 |
| echo | 测试工具 | 是 | 是 |

---

## 详细说明

### file_ops - 文件操作工具

安全地读取、写入、编辑文件，列出目录内容，glob模式匹配。

**参数**:
- `action` (string, 必需): 操作类型 read_file/write_file/edit_file/list_files/glob_files
- `path` (string): 文件或目录路径（相对于工作目录）
- `content` (string): 要写入的内容（write_file）
- `old_string` (string): 要替换的旧字符串（edit_file）
- `new_string` (string): 替换后的新字符串（edit_file）
- `pattern` (string): glob模式（glob_files）
- `offset` (integer): 读取起始行号，默认1
- `limit` (integer): 读取最大行数，默认2000

**安全限制**:
- 路径必须在工作目录内（防 path traversal）
- 文件大小限制 1MB

---

### code_search - 代码搜索工具

在工作目录中搜索代码内容和文件名。

**参数**:
- `action` (string, 必需): 搜索类型 grep/find_files/search_code
- `pattern` (string, 必需): 搜索模式（正则表达式或 glob）
- `path` (string): 搜索目录
- `include` (string): 文件名过滤，如 *.py
- `context_lines` (integer): 上下文行数，默认3

**安全限制**:
- 只读操作，可并发执行
- 结果数量限制 50 条

---

### git_ops - Git 操作工具

安全地执行 git 命令。

**参数**:
- `action` (string, 必需): git_status/git_diff/git_log/git_commit
- `path` (string): 文件路径（git_diff）
- `message` (string): 提交消息（git_commit）
- `files` (array): 要提交的文件列表
- `n` (integer): 显示最近n条提交，默认10

---

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
