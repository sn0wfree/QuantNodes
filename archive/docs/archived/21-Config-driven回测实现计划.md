# Config-driven 回测实现计划

**创建日期**: 2026-04-30  
**更新日期**: 2026-04-30  
**分支**: refactor/factor-node-cleanup  
**状态**: 实施中

---

## 一、架构决策记录

### 1.1 原始方案（已废弃）

原方案采用 `Config → Code → BacktestTool` 路径：

```
YAML Config → ConfigLoader → ConfigCodeGenerator (regex 转换) → BacktestTool (sandbox) → 交易
```

**废弃原因**：
- regex 替换不可靠 — 试图用正则表达式重新实现表达式求值器
- 算子覆盖不全 — 只覆盖 ~9/46 个算子
- 跨截面语义错误 — `rank()` 生成 `.rank(pct=True)` 是行级排名，非跨截面排名
- 双重实现 — 同一个因子逻辑在 Polars 和 Pandas 各实现一遍

### 1.2 当前方案：直接 Polars 桥接

```
YAML Config → ConfigLoader → ConfigExecutor (Polars 因子计算) → ConfigBacktestRunner → backtest/ 引擎 (Pandas) → 交易
```

**优势**：
- 单点实现 — 因子逻辑只在 Polars 中实现一次
- 46/46 算子覆盖 — 所有已注册算子都可用
- 语义正确 — Polars 的跨截面操作是正确的
- 无 regex 依赖 — 使用 recursive descent parser

### 1.3 路径对比

| 维度 | 原方案 (Code Generator) | 当前方案 (Direct Bridge) |
|------|------------------------|-------------------------|
| 因子计算 | regex 替换为 Pandas | ConfigExecutor (Polars) |
| 算子覆盖 | ~9/46 | 46/46 |
| 跨截面语义 | 错误（行级 rank） | 正确 |
| 代码重复 | 有（Polars + Pandas 两套） | 无 |
| 安全机制 | CodeSandbox (AST 静态分析) | 声明式表达式（更严格） |

---

## 二、数据流架构

```
StrategyConfig + pl.LazyFrame
        │
        ▼
ConfigExecutor.run_backtest(config, data)
        │  result.data = LazyFrame (所有因子列 + signal 列)
        │  result.backtest = {signals, config, thresholds}
        ▼
┌─ ConfigBacktestRunner.run() ────────────────────────────────┐
│                                                              │
│  1. df = result.data.collect().to_pandas()                   │
│  2. df = _normalize_columns(df)  # 统一 Code/close 大小写   │
│  3. ConfigStrategyNode._generate_signals(df) → OrdersResult  │
│  4. PositionLimitRiskNode.execute(orders) → RiskResult       │
│  5. ExecutionBrokerNode.execute((orders, df)) → TradeResult  │
│  6. _compute_statistics(trade_result) → BacktestResult       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、文件变更清单

### 新建文件

| 文件 | 说明 | 预估行数 |
|------|------|---------|
| `backtest/config_strategy.py` | ConfigStrategyNode — 从 signal 列生成 Orders | ~50 |
| `backtest/config_runner.py` | ConfigBacktestRunner — 端到端编排器 | ~150 |

### 修改文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `agent/config/executor.py` | run_backtest() | 将 signal 列加入 result.data |
| `agent/tools/config_backtest.py` | 重写 | 改用 ConfigBacktestRunner |
| `backtest/__init__.py` | 追加导出 | ConfigStrategyNode, ConfigBacktestRunner |
| `tests/agent/test_config_executor.py` | 更新 | 删除 TestConfigCodeGenerator，新增 3 个测试类 |

### 删除文件

| 文件 | 原因 |
|------|------|
| `agent/config/generator.py` | ConfigCodeGenerator 已废弃 |

---

## 四、实施步骤

### Step 1: 修改 `executor.py` run_backtest()

**问题**：当前 `result.data` 不包含 `signal` 列，signal 只在 `result.backtest["signals"]` 中。

**修改**：在 `run_backtest()` 末尾，将 signal 列也加入 `result.data`：

```python
# 在 result.backtest = {...} 之后追加：
result.data = data.select(data.collect_schema().names())
```

**文件**: `agent/config/executor.py`，`run_backtest()` 方法

### Step 2: 新建 `ConfigStrategyNode`

**文件**: `backtest/config_strategy.py`

```python
class ConfigStrategyNode(StrategyNode):
    """从 DataFrame 的 signal 列生成交易信号。"""

    def __init__(self, signal_col: str = "signal", **kwargs):
        super().__init__(name="ConfigStrategy", **kwargs)
        self._signal_col = signal_col

    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]:
        signals = []
        for _, row in input_data.iterrows():
            sig_val = row.get(self._signal_col, 0)
            if sig_val == 0:
                continue
            signals.append(Signal(
                code=str(row.get("Code", row.get("code", ""))),
                signal_type="buy" if sig_val > 0 else "sell",
                strength=abs(float(sig_val)),
                price=float(row.get("Close", row.get("close", 0))),
                date=str(row.get("date", "")),
            ))
        return signals
```

### Step 3: 新建 `ConfigBacktestRunner`

**文件**: `backtest/config_runner.py`

核心方法：
- `run(config, data) → BacktestResult` — 主入口
- `_normalize_columns(df)` — 统一列名大小写
- `_build_risk_nodes(config)` — 从 config 构建风控节点
- `_build_broker(config)` — 从 config 构建经纪商
- `_apply_risk(orders, nodes)` — 应用风控过滤
- `_compute_statistics(trade_result, df, config)` — 计算绩效统计

### Step 4: 重写 `ConfigBacktestTool`

**文件**: `agent/tools/config_backtest.py`

核心变更：
- 移除 `ConfigCodeGenerator` 依赖
- 改用 `ConfigBacktestRunner`
- 新增 `_load_data()` 辅助方法

### Step 5: 更新 `backtest/__init__.py`

追加导出 `ConfigStrategyNode` 和 `ConfigBacktestRunner`。

### Step 6: 更新测试

**删除**: `TestConfigCodeGenerator`（3 个测试）

**新增**:
- `TestConfigStrategyNode` — 4 个测试
- `TestConfigBacktestRunner` — 5 个测试
- `TestConfigBacktestToolUpdated` — 2 个测试

### Step 7: 删除 `generator.py`

---

## 五、关键类接口

### ConfigStrategyNode

```python
class ConfigStrategyNode(StrategyNode):
    def __init__(self, signal_col: str = "signal", **kwargs): ...
    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]: ...
```

输入: DataFrame 包含 `Code`, `date`, `Close`, `signal` 列
输出: `List[Signal]`

### ConfigBacktestRunner

```python
class ConfigBacktestRunner:
    def run(self, config: StrategyConfig, data: pl.LazyFrame) -> BacktestResult: ...
```

输入: `StrategyConfig` + `pl.LazyFrame`
输出: `BacktestResult` (trades, orders, equity_curve, statistics, final_cash, total_return, sharpe_ratio, max_drawdown, win_rate)

### BacktestResult (已有)

```python
@dataclass
class BacktestResult:
    positions: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    equity_curve: pd.DataFrame
    statistics: Dict[str, Any]
    final_cash: float
    final_positions: Dict[str, float]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
```

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Polars→Pandas 列名大小写不一致 | 数据丢失 | `_normalize_columns()` 统一处理 |
| 零信号场景 | 空交易 | ConfigStrategyNode 返回空 OrdersResult |
| equity curve 计算复杂 | MVP 不完整 | 先返回 trades + cash，equity curve 延后 |
| 删除 generator.py 后旧测试失败 | 测试不通过 | 同步删除 TestConfigCodeGenerator |
| BacktestConfig.positions 未使用 | 仓位不限 | PositionLimitRiskNode 从 config 读取 |

---

## 七、验收标准

1. ✅ `ConfigBacktestRunner.run()` 可以端到端执行回测
2. ✅ ConfigStrategyNode 正确从 signal 列生成 Orders
3. ✅ PositionLimitRiskNode 从 BacktestConfig.positions 读取 max_positions
4. ✅ ExecutionBrokerNode 使用 BacktestConfig.slippage
5. ✅ 所有现有测试继续通过
6. ✅ 新增测试覆盖 ConfigStrategyNode 和 ConfigBacktestRunner
7. ✅ ConfigCodeGenerator 已删除，无残留引用
