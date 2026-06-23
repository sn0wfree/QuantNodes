---
name: config-driven
description: 配置驱动策略 — YAML 配置文件编写、配置验证、回测自动闭环。
---

# Config-Driven Strategy

QuantNodes v2.x 引入的**配置即策略**工作流：LLM 编写 YAML 配置而非 Python 代码，
自动闭环到 ConfigLoader → OperatorRegistry → ConfigBacktestRunner。

## YAML 结构

```yaml
version: "1.0"
name: "momentum_alpha_v1"

data:
  source: "clickhouse"  # or duckdb/mysql/csv/parquet
  conn_ini: "conn.ini"
  table: "quote.cn_stock"
  date_column: "date"
  code_column: "code"

factors:
  - name: momentum_20d
    formula: "close / close.shift(20) - 1"

operations:
  - type: time_series
    name: momentum_ma
    category: ts_mean
    inputs: [momentum_20d]
    params: {window: 20}

composite:
  - name: alpha
    formula: "momentum_ma"

backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
```

## 工作流

1. **配置编写** — LLM 根据研究主题生成 YAML 配置（无需写 Python）
2. **配置验证** — 调用 `config_backtest` 工具（带 `validate_only=true`），检查：
   - 数据源可达
   - 因子公式合法
   - 算子在 OperatorRegistry 中可查
3. **Agent 兜底** — 若算子不可表达，Agent 编写自定义算子补充
4. **回测执行** — 调用 `config_backtest` 自动闭环：config → 代码生成 → 验证 → 回测
5. **结果沉淀** — 调用 `wiki_write` 写入 Wiki（附 YAML 配置 + 回测结果）

## 工具集

| 工具 | 用途 |
|------|------|
| `config_backtest` | 配置驱动回测（核心） |
| `wiki_write` | 写入配置和结果 |

## 算子覆盖

默认 OperatorRegistry 覆盖 162 个常用算子（OperatorVocab）。
不可覆盖时 Agent 兜底写自定义算子。

## 反模式

- 不要在 YAML 中写 Python 代码（破坏配置驱动意义）
- 不要依赖未注册的算子（必须先验证）
- 不要忽略数据源配置（start with data section）
