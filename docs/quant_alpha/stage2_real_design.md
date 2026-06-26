# Stage 2 Real Table 4 — 设计文档

## 数据源

| 项 | 值 |
|----|---|
| ClickHouse | localhost:8123, user=data, db=quote |
| 表名 | `quote.stock_quote` |
| 字段 | ts_code, trade_date, open, high, low, close, vol, amount |
| 2019-2024 | 6.62M rows, 5570 stocks |
| ts_code 格式 | "601107.SH" |
| trade_date 格式 | DateTime → cast Date |

## 字段映射

| ClickHouse | polars | 说明 |
|------------|--------|------|
| ts_code | code | 保持原格式 |
| trade_date | date | CAST AS Date |
| open/high/low/close | 同名 | 直接映射 |
| vol | vol | 直接映射 |
| amount | amount | 直接映射 |
| — | industry | 暂缺，Stage 2 先跳过 |

## 文件清单

| 文件 | 操作 |
|------|------|
| `evaluation/clickhouse_data_loader.py` | 新建 |
| `evaluation/baselines/g2_llm_only.py` | 改造 |
| `evaluation/baselines/g3_alpha_gpt.py` | 改造 |
| `evaluation/runner.py` | 新增 RealTable4Runner |
| `evaluation/__init__.py` | 导出 |
| `scripts/reproduce_table4_real.py` | 新建 |
| `tests/quant_alpha/test_table4_real_*.py` | 新建 |

## LLM

MiniMax 已配置:
- QUANTNODES__LLM__API_KEY=sk-cp-...
- QUANTNODES__LLM__BASE_URL=https://api.minimaxi.com/v1
- QUANTNODES__LLM__MODEL=minimax-M3

LLMGateway 自动路由到 MiniMax (nanobot → QuantNodesProvider → litellm → MiniMax API)。
