# Alpha-GPT 用户指南

> **版本**：v1.0
> **日期**：2026-06-24
> **状态**：M6 PR 用户文档（doc-first 阶段）
> **适用项目**：QuantNodes v2.7.0+

---

## 1. 快速开始

### 1.1 安装

```bash
# 安装 QuantNodes（含 nanobot extras）
pip install 'quantnodes[all]'

# 配置 LLM（DeepSeek / OpenAI / Qwen 任选）
export QUANTNODES_LLM_PROVIDER=deepseek
export QUANTNODES_LLM_API_KEY=sk-...
export QUANTNODES_LLM_MODEL=deepseek-chat
# 或：openai / qwen / azure
```

### 1.2 准备数据

Alpha-GPT 需要包含 OHLCV + 至少 60 日历史的 Parquet 文件：

```python
import polars as pl
df = pl.read_parquet("data.parquet")
# 必须包含列：date, code, close, open, high, low, vol
# 可选列：vwap, amount, adj_close
```

数据格式要求：

| 列 | 类型 | 必须 | 说明 |
|----|------|------|------|
| `date` | date / str | ✅ | 交易日期 |
| `code` | str | ✅ | 股票代码 |
| `close` | float | ✅ | 收盘价 |
| `open` | float | ✅ | 开盘价 |
| `high` | float | ✅ | 最高价 |
| `low` | float | ✅ | 最低价 |
| `vol` | float | ✅ | 成交量 |
| `vwap` | float | ❌ | 成交量加权均价 |
| `amount` | float | ❌ | 成交额 |
| `adj_close` | float | ❌ | 后复权收盘价 |

### 1.3 第一次运行

```bash
quantnodes alpha-gpt \
  --objective "捕捉 A 股反转效应" \
  --iterations 5 \
  --pool-size 10 \
  --data data.parquet
```

输出：
```
🎯 Alpha-GPT 自动化因子挖掘
📊 数据：5000 票 × 2520 日（10 年）
🧠 LLM：DeepSeek-V3（via nanobot）
🔄 5 轮 × 10 候选 = 50 个公式
💹 Trading 回测：禁用

[Round 1/5] spawn idea-generator (3.2s)
  → 10 ideas
[Round 1/5] spawn formula-translator (2.8s)
  → 10 formulas
[Round 1/5] spawn evaluator (45.3s)
  → 10 evaluated, 8 success, 2 failed
  best: IC=0.045, IR=2.05
[Round 1/5] spawn reflector (4.1s)
  → 4 keep, 4 mutate, 2 drop

[Round 2/5] ...

🏆 Top 10 公式：
 1. rank(-ts_mean(returns, 20))              IR=2.05  IC=0.045
 2. ts_zscore(close / ts_delay(close, 60))   IR=1.78  IC=0.038
 3. rank(-ts_corr(close, vol, 10))           IR=1.65  IC=0.035
 ...

⏱️ 总耗时：315.4s
💾 结果已保存：alpha_pool.json
```

---

## 2. 完整 CLI 参数

```bash
quantnodes alpha-gpt [OPTIONS]
```

### 2.1 必选参数

| 参数 | 说明 |
|------|------|
| `--objective` / `-o` | 研究目标（如 "捕捉反转效应" / "动量因子挖掘"）|
| `--data` / `-d` | 数据路径（Parquet/CSV）|

### 2.2 工作流参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--iterations` / `-n` | 5 | 迭代轮次 |
| `--pool-size` / `-p` | 10 | 每轮想法/公式数量 |
| `--top-k` / `-k` | 10 | 最终返回的 top-K 公式数量 |
| `--min-ir` | 0.5 | IR 阈值（低于此的公式不入选 final pool）|
| `--max-mutual-ic` | 0.7 | mutual_IC 阈值（高于此的公式视为冗余）|

### 2.3 LLM 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--llm` | `deepseek` | LLM provider（`openai`/`deepseek`/`qwen`/`azure`/`mock`）|
| `--model` | 自动 | 模型名（如 `gpt-4o` / `deepseek-chat`）|
| `--temperature` | 0.7 | 采样温度 |
| `--max-tokens` | 4096 | 单次 LLM 调用最大输出 |

### 2.4 数据参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--date-column` | `date` | 日期列名 |
| `--code-column` | `code` | 股票代码列名 |
| `--forward-returns` | `1,5,20` | 前瞻收益期（逗号分隔）|
| `--start-date` | 全部 | 数据起始日期 |
| `--end-date` | 全部 | 数据结束日期 |

### 2.5 Trading 回测参数（需 `--backtest`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--backtest` | ❌ | 启用 Trading 回测 |
| `--top-k-backtest` | 10 | 跑回测的 top-K 数量 |
| `--initial-cash` | 1000000 | 初始资金 |
| `--commission` | 0.001 | 手续费率 |
| `--rebalance-freq` | 5 | 调仓频率（交易日）|

### 2.6 输出参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--output` / `-o` | `alpha_pool.json` | 结果保存路径 |
| `--verbose` / `-v` | ❌ | 详细输出 |
| `--quiet` / `-q` | ❌ | 安静模式（只输出 final pool）|

---

## 3. Python API

### 3.1 基础用法

```python
from QuantNodes.research.quant_alpha.workflow import (
    AlphaGptWorkflow, AlphaGptConfig,
)

config = AlphaGptConfig(
    objective="捕捉 A 股反转效应",
    iterations=5,
    pool_size=10,
    llm_provider="deepseek",
    forward_returns=[1, 5, 20],
)

workflow = AlphaGptWorkflow(config=config)
result = workflow.run(data_path="data.parquet")

print(f"Top {len(result.final_pool)} formulas:")
for f in result.final_pool:
    print(f"  IR={f.ir:.3f}  formula={f.formula}")
    print(f"    reason: {f.selection_reason}")
```

### 3.2 流式输出（实时观察）

```python
async def main():
    workflow = AlphaGptWorkflow(config=config)
    async for event in workflow.stream(data_path="data.parquet"):
        if event.type == "round_started":
            print(f"Round {event.round}/{event.total_rounds}")
        elif event.type == "subagent_done":
            print(f"  {event.subagent} done ({event.elapsed:.1f}s)")
        elif event.type == "formulas_evaluated":
            print(f"  best IR: {event.best_ir:.3f}")
        elif event.type == "final_pool_ready":
            print(f"🏆 {len(event.pool)} formulas selected")

import asyncio
asyncio.run(main())
```

### 3.3 REST API

```bash
# 启动工作流
curl -X POST http://localhost:8000/api/alpha/alpha-gpt/generate \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "捕捉 A 股反转效应",
    "iterations": 5,
    "data_path": "/path/to/data.parquet"
  }'
# → {"session_id": "abc-123", "status": "running"}

# 查询进度
curl http://localhost:8000/api/alpha/alpha-gpt/status/abc-123
# → {"round": 2, "total_rounds": 5, "best_ir": 1.78, "elapsed": 145.3}

# 获取结果
curl http://localhost:8000/api/alpha/alpha-gpt/results/abc-123
# → {"final_pool": [...], "summary": {...}}

# 停止
curl -X POST http://localhost:8000/api/alpha/alpha-gpt/stop/abc-123
```

### 3.4 WebSocket 实时流

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/api/alpha/alpha-gpt/stream") as ws:
    await ws.send(json.dumps({
        "objective": "捕捉反转效应",
        "iterations": 5,
    }))
    async for msg in ws:
        event = json.loads(msg)
        print(f"[{event['type']}] {event}")
```

事件类型：

| type | 说明 |
|------|------|
| `round_started` | 一轮开始 |
| `subagent_started` | subagent spawn |
| `subagent_done` | subagent 完成 + 输出 |
| `formulas_evaluated` | IC/IR 计算完成 |
| `round_completed` | 一轮结束 + summary |
| `final_pool_ready` | 全部完成 + final pool |
| `error` | 错误（会带 retry 信息）|

---

## 4. 结果格式

### 4.1 `alpha_pool.json` 结构

```json
{
  "metadata": {
    "objective": "捕捉 A 股反转效应",
    "iterations": 5,
    "pool_size": 10,
    "data_path": "data.parquet",
    "data_rows": 12600000,
    "llm_provider": "deepseek",
    "llm_model": "deepseek-chat",
    "started_at": "2026-06-24T10:30:00Z",
    "completed_at": "2026-06-24T10:35:15Z",
    "elapsed_seconds": 315.4
  },
  "final_pool": [
    {
      "rank": 1,
      "formula": "rank(-ts_mean(returns, 20))",
      "category": "reversal",
      "round_discovered": 1,
      "metrics": {
        "ic_mean": 0.045,
        "ic_std": 0.022,
        "ir": 2.05,
        "ic_decay": {"1d": 0.045, "5d": 0.038, "20d": 0.021},
        "turnover": 0.35
      },
      "backtest": {
        "annual_return": 0.142,
        "sharpe": 1.65,
        "max_drawdown": -0.123,
        "win_rate": 0.54
      },
      "selection_reason": "20 日反转因子...",
      "risk_notes": ["IC 在 5 日仍显著，但 20 日衰减较快"]
    }
  ],
  "summary": {
    "total_evaluated": 50,
    "passed_filters": 12,
    "selected": 10,
    "category_distribution": {"reversal": 4, "momentum": 3, "volatility": 2, "quality": 1},
    "avg_ir": 1.42,
    "best_ir": 2.05
  },
  "all_evaluations": [
    {"round": 1, "formula": "...", "status": "success", "ir": 2.05}
  ]
}
```

### 4.2 字段说明

| 字段 | 说明 |
|------|------|
| `formula` | polars 表达式字符串 |
| `ic_mean` | IC 均值（pearson） |
| `ic_std` | IC 标准差 |
| `ir` | IC 信息比率（= ic_mean / ic_std）|
| `ic_decay` | 不同前瞻期的 IC |
| `turnover` | top-K 组合的换手率 |
| `selection_reason` | Critic 的选择理由 |
| `risk_notes` | 风险提示（衰减、稳定性等）|

---

## 5. 实战技巧

### 5.1 选择 LLM

| LLM | 适用场景 | 成本 | 速度 |
|-----|----------|------|------|
| **DeepSeek-V3** (推荐) | 中文 prompt / 公式生成 / 成本敏感 | $0.1/次 | 快 |
| **GPT-4o** | 英文复杂 prompt / 最高质量 | $0.5/次 | 中 |
| **Qwen2.5-Coder-32B** | 代码生成优先 | $0.05/次 | 中 |
| **mock** | 单元测试 / 离线开发 | $0 | 即时 |

### 5.2 选择 Objective 写法

好的 objective：
- ✅ "捕捉 A 股反转效应"
- ✅ "挖掘 60 日长周期动量因子，避免换手率过高"
- ✅ "构造行业中性化的价值因子"

避免：
- ❌ "找好因子"（太宽泛）
- ❌ "复制 Alpha 101 全部公式"（应该用 M3 few-shot 直接调用）
- ❌ "预测明天涨跌"（不是因子问题，是预测问题）

### 5.3 调优 iterations vs pool_size

| 预算 | 推荐配置 |
|------|----------|
| 快速试错 | `--iterations 2 --pool-size 5`（10 个公式，~80s）|
| 标准 | `--iterations 5 --pool-size 10`（50 个公式，~5 分钟）|
| 深度 | `--iterations 10 --pool-size 20`（200 个公式，~15 分钟）|

### 5.4 与 M2 MCTS 对比

```bash
# LLM 驱动（创意强，适合"探索未知 alpha"）
quantnodes alpha-gpt --backend llm --objective "..." --iterations 5

# MCTS 驱动（基于 mutation，适合"在已知 alpha 基础上优化"）
quantnodes alpha-gpt --backend mcts --iterations 50

# Hybrid（未来 v2.9，先 LLM 出种子，再用 MCTS 优化）
quantnodes alpha-gpt --backend hybrid --llm-iterations 2 --mcts-iterations 30
```

### 5.5 启用 Trading 回测

```bash
quantnodes alpha-gpt \
  --objective "构造高 Sharpe 动量因子" \
  --iterations 5 \
  --pool-size 10 \
  --backtest \
  --top-k-backtest 10 \
  --initial-cash 1000000 \
  --commission 0.0015 \
  --data data.parquet
```

输出会包含每个 top-K 公式的 `backtest` 字段（年化收益 / Sharpe / 最大回撤）。

### 5.6 数据质量检查

在运行 Alpha-GPT 前，建议先用 `factor_test` 验证数据：

```python
from QuantNodes.research.quant_alpha.adapters import PolarsAlphaCalculator

calc = PolarsAlphaCalculator(data=df, forward_returns=[1, 5, 20])
ic = calc.calc_single_IC_ret("rank(ts_mean(close, 5))")
print(f"Sanity check IC: {ic.mean():.3f}")  # 应在 ±0.05 范围
```

---

## 6. 故障排除

### 6.1 spawn 超时

```
Error: spawn alpha-gpt-idea-generator timeout after 30s
```

解决：
- 增加 `--spawn-timeout 60`
- 检查 nanobot 是否正确安装 `pip show nanobot-ai`
- 检查 LLM API key 是否有效

### 6.2 公式执行失败

```
Error: Operator 'unknown_op' not in vocabulary
```

解决：
- 公式中的算子不在 162 算子白名单
- 检查 OperatorVocab: `vocab.list_vocab_operators()`
- 改用等价的标准算子

### 6.3 IC 全为 NaN

```
All formulas returned IC=NaN
```

解决：
- 数据不足（< 60 个交易日）
- 公式包含未来函数（Delay-0）
- 检查 `data` 列完整性

### 6.4 Token 限流

```
Error: Rate limit exceeded (429)
```

解决：
- 切换到 DeepSeek-V3（更宽松）
- 减少 `--pool-size` 或 `--iterations`
- 启用 TokenCountingClient 监控

### 6.5 Trading 回测慢

```
Backtest takes >5min per formula
```

解决：
- 减少 `--top-k-backtest` 到 5
- 增大调仓频率 `--rebalance-freq 20`
- 用更短的数据范围 `--start-date 2024-01-01`

---

## 7. 进阶用法

### 7.1 自定义 Subagent Prompt

高级用户可重写 `.agent/agents/alpha-gpt-*.md` 调整 agent 行为：

```bash
# 备份默认
cp .agent/agents/alpha-gpt-formula-translator.md \
   .agent/agents/alpha-gpt-formula-translator.local.md

# 编辑本地副本（会被优先加载）
vim .agent/agents/alpha-gpt-formula-translator.local.md
```

### 7.2 自定义 Few-shot

```python
from QuantNodes.research.quant_alpha.workflow import AlphaGptConfig

config = AlphaGptConfig(
    objective="...",
    custom_few_shot=[
        {"formula": "rank(-ts_mean(returns, 60))", "category": "reversal"},
        {"formula": "ts_zscore(high - low, 20)", "category": "volatility"},
    ]
)
```

### 7.3 分布式运行（未来 v2.9）

```bash
# 多机并行
quantnodes alpha-gpt \
  --objective "..." \
  --workers 4 \
  --shard-by stock
```

---

## 8. 引用

### 8.1 学术

如果 Alpha-GPT 对你的研究有帮助，请引用：

```bibtex
@article{wang2023alphagpt,
  title={Alpha-GPT: Human-Agent Collaborative Alpha Mining},
  author={Wang, Saizhuo and et al.},
  journal={arXiv preprint arXiv:2308.00016},
  year={2023}
}
```

### 8.2 内部文档

- [架构设计](alpha_gpt_architecture.md)
- [项目总规划](PROJECT_PLAN.md)
- [M4 PolarsAlphaCalculator 适配器](../Architecture-v2.6.md)

---

> **最后更新**：2026-06-24
> **版本**：v1.0（对应 QuantNodes v2.7.0）
