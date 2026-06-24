# Stage 2 Table 4 复现 — 数据需求清单

> v2.10 量化论文 Table 4 复现所需的全部数据、依赖、配置汇总。
> Stage 1 mock 不需要任何真实数据；Stage 2 real 需要以下全部就位。

---

## 1. 行情数据（必需，最重要）

### 1.1 全 A 股日线成交数据

| 项 | 值 |
|----|---|
| **存储路径** | `data/cache/full_a_2019_2024.parquet` |
| **VCS** | ❌ gitignore（不入仓） |
| **数据源** | iFinD（同花顺金融数据）|
| **粒度** | 日线 |
| **时间跨度** | 2019-01-01 ~ 2024-12-31（5 年）|
| **股票范围** | 全 A 股（沪深主板 + 中小板 + 创业板 + 科创板）|
| **Universe** | ~5000 票（含北交所 ~250 票）|
| **估算行数** | ~750 万行（5000 票 × 1250 日 × 1）|
| **估算大小** | ~600 MB（Float64 压缩 parquet）|

### 1.2 必需字段

| 字段 | 类型 | 必需 | 用途 |
|------|------|:----:|------|
| `date` | Date | ✅ | 时间索引 |
| `code` | Utf8 | ✅ | 股票代码（SH600000 / SZ000001）|
| `open` | Float64 | ✅ | 开盘价 |
| `high` | Float64 | ✅ | 最高价 |
| `low` | Float64 | ✅ | 最低价 |
| `close` | Float64 | ✅ | 收盘价 |
| `vol` | Float64 | ✅ | 成交量（股）|
| `amount` | Float64 | ✅ | 成交额（元）|
| `industry` | Utf8 | ✅ | 行业分类（IndNeutralize 用）|

### 1.3 可选字段（强烈建议）

| 字段 | 类型 | 必需 | 用途 |
|------|------|:----:|------|
| `vwap` | Float64 | ⭐ | 量价加权均价（Alpha 101 大量使用）|
| `adj_factor` | Float64 | ⭐ | 后复权因子（避免分红拆股失真）|
| `float_share` | Float64 | ⭐ | 流通股本（换手率计算）|
| `is_st` | Bool | ⭐ | 是否 ST / *ST（filter 剔除）|
| `market_cap` | Float64 | - | 总市值（市值中性化）|
| `list_date` | Date | - | 上市日期（新股剔除）|

### 1.4 Universe 过滤规则

| 过滤 | 阈值 | 原因 |
|------|------|------|
| ST / *ST | `is_st == True` 剔除 | 投机风险 |
| 上市 < 60 日 | `today - list_date < 60d` 剔除 | 新股不稳定 |
| 暂停上市 | `close.isnull()` 剔除 | 数据缺失 |
| 当日停牌 | `vol == 0` 剔除 | 无成交 |
| 北交所 | 可选剔除 | 流动性差 |

### 1.5 数据拉取建议

```python
from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcher

fetcher = IFindFetcher(token="your_ifind_token")

# 分批拉取（每次 100 票 × 250 日）
for batch_start in range(0, 5000, 100):
    codes = get_codes_batch(batch_start, 100)
    df = fetcher.query(
        server_type="stock_zh",
        tool_name="history_kline",
        params={
            "codes": codes,
            "start_date": "2019-01-01",
            "end_date": "2024-12-31",
            "fields": "date,code,open,high,low,close,vol,amount,industry",
        },
    )
    # 追加到 parquet
```

---

## 2. 行业分类数据（必需）

### 2.1 行业 schema

| 项 | 值 |
|----|---|
| **字段** | `code`, `industry_code`, `industry_name`, `level` |
| **来源** | 申万一级 / 中信一级 |
| **频次** | 季度更新（行业分类会变）|
| **存储路径** | `data/cache/sw_industry_2019_2024.parquet` |
| **估算大小** | ~1 MB |

### 2.2 推荐方案

- **方案 A**：iFinD `SW_industry` 接口直接拉取（需额外 iFinD 权限）
- **方案 B**：本地维护静态 csv + 季度手动更新（推荐，简单）
- **方案 C**：用 wind TDX 行业分类（需 wind 终端）

样例（`data/cache/sw_industry.csv`）：
```csv
code,industry_code,industry_name
SH600000,801010,农林牧渔
SH600519,801120,食品饮料
SZ000001,801190,银行
```

---

## 3. LLM 配置（必需，Stage 2 真实 LLM）

### 3.1 MiniMax LLM provider

| 项 | 值 |
|----|---|
| **API 格式** | OpenAI 兼容 |
| **Base URL** | `https://api.MiniMax.chat/v1`（待确认）|
| **API key** | 环境变量 `QUANTNODES__LLM__API_KEY` |
| **模型名** | `MiniMax-Text-01` 或类似（待确认）|
| **context window** | 200K tokens |
| **价格** | ~$0.2/1M tokens（参考）|

### 3.2 使用方式

```python
import os
os.environ["QUANTNODES__LLM__API_KEY"] = "sk-..."

from QuantNodes.research.quant_alpha.workflow.alpha_gpt import (
    AlphaGptConfig, AlphaGptWorkflow
)
from QuantNodes.ai.llm.nanobot_wrapper import NanobotLLMWrapper  # 或 MiniMax 直连

config = AlphaGptConfig(
    objective="...",
    iterations=3,
    pool_size=10,
    llm_provider="MiniMax",
)
workflow = AlphaGptWorkflow(config=config, llm_client=NanobotLLMWrapper(...))
```

### 3.3 LLM 调用预算估算

| 项 | 数量 | Tokens | 费用 |
|----|----:|-------:|-----:|
| idea-generator × 5 轮 | 5 × 10 = 50 | 100K | ~$0.02 |
| formula-translator × 5 轮 | 50 | 200K | ~$0.04 |
| reflector × 4 轮 | 4 × 20 = 80 | 100K | ~$0.02 |
| critic × 1 轮 | 1 | 30K | ~$0.01 |
| **合计** | | **~430K** | **~$0.10** |

Stage 2 真实数据 3 组对比 + 多次调参 ≈ $1-5 总费用。

---

## 4. 可选依赖

### 4.1 必需

| 包 | 版本 | 用途 | 安装 |
|----|------|------|------|
| polars | >= 0.20 | 数据处理（必装）| `pip install polars` |
| pyarrow | >= 14 | parquet 读写 | `pip install pyarrow` |
| pandas | >= 2.0 | iFinD 兼容 | `pip install pandas` |
| numpy | >= 1.24 | 数值计算 | `pip install numpy` |

### 4.2 推荐

| 包 | 版本 | 用途 | 安装 |
|----|------|------|------|
| ta-lib | >= 0.4 | 技术指标（Alpha 158）| `pip install ta-lib` |
| tables | >= 3.8 | h5 数据加载 | `pip install tables` |
| iFinD SDK | >= 2.0 | iFinD Python 接口 | 见同花顺官网 |
| scikit-learn | >= 1.3 | mutual IC 计算（可选）| `pip install scikit-learn` |
| matplotlib | >= 3.7 | 可视化（可选）| `pip install matplotlib` |

### 4.3 Stage 2 已有依赖

| 模块 | 路径 | 用途 |
|------|------|------|
| `IFindFetcher` | `QuantNodes/research/factor_test/ifind_db/fetcher.py` | Stage 2 DataLoader 复用 |
| `IFindFetcherStub` | 同上 | 测试 mock |
| `PolarsAlphaCalculator` | `QuantNodes/research/quant_alpha/adapters/calculator.py` | Stage 2 evaluator 复用 |

---

## 5. 配置文件

### 5.1 `.env` 需新增（Stage 2）

```bash
# === Stage 2 数据 ===
QUANTNODES__IFIND__TOKEN=your-ifind-token-here
QUANTNODES__IFIND__RATE_LIMIT=10          # 每秒请求数
QUANTNODES__DATA__CACHE_DIR=data/cache
QUANTNODES__DATA__FULL_A_PARQUET=data/cache/full_a_2019_2024.parquet
QUANTNODES__DATA__INDUSTRY_CSV=data/cache/sw_industry.csv

# === Stage 2 LLM ===
QUANTNODES__LLM__PROVIDER=MiniMax          # 或 nanobot
QUANTNODES__LLM__API_KEY=sk-MiniMax-...
QUANTNODES__LLM__BASE_URL=https://api.MiniMax.chat/v1
QUANTNODES__LLM__MODEL=MiniMax-Text-01
```

### 5.2 `config_mapper.py` 需扩展

`QuantNodes/agent/config_mapper.py` 当前只支持 LLM 配置，需扩展：
- `QUANTNODES__IFIND__TOKEN` → `.agent/nanobot_config.json`
- `QUANTNODES__DATA__CACHE_DIR` → `.agent/nanobot_config.json`

---

## 6. 仓库目录结构（Stage 2 准备）

```
QuantNodes/
├── data/                              # ⚠️ gitignored
│   ├── cache/
│   │   ├── full_a_2019_2024.parquet          # 必填 ~600 MB
│   │   └── sw_industry.csv                   # 必填 ~1 MB
│   └── output/
│       └── table4_real/
│           ├── table4_report.json
│           └── table4_report.md
├── QuantNodes/
│   ├── research/quant_alpha/
│   │   ├── evaluation/
│   │   │   ├── ifind_data_loader.py          # 新增
│   │   │   └── baselines/
│   │   │       ├── g2_llm_only.py            # 改造 (mock → MiniMax)
│   │   │       └── g3_alpha_gpt.py           # 改造 (注入 NanobotLLMWrapper)
│   │   └── ai/llm/
│   │       └── MiniMax.py                          # 新增
│   └── scripts/
│       ├── reproduce_table4_mock.py          # 已有
│       └── reproduce_table4_real.py          # 新增
└── tests/quant_alpha/
    ├── test_table4_*.py                       # 已有
    └── test_table4_real_*.py                  # 新增
```

---

## 7. Stage 2 启动检查清单

| # | 项 | 负责人 | 状态 | 备注 |
|---|----|----|:---:|------|
| 1 | iFinD 全 A 5 年 parquet | 用户 | ⏳ | `data/cache/full_a_2019_2024.parquet` |
| 2 | 行业分类 csv | 用户 | ⏳ | `data/cache/sw_industry.csv` |
| 3 | MiniMax API key | 用户 | ⏳ | `.env` + `config_mapper.py` |
| 4 | iFinD token | 用户 | ⏳ | `.env` |
| 5 | iFinD Python SDK | 用户 | ⏳ | `pip install iFinDPy`（如需要）|
| 6 | `QuantNodes/ai/llm/MiniMax.py` | 我 | 🔜 | Stage 2 #2.4 |
| 7 | `evaluation/ifind_data_loader.py` | 我 | 🔜 | Stage 2 #2.1 |
| 8 | `scripts/reproduce_table4_real.py` | 我 | 🔜 | Stage 2 #2.7 |
| 9 | Stage 2 测试 (~10) | 我 | 🔜 | Stage 2 #2.9 |
| 10 | graphify refresh | 我 | 🔜 | Stage 2 完成后 |
| 11 | v2.10 release | 我 | 🔜 | Stage 2 #2.10 |
| 12 | `config_mapper.py` 扩展 IFind | 我 | 🔜 | Stage 2 前置 |

---

## 8. 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| iFinD 限频（每秒 10 次）| 5000 票 × 50 批次 × 2 接口 ≈ 5000 次请求 → ~10 分钟 | 异步并发 + 本地缓存 |
| iFinD token 过期 | 数据拉取中断 | 用环境变量 + 错误重试 |
| MiniMax API 限频（每秒 60 次）| LLM 调用排队 | NanobotLLMWrapper 自带限频 |
| 数据不完整（停牌、退市）| Universe 过滤失败 | `load_summary()` 完整性检查 |
| 行业分类过期 | 中性化偏差 | 季度更新 csv |
| iFinD API 变更 | DataLoader 失效 | 集成测试 + Stub fallback |

---

## 9. 时间线

| 阶段 | 工作量 | 依赖 |
|------|------:|------|
| 用户准备数据 + API key | ~1d（用户并行）| 无 |
| Stage 2 #2.1 iFinDDataLoader | 0.3d | 数据就位 |
| Stage 2 #2.4 MiniMax provider | 0.3d | API key 就位 |
| Stage 2 #2.5-2.6 baseline 接入 | 0.4d | #2.1 + #2.4 |
| Stage 2 #2.7 main + 测试 | 0.6d | 全部就位 |
| Stage 2 #2.8 paper 对比 + 报告 | 0.3d | #2.7 跑通 |
| Stage 2 #2.9 集成测试 | 0.3d | #2.7 |
| Stage 2 #2.10 release prep | 0.2d | #2.9 |
| **合计** | **2.4d** | |

---

## 10. 相关文档

- `docs/quant_alpha/table4_reproduction.md` — Stage 1 mock + Stage 2 规划
- `docs/14-上游nanobot升级指南.md` — nanobot LLM 集成
- `docs/15-可选依赖安装指南.md` — Stage 2 依赖安装
- `docs/07-IFind集成.md`（如存在）— iFinD 接入文档