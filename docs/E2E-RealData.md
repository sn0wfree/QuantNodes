# 真实数据 E2E 测试 / Real-Data End-to-End Test

> 完整 12 节点 + 多轮演化 + RAG + 可视化, 一键跑通
> Version: 1.0  |  Date: 2026-06-11

---

## 1. 概述

Week 11 引入 2 个 E2E 脚本, 把 Weeks 1-10 的所有功能串成 1 个完整流程:

```
[数据准备] → [12 节点单回测] → [多轮演化] → [RAG 评估] → [可视化报告]
```

支持 3 种数据源:
- **合成数据 (默认)**: `data_prep.py` 生成 HDF5, 模拟 iFinD 真实数据结构
- **真实 iFinD**: 用 `IFindFetcher` + API key (在 `~/.agents/skills/ifind/mcp_config.json`)
- **自定义 H5**: 用户提供符合 `stk_daily.h5` / `index_daily.h5` 格式的数据

---

## 2. 快速开始

### 2.1 合成数据 (无需 API key)

```bash
# 1. 生成数据 (60 天 × 20 股票 × 3 因子)
python -m QuantNodes.research.factor_test.e2e.data_prep \
       --output-dir /tmp/e2e_data/ \
       --n-days 60 --n-stocks 20 \
       --factors momentum_20d,reversal_5d,volatility_60d

# 2. 跑 E2E
python -m QuantNodes.research.factor_test.e2e.run_evolution_e2e \
       --data-path /tmp/e2e_data/ \
       --output-dir /tmp/e2e_output/ \
       --max-rounds 3
```

### 2.2 真实 iFinD 数据

```bash
# 需要先在 ~/.agents/skills/ifind/mcp_config.json 配置 auth_token

# 1. 用 iFinD 拉数据
python -c "
from QuantNodes.research.factor_test.ifind_db import IFinDDatabase
db = IFinDDatabase(
    date_beg='20260101', date_end='20260630',
    universe='沪深300',
    # 默认从 ~/.agents/skills/ifind/mcp_config.json 读 auth_token
)
db.fetch_to_h5('/tmp/real_data/')  # 用户需实现此方法 (TODO)
"

# 2. 同样跑 E2E (--data-path 指向真实数据)
python -m QuantNodes.research.factor_test.e2e.run_evolution_e2e \
       --data-path /tmp/real_data/ \
       --max-rounds 3
```

---

## 3. 输出结构

```
{output_dir}/
├── trajectory/                       # TrajectoryPool 双层持久化
│   ├── trajectories.parquet          # 元数据 (15 列 × N entries)
│   └── {entry_id}.json               # 完整记录 (含 feedback + metrics)
├── factor_test_{stklist}_{ts}.json   # 12 节点结果
├── evolution_report.html             # Plotly 交互报告
└── evolution_summary.json            # 演化统计 + RAG 指标
```

### evolution_summary.json 示例

```json
{
  "data_path": "/tmp/e2e_data/",
  "output_dir": "/tmp/e2e_output/",
  "directions": ["momentum", "reversal", "volatility"],
  "max_rounds": 2,
  "pool_size": 4,
  "rounds_completed": 2,
  "total_count": 4,
  "rejected_count": 0,
  "best_entries": [
    {"id": "...", "name": "momentum_20d", "operation": "original",
     "round": 0, "sharpe": 0.0},
    ...
  ],
  "rag_metrics_history": [
    {"round": 1, "n_queries": 3, "hit_at_5": 1.0,
     "ndcg_at_5": 1.0, "mrr": 1.0, "lineage_coverage": 0.0,
     "diversity": 1.0},
    ...
  ]
}
```

---

## 4. CLI 参数

### 4.1 data_prep

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | 必填 | 输出目录 (将创建 H5 文件) |
| `--n-days` | 120 | 工作日数量 |
| `--n-stocks` | 30 | 股票数量 |
| `--factors` | `momentum_20d,reversal_5d,volatility_60d` | 因子列表 (决定 IC 方向) |
| `--seed` | 42 | 随机种子 |

### 4.2 run_evolution_e2e

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-path` | 必填 | data_prep 输出目录 (或真实数据目录) |
| `--factor-name` | `momentum_20d` | 起始因子名 (round 0 用) |
| `--directions` | `momentum,reversal,volatility` | 逗号分隔研究方向 |
| `--output-dir` | `/tmp/e2e_output/` | 输出目录 |
| `--max-rounds` | 3 | 演化轮数 (不含 round 0) |
| `--disable-quality-gate` | False | 禁用 QualityGate (默认启用) |
| `--disable-kb` | False | 禁用 KnowledgeBase + RAG (默认启用) |

---

## 5. 5 阶段流程详解

### 5.1 [1/5] 数据注入

- 从 H5 读 `cp / st / suspend / ud_limit / ipo_days / industry / mv_float`
- 构造 `DataLoader` 实例, 注入到 `runner._context['LoadData']`
- 设置 `_loader` 字段 (供 RiskCorrelationNode 使用)

### 5.2 [2/5] 12 节点单回测

- 跳过 `LoadDataNode` (已注入)
- 跑 `SamplePoolFilter → TradabilityFilter → AdjustDate → FactorPreprocess → FactorNeutralize`
- 跑 `ICAnalyzer → GroupAnalyzer → LongShort → FactorScore → RiskCorrelation`
- 跑 `FactorTestReport`
- 输出 ctx 包含 IC 均值 / 分组收益 / Sharpe / MDD 等指标

### 5.3 [3/5] 演化组件

- `TrajectoryPool`: `output_dir/trajectory/`
- `QualityGateNode`: 默认启用 (3 门都开)
- `KnowledgeBase + RAGEvaluator`: 索引 round 0 entry, 评估每轮

### 5.4 [4/5] 演化循环

- `EvolutionLoop.run(initial_directions=...)`
- round 0: 3 个 original (用 `Hypothesizer` mock)
- round 1: mutation (1 parent, 1 child)
- round 2: crossover (2 parents, 1 child)
- 每 round 1+ 自动 `_evaluate_rag()` 记录 5 指标

### 5.5 [5/5] 报告

- `generate_html()` → `evolution_report.html` (含 5 个 Plotly figure)
- `evolution_summary.json` (含 best entries + RAG 指标历史)

---

## 6. 验证项

运行 E2E 后, 应满足:

| 验证项 | 期望 | 失败排查 |
|--------|------|----------|
| `pool_size` ≥ 3 | round 0 + 至少 1 轮 | 检查 `max_rounds` 参数 |
| `total_count` ≥ 2 | QualityGate 没全拦 | 调整 gate 阈值 |
| `best_entries` 非空 | 至少 1 个通过 | 检查 factor 表达式 |
| `evolution_report.html` 存在 | > 10 KB | 写权限 |
| `evolution_summary.json` 有效 | JSON 解析成功 | - |
| `rag_metrics_history` 长度 | = `max_rounds` | 检查 KB sync |

---

## 7. 与真实 iFinD 集成

### 7.1 数据格式约定

`data_prep` 生成的 H5 与 `LoadDataNode` 期望一致:

```
stk_daily.h5:  cp, st, suspend, ud_limit, ipo_days, id_citic1, mv_float
index_daily.h5: index_cp
stklist.h5, trade_dt.h5: 单列 DataFrame
{factor_name}.h5: 单 key='data', shape=(n_days, n_stocks)
```

### 7.2 真实数据拉取 (TODO)

需要在 `IFinDDatabase` 增加 `fetch_to_h5(output_dir)` 方法:

```python
def fetch_to_h5(self, output_dir: str) -> None:
    """从 iFinD 拉数据, 写到 output_dir (HDF5 格式)。
    
    实现思路:
    1. cp = iFinD 查询 '收盘价'
    2. st = iFinD 查询 'ST 标记'
    3. ... (其他 key)
    4. pd.HDFStore 写每 key
    """
    # TODO: Week 12+
```

未来可参考:
- QuantaAlpha `quantaalpha/data/ifind_fetcher.py` 实现
- iFinD API: `THS_iFinDStock` / `THS_IndexCP` 等

### 7.3 iFinD 限流

- 免费版: 2 QPS
- `IFindFetcher` 已设 0.5s 间隔 (Week 4 iFinD Database)
- 建议: 缓存 7 天 (Parquet), 二次跑用本地缓存

---

## 8. 测试覆盖

`tests/e2e/test_realdata_e2e.py` 提供 8 个测试:

1. `test_data_prep_basic`: 生成 H5 验证格式
2. `test_run_evolution_e2e_synthetic`: 完整 E2E (合成数据)
3. `test_run_evolution_e2e_with_real_ifind`: 真实 iFinD (skip if 无 key)
4. `test_pipeline_runner_skip_load_data`: 注入 LoadData 后 run() 不再调 LoadDataNode
5. `test_quality_gate_rejects_in_e2e`: 启用 QG 时, 低质量因子被拦
6. `test_evolution_pool_size_meets_expectations`: pool size ≥ 3
7. `test_html_report_generated`: 报告文件存在 + size > 10KB
8. `test_summary_json_valid`: JSON 含必要字段

---

## 9. 故障排查

### 9.1 `数据加载器缺失` 错误

`RiskCorrelationNode` 需要 `_loader` 字段。E2E 脚本已自动注入, 若手动跑:

```python
from QuantNodes.research.factor_test.utils.data_loader import DataLoader
runner._context["LoadData"]["_loader"] = DataLoader("/tmp/e2e_data/")
```

### 9.2 IC 全部为 NaN

- 检查 `preprocess.adj_date_beg/end` 与因子 H5 时间范围一致
- 检查 `industry_neutral=True` 时 `id_citic1` 是否完整
- 检查 `min_group_size` 是否太大 (默认 5, 30 股票够用)

### 9.3 TrajectoryPool 写入失败

检查磁盘权限 + `output_dir` 是否可写:

```bash
chmod 755 /tmp/e2e_output/
```

### 9.4 iFinD API 超时

- 检查 `~/.agents/skills/ifind/mcp_config.json` 的 `auth_token` 是否过期
- 减小 `universe` (如 '沪深300' → '中证500')
- 增加缓存期 (7 天 → 30 天)

---

## 10. 参考

- QuantaAlpha `quantaalpha/pipeline/loop.py:209` — 5 步演化循环
- 12 节点定义: `docs/SingleFactorBacktest-Integration-Design.md`
- FactorFeedback: `docs/FactorFeedback.md`
- TrajectoryPool: `docs/TrajectoryPool.md`
- QualityGate: `docs/QualityGate.md`
- 演化框架: `docs/Evolution-Framework.md`
- RAG 评估: `docs/` (Week 10 metrics)

---

*Last updated: 2026-06-11*
