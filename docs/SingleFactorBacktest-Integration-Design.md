# 单因子回测节点化整合设计文档 / Single-Factor Backtest Node Integration Design

> **日期**: 2026-06-10
> **来源框架**: `~/Public/单因子回测` (DaisyZhou, 2019/11)
> **目标**: 将单因子回测的 12 项能力拆分为独立 Node，用 Pipeline 组合成完整回测管线

---

## 一、背景 / Background

### 1.1 来源框架概况

| 项 | 值 |
|---|---|
| 文件 | `date_utils.py` (303行) + `factor_utils.py` (830行) + `factor_performance.py` (1014行) + `factor_output.py` (616行) |
| 依赖 | pandas 0.24, numpy 1.16, scipy 1.2, statsmodels 0.10, xlwings, matplotlib, seaborn |
| 数据 | H5 文件 (固定路径 `./testdata/test_h5_new/`) |
| 输出 | xlwings 写 Excel |

### 1.2 核心能力 (QuantNodes 缺失的 5 项)

| 能力 | 说明 |
|------|------|
| 完整因子预处理管线 | 行业均值填充 + MAD 去极值 + 分位数缩尾 + Z-score 标准化 + 正态化 |
| 分组收益 + 多空净值 | N 分位分组、各组年化收益、多空组合净值、对冲净值 |
| 市值行业分层打分 | 3 市值组 × 29 中信行业 × N 分位 |
| 风险因子相关性 | Spearman 秩相关 + 稳定系数 |
| 完整评价体系 | 年化收益/累计收益/Sharpe/MDD(含持续回复时间)/胜率/盈亏比/Calmar |

### 1.3 安全问题

原框架 6 处 `exec()`/`eval()` 调用，迁移时全部替换为安全实现。

---

## 二、设计理念 / Design Principles

1. **万物皆 Node** — 每项能力封装为 `BaseNode` 子类
2. **Pipeline 是唯一组合原语** — 通过 YAML 配置编排节点
3. **先迁移后替换** — 先保持原框架行为正确，再逐步替换为 QuantNodes 算子
4. **混合模式数据传递** — 数据层严格串联，分析层 Context 共享

---

## 三、节点架构 / Node Architecture

### 3.1 总览

```
[严格 Pipeline 串联 — 数据层]
LoadData >> SampleFilter >> TradabilityFilter >> AdjustDate >> Preprocess >> Neutralize
                                                         │
[Context 共享 — 分析层]                                   │
context["factor_neutral"] ──┬──→ ICAnalyzer ──→ context["ic"]
                           ├──→ GroupAnalyzer ──→ context["group"]
                           │         │
                           │         ▼
                           │    LongShortNode ──→ context["longshort"]
                           │
                           ├──→ FactorScoreNode ──→ context["score"]
                           │
                           └──→ RiskCorrelationNode ──→ context["risk_corr"]
                                                         │
                                              FactorTestReportNode
```

### 3.2 12 个节点

| # | 节点 | 类名 | 输入 | 输出 | 原框架来源 |
|---|------|------|------|------|-----------|
| 1 | 加载数据 | `LoadDataNode` | config | `Dict[str, DataFrame]` | `factor_utils.Factor` |
| 2 | 样本池筛选 | `SamplePoolFilterNode` | stklist, trade_dt | `sample_mask` | `factor_utils.py:155-234` |
| 3 | 可交易性 | `TradabilityFilterNode` | factor, sample | `tradable_mask` | `factor_utils.py:250-308` |
| 4 | 调仓日 | `AdjustDateNode` | trade_dt | `adj_dates` | `date_utils.py:134-191` |
| 5 | 因子预处理 | `FactorPreprocessNode` | factor, tradable, industry | `factor_std` | `factor_utils.py:491-532` |
| 6 | 因子中性化 | `FactorNeutralizeNode` | factor_std, industry | `factor_neutral` | `factor_utils.py:534-625` |
| 7 | IC 分析 | `ICAnalyzerNode` | factor_neutral, price | `ic_result` | `factor_performance.py:111-158` |
| 8 | 分组分析 | `GroupAnalyzerNode` | factor_neutral, price | `group_result` | `factor_performance.py:361-560` |
| 9 | 多空组合 | `LongShortNode` | group_result | `longshort_result` | `factor_performance.py:562-617` |
| 10 | 市值行业打分 | `FactorScoreNode` | factor_neutral, mv, industry | `score_result` | `factor_performance.py:730-877` |
| 11 | 风险因子相关 | `RiskCorrelationNode` | factor_neutral, risk_factors | `corr_result` | `factor_performance.py:879-937` |
| 12 | 汇总报告 | `FactorTestReportNode` | all_results | `FactorTestReport` | `factor_output.py:539-616` |

### 3.3 混合模式数据传递

- **Phase 1 (严格串联)**: `LoadData >> SampleFilter >> TradabilityFilter >> AdjustDate >> Preprocess >> Neutralize`
  - 每个节点输出直接作为下一个的 `input_data`
  - 保证数据流的顺序性和确定性

- **Phase 2 (Context 共享)**: `ICAnalyzer / GroupAnalyzer / FactorScore / RiskCorrelation`
  - 所有分析节点从共享 `context` dict 按名字读取 `factor_neutral`、`price` 等
  - 可并行执行（未来优化点）

- **Phase 3 (依赖分析)**: `LongShortNode` 依赖 `GroupAnalyzerNode` 的输出
- **Phase 4 (输出)**: `FactorTestReportNode` 汇总所有结果

---

## 四、代码结构 / File Structure

```
QuantNodes/research/factor_test/
├── __init__.py
├── config.py
├── pipeline_runner.py
├── nodes/
│   ├── __init__.py
│   ├── load_data_node.py
│   ├── sample_pool_filter_node.py
│   ├── tradability_filter_node.py
│   ├── adjust_date_node.py
│   ├── factor_preprocess_node.py
│   ├── factor_neutralize_node.py
│   ├── ic_analyzer_node.py
│   ├── group_analyzer_node.py
│   ├── long_short_node.py
│   ├── factor_score_node.py
│   ├── risk_correlation_node.py
│   └── factor_test_report_node.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── date_utils.py
│   ├── performance_metrics.py
│   └── constants.py
└── tests/
    ├── __init__.py
    ├── test_nodes/
    └── test_pipeline_runner.py
```

---

## 五、YAML 配置示例 / YAML Config Example

```yaml
pipeline:
  name: "单因子回测 - EP"
  description: "EP因子有效性测试"

data:
  factor_path: "./testdata/test_h5_new/alpha/ep.h5"
  factor_key: "ep"
  format: "h5"
  api_path: "./testdata/test_h5_new/"
  load_keys: ["stklist", "trade_dt", "cp", "id_citic1", "mv_float", "index_cp"]

preprocess:
  sample_index: "HS300"
  sample_industry: "all"
  tradable:
    no_st: true
    no_suspended: true
    no_up_down_limit: false
    min_ipo_days: 360
    trace: null
  adj_mode: ["M", "end"]
  adj_date_beg: 20170801
  adj_date_end: 20171231
  missing: "ind_avg"
  extreme: "median"
  norm: "zscore"
  industry_neutral: false
  risk_neutral: false
  risk_factors:
    - ["risk_factor.h5", "Size"]
    - ["risk_factor.h5", "Value"]

analysis:
  ic:
    min_group_size: 5
  group:
    groups: 5
    factor_direction: 1
    floor_mode: "group"
    hedge: "equal"
    hedge_path: null
  longshort:
    factor_direction: 1
  score:
    enabled: true
  risk_corr:
    factors: "all"

output:
  dir: "./output/"
  format: ["parquet", "json"]
```

---

## 六、安全改造 / Security Fixes

| 原文件:行 | 原代码 | 改造为 |
|----------|--------|--------|
| `factor_utils.py:71` | `eval("\'/{}\' in h5_store.keys()")` | `key in h5_store` |
| `factor_utils.py:292` | `exec("trace_data = " + trace_i + ".copy()")` | `data = trace_dict[trace_i].copy()` |
| `factor_utils.py:726` | `exec("{}=Factor.get_apidata(dir_i)")` | `data[name] = loader.load(...)` |
| `factor_utils.py:728` | `exec("{}=Factor.get_customdata(dir_i)")` | `data[name] = loader.load(...)` |
| `factor_utils.py:730` | `eval("Factor.valid_shape({})")` | `loader.valid_shape(data[name])` |
| `factor_performance.py:901-909` | `exec()`/`eval()` | 字典查找替代 |

---

## 七、与 QuantNodes 现有能力的复用 / Reuse

| 新节点 | 可复用的 QuantNodes 能力 | 复用方式 |
|--------|------------------------|---------|
| `FactorPreprocessNode` | `section_ops.py:winsorize`, `standardizeZScore`, `fillNaNByFun` | Phase 2 替换 |
| `ICAnalyzerNode` | `factor_evaluator.py:230` `_compute_return_dimension()` | 提取逻辑 |
| `GroupAnalyzerNode` | `factor_evaluator.py:385` `_compute_monotonicity_dimension()` | 提取逻辑 |
| 其他节点 | 无直接复用 | 从原框架迁移 |

---

## 八、实施计划 / Implementation Plan

| Phase | 内容 | 工时 |
|-------|------|------|
| 1 | utils/ + config.py + __init__.py | 0.5 天 |
| 2 | 数据层 4 节点 | 1 天 |
| 3 | 预处理层 2 节点 | 1 天 |
| 4 | 分析层 4 节点 | 1.5 天 |
| 5 | 扩展层 2 节点 | 0.5 天 |
| 6 | pipeline_runner.py | 0.5 天 |
| 7 | 测试 | 1 天 |
| **总计** | | **~6 天** |

---

## 九、不做清单 / Out of Scope

- ❌ 不修改原 `~/Public/单因子回测/` 目录
- ❌ 不添加 xlwings 依赖
- ❌ 不修改 `core/` 或 `backtest/` 现有代码
- ❌ Phase 1 不替换算子（先迁移后替换）
