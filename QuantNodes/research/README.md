# QuantNodes.research

Wiki 因子库代理层 + 自动因子研究 + 研报复现。

- **功能3A**: WikiFactorProxy — 因子库基础设施
- **功能3B**: ResearchReportReproducer — 研报复现 (PDF→逻辑提取→验证)
- **功能3C**: AutoResearcher — 自动因子挖掘 (模板枚举 + MCTS + 6维度评估)

## 快速开始

### 自动因子挖掘

```python
from QuantNodes.research import AutoResearcher, EvalConfig
import polars as pl

data = pl.read_parquet("market_data.parquet")
researcher = AutoResearcher(wiki_path="/path/to/wiki")

# 模板枚举 + 6维度评估
result = researcher.run(data=data, max_factors=100)

# MCTS 搜索
result = researcher.run(data=data, use_mcts=True, mcts_iterations=50)

# 验证单个因子
result = researcher.mine_single_factor(
    formula="rank(close / ts_lag(close, 20) - 1)",
    data=data,
)
print(result.report_markdown)
```

### 6维度评估

| 维度 | 指标 | 阈值 |
|------|------|------|
| 收益 | IC, IC_IR | \|IC\| > 0.03, IC_IR > 0.5 |
| 稳定性 | 滚动IC | > 0.6 |
| 分散度 | 因子间相关 | < 0.7 |
| 换手率 | 排名变化 | < 0.5 |
| 单调性 | 分组收益 | > 0.7 |
| 覆盖率 | 非空比例 | > 0.8 |

## 研报复现 (功能3B)

```python
from QuantNodes.research import ResearchReportReproducer
import polars as pl

data = pl.read_parquet("market_data.parquet")
reproducer = ResearchReportReproducer(
    wiki_path="/path/to/wiki",
    llm_client=llm_client,  # 可选, 不传则用规则匹配
)

# 处理研报PDF
report = reproducer.process(
    pdf_path="research_report.pdf",
    data=data,
    store_to_wiki=True,
)
print(report.report_markdown)
# verified: 因子类逻辑通过IC/IR验证
# pending: 非因子类逻辑存入WikiLogic待人工验证
```

## Wiki 因子库 (功能3A)

### 初始化

```python
from QuantNodes.research import init_factor_wiki, WikiFactorProxy

init_factor_wiki("/path/to/wiki")
proxy = WikiFactorProxy("/path/to/wiki")
```

### 存储/读取因子

```python
from QuantNodes.research import WikiFactor, FactorSource, FactorCategory

factor = WikiFactor(
    name="momentum_20d",
    formula="close / delay(close, 20) - 1",
    source=FactorSource.RESEARCH_REPORT,
    category=FactorCategory.MOMENTUM,
    ic_mean=0.05, icir=0.5,
)
proxy.store_factor(factor)
factor = proxy.get_factor("momentum_20d")
```

### 关系管理

```python
proxy.add_relation("Logic/report_001", "Factor/momentum_20d", "derived_from")
neighbors = proxy.get_neighbors("Factor/momentum_20d")
```

## 测试

```bash
python3 -m pytest tests/research/ -v
```
