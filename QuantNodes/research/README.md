# QuantNodes.research

Wiki 因子库代理层，封装 llmwikify 调用，为研报复现 (3B) 和 AutoResearch (3C) 提供统一的因子/逻辑读写接口。

## 初始化

```python
from QuantNodes.research import init_factor_wiki, WikiFactorProxy

# 首次使用：初始化 Wiki 目录
init_factor_wiki("/path/to/wiki")

# 创建代理
proxy = WikiFactorProxy("/path/to/wiki")
```

## 存储因子

```python
from QuantNodes.research import WikiFactor, FactorSource, FactorCategory

factor = WikiFactor(
    name="momentum_20d",
    formula="close / delay(close, 20) - 1",
    source=FactorSource.RESEARCH_REPORT,
    category=FactorCategory.MOMENTUM,
    tags=["momentum", "medium_term"],
    ic_mean=0.05,
    icir=0.5,
    strategy_yaml="name: momentum_strategy\nfactors:\n  - momentum_20d",
)
proxy.store_factor(factor)
```

## 读取因子

```python
factor = proxy.get_factor("momentum_20d")
factors = proxy.search_factors("momentum")
factors = proxy.list_factors(source=FactorSource.RESEARCH_REPORT)
```

## 更新 / 删除

```python
proxy.update_factor("momentum_20d", {"ic_mean": 0.08})
proxy.delete_factor("momentum_20d")
```

## 存储研报逻辑

```python
from QuantNodes.research import WikiLogic, LogicSource

logic = WikiLogic(
    name="report_001",
    content="根据研报分析，动量因子在A股市场具有显著预测能力...",
    source=LogicSource.RESEARCH_REPORT,
    extracted_formula="close / delay(close, 20) - 1",
    related_strategies=["strategy_a"],
    related_factors=["momentum_20d"],
    validation_status="validated",
)
proxy.store_logic(logic)
```

## 关系管理

```python
proxy.add_relation("Logic/report_001", "Factor/momentum_20d", "derived_from")
neighbors = proxy.get_neighbors("Factor/momentum_20d")
```

## 状态检查

```python
proxy.ping()   # True/False
proxy.status() # dict
```

## 测试

```bash
python3 -m pytest tests/research/test_wiki.py -v
```
