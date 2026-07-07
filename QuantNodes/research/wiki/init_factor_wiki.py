"""init_factor_wiki() function + markdown templates (PR6.6 / M4.3 split).

The init function generates a 'wiki' documentation page (in markdown format)
explaining the wiki layout, page types, relation types, and usage examples.

向后兼容: `from QuantNodes.research.wiki import init_factor_wiki` 仍可用.
"""
from __future__ import annotations

from llmwikify import create_wiki

# 8 markdown section templates (Factor / Strategy / Reproduction / 核心工具 etc.)
WIKI_INIT_MARKDOWN = """# QuantNodes Strategy Wiki 配置

> 本 Wiki 专为量化策略研究设计，用于存储因子、策略、研报逻辑等知识。

## Page Types

| Directory | Description |
|----------|-------------|
| Factor | 验证有效的因子（通过回测验证） |
| Logic | 从研报提取的因子逻辑/公式 |
| Strategy | 策略配置（参数、因子组合、回测设置） |
| Reproduction | 研报复现对比报告 |

## Relation Types

| Relation | Description |
|----------|-------------|
| uses | 策略使用因子 |
| correlates_with | 因子之间相关性 |
| derived_from | 因子来源于研报逻辑 |
| related_to | 通用关联 |
| outperforms | 策略A优于策略B |
| similar_to | 相似策略/因子 |
| contradicts | 矛盾/负相关发现 |
| supports | 回测结果支持策略假设 |
| validated | 因子已通过回测验证 |

## 操作流程（init 之后）

### 1. 因子研究流程
```
1. 读取研报 → 提取逻辑 → 写入 wiki/Logic/{topic}.md
2. 设计因子 → 配置参数 → 写入 wiki/Factor/{name}.md
3. 编写策略 → 使用因子 → 写入 wiki/Strategy/{name}.md
4. 运行回测 → 生成报告 → 写入 wiki/Reproduction/{name}.md
5. 添加关系 → 连接因子/策略/逻辑
```

### 2. 写入因子示例
```python
from QuantNodes.research.wiki import WikiFactorProxy

proxy = WikiFactorProxy(wiki_path="wiki")
factor = WikiFactor(
    name="momentum_20d",
    formula="rank(corr(rank(close), rank(time), 20))",
    category=WikiFactorCategory.MOMENTUM,
    source="研报/某券商Alpha研究.pdf",
    description="20日动量因子"
)
proxy.store_factor(factor)
```

### 3. 写入策略示例
```python
strategy = WikiStrategy(
    name="momentum_alpha_v1",
    factors=["momentum_20d", "volume_ratio_5d"],
    weight_method="equal_weight",
    rebalance="monthly"
)
proxy.store_strategy(strategy)
```

## Page Format Examples

### Factor Page
```markdown
---
title: momentum_20d
type: factor
created: 2026-05-09
updated: 2026-05-09
sources: [raw/alpha_research.pdf]
tags: [momentum, time_series]
---

# momentum_20d

## 因子公式
rank(corr(rank(close), rank(time), 20))

## 描述
20日动量因子，衡量过去20天的价格动量效应

## 验证结果
- IC: 0.05 (样本内), 0.03 (样本外)
- 回测年化收益: 12.3%
- 最大回撤: -8.5%

## 来源
- [Source: Alpha研究.pdf](raw/alpha_research.pdf)

## 关联
- uses: [[logic/momentum_theory]]
- similar_to: [[factor/momentum_60d]]
```

### Strategy Page
```markdown
---
title: momentum_alpha_v1
type: strategy
created: 2026-05-09
updated: 2026-05-09
tags: [momentum, equal_weight]
---

# momentum_alpha_v1

## 策略描述
基于动量因子的等权重组合策略

## 因子组合
- momentum_20d (权重: 0.5)
- momentum_60d (权重: 0.5)

## 回测设置
- 标的: 全市场 A 股
- 频率: 月度调仓
- 手续费: 万三

## 回测结果
- 年化收益: 15.2%
- 夏普比率: 1.8
- 最大回撤: -12.3%

## 关联
- uses: [[factor/momentum_20d]], [[factor/momentum_60d]]
- derived_from: [[logic/momentum_theory]]
```

### Reproduction Page
```markdown
---
title: 研报复现_海通Alpha动量
type: reproduction
created: 2026-05-09
updated: 2026-05-09
sources: [raw/ht_alpha_momentum.pdf]
---

# 研报复现_海通Alpha动量

## 研报信息
- 标题: Alpha动量因子研究
- 机构: 海通证券
- 日期: 2025-12

## 复现结果
| 指标 | 研报结果 | 复现结果 | 差异 |
|------|----------|----------|------|
| IC | 0.062 | 0.058 | -6.5% |
| 年化收益 | 18.5% | 16.2% | -12.4% |

## 差异分析
1. 样本期间差异（研报2019-2024，复现2020-2025）
2. 因子计算细节略有不同

## 结论
基本复现成功，差异在可接受范围内

## 关联
- derived_from: [[logic/momentum_ht]]
- validates: [[factor/momentum_20d]]
```

## 核心工具

| 操作 | API |
|------|-----|
| 存储因子 | `proxy.store_factor(factor)` |
| 获取因子 | `proxy.get_factor(name)` |
| 存储策略 | `proxy.store_strategy(strategy)` |
| 添加关系 | `proxy.add_relation(from, relation, to)` |
| 搜索 | `proxy.search_factors(query)` |

## 最佳实践

1. **先写Logic再写Factor** - 从研报提取逻辑，验证后再创建因子
2. **回测验证后再存储** - Factor页面应包含验证结果
3. **策略引用因子** - Strategy页面使用wikilink引用Factor
4. **记录复现过程** - Reproduction页面详细记录差异分析
5. **定期更新** - 市场变化后更新因子表现
"""


def init_factor_wiki(wiki_path: str, force: bool = False) -> None:
    """Initialize a wiki directory for factor research storage.

    Args:
        wiki_path: Path to the wiki directory.
        force: If True, re-initialize even if directory exists.
    """
    wiki = create_wiki(wiki_path)
    if not force and wiki.root.exists():
        pass
    else:
        wiki.init()
    wiki.write_page("wiki", WIKI_INIT_MARKDOWN)


__all__ = ["init_factor_wiki"]