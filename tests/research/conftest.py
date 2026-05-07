# -*- coding: utf-8 -*-
"""Research 子系统测试 fixtures"""
import os
import tempfile
import pytest
import polars as pl


@pytest.fixture
def wiki_path(tmp_path):
    """Wiki 临时目录路径"""
    wiki_dir = tmp_path / "test_wiki"
    wiki_dir.mkdir(exist_ok=True)
    return str(wiki_dir)


@pytest.fixture
def wiki_proxy(wiki_path):
    """WikiFactorProxy 实例（使用临时目录）"""
    from QuantNodes.research.wiki import WikiFactorProxy
    return WikiFactorProxy(wiki_path)


@pytest.fixture
def factor_evaluator():
    """FactorEvaluator 实例"""
    from QuantNodes.research.factor_evaluator import FactorEvaluator, EvalConfig
    config = EvalConfig(
        min_ic=0.01,
        min_ir=0.0,
        max_correlation=0.95,
    )
    return FactorEvaluator(config)


@pytest.fixture
def factor_miner():
    """FactorMiner 实例"""
    from QuantNodes.research.factor_miner import FactorMiner
    return FactorMiner()


@pytest.fixture
def eval_data():
    """因子评估用的样本数据（Polars DataFrame）"""
    return pl.DataFrame({
        'date': ['2024-01-01', '2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02', '2024-01-02',
                 '2024-01-03', '2024-01-03', '2024-01-03',
                 '2024-01-04', '2024-01-04', '2024-01-04'],
        'code': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C',
                 'A', 'B', 'C'],
        'close': [100.0, 200.0, 300.0, 101.0, 202.0, 303.0, 99.0, 198.0, 297.0,
                  102.0, 204.0, 306.0],
        'volume': [1000, 2000, 3000, 1100, 2100, 3100, 900, 1900, 2900,
                   1200, 2200, 3200],
        'forward_return': [0.01, 0.02, 0.015, -0.01, 0.01, 0.005, 0.02, -0.01, 0.01,
                          0.015, 0.02, 0.01],
    })


@pytest.fixture
def sample_factor():
    """样本因子候选"""
    from QuantNodes.research.factor_miner import FactorCandidate
    return FactorCandidate(
        name="test_ma5",
        formula="ts_mean(close, 5)",
        template="time_series_ma",
        source="manual",
    )


@pytest.fixture
def mock_wiki_pages(wiki_path):
    """预填充的 Wiki 页面，用于测试读取"""
    from pathlib import Path

    page_content = """---
title: 测试 WikiFactor
category: momentum
source: manual
created_at: 2024-01-01
tags: [test, ma]
---

## 因子描述

测试用因子，描述信息。

## 因子公式

```yaml
name: test_ma5
formula: ts_mean(close, 5)
category: momentum
source: manual
parameters:
  window: 5
```

## 相关策略

- dual_ma
- momentum
"""
    page_path = Path(wiki_path) / "test_wikifactor.md"
    page_path.write_text(page_content, encoding="utf-8")
    return str(page_path)


@pytest.fixture
def sample_pdf_text():
    """样本研报文本（用于 LLM 提取测试）"""
    return """
    量化研究报告：基于均线系统的选股策略

    本报告提出一种基于移动平均线的量化选股策略。

    因子构建：
    1. 短期均线：MA5 = TS_MEAN(close, 5)
    2. 长期均线：MA20 = TS_MEAN(close, 20)
    3. 均线差值：MA_diff = MA5 - MA20

    交易规则：
    - 当 MA5 上穿 MA20 时买入
    - 当 MA5 下穿 MA20 时卖出

    回测结果：
    - 年化收益率：15.3%
    - Sharpe比率：1.52
    - 最大回撤：-8.2%
    """
