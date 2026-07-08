# -*- coding: utf-8 -*-
"""Research 子系统测试 fixtures"""
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
    from QuantNodes.research.wiki.proxy import WikiFactorProxy
    return WikiFactorProxy(wiki_path)


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
    """样本因子候选 (使用 quant_alpha operator_vocab 替代 _legacy_3c.factor_miner.FactorCandidate)"""
    return {
        "name": "test_ma5",
        "formula": "ts_mean(close, 5)",
        "template": "time_series_ma",
        "source": "manual",
    }


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


# ── API Client Fixtures (paper / reproduction) ────────────────
# 注: paper/reproduction routers 仍在 llmwikify (v0.40 才迁), 但 reproduction
# DB 已迁到 QuantNodes.research.persist.sessions.ReproductionDatabase (v4.0.0).


class _FakeWiki:
    """In-memory wiki mock for API tests."""

    def __init__(self, tmp):
        self.wiki_dir = tmp
        self.written = []
        self._pages = {}

    def write_page(self, name, content, page_type=None):
        self.written.append((page_type, name, content))
        self._pages[name] = content

    def read_page(self, name):
        if name in self._pages:
            return {"page_name": name, "content": self._pages[name]}
        raise FileNotFoundError(f"Page {name} not found")

    def list_pages(self):
        return list(self._pages.keys())

    def search(self, query, limit=10):
        return []


class _FakeRegistry:
    """In-memory wiki registry mock."""

    def __init__(self, wiki):
        self._wiki = wiki

    def get_default_wiki(self):
        return self._wiki

    def get_wiki(self, wiki_id):
        return self._wiki

    def get_default_wiki_id(self):
        return "test-wiki"


@pytest.fixture
def paper_client(tmp_path, monkeypatch):
    """Paper router with isolated DB, wiki, raw/upload dirs."""
    from llmwikify.interfaces.server.http import paper as mod
    from QuantNodes.research.persist.sessions import ReproductionDatabase

    wiki = _FakeWiki(tmp_path / "wiki")
    wiki.wiki_dir.mkdir(parents=True, exist_ok=True)
    db = ReproductionDatabase(tmp_path / "repro.db")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    upload_dir = tmp_path / "papers"
    upload_dir.mkdir(exist_ok=True)

    mod.set_paper_deps(_FakeRegistry(wiki), db=db, raw_dir=raw_dir, upload_dir=upload_dir)
    monkeypatch.setattr(mod, "_WIKI_REGISTRY", _FakeRegistry(wiki))
    monkeypatch.setattr(mod, "_DB", db)
    monkeypatch.setattr(mod, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(mod, "_UPLOAD_DIR", upload_dir)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llmwikify.interfaces.server.http.paper import router

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), wiki, db


@pytest.fixture
def repro_client(tmp_path, monkeypatch):
    """Reproduction router with isolated DB and wiki."""
    import pandas as pd
    from llmwikify.interfaces.server.http import reproduction as mod
    from QuantNodes.research.persist.sessions import ReproductionDatabase

    db = ReproductionDatabase(db_path=tmp_path / "r.db")
    wiki = _FakeWiki(tmp_path / "wiki")
    wiki.wiki_dir.mkdir(parents=True, exist_ok=True)

    mod.set_repro_deps(db, _FakeRegistry(wiki))
    monkeypatch.setattr(mod, "_REPRO_DB", db)
    monkeypatch.setattr(mod, "_WIKI_REGISTRY", _FakeRegistry(wiki))

    monkeypatch.setattr(
        "QuantNodes.research.data_source.router.SynthDataSource.get",
        lambda self, s, st, e: (pd.DataFrame({"close": [10.0] * 60}), "synth"),
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llmwikify.interfaces.server.http.reproduction import router

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), wiki, db


# ── Pytest collection ignore ────────────────────────────────────
# test_e2e_paper.py is a script that starts its own server. Pytest collecting it
# produces errors. Exclude from collection.
collect_ignore = ["test_e2e_paper.py"]
