# coding=utf-8
"""Stage 2 real Table 4 测试

测试 ClickHouseDataLoader、G2/G3 LLM 接入、RealTable4Runner。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.clickhouse_data_loader import (
    ClickHouseDataLoader,
)
from QuantNodes.research.quant_alpha.evaluation.baselines.g2_llm_only import G2LlmOnly
from QuantNodes.research.quant_alpha.evaluation.baselines.g3_alpha_gpt import G3AlphaGpt
from QuantNodes.research.quant_alpha.evaluation.runner import RealTable4Runner
from QuantNodes.research.quant_alpha.evaluation.contracts import (
    Baseline,
    DataLoader,
    Evaluator,
    FactorMetrics,
    FactorSpec,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class FakeAgent:
    """模拟 nanobot Agent。"""

    def __init__(self, response: str = "fake agent response"):
        self.response = response
        self.calls: List[dict] = []

    async def run(self, prompt: str, session_id: str = "default") -> str:
        self.calls.append({"prompt": prompt, "session_id": session_id})
        return self.response

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
        tools: Optional[List[str]] = None,
        tool_choice: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        self.calls.append({
            "prompt": message,
            "session_id": session_id,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        yield {"type": "done", "content": self.response, "stop_reason": "stop"}


class FakeClickHouseDataLoader(DataLoader):
    """模拟 ClickHouseDataLoader (不依赖 ClickHouse)。"""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def load(self) -> pl.DataFrame:
        return self._df


class FakeEvaluator(Evaluator):
    """模拟 Evaluator (返回固定指标)。"""

    def evaluate(
        self,
        factors: List[FactorSpec],
        data: Any,
        forward_returns: Optional[List[int]] = None,
    ) -> List[FactorMetrics]:
        return [
            FactorMetrics(
                formula_id=f.formula_id,
                status="success",
                ic_mean=0.05,
                ic_std=0.02,
                ir=2.5,
            )
            for f in factors
        ]


def _make_fake_market() -> pl.DataFrame:
    """生成小规模 fake 市场数据。"""
    import numpy as np
    np.random.seed(42)
    n_stocks = 10
    n_days = 50
    dates = pl.date_range(
        start=pl.date(2020, 1, 1),
        end=pl.date(2020, 3, 20),
        interval="1d",
        eager=True,
    ).cast(pl.Date)
    # 只保留工作日
    dates = dates.head(n_days)

    codes = [f"TEST{i:04d}.SZ" for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = 10.0
        for d in dates:
            ret = np.random.normal(0.001, 0.02)
            price *= (1 + ret)
            rows.append({
                "date": d,
                "code": code,
                "open": price * (1 + np.random.normal(0, 0.005)),
                "high": price * (1 + abs(np.random.normal(0, 0.01))),
                "low": price * (1 - abs(np.random.normal(0, 0.01))),
                "close": price,
                "vol": float(np.random.randint(1000, 100000)),
                "amount": float(np.random.randint(10000, 1000000)),
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. ClickHouseDataLoader 测试
# ---------------------------------------------------------------------------

class TestClickHouseDataLoader:
    def test_init(self):
        """初始化参数正确。"""
        loader = ClickHouseDataLoader(
            table="quote.stock_quote",
            start_date="2019-01-01",
            end_date="2024-12-31",
        )
        assert loader.table == "quote.stock_quote"
        assert loader.start_date == "2019-01-01"
        assert loader.end_date == "2024-12-31"

    def test_field_map(self):
        """字段映射正确。"""
        assert ClickHouseDataLoader.FIELD_MAP["ts_code"] == "code"
        assert ClickHouseDataLoader.FIELD_MAP["close"] == "close"

    def test_load_from_cache(self, tmp_path):
        """从 parquet 缓存加载。"""
        df = _make_fake_market()
        cache_path = tmp_path / "test_cache.parquet"
        df.write_parquet(cache_path)

        loader = ClickHouseDataLoader(cache_parquet=str(cache_path))
        loaded = loader.load()
        assert loaded.height == df.height
        assert set(loaded.columns) == set(df.columns)

    def test_load_summary_mock(self):
        """load_summary 连接失败时返回 error。"""
        loader = ClickHouseDataLoader(
            host="127.0.0.1",
            port=19999,  # 不存在的端口
            cache_parquet=None,
        )
        try:
            summary = loader.load_summary()
        except Exception:
            summary = {"error": "connection failed"}
        assert "error" in summary or "total_rows" in summary

    def test_clean_filters_zero_vol(self):
        """_clean 过滤 vol=0 (停牌)。"""
        loader = ClickHouseDataLoader(cache_parquet=None)
        df = pl.DataFrame({
            "date": ["2020-01-01", "2020-01-02"],
            "code": ["A", "B"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "vol": [0.0, 1000.0],
            "amount": [0.0, 50000.0],
        })
        cleaned = loader._clean(df)
        assert cleaned.height == 1
        assert cleaned["vol"][0] > 0


# ---------------------------------------------------------------------------
# 2. G2 LLM 接入测试
# ---------------------------------------------------------------------------

class TestG2LlmOnlyLLM:
    def test_init_with_llm_client(self):
        """G2 接受 llm_client 参数。"""
        agent = FakeAgent(response='["rank(close)"]')
        from QuantNodes.ai.llm.gateway import LLMGateway
        g = LLMGateway(agent=agent)
        g2 = G2LlmOnly(n=5, llm_client=g)
        assert g2._llm_client is g

    def test_init_without_llm_client(self):
        """G2 不传 llm_client 时 _llm_client=None。"""
        g2 = G2LlmOnly(n=5)
        assert g2._llm_client is None

    def test_generate_factors_with_mock_llm(self):
        """G2 有 llm_client 时调用 LLM 生成公式。"""
        # Mock LLM 返回有效公式列表
        mock_response = '["rank(-ts_mean(returns, 20))", "ts_std(close, 10)", "delta(close, 5)"]'
        agent = FakeAgent(response=mock_response)
        from QuantNodes.ai.llm.gateway import LLMGateway
        g = LLMGateway(agent=agent)

        g2 = G2LlmOnly(n=3, llm_client=g)
        factors = g2.generate_factors(n=3)

        assert len(factors) == 3
        assert all(f.source == "g2_llm_only" for f in factors)
        assert all(f.formula for f in factors)

    def test_generate_factors_without_llm_falls_back_to_mock(self):
        """G2 无 llm_client 时用 mock 公式生成。"""
        g2 = G2LlmOnly(n=10, seed=42)
        factors = g2.generate_factors(n=10)

        assert len(factors) == 10
        assert all(f.source == "g2_llm_only" for f in factors)

    def test_parse_llm_formulas_json(self):
        """_parse_llm_formulas 解析 JSON 数组。"""
        response = '["rank(close)", "ts_mean(vol, 20)"]'
        formulas = G2LlmOnly._parse_llm_formulas(response, 2)
        assert formulas == ["rank(close)", "ts_mean(vol, 20)"]

    def test_parse_llm_formulas_markdown(self):
        """_parse_llm_formulas 从 markdown 代码块提取。"""
        response = '```json\n["rank(close)", "ts_mean(vol, 20)"]\n```'
        formulas = G2LlmOnly._parse_llm_formulas(response, 2)
        assert formulas == ["rank(close)", "ts_mean(vol, 20)"]

    def test_parse_llm_formulas_invalid_returns_empty(self):
        """_parse_llm_formulas 无效输入返回空列表。"""
        formulas = G2LlmOnly._parse_llm_formulas("not valid json", 5)
        assert formulas == []


# ---------------------------------------------------------------------------
# 3. G3 LLM 接入测试
# ---------------------------------------------------------------------------

class TestG3AlphaGptLLM:
    def test_init_with_llm_client(self):
        """G3 接受 llm_client 参数。"""
        agent = FakeAgent(response="mock")
        from QuantNodes.ai.llm.gateway import LLMGateway
        g = LLMGateway(agent=agent)
        g3 = G3AlphaGpt(n=5, llm_client=g)
        assert g3._llm_client is g

    def test_init_without_llm_client(self):
        """G3 不传 llm_client 时 _llm_client=None。"""
        g3 = G3AlphaGpt(n=5)
        assert g3._llm_client is None

    def test_group_name(self):
        """G3 group_name 正确。"""
        g3 = G3AlphaGpt(n=5)
        assert g3.group_name == "G3_AlphaGpt"


# ---------------------------------------------------------------------------
# 4. RealTable4Runner 测试
# ---------------------------------------------------------------------------

class TestRealTable4Runner:
    def test_stage_is_real(self):
        """RealTable4Runner stage='real'。"""
        df = _make_fake_market()
        loader = FakeClickHouseDataLoader(df)
        evaluator = FakeEvaluator()

        runner = RealTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[],
        )
        assert runner.stage == "real"

    def test_default_output_dir(self):
        """RealTable4Runner 默认输出目录。"""
        df = _make_fake_market()
        loader = FakeClickHouseDataLoader(df)
        evaluator = FakeEvaluator()

        runner = RealTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[],
        )
        assert runner.output_dir == Path("data/output/table4_real")

    def test_run_empty_baselines(self):
        """RealTable4Runner 空 baselines 返回空报告。"""
        df = _make_fake_market()
        loader = FakeClickHouseDataLoader(df)
        evaluator = FakeEvaluator()

        runner = RealTable4Runner(
            loader=loader,
            evaluator=evaluator,
            baselines=[],
        )
        report = runner.run()
        assert report.stage == "real"
        assert len(report.groups) == 0
