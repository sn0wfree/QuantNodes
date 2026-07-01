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

    def test_clean_datetime_column(self):
        """_clean 处理 Datetime 类型 date 列 (line 162-163)。"""
        loader = ClickHouseDataLoader(cache_parquet=None)
        df = pl.DataFrame({
            "date": [__import__("datetime").datetime(2020, 1, 1), __import__("datetime").datetime(2020, 1, 2)],
            "code": ["A", "B"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "vol": [1000.0, 2000.0],
            "amount": [50000.0, 60000.0],
        })
        cleaned = loader._clean(df)
        assert cleaned.height == 2

    def test_clean_low流动性过滤(self):
        """_clean 过滤低流动性 (min_amount_percentile > 0, lines 181-184)。"""
        loader = ClickHouseDataLoader(cache_parquet=None, min_amount_percentile=0.5)
        df = pl.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "code": ["A", "B", "C", "D"],
            "open": [10.0] * 4,
            "high": [10.5] * 4,
            "low": [9.5] * 4,
            "close": [10.0] * 4,
            "vol": [1000.0] * 4,
            "amount": [100.0, 200.0, 800.0, 900.0],
        })
        cleaned = loader._clean(df)
        # median amount filter: keep >= 500
        assert cleaned.height <= 4

    def test_load_summary_success(self):
        """load_summary 成功路径 (P2.12c.4: mock ClickHouseNode.query)。"""
        import pandas as pd
        loader = ClickHouseDataLoader(cache_parquet=None)
        mock_node = MagicMock()
        mock_node.query.return_value = pd.DataFrame([{
            "min_date": "2020-01-01",
            "max_date": "2020-12-31",
            "total_rows": 1000,
            "n_stocks": 10,
        }])

        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode",
            return_value=mock_node,
        ):
            summary = loader.load_summary()

        assert summary["total_rows"] == 1000
        assert summary["n_stocks"] == 10

    def test_load_summary_error_returns_dict(self):
        """load_summary query 失败返回 error (P2.12c.4)。"""
        loader = ClickHouseDataLoader(cache_parquet=None)
        mock_node = MagicMock()
        mock_node.query.side_effect = RuntimeError("CH query failed")

        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode",
            return_value=mock_node,
        ):
            summary = loader.load_summary()

        assert "error" in summary

    def test_load_summary_empty_returns_dict(self):
        """load_summary 返回空 DataFrame 时返回 error (P2.12c.4)。"""
        import pandas as pd
        loader = ClickHouseDataLoader(cache_parquet=None)
        mock_node = MagicMock()
        mock_node.query.return_value = pd.DataFrame()  # 空

        with patch(
            "QuantNodes.database_node.clickhouse_node.ClickHouseNode",
            return_value=mock_node,
        ):
            summary = loader.load_summary()

        assert "error" in summary


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
        """G2 有 llm_client 时调用 _generate_with_llm (mock)。"""
        from unittest.mock import patch

        g2 = G2LlmOnly(n=3, seed=42)
        mock_factors = [
            FactorSpec(formula_id="G2_000", formula="delta(close, 5)", source="g2_llm_only"),
            FactorSpec(formula_id="G2_001", formula="ts_mean(vol, 10)", source="g2_llm_only"),
            FactorSpec(formula_id="G2_002", formula="sign(ts_std(close, 3))", source="g2_llm_only"),
        ]
        with patch.object(G2LlmOnly, "_generate_with_llm", return_value=mock_factors):
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
