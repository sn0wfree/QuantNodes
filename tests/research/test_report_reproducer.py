# coding=utf-8
"""ResearchReportReproducer 单元测试"""

import json
import pytest
import numpy as np
import polars as pl
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from QuantNodes.research.report_reproducer import (
    ResearchReportReproducer,
    ExtractedLogic,
    ReproductionResult,
    ReproductionReport,
)
from QuantNodes.research.wiki import init_factor_wiki


@pytest.fixture
def sample_data():
    """生成模拟行情数据"""
    np.random.seed(42)
    n = 200
    dates = ["2024-01-01"] * 50 + ["2024-01-02"] * 50 + \
            ["2024-01-03"] * 50 + ["2024-01-04"] * 50
    codes = [f"SZ{i:06d}" for i in range(50)] * 4

    close = np.random.uniform(10, 100, n)
    return pl.DataFrame({
        "date": dates,
        "code": codes,
        "close": close,
        "open": close + np.random.normal(0, 1, n),
        "high": close + abs(np.random.normal(0, 2, n)),
        "low": close - abs(np.random.normal(0, 2, n)),
        "vol": np.random.uniform(1000, 100000, n),
        "forward_return": np.random.normal(0, 0.02, n),
    })


@pytest.fixture
def tmp_wiki():
    d = tempfile.mkdtemp()
    wiki_path = str(Path(d) / "test_wiki")
    init_factor_wiki(wiki_path)
    yield wiki_path
    shutil.rmtree(d)


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock()
    logics = [
        {
            "logic_type": "factor",
            "title": "20日动量因子",
            "description": "close / delay(close, 20) - 1",
            "formula": "close / delay(close, 20) - 1",
            "evidence": "我们发现20日动量因子在A股市场具有显著预测能力",
            "confidence": 0.9,
        },
        {
            "logic_type": "rule",
            "title": "金叉买入规则",
            "description": "当短期均线上穿长期均线时买入",
            "formula": None,
            "evidence": "金叉买入，死叉卖出",
            "confidence": 0.8,
        },
    ]
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=json.dumps(logics)))]
    client.chat.return_value = response
    return client


class TestExtractedLogic:

    def test_create(self):
        logic = ExtractedLogic(
            logic_type="factor",
            title="test",
            description="test desc",
            formula="rank(close)",
            evidence="evidence",
            confidence=0.9,
        )
        assert logic.logic_type == "factor"
        assert logic.formula == "rank(close)"
        assert logic.confidence == 0.9

    def test_optional_formula(self):
        logic = ExtractedLogic(
            logic_type="rule",
            title="test",
            description="desc",
            formula=None,
            evidence="ev",
            confidence=0.5,
        )
        assert logic.formula is None


class TestReproductionResult:

    def test_create(self):
        logic = ExtractedLogic(
            logic_type="factor", title="t", description="d",
            formula="rank(close)", evidence="e", confidence=0.9,
        )
        result = ReproductionResult(
            logic=logic,
            verification_status="verified",
            deviation="IC=0.05",
        )
        assert result.verification_status == "verified"
        assert result.deviation == "IC=0.05"


class TestParsePdf:

    def test_parse_pdf_fallback(self, tmp_wiki):
        """无 llmwikify/pymupdf 时回退"""
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        # 非PDF文件, 会触发回退
        title, text = reproducer._parse_pdf_fallback("/nonexistent.pdf")
        assert isinstance(title, str)
        assert isinstance(text, str)


class TestRuleBasedExtract:

    def test_extract_formula_patterns(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        text = """
        公式: close / delay(close, 20) - 1
        因子: rank(ts_mean(close, 20))
        """
        logics = reproducer._rule_based_extract(text)
        assert len(logics) >= 1
        assert all(isinstance(logic, ExtractedLogic) for logic in logics)

    def test_extract_no_match(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logics = reproducer._rule_based_extract("没有量化逻辑的普通文本")
        assert isinstance(logics, list)

    def test_extract_dedup(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        text = "公式: rank(close)\n因子: rank(close)"
        logics = reproducer._rule_based_extract(text)
        formulas = [logic.formula for logic in logics if logic.formula]
        # 应去重
        assert len(formulas) == len(set(formulas))


class TestLLMExtract:

    def test_llm_extract_success(self, tmp_wiki, mock_llm_client):
        reproducer = ResearchReportReproducer(
            wiki_path=tmp_wiki, llm_client=mock_llm_client
        )
        logics = reproducer._llm_extract("测试文本", "测试标题")
        assert len(logics) == 2
        assert logics[0].logic_type == "factor"
        assert logics[1].logic_type == "rule"

    def test_llm_extract_fallback_on_error(self, tmp_wiki):
        bad_client = MagicMock()
        bad_client.chat.side_effect = Exception("API error")
        reproducer = ResearchReportReproducer(
            wiki_path=tmp_wiki, llm_client=bad_client
        )
        logics = reproducer._llm_extract("公式: rank(close)", "test")
        # 应回退到规则匹配
        assert isinstance(logics, list)


class TestVerifyFactor:

    def test_verify_factor_valid(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logic = ExtractedLogic(
            logic_type="factor",
            title="test_factor",
            description="rank(close)",
            formula="rank(close)",
            evidence="test",
            confidence=0.9,
        )
        result = reproducer.verify_factor(logic, sample_data)
        assert isinstance(result, ReproductionResult)
        assert result.verification_status in ("verified", "failed")

    def test_verify_no_formula(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logic = ExtractedLogic(
            logic_type="rule",
            title="test",
            description="desc",
            formula=None,
            evidence="ev",
            confidence=0.5,
        )
        result = reproducer.verify_factor(logic, sample_data)
        assert result.verification_status == "unverifiable"

    def test_verify_invalid_formula(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logic = ExtractedLogic(
            logic_type="factor",
            title="bad",
            description="bad formula",
            formula="invalid_func(close)",
            evidence="ev",
            confidence=0.3,
        )
        result = reproducer.verify_factor(logic, sample_data)
        assert result.verification_status == "failed"


class TestWikiStorage:

    def test_store_verified_factor(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logic = ExtractedLogic(
            logic_type="factor",
            title="store_test",
            description="rank(close)",
            formula="rank(close)",
            evidence="test",
            confidence=0.9,
        )
        result = reproducer.verify_factor(logic, sample_data)
        if result.verification_status == "verified":
            reproducer._store_to_wiki(result)
            assert result.wiki_page_name is not None

    def test_store_pending_logic(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logic = ExtractedLogic(
            logic_type="rule",
            title="pending_test",
            description="金叉买入",
            formula=None,
            evidence="ev",
            confidence=0.8,
        )
        result = ReproductionResult(
            logic=logic,
            verification_status="pending",
        )
        reproducer._store_to_wiki(result)
        assert result.wiki_page_name is not None


class TestReportGeneration:

    def test_generate_report(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        results = [
            ReproductionResult(
                logic=ExtractedLogic(
                    logic_type="factor", title="f1", description="d1",
                    formula="rank(close)", evidence="e1", confidence=0.9,
                ),
                verification_status="verified",
                deviation="IC=0.05",
            ),
            ReproductionResult(
                logic=ExtractedLogic(
                    logic_type="rule", title="r1", description="d2",
                    formula=None, evidence="e2", confidence=0.8,
                ),
                verification_status="pending",
            ),
        ]
        report = reproducer.generate_report(results, "测试研报")
        assert "测试研报" in report
        assert "✅" in report
        assert "⏳" in report
        assert "verified" in report


class TestProcessE2E:

    def test_process_with_rule_extract(self, tmp_wiki, sample_data):
        """端到端: 规则提取 + 验证"""
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)

        # 模拟 PDF 文本 (跳过真实PDF解析)
        with patch.object(reproducer, 'parse_pdf', return_value=("test", "公式: rank(close)")):
            result = reproducer.process(
                pdf_path="/fake/report.pdf",
                data=sample_data,
                store_to_wiki=True,
            )

        assert isinstance(result, ReproductionReport)
        assert result.title == "test"
        assert result.elapsed_seconds > 0
        assert len(result.report_markdown) > 0

    def test_process_without_data(self, tmp_wiki):
        """无数据时只提取不验证"""
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)

        with patch.object(reproducer, 'parse_pdf', return_value=("test", "没有公式")):
            result = reproducer.process(
                pdf_path="/fake/report.pdf",
                data=None,
                store_to_wiki=False,
            )

        assert isinstance(result, ReproductionReport)
        assert result.total_logics >= 0


class TestExtractLogicFromText:

    def test_extract_logic_from_text_public(self, tmp_wiki, sample_pdf_text):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logics = reproducer.extract_logic_from_text(sample_pdf_text, "测试研报")
        assert isinstance(logics, list)

    def test_extract_logic_from_text_empty(self, tmp_wiki):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        logics = reproducer.extract_logic_from_text("", "空文本")
        assert isinstance(logics, list)

    def test_extract_logic_from_text_with_llm(self, tmp_wiki, sample_pdf_text, mock_llm_client):
        reproducer = ResearchReportReproducer(
            wiki_path=tmp_wiki, llm_client=mock_llm_client
        )
        logics = reproducer.extract_logic_from_text(sample_pdf_text, "测试标题")
        assert len(logics) == 2


class TestReproductionReportDataclass:

    def test_reproduction_report_fields(self, tmp_wiki):
        report = ReproductionReport(
            pdf_path="/path/to.pdf",
            title="测试研报",
            total_logics=5,
            verified=3,
            failed=1,
            pending=1,
            unverifiable=0,
            results=[],
            report_markdown="# Test",
            elapsed_seconds=1.5,
        )
        assert report.total_logics == 5
        assert report.verified == 3
        assert report.failed == 1
        assert report.elapsed_seconds == 1.5


class TestProcessWithInvalidPdf:

    def test_process_invalid_pdf_path(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        result = reproducer.process(
            pdf_path="/nonexistent/invalid.pdf",
            data=sample_data,
            store_to_wiki=False,
        )
        assert isinstance(result, ReproductionReport)


class TestProcessWithDataNoFormulas:

    def test_process_with_data_no_formulas(self, tmp_wiki, sample_data):
        reproducer = ResearchReportReproducer(wiki_path=tmp_wiki)
        with patch.object(reproducer, 'parse_pdf', return_value=("test", "普通文本没有公式")):
            result = reproducer.process(
                pdf_path="/fake/report.pdf",
                data=sample_data,
                store_to_wiki=True,
            )
        assert isinstance(result, ReproductionReport)
        assert result.failed == 0
