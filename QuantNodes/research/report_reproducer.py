# coding=utf-8
"""
研报复现 - ResearchReportReproducer

从研报PDF中提取因子公式、交易规则等量化逻辑，
验证其有效性，存入Wiki因子库。

流程: PDF解析 → LLM逻辑提取 → 因子验证 → Wiki存储 → 报告生成
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from QuantNodes.research.quant_alpha.evaluation.contracts import (
    FactorSpec,
    FactorMetrics,
    VerifyConfig,
)
from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.wiki import (
    FactorCategory,
    FactorSource,
    LogicSource,
    WikiFactor,
    WikiFactorProxy,
    WikiLogic,
)


# ==================== 数据模型 ====================

@dataclass
class ExtractedLogic:
    """从研报提取的逻辑"""
    logic_type: str           # factor | rule | condition | combination
    title: str                # 逻辑标题
    description: str          # 原文描述
    formula: Optional[str]    # 因子公式 (factor 类型必填)
    evidence: str             # 原文依据
    confidence: float         # LLM 提取置信度 (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproductionResult:
    """单条逻辑的复现结果"""
    logic: ExtractedLogic
    verification_status: str = "pending"  # verified | failed | pending | unverifiable
    factor_result: Optional[FactorMetrics] = None
    deviation: str = ""
    wiki_page_name: Optional[str] = None


@dataclass
class ReproductionReport:
    """研报复现报告"""
    pdf_path: str
    title: str
    total_logics: int = 0
    verified: int = 0
    failed: int = 0
    pending: int = 0
    unverifiable: int = 0
    results: List[ReproductionResult] = field(default_factory=list)
    report_markdown: str = ""
    elapsed_seconds: float = 0.0


# ==================== Prompt 模板 ====================

EXTRACTION_PROMPT = """你是一个量化研究分析助手。从以下研报文本中提取所有量化逻辑。

研报标题: {title}

研报文本:
{text}

请提取以下类型的逻辑:
1. factor: 因子公式 (如 "close / delay(close, 20) - 1")
2. rule: 交易规则 (如 "金叉买入，死叉卖出")
3. condition: 筛选条件 (如 "市值 > 100亿")
4. combination: 组合规则 (如 "等权配置前20只")

对每个逻辑，返回 JSON:
{{
  "logic_type": "factor|rule|condition|combination",
  "title": "逻辑标题",
  "description": "原文描述",
  "formula": "因子公式 (factor类型必填, 其他类型为null)",
  "evidence": "原文依据 (引用原文段落)",
  "confidence": 0.0-1.0
}}

返回 JSON 数组，不要其他内容。"""


# ==================== 编排器 ====================

class ResearchReportReproducer:
    """研报复现系统"""

    def __init__(
        self,
        wiki_path: str,
        llm_client=None,
        verify_config: VerifyConfig = None,
    ):
        """
        Args:
            wiki_path: Wiki 因子库路径
            llm_client: LLM 客户端 (可选, 默认使用 LLMGateway)
            verify_config: 因子验证配置
        """
        self.wiki_path = wiki_path
        if llm_client is None:
            from QuantNodes.ai.llm.gateway import get_llm_gateway
            llm_client = get_llm_gateway()
        self.llm_client = llm_client
        self.proxy = WikiFactorProxy(wiki_path)
        self.evaluator = PolarsAlphaCalculatorEvaluator()
        self.verify_config = verify_config or VerifyConfig()

    def process(
        self,
        pdf_path: str,
        data: pl.DataFrame = None,
        store_to_wiki: bool = True,
    ) -> ReproductionReport:
        """处理单个研报 PDF

        Args:
            pdf_path: PDF 文件路径
            data: 行情数据 (用于因子验证, 可选)
            store_to_wiki: 是否存入 Wiki

        Returns:
            ReproductionReport
        """
        start = time.time()

        # 1. PDF 解析
        title, text = self.parse_pdf(pdf_path)

        # 2. 逻辑提取
        logics = self.extract_logic_from_text(text, title)

        # 3. 逐条验证
        results: List[ReproductionResult] = []
        for logic in logics:
            if data is not None and logic.logic_type == "factor" and logic.formula:
                result = self.verify_factor(logic, data)
            else:
                result = ReproductionResult(
                    logic=logic,
                    verification_status=(
                        "pending" if logic.logic_type != "factor" else "unverifiable"
                    ),
                    deviation=(
                        "非因子类型, 待人工验证"
                        if logic.logic_type != "factor"
                        else "缺少数据, 无法验证"
                    ),
                )
            results.append(result)

        # 4. 存入 Wiki
        if store_to_wiki:
            for result in results:
                self._store_to_wiki(result)

        # 5. 生成报告
        elapsed = time.time() - start
        report_md = self.generate_report(results, title)

        verified = sum(1 for r in results if r.verification_status == "verified")
        failed = sum(1 for r in results if r.verification_status == "failed")
        pending = sum(1 for r in results if r.verification_status == "pending")
        unverifiable = sum(1 for r in results if r.verification_status == "unverifiable")

        return ReproductionReport(
            pdf_path=pdf_path,
            title=title,
            total_logics=len(results),
            verified=verified,
            failed=failed,
            pending=pending,
            unverifiable=unverifiable,
            results=results,
            report_markdown=report_md,
            elapsed_seconds=elapsed,
        )

    # ==================== PDF 解析 ====================

    def parse_pdf(self, pdf_path: str) -> Tuple[str, str]:
        """解析 PDF, 返回 (title, text)"""
        try:
            from llmwikify.extractors import extract
            result = extract(pdf_path)
            return result.title or Path(pdf_path).stem, result.text
        except ImportError:
            return self._parse_pdf_fallback(pdf_path)
        except Exception:
            return self._parse_pdf_fallback(pdf_path)

    def _parse_pdf_fallback(self, pdf_path: str) -> Tuple[str, str]:
        """PDF 解析回退方案 (尝试 pymupdf)"""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            title = doc.metadata.get("title", "") or Path(pdf_path).stem
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return title, "\n---\n".join(text_parts)
        except ImportError:
            return Path(pdf_path).stem, "[PDF 解析不可用: 请安装 llmwikify 或 pymupdf]"
        except Exception as e:
            return Path(pdf_path).stem, f"[PDF 解析失败: {e}]"

    # ==================== 逻辑提取 ====================

    def extract_logic_from_text(
        self,
        text: str,
        title: str = "",
    ) -> List[ExtractedLogic]:
        """从文本提取逻辑"""
        if self.llm_client is not None:
            return self._llm_extract(text, title)
        return self._rule_based_extract(text)

    def _llm_extract(self, text: str, title: str) -> List[ExtractedLogic]:
        """LLM 逻辑提取"""
        try:
            from QuantNodes.ai.llm.base import Message

            # 截断防溢出
            max_chars = 8000
            truncated = text[:max_chars] if len(text) > max_chars else text

            prompt = EXTRACTION_PROMPT.format(title=title, text=truncated)
            response = self.llm_client.chat([
                Message(role="user", content=prompt)
            ])

            content = response.choices[0].message.content

            # 提取 JSON (处理可能的 markdown 代码块)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                content = json_match.group(1)
            content = content.strip()

            logics_data = json.loads(content)
            return [ExtractedLogic(**item) for item in logics_data]

        except Exception:
            return self._rule_based_extract(text)

    def _rule_based_extract(self, text: str) -> List[ExtractedLogic]:
        """基于规则的逻辑提取 (不需要 LLM)"""
        logics = []

        # 匹配因子公式模式
        formula_patterns = [
            (r'(?:公式|factor|因子)[：:]\s*(.+)', "factor"),
            (r'(\w+\s*/\s*\w+[\w\s/*+-]*?-\s*1)', "factor"),
            (r'(ts_\w+\([^)]+\))', "factor"),
            (r'(rank\([^)]+\))', "factor"),
        ]

        seen_formulas = set()
        for pattern, logic_type in formula_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                formula = match.strip()
                if formula in seen_formulas or len(formula) < 5:
                    continue
                seen_formulas.add(formula)
                logics.append(ExtractedLogic(
                    logic_type=logic_type,
                    title=f"提取因子: {formula[:50]}",
                    description=formula,
                    formula=formula,
                    evidence=match,
                    confidence=0.5,
                ))

        # 匹配交易规则模式
        rule_patterns = [
            r'(?:当|如果|若)(.+?)(?:时|买入|卖出|做多|做空)',
            r'(金叉|死叉|上穿|下穿)(.+?)(?:买入|卖出)',
        ]
        for pattern in rule_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                desc = match.strip() if isinstance(match, str) else " ".join(match)
                if len(desc) < 5:
                    continue
                logics.append(ExtractedLogic(
                    logic_type="rule",
                    title=f"交易规则: {desc[:50]}",
                    description=desc,
                    formula=None,
                    evidence=desc,
                    confidence=0.4,
                ))

        return logics

    # ==================== 因子验证 ====================

    def verify_factor(
        self,
        logic: ExtractedLogic,
        data: pl.DataFrame,
    ) -> ReproductionResult:
        """验证单条因子逻辑"""
        result = ReproductionResult(logic=logic)

        if not logic.formula:
            result.verification_status = "unverifiable"
            result.deviation = "缺少公式"
            return result

        # 构造 FactorSpec
        spec = FactorSpec(
            formula_id=f"report_{logic.title}",
            formula=logic.formula,
            source="research_report",
            category="other",
            meta={
                "description": logic.description,
                "template_name": "research_report",
            },
        )

        # 评估因子
        metrics_list = self.evaluator.evaluate([spec], data)
        if not metrics_list:
            result.verification_status = "failed"
            result.deviation = "评估失败"
            return result

        eval_result = metrics_list[0]

        # 6 维验证
        eval_result = self.evaluator.verify(
            eval_result,
            data,
            config=self.verify_config,
        )

        if eval_result.is_valid:
            result.verification_status = "verified"
            result.factor_result = eval_result
            result.deviation = f"IC={eval_result.ic_mean:.4f}, IR={eval_result.ir:.4f}"
        else:
            result.verification_status = "failed"
            result.factor_result = eval_result
            if eval_result.fail_reasons:
                result.deviation = "; ".join(eval_result.fail_reasons)
            else:
                result.deviation = "未通过验证"

        return result

    # ==================== Wiki 存储 ====================

    def _store_to_wiki(self, result: ReproductionResult):
        """存储到 Wiki"""
        if result.verification_status == "verified" and result.factor_result:
            self._store_verified_factor(result)
        else:
            self._store_pending_logic(result)

    def _store_verified_factor(self, result: ReproductionResult):
        """存储验证通过的因子"""
        factor = WikiFactor(
            name=result.logic.title,
            formula=result.logic.formula,
            source=FactorSource.RESEARCH_REPORT,
            category=FactorCategory.OTHER,
            tags=["research_report"],
            # WikiFactor V2: 主动填充新字段
            factor_params={
                "logic_type": result.logic.logic_type,
                "evidence": result.logic.evidence,
            },
            status="validated",
            ic_mean=result.factor_result.ic_mean,
            ic_std=result.factor_result.ic_std,
            icir=result.factor_result.ir,
            rank_ic_mean=result.factor_result.rank_ic_mean,
            turnover=result.factor_result.turnover,
            metadata={
                "source_evidence": result.logic.evidence,
                "confidence": result.logic.confidence,
                "logic_type": result.logic.logic_type,
                "stability_score": result.factor_result.stability_score,
                "diversification_score": result.factor_result.diversification_score,
                "monotonicity_score": result.factor_result.monotonicity_score,
                "coverage": result.factor_result.coverage,
                "overall_score": result.factor_result.overall_score,
            },
        )
        page_name = self.proxy.store_factor(factor)
        result.wiki_page_name = page_name

    def _store_pending_logic(self, result: ReproductionResult):
        """存储待验证的逻辑"""
        logic = WikiLogic(
            name=result.logic.title,
            content=result.logic.description,
            source=LogicSource.RESEARCH_REPORT,
            extracted_formula=result.logic.formula,
            validation_status="pending" if result.verification_status == "pending" else "rejected",
            metadata={
                "logic_type": result.logic.logic_type,
                "evidence": result.logic.evidence,
                "confidence": result.logic.confidence,
                "deviation": result.deviation,
            },
        )
        page_name = self.proxy.store_logic(logic)
        result.wiki_page_name = page_name

    # ==================== 报告生成 ====================

    def generate_report(
        self,
        results: List[ReproductionResult],
        title: str = "",
    ) -> str:
        """生成 Markdown 复现报告"""
        verified = sum(1 for r in results if r.verification_status == "verified")
        failed = sum(1 for r in results if r.verification_status == "failed")
        pending = sum(1 for r in results if r.verification_status == "pending")
        unverifiable = sum(1 for r in results if r.verification_status == "unverifiable")

        lines = [
            f"# 研报复现报告: {title}",
            "",
            f"**提取逻辑数**: {len(results)}",
            f"**验证通过**: {verified}",
            f"**验证失败**: {failed}",
            f"**待验证**: {pending}",
            f"**无法验证**: {unverifiable}",
            "",
        ]

        # 按状态分组展示
        status_icons = {
            "verified": "✅",
            "failed": "❌",
            "pending": "⏳",
            "unverifiable": "⚠️",
        }

        for i, r in enumerate(results, 1):
            icon = status_icons.get(r.verification_status, "?")
            lines.append(f"### {i}. {r.logic.title} {icon}")
            lines.append(f"- **类型**: {r.logic.logic_type}")
            if r.logic.formula:
                lines.append(f"- **公式**: `{r.logic.formula}`")
            lines.append(f"- **描述**: {r.logic.description}")
            lines.append(f"- **原文依据**: {r.logic.evidence}")
            lines.append(f"- **置信度**: {r.logic.confidence:.2f}")
            lines.append(f"- **验证状态**: {r.verification_status}")
            if r.deviation:
                lines.append(f"- **详情**: {r.deviation}")
            if r.factor_result:
                lines.append(f"- **IC Mean**: {r.factor_result.ic_mean:.4f}")
                lines.append(f"- **IR**: {r.factor_result.ir:.4f}")
                lines.append(f"- **稳定性**: {r.factor_result.stability_score:.4f}")
                lines.append(f"- **综合分数**: {r.factor_result.overall_score:.4f}")
            lines.append("")

        return "\n".join(lines)
