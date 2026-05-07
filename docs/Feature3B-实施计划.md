# 功能3B 实施方案：研报复现 (Research Report Reproduction)

> 文档版本: v1.0
> 创建日期: 2026-05-07
> 状态: ✅ 已完成

---

## 一、背景

功能3B 是量化研究的知识提取层，从研报PDF中提取因子公式、交易规则等量化逻辑，验证其有效性，存入 Wiki 因子库。

**依赖链**：
```
llmwikify (PDF解析) + OpenAIClient (LLM)
    ↓
ResearchReportReproducer (编排器)
    ├── FactorEvaluator (功能3C复用) → 因子验证
    └── WikiFactorProxy (功能3A复用) → Wiki存储
```

**设计原则**：
- LLM 提取 + 规则匹配混合策略
- 因子类逻辑自动验证，非因子类存入 WikiLogic 待人工验证
- 复用 3A/3C 已有能力，最小化新代码

---

## 二、文件结构

```
QuantNodes/research/
├── __init__.py                  # 更新导出
├── wiki.py                      # 功能3A (已有)
├── factor_miner.py              # 功能3C (已有)
├── factor_evaluator.py          # 功能3C (已有)
├── mcts_search.py               # 功能3C (已有)
├── auto_researcher.py           # 功能3C (已有)
├── report_reproducer.py         # 功能3B: 研报复现编排器
└── README.md                    # 更新
```

---

## 三、数据模型

### 3.1 ExtractedLogic — 从研报提取的逻辑

```python
@dataclass
class ExtractedLogic:
    """从研报提取的逻辑"""
    logic_type: str           # factor | rule | condition | combination
    title: str                # 逻辑标题
    description: str          # 原文描述
    formula: Optional[str]    # 因子公式 (factor 类型必填)
    evidence: str             # 原文依据 (页码/段落)
    confidence: float         # LLM 提取置信度 (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.2 ReproductionResult — 单条逻辑的复现结果

```python
@dataclass
class ReproductionResult:
    """单条逻辑的复现结果"""
    logic: ExtractedLogic
    verification_status: str  # verified | failed | pending | unverifiable
    factor_result: Optional[FactorEvaluationResult] = None  # 因子验证结果
    deviation: str = ""       # 与原文声称的偏差描述
    wiki_page_name: Optional[str] = None
```

### 3.3 ReproductionReport — 研报复现报告

```python
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
```

---

## 四、核心接口

### 4.1 ResearchReportReproducer

```python
class ResearchReportReproducer:
    """研报复现系统"""

    def __init__(self, wiki_path: str, llm_client=None):
        """
        Args:
            wiki_path: Wiki 因子库路径
            llm_client: OpenAIClient 实例 (可选, 不传则跳过LLM提取, 使用规则匹配)
        """

    def process(
        self,
        pdf_path: str,
        data: pl.DataFrame = None,
        store_to_wiki: bool = True,
    ) -> ReproductionReport:
        """处理单个研报 PDF

        流程:
        1. PDF 解析 (llmwikify.extract)
        2. LLM 逻辑提取 (可选, 需 llm_client)
        3. 因子验证 (FactorEvaluator, 需 data)
        4. 存入 Wiki
        5. 生成报告
        """

    def extract_logic_from_text(
        self,
        text: str,
        title: str = "",
    ) -> List[ExtractedLogic]:
        """从文本提取逻辑 (调用 LLM)"""

    def verify_factor(
        self,
        logic: ExtractedLogic,
        data: pl.DataFrame,
    ) -> ReproductionResult:
        """验证单条因子逻辑"""

    def generate_report(
        self,
        results: List[ReproductionResult],
        title: str = "",
    ) -> str:
        """生成 Markdown 复现报告"""
```

---

## 五、PDF 解析

### 5.1 使用 llmwikify.extract()

```python
from llmwikify.extractors import extract

def parse_pdf(self, pdf_path: str) -> Tuple[str, str]:
    """解析 PDF, 返回 (title, text)"""
    result = extract(pdf_path)
    return result.title, result.text
```

### 5.2 llmwikify 提取能力

| 能力 | 说明 |
|------|------|
| 文本提取 | MarkItDown (优先) 或 pymupdf (回退) |
| OCR 支持 | MarkItDown + markitdown-ocr 插件 |
| 表格提取 | MarkItDown 自动处理 |
| 公式提取 | 文本形式保留 |

---

## 六、LLM 逻辑提取

### 6.1 Prompt 设计

```python
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
```

### 6.2 LLM 调用

```python
def extract_logic_from_text(self, text: str, title: str = "") -> List[ExtractedLogic]:
    """从文本提取逻辑"""
    if self.llm_client is None:
        return self._rule_based_extract(text)

    prompt = EXTRACTION_PROMPT.format(title=title, text=text[:8000])  # 截断防溢出
    response = self.llm_client.chat([
        Message(role="user", content=prompt)
    ])

    # 解析 JSON
    content = response.choices[0].message.content
    logics_data = json.loads(content)

    return [ExtractedLogic(**item) for item in logics_data]
```

### 6.3 规则匹配回退 (无 LLM 时)

```python
def _rule_based_extract(self, text: str) -> List[ExtractedLogic]:
    """基于规则的逻辑提取 (不需要 LLM)"""
    logics = []

    # 正则匹配因子公式模式
    import re
    formula_patterns = [
        r'公式[：:]\s*(.+)',
        r'factor[：:]\s*(.+)',
        r'(\w+\s*/\s*\w+.*?-\s*1)',
    ]
    for pattern in formula_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            logics.append(ExtractedLogic(
                logic_type="factor",
                title=f"提取因子: {match[:30]}",
                description=match,
                formula=match.strip(),
                evidence=match,
                confidence=0.5,
            ))

    return logics
```

---

## 七、因子验证

### 7.1 验证流程

```python
def verify_factor(self, logic: ExtractedLogic, data: pl.DataFrame) -> ReproductionResult:
    """验证单条因子逻辑"""
    result = ReproductionResult(logic=logic)

    if logic.logic_type != "factor" or not logic.formula:
        result.verification_status = "unverifiable"
        result.deviation = "非因子类型或缺少公式, 需人工验证"
        return result

    # 使用 FactorEvaluator 验证
    candidate = FactorCandidate(
        name=f"report_{logic.title}",
        formula=logic.formula,
        description=logic.description,
        operators_used=[],
        category=FactorCategory.OTHER,
        template_name="research_report",
    )

    eval_result = self.evaluator.evaluate(candidate, data)

    if eval_result.is_valid:
        result.verification_status = "verified"
        result.factor_result = eval_result
        result.deviation = f"IC={eval_result.ic_mean:.4f}, ICIR={eval_result.icir:.4f}"
    else:
        result.verification_status = "failed"
        result.factor_result = eval_result
        result.deviation = "; ".join(eval_result.fail_reasons)

    return result
```

### 7.2 验证状态定义

| 状态 | 说明 |
|------|------|
| `verified` | 因子通过6维度验证 |
| `failed` | 因子未通过验证 (附原因) |
| `pending` | 非因子类型, 待人工验证 |
| `unverifiable` | 无法验证 (缺少公式/数据) |

---

## 八、Wiki 存储

### 8.1 因子类逻辑 → WikiFactor

```python
def _store_verified_factor(self, result: ReproductionResult):
    """存储验证通过的因子"""
    factor = WikiFactor(
        name=result.logic.title,
        formula=result.logic.formula,
        source=FactorSource.RESEARCH_REPORT,
        category=FactorCategory.OTHER,
        tags=["research_report"],
        ic_mean=result.factor_result.ic_mean,
        ic_std=result.factor_result.ic_std,
        icir=result.factor_result.icir,
        rank_ic_mean=result.factor_result.rank_ic_mean,
        turnover=result.factor_result.turnover,
        metadata={
            "source_pdf": result.logic.evidence,
            "confidence": result.logic.confidence,
        },
    )
    self.proxy.store_factor(factor)
```

### 8.2 非因子类逻辑 → WikiLogic

```python
def _store_pending_logic(self, result: ReproductionResult):
    """存储待验证的逻辑"""
    logic = WikiLogic(
        name=result.logic.title,
        content=result.logic.description,
        source=LogicSource.RESEARCH_REPORT,
        extracted_formula=result.logic.formula,
        validation_status="pending",
        metadata={
            "logic_type": result.logic.logic_type,
            "evidence": result.logic.evidence,
            "confidence": result.logic.confidence,
        },
    )
    self.proxy.store_logic(logic)
```

---

## 九、报告生成

### 9.1 Markdown 报告模板

```markdown
# 研报复现报告: {title}

**PDF**: {pdf_path}
**耗时**: {elapsed}s
**提取逻辑数**: {total}
**验证通过**: {verified}
**验证失败**: {failed}
**待验证**: {pending}

## 提取的逻辑

### 1. {logic.title} ✅
- **类型**: factor
- **公式**: `{logic.formula}`
- **原文依据**: {logic.evidence}
- **验证结果**: IC={ic_mean:.4f}, ICIR={icir:.4f}
- **状态**: verified

### 2. {logic.title} ❌
- **类型**: rule
- **描述**: {logic.description}
- **验证结果**: 非因子类型, 待人工验证
- **状态**: pending
```

---

## 十、实施步骤

| Step | 任务 | 文件 | 预估行数 |
|------|------|------|----------|
| 1 | 数据模型 (ExtractedLogic 等) | `report_reproducer.py` | ~60 |
| 2 | PDF 解析封装 (llmwikify 集成) | `report_reproducer.py` | ~40 |
| 3 | LLM 逻辑提取 (Prompt + 调用) | `report_reproducer.py` | ~100 |
| 4 | 规则匹配回退 | `report_reproducer.py` | ~50 |
| 5 | 因子验证集成 (复用 FactorEvaluator) | `report_reproducer.py` | ~60 |
| 6 | Wiki 存储集成 (复用 WikiFactorProxy) | `report_reproducer.py` | ~50 |
| 7 | 报告生成 (Markdown) | `report_reproducer.py` | ~80 |
| 8 | ResearchReportReproducer 编排器 | `report_reproducer.py` | ~80 |
| 9 | 更新 `__init__.py` | `__init__.py` | ~10 |
| 10 | 单元测试 | `tests/research/test_report_reproducer.py` | ~200 |
| 11 | 更新设计文档 + README | `docs/` + `README.md` | ~50 |

**总计**: ~780 行代码

---

## 十一、测试策略

### 11.1 Mock 策略

- **Mock LLM**: 不依赖真实 API, 返回预定义 JSON
- **Mock PDF**: 使用字符串模拟 PDF 文本, 不依赖真实 PDF 文件
- **Mock FactorEvaluator**: 可选, 验证集成逻辑

### 11.2 测试覆盖

| 测试 | 说明 |
|------|------|
| `test_parse_pdf` | PDF 文本提取 |
| `test_extract_logic_llm` | LLM 逻辑提取 (mock) |
| `test_extract_logic_rule` | 规则匹配回退 |
| `test_verify_factor_valid` | 因子验证通过 |
| `test_verify_factor_invalid` | 因子验证失败 |
| `test_verify_non_factor` | 非因子类型处理 |
| `test_store_to_wiki` | Wiki 存储集成 |
| `test_generate_report` | 报告生成 |
| `test_process_e2e` | 端到端流程 |

---

## 十二、与现有模块的复用关系

| 现有模块 | 复用方式 | 位置 |
|----------|----------|------|
| `llmwikify.extract()` | PDF 文本提取 | 外部依赖 |
| `ai.llm.openai.OpenAIClient` | LLM 调用 | `QuantNodes/ai/llm/openai.py` |
| `FactorEvaluator` | 因子 IC/IR 验证 | `research/factor_evaluator.py` |
| `FactorCandidate` | 候选因子数据 | `research/factor_miner.py` |
| `WikiFactorProxy` | Wiki 读写 | `research/wiki.py` |
| `WikiFactor` | 因子数据模型 | `research/wiki.py` |
| `WikiLogic` | 逻辑数据模型 | `research/wiki.py` |
| `FactorSource.RESEARCH_REPORT` | 来源标记 | `research/wiki.py` |
| `LogicSource.RESEARCH_REPORT` | 来源标记 | `research/wiki.py` |

---

**文档版本**: v1.0
**最后更新**: 2026-05-07
