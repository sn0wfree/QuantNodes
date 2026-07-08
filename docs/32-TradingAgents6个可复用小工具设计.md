# TradingAgents 范式 → 可复用小工具 (6 Tools Design)

> **编号**: 32
> **状态**: ✅ 设计完成
> **依赖**: docs/28 (调研) + docs/29 (集成计划) + docs/30 (讨论) + docs/31 (深度梳理)
> **日期**: 2026-06-24
> **原则**: 把 TradingAgents 4-Phase Pipeline 拆成 6 个独立小工具

---

## 一、设计原则

| 原则 | 说明 |
|------|------|
| **细粒度** | 每个工具单一职责，可独立调用 |
| **可组合** | 工具之间可自由组合（用户拼装工作流） |
| **结构化** | 输入输出全部 Pydantic |
| **可配置** | 提示词/角色支持自定义，但有默认值 |
| **双入口** | Python API + Agent tool (LLM 可直接调用) |
| **无状态** | 不依赖 LangGraph，session state 可选 |
| **可降级** | LLM 失败时降级为 regex parser |

---

## 二、6 个工具总览

| # | 工具 | Phase | 核心职责 | 依赖 |
|---|------|-------|---------|------|
| 1 | `structured_judge` | **Phase 1** ⭐ | 文本 → Pydantic 结构化裁决 | LLMClientBase |
| 2 | `reflect` | Phase 1 | 决策 + 结果 → 反思 | LLMClientBase |
| 3 | `market_sanity_check` | Phase 1 | ticker+date → 确定性数据快照 | 因子引擎 |
| 4 | `multi_perspective_review` | Phase 2 | 标的 → 多视角独立报告 | LLMClientBase + structured_judge |
| 5 | `debate` | Phase 2 | 议题 → N 角色辩论 + 裁决 | LLMClientBase + multi_perspective_review + structured_judge |
| 6 | `kb_recall` | Phase 3 | 查询 → 历史经验召回 | KnowledgeBase |

**Phase 1** = 基础工具（独立可用）
**Phase 2** = 辩论工具（依赖 Phase 1）
**Phase 3** = 知识闭环（依赖 KnowledgeBase）

---

## 三、文件结构

```
QuantNodes/ai/llm_decision/                          # 纯函数层（无 nanobot 依赖）
  __init__.py
  schemas.py                                         # 5 个 Pydantic schemas
  prompts.py                                         # Prompt 模板（视角 + 角色）
  parser.py                                          # Rating regex parser

QuantNodes/agent/tools/llm_decision/                 # Tool 层（暴露给 LLM agent）
  __init__.py                                        # 注册到 _QUANT_TOOL_FACTORIES
  structured_judge.py                                # Phase 1 ⭐
  reflect.py                                         # Phase 1
  market_sanity_check.py                             # Phase 1
  multi_perspective_review.py                        # Phase 2
  debate.py                                          # Phase 2
  kb_recall.py                                       # Phase 3

QuantNodes/agent/skills/llm_decision/                 # Skill 层（可组合工作流）
  __init__.py
  strategy_review.py                                 # 策略评审（Phase 3）
  factor_verdict.py                                  # 因子过滤（Phase 3）
  reflection_loop.py                                 # 反思闭环（Phase 3）

tests/test_llm_decision/
  test_structured_judge.py
  test_reflect.py
  test_market_sanity_check.py
  test_multi_perspective_review.py
  test_debate.py
  test_kb_recall.py
  test_strategy_review_skill.py
  test_factor_verdict_skill.py
  test_reflection_loop_skill.py
```

---

## 四、Phase 1: 基础工具（3 个）

### 4.1 `structured_judge` ⭐ Phase 1 优先

**职责**: 把任意分析文本解析为 Pydantic 结构化裁决

#### 输入
```python
class StructuredJudgeInput(BaseModel):
    text: str                                         # 待解析的分析文本
    schema: str                                       # schema name (PortfolioDecision/FactorVerdict/StrategyVerdict/ReflectionNote)
    context: Optional[Dict[str, Any]] = None         # 额外上下文
    fallback: bool = True                             # 是否 graceful fallback to regex
```

#### 输出
```python
class StructuredJudgeOutput(BaseModel):
    parsed: BaseModel                                 # 解析后的 Pydantic 实例
    confidence: float                                 # 0-1
    method: Literal["structured", "regex_fallback", "failed"]
    raw_text: str                                     # 原始文本
```

#### Schema 定义 (`QuantNodes/ai/llm_decision/schemas.py`)

```python
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class PortfolioRating(str, Enum):
    """5-tier rating scale (从 TradingAgents 借鉴)"""
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"

class FactorVerdict(BaseModel):
    """因子质量评估"""
    accept: bool
    confidence: float = Field(ge=0, le=1, default=0.5)
    rationale: str
    concerns: List[str] = []

class StrategyVerdict(BaseModel):
    """策略上盘评审"""
    rating: PortfolioRating
    suggested_position_pct: float = Field(ge=0, le=100, default=0)
    rationale: str
    risk_factors: List[str] = []

class PortfolioDecision(BaseModel):
    """完整投资决策"""
    rating: PortfolioRating
    executive_summary: str
    investment_thesis: str
    price_target: Optional[float] = None
    time_horizon: Optional[str] = None

class ReflectionNote(BaseModel):
    """事后反思"""
    what_worked: str
    what_failed: str
    lesson: str
    confidence: float = Field(ge=0, le=1, default=0.5)

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "PortfolioRating": PortfolioRating,
    "FactorVerdict": FactorVerdict,
    "StrategyVerdict": StrategyVerdict,
    "PortfolioDecision": PortfolioDecision,
    "ReflectionNote": ReflectionNote,
}
```

#### 实现要点

```python
class StructuredJudge:
    """Text → Pydantic instance (复用 TradingAgents 的 graceful fallback 模式)"""

    def __init__(self, llm_client: Optional[LLMClientBase] = None):
        self.llm = llm_client

    async def judge(self, text: str, schema_name: str, context: Optional[Dict] = None) -> StructuredJudgeOutput:
        schema_cls = SCHEMA_REGISTRY[schema_name]

        # 路径 1: LLM with_structured_output (优先)
        if self.llm:
            try:
                structured_llm = self.llm.with_structured_output(schema_cls)
                result = structured_llm.invoke(self._build_prompt(text, context))
                return StructuredJudgeOutput(
                    parsed=result,
                    confidence=0.95,
                    method="structured",
                    raw_text=text,
                )
            except Exception as e:
                if not fallback:
                    raise
                # Fall through to regex

        # 路径 2: Regex parser (降级)
        result = self._regex_parse(text, schema_cls)
        return StructuredJudgeOutput(
            parsed=result,
            confidence=result.get("confidence", 0.5),
            method="regex_fallback",
            raw_text=text,
        )

    def _regex_parse(self, text: str, schema_cls: Type[BaseModel]) -> BaseModel:
        """无 LLM 时的纯 regex 解析（仅适用于简单 schema）"""
        if schema_cls == PortfolioRating:
            rating = parse_rating(text)  # 复用 TA 的两阶段策略
            return PortfolioRating(rating)
        # ... 其他 schema 的 regex
```

#### Rating Parser (`QuantNodes/ai/llm_decision/parser.py`)

```python
import re

_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)
_RATING_SET = {"buy", "overweight", "hold", "underweight", "sell"}

def parse_rating(text: str) -> str:
    """从文本中提取 5-tier rating (Pass 1: 显式标签; Pass 2: 首次出现; 默认: Hold)"""
    # Pass 1: explicit label
    match = _RATING_LABEL_RE.search(text)
    if match:
        word = match.group(1).lower().strip("*:.,")
        if word in _RATING_SET:
            return word.capitalize()

    # Pass 2: first occurrence
    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return "Hold"
```

#### 验收标准

```python
# Test 1: 独立调用（无 LLM, regex fallback）
result = await StructuredJudge().judge(
    text="基于市场分析，我建议 Buy AAPL",
    schema_name="PortfolioRating"
)
assert result.parsed == PortfolioRating.BUY
assert result.method == "regex_fallback"

# Test 2: 有 LLM 时使用 structured output
result = await StructuredJudge(llm_client=mock_llm).judge(
    text="建议买入 AAPL，目标价 200",
    schema_name="PortfolioDecision"
)
assert result.parsed.rating == PortfolioRating.BUY
assert result.method == "structured"

# Test 3: 通过 Agent 调用
agent.chat("用 structured_judge 工具解析: ...")
```

---

### 4.2 `reflect` (Phase 1)

**职责**: 给定决策 + 实际结果，生成 2-4 句反思

#### 输入
```python
class ReflectInput(BaseModel):
    decision: Dict[str, Any]                           # 决策内容 (action/ticker/reason)
    actual_return: float                               # 实际收益 (如 0.05)
    alpha: float                                       # 超额收益 (如 0.03)
    benchmark: str = "SPY"                             # 基准
    context: Optional[Dict[str, Any]] = None          # 额外上下文（持仓期等）
```

#### 输出
```python
class ReflectOutput(BaseModel):
    note: ReflectionNote                               # Pydantic 反思
    raw_text: str                                      # 原始反思文本
    method: Literal["structured", "free_text"]
```

#### 实现

```python
class Reflector:
    """Decision + outcome → Reflection (借鉴 TA 的 Reflector 模式)"""

    REFLECTION_PROMPT = """You are a trading analyst reviewing your own past decision.
Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).

Cover in order:
1. Was the directional call correct? (cite the alpha figure)
2. Which part of the investment thesis held or failed?
3. One concrete lesson to apply to the next similar analysis.

Be specific and terse. Output will be stored verbatim and re-read by future analysts."""

    async def reflect(self, input: ReflectInput) -> ReflectOutput:
        # Step 1: Generate reflection text via LLM
        prompt = f"""{self.REFLECTION_PROMPT}

Raw return: {input.actual_return:+.1%}
Alpha vs {input.benchmark}: {input.alpha:+.1%}

Decision:
{json.dumps(input.decision, indent=2)}

{self._format_context(input.context)}
"""
        response = self.llm.chat(prompt)
        text = response.content

        # Step 2: Parse via structured_judge (graceful fallback to free text)
        judge = StructuredJudge(self.llm)
        result = await judge.judge(text=text, schema_name="ReflectionNote")
        return ReflectOutput(note=result.parsed, raw_text=text, method=result.method)
```

#### 验收标准

```python
note = await Reflector(llm_client=mock_llm).reflect(
    decision={"action": "Buy", "ticker": "AAPL", "reason": "动量策略"},
    actual_return=0.052, alpha=0.031
)
assert isinstance(note.note, ReflectionNote)
assert note.note.lesson  # 非空
```

---

### 4.3 `market_sanity_check` (Phase 1)

**职责**: 给定 ticker + date，返回确定性计算的 OHLCV + 技术指标快照

#### 输入
```python
class MarketSanityCheckInput(BaseModel):
    ticker: str
    date: str                                          # YYYY-MM-DD
    indicators: List[str] = ["close_50_sma", "close_10_ema", "rsi", "macd", "boll_ub", "boll_lb", "atr"]
    look_back_days: int = 30
```

#### 输出
```python
class MarketSanityCheckOutput(BaseModel):
    snapshot_markdown: str                             # 完整 markdown 快照
    latest_close: float
    latest_volume: int
    computed_indicators: Dict[str, Optional[float]]   # 指标名 → 值
    source: str                                        # "quantnodes_factor_engine"
```

#### 实现（复用 QuantNodes 因子引擎）

```python
from QuantNodes.factor_node.factor_functions import get_operator

class MarketSanityChecker:
    """OHLCV + 指标快照 (复用 QuantNodes 因子引擎，不重实现)"""

    DEFAULT_INDICATORS = [
        "close_50_sma", "close_10_ema", "rsi",
        "macd", "boll_ub", "boll_lb", "atr"
    ]

    def __init__(self, data_loader):
        self.data_loader = data_loader  # 注入数据源

    def check(self, input: MarketSanityCheckInput) -> MarketSanityCheckOutput:
        # Step 1: 加载 OHLCV (复用 QuantNodes 数据加载)
        ohlcv = self.data_loader.load(input.ticker, input.date, look_back=input.look_back_days)

        # Step 2: 计算指标 (复用 QuantNodes 因子算子)
        computed = {}
        for ind_name in input.indicators:
            op = get_operator(ind_name)
            computed[ind_name] = op.compute(ohlcv).iloc[-1]

        # Step 3: 格式化为 markdown (带防幻觉指令)
        snapshot = self._format_snapshot(input.ticker, input.date, ohlcv, computed)

        return MarketSanityCheckOutput(
            snapshot_markdown=snapshot,
            latest_close=ohlcv["close"].iloc[-1],
            latest_volume=int(ohlcv["volume"].iloc[-1]),
            computed_indicators=computed,
            source="quantnodes_factor_engine",
        )

    def _format_snapshot(self, ticker, date, ohlcv, indicators) -> str:
        latest = ohlcv.iloc[-1]
        rows = "\n".join(f"| {name} | {val:.2f} |" for name, val in indicators.items())

        snapshot = f"""## Verified Market Snapshot: {ticker} as of {date}

### Latest OHLCV
| Open | High | Low | Close | Volume |
|------|------|-----|-------|--------|
| {latest['open']:.2f} | {latest['high']:.2f} | {latest['low']:.2f} | {latest['close']:.2f} | {int(latest['volume']):,} |

### Verified Technical Indicators
{rows}

---
Use this snapshot as the source of truth for exact OHLCV, price-level, and indicator-value claims.
If conflicting with other tool output, flag the discrepancy. Do not invent numbers.
"""
        return snapshot
```

#### 验收标准

```python
snapshot = MarketSanityChecker(data_loader).check(
    MarketSanityCheckInput(ticker="AAPL", date="2024-05-10")
)
assert "EMA" in snapshot.snapshot_markdown or "SMA" in snapshot.snapshot_markdown
assert "Use this snapshot as the source of truth" in snapshot.snapshot_markdown
assert snapshot.computed_indicators["rsi"] is not None
```

---

## 五、Phase 2: 辩论工具（2 个）

### 5.1 `multi_perspective_review`

**职责**: 让 N 个独立视角评估同一标的，返回多份报告

#### 输入
```python
class MultiPerspectiveInput(BaseModel):
    subject: str                                       # 标的 (因子名/策略名/决策)
    subject_type: Literal["factor", "strategy", "decision"]
    context: Dict[str, Any]                            # 评估上下文 (IC/Sharpe/etc.)
    perspectives: List[str] = ["technical", "fundamental", "risk", "regime"]
    custom_prompts: Optional[Dict[str, str]] = None   # 自定义每个视角的提示词
```

#### 输出
```python
class PerspectiveReport(BaseModel):
    name: str                                          # "technical"
    report: str                                        # 评估文本
    score: Optional[float] = None                      # 0-1 (如果 perspective 输出评分)

class MultiPerspectiveOutput(BaseModel):
    reports: Dict[str, PerspectiveReport]
    aggregate_score: Optional[float] = None
```

#### 默认 Perspectives (`QuantNodes/ai/llm_decision/prompts.py`)

```python
DEFAULT_PERSPECTIVES: Dict[str, str] = {
    "technical": "Evaluate the technical merit and statistical significance. Focus on: IC/IR, decay, turnover, regime dependence.",
    "fundamental": "Evaluate the economic rationale. Focus on: Why does this work? What's the behavioral/structural driver?",
    "risk": "Evaluate downside risk and tail scenarios. Focus on: max drawdown, volatility clustering, correlation with other factors.",
    "regime": "Evaluate market regime dependence. Focus on: bull vs bear, high vs low vol, performance across cycles.",
    "liquidity": "Evaluate liquidity and tradability. Focus on: impact cost, capacity, slippage.",
}
```

#### 实现

```python
class MultiPerspectiveReviewer:
    def __init__(self, llm_client, perspectives: Optional[Dict[str, str]] = None):
        self.llm = llm_client
        self.perspectives = perspectives or DEFAULT_PERSPECTIVES

    async def review(self, input: MultiPerspectiveInput) -> MultiPerspectiveOutput:
        prompts = input.custom_prompts or self.perspectives
        reports = {}

        # 并行调用 N 个视角
        tasks = {
            name: self._review_one(name, focus, input)
            for name, focus in prompts.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                reports[name] = PerspectiveReport(name=name, report=f"ERROR: {result}")
            else:
                reports[name] = result

        # Aggregate score (average of scores)
        scores = [r.score for r in reports.values() if r.score is not None]
        aggregate = sum(scores) / len(scores) if scores else None

        return MultiPerspectiveOutput(reports=reports, aggregate_score=aggregate)

    async def _review_one(self, name: str, focus: str, input: MultiPerspectiveInput) -> PerspectiveReport:
        prompt = f"""You are evaluating a {input.subject_type} from the {name} perspective.

Focus: {focus}

Subject: {input.subject}

Context:
{json.dumps(input.context, indent=2)}

Provide:
1. A 3-5 sentence evaluation
2. A score from 0 to 1 (higher = more positive)

Format your response as:
<evaluation>...</evaluation>
<score>0.XX</score>"""
        response = self.llm.chat(prompt)
        return self._parse_perspective_response(name, response.content)
```

---

### 5.2 `debate`

**职责**: N 个角色就同一议题辩论 R 轮，返回辩论历史 + 最终裁决

#### 输入
```python
class DebateInput(BaseModel):
    question: str                                      # 议题
    positions: List[str]                               # ["bull", "bear"] 或 ["aggressive", "conservative", "neutral"]
    context: Dict[str, Any]                            # 上下文（reports, data, multi_perspective result）
    rounds: int = 1                                    # 1-5
    judge_schema: str = "PortfolioDecision"            # 最终裁决的 schema
```

#### 输出
```python
class DebateTurn(BaseModel):
    speaker: str
    text: str

class DebateOutput(BaseModel):
    history: str                                       # 完整辩论记录
    turns: List[DebateTurn]
    judge_decision: BaseModel                          # 裁决（按 judge_schema）
    judge_method: Literal["structured", "regex_fallback"]
```

#### 默认角色 Prompts (`prompts.py`)

```python
ROLE_PROMPTS: Dict[str, str] = {
    "bull": """You are a Bull Analyst advocating for investing in {subject}.
Build a strong, evidence-based case emphasizing:
- Growth potential
- Competitive advantages
- Positive indicators

Resources: {context}
Last opponent argument: {last_argument}

Engage directly with the opponent's points and debate effectively.""",

    "bear": """You are a Bear Analyst making the case against investing in {subject}.
Present a well-reasoned argument emphasizing:
- Risks and challenges
- Competitive weaknesses
- Negative indicators

Resources: {context}
Last opponent argument: {last_argument}

Engage directly with the opponent's points and debate effectively.""",

    "aggressive": """You are an Aggressive Risk Analyst championing high-reward opportunities.
Focus on: upside, growth potential, competitive advantages.
Counter cautious views with data.""",

    "conservative": """You are a Conservative Risk Analyst protecting assets and minimizing volatility.
Focus on: stability, security, risk mitigation, capital preservation.""",

    "neutral": """You are a Neutral Risk Analyst providing balanced perspective.
Focus on: weighing both benefits and risks, broader market trends, diversification.""",
}
```

#### 路由逻辑

```python
def _next_speaker(debate_state: DebateState) -> str:
    """轮次管理: 简单字符串前缀路由 (复用 TA 模式)"""
    if not debate_state.turns:
        return debate_state.positions[0]  # 第一个角色

    last_speaker = debate_state.turns[-1].speaker
    current_idx = debate_state.positions.index(last_speaker)
    next_idx = (current_idx + 1) % len(debate_state.positions)
    return debate_state.positions[next_idx]
```

#### 实现

```python
class DebateEngine:
    """多角色对抗辩论 (借鉴 TradingAgents 的 state machine)"""

    def __init__(self, llm_client, judge: StructuredJudge):
        self.llm = llm_client
        self.judge = judge

    async def debate(self, input: DebateInput) -> DebateOutput:
        history = ""
        turns = []
        position_count = len(input.positions)
        max_turns = input.rounds * position_count  # 每轮每个角色发言一次

        for turn_idx in range(max_turns):
            speaker = input.positions[turn_idx % position_count]
            last_arg = turns[-1].text if turns else ""

            # Step 1: 生成发言
            prompt = ROLE_PROMPTS[speaker].format(
                subject=input.question,
                context=json.dumps(input.context, indent=2),
                last_argument=last_arg,
            )
            response = self.llm.chat(prompt)
            text = response.content

            # Step 2: 更新状态
            turns.append(DebateTurn(speaker=speaker, text=text))
            history += f"\n{speaker}: {text}"

        # Step 3: 最终裁决 (复用 structured_judge)
        judge_result = await self.judge.judge(
            text=history,
            schema_name=input.judge_schema,
            context={"question": input.question},
        )

        return DebateOutput(
            history=history,
            turns=turns,
            judge_decision=judge_result.parsed,
            judge_method=judge_result.method,
        )
```

#### 验收标准

```python
debate = DebateEngine(llm_client=mock_llm, judge=StructuredJudge(mock_llm))
result = await debate.debate(DebateInput(
    question="momentum_20d 因子是否值得用于实盘?",
    positions=["bull", "bear"],
    context={"ic": 0.05, "sharpe": 1.2},
    rounds=1,
    judge_schema="FactorVerdict",
))
assert len(result.turns) == 2  # 1 轮 × 2 角色
assert isinstance(result.judge_decision, FactorVerdict)
```

---

## 六、Phase 3: 知识闭环（1 个工具 + 3 个 Skill）

### 6.1 `kb_recall`

**职责**: 从 KnowledgeBase 召回历史经验

#### 输入
```python
class KBRecallInput(BaseModel):
    query: str                                         # 查询文本
    top_k: int = 3
    ticker_filter: Optional[str] = None               # 限定 ticker
    source_filter: Optional[List[str]] = None         # ["reflection", "experiment", "wiki"]
    min_score: float = 0.3
```

#### 输出
```python
class KBRecallHit(BaseModel):
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = {}

class KBRecallOutput(BaseModel):
    results: List[KBRecallHit]
    query: str
```

#### 实现（复用 QuantNodes KnowledgeBase）

```python
from QuantNodes.core.knowledge import KnowledgeBase

class KBRecall:
    """历史经验召回 (复用 KnowledgeBase)"""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def recall(self, input: KBRecallInput) -> KBRecallOutput:
        # Step 1: KnowledgeBase query (TF-IDF)
        hits = self.kb.query(input.query, top_k=input.top_k * 2)  # 多取一些过滤

        results = []
        for entry, score in hits:
            # 应用 filters
            if input.ticker_filter and entry.ticker != input.ticker_filter:
                continue
            if input.source_filter and entry.source_type not in input.source_filter:
                continue
            if score < input.min_score:
                continue

            results.append(KBRecallHit(
                text=entry.text,
                score=score,
                source=entry.source_type,
                metadata={"entry_id": entry.id, "created_at": entry.created_at},
            ))

            if len(results) >= input.top_k:
                break

        return KBRecallOutput(results=results, query=input.query)
```

---

### 6.2 Skill: `strategy_review`

**职责**: 自动组合 5 个工具做完整策略评审

```python
class StrategyReviewSkill(Skill):
    name = "strategy_review"
    description = "策略上盘评审: 数据验证 → 多视角 → 辩论 → 裁决"

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        strategy = context["strategy"]
        ticker = context["ticker"]
        date = context["date"]

        # Step 1: Market sanity check (防幻觉)
        sanity = await market_sanity_check(ticker, date)

        # Step 2: Multi-perspective review (4 视角)
        reviews = await multi_perspective_review(
            subject=strategy["name"],
            subject_type="strategy",
            context={**strategy, "snapshot": sanity.snapshot_markdown},
            perspectives=["technical", "risk", "regime", "liquidity"],
        )

        # Step 3: Debate (Bull vs Bear)
        debate = await debate(
            question=f"策略 {strategy['name']} 是否上盘?",
            positions=["bull", "bear"],
            context={"reviews": reviews.reports, "snapshot": sanity.snapshot_markdown},
            rounds=1,
            judge_schema="StrategyVerdict",
        )

        # Step 4: Return structured verdict
        return SkillResult(success=True, data={
            "snapshot": sanity.snapshot_markdown,
            "perspectives": reviews.reports,
            "debate_history": debate.history,
            "verdict": debate.judge_decision,
        })
```

### 6.3 Skill: `factor_verdict`

**职责**: 在因子挖掘流水线中过滤低质量因子

```python
class FactorVerdictSkill(Skill):
    name = "factor_verdict"
    description = "因子质量过滤: 多视角评审 → 接受/拒绝"

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        factor = context["factor"]
        metrics = context["metrics"]  # IC, IR, decay, etc.

        # Multi-perspective review
        reviews = await multi_perspective_review(
            subject=factor["name"],
            subject_type="factor",
            context=metrics,
            perspectives=["ic_quality", "stability", "redundancy"],
        )

        # Structured judgment
        verdict = await structured_judge(
            text=format_reviews(reviews.reports),
            schema_name="FactorVerdict",
        )

        return SkillResult(success=True, data={
            "verdict": verdict.parsed,
            "perspectives": reviews.reports,
        })
```

### 6.4 Skill: `reflection_loop`

**职责**: 回测完成后自动反思 + 写入 KB

```python
class ReflectionLoopSkill(Skill):
    name = "reflection_loop"
    description = "回测后反思: 生成反思 → 写入知识库 → 下次自动召回"

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        decision = context["decision"]
        actual_return = context["actual_return"]
        alpha = context["alpha"]

        # Step 1: Generate reflection
        reflection = await reflect(decision, actual_return, alpha)

        # Step 2: Recall similar past experiences (context injection)
        past = await kb_recall(query=f"similar strategy {decision.get('strategy_name', '')}")

        # Step 3: Store reflection in KB (idempotent)
        kb.store(experiment_id=context["experiment_id"], content=reflection.raw_text,
                 metadata={"source": "reflection", "alpha": alpha})

        return SkillResult(success=True, data={
            "reflection": reflection.note,
            "past_experiences": past.results,
        })
```

---

## 七、Agent 集成（暴露给 LLM）

### 7.1 Tool 注册

**修改 `QuantNodes/agent/tools/__init__.py`**:
```python
from .llm_decision import (
    LLMStructuredJudgeTool,
    LLMReflectTool,
    LLMMarketSanityCheckTool,
    LLMMultiPerspectiveReviewTool,
    LLM debateTool,
    LLMKBRecallTool,
)

_QUANT_TOOL_FACTORIES = [
    ...existing 20+ tools...,
    LLMStructuredJudgeTool,
    LLMReflectTool,
    LLMMarketSanityCheckTool,
    LLMMultiPerspectiveReviewTool,
    LLM debateTool,
    LLMKBRecallTool,
]
```

### 7.2 Tool Schema (LLM 看到的)

```python
# structured_judge
{
    "name": "structured_judge",
    "description": "把分析文本解析为结构化裁决（Pydantic schema）",
    "parameters": {
        "text": {"type": "string"},
        "schema": {"type": "string", "enum": ["PortfolioDecision", "FactorVerdict", "StrategyVerdict", "ReflectionNote"]},
        "fallback": {"type": "boolean", "default": True}
    }
}

# debate
{
    "name": "debate",
    "description": "多角色对抗辩论，最后给出结构化裁决",
    "parameters": {
        "question": {"type": "string"},
        "positions": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "object"},
        "rounds": {"type": "integer", "default": 1, "minimum": 1, "maximum": 5},
        "judge_schema": {"type": "string", "default": "PortfolioDecision"}
    }
}
```

### 7.3 LLM 调用示例

```
User: "请评估 momentum_20d 因子是否值得用"
Agent: → 调用 structured_judge → 调用 multi_perspective_review → 调用 debate
     → 返回 PortfolioDecision
```

---

## 八、实施计划（3 阶段）

### Phase 1（基础工具）— 3-4 天

1. **`schemas.py`** — 5 个 Pydantic schemas + SCHEMA_REGISTRY
2. **`parser.py`** — Rating regex parser
3. **`structured_judge.py`** ⭐ 第一个实现
4. **`reflect.py`** — 复用 TA 提示词
5. **`market_sanity_check.py`** — 复用 QuantNodes 因子引擎

### Phase 2（辩论工具）— 3-4 天

1. **`prompts.py`** — 视角 + 角色提示词模板
2. **`multi_perspective_review.py`** — N 视角并行
3. **`debate.py`** — 轮次管理 + 路由 + 复用 structured_judge

### Phase 3（知识闭环）— 2-3 天

1. **`kb_recall.py`** — 复用 KnowledgeBase
2. **`strategy_review.py` Skill** — 组合 1
3. **`factor_verdict.py` Skill** — 组合 2
4. **`reflection_loop.py` Skill** — 组合 3

**总预估**: 8-11 天

---

## 九、与 TradingAgents 的关键差异

| 维度 | TradingAgents | QuantNodes 6 Tools |
|------|--------------|-------------------|
| 粒度 | 1 个完整 4-phase pipeline | 6 个独立小工具 |
| 状态 | LangGraph state machine | 无状态（最小 session） |
| 入口 | CLI `tradingagents analyze` | Python API + nanobot tool |
| 组合方式 | 固定 pipeline | 用户自由组合 |
| 复用性 | 低（绑定 framework） | 高（独立小工具） |
| 数据 | yfinance / Alpha Vantage | **复用 QuantNodes 因子引擎** |

---

## 十、与 docs/28-31 的关系

| 文档 | 关系 |
|------|------|
| docs/28 | 提供 TradingAgents prompt/schemas 参考 |
| docs/29 | 集成计划的"可复用工具化"版本 |
| docs/30 | 核心价值分析（self-evolution 闭环） |
| docs/31 | 功能细节参考（实现时查阅） |
| **docs/32** | **6 工具最终设计（本文档）** |

---

## 十一、待实施检查清单

- [ ] Phase 1: `schemas.py` + `parser.py` + 3 个工具 + 测试
- [ ] Phase 2: `prompts.py` + 2 个辩论工具 + 测试
- [ ] Phase 3: `kb_recall.py` + 3 个 Skill + 测试
- [ ] Agent 集成: 注册到 `_QUANT_TOOL_FACTORIES`
- [ ] LLM Schema 验证（让 Agent 能看到并调用）
- [ ] 与现有 KnowledgeBase / 因子引擎的兼容性测试

---

## 十二、附录：核心 Schema 速查

```python
# 5 个核心 schemas（详 §4.1）
PortfolioRating        # 5-tier: Buy/Overweight/Hold/Underweight/Sell
FactorVerdict          # accept + confidence + rationale + concerns
StrategyVerdict        # rating + suggested_position_pct + risk_factors
PortfolioDecision      # rating + executive_summary + investment_thesis + price_target + time_horizon
ReflectionNote         # what_worked + what_failed + lesson + confidence
```