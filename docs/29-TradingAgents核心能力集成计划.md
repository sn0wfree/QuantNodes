# TradingAgents 核心能力集成计划

> **编号**: 29
> **状态**: 📋 计划中
> **依赖**: docs/28-TradingAgents调研报告.md
> **目标**: 将 TradingAgents 的高价值能力嫁接到 QuantNodes，不复制 TradingAgents

---

## 一、集成范围

从 TradingAgents 中提取 7 项高价值能力，按优先级排列：

| # | 能力 | 来源 | 价值 | 预估 |
|---|------|------|------|------|
| 1 | **Structured Output** | `schemas.py` + `structured.py` | 🔴 极高 | 2-3d |
| 2 | **多 Agent 辩论框架** | `researchers/` + `risk_mgmt/` + `managers/` | 🔴 极高 | 3-5d |
| 3 | **Market Data Validation** | `market_data_validator.py` | 🔴 高 | 1-2d |
| 4 | **双模型架构** | `default_config.py` (deep_think + quick_think) | 🟡 高 | 1d |
| 5 | **Multi-Provider LLM** | `llm_clients/` (20+ providers) | 🟡 高 | 2-3d |
| 6 | **Deferred Reflection** | `reflection.py` + `memory.py` | 🟡 中 | 1-2d |
| 7 | **数据源抽象** | `dataflows/interface.py` (vendor 回退链) | 🟢 中 | 1-2d |

**总预估**: 11-18d

---

## 二、Structured Output (P0)

### 2.1 做什么

在 QuantNodes 的 LLM 客户端中增加 Structured Output 能力，让 LLM 输出受 Pydantic schema 约束的结构化数据。

### 2.2 当前状态

- QuantNodes `LLMClientBase` 只有 `chat()` 和 `chat_stream()` 返回自由文本
- TradingAgents 用 `llm.with_structured_output(Schema)` + graceful fallback

### 2.3 实现方案

**新增文件**：
```
QuantNodes/ai/llm/structured.py    # StructuredOutputMixin
QuantNodes/ai/llm/schemas.py       # Pydantic schemas (PortfolioRating, ResearchPlan, etc.)
```

**StructuredOutputMixin**：
```python
class StructuredOutputMixin:
    """Mixin for LLM clients that support structured output."""

    def with_structured_output(self, schema: Type[BaseModel]) -> "StructuredLLM":
        """Bind a Pydantic schema to the LLM for structured generation."""
        ...

    def invoke_structured(self, messages, schema, fallback_llm=None):
        """Try structured output; fall back to free text + regex extraction."""
        ...
```

**Schema 定义**（复用 TradingAgents 的 5-tier rating）：
```python
class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"

class ResearchPlan(BaseModel):
    recommendation: PortfolioRating
    rationale: str
    strategic_actions: str

class TraderProposal(BaseModel):
    action: Literal["Buy", "Hold", "Sell"]
    reasoning: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    position_sizing: Optional[str]

class PortfolioDecision(BaseModel):
    rating: PortfolioRating
    executive_summary: str
    investment_thesis: str
    price_target: Optional[float]
    time_horizon: Optional[str]
```

**Graceful Fallback**：
```python
def invoke_structured_or_freetext(llm, schema, prompt, render_fn, agent_name):
    try:
        result = llm.with_structured_output(schema).invoke(prompt)
        return render_fn(result)
    except Exception:
        response = llm.chat(prompt)
        return response.content  # 降级为自由文本
```

### 2.4 验收

- [ ] `with_structured_output(Schema)` 对 OpenAI/Azure 正常工作
- [ ] 不支持的 provider 自动降级为自由文本
- [ ] 5 个 schema 可正常序列化/反序列化

---

## 三、多 Agent 辩论框架 (P0)

### 3.1 做什么

在 QuantNodes agent 中新增辩论 skill，支持 Bull/Bear 2 方辩论 + Risk 3 方辩论，产出结构化决策。

### 3.2 当前状态

- QuantNodes 单 Agent + 26 工具，无辩论机制
- 已有 `agent/skills_quant/` 工作流定义

### 3.3 实现方案

**新增文件**：
```
QuantNodes/agent/tools/debate.py           # 辩论工具 (Bull/Bear + Risk 3-way)
QuantNodes/agent/skills_quant/debate-trading/SKILL.md  # 辩论工作流
```

**辩论工具设计**：
```python
class DebateTool(Tool):
    name = "debate"
    description = "Run multi-agent debate on a trading decision. Supports Bull/Bear investment debate and 3-way risk debate."

    parameters = {
        "action": {
            "type": "string",
            "enum": ["investment_debate", "risk_debate", "full_pipeline"],
            "description": "Which debate phase to run"
        },
        "ticker": {"type": "string"},
        "analysis_date": {"type": "string"},
        "reports": {  # Analyst reports (from prior analysis)
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "sentiment": {"type": "string"},
                "news": {"type": "string"},
                "fundamentals": {"type": "string"}
            }
        },
        "debate_rounds": {"type": "integer", "default": 1},
        "trader_proposal": {"type": "string"}  # For risk_debate only
    }

    async def execute(self, action, ticker, analysis_date, reports, **kwargs):
        if action == "investment_debate":
            return await self._run_investment_debate(ticker, reports, kwargs.get("debate_rounds", 1))
        elif action == "risk_debate":
            return await self._run_risk_debate(ticker, reports, kwargs.get("trader_proposal"), kwargs.get("debate_rounds", 1))
        elif action == "full_pipeline":
            return await self._run_full_pipeline(ticker, analysis_date, reports, kwargs.get("debate_rounds", 1))
```

**辩论引擎**：
```python
class DebateEngine:
    """Multi-agent debate engine ported from TradingAgents."""

    def __init__(self, llm_client, config=None):
        self.llm = llm_client
        self.config = config or {}

    async def run_investment_debate(self, reports, max_rounds=1):
        """Bull vs Bear debate. Returns ResearchPlan."""
        history = ""
        bull_history = ""
        bear_history = ""

        for round_num in range(max_rounds):
            # Bull speaks
            bull_prompt = self._build_bull_prompt(reports, history, bear_history)
            bull_response = await self.llm.chat(bull_prompt)
            history += f"\nBull: {bull_response.content}"
            bull_history += f"\n{bull_response.content}"

            # Bear speaks
            bear_prompt = self._build_bear_prompt(reports, history, bull_history)
            bear_response = await self.llm.chat(bear_prompt)
            history += f"\nBear: {bear_response.content}"
            bear_history += f"\n{bear_response.content}"

        # Research Manager decides (structured output)
        return await self._judge_investment(reports, history)

    async def run_risk_debate(self, reports, trader_proposal, max_rounds=1):
        """3-way risk debate. Returns PortfolioDecision."""
        history = ""
        speakers = ["Aggressive", "Conservative", "Neutral"]

        for round_num in range(max_rounds):
            for speaker in speakers:
                prompt = self._build_risk_prompt(speaker, reports, trader_proposal, history)
                response = await self.llm.chat(prompt)
                history += f"\n{speaker}: {response.content}"

        # Portfolio Manager decides (structured output)
        return await self._judge_risk(reports, trader_proposal, history)

    async def _judge_investment(self, reports, history):
        """Research Manager structured output."""
        prompt = self._build_judge_prompt(reports, history)
        llm = self.llm.with_structured_output(ResearchPlan)
        result = llm.invoke(prompt)
        return result

    async def _judge_risk(self, reports, trader_proposal, history):
        """Portfolio Manager structured output."""
        prompt = self._build_pm_prompt(reports, trader_proposal, history)
        llm = self.llm.with_structured_output(PortfolioDecision)
        result = llm.invoke(prompt)
        return result
```

**辩论 Prompt 模板**（复用 TradingAgents 的 prompt 设计）：
```python
BULL_PROMPT = """You are a Bull Analyst advocating for investing in {ticker}.
Your task is to build a strong, evidence-based case emphasizing growth potential,
competitive advantages, and positive market indicators.

Key points to focus on:
- Growth Potential: market opportunities, revenue projections, scalability
- Competitive Advantages: unique products, strong branding, dominant positioning
- Positive Indicators: financial health, industry trends, positive news
- Bear Counterpoints: critically analyze the bear argument with specific data

Resources:
{reports}

Conversation history: {history}
Last bear argument: {last_bear}

Engage directly with the bear analyst's points and debate effectively."""

BEAR_PROMPT = """You are a Bear Analyst making the case against investing in {ticker}.
Your goal is to present a well-reasoned argument emphasizing risks, challenges,
and negative indicators.

Key points to focus on:
- Risks and Challenges: market saturation, financial instability, macro threats
- Competitive Weaknesses: weaker positioning, declining innovation, competitor threats
- Negative Indicators: financial data, market trends, adverse news
- Bull Counterpoints: expose weaknesses or over-optimistic assumptions

Resources:
{reports}

Conversation history: {history}
Last bull argument: {last_bull}

Engage directly with the bull analyst's points and debate effectively."""
```

### 3.4 SKILL.md 定义

```markdown
# Debate Trading

Multi-agent debate for trading decisions.

## Workflow

1. Run analyst tools to gather reports (market/sentiment/news/fundamentals)
2. Run investment debate (Bull vs Bear) → ResearchPlan
3. Convert ResearchPlan to TraderProposal
4. Run risk debate (Aggressive/Conservative/Neutral) → PortfolioDecision
5. Store result in Wiki knowledge base

## Tools Used
- debate (investment_debate, risk_debate, full_pipeline)
- factor (for quantitative backing)
- wiki (for result storage)

## Acceptance Criteria
- Debate produces structured output (not free text)
- Bull/Bear each cite specific analyst data
- Risk debate considers all 3 perspectives
- Final decision includes entry/stop/sizing
```

### 3.5 验收

- [ ] `DebateTool` 可执行 Bull/Bear 辩论
- [ ] `DebateTool` 可执行 3-way Risk 辩论
- [ ] 结构化输出 (ResearchPlan / PortfolioDecision)
- [ ] 辩论历史可持久化到 Wiki
- [ ] 支持可配置轮次 (1-5 rounds)

---

## 四、Market Data Validation (P0)

### 4.1 做什么

新增防幻觉数据验证工具，让 LLM 分析师引用确定性计算的价格/指标数据，而不是幻觉数字。

### 4.2 当前状态

- QuantNodes 有完整的因子计算引擎 (317+ 算子)
- 但没有"验证快照"工具让 LLM 引用确定性数据

### 4.3 实现方案

**新增文件**：
```
QuantNodes/agent/tools/market_validation.py  # Market data validation tool
```

**MarketValidationTool**：
```python
class MarketValidationTool(Tool):
    name = "market_validation"
    description = "Generate a verified market snapshot with exact OHLCV and technical indicators. Use this as the source of truth for all price and indicator claims."

    async def execute(self, ticker, date, indicators=None):
        """Compute deterministic ground-truth snapshot."""
        # 1. Load OHLCV data
        ohlcv = self._load_ohlcv(ticker, date)

        # 2. Compute indicators
        if indicators is None:
            indicators = ["ema_10", "ema_50", "ema_200", "rsi", "macd", "bollinger", "atr"]

        computed = {}
        for ind in indicators:
            computed[ind] = self._compute_indicator(ohlcv, ind)

        # 3. Format as structured markdown
        return self._format_snapshot(ticker, date, ohlcv, computed)
```

**输出格式**（复用 TradingAgents 的 anti-confabulation 指令）：
```markdown
## Verified Market Snapshot: {ticker} as of {date}

### Latest OHLCV
| Open | High | Low | Close | Volume |
|------|------|-----|-------|--------|
| 150.23 | 152.10 | 149.80 | 151.50 | 45,230,000 |

### Verified Technical Indicators
| Indicator | Value |
|-----------|-------|
| EMA 10 | 150.82 |
| EMA 50 | 148.35 |
| EMA 200 | 142.10 |
| RSI (14) | 58.3 |
| MACD | 1.23 |
| Bollinger Upper | 153.20 |
| Bollinger Lower | 147.50 |
| ATR (14) | 2.15 |

### Recent Closes (last 10 days)
[150.2, 149.8, 151.1, 152.3, 148.9, 150.5, 151.2, 149.5, 150.8, 151.5]

---
Use this snapshot as the source of truth for exact OHLCV, price-level, and indicator-value claims.
If another tool output conflicts with it, flag the discrepancy rather than inventing a reconciled number.
```

### 4.4 验收

- [ ] 输出确定性 OHLCV + 指标 (不依赖 LLM)
- [ ] 防幻觉指令包含在输出中
- [ ] 支持自定义指标列表

---

## 五、双模型架构 (P1)

### 5.1 做什么

支持 deep_think (关键决策) + quick_think (分析/辩论) 分层 LLM 调用。

### 5.2 实现方案

**修改文件**：
```
QuantNodes/ai/llm/base.py          # 新增 DualModelClient
QuantNodes/agent/config_mapper.py  # 新增 deep/quick model 配置
```

**DualModelClient**：
```python
class DualModelClient:
    """Two-tier LLM client: deep_think for decisions, quick_think for analysis."""

    def __init__(self, deep_client, quick_client):
        self.deep = deep_client   # GPT-5.5, Claude Opus, etc.
        self.quick = quick_client  # GPT-5.4-mini, etc.

    def chat_deep(self, messages, **kwargs):
        """Deep reasoning for critical decisions."""
        return self.deep.chat(messages, **kwargs)

    def chat_quick(self, messages, **kwargs):
        """Fast analysis for routine tasks."""
        return self.quick.chat(messages, **kwargs)

    def with_structured_output(self, schema, tier="deep"):
        """Bind schema to the specified tier."""
        client = self.deep if tier == "deep" else self.quick
        return client.with_structured_output(schema)
```

**配置**：
```python
# .env
LLM_DEEP_MODEL=gpt-5.5
LLM_QUICK_MODEL=gpt-5.4-mini
LLM_PROVIDER=openai
```

### 5.3 验收

- [ ] Debate 用 quick_think, Judge 用 deep_think
- [ ] 配置可切换 provider/model
- [ ] 成本降低 ~60% (大部分调用走 quick)

---

## 六、Multi-Provider LLM (P1)

### 6.1 做什么

扩展 QuantNodes 的 LLM 支持，从 2 个 provider (OpenAI + Azure) 扩展到 20+。

### 6.2 实现方案

**复用 TradingAgents 的 ProviderSpec 注册表**：

```python
@dataclass(frozen=True)
class ProviderSpec:
    base_url: str | None = None
    key_env: str | None = None
    key_optional: bool = False
    use_responses_api: bool = False
    chat_class: type = None  # 默认 NormalizedChatOpenAI

PROVIDER_REGISTRY = {
    "openai":     ProviderSpec(use_responses_api=True),
    "anthropic":  ProviderSpec(base_url="https://api.anthropic.com/v1", key_env="ANTHROPIC_API_KEY"),
    "google":     ProviderSpec(base_url="https://generativelanguage.googleapis.com/v1beta", key_env="GOOGLE_API_KEY"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY"),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", key_env="DASHSCOPE_API_KEY"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/", key_env="ZHIPU_API_KEY"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", key_env="MINIMAX_API_KEY"),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", key_optional=True),
    # ... more providers
}
```

### 6.3 验收

- [ ] 支持 10+ providers
- [ ] 自动检测 API key
- [ ] 优雅降级 (key 缺失时 skip)

---

## 七、Deferred Reflection (P1)

### 7.1 做什么

回测后自动反思决策结果，注入 knowledge base 供未来决策参考。

### 7.2 实现方案

**新增文件**：
```
QuantNodes/agent/tools/reflection.py  # Reflection tool
```

**ReflectionTool**：
```python
class ReflectionTool(Tool):
    name = "reflect_on_trade"
    description = "Reflect on a past trading decision now that the outcome is known. Generates lessons for future decisions."

    async def execute(self, ticker, decision_date, holding_days=5):
        # 1. Fetch actual return
        actual_return = self._fetch_return(ticker, decision_date, holding_days)

        # 2. Compute alpha vs benchmark
        alpha = self._compute_alpha(ticker, decision_date, holding_days)

        # 3. Generate reflection via LLM
        prompt = f"""You are a trading analyst reviewing your past decision.

Raw return: {actual_return:+.1%}
Alpha vs benchmark: {alpha:+.1%}

Write 2-4 sentences covering:
1. Was the directional call correct?
2. Which part of the thesis held or failed?
3. One concrete lesson for next time."""

        reflection = await self.llm.chat(prompt)

        # 4. Store in knowledge base
        self._store_reflection(ticker, decision_date, reflection.content, actual_return, alpha)

        return {"reflection": reflection.content, "return": actual_return, "alpha": alpha}
```

### 7.3 验收

- [ ] 自动 fetch 实际收益
- [ ] 生成 2-4 句反思
- [ ] 存入 knowledge base

---

## 八、数据源抽象 (P2)

### 8.1 做什么

新增多 vendor 回退链，支持 yfinance + Alpha Vantage + FRED + Polymarket。

### 8.2 实现方案

**新增文件**：
```
QuantNodes/data/vendor_router.py      # Vendor routing
QuantNodes/data/vendors/yfinance.py   # yfinance adapter
QuantNodes/data/vendors/alpha_vantage.py  # Alpha Vantage adapter
QuantNodes/data/vendors/fred.py       # FRED macro data
QuantNodes/data/vendors/polymarket.py # Prediction markets
```

**VendorRouter**：
```python
class VendorRouter:
    """Multi-vendor data routing with fallback chains."""

    def __init__(self, config):
        self.config = config
        self.vendors = {
            "yfinance": YFinanceVendor(),
            "alpha_vantage": AlphaVantageVendor(),
            "fred": FredVendor(),
            "polymarket": PolymarketVendor(),
        }

    def get_stock_data(self, ticker, start, end, preferred_vendor=None):
        vendors = self._resolve_chain("stock_data", preferred_vendor)
        for vendor_name in vendors:
            try:
                return self.vendors[vendor_name].get_stock_data(ticker, start, end)
            except VendorError as e:
                logger.warning(f"Vendor {vendor_name} failed: {e}")
                continue
        raise NoDataAvailable(f"All vendors failed for {ticker}")
```

### 8.3 验收

- [ ] 支持 4 个 vendor
- [ ] 自动回退
- [ ] 可选 vendor 降级

---

## 九、文件变更清单

### 9.1 新增 (~15 文件)

```
QuantNodes/ai/llm/structured.py              # StructuredOutputMixin + schemas
QuantNodes/ai/llm/dual_model.py              # DualModelClient
QuantNodes/ai/llm/providers.py               # Multi-provider registry
QuantNodes/agent/tools/debate.py             # Debate tool
QuantNodes/agent/tools/market_validation.py  # Market validation tool
QuantNodes/agent/tools/reflection.py         # Reflection tool
QuantNodes/agent/skills_quant/debate-trading/SKILL.md  # Debate workflow
QuantNodes/data/vendor_router.py             # Vendor routing
QuantNodes/data/vendors/__init__.py
QuantNodes/data/vendors/yfinance.py
QuantNodes/data/vendors/alpha_vantage.py
QuantNodes/data/vendors/fred.py
QuantNodes/data/vendors/polymarket.py
tests/test_structured_output.py
tests/test_debate_tool.py
tests/test_market_validation.py
tests/test_reflection_tool.py
tests/test_vendor_router.py
docs/29-TradingAgents核心能力集成计划.md       # 本文档
```

### 9.2 修改 (~3 文件)

```
QuantNodes/agent/tools/__init__.py           # 注册新 tools
QuantNodes/agent/config_mapper.py            # deep/quick model 配置
CHANGELOG.md                                  # 记录变更
```

---

## 十、实施顺序

### Phase 1: 基础能力 (3-5d)
- Structured Output (schemas + mixin + fallback)
- 双模型架构 (DualModelClient)
- Market Data Validation tool

### Phase 2: 辩论框架 (3-5d)
- DebateTool (investment + risk debate)
- DebateEngine (轮次管理 + prompt 模板)
- debate-trading SKILL.md

### Phase 3: 学习与数据 (3-5d)
- Deferred Reflection tool
- Multi-Provider LLM registry
- Vendor Router + 4 vendors

### Phase 4: 集成与测试 (2-3d)
- 集成到 nanobot agent
- 端到端测试
- 文档更新

---

## 十一、与现有系统的兼容性

| 现有组件 | 影响 | 兼容性 |
|---------|------|--------|
| `LLMClientBase` | 新增 mixin，不改接口 | ✅ 完全兼容 |
| `agent/tools/` | 新增 3 个 tool，不改现有 | ✅ 完全兼容 |
| `skills_quant/` | 新增 1 个 skill | ✅ 完全兼容 |
| `nanobot_bridge.py` | 无修改 | ✅ 完全兼容 |
| `config_mapper.py` | 新增 2 个 env var | ✅ 向后兼容 |
| `mcp_server/` | 可选暴露新 tools | ✅ 可选 |

---

## 十二、风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM API 成本增加 | 双模型架构 (60% 走 quick_think) |
| 辩论轮次过多导致延迟 | 默认 1 轮，可配置上限 5 |
| Structured output 不支持 | graceful fallback 到自由文本 |
| Vendor API 限流 | 回退链 + 可选 vendor 降级 |
| 辩论质量不稳定 | Structured output 约束 + 多轮反思 |
