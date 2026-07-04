# TradingAgents 功能深度梳理

> **编号**: 31
> **状态**: ✅ 梳理完成
> **依赖**: docs/28 (调研报告) + docs/29 (集成计划) + docs/30 (讨论总结)
> **日期**: 2026-06-24

---

## 一、整体架构

TradingAgents 采用**三层数据 + 十 Agent 决策**架构：

```
Layer 1: Agent Tool Wrappers  (@tool 装饰器函数, LangChain 接口)
    ↓
Layer 2: Vendor Router         (route_to_vendor, 配置驱动回退链)
    ↓
Layer 3: Vendor Implementations (yfinance / Alpha Vantage / FRED / Polymarket)

+ 10 Agent (4 Analysts + 2 Researchers + 3 Risk Debators + 1 Trader/PM)
+ Memory Log (append-only markdown, pending → resolved → reflection)
+ Structured Output (Pydantic schema 约束 LLM 输出)
+ CLI (Typer-based 8 步向导 + 实时显示)
```

---

## 二、数据层 (3 层架构)

### 2.1 Layer 1: Agent Tool Wrappers

所有工具用 LangChain `@tool` 装饰器包装，每个工具都委托给 `route_to_vendor()`：

```python
@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve stock price data (OHLCV) for a given ticker symbol."""
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)
```

### 2.2 工具清单 (12 个 @tool)

| 工具 | 数据 | 返回格式 | 使用的 Analyst |
|------|------|---------|---------------|
| `get_stock_data` | OHLCV | CSV with header | Market |
| `get_indicators` | 技术指标 (13种) | 时间序列 + 描述 | Market |
| `get_verified_market_snapshot` | 确定性验证快照 | 结构化 markdown | Market |
| `get_fundamentals` | 公司概览 (28字段) | key: value 列表 | Fundamentals |
| `get_balance_sheet` | 资产负债表 | CSV | Fundamentals |
| `get_cashflow` | 现金流 | CSV | Fundamentals |
| `get_income_statement` | 利润表 | CSV | Fundamentals |
| `get_news` | 个股新闻 | 文章列表 | Sentiment/News |
| `get_global_news` | 全球宏观新闻 | 文章列表 | News |
| `get_insider_transactions` | 内部交易 | CSV | News |
| `get_macro_indicators` | FRED 宏观数据 | markdown 表格 | News |
| `get_prediction_markets` | Polymarket 预测 | markdown 概率列表 | News |

### 2.3 Layer 2: Vendor Router (核心路由)

```python
VENDOR_METHODS = {
    "get_stock_data":          {"alpha_vantage": ..., "yfinance": ...},
    "get_indicators":          {"alpha_vantage": ..., "yfinance": ...},
    "get_fundamentals":        {"alpha_vantage": ..., "yfinance": ...},
    "get_balance_sheet":       {"alpha_vantage": ..., "yfinance": ...},
    "get_cashflow":            {"alpha_vantage": ..., "yfinance": ...},
    "get_income_statement":    {"alpha_vantage": ..., "yfinance": ...},
    "get_news":                {"alpha_vantage": ..., "yfinance": ...},
    "get_global_news":         {"yfinance": ..., "alpha_vantage": ...},
    "get_insider_transactions": {"alpha_vantage": ..., "yfinance": ...},
    "get_macro_indicators":    {"fred": ...},
    "get_prediction_markets":  {"polymarket": ...},
}
```

#### 路由逻辑

```python
def route_to_vendor(method, *args, **kwargs):
    # 1. 解析 vendor 链
    vendor_config = get_vendor(category, method)
    vendor_chain = [v.strip() for v in vendor_config.split(',')]

    # 2. 依次尝试
    for vendor in vendor_chain:
        try:
            return vendor_impl(*args, **kwargs)
        except VendorRateLimitError:
            continue  # 限流 → 下一个
        except VendorNotConfiguredError:
            continue  # 未配置 → 下一个
        except NoMarketDataError as e:
            last_no_data = e
            continue
        except Exception:
            continue

    # 3. 全部失败 → 返回 sentinel 字符串
    if last_no_data:
        return "NO_DATA_AVAILABLE: ..."
    if category in OPTIONAL_CATEGORIES:
        return "DATA_UNAVAILABLE: ..."  # 优雅降级
    raise first_error  # 核心类别 → 抛出
```

#### 错误类型

```
VendorError
  +-- NoMarketDataError          (空结果或过时数据)
  +-- VendorRateLimitError       (限流 → 下一个 vendor)
  +-- VendorNotConfiguredError   (缺 API key)
```

#### 关键设计：No Silent Fallback

配置的 vendor 链就是 vendor 链——不会"自动尝试"用户没配置的 vendor（issue #988/#289 修复）。

### 2.4 Layer 3: Vendor 实现详情

#### 2.4.1 yfinance (主 vendor, 免费)

```python
def get_YFin_data_online(symbol, start_date, end_date):
    canonical = normalize_symbol(symbol)  # XAUUSD+ -> GC=F
    ticker = yf.Ticker(canonical)
    # yfinance end is EXCLUSIVE, +1 day to make inclusive
    end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_inclusive))
    if data.empty:
        raise NoMarketDataError(...)
    _assert_ohlcv_not_stale(data, end_date, ...)
    csv_string = data.to_csv()
    return header + csv_string
```

**支持的 13 个技术指标**:
```python
"close_50_sma", "close_200_sma", "close_10_ema",
"macd", "macds", "macdh",
"rsi", "boll", "boll_ub", "boll_lb",
"atr", "vwma", "mfi"
```

**基本面字段 (28 个)**: Name, Sector, Industry, Market Cap, PE/Forward PE/PEG, P/B, EPS, Dividend Yield, Beta, 52W High/Low, 50D/200D Avg, Revenue, Gross Profit, EBITDA, Net Income, Margins, ROE, ROA, Debt/Equity, etc.

#### 2.4.2 Alpha Vantage (付费备选)

**限流检测**（HTTP 200 但 body 含 "Information" / "Note"）:
```python
notice = response_json.get("Information") or response_json.get("Note")
if notice:
    low = notice.lower()
    if "rate limit" in low or "requests per day" in low:
        raise AlphaVantageRateLimitError(...)
    if "api key" in low:
        raise AlphaVantageNotConfiguredError(...)
```

**支持的端点**: TIME_SERIES_DAILY_ADJUSTED, OVERVIEW, BALANCE_SHEET, CASH_FLOW, INCOME_STATEMENT, SMA/EMA/MACD/RSI/BBANDS/ATR, NEWS_SENTIMENT, INSIDER_TRANSACTIONS

#### 2.4.3 FRED (免费, 宏观数据)

**30+ 别名**:
```python
MACRO_SERIES = {
    # 利率
    "fed_funds_rate": "FEDFUNDS", "2y_treasury": "DGS2",
    "10y_treasury": "DGS10", "30y_treasury": "DGS30",
    "yield_curve": "T10Y2Y",  # 10Y-2Y 利差
    # 通胀
    "cpi": "CPIAUCSL", "core_cpi": "CPILFESL",
    "pce": "PCEPI", "core_pce": "PCEPILFE",
    # 增长
    "real_gdp": "GDPC1", "industrial_production": "INDPRO",
    # 就业
    "unemployment": "UNRATE", "nonfarm_payrolls": "PAYEMS",
    # 市场
    "vix": "VIXCLS", "dollar_index": "DTWEXBGS",
    # 消费
    "consumer_sentiment": "UMCSENT", "retail_sales": "RSAFS",
}
```

**输出格式** (markdown 表格):
```markdown
## FRED: 10-Year Treasury Constant Maturity Rate (DGS10)
- Units: Percent
- Frequency: Daily
- Window: 2025-01-01 to 2025-07-04
**Latest:** 4.35 (2025-07-03) | **Change over window:** -0.12 (-2.69%)

| Date | Value |
| --- | --- |
| 2025-07-03 | 4.35 |
| 2025-07-02 | 4.38 |
```

#### 2.4.4 Polymarket (免费, 预测市场)

```python
def get_prediction_markets(topic, limit=6):
    data = _request("public-search", {"q": topic, "limit_per_type": 20})
    candidates = [m for event in data["events"] for m in event["markets"]
                  if _is_forward_looking(m, now)]
    candidates.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)
    return header + "\n".join(lines) + "\n"
```

**输出格式**:
```markdown
## Polymarket prediction markets: "Fed rate cut"
- **Will the Fed cut rates in July 2025?** -- Yes 68% ($1.2M volume, resolves 2025-07-31, 1-week +5.2pp)
- **US recession in 2025?** -- No 82% ($890K volume, resolves 2025-12-31)
```

---

## 三、Agent 层 (10 个角色)

### 3.1 Analyst Team (4 个, ReAct + tools)

| Analyst | 工具 | 输出 | LLM 层级 |
|---------|------|------|---------|
| Market | `get_stock_data` + `get_indicators` + `get_verified_market_snapshot` | 技术趋势报告 (markdown 表格) | quick_think |
| Sentiment | `get_news` | 情绪分析报告 | quick_think |
| News | `get_news` + `get_global_news` + `get_insider_transactions` + `get_macro_indicators` + `get_prediction_markets` | 宏观/政治/新闻报告 | quick_think |
| Fundamentals | `get_fundamentals` + `get_balance_sheet` + `get_cashflow` + `get_income_statement` | 财务分析报告 | quick_think |

**ReAct Tool Loop**:
```
LLM 调用 → 返回 tool_calls? 
  → 是: 执行工具 → 结果返回 LLM → 循环
  → 否: 输出最终报告
```

**Msg Clear**: 每个 analyst 完成后清除 messages 列表，只保留最终报告传给下一个。

### 3.2 Analyst Node Specs

```python
ANALYST_NODE_SPECS = {
    "market": AnalystNodeSpec(
        agent_node="Market Analyst",
        tool_node="tools_market",
        report_key="market_report",
    ),
    "social": AnalystNodeSpec(  # Sentiment Analyst
        agent_node="Sentiment Analyst",
        tool_node="tools_social",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        agent_node="News Analyst",
        tool_node="tools_news",
        report_key="news_report",
    ),
    "fundamentals": AnalystNodeSpec(
        agent_node="Fundamentals Analyst",
        tool_node="tools_fundamentals",
        report_key="fundamentals_report",
    ),
}
```

### 3.3 Researcher Team (2 个, 辩论)

| Researcher | 角色 | Prompt 关键词 |
|-----------|------|-------------|
| Bull | 看多 | growth potential, competitive advantages, positive indicators |
| Bear | 看空 | risks, challenges, negative indicators, competitive weaknesses |

**辩论状态**:
```python
InvestDebateState = {
    "bull_history": "",      # Bull 所有论点
    "bear_history": "",      # Bear 所有论点
    "history": "",           # 完整辩论记录
    "current_response": "",  # 最近一轮论点
    "count": 0,              # 轮次计数
}
```

**路由逻辑**:
```python
if count >= 2 * max_debate_rounds:  # 退出
    return "Research Manager"
if current_response.startswith("Bull"):
    return "Bear Researcher"
return "Bull Researcher"
```

### 3.4 Decision Makers (3 个, Structured Output)

| Agent | Schema | 输出字段 | LLM 层级 |
|-------|--------|---------|---------|
| Research Manager | `ResearchPlan` | recommendation (5-tier) + rationale + strategic_actions | **deep_think** |
| Trader | `TraderProposal` | action + reasoning + entry_price + stop_loss + position_sizing | quick_think |
| Portfolio Manager | `PortfolioDecision` | rating + executive_summary + investment_thesis + price_target + time_horizon | **deep_think** |

### 3.5 Risk Management Team (3 个, 辩论)

| Debator | 角色 | 关注点 |
|---------|------|--------|
| Aggressive | 激进 | high-reward, growth potential, competitive advantages |
| Conservative | 保守 | protect assets, minimize volatility, stable growth |
| Neutral | 中性 | balanced perspective, well-rounded approach |

**路由逻辑**:
```python
if count >= 3 * max_risk_discuss_rounds:  # 退出
    return "Portfolio Manager"
if latest_speaker == "Aggressive":
    return "Conservative Analyst"
if latest_speaker == "Conservative":
    return "Neutral Analyst"
return "Aggressive Analyst"
```

### 3.6 Instrument Context (防错)

```python
def build_instrument_context(ticker, asset_type="stock", identity=None):
    context = (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation."
    )
    if identity:
        context += f" Resolved identity: Company: {name}; Sector: {sector}; ..."
    return context
```

**关键**: 防止 LLM 把股票 "AAPL" 误解为别的公司（issue #814）。

---

## 四、决策层 (Structured Output)

### 4.1 5-Tier Rating Scale

```
Buy → Overweight → Hold → Underweight → Sell
(最看多)                        (最看空)
```

### 4.2 Rating Parser (确定性, 无 LLM)

```python
# Pass 1: 显式标签匹配
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

# Pass 2: 首次出现匹配
for line in text.splitlines():
    for word in line.lower().split():
        clean = word.strip("*:.,")
        if clean in _RATING_SET:
            return clean.capitalize()

# 默认: "Hold"
```

**使用点**: 4 个 (Research Manager, Portfolio Manager, Signal Processor, Memory Log)

### 4.3 Structured Output Binding

```python
def bind_structured(llm, schema, agent_name):
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError):
        return None  # 降级为自由文本

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name):
    if structured_llm:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)  # Pydantic → markdown
        except Exception:
            logger.warning("structured-output failed; retrying as free text")
    return plain_llm.invoke(prompt).content
```

### 4.4 Schema 定义

```python
class PortfolioRating(str, Enum):
    BUY = "Buy"; OVERWEIGHT = "Overweight"; HOLD = "Hold"
    UNDERWEIGHT = "Underweight"; SELL = "Sell"

class ResearchPlan(BaseModel):
    recommendation: PortfolioRating
    rationale: str
    strategic_actions: str

class TraderProposal(BaseModel):
    action: TraderAction  # Buy/Hold/Sell
    reasoning: str
    entry_price: float | None
    stop_loss: float | None
    position_sizing: str | None

class PortfolioDecision(BaseModel):
    rating: PortfolioRating
    executive_summary: str
    investment_thesis: str
    price_target: float | None
    time_horizon: str | None
```

---

## 五、Memory 层 (自我进化)

### 5.1 三阶段生命周期

```
Phase A: 决策时 → store_decision() → 写入 pending entry
Phase B: 结果已知时 → batch_update_with_outcomes() → 更新为 resolved + reflection
Phase C: 下次决策时 → get_past_context() → 注入历史经验到 prompt
```

### 5.2 日志格式

```
[2024-05-10 | NVDA | Buy | pending]
DECISION:
<full Portfolio Manager output>
<!-- ENTRY_END -->

[2024-05-05 | NVDA | Overweight | +5.1% | +3.2% | 5d]
DECISION:
<full Portfolio Manager output>
REFLECTION:
The decision to overweight was justified as the stock outperformed SPY by 3.2%...
<!-- ENTRY_END -->
```

**Tag 字段** (pipe-delimited): `[date | ticker | rating | status_or_return | alpha | holding_days]`

### 5.3 Context Injection

```python
get_past_context() →
  "Past analyses of NVDA (most recent first):"
  "[2024-04-15 | NVDA | Overweight | +5.1% | +3.2% | 5d]"
  "DECISION: The stock shows strong momentum..."
  "REFLECTION: The decision was justified..."

  "Recent cross-ticker lessons:"
  "[2024-04-20 | AMD | Buy | +2.3%]"
  "The AI semiconductor sector continues to show strength..."
```

注入到 Portfolio Manager 的 prompt 中。

### 5.4 Deferred Reflection

```python
def reflect_on_final_decision(final_decision, raw_return, alpha_return, benchmark_name):
    prompt = f"""You are a trading analyst reviewing your own past decision.

Raw return: {raw_return:+.1%}
Alpha vs {benchmark_name}: {alpha_return:+.1%}

Write 2-4 sentences covering:
1. Was the directional call correct?
2. Which part of the thesis held or failed?
3. One concrete lesson for next time."""
```

### 5.5 Benchmark Resolution

```python
benchmark_map = {
    ".NS": "^NSEI", ".T": "^N225", ".HK": "^HSI",
    ".L": "^FTSE", ".TO": "^GSPTSE", ".AX": "^AXJO",
    ".SS": "000001.SS", ".SZ": "399001.SZ", "": "SPY",
}
```

自动根据 ticker 后缀选择对应市场的基准指数。

---

## 六、CLI 层 (交互式)

### 6.1 8 步交互式向导

```
1. Ticker (自动检测 stock/crypto)
2. 分析日期 (YYYY-MM-DD, 不能未来)
3. 输出语言 (English/中文/日本語/...)
4. Analyst 选择 (market/sentiment/news/fundamentals)
5. 研究深度 (debate + risk round counts)
6. LLM Provider (OpenAI/Anthropic/DeepSeek/Qwen/GLM/MiniMax/...)
7. 模型选择 (quick_think + deep_think)
8. Provider 特定配置 (thinking_level/reasoning_effort)
```

**Env-var 优先级**: 每步都可以通过 `TRADINGAGENTS_*` 环境变量跳过。

### 6.2 实时显示布局

```
+------------------------------------------+
| Progress Table    | Messages & Tools     |
| (agent status)    | (recent activity)    |
+------------------------------------------+
| Current Report Panel (markdown)          |
+------------------------------------------+
| Agents: 5/8 | LLM: 12 | Tools: 24      |
+------------------------------------------+
```

### 6.3 关键 CLI 特性

- 4 Hz 刷新率实时显示
- Agent 状态跟踪 (pending/in_progress/completed)
- 报告分段流式渲染
- 工具调用日志
- Token/成本跟踪 (LangChain callbacks)
- 消息去重 (by ID)
- 分段文件保存 (during execution)
- 分析完成后保存报告

---

## 七、版本演进 (v0.1.0 → v0.3.0)

| 版本 | 日期 | 关键特性 |
|------|------|---------|
| v0.1.0 | 2025-06-05 | 初始发布: 4 analysts + bull/bear + trader + risk + PM |
| v0.2.0 | 2026-02-04 | Multi-provider LLM + Alpha Vantage + 工具回退 |
| v0.2.1 | 2026-03-15 | LangChain 安全补丁 + pyproject.toml |
| v0.2.2 | 2026-03-22 | 5-tier rating + OpenAI Responses API + Anthropic effort |
| v0.2.3 | 2026-03-29 | 多语言 + GPT-5.4 + 统一模型目录 |
| **v0.2.4** | 2026-04-25 | **Structured Output + Checkpoint Resume + Persistent Decision Log** |
| v0.2.5 | 2026-05-11 | Grounded Sentiment + MiniMax + Dual-region Qwen/GLM |
| **v0.3.0** | 2026-06-22 | **FRED + Polymarket + Provider Registry + Verified Data Contract** |

**关键里程碑**:
- v0.2.4: 从"纯自由文本"升级为"结构化决策"
- v0.3.0: 从"只用 yfinance"扩展为"多数据源生态"

---

## 八、Provider 兼容性与能力表

### 8.1 4 Native API Clients

| Provider | SDK | 模型示例 |
|----------|-----|---------|
| Anthropic | `anthropic` | Claude Opus/Sonnet/Haiku |
| Google | `google-generativeai` | Gemini 3.x |
| Azure | `openai` (Azure endpoint) | GPT-5.x |
| Bedrock | `boto3` | Claude on AWS |

### 8.2 16 OpenAI-Compatible Providers

| Provider | base_url | 特殊处理 |
|----------|----------|---------|
| OpenAI | (default) | Responses API |
| xAI (Grok) | `https://api.x.ai/v1` | - |
| DeepSeek | `https://api.deepseek.com` | `reasoning_content` roundtrip |
| Qwen (Intl) | `https://dashscope-intl.aliyuncs.com/...` | - |
| Qwen (CN) | `https://dashscope.aliyuncs.com/...` | - |
| GLM (Intl) | `https://api.z.ai/api/paas/v4/` | - |
| GLM (CN) | `https://open.bigmodel.cn/...` | - |
| MiniMax (Intl) | `https://api.minimax.io/v1` | `reasoning_split=True` |
| MiniMax (CN) | `https://api.minimaxi.com/v1` | `reasoning_split=True` |
| OpenRouter | `https://openrouter.ai/api/v1` | 100+ 模型 |
| Mistral | `https://api.mistral.ai/v1` | - |
| Kimi | `https://api.moonshot.ai/v1` | - |
| Groq | `https://api.groq.com/openai/v1` | - |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | - |
| Ollama | `http://localhost:11434/v1` | key_optional |
| OpenAI-Compatible | (user-defined) | - |

### 8.3 Per-Model Capabilities

```python
@dataclass(frozen=True)
class ModelCapabilities:
    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: str  # function_calling/json_mode/json_schema/none
    requires_reasoning_content_roundtrip: bool  # DeepSeek
    requires_reasoning_split: bool              # MiniMax
```

**关键约束**:
- DeepSeek thinking mode: `tool_choice` 不支持 → 用 `function_calling`
- MiniMax M2.x: `{"none","auto"}` only → 用 `function_calling`
- 本地模型: 总是 `suppress tool_choice` (varied support)

### 8.4 Content 归一化

```python
def normalize_content(response):
    """OpenAI Responses API / Gemini 返回 content 为 list of typed blocks"""
    content = response.content
    if isinstance(content, list):
        texts = [item.get("text","") if isinstance(item,dict) and item.get("type")=="text"
                 else item if isinstance(item,str) else ""
                 for item in content]
        response.content = "\n".join(t for t in texts if t)
    return response
```

---

## 九、配置系统

### 9.1 Environment Variable Override

```python
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":       "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":     "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":       "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":    "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":      "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":           "temperature",
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL": "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":      "anthropic_effort",
}
```

**Type Coercion**:
- Boolean: `true/1/yes/on` → True, `false/0/no/off` → False
- Invalid values → `ValueError` (fail-loud)

### 9.2 Default Config

```python
DEFAULT_CONFIG = {
    # LLM
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    "temperature": None,
    # 辩论
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # 数据
    "news_article_limit": 20,
    "global_news_article_limit": 10,
    "global_news_lookback_days": 7,
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
        "macro_data": "fred",
        "prediction_markets": "polymarket",
    },
    # 持久化
    "checkpoint_enabled": False,
    "memory_log_path": "~/.tradingagents/memory/trading_memory.md",
    # 基准
    "benchmark_map": {".NS": "^NSEI", ".T": "^N225", ... "": "SPY"},
}
```

---

## 十、与 QuantNodes 的功能映射

| TradingAgents 功能 | QuantNodes 对应 | 差距 |
|-------------------|----------------|------|
| 10 个 @tool (数据获取) | 26 个 agent tools | QuantNodes 更丰富 |
| Vendor Router (4 vendors) | 6 数据库后端 (SQLite/DuckDB/MySQL/ClickHouse/CSV/Parquet) | 各有侧重 |
| yfinance 价格数据 | 需自接 | 🔴 缺失 |
| FRED 宏观数据 | ❌ 无 | 🔴 缺失 |
| Polymarket 预测 | ❌ 无 | 🟡 可选 |
| Market Validation (防幻觉) | ❌ 无 | 🔴 缺失 |
| 5-tier Rating | ❌ 无 | 🔴 缺失 |
| Structured Output (Pydantic) | ❌ 无 | 🔴 缺失 |
| Bull/Bear Debate | ❌ 无 | 🔴 缺失 |
| Risk 3-way Debate | RiskNode (规则引擎) | 机制不同 |
| Memory Log (append-only MD) | Knowledge Base (TF-IDF + lineage) | 机制不同 |
| Deferred Reflection | ❌ 无 | 🔴 缺失 |
| 20+ Provider LLM | 2 provider (OpenAI + Azure) | 🔴 缺失 |
| CLI 向导 (8 步) | `quantnodes` CLI | QuantNodes 更完整 |
| Multi-vendor fallback | Factory + Adapter (设计模式) | QuantNodes 已有类似 |

### 关键差异总结

| 维度 | TradingAgents | QuantNodes |
|------|--------------|------------|
| **定位** | 决策流水线 (单品种) | 研究框架 (多品种) |
| **回测** | 无 (仅 deferred reflection) | 完整引擎 |
| **因子** | 无 | 317+ 算子 + MCTS |
| **风控** | LLM 对话 (非量化) | 可组合风险链 (量化) |
| **数据验证** | Market snapshot 验证 | 无 |
| **LLM 集成** | 20+ Provider | 2 Provider |
| **结构化** | Pydantic schema | 无 |
| **自我进化** | Memory log + reflection | Knowledge base (静态) |
| **辩论** | Bull/Bear + Risk 3-way | 无 |

---

## 十一、关键设计模式

| 模式 | 实现位置 | 说明 |
|------|---------|------|
| **Factory** | `llm_clients/factory.py` | 20+ provider 统一创建 |
| **Strategy** | `dataflows/interface.py` | Vendor 回退链 |
| **Chain of Responsibility** | Vendor router | 依次尝试 vendor |
| **Null Object** | (未直接使用, 但 `bind_structured` 返回 None) | - |
| **Decorator** | `llm/decorators.py` (LLM), `@tool` | LLM 装饰器, tool 包装 |
| **State Pattern** | `InvestDebateState` / `RiskDebateState` | 辩论状态机 |
| **Visitor** | (未使用) | - |
| **Adapter** | 每个 vendor 适配相同接口 | yfinance / Alpha Vantage / FRED |
| **Template Method** | `route_to_vendor()` | 模板方法处理回退 |
| **Memento** | Memory Log | 保存决策历史 |

---

## 十二、文件结构参考

```
tradingagents/
  default_config.py          # 配置 + env-var override
  reporting.py               # 报告生成 (write_report_tree)

  agents/
    schemas.py               # Pydantic structured-output schemas
    analysts/                # 4 analyst agents
    researchers/             # Bull/Bear researchers
    managers/                # Research Manager + Portfolio Manager
    risk_mgmt/               # 3-way risk debators
    trader/                  # Trader agent
    utils/
      agent_states.py        # TypedDict state definitions
      agent_utils.py         # Tool functions + instrument context
      core_stock_tools.py    # get_stock_data
      technical_indicators_tools.py
      fundamental_data_tools.py
      news_data_tools.py
      macro_data_tools.py
      prediction_markets_tools.py
      market_data_validation_tools.py
      memory.py              # TradingMemoryLog
      rating.py              # parse_rating
      structured.py          # bind_structured + fallback

  graph/
    trading_graph.py         # TradingAgentsGraph main class
    setup.py                 # StateGraph construction
    propagation.py           # Initial state creation
    conditional_logic.py     # Edge routing
    analyst_execution.py     # Analyst execution plan
    checkpointer.py          # SQLite checkpointing
    reflection.py            # Deferred reflection
    signal_processing.py     # Rating extraction

  llm_clients/
    base_client.py           # BaseLLMClient + normalize_content
    factory.py               # create_llm_client() factory
    openai_client.py         # 16 OpenAI-compatible providers
    anthropic_client.py      # Native Anthropic
    google_client.py         # Native Google
    azure_client.py          # Native Azure
    bedrock_client.py        # Native Bedrock
    capabilities.py          # Per-model capabilities
    model_catalog.py         # Known model lists
    validators.py            # Model validation
    api_key_env.py           # API key env mapping

  dataflows/
    interface.py             # VENDOR_METHODS routing
    y_finance.py             # yfinance implementations
    alpha_vantage_common.py  # Rate limiting
    alpha_vantage_stock.py
    alpha_vantage_indicator.py
    alpha_vantage_fundamentals.py
    alpha_vantage_news.py
    fred.py                  # FRED macro data
    polymarket.py            # Polymarket prediction markets
    market_data_validator.py # Deterministic validation
    symbol_utils.py          # Symbol normalization
    stockstats_utils.py      # stockstats wrapper
    errors.py                # VendorError hierarchy

cli/
  main.py                    # Typer-based 8-step wizard

main.py                      # 16-line programmatic entry
Dockerfile
docker-compose.yml
pyproject.toml
```

---

## 十三、可借鉴的功能优先级

按价值密度排序：

| 优先级 | 功能 | 价值 | 集成方式 |
|--------|------|------|---------|
| **P0** | Structured Output | 一切决策自动化的基础 | 新增 Pydantic schema + LLM mixin |
| **P0** | Market Validation | 防 LLM 幻觉价格/指标 | 新增 verification tool |
| **P1** | Deferred Reflection | 自我进化的核心 | 增强 knowledge base |
| **P1** | Dual-Model Architecture | 成本控制 | 拆分 deep/quick 调用 |
| **P2** | Multi-Agent Debate | 决策质量提升 | 新增 DebateTool |
| **P2** | Multi-Provider LLM | 灵活性 | 复用 ProviderSpec 注册表 |
| **P2** | Vendor Router | 数据源冗余 | 多数据源回退链 |
| **P3** | FRED/Polymarket 接入 | 宏观+预测数据 | 新增数据 vendor |
| **P3** | CLI 向导改进 | UX 提升 | 扩展现有 CLI |

---

## 十四、待决策问题 (持续)

| # | 问题 | 选项 |
|---|------|------|
| 1 | 辩论轮次默认值 | 1 轮 (TA 默认) vs 2-3 轮 |
| 2 | Phase 优先级 | P0 基础能力 vs 直接辩论框架 |
| 3 | LLM Provider | 是否立即支持 DeepSeek/Qwen 等 |
| 4 | 最小验证 | 选 3 因子对比有/无辩论 |
| 5 | 数据 vendor | 是否引入 yfinance / FRED / Polymarket |

---

## 十五、相关文档

| 文档 | 内容 |
|------|------|
| docs/28 | TradingAgents 调研报告 (架构/Prompt/Schemas/Provider) |
| docs/29 | TradingAgents 核心能力集成计划 (7 项能力) |
| docs/30 | TradingAgents 集成讨论总结 (价值/用法/增益/进化) |
| **docs/31** | **TradingAgents 功能深度梳理 (本文档)** |
