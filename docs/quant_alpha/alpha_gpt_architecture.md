# Alpha-GPT 架构设计

> **版本**：v1.0
> **日期**：2026-06-24
> **状态**：M5 PR 设计文档（doc-first 阶段）
> **适用项目**：QuantNodes v2.7.0+

---

## 1. 概述

Alpha-GPT 是 QuantNodes 的**第五个智能体编排范式**（继 factor-analyst / backtest-engineer / risk-manager / dream 之后），
用于**自动化因子挖掘**。基于 [Wang 2023 / EMNLP 2025 Demo](https://arxiv.org/abs/2308.00016) 的多智能体
对话框架，结合本项目的 5 大基础设施（OperatorVocab / MCTS / few-shot / PolarsAlphaCalculator / nanobot），
实现从自然语言目标到可交易 alpha 公式的端到端自动化。

### 1.1 核心目标

输入：「**捕捉 A 股反转效应**」
输出：「**Top 10 个有效反转因子公式 + IC/IR/Trading 回测指标**」

### 1.2 关键差异化

| vs 业界 | QuantNodes Alpha-GPT 优势 |
|---------|--------------------------|
| vs [Alpha-GPT 论文](https://arxiv.org/abs/2308.00016) | **集成 nanobot 多进程 spawn** + **Trading 回测** + **A 股适配** |
| vs [AlphaGen (KDD 2023)](https://github.com/rl-research/alphagen) | **LLM 而非 RL**，**单次 5 轮即可**，**无需 GPU** |
| vs [Alpha² 2024](https://arxiv.org/abs/2406.16505) | **零训练成本**，复用现有 162 算子 |
| vs [QuantaAlpha](https://github.com/quantalpha/quantalpha) | **5 智能体分阶段**，**显式 Reflector 反馈环** |

---

## 2. 5 智能体架构

### 2.1 总体流程

```
                 ┌──────────────────────────────────────────┐
                 │  AlphaGptWorkflow (Python 协调器)        │
                 │  - 状态：formula pool + IC 历史 + 轮次   │
                 │  - 调度：spawn 5 个 nanobot subagent     │
                 │  - 输出：final top-K + JSON              │
                 └────────────────┬─────────────────────────┘
                                  │
                                  ▼  spawn (nanobot 多进程)
   ┌─────────────────────── 5 智能体编排 ──────────────────────────┐
   │                                                                │
   │  [1] alpha-gpt-idea-generator                                 │
   │       输入：objective + 反思建议                               │
   │       输出：N 个 alpha 想法（JSON 列表）                       │
   │           ↓                                                    │
   │  [2] alpha-gpt-formula-translator                             │
   │       输入：想法 + OperatorVocab 算子清单                       │
   │       输出：polars 公式（可执行）                              │
   │           ↓                                                    │
   │  [3] alpha-gpt-evaluator  ← 调 alpha_evaluate + alpha_backtest│
   │       输入：公式 + 数据                                        │
   │       输出：IC / IR / Trading 指标                              │
   │           ↓                                                    │
   │  [4] alpha-gpt-reflector                                      │
   │       输入：评估结果                                           │
   │       输出：keep/mutate/drop verdicts + 下一轮建议              │
   │           ↓                                                    │
   │  [5] alpha-gpt-critic                                         │
   │       输入：所有轮次评估 + 反思                                 │
   │       输出：top-K final pool                                   │
   │                                                                │
   │  循环回到 [1]，直到 max_iterations 满足                       │
   └────────────────────────────────────────────────────────────────┘
```

### 2.2 角色职责

| Agent | 输入 | 输出 | 工具 | 输出格式 |
|-------|------|------|------|----------|
| **IdeaGenerator** | objective, reflector_suggestions | 想法列表 | 无 | JSON: `{ideas: [...]}` |
| **FormulaTranslator** | ideas, operators, columns | polars 公式 | 无 | JSON: `{formulas: [...]}` |
| **Evaluator** | formulas, data | IC/IR/Trading | alpha_evaluate, alpha_backtest | JSON: `{evaluations: [...]}` |
| **Reflector** | evaluations | verdicts + 改进建议 | 无 | JSON: `{verdicts, suggestions}` |
| **Critic** | all_history | final top-K | 无 | JSON: `{final_pool: [...]}` |

### 2.3 5 轮迭代设计

每轮 5 个 spawn = **25 spawn 总开销**（每 spawn ~3s 进程启动 ≈ 75s 总开销）。

| 轮次 | 目标 | 期望产出 |
|------|------|----------|
| Round 1 | **基线探索** — 覆盖 6 大 category，每类 1-2 个想法 | 10 个 baseline 公式 |
| Round 2 | **变异 + 反馈采纳** — Reflector 标记 mutate 的公式做窗口/算子调整 | 8-10 个改进公式 |
| Round 3 | **正交化** — 强制要求新想法与 top-5 mutual_IC < 0.5 | 8-10 个独立公式 |
| Round 4 | **长周期 / 跨 category** — 探索 60-120 日窗口 + 新组合 | 6-8 个差异化公式 |
| Round 5 | **Trading 验证** — 仅对 top-10 候选做完整 backtest | 最终 final pool |

### 2.4 Subagent 注册

5 个 subagent 通过 `.agent/agents/alpha-gpt-*.md` 定义，并在 `QuantNodes/agent/agents/definition.py` 注册：

```python
"alpha-gpt-idea-generator": AgentDefinition(
    id="alpha-gpt-idea-generator",
    mode="subagent",
    tools_denied={"*"},  # 纯文本生成，无工具
),
"alpha-gpt-formula-translator": AgentDefinition(
    id="alpha-gpt-formula-translator",
    mode="subagent",
    tools_denied={"*"},
),
"alpha-gpt-evaluator": AgentDefinition(
    id="alpha-gpt-evaluator",
    mode="subagent",
    tools_allowed={"alpha_evaluate", "alpha_backtest", "read", "glob"},
),
"alpha-gpt-reflector": AgentDefinition(
    id="alpha-gpt-reflector",
    mode="subagent",
    tools_denied={"*"},
),
"alpha-gpt-critic": AgentDefinition(
    id="alpha-gpt-critic",
    mode="subagent",
    tools_denied={"*"},
),
```

---

## 3. 复用基础设施

### 3.1 复用率

| 组件 | 复用 | 说明 |
|------|------|------|
| LLM 调度 | **100%** | nanobot upstream（OpenAI / DeepSeek / Qwen via base_url） |
| Agent 框架 | **100%** | `QuantNodes.agent.Agent` + nanobot `spawn` 工具 |
| 工具抽象 | **100%** | `nanobot.agent.tools.base.Tool` |
| IC 评估 | **100%** | M4 `PolarsAlphaCalculator`（7 方法） |
| Few-shot | **100%** | M3 `alpha101_design.few_shot_examples` + `alpha158_design.few_shot_examples` |
| 算子清单 | **100%** | M1 `OperatorVocab.list_vocab_operators()` |
| Trading 回测 | **80%** | 复用 `BacktestTool` + `FactorNode` + `MAStrategyNode` |
| **自建** | **< 30%** | 5 subagent .md + 2 工具 + 1 协调器 + 1 parser |

### 3.2 关键依赖关系

```
                    ┌────────────────────────────────────────┐
                    │  AlphaGptWorkflow (新)                  │
                    │  - 5 轮主循环                            │
                    │  - spawn 协调                            │
                    │  - 状态管理                              │
                    └─────────┬──────────────────────────────┘
                              │
                              ├─→ nanobot spawn (复用)
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
   ┌─────────────────────┐    ┌────────────────────────────┐
   │ 5 Subagent .md      │    │ 2 New Tools                 │
   │ (新)                │    │ (新)                        │
   │ - idea-generator    │    │ - alpha_evaluate            │
   │ - formula-translator│    │ - alpha_backtest            │
   │ - evaluator         │    └─────────┬──────────────────┘
   │ - reflector         │              │
   │ - critic            │              │
   └─────────┬───────────┘              │
             │                           │
             ▼                           ▼
   ┌────────────────────────────────────────────────────┐
   │  复用基设施                                          │
   │  - nanobot Agent + LLMClientBase (OpenAI/DeepSeek) │
   │  - M4 PolarsAlphaCalculator (7 IC methods)         │
   │  - M3 few-shot examples (20 个)                     │
   │  - M1 OperatorVocab (162 算子 + 元数据)             │
   │  - 现有 BacktestTool + FactorNode                   │
   └────────────────────────────────────────────────────┘
```

---

## 4. 关键决策与原理

### 4.1 为什么不用新 LLM client？

复用 nanobot upstream 的 4 个理由：
1. **成本**：避免重复实现 OpenAI / DeepSeek / Qwen 协议
2. **一致性**：与项目内 `factor-analyst` / `backtest-engineer` 等 agent 共享同一 LLM 配置
3. **流式**：nanobot 已支持 SSE / WebSocket 流式输出（v2.x 协议）
4. **可观测**：nanobot 的 hooks / token counting / logging decorators 直接可用

### 4.2 为什么用 multi-process spawn？

vs 单进程 prompt 切换：

| 维度 | multi-process spawn | 单进程 prompt 切换 |
|------|--------------------|--------------------|
| Context 隔离 | ✅ 完全独立 | ❌ 共享同一 context |
| 并行能力 | ✅ 真并行 | ❌ 串行 |
| Token 成本 | ❌ 每次重启 context（浪费） | ✅ 复用 context |
| 复杂度 | ❌ 高（进程协调） | ✅ 低 |
| 调试 | ✅ 易（每个 subagent 独立日志）| ❌ 难（混在一起） |

**选择 multi-process**：每个 subagent 的 system prompt 完全不同（5 套独立的 few-shot + 工具集 + 约束），
context 隔离的收益大于 token 重启成本。

### 4.3 为什么 JSON Schema 三层降级？

```python
def parse_formula(llm_output: str) -> FormulaDict:
    # Layer 1: JSON Schema validation
    try:
        return FormulaDict.parse_raw(llm_output)
    except ValidationError:
        pass
    # Layer 2: Regex extraction
    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    if match:
        try:
            return FormulaDict.parse_raw(match.group())
        except ValidationError:
            pass
    # Layer 3: Retry LLM
    raise FormulaParseError(f"Cannot parse: {llm_output[:200]}")
```

理由：
- **零新依赖**（不引 instructor / outlines）
- **3 重兜底**：严格 → 宽松 → 重试
- **可观测**：每层失败都有明确日志

### 4.4 Trading 回测为何可选？

```bash
quantnodes alpha-gpt ... --backtest
```

- 默认**禁用**（避免 50 个公式 × 5-30s/回测 = 25 分钟延迟）
- 启用后仅对 **top-K 候选**做回测（默认 K=10）
- 回测结果作为 Critic 的次要评分依据（Sharpe 权重 0.2）

---

## 5. 与 M1-M4 的衔接

### 5.1 M1 OperatorVocab → IdeaGenerator

`list_vocab_operators()` 提供 LLM 可用算子清单（按 category 分组），注入 IdeaGenerator 的 system prompt。
避免 LLM 幻觉出不存在算子（如 `ts_macd`、`vwap_corr`）。

### 5.2 M3 Few-shot → FormulaTranslator

`alpha101_design.few_shot_examples`（10 个）+ `alpha158_design.few_shot_examples`（10 个）
直接作为 FormulaTranslator 的 few-shot context。

### 5.3 M4 PolarsAlphaCalculator → Evaluator

`alpha_evaluate` 工具是 PolarsAlphaCalculator 的 thin wrapper：

```python
class AlphaEvaluateTool(Tool):
    async def execute(self, formula: str, data_path: str, ...):
        calc = PolarsAlphaCalculator(...)
        ic = calc.calc_single_IC_ret(formula, ...)
        return {"ic_mean": float(np.nanmean(ic)), "ir": float(np.nanmean(ic)/np.nanstd(ic))}
```

### 5.4 M2 MCTSSearch → Alpha-GPT (alternative backend)

用户可选：
- **`--backend llm`**（默认）：用 Alpha-GPT 5 智能体
- **`--backend mcts`**：用 M2 MCTSSearch（更便宜但无 LLM 创意）

CLI：
```bash
quantnodes alpha-gpt --backend llm --iterations 5   # Alpha-GPT
quantnodes alpha-gpt --backend mcts --iterations 50  # MCTS
```

---

## 6. 数据流与状态

### 6.1 Workflow 状态

```python
@dataclass
class AlphaGptState:
    round: int = 0
    objective: str
    all_evaluations: List[Evaluation] = field(default_factory=list)
    all_reflections: List[Reflection] = field(default_factory=list)
    formula_pool: Dict[str, Formula] = field(default_factory=dict)
    current_top_k: List[Formula] = field(default_factory=list)
```

### 6.2 一轮完整数据流

```
Round N 开始
   ↓
[Coordinator] spawn idea-generator
   ↓ (JSON: {ideas: [...]})
[Coordinator] spawn formula-translator
   ↓ (JSON: {formulas: [...]})
[Coordinator] spawn evaluator (调 alpha_evaluate 工具)
   ↓ (JSON: {evaluations: [...]})
[Coordinator] spawn reflector
   ↓ (JSON: {verdicts, suggestions})
[Coordinator] 更新 state.all_evaluations / reflections
   ↓
Round N+1 (继续) 或 Critic spawn → final output
```

### 6.3 持久化

- **中间状态**：内存（不持久化，崩溃重启动）
- **最终结果**：JSON 文件 / DB（CLI 默认 `alpha_pool.json`，API 默认 SQLite）

---

## 7. 失败模式与降级

| 失败 | 检测 | 降级策略 |
|------|------|----------|
| nanobot spawn 超时 | 30s timeout | 重试 1 次，跳过该 subagent |
| 公式执行错误 | alpha_evaluate 捕获 | 标记 `status="failed"`，Reflector 建议改算子 |
| 全部公式失败 | Evaluator 返回全 failed | 工作流终止，返回空 pool + 错误报告 |
| JSON 解析失败 | 3 层降级全失败 | 重试 LLM（最多 3 次）|
| LLM API 限流 | 429 response | TokenCountingClient + 指数退避 |
| 数据缺失 | column 不存在 | Evaluator 返回 invalid，Reflector 标注 data issue |

---

## 8. 性能与成本

### 8.1 单次工作流耗时（5 轮）

| 阶段 | 耗时 |
|------|------:|
| 25 × spawn 启动 | ~75s |
| 50 × LLM 调用（5×5×2 = 50 idea/translate/reflection + 10 evaluator + 5 critic）| ~150s |
| 50 × IC 评估（polars 本地计算）| ~30s |
| 10 × Trading 回测（仅 top-10）| ~50s |
| Critic 综合排序 | ~10s |
| **合计** | **~315s（5.3 分钟）** |

### 8.2 Token 成本估算（DeepSeek-V3 价格）

| 阶段 | 输入 tokens | 输出 tokens | 成本 |
|------|----------:|----------:|-----:|
| 5 × idea | 2k × 5 = 10k | 1k × 5 = 5k | $0.014 |
| 5 × formula | 3k × 5 = 15k | 1k × 5 = 5k | $0.019 |
| 5 × evaluator | 4k × 5 = 20k | 1k × 5 = 5k | $0.024 |
| 5 × reflector | 5k × 5 = 25k | 1k × 5 = 5k | $0.028 |
| 1 × critic | 10k | 2k | $0.010 |
| **合计** | **80k** | **22k** | **~$0.10** |

**单次工作流成本**：约 0.10 美元（DeepSeek-V3）/ 0.50 美元（GPT-4o）/ 0.05 美元（Qwen）

---

## 9. 测试策略

| 层级 | 测试内容 | 用例数 |
|------|----------|------:|
| 单元 | parser / tools / state | 25 |
| 集成 | 单个 subagent（mock LLM） | 15 |
| 工作流 | 完整 5 轮（mock 所有 LLM） | 8 |
| E2E | mock nanobot + mock 数据 | 5 |
| **合计** | | **53** |

测试覆盖目标：≥ 80%（per 质量门栏）。

---

## 10. 后续路线

| 阶段 | 内容 | 价值 |
|------|------|------|
| **v2.7.0** | 本文档描述的全部 | M5-M6 PR |
| v2.8 | Table 4 复现（沪深 300 全 A 股）| 论文复现 |
| v2.9 | WebSocket 流式输出 + 前端可视化 | 用户体验 |
| v3.0 | Multi-Objective 优化（IC + Sharpe + 换手） | 实战化 |
| v3.1 | 自定义策略池（auto-stacking） | 高阶 |

---

> **最后更新**：2026-06-24
> **作者**：QuantNodes Agent
> **状态**：M5 PR 设计文档，待实现
