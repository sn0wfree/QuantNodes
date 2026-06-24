# Table 4 复现规划文档

> **版本**：v1.0（设计阶段）
> **日期**：2026-06-24
> **目标版本**：v2.10.0
> **状态**：Stage 1（Mock）准备开始，Stage 2（真实）并行

---

## 1. 背景与目标

### 1.1 论文来源

[Alpha-GPT 论文](https://arxiv.org/abs/2308.00016)（Wang et al., 2023, EMNLP 2025 Demo）的核心实验结果 Table 4 证明了：

> 用 GPT-4 + 量化 prompt + OperatorVocab 工具生成的 alpha 因子，在 Rank IC、IC IR、# Factors 三个维度上显著优于传统手工因子和纯 LLM（无工具）方法。

### 1.2 复现目标

将论文 Table 4 的实验迁移到 QuantNodes 项目，用我们的 **OperatorVocab（162 算子）** + **PolarsAlphaCalculator** + **AlphaGptWorkflow（5 智能体）**，在 A 股数据（沪深 300）上重现 3 组对比：

| 组别 | 方法 | 实现路径 |
|------|------|----------|
| **G1 (Handcrafted)** | 100 手工公式（Alpha 101 + 经典动量/反转/波动率）| 静态公式集合 |
| **G2 (LLM-Only)** | 50 公式（LLM 直生成，无 OperatorVocab 工具）| prompt-only |
| **G3 (Alpha-GPT)** | 200 公式（5 智能体 + OperatorVocab）| `AlphaGptWorkflow.run()` |

### 1.3 验收指标（论文 Table 4 对齐）

| 指标 | 计算 | G3 期望趋势 |
|------|------|------------|
| **Rank IC mean** | per-date Spearman corr 均值 | G3 > G1 > G2 |
| **Rank IC IR** | IC mean / IC std | G3 > G1 > G2 |
| **# Successful Factors** | IC > 0.02 的公式数 | G3 > G1 > G2 |
| **Category Coverage** | 6 类（mom/rev/val/qua/vol/liq）覆盖度 | G3 ≈ 1.0（6/6 类）< G1 ≈ 1.0 |

---

## 2. 关键约束：两阶段并行 + 接口契约

### 2.1 为什么需要接口契约

| 阶段 | 实现 | 数据源 | LLM |
|------|------|--------|-----|
| **Stage 1 (Mock)** | 同步进行 | 合成 GBM + trend injection | hardcoded mock response |
| **Stage 2 (Real)** | 并行启动 | iFinD 5 年历史 parquet | MiniMax API |

两阶段**必须并行**（不串行等待），但**不能让 Stage 2 等 Stage 1 全部完成**。因此需要：

> **接口契约**：在 Stage 1 #1.1 一次性定义，Stage 1 后续任务按契约实现 mock，Stage 2 任务按契约实现 real，两边互不等待。

### 2.2 接口契约的 4 个核心抽象

```
┌──────────────────────────────────────────────────────────────┐
│                      Table4Runner                             │
│  (主入口：组合 DataLoader + 3 Baselines + Evaluator)          │
└──────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  Data    │  │ Baseline │  │ Baseline │  │ Baseline │
   │  Loader  │  │  G1      │  │  G2      │  │  G3      │
   │ (ABC)    │  │ (ABC)    │  │ (ABC)    │  │ (ABC)    │
   └──────────┘  └──────────┘  └──────────┘  └──────────┘
   Mock/IFinD    Static         Mock/MiniMax    Mock/MiniMax
                                +LLMClient     +AlphaGptWorkflow
         │
         ▼
   ┌──────────┐
   │ Evaluator│
   │  (ABC)   │
   └──────────┘
   Mock/Polars
```

每个抽象的接口契约：

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Return pl.DataFrame with schema:
        required: date (Date), code (Utf8), open/high/low/close/vol (Float64)
        optional: amount, industry, adj_factor, vwap
        """

class Baseline(ABC):
    @abstractmethod
    def name(self) -> str:  # "G1_handcrafted" | "G2_llm_only" | "G3_alpha_gpt"

    @abstractmethod
    def generate_factors(self) -> List[FactorSpec]:
        """Generate factor specs (no evaluation)."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Extra info: llm_provider, iterations, pool_size, etc."""

class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        factors: List[FactorSpec],
        data: pl.DataFrame,
        forward_returns: List[int] = (1, 5, 20),
    ) -> List[FactorMetrics]:
        """Compute per-factor IC/IR/hit_rate."""
```

### 2.3 数据契约

```python
# FactorSpec: 一个因子的完整规格（3 个 baseline 共用输出格式）
@dataclass
class FactorSpec:
    factor_id: str                  # e.g. "G1-001", "G3-M3R2-007"
    formula: str                    # polars 表达式字符串
    source: str                     # "G1" | "G2" | "G3"
    round_idx: Optional[int] = None # G3 用，其他 None
    category: Optional[str] = None  # 6 类: momentum/reversal/value/quality/volatility/liquidity
    metadata: Dict[str, Any] = field(default_factory=dict)


# FactorMetrics: 单个因子的评估结果（3 个 baseline 共用）
@dataclass
class FactorMetrics:
    factor_id: str
    formula: str
    status: str                     # "success" | "failed"
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    hit_rate: float = 0.0
    n_dates: int = 0
    n_stocks: int = 0
    error_msg: Optional[str] = None


# Table4GroupResult: 单个 baseline 汇总
@dataclass
class Table4GroupResult:
    name: str                       # "G1_handcrafted" / "G2_llm_only" / "G3_alpha_gpt"
    n_factors_total: int = 0
    n_factors_successful: int = 0
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    hit_rate: float = 0.0
    category_coverage: Dict[str, float] = field(default_factory=dict)
    factors: List[FactorMetrics] = field(default_factory=list)


# Table4Report: 3 组对比完整报告
@dataclass
class Table4Report:
    experiment_date: str
    data_meta: Dict[str, Any]
    groups: Dict[str, Table4GroupResult] = field(default_factory=dict)
    paper_comparison: Optional[Dict[str, Any]] = None
```

---

## 3. Stage 1（Mock）实施计划

### 3.1 目标

不依赖真实数据 / 真实 LLM，跑通完整 pipeline，验证接口契约可用性。

### 3.2 复用现有基础设施（85% 复用率）

Stage 1 最大化复用 QuantAlpha M1-M6 已实现模块：

| 现有模块 | Stage 1 复用方式 |
|----------|------------------|
| `OperatorVocab`（162 算子 + 12 元数据）| G1 动态生成 100 公式（不硬编码）|
| `PolarsAlphaCalculator`（M4）| PolarsAlphaCalculatorEvaluator 内部直接调 |
| `alpha_evaluate` tool（M5）| PolarsAlphaCalculatorEvaluator 包这一层 |
| `AlphaGptWorkflow`（M5）| G3 baseline 包一层 + mock LLM 注入 |
| `_mock_llm_response`（M5）| G2 baseline 直接调 |
| `state.py` dataclass 风格（5 个）| 借鉴 contracts.py 的 4 个 dataclass |
| `data_loader.py`（factor_test/utils）| 借鉴 MockDataLoader 接口模式 |
| `alpha-gpt` CLI（commands/alpha.py）| 借鉴 CLI argparse 模式 |
| `pandas 3.0 兼容`（AGENTS.md 规则）| 不依赖 applymap，使用 polars 主导 |
| `优雅降级`（nanobot 可选）| 无 nanobot 时走 mock 路径 |

**Stage 1 不重新实现**：metrics 计算（用 alpha_evaluate）、LLM 客户端（用 mock）、评估器核心（用 PolarsAlphaCalculator）、工作流编排（用 AlphaGptWorkflow）。

### 3.3 任务清单（重构 7 个，从原 11 压缩 50%）

| # | 任务 | 工作量 | 输出文件 | 复用现有 |
|---|------|------:|----------|---------|
| 2.1 | 接口契约（4 dataclass + 4 ABC）| 0.05d | `evaluation/contracts.py` | 借鉴 `state.py` 风格 |
| 2.2 | MockDataLoader（500 票 GBM + trend）| 0.1d | `evaluation/mock_data_loader.py` | 借鉴 `data_loader.py` 接口 |
| 2.3 | Table4Runner（聚合 + save + md）| 0.15d | `evaluation/runner.py` | - |
| 2.4 | PolarsAlphaCalculatorEvaluator | 0.1d | `evaluation/evaluators/polars_evaluator.py` | **复用 alpha_evaluate tool** |
| 2.5 | G1 Handcrafted（动态生成 100 公式）| 0.15d | `evaluation/baselines/g1_handcrafted.py` | **动态 OperatorVocab 模板** |
| 2.6 | G2 LLM-Only（mock LLM 直生成 50 字符串）| 0.1d | `evaluation/baselines/g2_llm_only.py` | **复用 _mock_llm_response** |
| 2.7 | G3 AlphaGptBaseline（包 AlphaGptWorkflow）| 0.1d | `evaluation/baselines/g3_alpha_gpt.py` | **直接调 AlphaGptWorkflow** |
| 2.8 | mock CLI 脚本 | 0.05d | `scripts/reproduce_table4_mock.py` | 借鉴 alpha-gpt CLI 模式 |
| 2.9 | 单元 + 集成测试（5 文件，~14 测试）| 0.25d | `tests/quant_alpha/test_table4_*.py` | 复用现有 alpha_evaluate fixture |
| 2.10 | 文档（增量）| 0.1d | `docs/quant_alpha/table4_reproduction.md` | - |
| **合计** | | **1.15d** | | **~85% 复用** |

vs 原 11 任务（2.1d）：节省 50% 工作量 + ~50% 代码行数。

### 3.4 G3 baseline 决策：保持 AlphaGptWorkflow（不直接调 nanobot.run）

**理由**：
1. AlphaGptWorkflow（M5）已封装 5 智能体编排 + 5 轮迭代 + 5 × N 次 spawn
2. NanobotLLMWrapper（M6）已实现同步 LLM 接口，可注入 nanobot.Agent 作为 LLM client
3. Stage 1 直接复用现有 `_mock_llm_response`，**0 新增代码**
4. Stage 2 通过 `NanobotLLMWrapper(agent)` 注入到 AlphaGptWorkflow
5. 如果直接调 `nanobot.run()`，需要重写 ~500 行 5 智能体编排代码（与 AlphaGptWorkflow 重复）

### 3.5 文件结构

```
QuantNodes/research/quant_alpha/evaluation/
├── __init__.py
├── contracts.py             # 接口契约
├── metrics.py               # IC/IR 计算
├── mock_data_loader.py      # Stage 1 only
├── mock_evaluator.py        # Stage 1 only
├── runner.py                # Table4Runner
└── baselines/
    ├── __init__.py
    ├── g1_handcrafted.py     # 100 公式
    ├── g2_llm_only.py        # mock LLM
    └── g3_alpha_gpt.py       # AlphaGptWorkflow 包装

scripts/
└── reproduce_table4_mock.py   # CLI

tests/quant_alpha/
├── test_table4_contracts.py
├── test_table4_metrics.py
├── test_table4_g1_baseline.py
├── test_table4_g2_baseline.py
├── test_table4_g3_alpha_gpt.py
└── test_table4_end_to_end.py
```

### 3.4 Mock 数据生成策略

```python
def generate_mock_market(n_stocks: int = 500, n_days: int = 500) -> pl.DataFrame:
    """生成全 A 股 mock 行情（subset 500 票 × 500 日，模拟真实数据规模 1/10）
    
    - 用 GBM 模拟 500 票 × 500 日（更接近真实数据规模，便于 Stage 2 性能评估）
    - 注入 trend + momentum 信号（保证 G1 也有信号）
    - 返回 schema 完整的 polars.DataFrame
    - 字段：date, code, open, high, low, close, vol, amount, industry
    """
    np.random.seed(42)
    # 注入已知信号:
    # - 20-日动量因子 ic ≈ 0.04
    # - 5-日反转因子 ic ≈ 0.02
    # - 波动率因子 ic ≈ 0.03
```

**为什么用 500 票而不是 30 票**：
- Stage 2 会用全 A (~5000 票)，Stage 1 用 500 票可在 ~5 分钟内跑完
- 500 票的因子 IC 估计比 30 票稳定（IC std 下降 ~4×）
- Polars 在 500 票 × 500 日上能真实测出 polars vs pandas 性能差距

### 3.5 验收标准

| 项 | 阈值 |
|----|------|
| Mock pipeline 跑通 | `python scripts/reproduce_table4_mock.py` 输出有效 JSON |
| Mock 数据规模 | 500 票 × 500 日 ≈ 25 万行（接近真实数据 1/10）|
| G3 > G1 > G2 (mock 趋势) | ✅ 期望出现（即使 IC 绝对值低）|
| 计算时间 | 单组评估 ≤ 5 分钟（500 票规模）|
| 测试 | ≥ 5317 passed (5303 + ~14 new，比原 17 少 3 因复用 fixture) |
| 文档 | ✅ `table4_reproduction.md` 包含 mock 结果 + 接口约定 |
| AGENTS.md | **不动**（按用户约定，独立 PR 处理）|

### 3.6 Mock 阶段不做什么

- ❌ 不拉真实数据
- ❌ 不调真实 LLM API
- ❌ 不 push 到 origin
- ❌ 不打 tag（v2.10 release 时）
- ❌ 不与论文 Table 4 数字精确对比（仅趋势对比）

---

## 4. Stage 2（Real）实施计划

### 4.1 目标

替换 Mock 实现为 iFinD 数据 + MiniMax LLM，验证完整 Table 4 复现。

### 4.2 任务清单（10 个，Stage 2 主要复用 Stage 1 接口 + 替换 mock 实现）

| # | 任务 | 工作量 | 输出文件 | 复用 Stage 1 |
|---|------|------:|----------|-------------|
| 2.1 | iFinD DataLoader（复用 `factor_test/ifind_db/fetcher.py`）| 0.3d | `evaluation/ifind_data_loader.py` | **MockDataLoader 接口直接复用** |
| 2.2 | 5 年历史数据加载脚本（全 A）| 0.3d | `scripts/table4/load_full_a_history.py` | - |
| 2.3 | Stage 1 PolarsAlphaCalculatorEvaluator 直接复用 | 0d | - | **无需新代码** |
| 2.4 | MiniMax LLM provider 包装（参考 `ai/llm/openai.py` 模式）| 0.3d | `QuantNodes/ai/llm/minimax.py` | - |
| 2.5 | G2 baseline 替换 mock LLM → MiniMax | 0.2d | `evaluation/baselines/g2_llm_only.py` | **同文件改 LLM client** |
| 2.6 | G3 baseline 注入 NanobotLLMWrapper(MiniMax) | 0.2d | `evaluation/baselines/g3_alpha_gpt.py` | **同文件改 llm_client** |
| 2.7 | 主入口 `reproduce_table4.py` | 0.2d | `scripts/reproduce_table4.py` | - |
| 2.8 | 论文 Table 4 对比 + 报告 | 0.3d | `docs/quant_alpha/table4_paper_comparison.md` | - |
| 2.9 | Stage 2 集成测试 | 0.3d | `tests/research/test_table4_real.py` | - |
| 2.10 | v2.10 release prep（CHANGELOG + tag）| 0.2d | - | - |
| **合计** | | **2.3d** | |

### 4.3 数据需求

**全 A 股日线成交数据**（约 5000 票 × 5 年，~750 万行 parquet）

| 字段 | 类型 | 必需 | 用途 |
|------|------|------|------|
| date | Date | ✅ | 时间索引 |
| code | Utf8 | ✅ | 股票代码（如 SH600000, SZ000001）|
| open/high/low/close | Float64 | ✅ | OHLC |
| vol | Float64 | ✅ | 成交量（股）|
| amount | Float64 | ✅ | 成交额（元）|
| industry | Utf8 | optional | 行业分类（IndNeutralize 用）|
| vwap | Float64 | optional | 量价加权均价（Alpha 101 用）|
| adj_factor | Float64 | optional | 后复权因子（避免分红拆股失真）|
| float_share | Float64 | optional | 流通股本（换手率计算用）|
| is_st | Bool | optional | 是否 ST/*ST（filter 剔除用）|

**保存位置**：`data/cache/full_a_2019_2024.parquet`（不入 VCS）

**Universe 定义**：
- 初始全集：全 A 股 ~5000 票
- 流动性过滤：剔除日均成交额后 20% 分位（约 ~4000 票）
- ST 过滤：剔除 ST/*ST（~50 票）
- 实际计算宇宙：~3950 票

**预期计算成本**：
- 数据量：5000 × 1250 = 625 万行 × 200 公式 × 5 前瞻期 ≈ 6.25 亿次 IC 计算
- 本地 polars：~10-30 分钟（视 CPU）
- 不要 pandas apply loop（×100 慢）

### 4.4 LLM 需求

**MiniMax**（参考 `llmwikify` 的配置模式）

| 配置项 | 值 |
|--------|-----|
| Provider | MiniMax |
| API key env | `MINIMAX_API_KEY` |
| 默认 model | 暂用 MiniMax 默认（待确认具体 model 名）|
| 温度 | 0.7 (idea 阶段), 0.3 (formula 阶段) |

### 4.5 验收标准

| 项 | 阈值 |
|----|------|
| iFinD 拉取 5 年数据（全 A ~5000 票）| parquet ≥ 600 万行 |
| MiniMax LLM 调通 | G2/G3 输出 valid JSON |
| G3 Rank IC > G1 > G2 (real) | ✅ |
| 与论文 Table 4 差距 | ≤ 50%（标注"近似"）|
| 测试 | ≥ 5330 passed |
| v2.10 release tag | ✅ |

---

## 5. Stage 1 vs Stage 2 资源对应

| 概念 | Stage 1 (Mock) | Stage 2 (Real) | 接口统一 |
|------|---------------|----------------|----------|
| DataLoader | `MockDataLoader` GBM 生成 | `IFinDDataLoader` 读 parquet | `DataLoader.load()` |
| LLM Client | `mock_llm_response()` 函数 | `MiniMaxClient.complete()` | `LLMClient.complete()` |
| Evaluator | `MockEvaluator` 直生成 IC | `PolarsEvaluator` 用 M4 PolarsAlphaCalculator | `Evaluator.evaluate()` |
| G1 Baseline | 100 静态公式 | 同（公式集不变）| `Baseline.generate_factors()` |
| G2 Baseline | mock LLM → 50 公式 | MiniMax → 50 公式 | 同上 |
| G3 Baseline | mock LLM + AlphaGptWorkflow | MiniMax + AlphaGptWorkflow | 同上 |
| 主入口 | `reproduce_table4_mock.py` | `reproduce_table4.py` | `Table4Runner.run()` |

---

## 6. 并行策略

| 角色 | 任务 | 依赖 |
|------|------|------|
| **用户（数据）** | Stage 2 #2.1 #2.2 iFinD 拉取 | 无 |
| **用户（API key）** | Stage 2 #2.4 提供 MiniMax API key | 无 |
| **我（接口契约）** | Stage 1 #1.1 接口定义 | 必须先完成 |
| **我（Stage 1 mock）** | Stage 1 #1.2 - #1.11 | 依赖 #1.1 |
| **我（Stage 2 实现）** | Stage 2 #2.3 - #2.10 | 依赖 Stage 1 + 用户提供数据 + API key |

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 接口契约定义不当导致 Stage 2 重构 | 中 | 高 | 接口 review + 单元测试覆盖 |
| Mock 数据无信号 → 3 组 IC 都接近 0 | 中 | 中 | mock 用 GBM + trend injection |
| LLM mock 输出不符合 schema | 低 | 低 | schema validator + retry |
| Stage 1 #1.1 接口定义被推迟 | 中 | 高（阻塞）| 先完成 #1.1 再继续 |
| iFinD 拉数据失败/不全 | 中 | 高 | 多数据源 fallback |
| MiniMax API key 无效 | 中 | 高 | 检查 env + 测试 |
| 计算超时（5 年 × 300 票）| 低 | 低 | 数据切片并行 |
| Alpha-GPT 跑出来不如手工 | 中 | 中 | 调 iteration + pool size |

---

## 8. 时间线

```
Day 1: Stage 1 #1.1 - #1.5 (接口 + 数据 + 指标 + mock evaluator + G1)
Day 2: Stage 1 #1.6 - #1.10 (G2 + G3 + runner + CLI + 测试)
Day 3: Stage 1 #1.11 (文档) + Stage 2 #2.3 #2.4 (evaluator + MiniMax)
Day 4: Stage 2 #2.1 #2.2 (iFinD + 数据) + #2.5-#2.7 (G2/G3/runner real)
Day 5: Stage 2 #2.8 #2.9 #2.10 (对比 + 集成 + release)
```

并行：你（数据 + key）从 Day 1 开始并行准备，我 Stage 1 mock 跑完后立即切 Stage 2。

---

## 9. 待你确认

进入 Stage 1 实施前需要你确认：

- [ ] 接口契约（4 个 ABC + 4 个 dataclass）设计 OK
- [ ] Stage 1 11 个任务 + 工作量分配 OK
- [ ] Stage 2 10 个任务 + 工作量分配 OK
- [ ] Mock 数据规模（500 票 × 500 日）OK
- [ ] 数据契约 schema OK
- [ ] 测试策略（7 文件 + 集成）OK

确认后退出 Plan Mode 开始 Stage 1 #1.1 接口契约实施。