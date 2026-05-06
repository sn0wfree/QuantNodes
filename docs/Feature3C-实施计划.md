# 功能3C 实施方案：AutoResearch 自动因子挖掘

> 文档版本: v1.0
> 创建日期: 2026-05-07
> 状态: 待实施

---

## 一、背景

功能3C 是量化研究的自动化层，基于模板枚举 + MCTS 搜索自动生成、验证、去重因子，存入 Wiki 因子库。

**依赖链**：
```
功能3A: WikiFactorProxy (wiki.py) ✅
    ↑
功能3C: AutoResearch (auto_researcher.py)
    ├── factor_miner.py      # 因子挖掘 (模板枚举)
    ├── factor_evaluator.py  # 6维度评估
    └── mcts_search.py       # MCTS 搜索树 (阶段2)
```

**设计原则**：
- 确定性优先：模板枚举可复现、可调试
- 6维度评估：收益、稳定性、分散度、换手率、单调性、覆盖率
- 分阶段实施：阶段1(确定性基线) → 阶段2(MCTS) → 阶段3(LLM增强)

---

## 二、文件结构

```
QuantNodes/research/
├── __init__.py              # 更新导出
├── wiki.py                  # 功能3A (已有)
├── auto_researcher.py       # 编排器 (3阶段)
├── factor_miner.py          # 模板枚举 + 公式生成
├── factor_evaluator.py      # 6维度评估 + 相关性去重
├── mcts_search.py           # MCTS 搜索树 (阶段2)
└── README.md                # 更新
```

---

## 三、数据模型

### 3.1 FactorCandidate — 候选因子

```python
@dataclass
class FactorCandidate:
    name: str                           # 自动生成的因子名
    formula: str                        # Polars 表达式字符串
    description: str                    # 人类可读描述
    operators_used: List[str]           # 使用的算子列表
    category: FactorCategory            # 推断的分类
    template_name: str = ""             # 来源模板名
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.2 FactorEvaluationResult — 6维度评估结果

```python
@dataclass
class FactorEvaluationResult:
    candidate: FactorCandidate
    factor_values: Any                  # pl.Series 因子值

    # 维度1: 收益
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    rank_ic_mean: float = 0.0

    # 维度2: 稳定性
    rolling_ic_mean: float = 0.0        # 滚动IC均值
    rolling_ic_std: float = 0.0         # 滚动IC波动
    stability_score: float = 0.0        # 0-1

    # 维度3: 分散度
    avg_corr_with_existing: float = 0.0 # 与已有因子平均相关性
    diversification_score: float = 1.0  # 0-1, 1=完全独立

    # 维度4: 换手率
    turnover: float = 0.0               # 排名变化率
    turnover_cost: float = 0.0          # 估计交易成本

    # 维度5: 单调性
    group_returns: List[float] = field(default_factory=list)  # 5组分位收益
    monotonicity_score: float = 0.0     # 0-1

    # 维度6: 覆盖率
    coverage: float = 0.0               # 非空比例

    # 综合判定
    is_valid: bool = False
    fail_reasons: List[str] = field(default_factory=list)
    overall_score: float = 0.0          # 综合评分 (加权)
```

### 3.3 MiningConfig — 挖掘配置

```python
@dataclass
class MiningConfig:
    # 数据
    input_columns: List[str] = field(default_factory=lambda: ["close", "open", "high", "low", "vol"])
    date_column: str = "date"
    code_column: str = "code"
    forward_return_column: str = "forward_return"

    # 模板
    template_categories: List[str] = None  # None=全部
    max_depth: int = 3                      # 表达式最大深度
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # 评估阈值
    ic_threshold: float = 0.03
    icir_threshold: float = 0.5
    stability_threshold: float = 0.6
    corr_threshold: float = 0.7            # 去重相关性阈值
    turnover_threshold: float = 0.5
    monotonicity_threshold: float = 0.7
    coverage_threshold: float = 0.8

    # 搜索
    max_factors: int = 100                  # 最大候选因子数
    search_strategy: str = "template"       # template | mcts
    mcts_iterations: int = 50              # MCTS迭代次数 (阶段2)
    seed: int = 42
```

### 3.4 AutoResearchResult — 挖掘结果

```python
@dataclass
class AutoResearchResult:
    valid_factors: List[FactorEvaluationResult]   # 通过验证的因子
    all_evaluated: List[FactorEvaluationResult]   # 所有评估过的因子
    rejected_count: int = 0                       # 被拒绝的因子数
    deduplicated_count: int = 0                   # 去重去掉的因子数
    config: MiningConfig = None
    report_markdown: str = ""                     # Markdown 报告
    elapsed_seconds: float = 0.0
```

---

## 四、模板因子库

### 4.1 模板分类

```python
TEMPLATES = {
    "momentum": {
        "description": "动量因子",
        "formulas": [
            ("ts_delta({col}, {w})", "动量: {col} 的 {w} 期变化"),
            ("ts_pct_change({col}, {w})", "收益率: {col} 的 {w} 期涨跌幅"),
            ("ts_mean({col}, {w}) / ts_std({col}, {w})", "夏普比: {col} 的 {w} 期均值/标准差"),
        ],
    },
    "mean_reversion": {
        "description": "均值回归因子",
        "formulas": [
            ("({col} - ts_mean({col}, {w})) / ts_std({col}, {w})", "Z-Score: {col} 偏离 {w} 期均值"),
            ("{col} / ts_mean({col}, {w}) - 1", "价格/均值比: {col} 相对 {w} 期均值"),
        ],
    },
    "volatility": {
        "description": "波动率因子",
        "formulas": [
            ("ts_std({col}, {w})", "波动率: {col} 的 {w} 期标准差"),
            ("ts_std({col}, {w}) / ts_mean({col}, {w})", "变异系数: {col} 的 {w} 期CV"),
            ("ts_max({col}, {w}) - ts_min({col}, {w})", "振幅: {col} 的 {w} 期极差"),
        ],
    },
    "volume_price": {
        "description": "量价因子",
        "formulas": [
            ("ts_corr({col1}, {col2}, {w})", "相关性: {col1} 与 {col2} 的 {w} 期相关"),
            ("ts_cov({col1}, {col2}, {w})", "协方差: {col1} 与 {col2} 的 {w} 期协方差"),
            ("{col1} * {col2}", "乘积: {col1} × {col2}"),
        ],
    },
}
```

### 4.2 输入列组合

```python
INPUT_COMBOS = [
    (["close"],),
    (["open"],),
    (["high"],),
    (["low"],),
    (["vol"],),
    (["close", "vol"],),
    (["close", "open"],),
    (["high", "low"],),
]
```

---

## 五、6维度评估模型

### 5.1 维度定义

| 维度 | 指标 | 阈值 | 权重 | 说明 |
|------|------|------|------|------|
| 收益 | IC, IC_IR, Rank IC | \|IC\| > 0.03, IC_IR > 0.5 | 0.30 | 预测力 |
| 稳定性 | 滚动IC均值/标准差 | 稳定性 > 0.6 | 0.20 | 样本内一致性 |
| 分散度 | 因子间Spearman相关 | 平均相关 < 0.7 | 0.20 | 与已有因子差异性 |
| 换手率 | 排名变化率 | < 0.5 | 0.15 | 交易成本意识 |
| 单调性 | 5组分位收益 | 单调性 > 0.7 | 0.10 | 因子区分度 |
| 覆盖率 | 非空比例 | > 0.8 | 0.05 | 数据质量 |

### 5.2 综合评分

```python
overall_score = (
    0.30 *收益评分 +
    0.20 * 稳定性评分 +
    0.20 * 分散度评分 +
    0.15 * (1 - turnover) +
    0.10 * 单调性评分 +
    0.05 * 覆盖率
)

is_valid = all([
    abs(ic_mean) > ic_threshold,
    abs(icir) > icir_threshold,
    stability_score > stability_threshold,
    avg_corr_with_existing < corr_threshold,
    turnover < turnover_threshold,
    monotonicity_score > monotonicity_threshold,
    coverage > coverage_threshold,
])
```

### 5.3 相关性去重算法

```python
def deduplicate(results, corr_threshold=0.7):
    """
    贪心聚类去重:
    1. 按 overall_score 降序排序
    2. 遍历，如果与已选因子相关性 > threshold，跳过
    3. 否则加入已选列表
    """
    sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)
    selected = []
    for r in sorted_results:
        if all(abs(spearman_corr(r.factor_values, s.factor_values)) < corr_threshold
               for s in selected):
            selected.append(r)
    return selected
```

---

## 六、MCTS 搜索树（阶段2）

### 6.1 搜索树结构

```
搜索树:
    Root (空公式)
    ├── ts_mean(close, 20)
    │   ├── rank(ts_mean(close, 20))
    │   │   ├── rank(ts_mean(close, 20)) / ts_std(close, 20)
    │   │   └── rank(ts_mean(close, 20)) - rank(ts_mean(vol, 20))
    │   └── ts_mean(close, 20) - ts_mean(close, 60)
    ├── ts_std(vol, 10)
    └── ...
```

### 6.2 MCTS 四步循环

```python
class MCTSNode:
    formula: str
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    visits: int = 0
    dimension_scores: Dict[str, float] = {}  # 6维度评分
    overall_score: float = 0.0

class MCTSSearch:
    def search(self, root, iterations=50):
        for _ in range(iterations):
            node = self._select(root)       # 1. SELECT: UCB1选择
            child = self._expand(node)       # 2. EXPAND: 生成子节点
            scores = self._evaluate(child)   # 3. EVALUATE: 多维度评分
            self._backpropagate(child, scores)  # 4. BACKUP: 回传评分
        return self._get_best_path(root)
```

### 6.3 UCB1 选择策略

```python
def ucb1(node, exploration_weight=1.414):
    if node.visits == 0:
        return float('inf')
    exploit = node.overall_score
    explore = exploration_weight * sqrt(log(node.parent.visits) / node.visits)
    return exploit + explore
```

### 6.4 维度化反馈

```python
def get_weak_dimensions(node):
    """找出当前节点的弱维度，指导扩展方向"""
    weak = []
    if node.dimension_scores.get("return", 0) < 0.5:
        weak.append("return")      # 收益不够 → 换算子
    if node.dimension_scores.get("diversification", 0) < 0.5:
        weak.append("diversification")  # 相关性高 → 换输入列
    if node.dimension_scores.get("turnover", 0) < 0.5:
        weak.append("turnover")    # 换手高 → 加大窗口
    return weak
```

---

## 七、AutoResearcher 编排器

### 7.1 核心接口

```python
class AutoResearcher:
    """自动因子研究系统"""

    def __init__(self, wiki_path: str):
        self.wiki_path = wiki_path
        self.proxy = WikiFactorProxy(wiki_path)
        self.miner = FactorMiner()
        self.evaluator = FactorEvaluator()
        self.mcts = MCTSSearch()  # 阶段2

    def run(
        self,
        data: pl.DataFrame,           # 行情数据
        config: MiningConfig = None,   # 挖掘配置
    ) -> AutoResearchResult:
        """执行完整挖掘流程"""

    def mine_single_factor(
        self,
        formula: str,
        data: pl.DataFrame,
    ) -> FactorEvaluationResult:
        """验证单个因子公式"""
```

### 7.2 执行流程

```python
def run(self, data, config=None):
    config = config or MiningConfig()

    # 1. 生成候选因子
    candidates = self.miner.generate(data, config)

    # 2. 逐个评估
    results = []
    for candidate in candidates:
        result = self.evaluator.evaluate(candidate, data, config)
        if result.is_valid:
            results.append(result)

    # 3. 相关性去重
    deduplicated = self.evaluator.deduplicate(results, config.corr_threshold)

    # 4. 存入 Wiki
    for result in deduplicated:
        self._store_to_wiki(result)

    # 5. 生成报告
    report = self._generate_report(deduplicated, results, config)

    return AutoResearchResult(
        valid_factors=deduplicated,
        all_evaluated=results,
        rejected_count=len(candidates) - len(results),
        deduplicated_count=len(results) - len(deduplicated),
        config=config,
        report_markdown=report,
    )
```

---

## 八、Wiki 存储集成

### 8.1 因子存储

```python
def _store_to_wiki(self, result: FactorEvaluationResult):
    """将验证通过的因子存入 Wiki"""
    factor = WikiFactor(
        name=result.candidate.name,
        formula=result.candidate.formula,
        source=FactorSource.AUTO_RESEARCH,
        category=result.candidate.category,
        tags=[result.candidate.template_name],
        ic_mean=result.ic_mean,
        ic_std=result.ic_std,
        icir=result.icir,
        rank_ic_mean=result.rank_ic_mean,
        n_dates=None,
        factor_return_corr=result.avg_corr_with_existing,
        ic_t_stat=None,
        turnover=result.turnover,
    )
    self.proxy.store_factor(factor)
```

---

## 九、实施步骤

| Step | 任务 | 文件 | 预估行数 |
|------|------|------|----------|
| 1 | 模板定义 + 公式生成 + 语法校验 | `factor_miner.py` | ~200 |
| 2 | IC/IR 计算 + 分组收益 + 单调性 | `factor_evaluator.py` | ~150 |
| 3 | 换手率 + 覆盖率 + 综合评分 | `factor_evaluator.py` | ~100 |
| 4 | 相关性去重 | `factor_evaluator.py` | ~50 |
| 5 | AutoResearcher 编排器 + Wiki集成 | `auto_researcher.py` | ~250 |
| 6 | MCTS 搜索树 + UCB1 + 维度反馈 | `mcts_search.py` | ~300 |
| 7 | 集成 MCTS 到 AutoResearcher | `auto_researcher.py` | ~50 |
| 8 | 单元测试 | `tests/research/test_auto_research.py` | ~300 |
| 9 | 更新设计文档 + README | `docs/` + `QuantNodes/research/README.md` | ~100 |

**总计**: ~1500 行代码

---

## 十、测试策略

### 10.1 单元测试

- `test_factor_miner.py`: 模板生成、公式语法校验、输入列组合
- `test_factor_evaluator.py`: IC/IR计算、分组收益、单调性、换手率、去重
- `test_mcts_search.py`: 节点创建、UCB1选择、扩展、回传

### 10.2 集成测试

- `test_auto_research_e2e.py`: 完整流程（生成→评估→去重→Wiki存储）

### 10.3 测试数据

使用 mock DataFrame，不依赖真实行情数据:
```python
@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    return pl.DataFrame({
        "date": ["2024-01-01"] * n,
        "code": [f"SZ{i:06d}" for i in range(n)],
        "close": np.random.uniform(10, 100, n),
        "open": np.random.uniform(10, 100, n),
        "high": np.random.uniform(10, 100, n),
        "low": np.random.uniform(10, 100, n),
        "vol": np.random.uniform(1000, 100000, n),
        "forward_return": np.random.normal(0, 0.02, n),
    })
```

---

**文档版本**: v1.0
**最后更新**: 2026-05-07
