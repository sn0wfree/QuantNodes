# 互信息去重设计文档

## 目标

在因子池中移除高度相关的因子，保留多样化的因子集合。

## 算法选择

### 贪心相关性去重（推荐）

**算法**：
1. 按 `overall_score` 降序排序
2. 逐个检查与已选因子的 Spearman 相关性
3. 如果 `|corr| > threshold`，跳过该因子
4. 否则加入已选集合

**复杂度**：O(n * k * m)
- n = 候选因子数
- k = 已选因子数
- m = 序列长度

**优点**：
- 简单、确定性、可复现
- 保留每组相关因子中得分最高的
- 已在旧系统验证

**缺点**：
- 依赖排序顺序
- 贪心选择可能不是全局最优

## 实现方案

### 1. 辅助函数

```python
def _spearman_corr(x: pl.Series, y: pl.Series) -> float:
    """Spearman 秩相关

    Args:
        x: 第一个序列
        y: 第二个序列

    Returns:
        相关系数 [-1, 1]
    """
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x_rank = x.head(n).rank()
    y_rank = y.head(n).rank()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])
```

### 2. 去重函数

```python
def deduplicate_mutual_ic(
    factors: List[FactorMetrics],
    get_values: Callable[[FactorMetrics], Optional[pl.Series]],
    threshold: float = 0.7,
) -> List[FactorMetrics]:
    """贪心互信息去重

    Args:
        factors: 候选因子列表
        get_values: 获取因子值的函数
        threshold: 相关性阈值（默认 0.7）

    Returns:
        去重后的因子列表
    """
    sorted_f = sorted(factors, key=lambda f: f.overall_score, reverse=True)
    selected = []
    for f in sorted_f:
        vals = get_values(f)
        if vals is None:
            continue
        is_dup = False
        for s in selected:
            s_vals = get_values(s)
            if s_vals is None:
                continue
            corr = _spearman_corr(vals, s_vals)
            if abs(corr) > threshold:
                is_dup = True
                break
        if not is_dup:
            selected.append(f)
    return selected
```

### 3. 集成点

#### Alpha-GPT 工作流

在 `_select_final_pool()` 中，critic 选择后调用去重：

```python
def _select_final_pool(self) -> List[FinalFormulaRecord]:
    # ... 现有逻辑 ...

    # 新增：互信息去重
    if self.config.max_mutual_ic_threshold < 1.0:
        final_pool = self._deduplicate_pool(final_pool)

    return final_pool
```

#### MCTS 搜索

在 `search()` 结束时，对 `best_k_nodes` 去重：

```python
def search(self, data, ...):
    # ... 现有逻辑 ...

    # 新增：互信息去重
    if self.config.dedup_threshold < 1.0:
        best_k = self._deduplicate_nodes(best_k, data)

    return MCTSSearchResult(...)
```

## 测试计划

```
tests/quant_alpha/test_dedup.py
├── TestSpearmanCorr
│   ├── test_perfect_correlation
│   ├── test_no_correlation
│   └── test_negative_correlation
├── TestDeduplicateMutualIC
│   ├── test_empty_list
│   ├── test_single_factor
│   ├── test_no_duplicates
│   ├── test_with_duplicates
│   └── test_threshold_sensitivity
└── TestIntegration
    ├── test_alphagpt_dedup
    └── test_mcts_dedup
```

## 预估工作量

| 任务 | 文件 | 代码行数 | 时间 |
|------|------|----------|------|
| 去重函数 | `polars_evaluator.py` | ~50 行 | 20min |
| 修改 Alpha-GPT | `alpha_gpt.py` | ~20 行 | 10min |
| 修改 MCTS | `search.py` | ~20 行 | 10min |
| 测试 | `test_dedup.py` | ~100 行 | 30min |
| **总计** | **4** | **~190 行** | **~1h** |
