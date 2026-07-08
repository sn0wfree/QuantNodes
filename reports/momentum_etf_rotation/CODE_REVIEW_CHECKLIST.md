# Code Review Checklist

> 版本: v1.0
> 基于: Stage 9-13 开发过程中遇到的 7 个 bug
> 用途: 提交前必过的检查清单

---

## 0. 使用方式

每个 PR / commit 前, 对照本清单逐项检查。任何一个 ❌ 都必须修复后才能提交。

```markdown
## Code Review Check
- [ ] A. 字段与参数 (5 项)
- [ ] B. 集成顺序 (4 项)
- [ ] C. 不变量 (3 项)
- [ ] D. 边界条件 (5 项)
- [ ] E. 错误处理 (3 项)
- [ ] F. 性能 (2 项)
```

---

## A. 字段与参数 (5 项)

### A.1 ✅ 强制使用关键字参数

```python
# ❌ 错误: 字段顺序陷阱 (Stage 9-B bug)
TrendFilter(True, 200, 0.5, bond_code="511260")
# 实际: enabled=True, benchmark_code=200 (int!), ma_window=0.5
# → trend_filter 永远默认多头, 完全失效

# ✅ 正确
TrendFilter(
    enabled=True,
    benchmark_code="510300",
    ma_window=200,
    exposure_bear=0.7,
    bond_code="511260",
)
```

**规则**: dataclass 字段 ≥ 3 个时, **必须**用关键字参数。

### A.2 ✅ 字段顺序与默认值一致

确保位置参数顺序与字段定义顺序一致, 避免理解偏差。

### A.3 ✅ 新增字段给默认值

```python
@dataclass
class NewFeature:
    enabled: bool = False        # 永远默认关闭
    lookback: int = 60           # 合理默认值
    min_scale: float = 0.3
```

**规则**: 新字段必须给默认值, 不破坏向后兼容。

### A.4 ✅ 必填字段无默认值时强制 kwarg

如果字段无默认值, 调用时只能用关键字参数传递。

### A.5 ✅ 检查字段拼写

`bond_code="511260"` vs `bench_code` vs `bond_cdoe` - 字段名拼写错误是常见 bug。

---

## B. 集成顺序 (4 项)

> 这是 bug 集中爆发区 (Stage 9-C, 10, 13 都中招)

### B.1 ✅ 画数据流图

修改 backtest 循环前, 用 ASCII 图画出每个 step 的副作用:

```
select_and_weight
  → 设置 state.chosen, ranked
apply_trend_filter
  → 修改 state.weights (加 511260)
apply_vol_targeting
  → 修改 state.weights (缩放) ← 不能被后面的归一化覆盖!
回测循环
  → 重计算 inverse_vol_weights (覆盖 caps!) ← Stage 10 bug
  → 重新应用 caps ← 必须二次应用
apply_trend_filter
  → 修改 state.weights
```

### B.2 ✅ 验证"后操作不被前操作覆盖"

```python
# Stage 9-C bug: vol_targeting 后又被归一化
state.weights = {k: v * scale for ...}  # vol_targeting
# 删除: state.weights = {k: v / total ...}  # 这会取消缩放!

# Stage 10 bug: apply_stops 重新 inverse_vol_weights 覆盖 caps
if cfg.weight_method == "inv_vol":
    state.weights = inverse_vol_weights(...)  # 这里覆盖了 caps
if cfg.concentration.enabled:  # ← 必须在 inv_vol 之后再应用
    state.weights = _apply_concentration_caps(...)
```

**规则**: **任何重新计算 state.weights 的操作后, 必须重新应用所有修饰 (caps, trend_filter, vol_targeting)**。

### B.3 ✅ 检查修改顺序的逻辑依赖

```python
# 趋势过滤: 先选股, 再加债券 → 加权后补债券
# 但如果先加权再加债券 → 权重变化会改变组合波动率
```

### B.4 ✅ 集成测试覆盖关键集成点

```python
def test_full_integration_no_corruption():
    """完整集成所有 feature, 验证不互相覆盖."""
    cfg = RotationConfig(
        trend_filter=...,   # 假设 enabled
        vol_targeting=...,  # 假设 enabled
        concentration=...,   # 假设 enabled
    )
    state = select_and_weight(panel, pool, cfg, date)
    # 验证每个 feature 都生效
    assert has_bond_511260(state)        # trend_filter
    assert sum_weights_less_than_full(state)  # vol_targeting
    assert max_weight_capped(state)       # concentration
```

---

## C. 不变量 (3 项)

### C.1 ✅ 文档化所有不变量

```python
# 不变量 1: 状态权重总和 <= 1 (允许持有现金)
# 不变量 2: 单 ETF 权重 ∈ [0, 1] 
# 不变量 3: OOS Calmar > 0.5 (策略有效)
```

**规则**: 任何"应该 = X"的属性都要写为 **不变量**, 在测试中 assert。

### C.2 ✅ 关键路径加 assert

```python
def test_weighting_invariant():
    state = select_and_weight(panel, pool, cfg, date)
    # 不变量 1
    total = sum(state.weights.values())
    assert total <= 1.0 + 1e-6, f"weight sum={total}"
    # 不变量 2
    for code, w in state.weights.items():
        assert 0 <= w <= 1.0, f"{code}={w}"
```

### C.3 ✅ NAV 计算不变量

```python
# Stage 13 bug: nav[i] = nav[i] * 1.0 用未初始化值
# 应改为: nav[i] = nav[i-1] * (1 + daily_ret)
# 关键: nav[i] 必须先从 nav[i-1] 或常量初始化
```

**不变量**: `nav[i] ∈ [0, ∞)`, 不可能为 None 或负。

---

## D. 边界条件 (5 项)

### D.1 ✅ 空 inputs

```python
def test_empty_weights():
    state = apply_xxx(empty_state)
    assert state.weights == {}
    assert state.chosen == []
```

### D.2 ✅ 0 换手 (无 turnover)

```python
def test_zero_turnover():
    state = ...  # 不变的 state
    cost = calculate_turnover_cost(0.0, cost_model)
    assert cost == 0.0
```

### D.3 ✅ 单 ETF (N=1)

```python
def test_single_etf():
    pool = make_pool_with_one_etf()
    state = select_and_weight(panel, pool, cfg, date)
    assert len(state.chosen) == 1
    assert state.weights[single_code] > 0
```

### D.4 ✅ 数据不足 (window > len(data))

```python
def test_insufficient_data():
    short_panel = panel.iloc[:5]  # 只有 5 天
    with pytest.raises(ValueError):
        select_and_weight(short_panel, pool, cfg, date)
    # 或: fallback 到合理默认
```

### D.5 ✅ benchmark/feature 代码不存在

```python
def test_unknown_benchmark():
    cfg.lookback = 90  # 默认 510300 在池中
    # 故意设置不存在的 benchmark
    cfg.trend_filter.benchmark_code = "999999"
    state = select_and_weight(panel, pool, cfg, date)
    # 应该 fallback 到默认行为 (默认多头)
```

---

## E. 错误处理 (3 项)

### E.1 ✅ 缺数据 fallback

```python
def calculate_turnover_cost(turnover, cost):
    if not cost.enabled:
        return 0.0  # fallback
    if turnover < 0:  # 异常
        return 0.0  # fallback 而非崩溃
```

### E.2 ✅ HMM 训练失败不崩溃

```python
# Stage 9-D 改进方向
try:
    detector.fit(train_nav)
except Exception as e:
    logger.warning(f"HMM failed: {e}, defaulting to neutral")
    detector = None  # fallback
```

### E.3 ✅ 资产不存在时跳过而非崩溃

```python
def test_missing_etf_in_pool():
    # 假设选中的 ETF 后来退市了
    cfg = RotationConfig()
    cfg.blacklist = ["retired_etf"]
    state = select_and_weight(panel, pool, cfg, date)
    # 不应崩溃, 只跳过被 blacklist 的
```

---

## F. 性能 (2 项)

### F.1 ✅ O(N²) 协方差计算限制规模

```python
# 44 ETF 的协方差 = 44×44 矩阵 = 1896 个值 = OK
# 但如果池子扩展到 200+ ETF, 内存和计算会变慢
# → 考虑用 factor model 或 Ledoit-Wolf 收缩
```

### F.2 ✅ 回测不反复重算

```python
# 缓存不依赖时间的纯函数
@lru_cache
def get_category_weights_static(code):
    return CATEGORIES.get(code)

# 只在调仓日重算权重
# 调仓日之间使用上一次结果
```

---

## G. 测试纪律 (3 项)

### G.1 ✅ 集成测试 > 单元测试

原因: 我们的 bug 全在集成点 (apply_stops, backtest loop, 集成顺序)。

```python
# 至少 50% 测试是集成测试, 而非纯单元测试
```

### G.2 ✅ 每个真实数据 bug 必有回归测试

Stage 9-13 的每个 bug 都应添加回归测试:
- `test_trend_filter_field_order_bug` (Stage 9-B)
- `test_vol_targeting_not_normalized_bug` (Stage 9-C)
- `test_concentration_reapplied_in_apply_stops_bug` (Stage 10)
- `test_nav_initialization_cost_bug` (Stage 13)

### G.3 ✅ OOS 测试是必过的测试

不要把 OOS 测试降级为 `@pytest.mark.skip`, 必须真跑。

---

## H. 文档纪律 (2 项)

### H.1 ✅ 报告先于 commit

每个 Stage 完成后:
1. 先写 `stage<N>_report.md`
2. 再生成 charts
3. **最后** git commit

### H.2 ✅ commit message 包含 bug 教训

```bash
# 好例子
git commit -m "feat(stage9-b): trend filter

Bugs caught in this stage:
- TrendFilter field order: positional args assigned to wrong fields
  Fix: Use keyword arguments

Configuration:
- TrendFilter(enabled=True, benchmark_code='510300', ...)"
```

---

## 实际案例 (我们的 7 个 bug 对照)

| Bug | 违反的 checklist 项 | 教训 |
|-----|-------------------|------|
| `TrendFilter` 字段顺序 | A.1, A.5 | 强制 kwarg |
| `vol_targeting` 后归一化 | B.2, B.4 | 集成顺序必须测试 |
| `apply_stops` 覆盖 caps | B.1, B.2, B.4 | 画数据流图 |
| `nav[i] = nav[i] * 1.0` 未初始化 | C.3, B.4 | 不变量 + 集成测试 |
| `fill_by_rank` 未检查 caps | B.4 | 集成测试 |
| `resample("ME")` 错位 | (pandas 误解) | 注意 pandas API 变化 |
| HMM 全协方差过拟合 | D.5 | 注意 p>>n 问题 |

---

## 提交前必过

```markdown
## Pre-commit Check (日期: ____)

### A. 字段与参数
- [ ] 所有 dataclass 调用都用关键字参数
- [ ] 新字段都有默认值
- [ ] 字段拼写正确

### B. 集成顺序
- [ ] 画了数据流图
- [ ] 所有重新计算 state.weights 后重新应用修饰
- [ ] 至少 1 个集成测试覆盖本次改动

### C. 不变量
- [ ] 文档化了相关不变量
- [ ] 关键路径有 assert

### D. 边界
- [ ] 空 inputs 测试
- [ ] 数据不足测试
- [ ] 单 ETF / 异常值测试

### E. 错误处理
- [ ] 缺数据有 fallback
- [ ] 异常不崩溃 (有 try/except 或 fallback)

### F. 性能
- [ ] 缓存了纯函数 (如有)
- [ ] 没有 O(N³) 的循环

### G. 测试
- [ ] 回归测试覆盖 bug 场景
- [ ] OOS 测试通过

### H. 文档
- [ ] 报告已写
- [ ] commit message 包含 bug 教训
```

---

**Checklist 结束**

建议每 3 个月复盘: 哪些项常被跳过? 哪些项总是有效?
