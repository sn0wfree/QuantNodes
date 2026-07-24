# v9e — 回测验证方案

> **编号**: 49e
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49-v9_cycle_timing.md, docs/49f-v9_v8_integration.md

---

## 1. 回测目标

### 1.1 主要目标

**验证 v9 大盘择时信号是否能产生优于 v7.14 baseline 的收益**:

| 指标 | v7.14 baseline | v8_method_b | **v9 目标** |
|------|----------------|-------------|------------|
| OOS Sharpe | 0.438 | 1.485 | ≥ 0.5 |
| OOS Calmar | 0.35-0.42 | 1.467 | ≥ 0.5 |
| OOS MaxDD | -22.7% | -12.30% | ≤ -20% |
| OOS AnnRet | 7% | 18.04% | ≥ 8% |

**目标含义**: v9 不追求超越 v8 (1.485), 只求显著超过 v7.14 baseline 0.44, 即 +15% 提升, 证明周期分析层的价值。

### 1.2 次要目标

- **大底/大顶检测**: 2014/2019/2024 三次大底前 4 周内发出做多信号 ≥ 2 次
- **回撤控制**: 黑天鹅 (2020 新冠, 2022 俄乌) 期间空仓时间 ≥ 50%
- **稳健性**: 多起点测试 Sharpe 标准差 < 0.3

---

## 2. 回测框架

### 2.1 数据范围

```
训练期: 2014-01-01 ~ 2017-12-31 (4 年, 用于拟合 HP 滤波参数)
样本内: 2018-01-01 ~ 2020-12-31 (3 年, 用于参数稳定)
OOS 期: 2021-01-01 ~ 2026-05-31 (5.4 年, 主要验证期)
```

**说明**: A 股数据从 1990 年至今, 但 v9 仅用 v7.14 的 43 ETF 池 (2018-2026), 保证可比性。

### 2.2 回测引擎

**复用**: `QuantNodes/strategy/momentum_etf_rotation/backtest.py`

```python
from QuantNodes.strategy.momentum_etf_rotation.backtest import BacktestEngine

engine = BacktestEngine(
    weekly_weights=v9_adjusted_weights,
    daily_returns=etf_returns,
    cost_bps=10,  # 10bp 默认
    rebalance_freq='W',  # 周频
)

result = engine.run()
```

### 2.3 调仓频率

**默认**: 周频调仓 (与 v7.14 一致)

**备选**: 月频调仓 (减少交易成本)

| 频率 | 优点 | 缺点 |
|------|------|------|
| 周频 | 信号更新快 | 交易成本高 |
| 月频 | 成本低 | 信号滞后 |

**v9 默认**: 周频 (因为周频 ETF 数据可用)。

---

## 3. 评估指标

### 3.1 主指标 (与 v7.14 对比)

**核心指标**: OOS Sharpe Ratio

```python
def compute_sharpe(returns: pd.Series, 
                   freq: str = 'W',
                   rf: float = 0.02) -> float:
    """
    计算年化 Sharpe.
    
    参数:
        returns: 收益率序列
        freq: 'D' | 'W' | 'M'
        rf: 无风险利率 (年化)
    """
    if freq == 'D':
        periods = 252
    elif freq == 'W':
        periods = 52
    elif freq == 'M':
        periods = 12
    
    excess = returns - rf / periods
    return excess.mean() / excess.std() * np.sqrt(periods)
```

### 3.2 辅助指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **Calmar** | AnnRet / |MaxDD| | 收益回撤比 |
| **Sortino** | AnnRet / DownsideDev | 只看下行波动 |
| **MaxDD** | min(Nav / Nav.cummax() - 1) | 最大回撤 |
| **AnnRet** | (Nav[-1] / Nav[0])^(252/n) - 1 | 年化收益 |
| **Vol** | returns.std() * sqrt(252) | 年化波动率 |
| **WinRate** | (returns > 0).mean() | 胜率 |

### 3.3 指标计算工具

**复用**: `QuantNodes/strategy/momentum_etf_rotation/fi_plus.py::performance_metrics`

```python
from QuantNodes.strategy.momentum_etf_rotation.fi_plus import performance_metrics

metrics = performance_metrics(
    nav=nav_series,
    returns=returns_series,
    rf=0.02,
    freq='W',
)
```

---

## 4. 多起点测试 (Multi-start Validation)

### 4.1 目的

**检验**: 不同训练起点是否对最终 Sharpe 有显著影响。

**风险**: 如果 v9 表现依赖于特定起点, 说明可能过拟合。

### 4.2 实现

```python
def multi_start_backtest(start_dates: list, 
                          end_date: str,
                          min_train_years: int = 4) -> pd.DataFrame:
    """
    多起点回测.
    
    参数:
        start_dates: 训练起点列表
        end_date: 测试终点
        min_train_years: 最小训练期
    
    返回:
        DataFrame: 每行一个起点的指标
    """
    results = []
    for start in start_dates:
        train_end = pd.Timestamp(end_date) - pd.DateOffset(years=min_train_years)
        
        # 训练
        v9_signal = train_v9(start, train_end)
        
        # 测试
        metrics = backtest_v9(v9_signal, train_end, end_date)
        metrics['start'] = start
        results.append(metrics)
    
    return pd.DataFrame(results)
```

### 4.3 起点选择

**默认 5 个起点**:
- 2014-01-01 (牛市初)
- 2015-06-01 (牛市顶)
- 2016-01-01 (熔断后)
- 2018-01-01 (贸易战初)
- 2020-01-01 (疫情前)

**目标**: 5 个起点的 Sharpe 均值 ≥ 0.5, 标准差 < 0.3。

---

## 5. 滚动窗口测试 (Walk-Forward)

### 5.1 目的

**检验**: 在不同时间窗口下, 策略表现是否稳定。

### 5.2 实现

```python
def walk_forward_backtest(start: str, 
                           end: str,
                           train_years: int = 4,
                           test_years: int = 2) -> list:
    """
    滚动窗口回测.
    
    4 年训练 + 2 年测试, 滚动推进。
    """
    dates = pd.date_range(start, end, freq='YS')
    results = []
    
    for i in range(len(dates) - 2):
        train_start = dates[i]
        train_end = dates[i + train_years]
        test_start = train_end
        test_end = dates[min(i + train_years + test_years, len(dates) - 1)]
        
        v9_signal = train_v9(train_start, train_end)
        metrics = backtest_v9(v9_signal, test_start, test_end)
        results.append({
            'train': (train_start, train_end),
            'test': (test_start, test_end),
            **metrics,
        })
    
    return results
```

### 5.3 期望

| 测试窗口 | 期望 Sharpe | 备注 |
|---------|------------|------|
| 2018-2019 | ≥ 0.4 | 熊市末 |
| 2020-2021 | ≥ 0.6 | 疫情后反弹 |
| 2022-2023 | ≥ 0.3 | 调整市 |
| 2024-2025 | ≥ 0.5 | 反弹 |

---

## 6. 成本敏感性测试

### 6.1 目的

**检验**: 不同交易成本下的策略表现。

### 6.2 成本档位

| 档位 | 成本 (bp) | 适用场景 |
|------|----------|---------|
| 极低 | 5 | 大资金 + 极低费率 |
| 低 | 10 | 散户典型 |
| 中 | 20 | 高频 + ETF 折溢价 |
| 高 | 50 | 流动性差 |

### 6.3 期望

| 成本 (bp) | 期望 Sharpe | 备注 |
|----------|------------|------|
| 5 | ≥ 0.7 | 最优场景 |
| 10 | ≥ 0.5 | 基线 |
| 20 | ≥ 0.3 | 仍优于 v7.14 |
| 50 | ≥ 0.1 | 接近无信号 |

---

## 7. 参数敏感性测试

### 7.1 关键参数

| 参数 | 默认 | 测试范围 |
|------|------|---------|
| upper_threshold | 50 | 40-60 |
| lower_threshold | 30 | 20-40 |
| lock_threshold_deg | 30° | 15-60 |
| lock_min_duration_weeks | 12 | 4-24 |
| hp_lambda | 100 | 50-500 |

### 7.2 实现

```python
def parameter_sensitivity(default_params: dict,
                          perturbations: dict,
                          n_random: int = 50) -> pd.DataFrame:
    """
    参数敏感性测试.
    
    围绕默认值随机扰动, 检验稳健性.
    """
    results = []
    
    for _ in range(n_random):
        params = default_params.copy()
        for key, (low, high) in perturbations.items():
            params[key] = np.random.uniform(low, high)
        
        metrics = backtest_with_params(params)
        results.append(metrics)
    
    return pd.DataFrame(results)
```

### 7.3 期望

**Sharpe 标准差 < 0.1** (相对于默认值的扰动)。

---

## 8. 退出标准 (Phase 2 决策点)

### 8.1 通过条件

**任一满足即可**:
- ✅ 2014/2019/2024 三次大底中 ≥ 2 次在底部前 4 周发出 Δφ<30° 锁定
- ✅ OOS Sharpe 0.44 → 0.5 显著提升 (t-test p<0.1)
- ✅ 多起点 Sharpe 均值 ≥ 0.5, 标准差 < 0.3
- ✅ 双相干系数在 ≥ 2 对 IMF 上 >0.6

### 8.2 失败应对

**如未通过** → 仅交付:
- 8 份文档
- CPD 模块 (美林+Pring)
- HTML 仪表盘
- 当前周期报告 (docs/50)

**不做**:
- ❌ 完整 v9 仓位合成
- ❌ 与 v8 集成
- ❌ 实盘部署

---

## 9. 报告输出

### 9.1 回测报告

```
reports/momentum_etf_rotation/v9/
├── backtest_results.csv         # 完整回测指标 (Sharpe, Calmar, etc.)
├── backtest_nav.csv             # NAV 时序
├── nav_comparison.png           # v9 vs v8 vs v7.14 NAV 对比
├── multi_start_results.csv      # 多起点测试
├── walk_forward_results.csv     # 滚动窗口
├── cost_sensitivity.csv         # 成本敏感性
├── param_sensitivity.csv        # 参数敏感性
└── summary.md                   # 总结报告
```

### 9.2 报告样例

```markdown
# v9 回测报告
> 回测期: 2021-01-01 ~ 2026-05-31
> 数据频率: 周频

## 主指标对比

| 指标 | v7.14 | v8_method_b | v9 |
|------|-------|-------------|-----|
| Sharpe | 0.438 | 1.485 | 0.62 |
| Calmar | 0.40 | 1.467 | 0.71 |
| MaxDD | -22.7% | -12.30% | -15.8% |
| AnnRet | 7.0% | 18.04% | 11.2% |

## 多起点测试

| 起点 | Sharpe | Calmar | MaxDD |
|------|--------|--------|-------|
| 2014 | 0.55 | 0.65 | -16% |
| 2015 | 0.61 | 0.72 | -15% |
| 2016 | 0.58 | 0.68 | -16% |
| 2018 | 0.65 | 0.78 | -14% |
| 2020 | 0.62 | 0.71 | -15% |
| **均值** | **0.60** | **0.71** | **-15.2%** |
| **标准差** | **0.04** | **0.05** | **0.8%** |

## 决策

✅ Phase 2 通过, 继续 Phase 3 (完整集成)
```

---

## 10. 实施清单

### 10.1 文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/
└── backtest.py               # 本模块
```

### 10.2 接口

```python
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest,
    multi_start_backtest,
    walk_forward_backtest,
    cost_sensitivity,
    parameter_sensitivity,
    generate_report,
)
```

### 10.3 主入口

```python
# scripts/v9/v9_cycle_timing_main.py

if __name__ == '__main__':
    # 1. 数据加载
    data = load_v9_data()
    
    # 2. 信号生成
    v9_signals = generate_v9_signals(data)
    
    # 3. 主回测
    main_result = run_backtest(v9_signals, data)
    
    # 4. 多起点
    multi_start = multi_start_backtest(...)
    
    # 5. 滚动窗口
    walk_forward = walk_forward_backtest(...)
    
    # 6. 敏感性
    cost_sens = cost_sensitivity(v9_signals, data)
    param_sens = parameter_sensitivity(...)
    
    # 7. 报告生成
    generate_report(main_result, multi_start, walk_forward, cost_sens, param_sens)
```

---

## 11. 风险与对冲

| 风险 | 缓解 |
|------|------|
| 起点偏差 | 多起点测试 |
| 过拟合 | 滚动窗口 + 留出测试 |
| 交易成本忽略 | 多档成本测试 |
| 参数过拟合 | 学术默认权重 + 敏感性测试 |
| 未来函数 | 严格无未来数据实现 |
| 信号回看偏差 | 严格滚动更新 |

---

## 12. 参考文献

1. **Harvey, C. R., Liu, Y. (2015)**. "Backtesting". *Journal of Portfolio Management* 42(1), 13-28.
2. **Bailey, D. H., López de Prado, M. (2014)**. "The Deflated Sharpe Ratio". *Journal of Portfolio Management* 40(5), 94-107.
3. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Chapter 11.

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中