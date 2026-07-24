# v9f — 与 v8 / v7.14 集成方案

> **编号**: 49f
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49-v9_cycle_timing.md, docs/47-v8_optimization_record.md

---

## 1. 集成架构

### 1.1 三层金字塔

```
┌──────────────────────────────────────────────────────────────┐
│  v9 大盘择时层 (顶层)                                          │
│  ─ 信号 0/1, 决定是否参与市场                                  │
│  ─ 输出: v9_factor(t) ∈ {0, 1}                                │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  v7.14 选股权重层 (中层)                                       │
│  ─ TV-PR 因子预测, 给出 43 ETF 周权重                          │
│  ─ 输出: v7_weight(t, etf) ∈ [0, 1]                          │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  v8 仓位调节层 (底层)                                          │
│  ─ Jump Model 检测单资产 bear 状态, 调整仓位                   │
│  ─ 输出: v8_factor(t, etf) ∈ [0, 1]                          │
└──────────────────────────────────────────────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │  final = v9 × v7.14 × v8            │
              │  持仓 = final * 总资金              │
              │  现金 = 1 - sum(final)              │
              └──────────────────────────────────────┘
```

### 1.2 数据流

```
1. v9 评分 → 大盘信号 0/1
   ├── 输入: 4 IMF + 耦合信号 + VIX
   └── 输出: v9_signal(t) ∈ {0, 1}
   
2. v7.14 TV-PR → 周权重
   ├── 输入: 39 因子 + 43 ETF 收益
   └── 输出: weekly_weights (43,)
   
3. v8 Jump Model → Bear%
   ├── 输入: 43 ETF 日收益
   └── 输出: bear_pct (43,)
   
4. 集成 → 最终仓位
   ├── final_weight(t, etf) = v9 × v7 × v8
   └── 归一化 + 现金管理
```

---

## 2. v8 接口调用

### 2.1 v8 现有 API

**已有模块** (`QuantNodes/strategy/momentum_etf_rotation/v8/`):
- `jump_model.py::jump_model_periodic_retrain()` — 核心 Jump Model
- `integration.py::position_sizing_weights()` — 仓位调整
- `signal_composer.py::compute_composite_signal()` — 复合信号

### 2.2 v9 调用 v8

```python
from QuantNodes.strategy.momentum_etf_rotation.v8 import (
    jump_model_periodic_retrain,
    position_sizing_weights,
)
from QuantNodes.strategy.momentum_etf_rotation.v7_14.tvpr_estimator import (
    expanding_window_tvpr,
    compute_weekly_weights,
)


def compute_v8_bear_pct(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 ETF 计算 v8 Bear%.
    
    参数:
        daily_returns: (T_daily, N_etf) 日频收益
    
    返回:
        bear_pct: (T_daily, N_etf) Bear% ∈ [0, 1]
    """
    bear_pct = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns)
    
    for col in daily_returns.columns:
        states = jump_model_periodic_retrain(
            returns=daily_returns[col],
            asset_type='equity',
            jump_penalty=50.0,
            train_window=1000,
            retrain_every=30,
        )
        # 60 日滚动 Bear%
        bear_pct[col] = states.rolling(60).mean()
    
    return bear_pct
```

### 2.3 v8 仓位因子转换

```python
def v8_bear_to_factor(bear_pct: float, 
                       threshold: float = 0.3) -> float:
    """
    v8 Bear% → 仓位因子.
    
    参数:
        bear_pct: Bear% ∈ [0, 1]
        threshold: 阈值 (默认 0.3)
    
    返回:
        factor ∈ [0, 1]
    """
    if bear_pct <= threshold:
        return 1.0  # 满仓
    else:
        # 线性降至 0
        return max(1 - (bear_pct - threshold) / (1 - threshold), 0)
```

---

## 3. v7.14 接口调用

### 3.1 v7.14 现有 API

**已有模块** (`QuantNodes/strategy/momentum_etf_rotation/v7/`):
- `tvpr_estimator.py::expanding_window_tvpr()` — TV-PR 训练
- `macro_substrategy_v7_6.py::compute_weekly_weights()` — 周权重计算

### 3.2 v9 调用 v7.14

```python
def compute_v7_weights(Y_weekly: pd.DataFrame,
                        X_panel: np.ndarray,
                        factor_names: list) -> pd.DataFrame:
    """
    v7.14 TV-PR 计算周权重.
    
    参数:
        Y_weekly: (T_weekly, N_etf) 周频收益
        X_panel: (T_weekly, N_etf, K) 因子面板
        factor_names: 因子名称列表
    
    返回:
        weekly_weights: (T_weekly, N_etf) 周权重
    """
    from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
        expanding_window_tvpr,
    )
    
    # 1. TV-PR 训练
    beta_path = expanding_window_tvpr(
        Y=Y_weekly,
        X=X_panel,
        factor_names=factor_names,
        method='expanding',  # 无未来数据
    )
    
    # 2. 计算得分
    scores = np.einsum('tnk,nk->tn', X_panel, beta_path)
    
    # 3. 周权重 (Top-N + 逆波动率)
    weekly_weights = compute_weekly_weights(
        scores=scores,
        returns=Y_weekly,
        top_n=5,
    )
    
    return weekly_weights
```

---

## 4. v9 信号生成

### 4.1 主入口

```python
def generate_v9_signals(hs300_index: pd.Series,
                          vix: pd.Series,
                          imfs: np.ndarray,
                          plv: dict,
                          bic_matrix: np.ndarray) -> pd.Series:
    """
    生成 v9 大盘信号时序.
    
    参数:
        hs300_index: 沪深300 等权指数
        vix: VIX 时序
        imfs: (K, T) IMF 数组
        plv: 相位锁定值字典
        bic_matrix: 双相干矩阵
    
    返回:
        v9_signal: (T,) 大盘信号 0/1
    """
    # 1. 计算评分时序
    score_series = compute_score_timeseries(
        imfs=imfs,
        plv=plv,
        bic_matrix=bic_matrix,
        vix=vix,
    )
    
    # 2. 迟滞信号
    signal = score_to_signal_hysteresis(
        score_series=score_series,
        upper=50,
        lower=30,
    )
    
    return signal
```

### 4.2 周频对齐

v9 信号可能是日频, 需要对齐到周频 (与 v7.14 一致):

```python
def align_v9_to_weekly(v9_signal_daily: pd.Series,
                        weekly_dates: pd.DatetimeIndex) -> pd.Series:
    """
    日频 v9 信号 → 周频.
    
    取每周最后一个交易日的信号.
    """
    v9_weekly = v9_signal_daily.resample('W').last()
    return v9_weekly.reindex(weekly_dates, method='ffill')
```

---

## 5. 仓位合成

### 5.1 主函数

```python
def synthesize_position(v9_signal: pd.Series,
                         v7_weights: pd.DataFrame,
                         v8_factors: pd.DataFrame,
                         cash_yield: float = 0.02) -> pd.DataFrame:
    """
    三层仓位合成.
    
    参数:
        v9_signal: (T_weekly,) 大盘信号 0/1
        v7_weights: (T_weekly, N_etf) v7.14 周权重
        v8_factors: (T_weekly, N_etf) v8 仓位因子
        cash_yield: 现金收益率 (年化)
    
    返回:
        final_weights: (T_weekly, N_etf) 最终权重
    """
    assert v9_signal.index.equals(v7_weights.index)
    assert v7_weights.index.equals(v8_factors.index)
    
    # 1. 三层相乘
    final = pd.DataFrame(
        index=v7_weights.index,
        columns=v7_weights.columns,
    )
    
    for date in v7_weights.index:
        v9 = v9_signal.loc[date]
        v7 = v7_weights.loc[date]
        v8 = v8_factors.loc[date]
        
        if v9 == 0:
            # 大盘看空 → 全现金
            final.loc[date] = 0
        else:
            # 大盘看多 → 三层叠加
            final.loc[date] = v9 * v7 * v8
    
    return final
```

### 5.2 归一化与现金

```python
def normalize_weights(final_weights: pd.DataFrame) -> tuple:
    """
    归一化权重, 处理负值和现金.
    
    返回:
        normalized_weights: 归一化权重
        cash_weights: 现金权重
    """
    # 1. 截断负值
    final_weights = final_weights.clip(lower=0)
    
    # 2. 行求和
    row_sums = final_weights.sum(axis=1)
    
    # 3. 归一化
    normalized = final_weights.div(row_sums, axis=0).fillna(0)
    
    # 4. 现金权重
    cash = 1 - row_sums
    
    return normalized, cash
```

---

## 6. 集成测试

### 6.1 单元测试

```python
def test_synthesize_position_v9_zero():
    """测试 v9 信号为 0 时, 全部为现金."""
    v9_signal = pd.Series([0], index=[pd.Timestamp('2024-01-01')])
    v7_weights = pd.DataFrame({'A': [0.4], 'B': [0.6]}, index=v9_signal.index)
    v8_factors = pd.DataFrame({'A': [1.0], 'B': [1.0]}, index=v9_signal.index)
    
    result = synthesize_position(v9_signal, v7_weights, v8_factors)
    
    assert result.loc['2024-01-01', 'A'] == 0
    assert result.loc['2024-01-01', 'B'] == 0


def test_synthesize_position_v9_one_full():
    """测试 v9 信号为 1 时, 全部生效."""
    v9_signal = pd.Series([1], index=[pd.Timestamp('2024-01-01')])
    v7_weights = pd.DataFrame({'A': [0.4], 'B': [0.6]}, index=v9_signal.index)
    v8_factors = pd.DataFrame({'A': [1.0], 'B': [0.5]}, index=v9_signal.index)
    
    result = synthesize_position(v9_signal, v7_weights, v8_factors)
    
    # 期望: A=0.4*1.0=0.4, B=0.6*0.5=0.3
    assert np.isclose(result.loc['2024-01-01', 'A'], 0.4)
    assert np.isclose(result.loc['2024-01-01', 'B'], 0.3)
```

### 6.2 集成测试

**对比三种配置**:

| 配置 | v9 | v7.14 | v8 | 期望 Sharpe |
|------|----|----|----|------------|
| v7.14 only | 关 | 开 | 关 | 0.44 (baseline) |
| v7.14 + v8 | 关 | 开 | 开 | ~1.0 |
| v7.14 + v8 + v9 | 开 | 开 | 开 | ≥ 0.5 |

### 6.3 端到端测试

```python
def test_end_to_end_pipeline():
    """端到端流水线测试."""
    # 1. 加载数据
    data = load_v9_data()
    
    # 2. 训练
    v7_weights = compute_v7_weights(data)
    v8_bear = compute_v8_bear_pct(data.daily_returns)
    v9_signal = generate_v9_signals(data)
    
    # 3. 合成
    final = synthesize_position(v9_signal, v7_weights, v8_bear)
    
    # 4. 回测
    nav = backtest(final, data)
    sharpe = compute_sharpe(nav.returns)
    
    # 5. 验证
    assert sharpe >= 0.5, f"Sharpe {sharpe} < 0.5"
```

---

## 7. 集成示例

### 7.1 完整流水线

```python
# scripts/v9/v9_cycle_timing_main.py

from QuantNodes.strategy.momentum_etf_rotation.v9 import (
    load_data,
    preprocess,
    decompose,
    coupling,
    scoring,
    position,
    backtest,
)

def main():
    # 1. 数据加载
    data = load_data()
    
    # 2. 预处理
    hs300 = data.hs300_index
    cycle_residual = preprocess.hp_filter(hs300, lamb=100)
    
    # 3. 周期分解
    imfs, omega, method = decompose.decompose_signal(
        signal=cycle_residual,
        K=4,
        method='both',
    )
    
    # 4. 周期耦合
    phases = coupling.compute_all_phases(imfs)
    delta_phases = coupling.compute_delta_phases(phases)
    plv = coupling.compute_phase_locking_value(delta_phases)
    bic_matrix = coupling.bicoherence(hs300)
    
    # 5. 评分
    score_series = scoring.compute_score_timeseries(
        imfs=imfs,
        plv=plv,
        bic_matrix=bic_matrix,
        vix=data.vix,
    )
    v9_signal = scoring.score_to_signal_hysteresis(score_series)
    
    # 6. v7.14 周权重
    v7_weights = compute_v7_weights(data)
    
    # 7. v8 Bear%
    v8_bear = compute_v8_bear_pct(data.daily_returns)
    v8_factors = v8_bear.apply(position.v8_bear_to_factor)
    
    # 8. 合成
    final_weights = position.synthesize_position(
        v9_signal, v7_weights, v8_factors,
    )
    final_weights, cash = position.normalize_weights(final_weights)
    
    # 9. 回测
    nav = backtest.run_backtest(final_weights, data.daily_returns)
    
    # 10. 报告
    metrics = compute_metrics(nav)
    print(metrics)
    
    return metrics

if __name__ == '__main__':
    main()
```

---

## 8. 与 v8_method_b 的对比

### 8.1 配置对比

| 配置 | v9_factor | v8_method_b | 差异 |
|------|-----------|-------------|------|
| 大盘信号 | 0/1 (新增) | 1 (满仓) | v9 增加空仓选项 |
| v7.14 权重 | ✅ | ✅ | 相同 |
| v8 Bear% | ✅ | ✅ | 相同 |
| 评分权重 | 40+40+20 | - | 新增 |

### 8.2 预期差异

| 维度 | v8_method_b | v9 |
|------|-------------|-----|
| 满仓时间 | ~95% | ~70% (有 30% 空仓期) |
| 最大回撤 | -12.30% | ~-15% (略大, 因空仓延迟) |
| Sharpe | 1.485 | 0.5-1.0 (目标) |

**关键差异**: v9 在大熊市会更早空仓, 但可能错过急反弹。

---

## 9. 风险与对冲

| 风险 | 缓解 |
|------|------|
| v9 信号滞后 | 迟滞宽度 20, 滚动更新 |
| v8/v7 接口不一致 | 严格类型注解 + 单元测试 |
| 归一化错误 | 单元测试 + 边界检查 |
| 数据对齐错误 | 严格日期对齐 + 缺失值检查 |

---

## 10. 实施清单

### 10.1 文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/
├── position.py               # 仓位合成
└── backtest.py               # 回测引擎
```

### 10.2 接口

```python
from QuantNodes.strategy.momentum_etf_rotation.v9.position import (
    v8_bear_to_factor,
    synthesize_position,
    normalize_weights,
    align_v9_to_weekly,
)
```

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中