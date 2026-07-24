# v10 自上而下完备框架 — 架构设计 (讨论稿)

> 日期: 2026-07-22
> 状态: 📋 讨论中
> 基于: v1-v9 全部策略核心因子/措施分析 (docs/54)

---

## 一、设计目标

v10 要做一个**自上而下的完备框架**,整合 v1-v9 所有优点:

- **宏观层**: v7 TV-PR 时变β + v9 银河因子配置
- **行业轮动**: v9 中信行业轮动 + 估值/基本面/regime 条件 (加强)
- **风格轮动**: v4 IC 驱动因子择时 + v9 中信多因子 + regime 条件 (加强)
- **因子选股**: 横截面 5 风格因子打分
- **风险管理**: v8 Jump Model + v9 动态仓位
- **组合构建**: v9 风险平价 + 因子加权 + 仓位缩放

## 二、自上而下 4 层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 4: 组合构建 (Portfolio Construction)                          │
│  ─ base: 风险平价 (v9)                                               │
│  ─ tilt: industry_rotation × style_rotation × factor_scores         │
│  ─ scale: position_size (v8+v9)                                     │
│  ─ final = position × risk_parity × tilt                            │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3: 风险管理 (Risk Management)                                 │
│  ─ Jump Model (v8): DD/Sortino → bull/bear                          │
│  ─ 动态仓位 (v9): pos = (0.7 - 0.5·z).clip(0.2, 1.0)              │
│  ─ position_size ∈ [0.2, 1.0]                                       │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2: 行业+风格轮动 (Industry & Style Rotation)  ← 加强          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 2A: 行业轮动 (23 个行业 ETF)                                   │  │
│  │   - 动量 (12-1 月)                                             │  │
│  │   - 估值 (P/E 分位数, 反向)                                     │  │
│  │   - 基本面 (营收增速, ROE)                                      │  │
│  │   - 质量 (波动率, Sharpe)                                       │  │
│  │   - Regime 条件: bull→进攻型, bear→防御型                       │  │
│  │   - 相关性约束: 避免冗余行业                                    │  │
│  │   → sector_weights (行业配置)                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 2B: 风格轮动 (6 种风格)                                        │  │
│  │   - Value / Growth / Momentum / Quality / Size / Volatility    │  │
│  │   - IC 加权 (v4): 滚动 IC → 因子权重                           │  │
│  │   - Regime 条件: bull→growth+mom, bear→value+quality           │  │
│  │   - Factor timing: 哪个风格当前最有效                           │  │
│  │   → style_weights (风格配置)                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 2C: 因子选股 (43 ETF 横截面)                                   │  │
│  │   - 用 style_weights 加权不同因子                              │  │
│  │   - 横截面 z-score → 综合得分                                  │  │
│  │   - softmax 加权 → 候选权重                                    │  │
│  │   → factor_scores (个股得分)                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1: 宏观择时 (Macro Regime)                                    │
│  ─ TV-PR (v7): 时变 β → 宏观状态                                    │
│  ─ 银河因子配置 (v9): 17 因子 → 熵权得分                            │
│  ─ 输出: macro_signal + regime_state                                │
│  ─ 作用: 驱动 Layer 2 的 regime 条件                                │
└──────────────────────────────────────────────────────────────────────┘
```

## 三、各层详细设计

### Layer 1: 宏观择时 (Macro Regime)

**来源**: v7 TV-PR + v9 银河因子配置

| 组件 | 输入 | 输出 | 方法 |
|------|------|------|------|
| TV-PR | 17 macro factors + 43 ETF returns | 时变 β_t | expanding_window_tvpr (v7) |
| 银河因子配置 | 17 macro factors | 因子得分 | 熵权法 + 风险预算 (v9) |
| 组合 | β_t + 因子得分 | macro_signal | z-score 标准化后平均 |

**输出**: macro_signal ∈ [-1, +1], regime_state ∈ {bull, neutral, bear}

**Regime 定义**:
- bull: macro_signal > 0.5
- neutral: -0.5 ≤ macro_signal ≤ 0.5
- bear: macro_signal < -0.5

### Layer 2A: 行业轮动 (Industry Rotation)

**来源**: v9 中信行业轮动 + 加强

| 因子 | 来源 | 计算方式 | 权重 | Regime 条件 |
|------|------|----------|------|-------------|
| 动量 | v9 中信 | 12-1 月累计收益 | 1.0 | bull: 1.5x, bear: 0.5x |
| 估值 | 新增 | P/E 分位数 (反向) | 1.0 | bear: 2.0x |
| 基本面 | 新增 | 营收增速 + ROE | 0.5 | bull: 1.5x |
| 质量 | v9 中信 | 波动率 + Sharpe | 0.5 | bear: 2.0x |
| 相关性约束 | 新增 | 行业间相关系数 > 0.7 → 剔除 | — | — |

**输出**: sector_weights (23 个行业 ETF 的权重)

**算法**:
1. 计算 4 个因子的横截面 z-score
2. 根据 regime_state 调整因子权重
3. 加权合成综合得分
4. 相关性约束: 剔除相关系数 > 0.7 的行业中得分较低者
5. Top-K 选优 (K=5-8)
6. softmax 加权 → sector_weights

### Layer 2B: 风格轮动 (Style Rotation)

**来源**: v4 IC 驱动因子择时 + v9 中信多因子 + 加强

| 风格 | 因子 | IC 加权窗口 | Regime 条件 |
|------|------|-------------|-------------|
| Value | P/B, P/E, 股息率 | 60 周 | bear: +50% |
| Growth | 营收增速, 盈利增速 | 120 周 | bull: +50% |
| Momentum | 12-1 月收益 | 120 周 | bull: +30% |
| Quality | ROE, 毛利率 | 180 周 | bear: +30% |
| Size | 市值 (反向) | 60 周 | — |
| Volatility | 波动率 (反向) | 60 周 | bear: +50% |

**输出**: style_weights (6 种风格的权重)

**算法**:
1. 计算每个风格的滚动 IC (与未来 4 周收益的相关性)
2. IC 加权: weight = max(0, IC + base)^power
3. 根据 regime_state 调整风格权重
4. 归一化 → style_weights

### Layer 2C: 因子选股 (Factor Scoring)

**来源**: v9 中信多因子 + 加强

| 因子 | 方向 | 计算 | 加权方式 |
|------|------|------|----------|
| Momentum | + | 12-1 月累计收益 | × style_weights['momentum'] |
| Volatility | - | 26 周 std | × style_weights['volatility'] |
| Quality | + | 26 周 Sharpe | × style_weights['quality'] |
| Size | - | 4 周均振幅 | × style_weights['size'] |
| Value | + | 52-104 周反转 | × style_weights['value'] |

**输出**: factor_scores (43 个 ETF 的得分)

**算法**:
1. 计算 5 个因子的横截面 z-score
2. 用 style_weights 加权: score = Σ(z_factor × style_weight)
3. softmax 加权 → factor_scores

### Layer 3: 风险管理 (Risk Management)

**来源**: v8 Jump Model + v9 动态仓位

| 组件 | 输入 | 输出 | 方法 |
|------|------|------|------|
| Jump Model | ETF returns | bull/bear 概率 | DD_10/Sortino_20/Sortino_60 |
| 动态仓位 | macro_signal + factor_scores | position_size | pos = (0.7 - 0.5·z).clip(0.2, 1.0) |
| 组合 | bull/bear + position_size | final_position | min(jump_pos, dynamic_pos) |

**输出**: position_size ∈ [0.2, 1.0]

**Jump Model 仓位映射**:
- bull: position_size = 1.0 (满仓)
- neutral: position_size = 0.6 (半仓)
- bear: position_size = 0.2 (最低仓)

### Layer 4: 组合构建 (Portfolio Construction)

**来源**: v9 风险平价 + 因子加权

| 组件 | 输入 | 输出 | 方法 |
|------|------|------|------|
| 风险平价 | ETF returns | base_weights | inv_vol / sum(inv_vol) |
| 行业轮动 | sector_weights | industry_tilt | 行业 ETF 权重调整 |
| 风格轮动 | style_weights | style_tilt | 风格因子加权 |
| 因子选股 | factor_scores | factor_tilt | softmax(scores) |
| 仓位缩放 | position_size | final_weights | position × base × tilts |

**输出**: final_weights (43 个 ETF 的最终权重)

**算法**:
1. base_weights = risk_parity(returns)
2. industry_tilt = apply_sector_weights(base_weights, sector_weights)
3. style_tilt = apply_style_weights(industry_tilt, style_weights)
4. factor_tilt = softmax(factor_scores × temperature)
5. final_weights = position_size × base_weights × industry_tilt × style_tilt × factor_tilt
6. 归一化: final_weights = final_weights / sum(final_weights)

## 四、行业轮动加强细节

### 4.1 估值因子 (新增)

由于 ETF 没有直接的 P/E 数据, 用以下代理:
- **指数估值**: 用 ETF 跟踪的指数的 P/E 分位数 (需要额外数据)
- **收益率反向**: 用过去 52 周累计收益的反向作为估值代理
- **波动率反向**: 低波动率 ≈ 价值股特质

### 4.2 基本面因子 (新增)

同样用代理:
- **营收增速**: 无法直接获取, 用动量因子代理
- **ROE**: 用 Sharpe ratio 代理 (高质量 ≈ 高 ROE)

### 4.3 相关性约束

```python
# 行业间相关系数矩阵
corr_matrix = returns[sector_codes].rolling(52).corr()
# 剔除相关系数 > 0.7 的行业中得分较低者
for i in range(n):
    for j in range(i+1, n):
        if corr_matrix.iloc[i, j] > 0.7:
            if scores[i] < scores[j]:
                scores[i] = 0  # 剔除
```

## 五、风格轮动加强细节

### 5.1 IC 加权 (v4 风格)

```python
# 滚动 IC 计算
for style in styles:
    ic = rolling_ic(factor[style], forward_return, window=60)
    # IC 加权
    weight = max(0, ic + base) ** power
# 归一化
style_weights = weights / weights.sum()
```

### 5.2 Regime 条件

```python
# 根据 regime_state 调整风格权重
if regime == 'bull':
    style_weights['growth'] *= 1.5
    style_weights['momentum'] *= 1.3
elif regime == 'bear':
    style_weights['value'] *= 1.5
    style_weights['quality'] *= 1.3
    style_weights['volatility'] *= 1.5
```

## 六、预期性能

| 指标 | v9 银河方案-动态仓位 | v10 预期 (加强后) |
|------|---------------------|-------------------|
| Sharpe | 1.23 | **1.5 - 1.8** |
| Calmar | 1.20 | **1.4 - 1.6** |
| MaxDD | -13.7% | **-8% ~ -11%** |
| 年化收益 | 16.4% | **20% - 25%** |

**提升来源**:
1. 行业轮动加强: 估值+基本面+regime 条件 (+0.1-0.2 Sharpe)
2. 风格轮动加强: IC 加权+regime 条件 (+0.1-0.2 Sharpe)
3. 相关性约束: 避免冗余行业 bets (+0.05-0.1 Calmar)

## 七、文件结构

```
QuantNodes/strategy/momentum_etf_rotation/v10/
├── __init__.py
├── macro_layer.py          # Layer 1: TV-PR + 银河因子配置
├── industry_rotation.py    # Layer 2A: 行业轮动 (加强版)
├── style_rotation.py       # Layer 2B: 风格轮动 (加强版)
├── factor_scoring.py       # Layer 2C: 因子选股 (用 rotation 加权)
├── risk_layer.py           # Layer 3: Jump Model + 动态仓位
├── portfolio_layer.py      # Layer 4: 组合构建
├── v10_strategy.py         # 主入口: 4 层串联
└── backtest_v10.py         # 回测引擎

scripts/v10/
├── v10_backtest.py         # 回测脚本
└── v10_compare.py          # 对比 v9 所有策略
```

## 八、数据需求

| 数据 | 来源 | 频率 | 用途 |
|------|------|------|------|
| 43 ETF 收益 | data/high_freq_macro/v7_10_Y_weekly.parquet | 周频 | Layer 2C, 3, 4 |
| 17 宏观因子 | data/high_freq_macro/v7_6_X_macro_weekly.parquet | 周频 | Layer 1 |
| 行业分类 | CITIC_ETF_CLASSIFICATION (v9) | 静态 | Layer 2A |
| 指数估值 | 需额外数据源 | 周频/月频 | Layer 2A 估值因子 |

## 九、待确认事项

1. **估值因子**: 是否需要接入外部数据 (指数 P/E/P/B)? 还是用代理因子?
2. **基本面因子**: 是否需要接入营收/ROE 数据? 还是用动量/Sharpe 代理?
3. **Jump Model 参数**: 是否沿用 v8 的最优参数? 还是需要重新调优?
4. **TV-PR 参数**: 是否沿用 v7 的 lambda_tv/lambda_l1? 还是需要针对 v10 重新调优?
5. **调仓频率**: 周频还是月频? (周频换手高, 月频可能错过信号)

## 十、下一步

确认上述待确认事项后, 开始实现 v10 核心模块。
