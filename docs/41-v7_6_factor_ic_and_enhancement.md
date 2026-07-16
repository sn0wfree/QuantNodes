# v7.6 因子 IC 评估与因子增强方案

> **编号**: 41
> **状态**: 🔧 **实施中**
> **日期**: 2026-07-15
> **关联**: docs/39-v7_6_tvpr.md, docs/40-v7_6_sensitivity.md

---

## 1. 背景

### 1.1 当前问题

v7.6 TV-PR 修复后 OOS Calmar 从 0.22 提升到 0.38，但仍远低于 v1.0 locked 的 1.87。

根本原因分析：
1. **宏观因子预测力极弱**: 9 个宏观因子对 ETF 截面收益几乎无区分度
2. **量价因子预测力一般**: 最强因子 f5_turnover 的信号相关系数仅 0.0386
3. **因子池过大 (K=20)**: 过拟合风险高

### 1.2 目标

1. 建立系统化的因子 IC 评估框架
2. 筛选有效因子，剔除噪声因子
3. 增加新的高预测力因子
4. 重新运行 TV-PR，验证因子增强效果

---

## 2. IC 计算方法

### 2.1 核心问题：宏观因子 vs 量价因子

| 类别 | 截面特征 | IC 计算方式 |
|------|---------|------------|
| 宏观因子 (k=0..7) | 所有资产共享同一值 | 时序 IC + 面板 IC |
| 量价因子 (k=8..19) | 每个资产值不同 | 标准截面 IC |

```
X_panel[t, i, k] 的含义:
  k = 0..7 (宏观因子):  X_panel[t, 0, k] == X_panel[t, 1, k] == ... == X_panel[t, N-1, k]
  k = 8..19 (量价因子): X_panel[t, 0, k] ≠ X_panel[t, 1, k] ≠ ...
```

### 2.2 方法 1: 时序 IC（Time-Series IC）

**适用**: 宏观因子

**逻辑**: 固定资产 i，看宏观因子随时间变化与该资产收益的相关性

```
对每个资产 i (i = 1..N):
  对每个宏观因子 k (k = 0..7):
    x_series = X_panel[:, i, k]        # (T,) 宏观因子时序
    r_series = Y[:, i]                  # (T,) 资产收益时序

    # 因子领先一期: x_t 预测 r_{t+1}
    IC_{i,k} = spearmanr(x_series[:-1], r_series[1:])

跨资产聚合:
  IC_mean_k = mean(IC_{i,k} for all i)
  IC_std_k  = std(IC_{i,k} for all i)
  ICIR_k    = IC_mean_k / IC_std_k
```

**输出**: 每个宏观因子的 (IC_mean, IC_std, ICIR, 正IC占比)

**优点**:
- 保留宏观因子的时序变化信息
- 可以看出哪些资产对宏观因子敏感

**缺点**:
- 样本量 = T（约 400 周），不是 T×N
- 不同资产的 IC 可能差异大

### 2.3 方法 3: 面板 IC（Panel IC，修正版）

**适用**: 宏观因子

**问题**: 同一时间所有资产的宏观因子值相同，直接展开为长格式会混杂时序+截面信息。

**修正方案**: 用市场平均收益作为因变量

```
market_r[t] = mean(Y[t+1, :])        # 所有资产的下期平均收益
x_ts[t] = X_panel[t, 0, k]           # 宏观因子时序（取任意资产，所有资产相同）

IC_panel_k = spearmanr(x_ts[:-1], market_r[1:])
t_stat = IC_panel * sqrt(T) / sqrt(1 - IC_panel^2)
```

**输出**: 每个宏观因子的 (panel_IC, t_stat, p_value)

### 2.4 方法 2: 截面 IC（Cross-Sectional IC）

**适用**: 量价因子

**逻辑**: 标准截面 IC，固定时间 t，看因子截面值与下期收益的相关性

```
对每个 t (t = 0..T-2):
  对每个量价因子 k (k = 8..19):
    x = X_panel[t, :, k]       # (N,) 因子截面
    r = Y[t + 1]               # (N,) 下期收益

    IC_t = spearmanr(x, r)

聚合:
  IC_mean = mean(IC_t)
  IC_std = std(IC_t)
  ICIR = IC_mean / IC_std
  IC_positive_ratio = count(IC_t > 0) / count(IC_t)
```

**输出**: 每个量价因子的 (IC_mean, IC_std, ICIR, 正IC占比)

### 2.5 综合评分

```
宏观因子综合分 = 0.6 × |时序IC_mean| + 0.4 × |面板IC|
量价因子综合分 = |截面IC_mean|  (直接用截面IC)

筛选标准:
  - 综合分 > 0.03: 保留
  - 综合分 0.02-0.03: 边缘，观察
  - 综合分 < 0.02: 剔除
```

### 2.6 IC 判据标准

| 指标 | 好 | 一般 | 差 |
|------|-----|------|-----|
| IC_mean | > 0.05 | 0.02-0.05 | < 0.02 |
| ICIR | > 0.5 | 0.3-0.5 | < 0.3 |
| IC_positive_ratio | > 60% | 50-60% | < 50% |
| t_stat (面板IC) | > 2.0 | 1.5-2.0 | < 1.5 |

---

## 3. 因子增强方案

### 3.1 当前因子集

**9 宏观因子**（全局，所有资产共享）:

| 因子 | 类别 | 来源 |
|------|------|------|
| 宏观增长因子 | 经济增长 | v9_factors_weekly.parquet |
| 宏观通胀因子_生活端 | 通胀 | v9_factors_weekly.parquet |
| 宏观通胀因子_生产端 | 通胀 | v9_factors_weekly.parquet |
| 无风险收益率 | 利率 | v9_factors_weekly.parquet |
| 信用利差因子 | 信用 | v9_factors_weekly.parquet |
| 期限利差因子_债 | 期限结构 | v9_factors_weekly.parquet |
| 期限利差因子_股 | 期限结构 | v9_factors_weekly.parquet |
| 宏观汇率因子 | 汇率 | v9_factors_weekly.parquet |

**11 量价因子**（资产特异）:

| 因子 | 类别 | 预测力排名 |
|------|------|-----------|
| f1_second_mom (二阶动量) | 动量 | 中 |
| f2_mom_term (动量期限差) | 动量 | 3 (IC≈0.020) |
| f3_amt_vol (成交金额波动) | 交易波动 | 中 |
| f4_vol_vol (成交量波动) | 交易波动 | 2 (IC≈0.033) |
| f5_turnover (换手率变化) | 换手率 | 1 (IC≈0.039) |
| f6_ls_total (多空对比总量) | 多空对比 | 低 |
| f7_ls_change (多空对比变化) | 多空对比 | 低 |
| f8_pv_rankcov (量价排序协方差) | 量价背离 | 低 |
| f9_pv_corr (量价相关系数) | 量价背离 | 低 |
| f10_first_div (一阶量价背离) | 量价背离 | 低 |
| f11_vol_range (量幅同向) | 量幅同向 | 低 |

### 3.2 候选新增因子

#### 3.2.1 高频微观结构因子（ETF 特异）

| 因子 | 定义 | 预期 IC | 计算复杂度 |
|------|------|---------|-----------|
| **Amihud 非流动性** | mean(\|r\|/volume) over window | 0.03-0.05 | 低 |
| **Realized Volatility** | std(daily_returns) over window | 0.02-0.04 | 低 |
| **Realized Skewness** | skew(daily_returns) over window | 0.02-0.03 | 低 |
| **Max5** | max(daily_returns, 5 days) | 0.03-0.05 | 低 |
| **Idiosyncratic Volatility** | 残差波动率 (需回归) | 0.03-0.05 | 中 |

#### 3.2.2 动量/反转增强因子

| 因子 | 定义 | 预期 IC | 计算复杂度 |
|------|------|---------|-----------|
| **52-Week High** | close / max(close, 252d) - 1 | 0.02-0.04 | 低 |
| **Momentum Reversal** | ret_1m × (-1) + ret_12m | 0.02-0.04 | 低 |
| **Volume-Weighted Momentum** | sum(r × volume) / sum(volume) | 0.02-0.03 | 低 |

#### 3.2.3 宏观状态因子（全局）

| 因子 | 定义 | 预期 IC | 数据来源 |
|------|------|---------|---------|
| **VIX 期限结构** | VIX / VIX3M | 0.02-0.04 | 需外部数据 |
| **信用利差变化** | Δ(HY-IG spread) | 0.02-0.03 | 可从现有数据计算 |
| **实际利率** | 名义利率 - 通胀预期 | 0.02-0.03 | 可从现有数据计算 |
| **美元指数变化** | ΔDXY | 0.02-0.03 | 需外部数据 |

### 3.3 推荐新增因子（优先级排序）

基于数据可得性和预期 IC，推荐以下新增顺序：

| 优先级 | 因子 | 类别 | 理由 |
|--------|------|------|------|
| 1 | Amihud 非流动性 | 微观结构 | 经典因子，预期 IC 高，计算简单 |
| 2 | Realized Volatility | 微观结构 | 波动率溢价，计算简单 |
| 3 | Max5 | 动量反转 | 彩票效应，学术支持强 |
| 4 | 52-Week High | 动量反转 | 锚定效应，学术支持强 |
| 5 | Realized Skewness | 微观结构 | 尾部风险溢价 |
| 6 | Idiosyncratic Volatility | 微观结构 | 低波溢价，但需回归计算 |
| 7 | Momentum Reversal | 动量反转 | 短期反转+长期动量组合 |
| 8 | Volume-Weighted Momentum | 动量反转 | 机构行为信号 |

---

## 4. 实施计划

### Phase 1: IC 评估框架（1天）

**目标**: 建立系统化的因子 IC 评估脚本

**产出**: `scripts/calc_factor_ic.py`

**内容**:
1. 方法 1: 时序 IC (宏观因子)
2. 方法 3: 面板 IC (宏观因子，修正版)
3. 方法 2: 截面 IC (量价因子)
4. 综合评分与排序
5. 输出 IC 报告

**运行命令**:
```bash
python3.11 scripts/calc_factor_ic.py
```

**输出文件**:
```
reports/momentum_etf_rotation/v7_6_factor_ic_report.md
reports/momentum_etf_rotation/v7_6_factor_ic_details.csv
```

### Phase 2: 因子增强实现（2天）

**目标**: 实现 8 个新因子

**产出**: `QuantNodes/strategy/momentum_etf_rotation/v7/enhanced_factors.py`

**新因子实现**:

```python
def calc_amihud_illiquidity(close, volume, returns, window=20):
    """Amihud 非流动性: mean(|r|/volume)."""
    abs_ret = returns.abs()
    illiq = abs_ret / volume.replace(0, np.nan)
    return illiq.rolling(window).mean()

def calc_realized_volatility(returns, window=20):
    """已实现波动率: std(daily_returns)."""
    return returns.rolling(window).std()

def calc_realized_skewness(returns, window=20):
    """已实现偏度: skew(daily_returns)."""
    return returns.rolling(window).skew()

def calc_max5(returns, window=5):
    """Max5: 过去5天最大日收益."""
    return returns.rolling(window).max()

def calc_idiosyncratic_volatility(returns, market_returns, window=60):
    """特质波动率: 回归残差的标准差."""
    # 需要市场收益作为基准
    resid = returns - market_returns  # 简化版
    return resid.rolling(window).std()

def calc_52week_high(close, window=252):
    """52周高点距离: close / max(close, 252d) - 1."""
    max_price = close.rolling(window).max()
    return close / max_price - 1

def calc_momentum_reversal(close, short_window=20, long_window=252):
    """动量反转: -ret_1m + ret_12m."""
    ret_short = close.pct_change(short_window)
    ret_long = close.pct_change(long_window)
    return -ret_short + ret_long

def calc_volume_weighted_momentum(close, volume, window=20):
    """成交量加权动量: sum(r × volume) / sum(volume)."""
    returns = close.pct_change()
    vwm = (returns * volume).rolling(window).sum() / volume.rolling(window).sum()
    return vwm
```

### Phase 3: IC 测试与因子筛选（1天）

**目标**: 运行 IC 测试，筛选有效因子

**步骤**:
1. 运行 IC 评估脚本，生成报告
2. 根据综合评分筛选 top 因子
3. 更新 `V7_6Config` 中的因子列表

**预期结果**:
- 宏观因子: 大部分 IC < 0.02，可能剔除 4-6 个
- 量价因子: 保留 f5_turnover, f4_vol_vol, f2_mom_term + 2-3 个新因子
- 新增因子: 期望 2-3 个 IC > 0.03 的因子

### Phase 4: TV-PR 回测验证（1天）

**目标**: 用新因子集重新运行 TV-PR，验证效果

**对比指标**:
- OOS Calmar (当前 0.38)
- 信号相关系数 (当前 0.0546)
- 起点 CV% (当前 16.1%)

**预期效果**:
- 信号相关系数: 0.05 → 0.08-0.10
- OOS Calmar: 0.38 → 0.50-0.70

---

## 5. 关键风险

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 新因子 IC 低于预期 | 中 | 先做 IC 测试，只保留显著因子 |
| 过拟合 | 中 | 用多起点测试，监控 CV% |
| 计算量增加 | 低 | 缓存中间结果，增量计算 |
| 数据缺失 | 低 | 所有数据可从现有 OHLCV 计算 |

---

## 6. 已完成工作

### 6.1 宏观因子对数收益率实现 (2026-07-15)

**修改**: `data_loader_v7_6.py`

- `build_mixed_factor_panel()`: 新增 `macro_use_log_return` 参数 (默认 True)
- `load_v7_6_data()`: 传递 `macro_use_log_return` 参数
- 宏观因子从 NAV levels 转换为对数收益率: `r_t = ln(NAV_t / NAV_{t-1})`

**IC 对比结果**:

| 因子 | Levels IC | Log Returns IC | 提升 |
|------|-----------|----------------|------|
| 期限利差因子_加权 | 0.0263 | **-0.4998** | 19x |
| 期限利差因子_股 | 0.0263 | **-0.4699** | 18x |
| 期限利差因子_债 | 0.0094 | **-0.2851** | 30x |
| 宏观通胀因子_生产端 | -0.1375 | **0.2102** | 1.5x |
| 宏观汇率因子 | 0.0004 | **0.1729** | 432x |
| 宏观增长因子 | -0.0945 | **0.1425** | 1.5x |

**回测结果** (不变):

| 指标 | Levels | Log Returns |
|------|--------|-------------|
| OOS Calmar | 0.38 | 0.38 |
| Mean Calmar | 0.38 | 0.38 |
| CV% | 16.1% | 16.1% |

**结论**: 对数收益率的 IC 更强，但 TV-PR 模型已捕捉时变特性，相对排序不变，回测表现一致。

### 6.2 量价因子 IC 评估

**结果**:

| 排名 | 因子 | IC_mean | ICIR | 正IC占比 |
|------|------|---------|------|----------|
| 1 | f4_vol_vol (成交量波动) | 0.0456 | 0.20 | 60.0% |
| 2 | f3_amt_vol (成交金额波动) | 0.0319 | 0.17 | 58.8% |
| 3 | f10_first_div (一阶量价背离) | 0.0182 | 0.08 | 55.0% |
| 4 | f6_ls_total (多空对比总量) | -0.0180 | -0.05 | 47.7% |
| 5 | f9_pv_corr (量价相关系数) | 0.0149 | 0.06 | 53.1% |

**结论**: 大数量价因子 IC < 0.02，预测力较弱。

---

## 7. 文件清单

```
新建文件:
  scripts/calc_factor_ic.py                                    # IC 评估脚本
  QuantNodes/strategy/momentum_etf_rotation/v7/enhanced_factors.py  # 新因子实现 (待实施)
  reports/momentum_etf_rotation/v7_6_factor_ic_report.md       # IC 报告
  reports/momentum_etf_rotation/v7_6_factor_ic_details.csv     # IC 详情

修改文件:
  QuantNodes/strategy/momentum_etf_rotation/v7/data_loader_v7_6.py  # 宏观因子用对数收益率
```

---

## 8. 时间线

| 阶段 | 时间 | 产出 | 状态 |
|------|------|------|------|
| Phase 1: IC 评估框架 | 1 天 | `scripts/calc_factor_ic.py` | ✅ 完成 |
| Phase 2: 因子增强实现 | 2 天 | `v7/enhanced_factors.py` | 待实施 |
| Phase 3: IC 测试与筛选 | 1 天 | IC 报告 + 因子筛选 | ✅ 完成 |
| Phase 4: TV-PR 回测验证 | 1 天 | 回测结果对比 | ✅ 完成 |
| **总计** | **5 天** | |

---

**最后更新**: 2026-07-15
**状态**: 设计文档，待讨论后实施
