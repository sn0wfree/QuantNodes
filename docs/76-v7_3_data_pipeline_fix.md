# v7.3 数据管道重构 + v7.3.2 β 预筛选

## 一、问题背景

### 1.1 审计发现的 8 个 Bug

| # | 级别 | Bug | 影响 |
|---|---|---|---|
| 1 | CRITICAL | `resample("W").last().pct_change()` 对收益数据 | Lasso 信号完全失效，β 几乎全零 |
| 2 | CRITICAL | expanded_panel 混合 simple + log return | 56 资产收益类型不一致 |
| 3 | CRITICAL | NAV `(1+log_return).cumprod()` | 数学错误，~1.2%/年误差 |
| 4 | HIGH | `compute_metrics` 用 `freq="W"` | Sharpe 高估 2.2x |
| 5 | HIGH | `run_v7_3_backtest` `>= curr_date` | 1 日前视偏差 |
| 6 | MODERATE | expanded_panel ETF 上市前填充 0 | `pct_change()` 产生 inf |
| 7 | MODERATE | `load_index_prices()` 缺少中债1-3年 | KeyError |
| 8 | LOW | 旧缓存存的是收益不是价格 | 数据不一致 |

### 1.2 根因分析

源 notebook 的 `main_idx.resample('W').last().pct_change(1)` 操作的是**价格水平**。`load_index_panel()` 被重构为返回对数收益后，`run_v7_3_backtest` 仍沿用原写法，但语义已完全改变。

## 二、修复方案

### 2.1 核心原则

**数据导入只返回价格/净值，不返回收益。收益计算在策略层完成。**

### 2.2 数据流（修复后）

```
Excel 价格 → load_aligned_prices() → 价格 DataFrame
                                        ↓
                              prices.resample("W").last().pct_change()  → 周频 simple return (信号)
                              prices.pct_change()                        → 日频 simple return (NAV)
                              factor_nav.pct_change()                    → 周频 simple return (因子)
```

### 2.3 改动文件清单

| 文件 | 改动 |
|---|---|
| `data_loader.py` | 新增 `load_aligned_prices(pool)`，删除 4 个旧函数 |
| `macro_substrategy_v7_3.py` | `run_v7_3_backtest()` 签名改为接收价格；line 549 `>=` → `>`；新增 v7.3.2 β 预筛选 |
| `__init__.py` | 更新导出列表，新增 `v7_3_2_expanded_tf()` |
| `adapters.py` | 适配新 API |
| 6 个测试文件 | 全部改用 `load_aligned_prices()` |
| 4 个脚本 | 全部改用新 API |
| 3 个缓存文件 | 删除（`v9_indices_daily.parquet`, `v56_expanded_daily.parquet`, `v9_factors_weekly_returns.parquet`） |

### 2.4 关键代码改动

#### data_loader.py

```python
# 新增公共函数
def load_aligned_prices(pool="index"|"expanded", start="2008-01-01"):
    """返回 {"asset_prices": DataFrame, "factor_nav": DataFrame, "benchmark": Series}"""
    # pool="index": 13 指数日价格 (从 Excel)
    # pool="expanded": 51 ETF NAV + 5 债券指数价格
    # factor_nav: 8 因子周频净值
    # benchmark: 沪深300日价格
```

#### macro_substrategy_v7_3.py

```python
# run_v7_3_backtest 签名变化
# 旧: run_v7_3_backtest(index_panel=日收益, factor_panel=周收益, ...)
# 新: run_v7_3_backtest(asset_prices=日价格, factor_nav=周净值, ...)

# 信号计算 (从价格)
asset_weekly_ret = asset_prices.resample("W").last().pct_change()
factor_weekly_ret = factor_nav.pct_change()

# NAV 计算 (从价格)
daily_returns = asset_prices.pct_change()
nav = (1 + daily_returns @ weights).cumprod()

# 1 日前视修复
mask = (daily_returns.index > curr_date) & (daily_returns.index < next_date)
```

## 三、v7.3.2：β 预筛选 + 分散度约束

### 3.1 设计

借鉴 CICC 方案（v0-v1），在 FRP 优化前增加 β 预筛选：

```python
V7_4Config 新增参数:
    n_assets: int = 15              # 最大持仓数
    div_a_share_max: int = 8        # A股 ≤ 8
    div_hk_max: int = 2             # 港股 ≤ 2
    div_commodity_min: int = 1      # 商品 ≥ 1
    div_commodity_max: int = 3      # 商品 ≤ 3
    div_overseas_min: int = 1       # 海外 ≥ 1
    div_overseas_max: int = 3       # 海外 ≤ 3
    div_bond_min: int = 1           # 债券 ≥ 1
    div_bond_max: int = 5           # 债券 ≤ 5
```

### 3.2 流程

```
56 资产 β 矩阵 (56×8)
  ↓ |β|_1 排序 + 分散度约束
15 资产 β 矩阵 (15×8)
  ↓ FRP 优化
15 资产权重
```

### 3.3 结论

修复数据管道后，FRP 已自然稀疏（6-8 只活跃资产），β 预筛选未触发。代码保留作为防御性约束。

## 四、业绩对比

### 4.1 三策略对比（bootstrap=100, freq="D"）

#### Full Period (2010~2026)

| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar | OOS Sharpe | OOS Calmar |
|---|---|---|---|---|---|---|---|
| baseline (13指数) | +6.08% | 9.69% | 0.628 | -31.44% | 0.193 | 1.321 | 1.015 |
| **+TF (13指数)** | +6.67% | 8.07% | 0.827 | -24.45% | 0.273 | **1.615** | **1.972** |
| v7.3.2 (expanded) | +7.92% | 6.97% | **1.137** | **-9.97%** | **0.795** | 1.195 | 1.135 |

#### 2018-2026

| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| baseline | +9.13% | 8.43% | 1.083 | -12.53% | 0.729 |
| **+TF** | +9.45% | 7.72% | **1.224** | **-7.32%** | **1.291** |
| v7.3.2 | +7.92% | 6.97% | 1.137 | -9.97% | 0.795 |

#### 2020-2026

| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| baseline | +9.73% | 8.47% | 1.149 | -12.53% | 0.777 |
| **+TF** | +10.38% | 7.84% | **1.323** | **-7.32%** | **1.418** |
| v7.3.2 | +8.34% | 7.03% | 1.186 | -9.97% | 0.836 |

### 4.2 逐年收益率

| 年份 | baseline | +TF | v7.3.2 | 熊市天数% |
|---|---|---|---|---|
| 2018 | +2.7% | +6.2% | -2.1% | 77.8% |
| 2019 | +13.3% | +8.1% | +16.2% | 12.3% |
| 2020 | +9.1% | +9.1% | +8.5% | 18.7% |
| 2021 | +4.9% | +4.9% | +7.6% | 52.5% |
| 2022 | -1.8% | +0.9% | -0.5% | 100.0% |
| 2023 | +7.2% | +8.7% | +8.0% | 67.3% |
| 2024 | +15.8% | +15.6% | +8.4% | 56.1% |
| 2025 | +21.0% | +21.0% | +18.6% | 3.1% |
| 2026 | +7.4% | +7.3% | +3.5% | 8.4% |

### 4.3 分析

**+TF(13)** 在 OOS 段表现最优（Calmar 1.972），2022/2024/2026 年正贡献显著。机制：熊市减仓债券对冲，"砍左尾不影响右尾"。

**v7.3.2(expanded)** 全期风险调整更优（Sharpe 1.137，Vol 6.97%，MaxDD -9.97%），2019/2021 年收益更高。但 OOS 段不如 +TF(13)。

## 五、测试验证

- 65/65 非慢速测试通过
- ruff check 通过（新增代码无新错误）
- 3 个缓存文件已删除，新缓存自动生成

## 六、遗留问题

1. **v7.3.2 β 预筛选冗余**：修复数据管道后 FRP 已自然稀疏，n_assets=15 未触发
2. **expanded ETF 上市前 NaN**：部分 ETF 2018-2022 期间 NaN >30%，影响因子估计
3. **bootstrap 敏感性**：expanded 池 bootstrap 次数对结果影响较大（30 vs 100 vs 500）
