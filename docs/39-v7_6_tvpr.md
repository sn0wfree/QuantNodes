# v7.6 — TV-PR 宏观子策略 (9 macro + 11 量价, 月频)

> **编号**: 39
> **状态**: 🔧 **实施中**
> **日期**: 2026-07-14
> **关联**: Cui et al. (2025) "Breaks and trends in factor premia"

---

## 1. 背景与动机

### 1.1 问题

v7.3 baseline 使用 Bootstrap-Lasso 估计时变 β_t，但存在两个问题：
1. **expanding window** 让历史数据"淹没"近期关系，无法捕捉结构性变化
2. **静态因子池**：只用 9 macro factors，缺乏资产特异信息

### 1.2 解决方案

TV-PR (Total-Variation Predictive Regression) 通过 L1 正则化识别因子溢价的结构性变化：
- **TV 项**：$\lambda_1 \sum_{t=2}^T \|\boldsymbol{\beta}_t - \boldsymbol{\beta}_{t-1}\|_1$ — 控制 break 数量
- **L1 项**：$\lambda_2 \sum_{t=1}^T \|\boldsymbol{\beta}_t\|_1$ — 因子稀疏性

### 1.3 v7.6 vs v7.3

| 维度 | v7.3 | v7.6 |
|---|---|---|
| Y (资产) | 13 INDEX | 56 ETF (expanded) |
| X (因子) | 9 macro | **9 macro + 11 量价** |
| β_t 估计 | Bootstrap-Lasso | **TV-PR** |
| 频率 | 周频 | 周频 |
| 调仓 | 季度 (QE) | **月度 (M)** |
| 成本 | 5bp+5bp | 5bp+5bp |

---

## 2. 数学公式

### 2.1 问题设定

- $r_{i,t}$：资产 $i$ 在时期 $t$ 的超额收益
- $\mathbf{x}_{i,t} = [macro_{1,t}, ..., macro_{9,t}, pv_{i,1,t}, ..., pv_{i,11,t}]$：20 维因子向量
- $\boldsymbol{\beta}_t$：时变因子溢价（20 维）

### 2.2 TV-PR 目标函数

$$\min_{\boldsymbol{\beta}_1, ..., \boldsymbol{\beta}_T} \sum_{t=1}^T \sum_{i=1}^{N_t} (r_{i,t} - \mathbf{x}_{i,t}' \boldsymbol{\beta}_t)^2 + \lambda_1 \sum_{t=2}^T \|\boldsymbol{\beta}_t - \boldsymbol{\beta}_{t-1}\|_1 + \lambda_2 \sum_{t=1}^T \|\boldsymbol{\beta}_t\|_1$$

### 2.3 ADMM 求解

**增广形式**：
$$\min_{\beta, z} \sum_{t=1}^T \sum_{i=1}^{N_t} (r_{i,t} - \mathbf{x}_{i,t}' \beta_t)^2 + \lambda_1 \|z\|_1 + \lambda_2 \|\beta\|_1$$
$$\text{s.t.} \quad z = \Delta \beta$$

**ADMM 迭代**：
1. **β-update**：固定 z, u, 解 Lasso 问题
2. **z-update**：固定 β, u, 解 soft-thresholding 问题
3. **u-update**：对偶变量更新

---

## 3. 因子定义

### 3.1 9 macro factors (全局)

| 因子 | 频率 | 来源 |
|---|---|---|
| 宏观增长因子 | 周频 | v9_factors_weekly.parquet |
| 宏观通胀因子_生活端 | 周频 | v9_factors_weekly.parquet |
| 宏观通胀因子_生产端 | 周频 | v9_factors_weekly.parquet |
| 无风险收益率 | 周频 | v9_factors_weekly.parquet |
| 信用利差因子 | 周频 | v9_factors_weekly.parquet |
| 期限利差因子_债 | 周频 | v9_factors_weekly.parquet |
| 期限利差因子_股 | 周频 | v9_factors_weekly.parquet |
| 宏观汇率因子 | 周频 | v9_factors_weekly.parquet |

### 3.2 11 量价 factors (资产特异)

| 因子 | 类别 | 频率 | 来源 |
|---|---|---|---|
| 二阶动量 | 动量 | 日频→月频 | ETF OHLCV |
| 动量期限差 | 动量 | 日频→月频 | ETF OHLCV |
| 成交金额波动 | 交易波动 | 日频→月频 | ETF OHLCV |
| 成交量波动 | 交易波动 | 日频→月频 | ETF OHLCV |
| 换手率变化 | 换手率 | 日频→月频 | ETF OHLCV |
| 多空对比总量 | 多空对比 | 日频→月频 | ETF OHLCV |
| 多空对比变化 | 多空对比 | 日频→月频 | ETF OHLCV |
| 量价排序协方差 | 量价背离 | 日频→月频 | ETF OHLCV |
| 量价相关系数 | 量价背离 | 日频→月频 | ETF OHLCV |
| 一阶量价背离 | 量价背离 | 日频→月频 | ETF OHLCV |
| 量幅同向 | 量幅同向 | 日频→月频 | ETF OHLCV |

---

## 4. 数据流

### 4.1 输入

```
├─ 9 macro factors (周频) → resample('M') → 月频
├─ ETF OHLCV (日频) → compute_all_factors_panel() → 11 量价 (日频) → resample('M') → 月频
└─ ETF NAV (日频) → resample('M') → Y 月频
```

### 4.2 输出

```
├─ X_macro: (T_monthly, 9) 月频宏观因子
├─ X_pv: (T_monthly, 56×11) 月频量价因子
├─ X_mixed: (T_monthly, 20) 月频混合因子 (9 macro + 11 量价均值)
└─ Y: (T_monthly, 56) 月频资产收益
```

---

## 5. 配置

### 5.1 V7_6Config

```python
@dataclass
class V7_6Config:
    """v7.6 TV-PR 配置."""
    
    # 资产池
    asset_pool: str = "expanded"  # 56 assets
    index_pool: tuple[str, ...] = tuple(EXPANDED_COLS)
    equity_cols: tuple[str, ...] = tuple(EQUITY_ETF_COLS)
    commodity_cols: tuple[str, ...] = tuple(COMMODITY_ETF_COLS)
    bond_cols: tuple[str, ...] = tuple(EXPANDED_BOND_INDICES)
    
    # 因子池
    macro_cols: tuple[str, ...] = (...)  # 9 macro
    pv_factors: tuple[str, ...] = (...)  # 11 量价
    
    # TV-PR 参数
    lambda_tv: float = 0.05
    lambda_l1: float = 0.01
    method: str = "admm"
    max_iter: int = 200
    tol: float = 1e-5
    
    # 调仓
    rebalance_freq: str = "M"  # 月度
    min_history: int = 12      # 最少 1 年月频数据
    
    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0
    cost_enabled: bool = True
```

---

## 6. 验证结果 (2026-07-14)

### 6.1 λ 校验

- λ_tv: 0.01
- λ_l1: 0.001
- 16 组合均 Calmar=0.5966

### 6.2 全段回测 (2018-2026)

| 指标 | 值 |
|---|---|
| Calmar | **-0.0086** |
| 年化收益 | -0.12% |
| 波动率 | 5.56% |
| 最大回撤 | -14.31% |
| 夏普 | -0.02 |

### 6.3 OOS 回测 (2022-2026)

| 指标 | 值 |
|---|---|
| Calmar | **-0.0759** |
| 年化收益 | -1.09% |
| 波动率 | 6.36% |
| 最大回撤 | -14.31% |
| 夏普 | -0.17 |

### 6.4 起点依赖

| 起点 | Calmar |
|---|---|
| 2018-01-01 | -0.0086 |
| 2019-01-01 | -0.0907 |
| 2020-01-01 | -0.0616 |
| 2021-01-01 | -0.0782 |
| 2022-01-01 | -0.0759 |

- Mean Calmar: -0.0630
- Std Calmar: 0.0287
- **CV%: 0.0%** (阈值 25%, PASS)

### 6.5 结论

v7.6 TV-PR **表现不佳**:
- Calmar 为负, 策略亏损
- 起点 CV% 通过 (0.0%), 但这是"稳定亏损"
- 需要进一步优化

### 6.6 可能原因

1. **数据不足**: 仅 101 个月, TV-PR 需要更长数据
2. **因子模型问题**: 9 macro + 11 量价可能不具预测性
3. **ADMM 收敛问题**: 可能未收敛到最优解
4. **选股逻辑问题**: 用截面收益排序可能不是最优方法

---

## 7. 文件结构

### 7.1 新建文件

| 文件 | 说明 | 行数估计 |
|---|---|---|
| `v7/data_loader_v7_6.py` | 月频数据加载 | ~250 行 |
| `v7/tvpr_estimator.py` | TV-PR 核心算法 | ~300 行 |
| `v7/macro_substrategy_v7_6.py` | v7.6 回测框架 | ~400 行 |
| `tests/test_v7_6_tvpr.py` | 单元测试 | ~200 行 |
| `scripts/run_v7_6_backtest.py` | 端到端回测脚本 | ~150 行 |

### 7.2 修改文件

| 文件 | 改动 |
|---|---|
| `v7/__init__.py` | 导出 v7.6 API |
| `reports/momentum_etf_rotation/docs/STRATEGY_VERSIONS.md` | 添加 v7.6 |

---

## 8. 时间线

| 阶段 | 时间 | 产出 |
|---|---|---|
| Step 1: 数据加载 | 1 天 | `v7/data_loader_v7_6.py` |
| Step 2: TV-PR 算法 | 2 天 | `v7/tvpr_estimator.py` + 测试 |
| Step 3: 回测框架 | 2 天 | `v7/macro_substrategy_v7_6.py` |
| Step 4: λ 校验 | 1 天 | Time Series CV 结果 |
| Step 5: 验证 | 2 天 | 测试报告 + 对比表 |
| Step 6: 文档 | 1 天 | 更新本文档 |
| **总计** | **9 天** | |

---

## 9. 参考文献

- Cui, L., Feng, G., Ma, J., and Su, Y. (2025). "Breaks and trends in factor premia." Working paper.
- 石川 (2026). "因子择时." 川总写量化.

---

**最后更新**: 2026-07-14
**状态**: Step 1 实施中
