# 华泰金工《时序CTA方法论综述》调研笔记 (2026-09-07)

> 来源: [新浪财经原文](https://finance.sina.com.cn/wm/2026-06-12/doc-inicazrw1882110.shtml)
> 状态: **调研归档 / 待实施 Phase 1**

---

## 一、文章核心论点

华泰金工系列研究开篇，从随机过程统计学视角重塑时序CTA：

| 问题 | 答案 |
|------|------|
| 如何判断趋势/震荡？ | (μ, ρ, σ) 三参数 + TSI 置信区间 |
| 如何构建开仓信号？ | 三层架构：线性滤波 → 去量纲 → 信号融合 |
| 何时退出？ | 四层防线：技术止损 → 逻辑证伪 → 时间强制 → 全局风控 |

**核心哲学**：可交易性 ≠ 趋势方向，而是 **TSI(μ/σ·√T)** 的统计显著性。

---

## 二、与 QuantNodes 现有能力的差距矩阵

| 文章组件 | 现有实现 | 缺口 |
|----------|---------|------|
| GARCH(1,1) | `GARCHFactor` `factors/volatility.py:123-160` | - |
| AR(1) 自相关 | `AutocorrelationFactor` `factors/statistical.py:142-167` | 需补显式含截距回归 |
| RSI | `RSIFactor` `factors/trend.py:150-172` | - |
| 布林带 | `BollingerFactor` `factors/trend.py:175-193` | - |
| MA/EMA | `MAFactor` + talib 12 种 | - |
| ATR | `indicators.atr` + `ATRBreakoutFactor` | - |
| **吊灯止损** | `ChandelierExitFactor` `factors/trend_advanced.py:177-212` | - |
| Z-score | `indicators.zscore` | - |
| 乖离率 | `BIASFactor` `factors/trend.py:269-296` | - |
| **Kaufman 效率比** | `EfficiencyFactor` `factors/trend.py:196-214` | - |
| ADX | `ADXFilterFactor` `factors/trend.py:217-236` | - |
| Hurst | `HurstExponentFactor` `factors/statistical.py:82-109` | - |
| 卡尔曼滤波 | `kalman_local_level` `strategy/commodity_fundamental/kalman_filter.py:19-78` | - |
| 市场状态识别 | HMM `regime_detector.py:51-121` + Hurst/ADX/Eff | - |
| **TSI (True Strength Index)** | ❌ **缺失** | **需新增 `TSIFactor`** |
| **TSI Bootstrap CI** | ❌ | **需新增 `tsi_bootstrap.py`** |
| **Regime 编排层** | ❌ | **需新增 `regime/` 目录** |
| CTA 回测引擎 | `run_cta_backtest` `strategy/cta/backtest/cta_engine.py:62-168` | - |
| 卡玛比/夏普/MaxDD | `compute_metrics` `metrics.py:79-176` | - |
| 多空双向 + 滑点 | `CtaBroker` + `SquareRootImpact` | - |
| 分钟级回测 | ❌ 当前仅日频 | 本期不实现 |

**结论**: 已有 90% 能力，仅缺 TSI 因子 + Regime 编排层 + 多层退出管理。

---

## 三、关键公式（落地伪代码）

### 3.1 三元分解 + 自适应窗口

```python
# AR(1) + GARCH 联合估计
def estimate_arp1_garch(returns, window, kaufman_er):
    h = (21 * kaufman_er + 250 * (1 - kaufman_er)).round().astype(int)  # 21-250 自适应
    mu, rho, sigma = [], [], []
    for t in range(window, len(returns)):
        seg = returns.iloc[t-window:t]
        am = arch_model(seg * 100, vol='GARCH', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        mu.append(res.params['mu'] / 100)
        rho.append(res.params['Const'])  # AR 系数
        sigma.append(res.conditional_volatility.iloc[-1] / 100)
    return pd.DataFrame({'mu': mu, 'rho': rho, 'sigma': sigma})
```

### 3.2 TSI Bootstrap 置信区间

```python
def tsi_bootstrap_ci(mu, sigma, n=1000, ci=0.95, rng=None):
    """对残差有放回重采样 → TSI 经验分布"""
    rng = rng or np.random.default_rng(42)
    samples = rng.normal(mu, sigma, size=(n, len(mu))).mean(axis=1)
    tsi_samples = samples / (sigma.mean() / np.sqrt(len(mu)))
    lo, hi = np.quantile(tsi_samples, [(1-ci)/2, 1-(1-ci)/2])
    return {'tsi': mu.mean() / (sigma.mean() / np.sqrt(len(mu))), 'lo': lo, 'hi': hi}
```

### 3.3 状态识别规则

```text
趋势主导 ← all(连续 10 日):
    TSI_CI 完全在零轴一侧
    OR |TSI| 历史分位数 > 50%
    AND ρ > 0
    AND (ρ_t_stat > 1.5 OR σ 历史分位数 < 95%)

否则 ⇒ 震荡
```

### 3.4 信号三层架构

```python
def entry_signal(close, state, params):
    # Layer 1: 线性滤波
    ema_fast, ema_slow = close.ewm(span=5).mean(), close.ewm(span=20).mean()
    # Layer 2: 去量纲
    rsi_14 = compute_rsi(close, 14)
    zscore = (close - close.rolling(20).mean()) / close.rolling(20).std()
    # Layer 3: 状态分支 + 多窗口共振
    if state == 'trend':
        return np.sign(ema_fast - ema_slow)
    return np.where(rsi_14 > 70, -1, np.where(rsi_14 < 30, 1, 0))
```

### 3.5 ATR 移动止损（吊灯 — 项目已具备 `ChandelierExitFactor`）

```python
# 参考: factors/trend_advanced.py:177-212
long_stop  = high.rolling(22).max() - k * atr(ohlc, 14)
short_stop = low.rolling(22).min()  + k * atr(ohlc, 14)
```

---

## 四、实施 Phase 清单

### Phase 1 — 因子层补齐（本期目标）

| 文件 | 类型 | 说明 |
|------|------|------|
| `QuantNodes/strategy/cta/factors/oscillators.py` | 新增 | `TSIFactor(close, r=25, s=13)` 双 EMA 归一化 |
| `QuantNodes/strategy/cta/factors/statistical_ext.py` | 新增 | `AR1Factor(returns, window)` 含截距 OLS |
| `QuantNodes/strategy/cta/regime/__init__.py` | 新增 | 公开 API |
| `QuantNodes/strategy/cta/regime/ar_garch_estimator.py` | 新增骨架 | (μ, ρ, σ) 估计 + Kaufman 自适应窗口 |
| `QuantNodes/strategy/cta/regime/tsi_bootstrap.py` | 新增骨架 | TSI 1000 次重采样 + 95% CI |
| `QuantNodes/strategy/cta/regime/regime_classifier.py` | 新增骨架 | 状态识别规则 |
| `factors/__init__.py` | 修改 | 导出 `TSIFactor` / `AR1Factor` |
| `strategy/cta/__init__.py` | 修改 | 注册新因子 |
| `tests/strategy/cta/test_factors_oscillators.py` | 新增 | TSI 边界/单调性 |
| `tests/strategy/cta/test_factors_ar1.py` | 新增 | AR1 还原已知序列 |
| `tests/strategy/cta/test_regime_classifier.py` | 新增 | 状态切换正确性 |

### Phase 2 — Regime 完整化（后续）
实装骨架文件中的 GARCH/AR 估计逻辑 + 状态判定完整规则。

### Phase 3 — 退出机制管理（后续）
四层防线编排：`cta/exits/{chandelier,logical_exit,time_exit,risk_manager,exit_manager}.py`。

### Phase 4 — 策略实证（后续）
**用户决策**：标的池 = **期货数据**（`IF + AU + T`），非原文 ETF。

| 文章维度 | 文章 ETF | 期货映射 |
|----------|---------|----------|
| 高波动权益 | 510300 | IF (沪深300期货) |
| 长周期商品 | 518880 | AU (沪金) |
| 低波动防御 | 511010 | T (10Y国债) |

### Phase 5 — 文档与 CHANGELOG（最后）
方法论笔记 + 业绩结果 + CHANGELOG 条目。

---

## 五、风险与红线

- **`arch` 是可选依赖**：参考 `GARCHFactor:142` 已有的 `from arch import arch_model` 函数内 import 模式
- **`statsmodels` 是核心依赖** (`pyproject.toml:62`)，可直接使用 `AutoReg`
- **Bootstrap 1000 次性能**：单标的 1-2s；多标的建议并行化
- **期货 vs ETF 可对比性**：仅做框架级（卡玛比改善方向）对比，不做逐数值对齐
- **AGENTS.md 红线**：Phase 1 仅新增 5 文件 + 修改 2 文件，禁止顺手重构

---

## 六、参考

- 华泰证券《时序CTA方法论综述：市场状态、开仓信号与退出机制》(2026-06-12)
- Moskowitz, Ooi, Pedersen (2011). *Time Series Momentum*
- Levine, Pedersen (2016). *Which Trend is Your Friend?*
- Curtis Faith (2013). *海龟交易法则*