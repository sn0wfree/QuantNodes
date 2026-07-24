# v8 — ML 因子择时 (非收益预测)

> **编号**: 46
> **状态**: 📋 设计中
> **日期**: 2026-07-21
> **关联**: docs/43-v7_7_lgbm.md, docs/39-v7_6_tvpr.md, docs/32-市场逻辑驱动因子挖掘设计.md

---

## 1. 背景与动机

### 1.1 v7 的 OOS 表现

| 版本 | OOS Sharpe | OOS MaxDD | OOS AnnRet | 方法 |
|------|------------|-----------|------------|------|
| v7.10 | 0.329 | -26.24% | 6.99% | TV-PR (fixed lambda) |
| v7.14 | 0.438 | -22.71% | ~7% | TV-PR (fixed lambda) |
| v7.14 CV | 0.402 | -24.06% | 8.56% | TV-PR (CV lambda) |
| 历史报告 | 1.60 | -14.29% | 30.64% | TV-PR (有 lookahead) |

**关键发现**: 历史 20%+ 年化来自 `method='admm'` (全样本，有前视偏差)。当前 `method='expanding'` 的 7-8% 年化是真实 OOS 表现。

### 1.2 v7.7 ML 失败教训

> **结论**: 39 因子对下一周截面收益 R2 ≈ 0，所有 ML 模型失败 (RF, LightGBM, Lasso, Ridge, GBR, ET, MLP)。
> **根因**: 信号质量问题，非模型问题。

| 问题 | 具体表现 |
|------|----------|
| **直接预测收益** | R2 ≈ 0，无预测力 |
| **因子弱** | 单因子 IC < 0.1，ICIR < 0.3 |
| **频率不匹配** | 周频因子 → 周频收益，噪声太大 |

### 1.3 v8 的核心思路

**不预测收益，用 ML 做别的事情**:

| 方向 | 用 ML 做什么 | 为什么可行 |
|------|-------------|-----------|
| A2 非线性交互 | 捕捉因子间交互 | 单因子弱，但组合可能有信号 |
| A3 因子权重 | 预测最优因子配置 | 宏观状态 → 因子偏好 |
| A4 市场状态 | 识别牛/熊/震荡 | 宏观因子有预测力 |
| A5 集成 | 组合多个估计器 | 降低单估计器风险 |

---

## 2. 五个方向

### 2.1 A1: 新因子生成 (`v8_factor`)

**目标**: 生成有真实预测力的新因子

**方法**:
- 复用 `QuantNodes/research/core/pipeline.py` 的 PaperPipeline
- 复用 `QuantNodes/research/codegen/llm_code.py` 的 LLM 因子生成
- 使用 AlphaLogics 市场逻辑驱动 (`docs/32-市场逻辑驱动因子挖掘设计.md`)
- Gamma 编译器约束 LLM 生成 (`docs/quant_alpha/gamma_compiler_design.md`)

**前置条件**: alpha mining pipeline V9 重启 (当前暂停中，见 `docs/quant_alpha/v8_paused_for_testing.md`)

**产出**: 新因子面板 `v8_X_panel.npy`

**优先级**: P2 (依赖外部 pipeline)

### 2.2 A2: 非线性因子交互 (`v8_interaction`)

**目标**: 用 LightGBM 捕捉因子间交互，即使单因子弱

**关键设计**:
- **不预测收益**，预测截面 rank (鲁棒)
- **Purged walk-forward CV** (避免 lookahead)
- **特征重要性分析** (哪些因子/交互最重要)
- **Softmax 连续权重** (替代硬 Top-N 选择)

**与 v7.7 的区别**:

| 维度 | v7.7 | v8 |
|------|------|-----|
| 模型 | PyCaret 25 模型盲目对比 | LightGBM + Optuna 调参 |
| CV | 无 purged | purged walk-forward CV |
| 特征重要性 | 无 | 记录每步 feature importance |
| 权重 | 硬 Top-N | softmax 连续权重 |
| 标签 | 截面 rank | 截面 rank (保留) |

**接口**:
```python
def lgbm_interaction_predict(
    Y: pd.DataFrame,              # (T, N) 周频收益
    X_panel: np.ndarray,          # (T, N, K) 因子面板
    min_history: int = 52,
    step: int = 4,
    n_estimators: int = 200,
    num_leaves: int = 31,
) -> np.ndarray:
    """
    Returns:
        scores: (T, N) 截面预测分数
    """
```

**优先级**: P0 (核心方向)

### 2.3 A3: 因子权重优化 (`v8_weight_opt`)

**目标**: 用 ML 预测最优因子配置 (非收益)

**方法**:
- 输入: 宏观状态特征 (VIX、利率、信用利差等)
- 输出: 因子权重向量 (哪些因子当前更重要)
- 模型: Ridge 或 LightGBM (轻量)

**数学框架**:

对每个时点 t:
1. 提取宏观状态: `s_t = [vix_t, rate_t, credit_t, ...]`
2. 预测因子权重: `w_t = f(s_t)` → (K,) 维向量
3. 加权因子: `x_weighted_{i,t} = x_{i,t} * w_t`
4. 用加权因子做 TV-PR 或直接排序

**接口**:
```python
def predict_factor_weights(
    macro_state: np.ndarray,      # (T, K_macro) 宏观因子
    history: dict,                # 历史因子表现
) -> np.ndarray:
    """
    Returns:
        weights: (K,) 每个因子的当前权重
    """
```

**优先级**: P1 (依赖 A4)

### 2.4 A4: 市场状态检测 (`v8_regime`)

**目标**: 识别牛/熊/震荡，动态调整策略

**方法**:
- 输入: 宏观因子 + 技术指标
- 输出: 市场状态标签 (bull/bear/sideways)
- 模型: LightGBM 分类器 或 Hidden Markov Model

**标签生成**:
```python
# 用未来 12 周收益定义市场状态
future_ret = Y.rolling(12).mean().shift(-12)
regime = pd.cut(future_ret, bins=[-inf, -0.05, 0.05, inf], labels=['bear', 'sideways', 'bull'])
```

**应用**:
- 牛市: 满仓，高 beta 因子
- 熊市: 半仓，低 beta 因子 + 债券
- 震荡: 均衡配置

**接口**:
```python
def detect_market_regime(
    macro_state: np.ndarray,      # (T, K_macro) 宏观因子
    technical: np.ndarray,        # (T, N_tech) 技术指标
    min_history: int = 52,
) -> np.ndarray:
    """
    Returns:
        regime: (T,) 市场状态标签 ('bull' | 'bear' | 'sideways')
    """
```

**优先级**: P0 (最简单，独立)

### 2.5 A5: 多估计器集成 (`v8_ensemble`)

**目标**: 组合多个估计器的预测

**方法**:
- 输入: TV-PR scores + LightGBM scores + 市场状态
- 输出: 集成 scores
- 模型: 简单平均 或 stacking

**集成策略**:
```python
# 方案 1: 简单平均
ensemble_scores = 0.5 * tvpr_scores + 0.5 * lgbm_scores

# 方案 2: 状态条件加权
if regime == 'bull':
    ensemble_scores = 0.3 * tvpr_scores + 0.7 * lgbm_scores
elif regime == 'bear':
    ensemble_scores = 0.7 * tvpr_scores + 0.3 * lgbm_scores
else:
    ensemble_scores = 0.5 * tvpr_scores + 0.5 * lgbm_scores

# 方案 3: Stacking (学习最优权重)
meta_model = Ridge()
meta_model.fit(meta_features_train, Y_train)
ensemble_scores = meta_model.predict(meta_features_test)
```

**接口**:
```python
def ensemble_predict(
    tvpr_scores: np.ndarray,      # (T, N) TV-PR 预测
    lgbm_scores: np.ndarray,      # (T, N) LightGBM 预测
    regime: np.ndarray,           # (T,) 市场状态
    method: str = 'state_weighted',
) -> np.ndarray:
    """
    Returns:
        scores: (T, N) 集成预测
    """
```

**优先级**: P1 (依赖 A2 + A4)

---

## 3. 实施顺序

```
Phase 1: A4 (市场状态检测) — 最简单，独立，1-2 天
Phase 2: A2 (非线性因子交互) — 核心方向，3-5 天
Phase 3: A3 (因子权重优化) — 依赖 A4，2-3 天
Phase 4: A5 (集成估计器) — 依赖 A2 + A4，2-3 天
Phase 5: A1 (新因子生成) — 依赖 alpha pipeline V9，待定
```

### Phase 1 详细计划

**A4: 市场状态检测**

| 步骤 | 任务 | 产出 | 时间 |
|------|------|------|------|
| 1.1 | 标签生成 (未来 12 周收益 → bull/bear/sideways) | `v8_regime_labels.npy` | 0.5 天 |
| 1.2 | 特征工程 (宏观因子 + 技术指标) | `v8_regime_features.npy` | 0.5 天 |
| 1.3 | LightGBM 分类器训练 | `v8_regime_model.pkl` | 0.5 天 |
| 1.4 | 集成到 walk-forward 框架 | `regime_detector.py` | 0.5 天 |
| 1.5 | 回测验证 | OOS 指标 | 0.5 天 |

### Phase 2 详细计划

**A2: 非线性因子交互**

| 步骤 | 任务 | 产出 | 时间 |
|------|------|------|------|
| 2.1 | 数据准备 (复用 v7.10 因子面板) | `v8_X_panel.npy` | 0.5 天 |
| 2.2 | LightGBM 滚动预测 (purged CV) | `lgbm_estimator.py` | 1 天 |
| 2.3 | Optuna 超参调优 | 最优参数 | 1 天 |
| 2.4 | 特征重要性分析 | `v8_feature_importance.csv` | 0.5 天 |
| 2.5 | Softmax 连续权重 | 组合构造改进 | 0.5 天 |
| 2.6 | 回测验证 | OOS 指标 | 0.5 天 |

---

## 4. 文件规划

### 4.1 新建文件

```
QuantNodes/strategy/momentum_etf_rotation/v8/
├── __init__.py
├── adapters.py                    # v8 统一适配器 (复用 v7 模式)
├── data_loader_v8.py              # v8 数据加载
├── lgbm_estimator.py              # A2: LightGBM 因子交互
├── regime_detector.py             # A4: 市场状态检测
├── factor_weight_optimizer.py     # A3: 因子权重优化
├── ensemble_estimator.py          # A5: 集成估计器
└── macro_substrategy_v8.py        # v8 组合构造

scripts/
├── v8_unified_runner.py           # v8 统一执行脚本
└── v8_regime_train.py             # A4 训练脚本

tests/strategy/momentum_etf_rotation/v8/
├── test_lgbm_estimator.py
├── test_regime_detector.py
└── test_ensemble_estimator.py
```

### 4.2 复用文件

| 文件 | 复用内容 |
|------|----------|
| `v7/adapters.py` | `make_v7_6_backtest_fn` 模式、`select_lambda_cv` |
| `v7/tvpr_estimator.py` | `expanding_window_tvpr` (A5 集成) |
| `v7/macro_substrategy_v7_6.py` | `compute_weekly_weights`、`construct_portfolio_components` |
| `common/walk_forward.py` | `walk_forward_rolling`、`generate_nav_from_weights` |

---

## 5. 接口设计

### 5.1 v8 统一接口

```python
# v8 adapters.py
def make_v8_backtest_fn(version: str = "v8.1") -> Callable:
    """生成 v8 的 backtest_fn.

    签名: backtest_fn(Y, X, **params) → (shares, prices, weights)
    """
    def backtest_fn(Y, X, **params):
        # 1. 市场状态检测 (A4)
        regime = detect_market_regime(Y, X, ...)

        # 2. LightGBM 因子交互 (A2)
        lgbm_scores = lgbm_interaction_predict(Y, X, ...)

        # 3. TV-PR 基准 (复用 v7)
        beta = expanding_window_tvpr(Y, X, ...)
        tvpr_scores = X @ beta  # (T, N)

        # 4. 集成 (A5)
        scores = ensemble_predict(tvpr_scores, lgbm_scores, regime, ...)

        # 5. 组合构造 (复用 v7)
        shares, prices, weights = construct_portfolio_components(Y, X, scores, cfg)
        return shares, prices, weights

    return backtest_fn
```

### 5.2 与 v7 的兼容性

v8 复用 v7 的组合构造逻辑，只替换信号生成部分:

| 步骤 | v7 | v8 |
|------|----|----|
| 信号 | `X @ beta_path` (线性) | `ensemble(tvpr, lgbm, regime)` (非线性) |
| 选择 | Top-N (硬) | Softmax (连续) |
| 加权 | 逆波动率 | 逆波动率 (保留) |
| 风控 | 止损桩 | 止损 + 趋势过滤 |

---

## 6. 评估方案

### 6.1 基准对比

| 策略 | 说明 |
|------|------|
| v7.14 TV-PR | 当前最优 (OOS Sharpe 0.438) |
| v8.1 LightGBM | A2 单独 |
| v8.2 Regime | A4 单独 |
| v8.3 Ensemble | A2 + A4 集成 |
| v8.4 Full | A2 + A3 + A4 + A5 |

### 6.2 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| OOS Sharpe | 年化收益 / 波动率 | > 0.5 |
| OOS MaxDD | 最大回撤 | < -25% |
| OOS AnnRet | 年化收益 | > 8% |
| Feature Importance | 特征重要性 | 可解释 |
| CV% | 多起点变异系数 | < 25% |

---

## 7. 风险与缓解

### 7.1 过拟合风险

| 风险 | 缓解 |
|------|------|
| LightGBM 过拟合 | purged CV + max_depth 限制 + early stopping |
| 因子过多 | 特征重要性筛选 + L1 正则 |
| 市场状态标签噪声 | 12 周窗口平滑 + 概率输出 |

### 7.2 计算成本

| 模型 | 单次训练 | 滚动 81 次 | 总时间估计 |
|------|----------|------------|-----------|
| LightGBM (A2) | ~0.2s | 81 | ~20s |
| 市场状态 (A4) | ~0.1s | 81 | ~10s |
| 集成 (A5) | ~0.01s | 81 | ~1s |
| **总计** | | | **~30s** |

### 7.3 数据依赖

| 数据 | 来源 | 状态 |
|------|------|------|
| v7.10 因子面板 | `data/high_freq_macro/v7_10_*.npy` | ✅ 可用 |
| 宏观因子 | `data/high_freq_macro/v7_6_X_macro_weekly.parquet` | ✅ 可用 |
| 市场状态标签 | 需生成 | 待实现 |

---

## 8. 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-07-21 | v7.7 ML 失败后重启 ML 方向 | 换用法：不预测收益，做因子交互/状态检测 |
| 2026-07-21 | LightGBM 为主模型 | 效率高、支持 NaN、可解释 |
| 2026-07-21 | 截面 rank 标签 (保留) | 鲁棒，消除极端值 |
| 2026-07-21 | purged walk-forward CV | 避免 lookahead |
| 2026-07-21 | A4 先行 | 最简单，独立，可验证思路 |

---

## 9. 参考文献

- Ke, G. et al. (2017). "LightGBM." NeurIPS.
- Cui et al. (2025). "Breaks and trends in factor premia." (TV-PR 原论文)
- AlphaLogics (arXiv 2603.20247) — 市场逻辑驱动因子挖掘
- QuantaAlpha — 进化实验框架
- De Prado, M. (2018). "Advances in Financial Machine Learning." — Purged CV

---

**最后更新**: 2026-07-21
**状态**: 📋 设计完成，待实施
