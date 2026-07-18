# v7.7 — PyCaret 多模型因子择时

> **编号**: 43
> **状态**: 📋 设计中
> **日期**: 2026-07-16
> **关联**: docs/39-v7_6_tvpr.md, docs/41-v7_6_factor_ic_and_enhancement.md

---

## 1. 背景与动机

### 1.1 TV-PR 的局限

v7.6 TV-PR OOS Calmar 稳定在 0.35-0.42，但远低于 v1.0 locked 的 1.87。

根本原因：

| 问题 | 具体表现 |
|------|----------|
| **线性假设** | TV-PR 本质是 β_t · x 线性模型，无法捕捉因子间的非线性交互 |
| **稀疏性偏向** | L1 罚项迫使大部分 β_t=0，但真实因子溢价可能非稀疏 |
| **时变性粗糙** | TV 罚项只控制断点数量，无法捕捉渐变的因子关系 |
| **截面利用不足** | 每个时点的 N 个资产样本被压缩为一个 β_t，丢失截面差异信息 |

### 1.2 为什么用 PyCaret

PyCaret 是 low-code ML 库，核心价值：
1. **一键对比 25 种回归模型**（线性/树/神经网络/SVM/KNN 等）
2. **自动预处理**（缺失值填充、标准化、异常值处理）
3. **统一 API**：setup → compare_models → predict_model
4. **底层模型可提取**：直接拿到 sklearn/LightGBM/CatBoost 原生对象

相比手写 LightGBM，PyCaret 让我们快速回答"什么模型最适合这个数据集"。

### 1.3 可选模型一览（PyCaret regression）

| ID | 模型 | 类型 | 适用场景 |
|----|------|------|----------|
| lr | Linear Regression | 线性 | 基准 |
| lasso | Lasso | 线性+L1 | 稀疏特征选择 |
| ridge | Ridge | 线性+L2 | 多重共线性 |
| en | Elastic Net | 线性+L1+L2 | 平衡 |
| br | Bayesian Ridge | 贝叶斯线性 | 不确定性估计 |
| huber | Huber Regressor | 鲁棒线性 | 抗异常值 |
| svm | SVR | 核方法 | 非线性小样本 |
| knn | KNN | 非参数 | 局部模式 |
| dt | Decision Tree | 树 | 可解释 |
| rf | Random Forest | 集成树 | 稳健非线性 |
| et | Extra Trees | 集成树 | 更随机的 RF |
| gbr | Gradient Boosting | boosting | 序列集成 |
| ada | AdaBoost | boosting | 自适应权重 |
| xgboost | XGBoost | boosting | 高性能 |
| **lightgbm** | **LightGBM** | **boosting** | **高效率+大规模** |
| catboost | CatBoost | boosting | 类别特征 |
| mlp | MLP | 神经网络 | 复杂非线性 |

---

## 2. 数学框架

### 2.1 问题设定

与 TV-PR 完全一致的 pooled cross-sectional regression：

- 训练样本: 每个 (t, i) 是一个样本
- 特征: x_{i,t} in R^{K}，K=39 维因子
- 标签: y_{i,t} = r_{i,t+1}（下周收益）或截面 rank

### 2.2 滚动训练

对每个预测时点 t >= T_min：

  f_t = argmin_f sum_{s=1}^{t-1} sum_{i=1}^{N} L(y_{i,s}, f(x_{i,s})) + Omega(f)

预测：

  y_{i,t} = f_t(x_{i,t})

### 2.3 标签选择

| 方案 | 定义 | 优劣 |
|------|------|------|
| **原始收益** | y_{i,t} = r_{i,t+1} | 保留幅度信息，但受极端值影响 |
| **截面 rank** | y_{i,t} = rank(r_{i,t+1}) / N | 鲁棒，消除异常值，但丢失幅度 |

**推荐**: 截面 rank（默认）。

### 2.4 组合构造

模型输出 scores (T, N) 后，复用 v7.6 的 construct_portfolio 逻辑：

1. 每周用 scores 选 top_n 资产
2. 逆波动率加权
3. 扣除交易成本

---

## 3. 因子体系

当前 39 维（8 原始宏观 + 9 增强宏观 + 22 量价），详见 docs/41。

关键高 IC 因子：

| 因子 | IC | IC_IR | 类型 |
|------|-----|-------|------|
| f22_rsi | +0.500 | +2.24 | 量价 |
| f19_mom_mid | +0.428 | +1.50 | 量价 |
| gold_oil_corr | -0.053 | -1.04 | 宏观 |

---

## 4. PyCaret 工作流设计

### 4.1 核心流程

```python
from pycaret.regression import RegressionExperiment

exp = RegressionExperiment()

# 每个滚动步：
exp.setup(
    data=train_df,       # DataFrame: [f0..f38, target]
    target='target',
    train_size=0.8,
    preprocess=False,    # 因子已预处理，不需要 PyCaret 自动处理
    session_id=42,
    verbose=False,
    html=False,
)

# 方案 A: compare_models 自动对比
best_models = exp.compare_models(
    include=['ridge', 'lightgbm', 'rf', 'xgboost', 'catboost'],
    sort='R2',
    n_select=5,
    verbose=False,
)

# 方案 B: create_model 单模型
model = exp.create_model('lightgbm', verbose=False)

# 预测
pred = exp.predict_model(model, data=test_df, verbose=False)
# pred['prediction_label'] = 预测值
```

### 4.2 滚动窗口包装

```python
def pycaret_estimator(
    Y: pd.DataFrame,              # (T, N) 周频收益
    X_panel: np.ndarray,          # (T, N, K) 因子面板
    min_history: int = 52,        # 最少训练期
    model_ids: list[str] = None,  # PyCaret 模型 ID 列表
    use_rank_label: bool = True,  # 截面 rank 标签
    compare: bool = True,         # True=compare_models, False=单模型
) -> dict[str, np.ndarray]:
    """
    Returns:
        dict[model_id -> scores (T, N)] 每个模型的预测分数
    """
```

### 4.3 模型选择策略

**Phase 1: 快速筛选**（compare_models）

```python
candidates = ['lr', 'ridge', 'lasso', 'en', 'rf', 'et', 'gbr',
              'lightgbm', 'xgboost', 'catboost', 'huber', 'knn']
```

用 compare_models 一次性跑完，按 R2 排序，选 top-5。

**Phase 2: 滚动回测**

对 top-5 模型分别做滚动训练+回测，比较 OOS Calmar/Sharpe。

**Phase 3: 超参调优**

对最佳模型用 tune_model 做贝叶斯优化。

### 4.4 preprocess 选择

**关键决策**: preprocess=False

原因：
1. 我们的因子已经过 z-score/rank 标准化
2. PyCaret 的自动预处理（缺失值填充、标准化）可能引入前瞻偏差
3. 树模型不需要标准化
4. 避免 PyCaret Pipeline 包装，直接拿到原生模型对象

### 4.5 注意事项

| 事项 | 处理方式 |
|------|----------|
| **NaN 处理** | 训练前用 dropna，不依赖 PyCaret 填充 |
| **prediction_label** | PyCaret 预测列名固定为 prediction_label，需 rename |
| **n_select** | compare_models(n_select=k) 返回 list，n_select=1 返回单对象 |
| **session_id** | 固定 session_id=42 保证可复现 |
| **verbose** | 全程 verbose=False，减少输出 |
| **html** | html=False，避免弹出 notebook |

---

## 5. 防过拟合策略

### 5.1 滚动窗口

扩展窗口（默认），与 TV-PR 一致。前 52 周不预测。

### 5.2 正则化层次

1. **模型层**: 各模型自带正则化（Ridge alpha, LGB num_leaves, RF max_depth）
2. **标签层**: 截面 rank（消除极端值）
3. **评估层**: OOS 时间序列验证

### 5.3 与 TV-PR 的对比

| 维度 | TV-PR | PyCaret 多模型 |
|------|-------|----------------|
| 模型 | 线性 beta_t*x | 25 种可选 |
| 特征交互 | 无 | 树模型自动学习 |
| 特征选择 | 手动调 lambda_l1 | importance 自动排序 |
| 样本利用率 | T*1 -> beta_t (430个) | T*N -> 18,000 个 |
| 验证方式 | 无 | CV + OOS |

---

## 6. 评估方案

### 6.1 基准对比

| 策略 | 说明 |
|------|------|
| v7.6 TV-PR | 当前线性基准 (Calmar ~0.38) |
| v7.7 PyCaret top-5 | 多模型对比最佳 |
| v7.7 LightGBM (单独) | 直接 LightGBM |
| v7.7 Ridge (单独) | 线性基准 |
| v1.0 locked | 历史最优 (Calmar 1.87) |

### 6.2 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Calmar | 年化收益 / 最大回撤 | > 0.5 |
| Sharpe | 年化收益 / 波动率 | > 0.8 |
| OOS 2022+ Calmar | 近期样本外表现 | > 0.4 |
| CV% | 多起点 Calmar 变异系数 | < 25% |

---

## 7. 文件规划

### 7.1 新建文件

| 文件 | 说明 | 行数估计 |
|------|------|----------|
| v7/pycaret_estimator.py | PyCaret 滚动训练 + 多模型预测 | ~250 行 |
| v7/macro_substrategy_v7_7.py | v7.7 配置 + 回测入口 | ~200 行 |
| scripts/run_v7_7_backtest.py | 端到端回测脚本 | ~100 行 |

### 7.2 修改文件

| 文件 | 改动 |
|------|------|
| v7/__init__.py | 导出 v7.7 API |

### 7.3 依赖

PyCaret 3.3.2 已安装 (python3.10 + python3.11)
LightGBM 4.6.0 已安装
不需要额外安装

---

## 8. 接口设计

### 8.1 与 v7.6 的兼容性

```python
# v7.6: tvpr_estimator
beta_path = tvpr_estimator(Y, X_panel, lambda_tv, lambda_l1, ...)
# -> beta_path (T, K), scores = X_panel[t] @ beta_path[t-1]

# v7.7: pycaret_estimator
scores_dict = pycaret_estimator(Y, X_panel, model_ids=['lightgbm', ...])
# -> scores_dict = {'lightgbm': (T,N), 'ridge': (T,N), ...}
```

### 8.2 组合构造

完全复用 v7.6 的 construct_portfolio(Y, scores, cfg) -- scores 是 (T, N)。

### 8.3 日频 NAV

完全复用 v7.6 的 calculate_daily_nav(weights_df, daily_returns, cfg)。

---

## 9. 风险与缓解

### 9.1 计算成本

| 模型 | 单次训练 | 滚动 378 次 | 总时间估计 |
|------|----------|------------|-----------|
| Ridge | ~0.01s | 378 | ~5s |
| LightGBM | ~0.1s | 378 | ~40s |
| Random Forest | ~0.5s | 378 | ~3min |
| CatBoost | ~1s | 378 | ~6min |
| compare_models (5个) | ~2s | 378 | ~13min |

**优化**: Phase 1 用 compare_models 只跑一次全量对比，Phase 2 只对 top-5 做滚动。

### 9.2 过拟合风险

| 风险 | 缓解 |
|------|------|
| 模型太复杂 | num_leaves/max_depth 限制 |
| 样本不均衡 | 截面 rank 标签 |
| 滚动窗口小 | 扩展窗口，min_history=52 |

### 9.3 PyCaret 前瞻偏差

PyCaret 的 preprocess=True 会自动做标准化/缺失值填充，可能用到整个数据集。
-> 使用 preprocess=False，手动处理数据。

---

## 10. 时间线

| 阶段 | 时间 | 产出 |
|------|------|------|
| Step 1: pycaret_estimator.py | 1 天 | 核心估计器 + 多模型支持 |
| Step 2: macro_substrategy_v7_7.py | 0.5 天 | 回测框架 |
| Step 3: 端到端回测 + compare_models | 0.5 天 | scripts/run_v7_7_backtest.py |
| Step 4: top-5 滚动回测对比 | 1 天 | 性能对比表 |
| Step 5: 最佳模型超参调优 | 0.5 天 | tune_model 结果 |
| Step 6: 文档更新 | 0.5 天 | 更新本文档 |
| **总计** | **4 天** | |

---

## 11. 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-07-16 | 用 PyCaret 替代手写 LightGBM | 一键对比多模型，统一 API |
| 2026-07-16 | preprocess=False | 避免前瞻偏差，因子已预处理 |
| 2026-07-16 | 截面 rank 标签（默认） | 鲁棒，消除极端值 |
| 2026-07-16 | 扩展窗口（非滚动） | 样本量小，扩展窗口更稳定 |
| 2026-07-16 | 复用 v7.6 的 construct_portfolio | 信号->组合逻辑独立于模型 |
| 2026-07-16 | 移除期限利差因子_加权 | 用户判断无价值 |
| 2026-07-16 | 新增 cn_us_spread + gold_oil_corr | IC_IR 分别 +0.12 和 -1.04 |

---

## 12. 参考文献

- PyCaret Documentation: https://pycaret.gitbook.io/docs
- Ke, G. et al. (2017). "LightGBM." NeurIPS.
- Pedregosa et al. (2011). "Scikit-learn." JMLR.

---

**最后更新**: 2026-07-16
**状态**: 设计完成，待实施
