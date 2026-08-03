# v10 Risk Warning System — ML-Based Market Timing

## 背景

方案 A (Regime-Based) 原本是 v10 的 5 个动态权重方案之一，仅用 `v7.10` 12M 收益符号做牛市/熊市二元判断。2026-08-03 精简方案时保留为独立模型，用于市场择时研究。

**核心困境：** 历史证明 108 种动态加权组合全部跑输静态 Vol-parity (docs/71)，CA-GCP v6 信号 AUC ≈ 0.5 被拒 (docs/83)。A 方案的 12M 收益符号是唯一幸存者。

**解决思路：** 不做择时 (预测方向)，做预警 (评估风险环境)。用外部信号 (VIX、宏观因子、尾部风险) 而非策略内部信号构建 ML 模型。

## 系统架构

```
┌───────────────────────────────────────────────────────────────┐
│                    ML Risk Warning System                     │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Layer: 4 类信号 (外部数据, 非策略内部)                  │
│  ├── Volatility: VIX, 已实现波动率, 尾部风险 (skew/kurt)     │
│  ├── Macro: 增长/通胀/流动性/汇率 等 8 个 v9 因子             │
│  ├── Market Structure: 截面离散度, 金油相关性, 相关性         │
│  └── Technical: 趋势, 动量, 回撤状态 (仅用作控制变量)          │
│                                                               │
│  Feature Engineering: CA-GCP v3 模式 (40-60 特征)             │
│  ├── Lag features (1/5/20 日)                                 │
│  ├── Z-score (252 日滚动, 对齐历史位置)                       │
│  ├── Change rate (diff, pct_change, 边际变化)                 │
│  ├── Threshold events (binary, 极端值突破)                     │
│  └── Interaction features (VIX × momentum, 非线性)            │
│                                                               │
│  Model: LightGBM Binary Classifier                            │
│  ├── Target: y_E (10 日 forward maxDD < -2.5% OR return < 20%ile) │
│  ├── Validation: Walk-forward (4y train / 1y test / 1y step) │
│  ├── Purge/Embargo: 5 日 (防数据泄露)                         │
│  └── Calibration: Platt scaling → 概率输出                    │
│                                                               │
│  Output: 风险等级 0-3 → 动态 min_defensive                    │
│  ├── Level 0: p < 0.25 → min_defensive = 25% (默认)          │
│  ├── Level 1: 0.25 ≤ p < 0.50 → min_defensive = 40%          │
│  ├── Level 2: 0.50 ≤ p < 0.75 → min_defensive = 60%          │
│  └── Level 3: p ≥ 0.75 → min_defensive = 90%                 │
│                                                               │
│  Integration: G 方案 _apply_constraints() 注入                 │
│  └── risk_level → min_defensive → 每周更新一次                │
└───────────────────────────────────────────────────────────────┘
```

## 执行计划

### Phase 1: 数据探索 (Data Exploration)

**文件:** `scripts/v10_3/v10_3_risk_data_exploration.py`

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1.1 | 加载 4 策略 NAV + VIX + 8 宏观因子 + 尾部风险, 对齐日频 | 特征矩阵 X_raw |
| 1.2 | 构建 y_E 目标 | 标签 y |
| 1.3 | 各信号分布统计 | 统计表 |
| 1.4 | 单信号 AUC (每个信号独立预测 y_E) | AUC 排名 |
| 1.5 | 信号间相关性矩阵 | 热力图 |
| 1.6 | y_E 在 OOS 期分布 | 正例率/时序 |
| 1.7 | 极端事件回测 (4 次已知事件) | 事件-信号表 |

### Phase 2: ML 模型训练

**文件:** `scripts/v10_3/v10_3_risk_model_training.py`

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | 特征工程 (40-60 特征, 含 lag/zscore/event/interaction) | 特征矩阵 X |
| 2.2 | Walk-forward split (4y / 1y / 1y, purge=5, embargo=5) | 5 folds |
| 2.3 | Optuna 超参调优 (30 trials, scale_pos_weight) | 最优参数 |
| 2.4 | LightGBM 训练 + 校准 | 模型 + 校准器 |
| 2.5 | Walk-forward 评估 (AUC, F1, PR-AUC, ECE, Brier) | 评估报告 |
| 2.6 | SHAP 特征重要性 | 特征排名 |
| 2.7 | 对比 XGBoost (备选) | 模型对比 |

### Phase 3: 回测验证

**文件:** `scripts/v10_3/v10_3_risk_alert_backtest.py`

| 测试 | 基准 | 对比 | 指标 |
|------|------|------|------|
| 3.1 OOS 整体 | G (min_def=25%) | G + RiskAlert | Sharpe, Calmar, MaxDD |
| 3.2 成本调整 | 无成本 | 10bp | 换手率, 净收益 |
| 3.3 假警报率 | — | 精确率/召回率 | 混淆矩阵 |
| 3.4 Walk-forward | G 基准 | G + RiskAlert | 滚动 2 年胜率 |
| 3.5 极端事件 | 4 次已知事件 | 预警触发时点 | 事件命中率 |

### 生产集成

**文件:** `dynamic_weight_schemes.py`

| 改动 | 内容 |
|------|------|
| 新增 `RiskAlert` 类 | 加载模型 + 特征工程 + 预测 + 风险等级 |
| 新增 `scheme_a_risk_alert()` | 升级版 A, 输出动态权重 |
| 修改 `scheme_g_trend_adaptive()` | 增加 `risk_alert` 参数, 动态 `min_defensive` |
| 修改 `main()` | 增加 `A_risk_alert` 方案 + 集成 |

## 交付标准

- OOS Sharpe 提升: G + RiskAlert > G 基准 (min delta +0.05)
- OOS Calmar 提升: G + RiskAlert > G 基准 (min delta +0.10)
- 假警报率 < 40%: 预警触发后 10 日内实际回撤 < -2.5%
- 年化换手率增量 < 5%: 不显著增加交易成本
- Walk-forward 胜率 > 60%: 滚动 2 年窗口 vs G 基准

## 已知风险

1. y_E 正例率低 (OOS 可能 15-25%), 需 scale_pos_weight
2. VIX 是美股代理, 对 A 股领先性待验证
3. ML 过拟合: walk-forward + purge/embargo + 简单模型 三重防护
4. 换手率增加: 每周更新 + hysteresis 防抖

## 参考

- docs/71: 动态加权失败总结
- docs/83: CA-GCP v6 被拒原因
- docs/85: Phase 0-1-2 完整报告
- CA-GCP v3: 特征工程模式 (reports/ca_gcp_ew/ca_gcp_training_v3.py)
- 宏观因子: v9/factor_score_basic.py
- VIX: v9/scoring.py compute_vix_score()
- Walk-forward: common/walk_forward.py