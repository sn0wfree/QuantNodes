# 决策日志 (Decision Log)

> 用途: 记录所有重大 go/no-go 决策, 形成时间序列
> 维护: 每次 Stage 5 (Decision) 完成后追加
> 格式: YYYY-MM-DD, 决策, 原因, 证据链接

---

## 2026-07-08: Stage 12A 斜率×R² 动量 + 策略 v1.0 锁定

- **决策**: ✅ GO (推荐 `momentum_type="hybrid"` 作为默认)
- **配置** (v1.0 锁定):
  ```python
  RotationConfig(
      lookback=90, top_n=10,
      momentum_type="hybrid",             # price + slope_r2
      momentum_fused_weight=0.5,
      vol_targeting=VolTargeting(enabled=True, target_vol=0.15, ...),
      cost_model=CostModel(enabled=True, commission_bp=5, ...),
  )
  ```
- **v1.0 关键指标 (2019-2026)**:
  - Calmar **1.60** (vs Stage 8 baseline 0.78, +105%)
  - DD **-3.93%** (vs -21.05%, 改善 17pp)
  - Ann 6.28% (vs 16.35%, 降低)
  - OOS Calmar 0.84 (vs 1.72, 退化但 DD 远优)
- **原因**:
  - hybrid 方式在样本内 Calmar 1.17 (+10% vs price)
  - 与 VT 组合达 Calmar 1.60, DD 仅 -3.93%
  - OOS 中等退化, 接受为风险厌恶型配置
- **证据**: `stage12a_report.md`
- **v1.0 状态**: 锁定为基准版本, 后续 v1.x 在此基础上迭代

---

## 2026-07-07: 协方差优化方向 (调研)

- **决策**: ⚠️ 继续调研, 暂不实施
- **调研产出**: `COVARIANCE_RESEARCH.md`
- **原因**:
  - Ledoit-Wolf / 风险平价有理论依据
  - 但 21 日窗口样本不足问题需先解决
  - 当前策略 Calmar 0.98 已不错, 避免风险性优化
- **证据**: `covariance_research.md`
- **下一步**: 单独立项实验 Ledoit-Wolf 收缩协方差

---

## 2026-07-07: Stage 13 交易成本建模

- **决策**: ✅ GO (推荐启用)
- **配置**: `CostModel(enabled=True, commission_bp=5, slippage_bp=10, impact_factor=0.1)`
- **原因**:
  - Calmar 仅降 1.6% (0.98→0.98)
  - 年化成本 -0.12%, 在合理范围
  - 回测更贴近实盘
- **证据**: `stage13_report.md`
- **默认启用**: 否 (用户在 Stage 9-C 基础上手动启用)

---

## 2026-07-07: Stage 10 集中度约束

- **决策**: ❌ NO-GO (放弃)
- **原因**:
  - 单独启用 Calmar 退化 22% (0.78→0.61)
  - 与 Stage 9-C 组合退化 23%
  - 逆波动加权已自然分散
- **证据**: `stage10_report.md`
- **归档**: `experiments/stage_10_caps_failed.md`
- **教训**: 风控过度叠加, 单一功能应测试独立性

---

## 2026-07-07: Stage 9-D HMM Regime 检测器

- **决策**: ❌ NO-GO (放弃)
- **原因**:
  - Calmar 退化 33% (0.78→0.52)
  - HMM 全协方差在 504 天 / 3 regime 下过拟合
  - OOS 显示 regime 切换不稳定
- **证据**: `stage9d_report.md`
- **归档**: `experiments/stage_9d_hmm_failed.md`
- **教训**: 低样本高维协方差 → 必须用收缩估计
- **可能复苏方向**: 改用 Ledoit-Wolf 收缩协方差 + 更长训练窗口

---

## 2026-07-07: Stage 9-C 波动率目标

- **决策**: ✅ GO (推荐默认启用)
- **配置**: `VolTargeting(enabled=True, target_vol=0.15, lookback=60, min_scale=0.3, max_scale=1.5)`
- **原因**:
  - Calmar 0.78 → 1.00 (+28%)
  - DD -21% → -7%
  - OOS Calmar 1.00 (验证 OOS 稳健)
- **证据**: `stage9c_report.md`
- **当前状态**: **默认推荐配置**

---

## 2026-07-07: Stage 9-B 趋势过滤器

- **决策**: ✅ GO (可选启用)
- **配置**: `TrendFilter(enabled=True, benchmark_code="510300", ma_window=200, exposure_bear=0.7, bond_code="511260")`
- **原因**:
  - Calmar 0.78 → 0.88 (+13%)
  - DD -21% → -17% (优于 CICC -18.78%)
  - OOS Calmar 维持
- **证据**: `stage9b_report.md`
- **推荐场景**: 希望保留部分高收益 + 中等风控的用户

---

## 2026-07-07: Stage 9-A 52 周新高信号融合

- **决策**: ✅ GO (可选启用)
- **配置**: `signal_type="fused", signal_fused_weight=0.6`
- **原因**:
  - Calmar 持平 (0.78), DD -20.38% (略降)
  - 与 baseline 互补
  - 边际效果较小
- **证据**: `stage9a_report.md`
- **推荐场景**: 希望降低 DD 而不降低收益的用户

---

## 2026-07-07: Stage 8 17 指标 + 4 维贡献分析

- **决策**: ✅ GO (基础设施)
- **配置**: N/A (分析工具)
- **产出**:
  - `extended_metrics.py` (17 指标)
  - `contribution.py` (4 维贡献)
  - `brinson.py` (Brinson 归因)
- **价值**: 为后续 Stage 提供量化基础

---

## 2026-07-07: Stage 7 Validation 修复

- **决策**: ✅ 修复
- **改动**:
  - `start_points`: (2018,2020,2022) → (2019,2020-06,2022,2023-06)
  - `perturb_lookbacks`: (120,144,168) → (80,100,120)
- **原因**: 原设置与实际敏感区不匹配 (lb=144 非最优)
- **证据**: `validation_fix_report.md`

---

## 2026-07-07: CICC 对齐 (pre-dedup, caps, vol_window, a_total)

- **决策**: ✅ 完成
- **配置**: `RotationConfig` 全面按 CICC 报告实现
- **Bug 修复**:
  - `resample("ME")` 标签错位 → `groupby period`
  - `fill_by_rank` 未检查 caps → 加 caps 检查
  - `apply_stops` 缺 base_categories → 添加参数
- **证据**: 各 stage 报告

---

## 2026-07-07 (之前): 基础阶段 (Stage 1-6)

- **Stage 1-5**: 数据/回测/校验/文档 baseline
- **Stage 6**: pandas 3.0 兼容, 5163 tests pass
- **决策**: 全部 ✅, 为后续阶段打好基础

---

## 决策模式总结 (Pattern Summary)

| 模式 | 决策 | 例子 |
|------|------|------|
| **强信号** | ✅ GO + 默认启用 | Stage 9-C (VolTarget) |
| **中信号** | ✅ GO + 可选启用 | Stage 9-B (Trend), 9-A (Signal) |
| **弱信号** | ✅ GO + 文档保留 | Stage 13 (Cost) |
| **反信号** | ❌ NO-GO + 归档 | Stage 10 (Caps), Stage 9-D (HMM) |

---

## 待决策

| 项 | 等待 |
|----|------|
| 协方差优化 (Ledoit-Wolf + RP) | 调研完成, 待立项 |
| 多策略扩展 | 未开始 |
| 实时数据接入 | 未开始 |
| ML 方法引入 | 未讨论 |

---

**日志结束**

如需新增决策, 请按格式追加本文件。
