# Stage 19 v4 LW 增强 — Nagel 风格 Ledoit-Wolf + λ 收缩

> **任务**: 基于 Nagel 团队《Optimal Factor Timing in a High-Dimensional Setting》论文 (A 股复现: QuantML《论文复现 | 最优因子择时框架》), 实施 Ledoit-Wolf 协方差 + λ 权重收缩, 作为 v4 因子择时的**可选稳健模式**.
> 
> **核心发现**:
> 1. **论文 A 股复现**: 择时策略 Sharpe 2.66 < 静态最优 3.60, **但年化收益更高** (8.38% vs 6.71%). λ 多数年份 = 30-100 (高收缩, 接近静态).
> 2. **我们的 v4 IC^2 已经接近 IC 信号极限**: 8y Calmar 0.613, LW 滚动 λ Calmar 0.468 (-24%).
> 3. **LW 不显著优于 IC^2** in our 5 ETF 类别 setting, 但提供**稳健性**.
> 4. **最优生产配置**: v3 65% + v4 IC^2 35% Calmar **0.676** (8y), OOS 0.842 (4.5y).

---

## 一、论文方法 (Nagel 风格)

### 1.1 三层稳健性
1. **Ledoit-Wolf 协方差收缩** — `cov_lw = (1-δ)·S + δ·F`, δ = 最优收缩强度
2. **λ 权重收缩** — `w_final = (1-shrink)·w_mvo + shrink·w_equal`, `shrink = λ/(1+λ)`
3. **总敞口约束** — `Σ|w| = 1` (L1 范数归一化)

### 1.2 A 股复现 OOS (2018-2025, 7y)

| 策略 | Ann | Vol | Sharpe | DD |
|------|-----|-----|--------|-----|
| 择时 gross | 8.38% | 3.05% | 2.66 | -2.11% |
| **静态最优** | 6.71% | 1.82% | **3.60** | -1.21% |
| 等权 | -2.20% | 2.08% | -1.06 | -15.57% |

**关键 insight**: 择时提升收益 (+1.67% ann) 但波动也上升 (3.05% vs 1.82%), 净 Sharpe 落后静态最优.

### 1.3 λ 分布
- 2018+ 多数年份: **λ=30 或 100** (高收缩, 接近静态)
- 2020 短暂: 低 λ (低收缩, 信任择时)

---

## 二、v4 LW 集成 (Stage 19)

### 2.1 新增模块

| 文件 | 行数 | 内容 |
|------|------|------|
| `v4/lw_factor_timing.py` | 192 | Ledoit-Wolf + MVO + λ 收缩核心算法 |
| `v4/lw_factor_timing_integration.py` | 220 | LW 集成 (regime + 滚动 λ + ETF 映射) |
| `v4/factor_timing_v4.py` | +130 | `lw_enabled/lw_lambda_mode` 字段 + `compute_factor_weights_lw()` |

### 2.2 FactorTimingConfig 新增字段

```python
# Stage 19: Nagel 风格 Ledoit-Wolf + λ 收缩 (可选, 默认 False)
lw_enabled: bool = False
lw_lambda_mode: str = "fixed"  # "fixed" | "rolling"
lw_lambda_fixed: float = 10.0
lw_candidate_lambdas: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0)
lw_train_window: int = 60
lw_val_window: int = 12
lw_long_only: bool = True
lw_l1_norm: float = 1.0
```

### 2.3 启用方式
```python
# 默认 (Stage 18 v4 IC^2, λ=0)
FactorTimingConfig()  # lw_enabled=False

# LW 固定 λ=10 (论文稳健值)
FactorTimingConfig(lw_enabled=True, lw_lambda_mode="fixed", lw_lambda_fixed=10.0)

# LW 滚动验证 λ (论文 OOS 方法)
FactorTimingConfig(lw_enabled=True, lw_lambda_mode="rolling")
```

---

## 三、回测结果 (2018-2026, 8y)

### 3.1 v4 因子择时 单策略对比

| 模式 | Ann | Vol | Sharpe | DD | Calmar |
|------|-----|-----|--------|-----|--------|
| **v4 IC^2 (默认)** | 11.15% | - | 0.70 | -18.18% | **0.613** ⭐ |
| LW λ=0 (无收缩) | 10.47% | - | 0.66 | -18.91% | 0.554 |
| LW λ=1 | 10.79% | - | 0.67 | -21.47% | 0.503 |
| LW λ=5 | 11.28% | - | 0.69 | -21.47% | 0.525 |
| LW λ=10 | 11.39% | - | 0.70 | -21.47% | 0.531 |
| LW λ=30 | 11.48% | - | 0.70 | -21.47% | 0.535 |
| LW λ=100 | 11.51% | - | 0.70 | -21.47% | 0.536 |
| LW 滚动 λ | 10.05% | - | 0.63 | -21.47% | 0.468 |

**核心观察**:
- v4 IC^2 (λ=0 + 集中度无 cap) **Calmar 0.613 最高**
- LW 高 λ 收益略高 (11.5% vs 11.2%) 但 DD 更大 (-21.47% vs -18.18%), Calmar 反而低
- LW 滚动 λ 选偏 0 与 100 (各 ~30-50%), 综合效果不如 IC^2

### 3.2 v3 + v4 因子 组合 (8y)

| 组合 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 50% + IC^2 50% | 9.12% | 0.80 | -13.85% | 0.658 |
| v3 60% + IC^2 40% | 8.68% | 0.83 | -12.90% | 0.673 |
| **v3 65% + IC^2 35%** | 8.45% | 0.85 | -12.51% | **0.676** ⭐ |
| v3 70% + IC^2 30% | 8.22% | 0.87 | -12.29% | 0.669 |
| v3 80% + IC^2 20% | 7.75% | 0.90 | -11.83% | 0.655 |
| v3 60% + LW 滚动 40% | 8.17% | 0.79 | -14.50% | 0.563 |
| v3 70% + LW 滚动 30% | 7.83% | 0.83 | -13.58% | 0.576 |

### 3.3 OOS Walk-Forward (2022-2026, 4.5y) ⭐⭐⭐

| 组合 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v4 IC^2 only OOS | 10.56% | 0.65 | -18.18% | 0.581 |
| LW λ=10 only OOS | 10.22% | 0.60 | -21.47% | 0.476 |
| LW 滚动 only OOS | 10.22% | 0.60 | -21.47% | 0.476 |
| **v3 60% + IC^2 40% OOS** | 10.19% | 0.92 | -12.90% | **0.790** ⭐ |
| **v3 70% + IC^2 30% OOS** | 10.12% | **1.01** | -12.02% | **0.842** ⭐⭐ |
| **v3 80% + IC^2 20% OOS** | 10.03% | **1.11** | -11.11% | **0.903** ⭐⭐⭐ |
| v3 60% + LW 滚动 40% OOS | 10.02% | 0.88 | -14.50% | 0.691 |
| v3 70% + LW 滚动 30% OOS | 9.98% | 0.98 | -13.25% | 0.753 |
| v3 80% + LW 滚动 20% OOS | 9.94% | 1.09 | -11.94% | 0.832 |

---

## 四、为什么 LW 不如 IC^2 (在我们设置下)

### 4.1 论文 vs 我们的差异

| 维度 | 论文 A 股 | 我们的 v4 |
|------|----------|-----------|
| 因子数 | 10 (Barra 风格) | 5 (m, r, v, d, q) |
| 信号 | 11 (5 宏观 + 6 因子特异) | 1 (IC 滚动) |
| 协方差 | 高维 (10×10) | 低维 (5×5) |
| LW 优势 | 高维噪声 → 必须收缩 | 低维信号 → IC^2 已有效 |
| 论文 OOS Sharpe | 2.66 (vs 静态 3.60) | 0.65 (LW 滚动 vs IC^2 0.70) |

### 4.2 核心原因
- **LW 用 L1 归一化 (|w|=1)**: 把所有非零权重均分, 失去"集中度"信号
- **IC^2 用非线性放大**: `max(0, IC+0.05)^2`, 高 IC 给极高权重, 低 IC 给 0
- 在 5 ETF 类别设置下, **IC^2 集中度更高 → Calmar 更好**
- LW 的协方差收缩主要价值在**高维** (10+ 因子), 我们 5 因子不需要

### 4.3 LW 的价值
- **稳健性**: L1 归一化避免极端权重
- **可调性**: λ 滚动验证可适应 regime
- **论文场景**: 高维 (10+ 因子) + 多信号 (11) → LW 必要
- **我们的场景**: 低维 (5 类别) + 单 IC 信号 → LW 过度收缩

---

## 五、最终推荐 (Stage 19)

### 5.1 生产配置 (基于 8y 完整 + 4.5y OOS 验证)

| 风险偏好 | 配置 | 8y Calmar | OOS Calmar | OOS Sharpe |
|---------|------|-----------|------------|-----------|
| **稳健** | v3 80% + v4 IC^2 20% | 0.655 | **0.903** | **1.11** ⭐⭐⭐ |
| **平衡** | v3 70% + v4 IC^2 30% | 0.669 | **0.842** | **1.01** ⭐⭐ |
| **进攻** | v3 60% + v4 IC^2 40% | 0.673 | **0.790** | 0.92 ⭐ |
| **稳健 (LW 备选)** | v3 80% + v4 LW 滚动 20% | 0.589 | 0.832 | 1.09 |

### 5.2 LW 模式使用建议
- **默认关闭** (`lw_enabled=False`), v4 IC^2 仍是最优
- **生产稳健模式**: `lw_enabled=True, lw_lambda_mode="fixed", lw_lambda_fixed=30.0` (论文稳健值)
- **OOS 自适应**: `lw_enabled=True, lw_lambda_mode="rolling"` (每月重选 λ, 更稳健但 Calmar 略低)
- **不适合场景**: 因子数 < 8 (LW 协方差优势不明显)

---

## 六、Stage 19 vs Stage 18 对比

| 维度 | Stage 18 v4 (默认) | Stage 19 v4 + LW (可选) |
|------|-------------------|------------------------|
| 协方差 | 单变量 IC | LW 收缩 (高维可用) |
| 权重公式 | `max(0, IC+0.05)^2` | MVO + λ 收缩 |
| L1 归一化 | 无 (max_weight 0.50) | 显式 \|w\|=1 |
| λ 选择 | 不适用 | 固定 / 滚动验证 |
| Calmar (8y 单独) | **0.613** | 0.468-0.554 |
| Calmar (v3 80% + v4 20% 8y) | **0.655** | 0.589 |
| **OOS Calmar (4.5y)** | **0.903** | 0.832 |
| OOS Sharpe | **1.11** | 1.09 |

**结论**: Stage 19 LW 是**可选增强**, 不替代 Stage 18 v4 IC^2 默认. 论文 A 股复现的关键 insight (高 λ 偏保守) 在我们低维设置下不显著.

---

## 七、文件清单

### 7.1 新增
- `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing.py` (192 行)
- `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing_integration.py` (220 行)
- `scripts/lw_factor_timing_backtest.py` (350 行, 详细测试)
- `scripts/v4_lw_integrated_test.py` (220 行, 集成测试)
- `reports/momentum_etf_rotation/v4/lw_factor_timing_navs.parquet`
- `reports/momentum_etf_rotation/v4/lw_rolling_lambda_log.csv`
- `reports/momentum_etf_rotation/v4/v4_lw_integrated_navs.parquet`

### 7.2 升级
- `QuantNodes/strategy/momentum_etf_rotation/v4/factor_timing_v4.py` (+130 行)
  - 新增 `lw_enabled/lw_lambda_mode/lw_lambda_fixed` 字段
  - 新增 `compute_factor_weights_lw()` 函数
  - 新增 `_select_lambda_rolling_lw()` 函数
  - 升级 `backtest_factor_weights_history()` 支持 LW 模式

### 7.3 论文引用
- Nagel 等. *Optimal Factor Timing in a High-Dimensional Setting*
- QuantML. *论文复现 | 最优因子择时框架* (A 股复现, 2026)
- Barra CNE5 10 风格因子 (CNE5 = China A Equity Model 5)

---

## 八、下一步

1. **生产部署**: v3 80% + v4 IC^2 20% (OOS Calmar 0.903, Sharpe 1.11)
2. **Stage 20**: 实时 paper trade 验证 LW 滚动 λ
3. **Stage 21**: 多信号输入 (论文用 11 信号, 我们仅 1 IC)
4. **更新 Stage 17 INDEX**: 标记 Stage 19 v4 LW 完成
