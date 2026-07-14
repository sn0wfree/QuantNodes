# Stage 29 — v6.2 ir_expanding 升级为默认 (PROMOTION 决策)

> **日期**: 2026-07-10
> **决策**: v6.2 默认 `sort_method` 从 `"warmup_ir"` → `"ir_expanding"`, 5-fold walk-forward 验证 4/5 胜 v6.1 IC12.
> **状态**: ⭐ **PROMISING** (升为 v6.2 主推, v6.1 IC12 仍为稳定对照)

---

## 1. 背景

Stage 28 默认 `sort_method="warmup_ir"` (12m), 单次 OOS 2022-2026 = **0.629**, 弱于 v6.1 IC12 baseline 0.748 (-15.9%)。

用户要求客观评估 v6.2 未来函数问题 → 决定是否能修复 → 修复后是否有价值。

### 1.1 未来函数审计结论 (Phase 1 C 方向)

完整审计 6 个文件, 16 条代码路径, 结果:

| 路径 | Look-ahead? |
|------|------------|
| `compute_cross_section_ic` | ⚠️ IC 测量用 21d forward (业界标准) |
| `compute_factor_weights` (clip) | ✅ `shift(1)` 防 look-ahead |
| `get_factor_ir_order_expanding` | ✅ 仅看 d_j < d_i, 21d horizon < 月度 30d |
| `get_factor_ir_order_warmup` | ✅ 仅用 warmup 期 (train set) |
| `orthogonalize_factor_panel` (Gram-Schmidt) | ✅ 截面 OLS, 当天数据 |
| `orthogonalize_factor_panel_qr` | ✅ 截面 QR, 当天数据 |
| `_DEPRECATED_get_factor_ir_order` (ir_full) | ❌ **严重** — 已从生产路径移出 |

**结论**: v6.2 5 个 sort_method (除 ir_full) **全部 look-ahead safe**。

### 1.2 1.113 thr=1.0 1.113 不可复现

`v6_2_ablation_metrics_v2.csv` 中 `v6.2_thr1.0` OOS Calmar 1.113, 但当前代码中 `min_ir_threshold` 是 no-op (`compute_factor_weights` 不读该参数), `compute_softmax_weights` 是 dead code (不被 `run_v6_2_backtest` 调用)。

**结论**: 1.113 来自已回退的老代码 (Stage 28 softmax 路线), 当前代码无法复现, **不应使用**。

---

## 2. 5-fold Walk-Forward 验证 (Phase 2 P1 方向)

跑 5-fold walk-forward, OOS = 1 年, training = 累积 2-7 年:

| Fold | 区间 | 训练 | v6.2 ir_expanding | v6.1 IC12 | Δ | 优者 |
|------|------|------|-------------------|-----------|---|------|
| 1 | 2020 | 2018-2019 | **3.022** | 2.060 | +0.96 | v6.2 |
| 2 | 2021 | 2018-2020 | **1.409** | -0.604 | +2.01 | v6.2 |
| 3 | 2022-2023.6 | 2018-2021 | **-0.016** | -0.439 | +0.42 | v6.2 |
| 4 | 2023.7-2024 | 2018-2023.6 | 0.636 | **1.893** | -1.26 | v6.1 |
| 5 | 2025-2026.6 | 2018-2024 | **2.510** | 1.427 | +1.08 | v6.2 |

**跨 fold 统计**:
- v6.2 ir_expanding: mean=**1.512**, min=-0.016
- v6.1 IC12: mean=0.867, min=-0.604
- v6.2 胜 fold: **4/5** (≥ 3/5 阈值)

**结论**: v6.2 ir_expanding **真实地稳定超越** v6.1 IC12, 不是单次 OOS 的运气。

---

## 3. 决策

### 3.1 sort_method 默认值变更

| 项 | 旧 (Stage 28) | 新 (Stage 29) |
|----|--------------|--------------|
| `V6_2Config.sort_method` | `"warmup_ir"` | `"ir_expanding"` ⭐ |
| `V6_2Config.warmup_months` | 12 | 0 (ir_expanding 不需要) |
| 文档标记 | 主推 | ⭐ **PROMISING** |

### 3.2 5 个 sort_method 状态 (Stage 29)

| sort_method | 状态 | OOS Calmar | 备注 |
|-------------|------|-----------|------|
| `ir_expanding` | ⭐ **PROMISING (新默认)** | 0.821 (5-fold mean 1.512) | 5-fold 4/5 胜 v6.1 |
| `predefined` | 备选 | 0.674 | 金融预定义, 固定顺序 |
| `warmup_ir` | 备选 | 0.629 | 早期主推, 5-fold 不如 expanding |
| `qr` | ❌ 不推荐 | 0.056 | Phase 3 失败, 对称旋转无金融意义 |
| `ir_full` | ❌ DEPRECATED | N/A | 严重 look-ahead, 见 `_helpers/deprecated_order.py` |

### 3.3 v6.1 状态

- 仍标 **RECOMMENDED** (5-fold 验证 1/5 胜 v6.2, 但**唯一稳 5-fold 跑过的老策略**)
- 用户可继续用 v6.1 作为保守主线, v6.2 ir_expanding 作为增强版

---

## 4. 代码改动清单 (Stage 29)

### 4.1 新增 (1 个)
- `tests/strategy/momentum_etf_rotation/_helpers/deprecated_order.py` (50 行): DEPRECATED 全样本 IR 排序, 仅供 ablation 对照

### 4.2 改动 (3 个生产文件)
- `QuantNodes/strategy/momentum_etf_rotation/v6_2/factor_orthogonal.py`: 删 `_DEPRECATED_get_factor_ir_order` 函数 (35 行), 从 `__all__` 移除
- `QuantNodes/strategy/momentum_etf_rotation/v6_2/industry_rotation_v6_2.py`:
  - 删 `_DEPRECATED_get_factor_ir_order` import
  - `run_v6_2_backtest` 起始加 `sort_method="ir_full"` → `NotImplementedError` 校验
  - `_orthogonalize_panel` 的 `if sort_method == "ir_full"` 分支: 抛 `NotImplementedError`
  - `V6_2Config.sort_method` 默认 `"warmup_ir"` → `"ir_expanding"`
  - `V6_2Config.warmup_months` 默认 12 → 0
  - `V6_2Config` docstring 重写 (Stage 28 → Stage 29 决策)
- `QuantNodes/strategy/momentum_etf_rotation/v6_2/__init__.py`: 顶部加 ⭐ **PROMISING** 标记 + 5-fold 数据

### 4.3 改动 (5 个 ablation 脚本)
- `scripts/v6_2_phase1_ablation.py`: ir_full 改从历史 CSV 读
- `scripts/v6_2_phase4_warmup_ablation.py`: 同上
- `scripts/v6_2_phase4_grid_ablation.py`: 同上
- `scripts/v6_2_phase3_qr_ablation.py`: 同上
- `scripts/generalization_test_v6_2.py`: 同上

### 4.4 改动 (1 个测试)
- `tests/strategy/momentum_etf_rotation/test_v6_1_v6_2.py`:
  - `test_with_orth_ir_full_runs`: 改 `pytest.raises(NotImplementedError)`
  - 新增 `test_deprecated_helper_runs`: helper 本身能跑通
  - `test_default_is_warmup_ir` → `test_default_is_ir_expanding` (test 名字改, 断言改)
  - `test_valid_sort_methods`: 排除 `ir_full`
  - 新增 `test_ir_full_raises`: 验证生产路径抛 NotImplementedError

### 4.5 新增 (1 个 ablation 脚本)
- `scripts/v6_2_ir_expanding_5fold.py` (115 行): 5-fold walk-forward 验证, 输出 `v6_2_ir_expanding_5fold.csv`

---

## 5. 测试结果

| 项 | 结果 |
|----|------|
| `test_v6_1_v6_2.py` | **38 passed** (新增 2: `test_deprecated_helper_runs`, `test_ir_full_raises`) |
| `v6_2_phase1_ablation.py` | ir_expanding 0.821 (一致), ir_full 仍 0.064 (历史 CSV 一致) |
| `v6_2_ir_expanding_5fold.py` | 4/5 胜 v6.1, mean 1.512 vs 0.867 |

---

## 6. 用户行动指南

### 6.1 想稳: 继续用 v6.1 IC12
```python
from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest
cfg = V6_1Config(ic_min_months=12)
nav = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
```
- 5-fold OOS Calmar mean 0.867, min -0.604
- 已有 production track record (Stage 27 验证)

### 6.2 想增: 用 v6.2 ir_expanding (新默认)
```python
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest
cfg = V6_2Config()  # 默认 sort_method="ir_expanding"
nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
```
- 5-fold OOS Calmar mean 1.512, min -0.016
- 4/5 fold 胜 v6.1
- 单 fold 4 (2023.7-2024) v6.1 强 3x, 需关注

### 6.3 备选: v6.2 predefined
```python
cfg = V6_2Config(sort_method="predefined")
```
- 金融预定义固定顺序, 不依赖数据
- 5-fold 未跑 (单次 OOS 0.674)

---

## 7. Stage 30 — 过拟合修复 (2026-07-14)

### 7.1 交易成本修复

**问题**: v6.2 回测未扣除交易成本, OOS Calmar 0.821 是毛收益.

**修复**: V6_2Config 新增 `cost_enabled=True`, `commission_bp=5.0`, `slippage_bp=10.0`.

**测试结果** (全期 2018-2026):

| 场景 | Calmar | 变化 |
|------|--------|------|
| 不扣成本 | 0.4449 | - |
| 扣成本 (5bp+10bp) | 0.3310 | **-25.6%** |

**结论**: 交易成本影响显著, v6.2 扣成本后 Calmar 仅 0.3310.

### 7.2 起点依赖测试

**测试方法**: 3 个起点 (2018-01-01, 2020-01-01, 2022-01-01), 扣成本.

| 起点 | Calmar (扣成本) |
|------|----------------|
| 2018-01-01 | 0.3310 |
| 2020-01-01 | 0.1923 |
| 2022-01-01 | 0.7668 |

**统计**:
- Mean Calmar: 0.4300
- Std Calmar: 0.2447
- **CV%: 56.9%** (阈值 25%)
- **结果: FAIL**

**结论**: v6.2 起点依赖严重, CV% 56.9% 远超 25% 阈值.

### 7.3 v6.2 定位调整

| 原定位 (Stage 29) | 新定位 (Stage 30) |
|-------------------|-------------------|
| PROMISING (可直接升级) | **研究版本 (需修复过拟合)** |
| combo 50/50 主力 | combo 主力需保守权重 |

### 7.4 后续方向

1. **ensemble v6.2 + v6.1**: 可能降低 CV%, 提升稳定性
2. **保持 combo 50/50**: v6.2 扣成本后弱于 v6.1, combo 依赖需重新评估
3. **直接上 v8 TV-PR**: 绕过 v6.2 问题, 用 13 INDEX_COLS 作为 X

---

## 8. 未来方向 (Stage 30+)

- v6.2 ir_expanding fold 4 (2023.7-2024) 弱于 v6.1: 调查为何这 18 个月正交化失效
- v6.2 + v6.1 ensemble: 借鉴 v3+v5 成功经验, 看是否能 +0.05-0.10
- v6.3 设计: 若 v6.2 跨年稳定性继续验证, 考虑 v6.3 = v6.2 + 风控 (TF/VT)
- 8 个 SmartBeta ETF 池扩展: 当前 44 主池, 加 8 个 SmartBeta 看是否可移植

---

**最后更新**: 2026-07-14
**状态**: Stage 30 完成, v6.2 扣成本后 Calmar 0.3310, 起点 CV% 56.9% (FAIL), 定位降级为研究版本
