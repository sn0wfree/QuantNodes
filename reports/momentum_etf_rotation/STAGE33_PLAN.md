# Stage 33 — 新因子挖掘 + HMM Regime + 跨资产信号 + 代码清理

> **日期**: 2026-07-20
> **前置**: Stage 32 v7.10 硬化完成 (expanding-window OOS Calmar 0.662)
> **分支**: `stage/32-v710-hardening` (继续使用)
> **状态**: 🚧 进行中

---

## 1. 背景

Stage 32 验证了 v7.10 TV-PR 的真实性能:
- expanding-window OOS (2022+): Calmar 0.662, Sharpe 0.779
- expanding-window OOS (2023+): Calmar 1.121, Sharpe 1.168
- full-sample 高估 44% (0.671 vs 0.466 全期)

v7.7 ML 路线失败 (R2 ≈ 0), 确认当前 36 因子用非线性模型无法提取额外信号.
需要从新维度扩展因子池.

**灵感来源**: `~/Public/comovement` (ARWS 资产共振预警系统)
- 40 维特征工程 (动量/风险/技术/跨资产)
- HMM 5 状态资产配置
- Diebold-Yilmaz 溢出指数
- TENET 尾部依赖网络

---

## 2. 工作分解

### A. 代码清理 + 补测试 (1-2天)

**A1. stop_loss 测试** (≥8 tests)
- 文件: `tests/strategy/momentum_etf_rotation/test_v7_6_stop_loss.py`
- 测试: default_none / triggers_at_threshold / cooldown_expires / no_positions_during_cooldown / peak_reset / factory_function / with_trend_filter / improves_calmar

**A2. v7.10 工厂函数测试** (≥5 tests)
- 文件: `tests/strategy/momentum_etf_rotation/test_strategy_versions_v7_10.py`
- 测试: factory_returns_config / lambda_values / stop_loss_enabled / get_version_routing / data_loads

**A3. 清理死代码**
- `pycaret_estimator.py` → ARCHIVED 标记
- `macro_substrategy_v7_7.py` → ARCHIVED 标记
- `adaptive_factor_selector.py` → ARCHIVED 标记
- 移除未使用导入
- `tvpr_estimator.py` `__all__` 移到文件末尾
- `_solve_tridiag` 重构冗余分母

**A4. 归档研究脚本**
- `mkdir scripts/research/`
- 移动 5 个一次性研究脚本

### C. 新因子挖掘: 40 维特征 (1周)

**来源**: `~/Public/comovement/resonance_warning/data/features.py` (只读, 不修改)

**10 个新因子** (不与现有 36 因子重叠):
1. skewness_60d — 60 日偏度
2. kurtosis_60d — 60 日峰度
3. max_dd_60d — 60 日最大回撤
4. macd — MACD 信号
5. bollinger_pct — 布林带位置
6. atr — 平均真实波幅
7. market_beta — 市场 beta
8. dispersion — 截面离散度
9. tail_co_occurrence — 尾部共现
10. vix_corr — 与 VIX 相关性

**执行**:
1. 在 `v7/enhanced_factors_v7_11.py` 实现 10 个新因子
2. 生成 v7.11 数据 (36 + 10 = 46 因子)
3. 计算每个新因子 IC (corr(X[t], Y[t+1]))
4. 筛选 IC > 0.03 的因子
5. expanding-window OOS 验证

**验收**: 新因子组合 Calmar > 0.662

### B. HMM Regime 集成 (1周)

**来源**: `~/Public/comovement/hmm_regime/` (只读)

**思路**: HMM 5 状态叠加 v7.10, regime-aware 权重调整
1. 复用 9 维观测向量 (VIX, TED, 信用利差, GPR, FSI, 油金相关性等)
2. 在 `v7/hmm_regime_overlay.py` 实现
3. 状态×权重乘数调整 v7.10 组合
4. OOS 验证

**验收**: HMM+v7.10 Calmar > 0.662

### D. 跨资产信号集成 (1周)

**来源**: `~/Public/comovement/resonance_warning/data/` (只读)

**信号**:
1. DY 溢出指数 — 跨资产波动溢出
2. TENET 尾部网络 — 风险传导方向
3. DCC regime — 相关性突变预警
4. 尾部依赖 spike — GPD 危机预警

**执行**:
1. 在 `v7/cross_asset_signals.py` 实现
2. 作为额外因子或 regime overlay
3. OOS 验证

**验收**: 叠加后 Calmar > 基线

---

## 3. 验收门槛

| # | 验收项 | 通过标准 |
|---|--------|---------|
| 1 | stop_loss 测试 | ≥ 8 tests pass |
| 2 | v7.10 工厂函数测试 | ≥ 5 tests pass |
| 3 | 死代码清理 | v7.7 标记 ARCHIVED |
| 4 | 新因子 IC | ≥ 3 个因子 IC > 0.03 |
| 5 | 新因子 OOS | Calmar > 0.662 |
| 6 | HMM Regime | Calmar > 0.662 |
| 7 | 跨资产信号 | Calmar > 基线 |
| 8 | 全量测试 | ≥ 40 passed |

---

## 4. 代码改动清单

### 4.1 新增
- `tests/.../test_v7_6_stop_loss.py`
- `tests/.../test_strategy_versions_v7_10.py`
- `v7/enhanced_factors_v7_11.py` (10 新因子)
- `v7/hmm_regime_overlay.py`
- `v7/cross_asset_signals.py`
- `scripts/research/` (归档目录)
- `reports/.../STAGE33_PLAN.md`

### 4.2 改动
- `v7/pycaret_estimator.py` (ARCHIVED 标记)
- `v7/macro_substrategy_v7_7.py` (ARCHIVED 标记)
- `v7/adaptive_factor_selector.py` (ARCHIVED 标记)
- `v7/macro_substrategy_v7_6.py` (移除未使用导入)
- `v7/tvpr_estimator.py` (__all__ 移动 + _solve_tridiag 重构)
- `v7/data_loader_v7_6.py` (section 编号修复)

---

**最后更新**: 2026-07-20
**状态**: 🚧 A 阶段进行中
