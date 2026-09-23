# 全天候策略 — 生产候选配置登记

> 登记日期: 2026-09-06 | 状态: **候选定版 (待实盘验证)** | 出处: §5.37-§5.42

---

## 1. 配置规格

```
池:      REF41 (41 资产: 10 ETF + 31 期货, 商品占比 ~64%)
引擎:    D v3 管线 (d_pipeline.py)
         └ cov=lw / W-FRI 调仓 / bond_cap=0.40 / window=252 / min_obs=60
杠杆:    vol_target = 15% (max_leverage=2.5)
叠加层:  ① D.1 负趋势 (trend, balanced 0.5/0.3) + rapid
         ② D.2 动量倾斜 (β=0.20, lookback=60)
         ③ P1-B 相关性覆盖 (θ=0.60, k=1.0)      ← §5.37
         ④ 共振门控 AR3 (60日/进P70/出P40∧volP50/dd_floor -3%/k=0.7)  ← §5.41
资本口径: 名义口径 + 融资 2% (回购)
```

## 2. 业绩 (样本内 2019-01 → 2026-06, 含成本与融资)

| 指标 | 值 | 基线 (无叠加层) | 改善 |
|------|-----|----------------|------|
| CAGR | **10.85%** | 10.51% | +0.34pp |
| Sharpe | **1.35** | 0.98 | **+0.37** |
| Vol | 8.20% | 11.30% | -3.1pp |
| **MaxDD** | **-10.30%** | -14.86% | **+4.56pp** |
| Calmar | ~1.05 | 0.71 | +0.34 |

## 3. Walk-forward 复验 (OOS 2022-2026)

| 指标 | 组合配置 | 基线 |
|------|---------|------|
| OOS 均值 Sharpe | **1.01** | 0.70 |
| OOS 均值 MaxDD | **-6.0%** | -9.2% |
| 2022 危机年 DD | **-5.2%** | -14.9% |
| 参数稳定性 | **5/5 折同一配置** | — |

## 4. 目标达成情况

| 目标 | 要求 | 实际 | 达成 |
|------|------|------|------|
| CAGR | ≥ 10% | 10.85% | ✅ |
| MaxDD | ≤ 10% | -10.30% | ⚠️ 差 0.30pp (用户确认可容忍, 不做微调) |
| Sharpe | ≥ 2.0 | 1.35 | ❌ (诚实基线, 路径在资产池/真 alpha) |

## 5. 启用边界 (重要)

| 池类型 | 特征 | P1-B | 共振门控 | 说明 |
|--------|------|------|---------|------|
| **REF41** | 商品 64%, N_eff 6.8 | ✅ k=1.0 | ✅ k=0.7 | **本候选** |
| P2b 宽基18 | N_eff 9.0, 分散充分 | ❌ | ❌ (OOS Sh -0.21) | 禁用 |
| P0 对齐12 | N_eff 5.5 | ❌ | ❌ | 禁用 |

**规律**: 结构类叠加层 (P1-B / 共振) 的价值与池的**结构性分散不足程度正相关**。

## 6. 关键机制

- **P1-B 与共振门控互补** (§5.42): 前者日频微调、后者事件级大幅降杠杆;
  单独 P1-B 在 vt=15% 高杠杆下有害 (8.81%), 与共振组合后 Sharpe 1.35 / DD -10.30%
- **共振门控入场条件**: 未深亏 (回撤 ≥ -3%) 才行动 — 已深亏时 AR3 是滞后信号
- **贝叶斯证据**: AR3∩组合波动 的 logBF10 = +9.31 (决定性), 但波动仅用于**出场确认**

## 7. 已知遗留 (不影响本候选)

| # | 项 | 优先级 |
|---|----|--------|
| 1 | OHLCV 复权与 NAV 交叉验证 9 ETF 不一致 | P2 |
| 2 | cash_proxy (中债1-3年财富) 停在 2026-05-29 | P2 |
| 3 | CTA v7 信号缓存截至 (影响 2026 C sleeve) | P3 |
| 4 | P2b 宽基18 工具化 (csi_1000 → 512100/IM) | P2 |
| 5 | 老脚本 upload_tick_data.py 5h37m 异常 | P3 |

## 8. 复现命令

```python
# v2 (2026-09-23 升级): 月频 + D.1 aggressive + 共振 k=0.9 + Carry sleeve 20%
from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.carry_sleeve import (
    build_carry_sleeve, combine_with_d)

res = run_d_sleeve(panel_ref41, bond_cap=0.40, window=252,
                   rebalance_freq='ME', cov_estimator='lw',          # 月频
                   neg_trend_mode='trend', neg_trend_preset='aggressive',
                   include_rapid=True, momentum_beta=0.20,
                   volume=vol, oi=oi, asset_classes=ASSET_CLASSES,
                   cash_ret=cash_ret, target_vol=0.15, max_leverage=2.5,
                   financing_rate=0.02,
                   correlation_k=1.0, correlation_threshold=0.60,
                   resonance_k=0.9, resonance_ar3_window=60,
                   resonance_enter=0.70, resonance_exit=0.40,
                   resonance_vol_exit=0.50, resonance_dd_floor=-0.05)
carry_ret, _ = build_carry_sleeve(panel_ref41, carry, freq='ME')
nav = combine_with_d(res.nav, carry_ret, carry_weight=0.2, leverage=1.0)
# 2019 起口径: CAGR 10.06% / Sharpe 1.81 / MaxDD -5.98% (IS)
# WF OOS: Sharpe 1.53 / MaxDD -3.6%
```

**验证脚本**: `model_ablation_test.py` / `wf_combo_overlay.py` / `production_validation.py`
