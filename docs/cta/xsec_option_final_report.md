# 期权池项目终局报告 — 三条路线全部验证完毕

> 日期: 2026-09-16
> 模块: `scripts/cta_xsec_option/`
> 结论: **截面 alpha / IV 增强 / 凸性 overlay 三条路线均不成立, 项目策略层归档, 基础设施保留**

---

## 一、三条路线的完整验证记录

### 路线 1: 期权 OI/IV 作为截面 alpha（Stage 0-1）

| 实验 | 结果 |
|---|---|
| 纯截面动量 (27品种, §26 S2 改进) | Sharpe -0.585, 截面 IC -0.068 (t=-2.2, 显著反向) |
| skew_reversal 反转方向 | 10 品种子集 IC -0.096 (t=-2.3) → 商品语境反向 |
| skew 全池 IC | |t| < 1.5 全部不显著 (新品种稀释) |

**结论: 持仓类信号 (CFTC/经纪/OI) 在本项目内已三次独立证伪。**

### 路线 2: IV 信息增强截面动量（IV 消融）

| 配置 | Sharpe | OOS |
|---|---|---|
| 基线纯动量 | -0.585 | -0.599 |
| 最好: mom+skew_trend×双乘数 | -0.412 | -0.011 |

IV 是"确认器"不是"生成器" — 相对改善 +0.17 真实存在, 但无法转正。

### 路线 3: 期权凸性 overlay 跟随主 CTA（A 路线）

| 配置 | 结果 |
|---|---|
| delta等效 0.30 NAV 持续持有 | 年 premium **46%**, MaxDD -10.6%, 腿 Sharpe -1.90 |
| 组合叠加 | 主 CTA 2.393 → 2.314 (恶化) |

**物理约束核算**:
- 30d OTM/价差的日 carry 中位 ≈ 1.4-2.0% 面值/日（含近到期异常, 稳健 ATM 量级 ~9%/年面值）
- **预算 2%/年 → delta 等效敞口上限 0.004-0.006 NAV**
- 凸性贡献上限 ≈ 0.02%/周, 对比主 CTA 周均波动 0.4% — **无感是物理必然**

## 二、根本认知（本轮学习的核心结论）

1. **凸性不免费**: 30d 期权买方的年化权利金 ≈ 8-10% 面值。要凸性有感（敞口 ≥ 0.3 NAV），年成本 ≥ 3-5% NAV — 只适合作为"危机保险"（低频事件驱动），不适合持续持有。
2. **截面无 alpha**: 该 27 品种池周频截面上不存在可持续相对强弱（动量 IC 显著负、skew 不显著）。IV/OI 数据改变不了这个事实。
3. **IV 的正确角色**: 确认器/执行结构选择器（低波 OTM / 高波价差），不是 alpha 源。

## 三、保留资产（基础设施残值高）

| 资产 | 路径 | 价值 |
|---|---|---|
| 全量 greeks cache | `data/cache/option_greeks_ext/option_greeks_ext.parquet` (445MB, 27品种×8.5年, valid 89.9%) | 期权研究通吃底座 |
| 日频 skew 面板 | `reports/cta/xsec_option/skew_daily_27.parquet` | 商品 skew 顺势证据 (10品种 h10 t=2.3) |
| 主 CTA 信号快照 | `reports/cta/xsec_option/main_cta_signals.parquet` (1941d×66品种) | 任何 overlay 实验的输入 |
| 信号层代码 | `iv_signals.py` / `option_convexity_layer.py` / `cta_xsec_momentum.py` | 可复用组件 |
| 参数快照脚本 | `snapshot_main_cta_signals.py` | 可重跑 |

## 四、G 门槛终判

| 门槛 | Stage 0 | IV 消融 | 凸性 overlay |
|---|---|---|---|
| OOS Sharpe > 0.15 | ❌ -0.60 | ❌ -0.011 | ❌ 腿 -1.9 |
| 毛价差 > 5% | ❌ 0.34% | ❌ | — |
| premium 预算 ≤ 2% | — | — | ❌ 46% (敞口版) / 敞口无感 (预算版) |

**0/9 → 项目策略层归档。**

## 五、后续选项（供讨论）

| 选项 | 内容 |
|---|---|
| A' | **归档**: 基础设施保留, 策略层不再推进 (推荐 — 三次独立验证已足够) |
| B' | 事件驱动凸性: 仅在主 CTA 信号+IV 极端时买入 (premium 磨损降量级), 但需新验证周期且偏事件挑选 |
| C' | 期权池转向卖方 VRP 收割 (docs/105 先例: 边际正期望需强对冲) — 独立新项目 |
| D' | 基础设施反哺: greeks cache 用于主 CTA 的 IV 过滤 (docs/106 已验证 VRP +0.08, 72品种 IVRegime +26%) |

## 六、产物索引

- `docs/cta/xsec_option_design.md` — 设计文档 (含全决策记录)
- `docs/cta/xsec_option_stage0_failure.md` — Stage 0 失败诊断
- `docs/cta/xsec_option_iv_ablation.md` — IV 消融结果
- `docs/cta/xsec_option_final_report.md` — 本文档
- `reports/cta/xsec_option/iv_enhance_ablation.csv` / `skew_daily_27.parquet` / `main_cta_signals.parquet`
- `data/cache/option_greeks_ext/` — 全量 cache (28 文件)
