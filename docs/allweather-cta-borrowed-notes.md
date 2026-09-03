# CTA 平行 agent 调研笔记 (2026-09-03, 用户指令)

> 目的: 从隔壁 agent 的 CTA 研究中找出对 Sleeve C (CTA) 与 ABC_ew 组合可借鉴的发现
> 调研对象: `/home/ll/Public/QuantNodes/docs/cta/` (23 文档, Sep 2-3 活跃) + `reports/cta/` (丰富 CSV/JSON)
> 核心来源: `cta_tuning_log.md` (711 行, 完整调优链) + `cta_combination_report.md` (子策略组合) + `cta_dynamic_risk_report.md` (风控+子权重)

---

## 1. T1 核心借鉴: EMA=0.15 的正确使用方式 (cta_tuning_log.md §3)

**重大发现**: 他们的 CTA 换手从 149x → 29x (-81%), Sharpe 1.688 → 1.884, MaxDD -10.3% → -5.6%, **用的就是 position_ema_alpha=0.15**!

我们的 α=0.10 失败原因: **应用对象错了**。我把 EMA 应用到了 raw sign 面板上 (sign 经 EMA → 均值≈0, 杀掉信号)。他们的做法是 EMA 应用到 **`_positions()` 产出的 fractional 仓位上** (CTA 模块内部: `signals → vol_target_positions → EMA → lots`), 仓位是 [-3, +3] 范围的浮点 (受 vol_target 控制), EMA 平滑的是仓位变化速度, 不是信号本身.

**复现路径**:
- 不要自己 EMA sign 面板, 直接调用 `combo._positions()` 拿平滑后的仓位面板
- 或按他们的方法: 先 `vol_target_positions(signals, price_panel)` → fractional weights → EMA 0.15 → 转 lots
- **正确预期**: 复现他们 29x 换手 + Sharpe 1.88+ 应该可行

**对 ABC_ew 的预期**: 当前 CTA sleeve Sharpe 1.092 / 换手 153x / 成本 1.53%; 修复后预期 Sharpe ~1.5-1.9 / 换手 ~30x / 成本 ~0.3%. 整个 ABC_ew 换手从 83x 降到 ~40x, 成本从 0.67% 降到 ~0.3%, Sharpe drag 0.13 → ~0.05.

---

## 2. 子策略组合: 80% 主CTA + 5% ZN库存 + 15% CU溢价TC (cta_combination_report.md)

**结果**: Sharpe **2.512**, OOS **2.001**, MaxDD **-4.45%**, AnnRet 16.4%, Calmar **3.69**.

| 子策略 | OOS Sharpe | 状态 |
|--------|-----------|------|
| 主CTA | 1.799 | 基线 |
| ZN库存 | 0.476 | ✅定版 (Inventory z-score) |
| CU溢价TC | **1.10** | ✅定版 (洋山铜溢价+TC, 迄今最强子品种) |
| M经纪持仓 | -0.262 | ❌止损 (PIT bug 修正后) |
| 价差套利 | — | ❌止损 (一致率 56%<60%) |
| 黄金独有信号 | — | ❌止损 |

**相关性矩阵** (关键):
- 主CTA × ZN库存 = 0.078 (低相关)
- 主CTA × CU溢价TC = 0.029 (极低相关)
- **ZN库存 × CU溢价TC = -0.068 (负相关)** — 子策略间本身互补

**启发**: 我们 ABC_ew 用单一 CTA combo sleeve, 不区分 ZN库存/CU溢价TC 子策略. **下一步可拆 sleeve C 为 C_main + C_ZN + C_CU 三个子 sleeve, 按风险预算混合, 预期 Sharper 进一步提升**. 

---

## 3. 动态风险 + 自适应子权重: 最佳 CTA 实战配置 (cta_dynamic_risk_report.md, Sep 3 最新)

**主推方案**: CTA55% (动态杠杆[1.0,1.5]) + CU溢价TC 31.5% + IH50动40×OI10 13.5% + 动态杠杆 + 自适应子权重

| 指标 | 主CTA单独 | 上轮最优 | **本轮主推** |
|------|----------|----------|-------------|
| Sharpe | 2.378 | 2.381 | **2.781** (+0.40) |
| AnnRet | 19.00% | 16.14% | **16.66%** (+0.52pp) |
| MaxDD | -6.48% | -4.32% | **-3.91%** |
| Calmar | 2.93 | 3.74 | **4.26** (+0.52) |
| OOS Sharpe | 1.799 | 1.955 | **2.414** (全系列最高) |

**动态杠杆机制**: vol60 驱动, vol≤8% → 杠杆 1.5, vol≥12% → 杠杆 1.0, 中间线性插值.

**自适应子权重**: 滚动 60d Sharpe < 全期中位 × 0.5 且 < 0 → 权重减半; 恢复后自动加回.

**对 ABC_ew 的启发**:
- 我们的 ABC_ew 静态等权 + cap6 fixed leverage, 不带 vol 调节
- 可借鉴动态杠杆: vol60 驱动的 cap ∈ [1.0, 2.5] (按 sleeve A vol 调), 替代我们的固定 cap6
- 可借鉴自适应子权重: 三 sleeve 滚动 Sharpe 监测, 弱 sleeve 自动降权

---

## 4. 杠杆结构性封顶: 国内商品 vol 中位 ~35% (cta_tuning_log.md §4)

**关键发现**: 组合杠杆被结构性钉在 ~0.55, 与 target_vol 不成比例.

**机制**:
- weights_i = signal × clip(target_vol/vol_i, 3.0)
- 国内商品 vol 中位 ~35% → tv=1.5 时 1.5/0.35=4.3 → clip 到 3.0
- tv=0.30 → 0.30/0.35=0.86 (未 clip) → 杠杆自由放大到 0.49
- tv=0.55 → 0.55/0.35=1.57 → 0.54
- tv=1.50 → 1.50/0.35=4.3 → clip 到 3.0 → 实际杠杆 0.53

**对 ABC_ew 的含义**: 我们 X2-L4 用 cap=4.0 但实际 avg_lev=1.75, 等于 1.4 倍 × cap 显示的 4. 想要真正加杠杆到 2.3, 必须用 cap6-7 (我们测的 (6,3) 是 avg_lev 2.28, 实际 cap 6 ≈ avg 1.93 与目标 2.3 之间). **他们的"动态杠杆[1.0,1.5]" 控制更精细, 可借鉴**.

---

## 5. 成本校准 (cta_tuning_log.md §5)

**陷阱**: DB 的 `contract_specs.tick_size` 有坏数据 (LC tick=0.01 错误, I 铁矿石 smoothed 表价格异常 200 元, AU/SN 价格异常高). 不能直接依赖 DB 建成本表.

**解决**: 硬编码交易所标准 tick + 典型价格, 建立 COMMISSION_BPS (47 品种) + SLIPPAGE_BPS (per tick ÷ 典型价格) 表.

**对 ABC_ew 的成本**: 我们 A/B sleeve 用 2bp blended, CTA sleeve 用 1bp (FUTURES_COMMISSION_RATE). 他们校准后 CTA sleeve cost 1.6%/年 (我们估算 1.53%) — 一致, 我们的近似合理.

---

## 6. CTA 实战 OOS 衰减对照 (重要基准)

| | IS | OOS | 衰减 |
|---|----|----|------|
| 主CTA alone | 2.956 | 1.799 | -39% |
| 主CTA + 子策略 (2.512) | 3.013 | 2.001 | -34% |
| 主推 (2.781) | — | 2.414 | — |

**OOS 衰减 ~30-40%** 是 CTA 策略的真实特征. 我们的 ABC_ew **没有做 OOS 测试** (全期 in-sample). 这是一个显著的盲点 — 用户的 ABC_ew Sharpe 1.897 可能严重 OOS-衰减.

**建议**: 对 ABC_ew 进行 OOS 切段测试 (如 2018-2022 IS / 2023-2026 OOS), 评估真实表现, 与文章 OOS 衰减对照.

---

## 7. 借鉴优先级 (下一步行动清单)

| 优先级 | 借鉴项 | 预期收益 | 风险 |
|--------|--------|----------|------|
| **P0** | **CTA sleeve 修复 EMA 应用对象** (调用 `_positions()` 取平滑仓位, 不再 raw-sign) | Sharpe 1.09→1.5-1.9, 换手 153x→30x | 需验证模块 API 不变 |
| **P1** | 拆 CTA sleeve 为 C_main + C_ZN + C_CU 三子 sleeve 按风险预算混合 | ABC_ew Sharpe 1.897→可能 2.0+ | 需对接 ZN库存/CU溢价TC 子脚本 |
| **P1** | ABC_ew 加动态杠杆 (vol60 驱动 cap 范围) | 压力段更稳, Calmar 提升 | 需引擎支持动态 cap |
| **P2** | ABC_ew OOS 切段测试 | 暴露真实 OOS 衰减 | 无 (纯测试) |
| **P3** | 自适应子权重 (滚动 Sharpe 减半机制) | 弱 sleeve 抗衰 | 需引擎加 sleeve 权重 hook |

P0 优先级最高 (修复换手 + Sharpe 同步), 也是回报最直接的.

---

## 8. 三子因子调研 (2026-09-04, 用户指令)

### 8.1 核心发现: 三个维度全部已在 CTA 因子库中实现

V5 归档时判定"量能维度无法从公开信息复原"——但调研发现**我们的代码库已完整实现三个维度**，无需从零构建。

| 子因子 | 信息维度 | 已有因子 (CTA因子库) | 文件 |
|--------|----------|---------------------|------|
| **Sub-A** 时序波动率 | 波动 regime | `VolatilityConeFactor` | volume_decay.py:45 |
| **Sub-B** 趋势力度 | 价格方向+强度 | `TrendQualityFactor` + `MomentumSlopeFactor` | trend_quality.py:32, 70 |
| **Sub-C** 量能确认 | 成交量/OI 方向 | `VolumeTrendFactor` + `OpenInterestTrendFactor` + `OIPriceDivergenceFactor` + `VolumeConfirmationFactor` + `OBVFactor` + `VolumePriceCorrFactor` | volume.py:21, 70, 166, 192; 93因子调研 |

**关键**: V5 遗漏的 Sub-C (量能确认) 在 CTA 因子库中有 **6 个相关因子**， 全部可用。

### 8.2 DVS 策略的量价共振模式 (docs/109, 111)

DVS（动态波动率策略）已实现"量价共振拐点检测"：

```python
# turn_point.py — TurnPointBreakoutFactor
# 价格突破 + 成交量确认 = 真实拐点
# 参数: lookback=20, atr_mult=0.5, vol_mult=1.5, vol_lookback=10
# 测试: 对2024年有+0.2%收益改善, MaxDD -5.1%→-4.8%
```

**对 AllWeather 的启示**: 量价共振 = Sub-B (价格突破) + Sub-C (放量确认), 这正是避险猫因子的"三维合成"中的 B+C 维度组合. DVS 已验证该模式在期货上可行.

### 8.3 黄金多因子组合验证 (docs/gold)

- **6因子等权**: 年化 13.94%, Sharpe 1.35, OOS/IS = 4.47
- **趋势B&H (60%) + 因子信号 (40%)**: 年化 16.39%, Sharpe 1.37
- 过拟合检测: 7/9 通过 OOS/IS > 0.8 阈值

**对 AllWeather 的启示**: 多因子组合在黄金上已验证有效, 且等权 + 因子择时是稳健的组合方式. OOS/IS = 4.47 远超 0.8 阈值, 说明该方法不依赖样本内过拟合.

### 8.4 Quant Alpha 因子挖掘教训 (docs/quant_alpha)

- **简单逻辑 > 复杂逻辑**: 单算子约束成功率高, 6算子组合 (price_volume_divergence) 全部失败
- **波动率因子反向**: 高波动率股票表现差 (IR = -0.12), 符合金融学直觉
- **变量白名单有效**: 聚焦 close/volume/open 减少无效因子 80%

**对 AllWeather 的启示**: Sub-A/B/C 作为连续因子 (非二值触发器) 保留信息量, 避免 trigA/trigB 失败的"硬阈值信息丢失"问题.

### 8.5 V5 → V6 修正路径

| 维度 | V5 (已归档) | V6 (新方案) |
|------|------------|-------------|
| Sub-A/B 实现 | trigA(trigB)自写, ad-hoc | 直接用 CTA 因子库已有因子 |
| Sub-C (量能) | 完全缺失, 判死主因 | **6 个已有因子**, 无需新数据管线 |
| 信号类型 | 二值触发器 (0/1) | 连续因子分数 (信息更丰富) |
| 组合方式 | 未到组合就失败 | 可用黄金多因子方法 (等权+因子择时) |
| 数据需求 | 需新 volume/OI 管线 | 全部已有, 零新管线 |
| V5 gate 失败原因 | Sub-A/B 混合无法归因 + Sub-C 缺失 | 三维度独立可归因, Sub-C 已实现 |
