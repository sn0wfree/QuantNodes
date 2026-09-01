# 黄金策略最终方案文档

> 最后更新: 2026-09-01 (v4 执行层 + 完整链路 + 归因分析)
> 最终策略: `compute_layered_ensemble_signal` (统一入口真实shift2 + 无前视Open-to-Open)
> 全周期真实业绩: CAGR 6.35% / Sharpe 0.886 / MaxDD -17.33% (O2O+5bps, 2010-2026)

---

## 🆕 v4 执行层 (2026-09-01)

> 状态: **模块已落地, 生产未切换** (建议纸面跟踪3个月后再切换)
> 模块: `scripts/cmf/gold_backtest.py` (`build_full_stack` / CLI `--stack`)

### 4.1 架构: 预算制执行栈 (三层)

```
生产链(-5%/-5%/-20%) → chain (0~1.8)
  ↓ 核心裁剪: clip(chain, 0, 1.25)            ← 核心信号最多用125%
  ↓ 周网格库存 (每周五收盘一次):                ← 多余25%杠杆让渡给网格
  |    价格在锚点下方每跌1档(2%) → 库存+0.083
  |    回到锚点上方归零; 锚点漂移>3档重置
  |    库存 ∈ [0, 0.25] (3档×0.083)
  ↓ 总仓位 = clip(核心+库存, 0, 1.5)           ← 总杠杆硬封顶150%
  ↓ 日平滑: 每日调仓限速cap=0.20
  ↓ 应急: 单日跌幅<-5% → 立即压向0 (0.5次/年)
```

**设计语义**: 网格不是叠加仓位, 而是从杠杆预算中划出的独立仓位——总敞口永远≤1.5,
网格只使用核心信号用不完的闲置预算。

### 4.2 B3参数与业绩锚点 (2018-2026, O2O+5bps)

| 层 | 参数 |
|---|---|
| 链层 | L=1.5, first_cap=-5%, second_cap=-5%, hard=-20%, soft=-12%, vol_cap=0.20 |
| 网格 | core_cap=1.25, sp=2%, step=0.0833, tiers=3, budget=1.5 |
| 平滑 | cap=0.20, emergency=-5% |

| 配置 | CAGR | Sharpe | MaxDD | Calmar | 26YTD | 26MDD | 最大日调仓 |
|---|---|---|---|---|---|---|---|
| 生产现状 (链-2/-3/-15) | +15.42% | 1.52 | -13.97% | 1.10 | +9.93% | -4.05% | 0.88 |
| **B3全栈** | **+18.47%** | **1.56** | -16.84% | 1.10 | **+11.94%** | **-4.71%** | **0.57** |

- MaxDD谷底位置不变 (2022-07-15), 深度-16.84%在-20%回撤预算内
- 2026-03崩盘段 +0.4% (正收益穿越); 逐年9年中7年≥现状
- 最大单日调仓 0.88→0.57 (-35%), 执行风险显著下降

### 4.3 应急通道参数回测 (3轮压力测试定案 -5%)

| 应急配置 | CAGR | Sharpe | MaxDD | Calmar | 触发频率 |
|---|---|---|---|---|---|
| 关闭 | +18.24% | 1.54 | -16.84% | 1.08 | — |
| -3% 压0 | +18.68% | 1.59 | -15.82% | 1.18 | 2次/年 |
| **-5% 压0 (定案)** | +18.47% | 1.56 | -16.84% | 1.10 | 0.5次/年 |
| -6%/-7% 压0 | +18.50% | 1.57 | -16.84% | 1.10 | 0.4次/年 |

**关键结论**:
1. -5%~-7%结果完全相同 → **稳健平台**; -3%~-4%是路径敏感区 (-4% MaxDD反而恶化至-17.10%)
2. -3%子样本测试虽全面占优, 但趋势年付保险费 (2024 -0.84% / 2025 -2.17%) —
   **实盘即"震荡洗出"成本, 每年2次反复清仓伤害执行纪律**, 故弃用
3. -5%触发全落在2025-2026极端日 (含2026-02-02单日-13.2%), 平时零干预
4. 压0 优于"收敛到目标" (26MDD -4.71% vs -5.07%)

### 4.4 鲁棒性与过拟合声明

| 检验 | 结果 |
|---|---|
| 网格sp敏感性 | 1.5%~2.0%平台全组合赢基准; 2.5%退化 |
| 逐年 | 7/9年改善, 仅2020/2021略弱, 无灾难年 |
| 周T对比 | 日频做T全负 (自相关-0.006); 周频二值T微正但噪音级; **网格式+升势停卖**是唯一稳定正贡献形态 |

⚠️ **诚实声明**: B3参数为样本内优化 (9年单资产), +3.05pp增益存在过拟合风险。
已按决策**暂缓生产切换**, 纸面跟踪3个月验证后再定。生产当前仍为链-2/-3/-15。

### 4.5 模块使用

```bash
python3 scripts/cmf/gold_backtest.py                    # 生产现状链报告 (锚点15.42%/1.52/-13.97%)
python3 scripts/cmf/gold_backtest.py --stack            # B3全栈报告 (锚点18.47%/1.56/-16.84%)
python3 scripts/cmf/gold_backtest.py --stack --compare  # 现状 vs 自定义栈 vs B3
python3 scripts/cmf/gold_backtest.py --fc -0.05 --sc -0.05 --hard -0.20   # 自定义链
python3 scripts/cmf/gold_track.py                       # 每日跟踪: 当日仓位/状态/动作
python3 scripts/cmf/gold_track.py --year 2026           # 全年持仓CSV
python3 scripts/cmf/gold_attribution.py                 # 归因: 链路+七级消融+子信号+调仓日志
python3 scripts/cmf/gold_attribution.py --year 2026     # + trades_2026.csv
```

核心函数: `build_chain` (链) / `weekly_grid_inventory` (周网格) /
`smooth_execution` (日平滑+应急) / `build_full_stack` (全栈组装)。
注意: 数据窗口为2018-06-20起 (`futures_main_close_au_adj`), 所有对比须同窗口。

### 4.6 完整链路 (数据 → 仓位 → 收益)

```
数据层   DuckDB: futures_main_close_au_adj + vix/gvz(shift2) + tips10y
信号层   4路加权: macro 0.2667 | real_rate 0.2667 | momentum 0.2667 | breakout 0.2
一层风控 内置: daily-5% / hard-10% / soft-7% / vol 15%
杠杆层   ×1.5 → clip 1.8
二层风控 hard-20% / soft-12% / daily-5% / vol 20% / cooldown 5d
死带     0.05
核心裁剪 ≤1.25
网格库存 +0~0.25 (sp=2%, 3档, 周频)
总封顶   1.5
日平滑   cap=0.20
应急     单日<-5% 压0
执行     O2O +5bps
```

唯一权威实现: `QuantNodes/strategy/commodity_fundamental/gold/gold_production.py`
(`PRODUCTION_PARAMS` + `build_production_position()`), 回测/跟踪/归因均为其委托或消费方。

### 4.7 归因分析 (2018-2026, 七级消融)

| 级 | 变体 | CAGR | Sharpe | MaxDD | 边际ΔCAGR |
|---|---|---|---|---|---|
| L0 | 裸信号 | +14.29% | 1.27 | -16.51% | — |
| L1 | +一层风控 | +13.15% | **1.51** | **-11.0%** | -1.1pp |
| L2 | +杠杆×1.5 | +20.13% | 1.53 | -16.2% | **+6.9pp** |
| L3 | +二层风控(链) | +17.73% | 1.52 | -17.3% | -2.4pp |
| L4 | +核心裁剪+网格 | +18.30% | 1.55 | -17.2% | +0.6pp |
| L5 | +日平滑 | +18.25% | 1.54 | -16.8% | -0.0pp |
| L6 | +应急(生产B3) | **+18.47%** | **1.56** | -16.8% | +0.2pp |

**归因结论** (B3 vs L0):
- **杠杆层是最大收益来源** (+6.9pp), 一层风控是最大风控贡献 (Sharpe+0.24, MaxDD改善5.5pp)
- 二层风控边际CAGR -2.4pp: 在一层已控回撤的基础上主要是"保险费", 但换 -5/-5/-20 参数后代价可控
- 网格+平滑+应急合计 +0.8pp 且 Sharpe +0.05: 执行层为正贡献

**子信号** (standalone, 生产价格口径): macro_regime 最优 (CAGR 16.6%/Sharpe 1.51/MaxDD -9.5%);
real_rate 最弱 (9.8%); breakout 2026年贡献最佳 (+13.3%)。

**2026调仓31次**: 网格12 / 链层12 / 平滑追赶5 / 应急2 (另有年初结转1次, tracker口径32)。

---

## 🆕 v3 更新 (2026-08-31)

### 3.1 数据修复与刷新

| 问题 | 修复 | 结果 |
|---|---|---|
| VIX/GVZ被管道覆盖截断 (仅169行) | 修复上游SQLite采集 + 重跑1e迁移 | 恢复至9261行 (1990-2026) |
| etf_tonnes停更 (iFind token过期) | akshare macro_cons_gold替代 (corr=1.000验证) | 2861行, 2004起, 至08-28 |
| gprd停更 | 官方渠道 --download-gprd | 至07-31 |
| dxy停更 | akshare_boc源延长 (corr=0.993) | 至08-25 |

**数据刷新脚本**: `scripts/cmf/refresh_gold_data.py` (幂等, 可重复执行)

### 3.2 渐进式换月 (新增数据管道)

- `fetch_gold_contracts.py` → `detect_gold_main.py` → `build_gold_smooth.py`
- 4天线性插值换月, 消除AU0换月跳空 (3-8% → 平滑过渡)
- 详见 `docs/gold/gold_rollover_plan.md`

### 3.3 代码重构 (5 Phase完成)

- `gold_utils.py`: 公共函数模块 (消除6份重复代码)
- `gold_ensemble.py`: 集成策略拆分 (gold_trend_bnh从1047行减负)
- **96个单元测试全部通过, 0警告** (tests/test_gold_*.py)

### 3.4 有效性评估 (统计检验通过 ✅)

| 检验 | 结果 | 判定 |
|---|---|---|
| Bootstrap Sharpe (1000次) | 95% CI [0.433, 1.366], P(<0)=**0%** | ✅ 显著为正 |
| Monte Carlo 择时 (200次随机) | Sharpe 0.886 超过 **100%** 随机排列 | ✅ 择时真实 |
| 参数敏感性 | hard_cap/vol_cap/bw/deadband全扰动Sharpe 0.82~0.91 | ✅ 无悬崖 |
| 成本敏感性 | 50bps下Sharpe仍+0.34 | ✅ 稳健 |
| 滚动5年Sharpe | 仅0.2%窗口为负 | ✅ 长期稳定 |
| 前视偏差 | shift2 + O2O对齐, 无已知前视 | ✅ 干净 |

### 3.5 胜率结构 (口径澄清)

| 周期 | 全样本胜率 | 有效持仓胜率 |
|---|---|---|
| 日 | 36.9% | **52.7%** |
| 周 | 38.0% | **51.1%** |
| 年 | 63.2% (12/19) | — |

- 全样本被空仓期稀释 (空仓日25.9%, 风控主动离场, 收益=0)
- 盈亏比 1.52 (盈利周+0.87% / 亏损周-0.57%), 期望+0.119%/周
- **盈利公式** = 52.7%胜率 × 1.52盈亏比 × 66%持仓时间 + 34%空仓避险

### 3.6 风控层贡献分解

| | CAGR | Sharpe | MaxDD | 均仓 |
|---|---|---|---|---|
| 无风控 | 7.45% | 0.675 | -34.55% | 66.6% |
| **有风控** | 6.35% | **0.886** | **-17.33%** | 38.6% |

> 用1.1pp年化换Sharpe+31%和回撤减半 — 策略本质是**金价beta+回撤管理**的配置型工具, 不是独立alpha来源。

### 3.7 逐年表现 vs B&H (O2O+5bps)

| 年份 | 策略 | 策略MaxDD | B&H | B&H MaxDD | 均仓 |
|---|---|---|---|---|---|
| 2010 | +21.17% | -3.88% | +26.62% | -8.31% | 46% |
| 2011 | +18.98% | -3.65% | +3.94% | -20.05% | 52% |
| 2012 | -4.90% | -6.71% | +4.13% | -13.93% | 21% |
| 2013 | +0.10% | -0.27% | -30.24% | -30.90% | 1% |
| 2014 | -4.27% | -7.13% | -1.22% | -16.88% | 37% |
| 2015 | -2.02% | -2.01% | -6.83% | -16.80% | 5% |
| 2016 | +4.65% | -7.57% | +19.67% | -11.26% | 59% |
| 2017 | -1.67% | -7.23% | +3.76% | -6.57% | 67% |
| 2018 | +2.28% | -3.29% | +3.15% | -6.61% | 55% |
| 2019 | +12.66% | -6.39% | +21.46% | -9.51% | 73% |
| 2020 | +5.05% | -5.83% | +14.70% | -17.43% | 17% |
| 2021 | -3.82% | -4.10% | -6.59% | -11.49% | 20% |
| 2022 | +5.09% | -4.27% | +10.99% | -11.23% | 39% |
| 2023 | +12.24% | -3.35% | +17.99% | -4.58% | 78% |
| 2024 | +19.09% | -4.04% | +29.62% | -7.13% | 49% |
| 2025 | +39.05% | -4.72% | +59.56% | -11.02% | 60% |
| **2026** | **+11.26%** | **-2.77%** | **+0.69%** | **-30.37%** | **6%** |
| **全周期** | **+6.35%** | **-17.33%** | **+8.67%** | **-44.81%** | 39% |

- 2026年OOS: 策略+11.26% vs B&H+0.69%, MaxDD -2.77% vs -30.37% (风控躲过3月崩盘)
- 熊市保护显著 (2013/2021/2026), 牛市跟涨略弱 (2024/2025)

### 3.8 最新信号 (2026-08-28)

| 信号 | 仓位 |
|---|---|
| **Layered Ensemble (生产)** | **23.62%** |
| 宏观机制 | 34.90% |
| 动量 / 突破 / 美元趋势 | 100% |
| 实际利率 / 避险情绪 / 仓位流 | 0% |

### 3.9 ⚠️ 全因子信号过拟合警告

`gold_all_factor` (5层11指标) Walk-Forward验证:
- IS(2018-2025) Sharpe=4.13 → OOS(2026) Sharpe=**-0.32** (衰减108%)
- 2025年Sharpe=19.38为极端过拟合信号
- 已加shift(1)修正前视, 但IS/OOS表现不变 — **过拟合根因是参数过多, 非前视**
- **结论: 全因子仅作辅助参考, 不可作为主信号; 生产信号(Layered)无此问题**

### 3.10 生产信号数据依赖 (仅5字段)

| 表 | 字段 | 状态 |
|---|---|---|
| futures_main_close_au | open, close | ✅ 至08-28 |
| fundamentals_daily (gold_fred) | vix | ✅ 至08-27 |
| fundamentals_daily (gold_fred) | gvz | ✅ 至08-27 |
| fundamentals_daily (fred_dfii10) | tips10y | ✅ 至08-21 |
| fundamentals_daily (gold_fred) | rir (辅助) | ✅ 至08-01 |

> etf/gprd/dxy/cftc等数据**不在生产信号依赖链上**, 更新它们不影响生产仓位。

---

## ⚠️ v2 重大修正 (2026-08-28)

### 发现的双重前视偏差 (用户质疑触发)

**偏差1: O2O收益对齐错误**
- 原计算: `sr[t] = pos[t] × (open[t+1]-open[t])/open[t]`
- pos[t]是t日**收盘**信号, 却乘t日**开盘**起算的收益 → 信号提前6小时使用
- 修复: `sr[t] = pos[t-1] × (open[t+1]-open[t])/open[t]`

**偏差2: macro/rr层内部自加载未shift数据**
- compute_macro_regime_signal / compute_real_rate_signal 内部自加载原始VIX/GVZ/TIPS
- 修复: 两函数新增数据参数, 统一入口传入shift后数据

### 修正前后对比 (shift2基线, WF CAGR)

| 版本 | WF CAGR | 说明 |
|---|---|---|
| 修正前 (双重前视) | 9.92% | 虚高 |
| 修正后 (无前视) | 5.89% | 真实 |

### 真实业绩 (无前视O2O + shift2 + 5bps)

| 策略 | 年化 | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| L=1.5 CPPI | **9.98%** | 0.93 | **-28.53%** | 0.35 |
| 基线 L=1 | 7.34% | 0.92 | -21.45% | 0.34 |
| B&H | 8.99% | 0.57 | -44.52% | 0.20 |

### 诚实结论

1. 策略CAGR与B&H相当(+1pp), MaxDD改善16pp — 风险调整仍优, 但**收益不再大幅跑赢**
2. **CPPI的-15%硬约束失守** (实际-28.53%): rolling DD计算滞后 + O2O隔夜跳空穿越
3. 三档shift真实排序: shift0(7.67%) > shift2(6.62%) > shift1(6.08%) — shift2非最差, 保守化代价可接受
4. 此前所有"策略大幅跑赢B&H"的阶段性结论均不可信, 以本v2数字为准

---

## ⭐ 数据延迟修复说明 (2026-08-28)

**发现的问题**: compute_macro_regime_signal / compute_real_rate_signal 内部自加载未shift数据,
导致此前回测的shift1/shift2仅覆盖delta_bin和CPPI层, macro_ov(21%权重)/rr_ov(21%权重)存在前视偏差。

**修复方案**: 两函数新增数据参数(统一入口), 调用方传入shift后数据; 未传参时保持旧行为(向后兼容)。

**修复后真实三档对比** (Open-to-Open, 5bps):

| 方案 | OOS年化 | OOS DD | IS年化 | IS DD | WF CAGR | WF MaxDD | Calmar |
|---|---|---|---|---|---|---|---|
| shift0 | 8.14% | -10.64% | 20.94% | -6.57% | 12.17% | -8.21% | 1.48 |
| shift1 | 6.86% | -12.54% | 19.95% | -6.19% | 11.10% | -10.38% | 1.07 |
| **shift2 (生产)** | **5.41%** | **-13.42%** | **18.36%** | **-7.12%** | **9.61%** | **-11.07%** | **0.87** |

**2026 YTD逐月收益 (shift2, 非年化)**:

| 月份 | 月收益 | MaxDD |
|---|---|---|
| 1月 | +11.02% | -0.54% |
| 2月 | +0.39% | -0.59% |
| 3月 | -1.61% | -0.74% |
| 4月 | -0.31% | -0.82% |
| 5月 | +0.05% | -0.05% |
| 6月 | -0.61% | -0.66% |
| 7月 | -0.33% | -0.33% |
| 8月 | +1.15% | -0.28% |
| **YTD** | **+9.6%** | **-1.46%** |
> 换手优化: 死带th=0.05, 变化次数-47%, 换手-5.5%, 收益损失<0.05pp

---

## 一、策略概述

### 1.1 目标

用户原始需求: **"大级别止损，小级别波动容忍"**

核心目标: 在-15% MaxDD约束下最大化风险调整收益。

### 1.2 最终策略架构

```
四路分层集成 + Δtips信号 + CPPI杠杆 + 三级风控 + 数据延迟处理

信号层 (收盘计算):
  macro_ov  (0.2125)  宏观机制策略 + 独立风控(hc=-0.10)
  rr_ov     (0.2125)  TIPS实际利率择时 + 独立风控
  mom_ov    (0.2125)  52周动量 + 独立风控
  bo_ov     (0.2125)  20日突破 + 独立风控
  delta_bin (0.15)    Δtips10y二值信号(window=10, rank≤0.5)

数据延迟处理:
  VIX/GVZ: shift(1) (美股收盘+美中时差, T日信号用T-1数据)
  TIPS10y: shift(1) (FRED延迟1个交易日)
  金价: 当天收盘可用

集成: base = 0.2125×(macro+rr+mom+bo) + 0.15×Δtips (shift后)

杠杆: L=1.5, clip上限1.8

死带平滑: th=0.05 (忽略<5%的仓位变化, 降低换手47%)

回测方式: Open-to-Open (收盘算信号 → 次日开盘成交 → 捕捉隔夜跳空)

风控: CPPI三层重铺
  Level 1: 波动率压制 (vol_cap=0.20)
  Level 2: 软减仓 (soft_cap=-0.12)
  Level 3: 硬止损 + 智能再入场 (hard_cap=-0.15, daily_cap=-0.03, cooldown=5)
```

---

## 二、核心性能指标

### 2.1 Walk-Forward验证 (2012-2026, Open-to-Open, 含5bps成本, 真实shift2)

| 指标 | L=1.5 CPPI (shift2) | Buy&Hold |
|---|---|---|
| **CAGR** | **9.61%** | 9.06% |
| **Sharpe** | **1.34** | 0.55 |
| **MaxDD** | **-11.07%** | -44.52% |
| **Calmar** | **0.87** | 0.20 |

### 2.2 双窗口指标 (真实shift2)

| 指标 | OOS 2010-2019 | IS 2020-2026 |
|---|---|---|
| 年化收益 | 5.41% | **18.36%** |
| Sharpe | ~0.5 | **~2.6** |
| MaxDD | -13.42% | **-7.12%** |
| Calmar | 0.40 | **2.58** |

### 2.3 2026 YTD逐月收益 (Open-to-Open, 非年化)

| 月份 | 月收益 | MaxDD | 均仓 |
|---|---|---|---|
| 1月 | +11.02% | -0.54% | — |
| 2月 | +0.39% | -0.59% | — |
| 3月 | -1.61% | -0.74% | — |
| 4月 | -0.31% | -0.82% | — |
| 5月 | +0.05% | -0.05% | — |
| 6月 | -0.61% | -0.66% | — |
| 7月 | -0.33% | -0.33% | — |
| 8月 | +1.15% | -0.28% | — |
| **YTD** | **+9.6%** | **-1.46%** | — |

---

## 三、各层详解

### 3.1 Δtips二值信号 (核心alpha来源)

**设计**: tips10y.diff(10) → 504日滚动百分位rank → rank≤0.5=1(看多), >0.5=0(看空)

**逻辑**: 当实际利率下降中(rank≤0.5) → 看多黄金; 当实际利率上升中 → 看空

**IC**: 0.126 (20日前瞻收益)

**权重**: 0.15 (15%仓位贡献)

### 3.2 四路基础信号

| 信号 | 权重 | 数据源 | 信号类型 |
|---|---|---|---|
| macro_regime | 0.2125 | 金价+VIX+GVZ+TIPS | 趋势×宏观 |
| real_rate_tips_ma | 0.2125 | TIPS实际利率 | 均值回归 |
| momentum_long_only | 0.2125 | 金价52周动量 | 动量 |
| breakout | 0.2125 | 金价20日高点突破 | 突破 |

每路独立过 `compute_risk_overlay(hc=-0.10)` 后加权平均。

### 3.3 CPPI三层风控

| 层级 | 触发条件 | 动作 |
|---|---|---|
| Level 1 波动率压制 | realized_vol > 0.20 | 仓位×(0.20/realized_vol) |
| Level 2 软减仓 | rolling_dd < -0.12 | 仓位×0.5 |
| Level 3 硬止损 | rolling_dd < -0.15 OR daily_ret < -0.03 | 清仓+冷却5天 |
| 再入场 | VIX<60%百分位 AND GVZ_z<1.0 | 恢复原仓位 |

### 3.4 智能再入场机制

硬止损触发后:
1. 冷却期5天强制空仓
2. 冷却期结束检查: `vix_pct < 0.6 AND gvz_z < 1.0`
3. 条件不满足 → 继续空仓(延长冷却)
4. 条件满足 → 恢复信号层仓位

---

## 四、换手率与成本

| 策略 | 年均换手 | 5bps成本 | 10bps成本 |
|---|---|---|---|
| L=1.5 CPPI (死带0.05) | **10.1x** | **0.51%** | **1.01%** |
| L=1.5 CPPI (原始) | 14.4x | 0.72% | 1.44% |
| 基线(L=1) | 12.4x | 0.62% | 1.24% |

**成本敏感度**: 5bps下策略仍合规; 15bps仍合规; 20bps越线。

**死带平滑效果** (th=0.05):
- 变化次数: 2240 → 908 (-59.5%)
- MaxDD: -13.32% → -13.08% (+0.24pp改善)
- 年化损失: -0.06pp (几乎无损)
- 净效果: 降低换手成本0.21pp, 净正收益

---

## 五、研究历程

```
期权保护 (Buy Put / Put Spread)
    ↓ 失败: VIX<25触发线,保护未启动
VIX自适应仓位
    ↓ 失败: VIX对黄金暴跌盲区
ATR自适应止损
    ↓ 失效: 止损触发本就稀少,宏观层已主导
多信号集成 + 风控覆盖层
    ↓ 成功: 分层集成翻案
Δtips二值信号
    ↓ 突破: Walk-Forward CAGR+0.81pp, DD+3.43pp
CPPI杠杆
    ↓ 最终: CAGR 9.59%, DD -11.48%
```

### 关键里程碑

1. **分层集成**: 先各自风控→再平均, 解决中间仓位问题
2. **Δtips突破**: 利率变化率(window=10)二值信号, Walk-Forward 10/15折跑赢基线
3. **CPPI杠杆**: L=1.5 + hard_cap=-0.15, 年化从6.99%提升到9.59%

---

## 六、风险与诚实声明

### 6.1 已知局限

1. **2014-2017熊市表现平平**: 年化1-2%, 策略在震荡市缺乏alpha
2. **CPPI触发频率**: 2026年1月30日触发清仓后锁仓至8月, 错过部分反弹
3. **换手率偏高**: 14.4x年均, 成本敏感
4. **Δtips窗口敏感**: window=10最优, 但window=21/42效果递减

### 6.2 与B&H的差距

- Walk-Forward CAGR: 策略9.59% vs B&H 7.92% (**策略领先1.67pp**)
- 但2025年: 策略+48.19% vs B&H +56.63% (牛市跟涨不足)
- 2023年: 策略+19.98% vs B&H +17.14% (策略领先)

**本质**: 风险控制策略在熊市保护远优于B&H(DD -11.48% vs -41.09%), 但牛市跟涨幅度受限。

### 6.3 参数鲁棒性

- Walk-Forward 15折验证, 10/15折跑赢基线
- OOS 2010-2019与IS 2020-2026方向一致
- 子周期(10-14/15-19/20-23/24-26)均有正收益

---

## 七、代码结构

```
QuantNodes/strategy/commodity_fundamental/gold/
├── gold_utils.py                           # 公共函数 (数据加载/回测/周收益)
├── gold_trend_bnh.py
│   ├── compute_macro_regime_signal()       # 宏观机制策略
│   ├── compute_hysteresis_trend_d()        # 滞回带趋势
│   ├── compute_risk_overlay()              # 三级风控覆盖层
│   ├── compute_trend_bnh_signal()          # 基础趋势B&H
│   ├── _zscore_scale()                     # GVZ zscore缩放
│   ├── _rate_gate()                        # Δtips利率门控
│   └── _rolling_drawdown()                 # 滚动回撤计算
├── gold_ensemble.py
│   ├── compute_layered_ensemble_signal()   # 分层集成 (生产入口)
│   └── compute_adaptive_ensemble()         # 自适应集成(备选)
├── gold_real_rate.py
├── gold_momentum.py
├── gold_breakout.py
└── __init__.py

tests/
├── test_gold_rollover.py    # 换月 (16用例)
├── test_gold_utils.py       # 公共函数 (11用例)
├── test_gold_signals.py     # 6单因子 (40用例)
├── test_gold_trend_bnh.py   # 趋势+风控 (18用例)
└── test_gold_ensemble.py    # 集成 (7用例)
# 合计96个用例全部通过

scripts/cmf/ (数据管道)
├── 1e_fetch_gold.py          # SQLite→DuckDB迁移
├── refresh_gold_data.py      # etf/gprd/dxy幂等刷新
├── fetch_gold_contracts.py   # 全合约数据 (换月用)
├── detect_gold_main.py       # 主力识别+换月检测
├── build_gold_smooth.py      # 渐进式换月价格
├── prepare_gold_overview_data.py  # 报告数据缓存
└── gen_html_report.py        # HTML报告生成
```

---

## 八、使用方法

```python
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal,
    compute_risk_overlay,
)

# 基础分层集成 (bw=0.2默认)
sig = compute_layered_ensemble_signal()
close, position = sig["close"], sig["position"]

# CPPI杠杆版 (L=1.5)
pos_base = compute_layered_ensemble_signal(breakout_weight=0.2)["position"]
pos_15 = compute_risk_overlay(
    close=close,
    position=(pos_base * 1.5).clip(0, 1.8),
    vix=vix, gvz=gvz,
    hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5,
)
```

---

## 九、文档索引

| 文件 | 内容 |
|---|---|
| `docs/gold/gold_strategy_research_summary.md` | 完整研究总结 |
| `docs/gold/gold_atr_stop_research.md` | ATR止损调研 |
| `docs/gold/risk_overlay_design.md` | 风控覆盖层设计 |
| `docs/gold/substrategies_evaluation.md` | 子策略诊断 |
| `docs/gold/pareto_frontier_results.md` | Pareto前沿结果 |
| `docs/gold/final_evaluation_results.md` | 候选策略评估 |
| `docs/gold/adaptive_ensemble_results.md` | 自适应集成评估 |
| `docs/gold/ensemble_v2_results.md` | 分层集成评估 |
| `docs/gold/walk_forward_ensemble_results.md` | Walk-Forward验证 |
| `docs/gold/gold_rollover_plan.md` | 渐进式换月方案 |
| `docs/gold/gold_code_refactor_plan.md` | 代码整理计划 (已完成) |
| `reports/gold/report.html` | HTML业绩报告 (SVG自包含) |
| `reports/gold/gold_nav_curve.png` | 净值曲线 |
| `reports/gold/gold_nav_curve.csv` | 净值数据 |

---

## 十、Git提交历史 (近期)

```
bfc84fbd fix(gold): 全因子信号加shift=1周修正前视效应
b1c96d78 feat(gold): 刷新缺失数据 — etf/gprd/dxy 全部更新至最新
798be316 feat(gold): 更新CTA策略数据 + 生成最新信号
67727551 chore(gold): 重新生成业绩报告 — 验证整理后代码正确性
4cfdbe3d docs(gold): 代码整理计划状态更新 — 全部完成
55b72deb refactor(gold): Phase 5 — 修复fillna弃用警告
2a282db5 refactor(gold): Phase 4 — 合并重复测试脚本
c90b56dc refactor(gold): Phase 3 — 添加单元测试 (96个用例全部通过)
67529369 refactor(gold): Phase 2 — 拆分集成策略到 gold_ensemble.py
9d5dcff5 refactor(gold): Phase 1 — 提取公共模块 gold_utils.py
3a7ec09d docs(gold): 代码整理计划文档
0cc6a00f refactor(gold): 重构build_gold_smooth.py + 添加单元测试
941513c6 feat(gold): 使用渐进式换月价格重新回测
670ce606 feat(gold): 渐进式换月数据管道 — Step 1-3完成
b759ce01 docs(gold): 渐进式换月方案文档
```
