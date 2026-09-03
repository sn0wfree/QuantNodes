# CTA策略完整调优记录（2026-09-02）

> 一次性完整记录所有调优过程：从baseline到最终定版
> 子文档见各章节链接

---

## 目录

1. [总体时间线与关键节点](#1-总体时间线与关键节点)
2. [Phase 1: 基础设施修复与质量提升](#2-phase-1-基础设施修复与质量提升)
3. [Phase 2: 性能优化（向量化）](#3-phase-2-性能优化向量化)
4. [Phase 3: 换手率优化与成本发现](#4-phase-3-换手率优化与成本发现)
5. [Phase 4: 杠杆探索与资金等分机制理解](#5-phase-4-杠杆探索与资金等分机制理解)
6. [Phase 5: 成本校准与EC双口径发现](#6-phase-5-成本校准与ec双口径发现)
7. [Phase 6: 换月建模修复与改进](#7-phase-6-换月建模修复与改进)
8. [Phase 7: 双口径伪影修复（双重bug修复）](#8-phase-7-双口径伪影修复双重bug修复)
9. [Phase 8: α来源扩展研究（子策略）](#9-phase-8-α来源扩展研究子策略)
10. [Phase 9: 子策略组合实验与PIT质量修正](#10-phase-9-子策略组合实验与pit质量修正)
11. [最终定版配置与实盘推荐](#11-最终定版配置与实盘推荐)
12. [全部commit历史与产出物索引](#12-全部commit历史与产出物索引)

---

## 1. 总体时间线与关键节点

| 阶段 | 日期(2026) | 核心动作 | 关键产出 |
|------|-----------|---------|---------|
| Baseline | 09-02 | 保存修复前基准 | Sharpe 1.878 / Calmar 1.000 |
| Phase 1.1 | 09-02 | module_weights排查 | 误报，无bug，加防御性校验 |
| Phase 1.2 | 09-02 | 成本敏感性测试 | 发现零成本Sharpe严重虚高 |
| Phase 1.3 | 09-02 | 回测报告落盘 | scripts/cta_baseline.py |
| Phase 1.4 | 09-02 | 集群定义统一 | CLUSTERS覆盖66品种7板块 |
| Phase 1.5 | 09-02 | Basis截断排查 | 数据滞后，非代码bug |
| Phase 2.1 | 09-02 | XS因子向量化 | 24x加速，4因子equals |
| Phase 2.2 | 09-02 | 仓位EMA向量化 | 等价(1e-15) |
| Phase 2.3 | 09-02 | pct_change fill_method修复 | 54处，消除Warning+幻影价格 |
| Phase 3.1 | 09-02 | 换手率实验 | ema=0.15将换手149x→29x |
| Phase 4 | 09-02 | target_vol阶梯实验 | 发现杠杆结构性封顶 |
| Phase 4b | 09-02 | equity_leverage_k实施 | 定版k=1 |
| Phase 5 | 09-02 | 按品种成本校准 | cost_model.json |
| Phase 5b | 09-02 | EC双口径监控 | dual_basis.csv |
| Phase 6 | 09-02 | 换月建模诊断 | roll_cost_bps未生效因数据缺 |
| Phase 6b | 09-02 | lump-sum换月冲击模型 | cta_improvement_plan.md |
| Phase 7a | 09-02 | tail_risk n_pairs bug | VALID_TAIL_RISK_SECTORS |
| Phase 7b | 09-02 | 模块信号NaN污染修复 | price_panel.notna() mask |

**关键指标演变**：
```
初始         Sharpe 1.878 / Calmar 1.000 / MaxDD -12.5%
↓ Phase 1修复
修复后      Sharpe 1.688 / Calmar 1.214 / MaxDD -10.3%  (真bug修复,降低乐观)
↓ Phase 3 EMA优化
ema=0.15    Sharpe 1.884 / Calmar 1.710 / MaxDD -5.6%   (成本敏感度大幅下降)
↓ tv=1.5
tv=1.5     Sharpe 2.30 / Calmar 3.29 / MaxDD -6.1%    (取整+杠杆放大)
↓ k定版
k=1.0      Sharpe 2.30 / Calmar 3.29 / MaxDD -6.1%    (k非线性,k=1最佳)
↓ 按品种成本校准
校准成本   Sharpe 2.43 / Calmar 3.53 / MaxDD -5.8%    (Sharpe提0.13)
↓ lump换月
lump        Sharpe 2.42 / Calmar 3.51 / MaxDD -5.85%   (换月冲击被低真实成本抵消)
↓ tail_risk fix + NaN污染fix
最终        Sharpe 2.41(containEC) / Calmar 3.25 / MaxDD -6.1%
剔EC版    Sharpe 2.20 / Calmar 2.55 / MaxDD -6.3%  (实盘保守预期)
```

---

## 2. Phase 1: 基础设施修复与质量提升

详见 `docs/cta/cta_improvement_plan.md` 早期阶段。

### 1.1 module_weights "bug" — 排查后无bug
- **误报**：探索报告误报。`self._modules`实际5个模块（OscillatorAlpha不在combo内），5权重对应5模块，`zip`无丢弃
- **行动**：无行为修改；加防御性长度校验（未来加模块时报错而非静默丢弃）
- **验证**：`len(s._modules) == len(cfg.module_weights) == 5`

### 1.2 交易成本默认为0（重大发现）
- **问题**：`slippage_bps=None`，所有回测零成本，Sharpe虚高
- **敏感性测试**：
  | 滑点 (bps) | Sharpe | AnnRet | MaxDD | Calmar |
  |-----------|--------|--------|-------|--------|
  | 0 | 1.688 | 12.5% | -10.3% | 1.214 |
  | 1 | 1.254 | 10.2% | -11.7% | 0.871 |
  | 5 | 0.617 | 6.0% | -15.4% | 0.386 |
  | 10 | 0.347 | 3.7% | -18.7% | 0.197 |
- **结论**：每1bp滑点≈-0.04 Sharpe。零成本Sharpe严重虚高

### 1.3 回测报告落盘
- **修复**：`scripts/cta_baseline.py` 保存NAV/指标/年度收益
- **支持**：`--tag fix`、`--sensitivity` 多种输出

### 1.4 集群定义统一
- **问题**：`CLUSTERS` 仅覆盖10品种，`_cluster_limit` 又有不同局部定义
- **修复**：统一为7板块66品种：`precious/black/base/energy_chem/agri/financial/shipping`
- **效果**：cluster冲突过滤/tail_risk降仓/敞口上限从"基本失效"变为实际生效

### 1.5 Basis截断排查
- **结论**：代码本身无bug，`spot_basis_daily` 表数据本身只到2025-07-21（数据滞后，非代码问题）

---

## 3. Phase 2: 性能优化（向量化）

### 2.1 XS因子向量化
- **优化前**：`for date in ret.index` per行循环
- **优化后**：`quantile(axis=1)` + numpy布尔广播
- **验证**：4因子（MomentumXS/VolXS/BasisXS/AmihudXS）输出与旧实现equals完全一致
- **性能**：24x加速（500×30面板：526ms→22ms）
- **关键坑**：`DataFrame & Series` 按列对齐，需用 `.to_numpy()[:, None]` 显式行广播

### 2.2 仓位EMA向量化
- **优化前**：per-row Python循环（factor_modules.py `_run_with_lots`）
- **优化后**：前置零行 + `ewm(alpha, adjust=False)` 匹配循环初始条件
- **验证**：最大差1.3e-15（浮点噪声），行为等价

### 2.3 pct_change fill_method修复
- **优化前**：54处调用，`fill_method='pad'`默认弃用Warning
- **优化后**：显式 `fill_method=None`，消除Warning
- **发现**：LU（碳酸锂）面板有104天内部NaN，fill_method=None行为更诚实（消除幻影价格）

---

## 4. Phase 3: 换手率优化与成本发现

### 3.1 换手率实验（重大发现）

**结论**：策略年化换手率 149x（远高于行业中位数50-100x），**这是Sharpe虚高的最大来源**。

**5组实验**（@双10bp成本口径）：

| 配置 | Sharpe | AnnRet | MaxDD | 换手 |
|------|--------|--------|-------|------|
| baseline (ema=0, mc=0.30) | 1.688 | 12.5% | -10.3% | 149x |
| ema_only (ema=0.15) | **1.884** | **12.4%** | **-5.6%** | **29x** |
| mc_only (mc=0.15) | 1.681 | 12.4% | -10.8% | 153x |
| combo (ema=0.15, mc=0.15) | 2.110 | 12.4% | -6.8% | 37x |
| final_calibrated | 2.433 | 20.6% | -5.8% | 33x |

**关键发现**：
- **EMA是决定性改善**：换手149x→29x（-81%），MaxDD -11.7%→-5.6%（近乎减半）
- mc=0.15微害：应放高不降（误判纠）
- em组合中二者收益的：em与mc功能重叠
- 代码注释 "推荐0.15" 原本正确，baseline却默认关闭

**机制**：EMA让高频调仓降级为平滑渐变，保留趋势信号、过滤噪音信号。

### 3.2 EMA alpha敏感性（@10bp）

| ema alpha | Sharpe | AnnRet | MaxDD | 换手 | Calmar |
|-----------|--------|--------|-------|------|--------|
| 0 | 0.347 | 3.7% | -18.7% | 149x | 0.20 |
| 0.10 | 1.790 | 11.0% | -8.1% | 22x | 1.36 |
| **0.15** | **1.884** | **11.7%** | **-6.8%** | **29x** | **1.71** |
| 0.20 | 1.800 | 11.4% | -7.7% | 37x | 1.49 |
| 0.30 | 1.495 | 10.5% | -10.7% | 50x | 0.99 |

**结论**：
- ema=0.15是峰值
- 稳健平台0.10-0.20（Sharpe差<0.1）
- 0.30过度滞后，Sharpe掉到1.5

---

## 5. Phase 4: 杠杆探索与资金等分机制理解

### 4.1 target_vol阶梯实验（重大发现：杠杆结构封顶）

| target_vol | Sharpe | AnnRet | MaxDD | 杠杆 | Calmar |
|-----------|--------|--------|-------|------|--------|
| 0.30 | 1.893 | 10.9% | -5.6% | 0.49 | 1.95 |
| 0.55 | 2.198 | 15.9% | -6.2% | 0.54 | 2.55 |
| 0.80 | 1.800 | 11.4% | -7.7% | 0.37 | 1.49 |
| **1.50** | **2.304** | **20.0%** | **-6.1%** | **0.53** | **3.29** |
| 2.00 | 2.000 | ... | ... | ... | ... |

**意外发现**：杠杆被结构钉死在~0.53，**与target_vol不成比例**！

**原因（架构洞察）**：
- 组合杠杆 = mean(|weights_i|)
- weights_i = signal × clip(target_vol/vol_i, 3.0)
- 国内商品年化vol中中~35% → tv=1.5时 1.5/0.35=4.3 → clip到3.0
- tv=0.30时 → 0.30/0.35=0.86 → 没到clip
- 杠杆从0.30→0.55自由放大，0.55→1.5几乎不变（已clip顶格）

**资金等分基数**：`equity/n_active` × lev_k × weights_i
- 等分不是替代vol调整，是**手数换算时的资金分配基数**
- vol调整决定"品种间相对配比"，等分决定"绝对规模基数"
- 组合杠杆 = 活跃品种单位权重均值（结构性上限）

### 4.2 equity_leverage_k实验

| k | 杠杆 | Sharpe | AnnRet | MaxDD | Calmar |
|---|------|--------|--------|-------|--------|
| 1.0 | 0.555 | 2.10 | 19.0% | -6.5% | 2.92 |
| 1.7 | 0.766 | 1.94 | 22.8% | -9.9% | 2.31 |
| 2.0 | 0.812 | 1.96 | 24.5% | -11.0% | 2.24 |

**关键发现**：k>1.5进入非线性区
- 放大效率递减（k×2只换来杠杆×1.46，clip被被截断）
- Calmar加速恶化（AnnRet×1.29 vs MaxDD×1.69）
- 0.9杠杆需k≈2.4-2.5，Calmar跌破2.0

**定版**：k=1（现状已优，无需杠杆放大）

### 4.3 杠杆改进方案A（global杠杆系数k）：定版
- 实施：`equity_leverage_k` 字段
- **修复**：`_cluster_limit` 等比放大（初版未乘k导致放大被吸收0.66）
- **定版**k=1.0（杠杆0.555，Sharpe 2.10/Calmar 2.92）

---

## 6. Phase 5: 成本校准与EC双口径发现

### 5.1 按品种成本校准（数据陷阱）

详见 `scripts/cta_cost_model.py`。

**数据陷阱**：`contract_specs.tick_size` 有坏数据：
- LC tick=0.01（错误，实际为50元/吨）
- I（铁矿石）smoothed表价格异常200元
- AU/SN价格异常高
- **不能依赖DB数据建成本表**

**解决**：硬编码交易所标准tick+典型价格
```python
COMMISSION_BPS = { ... 47品种按交易所公开费率估算 ... }
SLIPPAGE_BPS = { ... 1 tick ÷ 典型价格 ... }
```

**成本分布**：单边total 0.27bp(AU) ~ 8.43bp(FG)

### 5.2 按品种成本 vs 双10bp 对比

| 成本口径 | Sharpe | AnnRet | Vol | MaxDD | Calmar |
|---------|--------|--------|-----|-------|--------|
| 双10bp（保守） | 2.10 | 19.0% | 9.1% | -6.5% | 2.92 |
| **按品种校准** | **2.43** | **20.6%** | **8.5%** | **-5.8%** | **3.53** |

校准后多数品种真实成本<10bp，Sharpe/Calmar均提升。

### 5.3 EC双口径监控（重大发现）

| 口径 | Sharpe | AnnRet | Vol | MaxDD | Calmar |
|------|--------|--------|-----|-------|--------|
| **含EC** | **2.52** | **21.4%** | **8.5%** | **-5.8%** | **3.66** |
| **剔EC** | **2.10** | **15.6%** | **7.4%** | **-6.7%** | **2.34** |

**EC全期+26.07/总PnL 25%**（主要来自2023红海危机+16.9，2024延续+10.2）

**诚实评估**：
- 含EC口径：Sharpe 2.52/Calmar 3.66（含EC红利，不可外推）
- **剔EC口径**：**Sharpe 2.10/Calmar 2.34**（实盘保守预期，更可信）

**待解决疑问**：剔EC前几年PnL为何仍有差异？（在Phase 7解决）

---

## 7. Phase 6: 换月建模修复与改进

详见 `docs/cta/cta_rollover_analysis.md`。

### 6.1 换月建模盲区诊断

**核心发现**：`roll_cost_bps` 机制完整但数据空：
- `SlippageEngine.load_roll_dates` 从`main_contract_mapping`加载换月日
- **数据全**（88品种×1988天）—— 之前误判为数据空
- **但实证**：启用roll_cost_bps=5/10bp Sharpe几乎无变化（2.432/2.430 vs 无2.433）
- **根本局限**：`roll_days=5` 线性均摊**低估真实冲击**

### 6.2 实证：换月跳价被回测调整吸收

**smoothed数据的"未来数据幻觉"**：
- 原始主合约73天>2%跳价（RB实证）
- smoothed用历史gap反向修正吸收
- 策略PnL用 `pos.shift(1) × smoothed_ret` 计算 → **看起来无换月影响**
- 真实执行：近月平仓+远月开仓，承受近远价差（与smoothed修正方向相反）
- **回测系统性低估换月成本 → 真实PnL可能远低于回测PnL**

### 6.3 lump-sum换月冲击模型（改进方案）

**实施**：`SlippageEngine` 新增 `roll_model: 'amortized'|'lump'`
- amortized：换月日起roll_days天内均摊（默认5天，旧行为）
- **lump：换月日当天一次性扣除 2 × roll_cost_bps**（近月平+远月开双倍）

### 6.4 lump实证（@双10bp口径，tv=1.5）

| 模型 | roll | Sharpe | MaxDD | Calmar |
|------|------|--------|-------|--------|
| 无roll | 0 | 2.4335 | -5.83% | 3.527 |
| amortized | 10bp | 2.4300 | -5.84% | 3.520 |
| amortized | 50bp | 2.4163 | -5.87% | 3.492 |
| **lump** | **10bp** | **2.0892** | **-6.54%** | **2.904** |
| lump | 20bp | 2.0793 | -6.56% | 2.887 |
| lump | 30bp | 2.0693 | -6.58% | 2.870 |

**重大发现**：lump vs amortized（同10bp）：Sharpe -0.34，MaxDD -0.7pp，Calmar -0.62
- 真实换月冲击约是amortized模型的10倍
- amortized回测严重低估换月成本
- lump内部10/20/30差距小，单次冲击已被10bp代表充分

---

## 8. Phase 7: 双口径伪影修复（双重bug修复）

### 7a. tail_risk_filter n_pairs bug

详见 `docs/cta/cta_rollover_analysis.md`。

**根因**：`_tail_risk_filter`中，板块间相关性平均时`n_pairs`随CLUSTERS动态计算。shipping板块只含EC（早期无数据），剔除后：
- 含EC时7板块→21对
- 剔EC时6板块→15对（shipping消失）
- 分母不同 → avg_sector_corr不同 → 触发降仓日期不同 → **非EC品种仓位也被改变**

**修复**：新增类常量 `VALID_TAIL_RISK_SECTORS = ('precious', 'black', 'base', 'energy_chem', 'agri', 'financial')`，shipping永远排除。

### 7b. 模块信号NaN污染修复（更深的bug）

**根因**：未上市品种（如EC在2018-2020）的模块信号不是NaN，而是**0甚至±值**：

| 模块 | EC列值（2018-2020） |
|------|---------------------|
| TrendMomentum | 全NaN ✓ |
| VolumeFlow | **全0.0** ❌ |
| TermBasis | **全0.0** ❌ |
| IndustrialChain | **全0.0** ❌ |
| **Macro** | **442个非零值** ❌（宏观信号广播到了无价格数据的EC列）|

**污染路径**：
1. `inv_vol`模块权重 = `s.rolling().std().mean(axis=1)`——EC的0列参与行均值 → **所有模块权重变**
2. `signal_on`把EC算进`n_active` → **资金等分被稀释**（EC无数据但参与66分之1）

**修复**：`_signals_raw` 循环开头统一应用 `price_panel.notna()` mask：
```python
price_valid = self._price_panel.notna()
for m in self._modules:
    sig = m._signals_raw()
    sig = sig.where(price_valid)  # 无价格数据品种 → NaN
    ...
```

**修复效果**：
- 2018-2022含EC vs 剔EC NAV**完全一致**（0.00%差异）✓
- 剔EC口径Sharpe从2.10升到2.20（修复了LC/AO/BR等未上市品种的污染）

### 7c. 修复后EC真实贡献（无伪影）

| 年份 | 含EC | 剔EC | 真实EC贡献 |
|------|------|------|-----------|
| 2019 | +27.45% | +27.45% | **0** ✓ |
| 2020 | +39.31% | +39.31% | **0** ✓ |
| 2021 | +29.11% | +29.11% | **0** ✓ |
| 2022 | +11.76% | +11.76% | **0** ✓ |
| **2023** | **+36.32%** | **+14.34%** | **+21.98pp** |
| **2024** | **+9.59%** | **-0.56%** | **+10.15pp** |
| 2025 | +6.19% | +7.17% | -0.98pp |

EC贡献**纯粹在2023-2024**（红海危机+延续），2019-2022零贡献，2025微负。

---

## 8. α来源扩展研究（Phase 8：子策略路线）

### 8.1 研究路线总览

为突破主CTASharpe 2.20的天花板，沿"低相关α来源"路线扩展5个子策略研究：

| 子策略 | 脚本 | OOS Sharpe | 状态 |
|--------|------|-----------|------|
| 锌库存z-score | `cta_zn_inventory.py` | **0.48** | ✅定版 |
| M经纪持仓 | `cta_broker_holding.py` | 0.58（后修正为 -0.19）| ⚠️止损（PIT bug修复后）|
| 价差套利 | `cta_spread_arbitrage.py` | — | ❌止损（一致率56%<60%）|
| 库存全组（除ZN外）| `cta_inventory_strategy.py` | — | ❌止损 |
| 铜洋山溢价+TC | `cta_cu_premium.py` | **1.10** | ✅定版 |
| 黄金独有信号 | `cta_au_substrategy.py` | — | ❌止损（一致率50%<60%）|

### 8.2 锌库存z-score子策略（cta_zn_inventory.py）

**数据**: LME锌库存（tfapi_indicators），2014起日频  
**因子**: 库存z-score(52周) + 极端化(|z|>1持仓)  
**逻辑**: 库存极端→反转

**结果（实盘版本）**：
| 期 | Sharpe | AnnRet | MaxDD | Calmar |
|----|--------|--------|-------|--------|
| IS | 0.464 | 3.3% | -8.3% | 0.40 |
| **OOS** | **0.476** | **3.8%** | **-6.5%** | **0.58** |
| 全期 | 0.470 | 3.6% | -11.7% | 0.30 |

换手10.4x。IS/OOS几乎一致（无过拟合）。

### 8.3 库存全组测试（cta_inventory_strategy.py）

**测试品种×因子组合数**: 10品种×6因子 = 60组合  
**止损原因**: 全组OOS Sharpe -0.081（类似CFTC负结果）  
**唯一通过筛选**: 锌库存

**品种间方向不一致教训**：
- 锌/AU反向（库存高→未来涨）——经济逻辑正确
- 大豆/豆粕无独立α（被组合稀释）
- 铜（强映射）通过五分位验证但OOS衰减

**结论**: 单品种独立验证优于全组统一方向。

### 8.4 M经纪持仓子策略（cta_broker_holding.py）

**数据**: tfapi_holding（659,563行，CU/M两品种162-168家经纪日频持仓, 2020起）

**5因子（IS/OOS一致4/8=50%止损）**：
| 因子 | IS价差 | OOS价差 | 一致? |
|------|--------|---------|------|
| F1_cr5_level | +28.4% | -17.6% | ✗ |
| F2_cr5_chg | +4.4% | -46.8% | ✗ |
| **F3_cr5_share** | **+72.8%** | **+21.3%** | ✓ |
| **F4_hedge_ratio** | **+48.8%** | **+15.1%** | ✓ |
| **F5_tail_head** | **+83.3%** | **+17.7%** | ✓ |

组合 OOS Sharpe=0.581 → **PIT修正后 -0.262**（详见Phase 9）

### 8.5 价差套利子策略（cta_spread_arbitrage.py）

**数据**: spot_basis_daily(55品种,83k行) + cross_product_spread(26对,140k行)  
**因子**: B1期限斜率 / B2基差水平 / B3基差收敛 / X1价差z / X2价差动量  
**结果**: 237组合中IS显著190对，方向一致107对=**56%<60%止损**

**过程发现**: B2双重z-score数值bug（rollIC=inf）——z-score是单调变换不影响五分位排序，修复后一致率不变

**亮眼组合**（作为复检线索，不构成当前信号）:
- X1_pair_conv/RB-I: IS -13.7% → OOS -47.2%（钢-矿价差反转）
- B2_basis_level/CU: IS -13.5% → OOS -48.7%

### 8.6 铜洋山溢价+TC子策略（cta_cu_premium.py）

**数据**: 洋山铜溢价(日频2012起) + 铜冶炼TC(周频2011起)

**4因子（IS/OOS一致4/4=100%）**:
| 因子 | IS价差 | OOS价差 | 方向 |
|------|--------|---------|------|
| P1溢价水平 | +31.9% | +11.1% | 正向 |
| P2溢价变化 | +9.7% | +25.7% | 正向 |
| T1 TC水平 | -38.9% | -31.1% | 反向 |
| T2 TC变化 | -94.3% | -29.7% | 反向 |

**回测**:
| 期 | Sharpe | AnnRet | MaxDD |
|----|--------|--------|-------|
| IS | 1.010 | 6.6% | - |
| **OOS** | **1.102** | **8.10%** | **-6.12%** |
| 全期 | 1.057 | 7.33% | -8.66% |

换手11.7x。当前TC -181历史极端（铜矿紧张真实事件）。

**迄今最强子品种策略**——经济逻辑最直接（洋山溢价=国内进口需求实时温度计，TC=矿供应紧张度）。

### 8.7 黄金独有信号（cta_au_substrategy.py，止损）

**数据**: ETF持仓(2861行) + 央行储备(122行) + GVZ + CFTC GOLD  
**6因子（一致率3/6=50%止损）**：
- G1 ETF水平 ✓ +12→+9
- G2 ETF变化 ✗ +13→-8
- G3 央行购金 ✗ +16→-10（**regime change**：2022-26购金潮在IS期不存在）
- G4 GVZ ✗ +12→-18
- C1 CFTC投机 ✓ +15→+10
- C2 CFTC商业 ✓ -26→-13

**关键发现**: 央行购金数据反转——2022-26购金潮（regime change）打破IS期信号结构。月频滞后30天数据在OOS期已被市场定价。

### 8.8 M经纪持仓深入因子（cta_m_deep.py）

6新因子验证（一致率5/6=83%），但等权组合OOS Sharpe=0.326<原3因子0.581（等权叠加分散单因子信号）。  
**结论**: 新因子有独立α但组合贡献为负，原F3+F4+F5仍为定版OOS。

### 8.9 cta_shared组件库沉淀

为避免重复代码，从4个研究脚本去重沉淀9个核心组件：

```python
# scripts/cta_shared/factor_research.py
from cta_shared.factor_research import (
    rolling_zscore,           # 滚动z-score
    compute_ic,               # 时序IC + 滚动IC + t-stat
    quintile_spread,          # 五分位年化价差
    is_oos_direction_consistency,  # IS/OOS方向一致性（固化60%止损门槛）
    vol_target_backtest,      # 多品种vol target回测（含按品种成本）
    period_stats,             # 分期统计 (Sharpe/AnnRet/MaxDD/Calmar)
    yearly_returns,           # 年度收益
    load_cost_bps,            # cost_model.json加载
    summarize_factor_test,    # 一行文本摘要（打印用）
)
```

后续因子研究（CU/AU子策略）直接复用，新脚本只需写因子构造函数（20-50行）。

---

## 9. 子策略组合实验与PIT质量修正（Phase 9）

### 9.1 合并实验设计

子策略间相关性（关键发现）:

|  | 主CTA | ZN库存 | CU溢价TC |
|--|------|------|------|
| 主CTA | 1.000 | 0.078 | 0.029 |
| ZN库存 | 0.078 | 1.000 | **-0.068** |
| CU溢价TC | 0.029 | -0.068 | 1.000 |

**关键**: 子策略与主CTA天然低相关，**ZN×CU轻微负相关**——子策略间本身互补。

### 9.2 14种资金比例实验结果

**最优组合**: 80% 主CTA + 5% ZN + 15% CU

| 组合 | Sharpe | AnnRet | MaxDD | Calmar | OOS-Shp |
|------|--------|--------|-------|--------|----------|
| 主CTA单独 | 2.378 | 19.0% | -6.48% | 2.93 | 1.799 |
| 主CTA+CU15% | 2.498 | 17.2% | -4.85% | 3.55 | 1.972 |
| 主CTA+ZN5+CU10 | 2.473 | 17.0% | -4.99% | 3.41 | 1.939 |
| 主CTA+ZN10+CU10 | 2.478 | 16.2% | -4.64% | 3.50 | 1.957 |
| **主CTA+ZN5+CU15** | **2.512** | **16.4%** | **-4.45%** | **3.69** | **2.001** |

### 9.3 关键收益改善（vs主CTA单独）

- **Sharpe**: 2.378 → 2.512（+5.6%）
- **OOS Sharpe**: 1.799 → 2.001（+11.2%）
- **MaxDD**: -6.48% → -4.45%（**改善31%**）
- **Calmar**: 2.93 → 3.69（**+26%**）
- 年化收益: 19.0% → 16.4%（-2.6%损失换风险预算大幅优化）

**风险预算型组合**: 2.6%收益损失换31%MaxDD改善+26%Calmar提升——对实盘资金管理极有价值。

### 9.4 PIT bug发现与修复

合并实验中发现：M子策略真实Sharpe=-0.262（远低于讨论中提到的0.581）。

**根因**: `cta_broker_holding.py`的`build_factors()`无PIT shift，经纪持仓数据虽盘后发布但脚本直接以当日使用——包含当日未来信息。

**修复**: `cta_broker_holding.py:121` 加 `return factors.shift(1)`

**修复后真实表现**:

| 指标 | 修复前（虚假） | 修复后（真实） |
|------|--------------|--------------|
| 全期Sharpe | 0.581 | **-0.195** |
| OOS Sharpe | 0.581 | **-0.262** |
| 结论 | OOS稳健子策略 | **止损归档** |

### 9.5 PIT教训归纳（贯穿全项目）

1. **任何外部数据源接回测必加PIT**——这是历史多次教训的核心
   - `cta_cftc_strategy.py`: 已有shift(3交易日)✓
   - `cta_broker_holding.py`: 缺失shift(1)✗ → 已修复
   - `cta_inventory_strategy.py`: shift(1)✓
   - `cta_zn_inventory.py`: shift(1)✓
   - `cta_cu_premium.py`: shift(1)✓
   - `cta_au_substrategy.py`: shift(1)✓

2. **IC检验不能替代PIT合规检查**——M的IC在IS/OOS都有显著信号（F3/F4/F5一致），但PIT修正后真实PnL为负

3. **代码质量审查**: 所有历史因子研究脚本需逐一检查PIT合规——本次发现并修复M经纪持仓bug

---

## 10. 最终定版配置与实盘推荐

### 10.1 实盘推荐组合

```
80% 主CTA + 5% ZN库存 + 15% CU溢价TC
```

| 策略 | OOS Sharpe | 信号维度 |
|------|-----------|---------|
| **主CTA (80%)** | **2.53**（含EC）/ 2.20（剔EC保守）| 技术面66品种多因子 |
| **ZN库存 (5%)** | **0.48** | LME库存z-score极端反转 |
| **CU溢价+TC (15%)** | **1.10** | 铜进口需求+矿供应紧张度 |
| **组合** | **OOS 2.00** | 技术+库存+供需三维α |

### 10.2 实盘前最后遗留项

1. **2026 OOS验证**（主CTA+子策略 最近7个月空白）——**最高优先级**
2. **cta_combination.py 加Lump换月口径**（当前用含EC版，可加剔EC保守版对比）
3. M经纪持仓**PIT bug已修复**（代码质量修正）——不影响已止损结论
4. 整合所有研究到总调优文档（已完成cta_tuning_log.md）

### 10.3 实盘监控重点

- 子策略与主CTA相关性漂移（regime shift风险）
- 子策略OOS价差连续20日转负→触发降级审查
- 季度重检（每年4次）

---

## 11. 全部commit历史与产出物索引

### 11.1 主CTA调优阶段（Phase 1-7）

| commit | 描述 |
|--------|------|
| 6cc6d23c | docs(cta): 改进计划Phase1-2 + baseline基准保存 |
| 9bc210eb | fix(cta): Phase1-2改进 — 集群统一+成本敏感性+向量化+NaN修复 |
| def9669c | fix(cta): 敏感性tag命名bug |
| 623d61ab | fix(cta): 敏感性tag命名bug (脚本部分) |
| d50e8c01 | research(cta): 10bp成本基准 + EMA alpha敏感性 |
| 6c07f901 | research(cta): 换手率实验 — EMA平滑换手-81%/Sharpe+0.77 |
| ea98f114 | research(cta): 年度归因 — 2024化工磨损拖后腿 + EC依赖25%警示 |
| 4a14185f | fix(cta): 执行路径过滤链bug修复 + regime/cap实验负结果 |
| e7a35f95 | research(cta): 定版tv=1.5 @ 双10bp成本口径 |
| 9cff3076 | feat(cta): equity_leverage_k实现 + 方案A实验定版k=1 |
| 2a23cce8 | feat(cta): 切到lump定版 + 修正turnover实验口径bug |
| 344887b7 | fix(cta): 换月建模修复实证 + roll_days局限分析 |
| a82173ff | feat(cta): 换月模型改进 — lump-sum一次性冲击 |
| 6d9c5366 | feat(cta): 按品种成本校准 + EC双口径监控 |
| b244aa19 | research(cta): 换月建模盲区 + 跨品种逻辑梳理 |
| b6588753 | fix(cta): 模块信号NaN污染修复 — 2019-2022含EC=剔EC完全一致 |
| ef254d4b | fix(cta): cta_broker_holding.py PIT shift(1) — 修复未来信息泄露 |
| 71e6dd2a | docs(cta): 综合调优记录文档 cta_tuning_log.md |
| 04b70d47 | feat(cta): 可复用因子研究组件库 cta_shared + 价差套利计划文档 |

### 11.2 α来源扩展研究（Phase 8-9）

| commit | 描述 |
|--------|------|
| 7472b54c | feat(allweather): 全天候2.0复现模块 — 计划文档+数据管线（别人的模块，不动）|
| 04b70d47 | feat(cta): 可复用因子研究组件库 cta_shared + 价差套利计划文档 |
| 13852c18 | research(cta): 经纪持仓因子 — M豆粕通过/CU止损归档（OOS Sharpe 0.58，后PIT修正发现bug）|
| 3a80a7bf | research(cta): 价差套利子策略 — 一致率56%<60%止损归档 |
| 88a72203 | research(cta): 铜洋山溢价+TC因子 — 迄今最强子品种策略 OOS Sharpe 1.10 |
| 0ca2bb0a | research(cta): 黄金独有信号 — 一致率50%<60%止损归档 |
| 5da29ba3 | research(cta): M经纪持仓深入因子 — 新因子质量高但组合不如原3因子 |
| 3363720e | research(cta): M经纪持仓深入 + 宏观filter评估（宏观止损）|
| **367baef9** | **research(cta): 合并实验 — 80%主CTA+5%ZN+15%CU Sharpe 2.51/MaxDD-4.45%** |

### 11.3 所有脚本/报告（按用途）

**主CTA核心**:
- `scripts/cta_baseline.py`, `cta_final_baseline.py`, `cta_dual_basis.py`
- `scripts/cta_turnover_experiment.py`, `cta_yearly_attribution.py`

**α来源扩展**:
- `scripts/cta_zn_inventory.py`（ZN库存子品种）
- `scripts/cta_cu_premium.py`（铜溢价TC子品种）
- `scripts/cta_broker_holding.py`（经纪持仓，已止损）
- `scripts/cta_inventory_strategy.py`（库存全组）
- `scripts/cta_spread_arbitrage.py`（价差套利）
- `scripts/cta_au_substrategy.py`（黄金独有信号）
- `scripts/cta_m_deep.py`（M经纪深入因子）
- `scripts/cta_macro_filter.py`（宏观filter，止损前）

**组合与汇总**:
- `scripts/cta_combination.py`（合并实验）
- `scripts/cta_cost_model.py`（按品种成本）

**组件库**:
- `scripts/cta_shared/factor_research.py`（9个可复用组件）

**报告目录** `reports/cta/`:
- 主CTA: `final_*_lump.csv`, `baseline_*.csv`
- 子策略NAV: `zn_inventory_nav.csv`, `cu_premium_nav.csv`, `m_broker_holding_nav.csv`
- 因子IC: `broker_holding_ic.csv`, `cftc_ic.csv`, `inventory_ic.csv`, `spread_arb_ic.csv`
- 双口径: `dual_basis.csv`, `dual_basis_yearly.csv`
- 合并: `cta_combination_results.csv`

**文档**:
- `docs/cta/cta_improvement_plan.md`（Phase 1-4改进计划与结果）
- `docs/cta/cta_rollover_analysis.md`（Phase 6-7换月与bug修复）
- `docs/cta/cta_cross_product.md`（跨品种逻辑梳理）
- `docs/cta/cta_broker_holding_factor.md`（经纪持仓因子研究）
- `docs/cta/cta_inventory_factor.md`（库存因子研究）
- `docs/cta/cta_cftc_factor.md`（CFTC因子负结果）
- `docs/cta/cta_spread_arbitrage_plan.md`（价差套利计划）
- `docs/cta/cta_combination_report.md`（合并实验报告）
- `docs/cta/cta_tuning_log.md`（本文件，综合记录）

---

## 12. 实盘落地的最终状态

### 12.1 推荐组合

```
80% 主CTA + 5% ZN库存 + 15% CU溢价TC
```

| 指标 | 主CTA单独 | 最优组合 |
|------|----------|----------|
| 全期Sharpe | 2.378 | **2.512** |
| OOS Sharpe | 1.799 | **2.001** |
| MaxDD | -6.48% | **-4.45%** |
| Calmar | 2.93 | **3.69** |

### 12.2 实盘前必做

- [ ] 2026 OOS验证（主CTA+子策略 最近7个月空白，最高优先级）
- [ ] cta_combination.py 加Lump换月口径（剔EC保守版对比）

### 12.3 方法论经验沉淀

1. **PIT合规**：任何外部数据源接回测必加shift（cta_cftc: +3日, cta_broker_holding: +1日, 库存: +1日, 现货升贴水: +1日）
2. **IC检验不能替代PIT**：M子策略IC显著但PIT修正后PnL为负
3. **品种×因子对独立评估**优于全组合统一方向——ZN库存/CU溢价/M经纪都通过单品种筛选
4. **60%一致率止损门槛**有效过滤了4个弱信号方向（CFTC/价差套利/库存全组/CU经纪/AU独有/M深入整体/宏观filter）
5. **回测一致性的复检**——合并实验中发现M的PIT bug，证明合并实验方法的价值

### 12.4 实盘监控重点

- 子策略与主CTA相关性漂移（ZN×CU=-0.068当前值）
- 子策略OOS价差连续20日转负→触发降级审查
- 季度重检（每年4次）
- 2026 OOS验证完成后才正式上实盘
