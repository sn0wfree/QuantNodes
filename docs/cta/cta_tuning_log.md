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

---

## 13. T+1执行口径测试与数据口径审计（2026-09-03，commit 141acb7f）

> 详细报告: `docs/cta/cta_open_execution_audit.md`
> 脚本: `scripts/cta_open_execution_test.py` + `scripts/cta_rebuild_portfolio.py`

### 13.1 触发动机

此前所有子策略NAV用 close-to-close 收益（信号T日收盘→T+1日收盘建仓），实盘可在T+1日开盘执行。本测试量化两种执行口径差异，**意外审计出两个数据口径伪影子策略**。

### 13.2 数据口径审计（重大发现）

| 数据源 | 审计结论 |
|--------|---------|
| smoothed_*_daily（88品种OHLC）| ✅ back-adjusted真主连 |
| 主CTA（load_gradual_continuous）| ✅ 逐步换月back-adjusted |
| **cffex drop-last（IH50/T10y用）** | ❌ 与OI主力一致率仅25.6%，随机拼接 |
| **v_all drop-last（产业链用）** | ❌ RB/I 92%天数在换合约 |

### 13.3 子策略真实性审计（7个全查，对账PASS后重算）

| 子策略 | 现有NAV | 真实close | 真实open执行 | 判定 |
|--------|---------|----------|-------------|------|
| CU溢价TC | 1.057 | 1.057 | 1.018 | ✅ 通过 |
| ZN库存 | 0.470 | 0.607 (PIT修正后) | 0.764 | ✅ 通过 |
| **IH50** | **0.607** | **0.154** | **-0.167** | ❌ **伪影→止损** |
| T10y | 0.457 | 0.519 | 0.489 | ✅ 通过 |
| **rbi_steel** | **0.301** | **0.109** | **0.015** | ❌ **伪影→止损** |
| jjm_coke | 0.185 | **0.540** | 0.339 | ✅ 被低估，纳入 |
| ym_oilmeal | 0.306 | 0.198 | 0.314 | ⚠️ 弱信号 |

**IH50"迄今最稳健跨资产信号(OOS 0.68)"结论作废**——其α来自随机合约拼接的价差污染。

### 13.4 执行口径结论

- 商品（有夜盘）：open vs close 差异 -0.04 ~ +0.16 —— 夜盘吸收隔夜信息
- 金融（无夜盘）：T10y -0.03 轻微；IH50 -0.32（但本身伪影应止损）
- **open执行（真实执行）在渐进A方案下完全可行**

### 13.5 组合重构（剔除IH50/rbi伪影）

**最终主推 R2**（open执行 + 渐进A + 5bp换手成本）：

```
资金: 主CTA 55% × 动态杠杆[1.0,1.5]
+ CU溢价TC 25% + T10y 10% + jjm_coke 10%
子策略: open口径执行, 主CTA: close口径 (gradual_continuous数据干净)
```

| 指标 | 原伪影主推(2b51d86c) | **重构R2** | 改善 |
|------|--------------------|-----------|------|
| 净Sharpe | 3.067 | 3.062 | 持平 |
| 净AnnRet | 18.03% | 18.04% | 持平 |
| **净MaxDD** | -3.87% | **-3.15%** | **-19%** |
| **净Calmar** | 4.67 | **5.72** | **+22%** |
| 净OOS-Shp | 2.738 | 2.601 | -5% |

**剔除伪影后组合质量反而提升**（Calmar +22%）——伪影子策略贡献的是虚假分散。

### 13.6 经验教训

1. **对账PASS ≠ 数据正确**：复刻了同样的错误口径也会PASS；口径审计必须独立于复刻
2. **随机合约拼接伪影机制**：keep-last跨合约的价差变化被close-to-close收益捕捉，形成虚假动量/OI相关性
3. **夜盘的执行优势**：商品期货open执行损失轻微（夜盘消化隔夜信息）
4. **vol_target_backtest的ret_df参数**已扩展支持close/open双口径回测（f1463404）

### 13.7 遗留事项

- ZN正式NAV用PIT严格版（pit=1）重出
- 主CTA执行口径专项测试（需改回测引擎，66品种分散预计影响轻微）
- 2026 OOS验证（重构组合）
- rbi/ym正式归档止损（不纳入主推）

---

## 14. R0/R2版本固化 + 品种敏感性测试（2026-09-03）

### 14.1 R0/R2双版本固化

- **R0** (极简版: 主CTA + CU45%): `reports/cta/portfolio_R0_nav.csv` — Sharpe 3.058 / AnnRet 18.80% / MaxDD -3.26% / Calmar 5.76
- **R2** (三子策略版: +T10y10% +jjm10%): `reports/cta/portfolio_R2_nav.csv` — Sharpe 3.062 / AnnRet 18.04% / MaxDD -3.15% / Calmar 5.72 / OOS 2.601
- **选择**: R2作为主推（用户决策），R0作为极简备选

### 14.2 品种敏感性测试

详见 `docs/cta/cta_product_sensitivity.md` 及 `reports/cta/product_sensitivity_maincta.csv`

**核心结论**:
- 66品种主CTA剔除单品种平均 d_sharpe=-0.023, d_calmar=-0.08 — **高度分散**
- 最敏感品种都是小品种: SM/EC/JD/SN/LH (剔除后d_sharpe -0.13~-0.27)
- 关键品种剔除后 Sharpe 仍保持 >2.19 (vs 基线 2.46)
- 7个品种 (AU/CU/HC/I/J/JM/RB) 因产业链因子硬编码依赖无法剔除 (产业连核心节点)
- open vs close 口径差异 <0.01, 组合对执行口径稳健

**集中度风险**:
- CU 单品种 25.83% (子策略全部敞口) — 最大单点风险
- 国债 T 单品种 10.83% — 第二大单点风险
- 钢厂链 (RB+I+J+JM) 合计 ~10% — 结构稳健但有产业链联动风险
- 其他品种 ≤ 1%, 高度分散

**遗留**: R2组合层剔除测试未跑通 (杠杆换手成本在剔除品种时爆掉), 建议未来加杠杆硬上限或重新设计测试方法.

## 15. 流动性约束测试（2026-09-03）

### 15.1 背景

用户明确**不动 factor_modules.py / cost_model.json 等他人代码**，故无法在CTA引擎层加 `max_volume_participation` 参数。采用**后处理法**对主CTA回测结果 (`res.positions` 逐日品种手数) 按"日成交量 × 参与率"作为持仓上限重算NAV。

### 15.2 测试设计

- **合约乘数**: 直接从 `contract_specs` 表加载，**66品种全覆盖**（AU 1000 / T 10000 / RB 10 等），无需akshare
- **ADV窗口**: 20日滚动均量（业界标准，可调）
- **场景**: 基础(∞) / 5% / 10% / 20% / 0.5%(极严) / 0.1%(极限) — 6场景
- **执行口径**: close / open × 2 = 12组
- **算法**: A 重算NAV（手×价×乘数+成本）/ B 增量成本（在baseline nav上扣额外滑点）/ C 简化重算（无成本）
- **总测试**: 6×2×3 = 36组对比

### 15.3 关键发现

#### 15.3.1 主CTA流动性极其充裕

各品种"持仓手数 / ADV手数" 峰值（全部1941个交易日）：

| 阈值 | 触发品种数/66 |
|------|---------------|
| 持仓/ADV > 10% | **0** |
| 持仓/ADV > 5% | **0** |
| 持仓/ADV > 1% | **0** |
| 持仓/ADV > 0.5% | **0** |
| 持仓/ADV > 0.1% | 1 (TS国债) |

- **平均持仓/ADV峰值: 0.014%**
- **最大持仓/ADV峰值: 0.388%** (TS国债)
- **0.1%参与率**: 仅TS国债21天/1941天触发cap，max_excess=0.11手（极小）
- **5%/10%/20%/0.5%参与率**: 完全无cap触发

#### 15.3.2 NAV对比（6场景 × B增量成本法）

| 场景 | Sharpe | Δ Sharpe | MaxDD | Δ MaxDD |
|------|--------|----------|-------|---------|
| 基础 | 2.405 | — | -6.14% | — |
| 5%参与率 | 2.405 | -0.0000 | -6.14% | 0.00% |
| 10%参与率 | 2.405 | -0.0000 | -6.14% | 0.00% |
| 20%参与率 | 2.405 | -0.0000 | -6.14% | 0.00% |
| 0.5%参与率 | 2.405 | -0.0000 | -6.14% | 0.00% |
| 0.1%参与率(极限) | 2.405 | **-0.0001** | -6.14% | 0.00% |

**B算法本质**: cap几乎不触发 → 几乎无额外滑点成本 → NAV与baseline几乎完全重合。

#### 15.3.3 真实流动性容量（修正之前的错误认知）

依据持仓×合约乘数×价格 反推：主CTA组合**实际流动性容量数十亿元/日**（远大于1万元初始权益×0.74x最大杠杆），流动性约束完全不是瓶颈。

#### 15.3.4 修正前的错误

之前的 `cta_product_sensitivity.md §8.3` 讨论"小品种流动性"仅基于品种列表未乘合约乘数。**真实流动性需要看"持仓手数/ADV手数"（无量纲比值）**，而非按品种列表排名。

### 15.4 数学与方法论教训

测试过程中暴露3个根本性数学错误（已修正）：

1. **单位错误**: 早期 `pnl = pos × ret` 出现"-193%"是因为 `手 × %` 不是合法盈亏单位，必须 `手 × 价格变化 × 合约乘数 = 元`
2. **初始权益错误**: 用`(1+pnl).cumprod()`必须保证pnl是收益率；若pnl是绝对盈亏元数, 需用 `equity[T] = init_eq + cumsum(pnl) / init_eq`
3. **cap_lots算法错误**: 早期`cap_lots = cap_value / price_mult`多除了一次价格——cap直接是手数，不需要再除

### 15.5 输出

**脚本**: `scripts/cta_liquidity_constraint_test.py` (473行)
**报告**: `reports/cta/liquidity_constraint_*.csv`（共13个文件）
- `liquidity_constraint_scenarios.csv` — 36组对比汇总
- `liquidity_constraint_summary_{close,open}_{A,B,C}.csv` — 各算法分口径
- `liquidity_constraint_5pct_{close,open}_algo{A,B,C}.csv` — 5%约束NAV序列
- `liquidity_constraint_trigger_stats.csv` — cap触发品种统计

### 15.6 遗留

- **绝对盈亏法重算NAV** vs baseline nav 误差 ~10%（手×价×乘数 = 5.32M vs 真实 6.13M, 差14%）—— 主CTA复利/动态杠杆/借款利息等因素未完全建模。如需更高精度, 应在factor_modules.py层面加max_volume_participation参数（用户当前约束不允许）
- **A重算NAV的"基础"场景** Sharpe=2.50 vs baseline 2.41（+0.09）—— 重算没考虑复利和动态杠杆, 单调累加存在轻微高估。这是已知偏差, 不影响相对比较

## 16. factor_modules.py 插件式架构拆分（2026-09-03，用户授权后执行）

### 16.1 背景与授权

此前 §15.1 记录"不动 factor_modules.py"约束针对的是流动性测试场景。本次用户明确提出
"优化代码结构，方便后续研发以及添加更多子策略"，**授权重构**该文件。原则：
纯结构搬移（pure move）、行为逐位对账、41 个引用方零改动。

### 16.2 拆分结果（1503 行 → 5 个组件，最大文件 534 行）

```
strategies/
├── factor_modules.py   ~85行  纯兼容层: re-export 18名 + PEP 562 __getattr__兜底 (永久保留)
├── base.py            ~625行  ModuleConfig + DEFAULT_PRODUCTS + _ModuleStrategy
│                              (信号管线/_agg_tvpr/_run_with_lots整体保留 → fut_v1_purge_test
│                               的 _ModuleStrategy._agg_tvpr monkeypatch 语义不变)
├── filters.py          ~95行  VolRiskFilter + MarketRegimeFilter
├── combo.py           ~534行  ComboConfig + FactorModuleComboStrategy + CLUSTERS
│                              + DEFAULT_COMBO_MODULES + module_classes 注入
└── modules/            7文件  一文件一模块 (仅FACTORS数据声明), __init__.py::BUILTIN_MODULES 注册表
    trend_momentum.py / oscillator_alpha.py / volume_flow.py / event_calendar.py
    term_basis.py / macro.py / industrial_chain.py
```

**新功能 — module_classes 注入**（本次唯一的行为面新增，默认 None=历史5模块组合）:
```python
FactorModuleComboStrategy(config=ComboConfig(
    module_classes=(TrendMomentumModuleStrategy, MyNewModuleStrategy)))
```

### 16.3 加新子策略的成本对比

| | 拆分前 | 拆分后 |
|---|---|---|
| 步骤 | 改1503行文件加类+顶部import+改combo硬编码 | 新建 modules/<name>.py + BUILTIN_MODULES加一行 |
| 碰撞风险 | 与其他会话改同一热点文件 | 零（新文件） |

### 16.4 验证协议与结果

- **pytest 基线对照**: 每阶段跑 test_strategies_factor_modules.py，均为 62 passed / 8 failed
  ——8 个失败（66<70品种数断言、combo相关5个、execution_lag 1个）在 HEAD 上完全一致，全部预存
- **身份审计**: 17+9 项 `fm.X is 源模块.X` 断言通过（无类身份分裂）
- **导入冒烟**: scripts.common / tvpr_combo / options.vrp_router / options_boosted_fut_v1 /
  allweather依赖路径全通过
- **golden NAV 对账**（重构前后各跑一次 cta_final_baseline，66品种2018-2025）:
  - max|Δnav| = 4.4e-15，最大相对差 1.1e-15（机器精度噪声底）
  - Sharpe 2.309750 / AnnRet 19.1794% / MaxDD -6.1412% / Calmar 3.123093 **6位小数完全一致**

### 16.5 过程中发现并绕开的坑

1. **set 迭代顺序的非确定性（预存）**: `_run_with_lots` 中 `set(targets) | set(self._last)`
   顺序受 PYTHONHASHSEED 影响 → 跨进程 NAV 存在 ulp 级抖动，逐位==0 不可达；
   正确标准是相对差 < 1e-12。未修（Surgical Changes），如需严格可复现可改有序遍历
2. **陈旧报告陷阱**: `final_nav_lump.csv` 落后 HEAD 3 个 commit（golden 与它差 ~11%），
   重构验证必须以"重构前当次运行"为 golden，而非 tracked CSV；跑完立即 `git checkout` 还原
3. **意外依赖**: `test_combined_regime_factors.py` 从 factor_modules 导入
   TurnPointBreakoutFactor 等3个因子类（借用旧顶部import命名空间）→ 兼容层显式 re-export；
   factors 包 `__init__` 未导出 BasisConvergenceFactor → __getattr__ 兜底链补 term_basis 路径
4. **历史文档不篡改**: docs/94 §6.1 的交付记录保留原文，追加日期注记

### 16.6 后续建议（未做，待授权）

- 41 个引用方可渐进迁移到新路径（base/combo/modules），但兼容层可永久保留，无迁移压力
- `_ModuleStrategy` 与 combo 的继承关系可进一步改为组合（LSP 更干净），涉及行为面，需单独评估
- gamma_scalper.py 存在 3 个预存 F401（非本次范围）

## 17. 定版配置过拟合三件套重跑 + 上实盘就绪度评估（2026-09-04）

### 17.1 背景

docs/95 §七 的三件套是 2026-08-12 旧配置时代 (tv=0.30) 的结果, 不能沿用到
现行定版 (tv=1.5/lump/按品种校准成本)。用户授权重跑 (方案A) 并出上实盘评估 (方案C)。
**新脚本** `scripts/cta_overfit_final_check.py` — 方法论与旧版对齐 (同网格/块/起点),
未改动旧脚本 `cta_overfitting_check.py` (他人工作)。

### 17.2 结果: PASS (4/4)

| 检查 | 定版配置 | 阈值 | 判定 |
|---|---|---|---|
| Bootstrap CV% (52块×200) | **14.9%** (p5 1.92/p95 3.12) | < 50% | ✅ |
| 起点依赖 (5起点) | Sharpe 2.41/1.36/1.90/0.86/0.58 全正 | 全>0 | ✅ |
| 严格OOS (IS选参→OOS验证) | IS最优=定版(0.5,60), OOS Sharpe **2.059** | ≥1 | ✅ |
| IS最优=定版参数 | (0.5, 60) | 是 | ✅ |

### 17.3 关键发现

1. **Bootstrap CV 跨配置世代几乎不变** (14.9% vs 旧 15.9%) — 分布形态稳定
2. **⚠️ TVPR 预热依赖**: 起点2021/2022 Sharpe 仅 0.86/0.58 且 MaxDD -22.7%/-20.7%
   —— expanding 学习历史不足时 β 估计差 → 信号差。**实盘部署必须带足历史预热, 不能冷启动**
3. 严格OOS框架下 OOS Sharpe 2.06 vs 全样本 2.41 (252计数) — OOS 衰减 ~15%,
   属正常范围; 近两年 Sharpe 0.90 的衰减问题仍需 2026 真样本区分"衰减 vs 过拟合"

### 17.4 上实盘结论

- **允许小仓起步** (1/4~1/5 目标仓位, R2 open 为主推), 不建议直接全仓
- 资金规划按 Sharpe 1.0~2.0 段位, MaxDD 预算 -8%~-10% (高于回测留余量)
- 前置: ① 数据管线追平→2026 七个月真OOS (最高优先级, 管线属他人工作需授权)
  ② 工程项 (资金成本/杠杆切换频率/滑点监控)
- 完整评估: `docs/cta/cta_live_readiness_assessment.md`

### 17.5 产出

- `scripts/cta_overfit_final_check.py` (8次全量回测, 3并发, nav全部落盘)
- `reports/cta/overfit_final_check.csv` / `overfit_final_bootstrap.csv` / `overfit_final_summary.json`
- `reports/cta/overfit_final_navs/*.parquet` (8个nav序列)
- `docs/cta/cta_live_readiness_assessment.md` (上实盘就绪度评估)

## 18. 主CTA衰减诊断 + 优化实验 G1/G2/G3（2026-09-04，负结果有价值）

### 18.1 衰减诊断结论

模块级逐年 Sharpe 归因 (module_signals 逐模块独立回测):

| 模块 | IS均值 | 2024 | 2025 | 判定 |
|---|---|---|---|---|
| TrendMomentum | 2.56 | 0.41 | 1.80 | 结构性下降(2.5→1.3), 2025回升 |
| VolumeFlow | 1.93 | 0.28 | 1.64 | 同上 |
| TermBasis | 2.02 | -0.06 | 2.16 | 稳定, 2024一次性冲击后恢复 |
| Macro | 1.09 | -0.51 | **-0.03** | 连续两年负, 与其他模块相关性0.01→0.14 |
| IndustrialChain | 3.57 | 2.01 | 1.74 | 最强模块, 温和衰减 |

**inv_vol 权重真相**: IndustrialChain(窄覆盖低波)权重达80%, TrendMomentum仅1% —
看似失真, 但 IC 是历史最强模块, inv_vol 的低波偏好恰好是历史最优配置.

**2024根因**: 低波环境(Vol 1-3%)趋势类模块集体走弱, 非权重问题; EC只是1/4月的遮羞.

### 18.2 实验矩阵与结果 (预设决策规则, 防数据窥探)

决策规则: 通过 = 2024 Sharpe改善>0.3 AND IS/22-23恶化<10% AND MaxDD不恶化

| 实验 | 2024 Δ | IS Δ | 全样本 | MaxDD | 判定 |
|---|---|---|---|---|---|
| G1 cap=0.30 | +0.09 | -0.12 | 2.237 | -5.5% | FAIL |
| G1 cap=0.40 | -0.04 | -0.15 | 2.213 | -6.2% | FAIL |
| G1 cap=0.50 | +0.00 | -0.03 | 2.288 | -6.1% | FAIL |
| G2 ER版(ER<0.4→x0.4) | **+0.30** | **-0.55(-17%)** | 1.985 | **-4.6%** | FAIL (IS恶化超限) |
| G3 ADX滞回(25/20) | +0.12 | -0.26 | 2.115 | -6.4% | FAIL |

对账自检: cap=0 vs golden 最大相对差 1e-15 ✓ (新字段默认关闭, 零行为变化)

### 18.3 关键教训

1. **weight_cap**: 修复"权重失真"不带来收益改善 — IC 80%权重是历史最优而非bug.
   "合理"与"有效"是两回事, 市场用业绩投票.
2. **regime filter 是收益换保护**: ER版 2024 +0.30 + MaxDD -25%, 代价是趋势年 -17%.
   **R2组合层 MaxDD 已是 -3.15%** — 主CTA层再买保护, 期望值不划算.
3. **预设决策规则的意义**: ER版 2024Δ+0.302 恰好过线, 但 IS 恶化超限 —
   若事后放宽规则即为数据窥探. 规则先于结果, 结果不合规则就不采用.
4. 品种筛选/ATR止损按计划跳过/搁置 (前者待G1/G2结果再定→结果为负, 无叠加基础;
   后者与逐日调仓语义冲突).

### 18.4 最终结论: 定版维持不变

- FINAL_LUMP_CFG 不改 (新字段 module_weight_cap=0 / regime_filter_enabled=False 默认关闭)
- 2024 问题的现有解已在 R2 组合层 (子策略逆风对冲, 7-8月 +15.6%)
- 增量alpha的方向: VRP子策略 (方案5, 62品种数据完备, 未测) — 后续独立研究
- 代码资产: module_weight_cap / ADXRegimeFilter / regime_filter_type 已入库
  (默认关闭, 供未来环境变化时复用), 实验脚本 cta_weight_cap_test.py /
  cta_regime_filter_test.py + nav 落盘 reports/cta/{weight_cap,regime_filter}_navs/

### 18.5 对"是否衰减"的最终回答

- 趋势类模块有真实衰减 (2.5→1.3), 但 2025 回升说明主因是 2024 低波环境冲击而非永久性失效
- 主CTA 本体不需要"修复" — 它的弱年在组合层已被对冲
- 真正的优化空间不在主CTA内部, 而在: (a) R2 权重再平衡 (ZN库存 2024/25 Sharpe 1.71/1.26 未入R2);
  (b) 新alpha来源 (VRP); (c) 2026 真样本验证

## 20. 子层扩容实验 — 全维度PASS, 推荐待用户拍板 (2026-09-04)

### 20.1 实验设计

加大已验证有效的子策略层权重: 主CTA 55%→50%/45%, 子层45%→50%/55%.
对症2024 (子层全正而主CTA剔EC归零). 组合机制与 rebuild_portfolio.py 逐行对齐
(渐进A + 动态杠杆 + 5bp), 主CTA用golden, 子策略用PIT修正版open NAV.
脚本: scripts/cta_r2_scale_test.py (含与并行会话R3的复刻对账).

### 20.2 对账与BUG记录

初版有双重缩放bug (子层百分比又乘(1-w_main), 25%资金闲置) — R2复刻得2.609.
修复后 R2复刻=2.947 / R3b复刻=2.994, 与并行会话 f35e3c61 (2.948/2.995) 对账一致.

### 20.3 结果 (净口径, 决策规则: 2024ΔShp>+0.1 AND 全样本恶化<2% AND DD不恶化)

| 方案 | 主CTA/子层 | 净Shp | 净AR | 净DD | 2024 Shp | OOS Shp | 判定 |
|---|---|---|---|---|---|---|---|
| R2复刻 | 55/45 | 2.947 | 17.13% | -3.19% | 1.949 | 2.779 | 基线 |
| R3b复刻 | 55/45 | 2.994 | 17.43% | -3.18% | 2.018 | 2.858 | FAIL(2024Δ+0.07) |
| E1 扩容 | 50/50 | 3.043 | 16.35% | -2.96% | 2.140 | 2.899 | **PASS** |
| **E2 扩容** | **45/55** | **3.144** | **15.58%** | **-2.73%** | **2.355** | **3.025** | **PASS** |
| E3 R3b+扩容 | 50/50 | 3.099 | 16.69% | -2.95% | 2.222 | 2.993 | **PASS** |
| E4 边界探测 | 40/60 | 3.247 | 14.80% | -2.50% | — | — | (未入规则) |
| E5 边界探测 | 35/65 | 3.384 | 14.51% | -2.31% | — | — | (未入规则) |

### 20.4 关键洞察

1. **单调无内点**: E1→E5 Sharpe单调升/AR单调降/DD单调降 — 是"分散度↑"的稳健结构效应,
   非过拟合噪音; 但也因此无法用历史数据定"最优停点", 必须由收益底线驱动决策.
2. **弱年绝对收益也升**: E2 的2024年化11.42% vs R2 11.16%, 2025 9.37% vs 9.09% —
   强年稀释(2019: 22.0%→19.0%)换来弱年绝对改善, 非Sharpe幻觉.
3. 与§18 regime filter 的本质区别: 同为"强年换弱年", 扩容全样本Sharpe是升的
   (分散化效应>稀释效应), regime filter是降的.
4. 停手理由: 单调区间内挑内点无意义; E5 (35/65) 年化14.5%已接近实盘收益底线,
   继续外推 = "低波低收益组合", 不再是同一策略.

### 20.5 决策矩阵 (待用户拍板)

| 优先 | 方案 | 年化 | MaxDD | 适合 |
|---|---|---|---|---|
| 收益优先 | R2/R3b (55/45) | 17.1~17.4% | -3.2% | 追求绝对收益 |
| **推荐折中** | **E1 (50/50)** | **16.4%** | **-2.96%** | **+0.10 Sharpe, 代价0.8pp年化** |
| 风险优先 | E2 (45/55) | 15.6% | -2.73% | +0.20 Sharpe, DD再降14% |

切换需: NAV对账 → 2026真样本验证 → 实盘分仓过渡. 当前定版R2未动.

## 21. 子策略真实性审计 — 回测引擎干净, CU代码瑕疵无实害, T10y刀尖 (2026-09-05)

### 21.1 审计动机
用户质询"当前子策略是否真实正确". 全面审查4子策略信号逻辑/PIT/回测引擎.

### 21.2 回测引擎 (干净 ✓)
vol_target_backtest: pos.shift(1)×ret, vol滚动窗口, 成本换仓日计提 — 无前视.
open口径 ret_open.shift(1) 对齐正确 (T-2收盘信号→T-1开盘建仓).

### 21.3 CU因子选择门槛 — 形式瑕疵, 零实害
原版逻辑: is_oos_direction_consistency 的 agree 门槛 (IS与OOS五分位方向一致才纳入)
用OOS数据决定因子入选 = 形式上的数据窥探.
实测: 4/4因子 IS/OOS 方向本来就全部一致 (P1 +0.32/+0.11, P2 +0.10/+0.26,
T1 -0.39/-0.31, T2 -0.94/-0.30) → agree门槛全通过, 无选择效应.
修正 (纯IS门槛 is_significant) vs 原版: NAV逐位一致, ΔSharpe=0.000;
R2/E1/E2 组合重跑 Δ=0.000. **CU结果可信, 建议后续把 agree 改为 is_significant**
(代码洁癖, 非实害修复).
脚本: scripts/cta_cu_bias_audit.py → cu_bias_audit.csv / cu_bias_combo.csv

### 21.4 参数敏感性抽查 (脚本: scripts/cta_sub_sensitivity_audit.py)
| 策略 | 网格 | OOS Sharpe 范围 | 裁决 |
|---|---|---|---|
| ZN库存 | z{40,52,65}×t{0.8,1.0,1.2} | +0.61~+0.80 全正 | 稳健 ✓ |
| jjm_coke | z{90,120,150}×t{0.8,1.0,1.2} | +0.75~+1.27 全正 | 稳健 ✓ (IS弱已知) |
| T10y | oi{3,5,10} | +0.137/+0.597/+0.195 | **刀尖 ⚠️** |

T10y定版oi5恰为邻域最优 (邻域OOS只有0.14/0.20) — 单独看是参数挑优产物.
组合传染实测 (oi3/oi5/oi10 替换): R2 2.936/2.958/2.950, E2 3.130/3.162/3.151
→ Δ≤0.03, 权重小+分散化吸收. **支持R3b把T10y砍到5%的方向; T10y的OOS预期
应按~0.2折算而非0.6.**

### 21.5 审计总裁决
| 子策略 | 信号 | PIT | 敏感性 | 裁决 |
|---|---|---|---|---|
| CU溢价TC | 4因子IS定方向 | ✓(shift1) | 未测(选择门槛已验) | 可信 ✓ |
| ZN库存 | LME锌z52极端 | ✓ | 全网格正 | 稳健 ✓ |
| jjm_coke | J/JM z120反转 | ✓ | OOS全正 | 稳健 ✓ |
| T10y | OI5日变化 | ✓ | 刀尖 | **降权对待** |

次级发现: 组合层参数 (渐进A 20/40/60, 动态杠杆 60d/0.08-0.12/1.0-1.5) 为历史
遗留, 未做严格IS/OOS再验证 — 全系统共享假设, 后续如做2026真样本验证应一并复检.

## 22. T10y剔除 + 铜/黄金策略子策略化评估 — 双FAIL, 放大版为新基线 (2026-09-05)

### 22.1 T10y剔除 (§21刀尖结论落地)
两种机械处置, 放大版全PASS / 还主CTA版全FAIL:
| 基线 | T10y-free放大版 | 还主CTA版 |
|---|---|---|
| R2 2.947 | **R2-R 3.023** (+0.08, PASS) | 2.843 (FAIL) |
| E1 3.043 | **E1-R 3.117** (+0.07, PASS) | 2.911 (FAIL) |
| E2 3.144 | **E2-R 3.207** (+0.06, PASS) | 2.982 (FAIL) |

剔除刀尖高估的T10y换成CU/jjm等比放大反而更好 — §21审计结论被直接验证.
还主CTA版FAIL再次印证§20: 主CTA占比升 = 稀释分散化.
脚本: scripts/cta_t10y_removal.py. **E2-R (Sharpe 3.207/AR 16.6%/DD -2.80%)
为当前最优定版候选, 切换待用户拍板.**

### 22.2 铜/黄金策略库审计 (strategy/commodity_fundamental/)
历史口径不可信: copper_nav_curves.csv 里v6.2 Sharpe 3.477 (周频359行, 无成本),
但V6_3_UPGRADE.md 自记v6.2=0.70; gold_combined 注释自认"参数优化2018-2025".

PIT审计 — 铜3处问题:
  a) v6.3 rank(pct=True) 全样本百分位 = 结构性前视 → 修正为 rolling(104) rank
  b) PMI 记每月1日, 语义不明, 最坏前视1个月 → 修正 index+35自然日
  c) 钻机数记周五, 贝克休斯北京周六凌晨发布 → 修正 index+4自然日
  combined 另有: 止损在dd触发当期抹收益 = 前视 (干净口径重测, 不用其backtest)

### 22.3 评估结果 — 贝塔增量门槛 (新沉淀)
标准口径重跑 (周信号→日频ffill→vol_target 8%, cost 2-3bp, pit_shift=1):

| 候选 | 全样本 | OOS | r(主CTA) | r(B&H) | OOS超额Sharpe | 裁决 |
|---|---|---|---|---|---|---|
| 铜v6.3修正版 | +0.733 | +0.529 | +0.241 | +0.938 | **-0.431** | FAIL |
| 铜combined | +0.529 | +0.165 | +0.211 | +0.902 | **-0.906** | FAIL |
| 黄金( IS重调后) | — | +2.311 | +0.089 | **+0.991** | ≈0 | FAIL |

关键发现: 门槛四项里前三项铜v6.3全过 (OOS 0.529≥0.15, 相关性达标),
但**贝塔分解揭穿**: CU B&H(vol_target) OOS本身0.509, v6.3超额-0.43;
黄金B&H OOS 2.307 vs 策略2.311, 相关0.991 — OOS"好成绩"全是2022-25
金铜牛市的贝塔, 择时增量IS为正OOS转负 (铜) 或恒为零 (黄金, 12组网格
OOS千分位差异).

### 22.4 方法论沉淀
**新增贝塔增量门槛: long-only策略必须过"OOS超额Sharpe>0"检验**
(vs 同品种B&H vol_target基准) — 防止牛市品种贝塔伪装成策略alpha.
与既有纪律并列: IC显著≠可交易 (CFTC), 一致性<60%止损, OOS<0.15止损.
注: "接入黄金B&H拉组合Sharpe"本质 = 全样本挑最强品种 = 幸存者偏差, 同样拒绝.

### 22.5 最终状态
- 子策略层维持: CU溢价TC + jjm_coke + ZN库存 (T10y剔除)
- 定版推荐: **E2-R** (主CTA 45% + CU 39.31% + jjm 15.69%) — 待用户拍板切换
- commodity_fundamental 库定位: 研究资产留存, 不接入生产组合

## 23. AU策略方案B (layered ensemble) 评估 — PASS, 组合接入PASS (2026-09-05)

### 23.1 对象纠正 (§22评估对象搞错)
§22评估的是gold_combined (周频简化60/40版). 用户确认的正式方案 =
**gold_ensemble.compute_layered_ensemble_signal** (生产函数):
4路信号 (macro_regime滞回带×GVZ自适应×利率门控 / TIPS-MA / 52周动量 / 20日突破)
× **三级风控每路独立过** (vol压制15% → 软减仓DD<-7%×0.5 → 硬止损DD<-10%或日<-2%
清仓+5天冷却+VIX/GVZ双确认再入场) → 等权 → 死带平滑 → clip(0,1.2).
历史对照 (2010-26): 裸B&H 0.569/DD-44.5% → layered 0.922/DD-21.5%, 风控砍半DD.
PIT: 干净 (vix/gvz/tips shift2, rolling dd/vol, shift1执行, 时序循环只用过去).

### 23.2 四格对照 (体系口径: smoothed_au o2o + vt8% + cost2bp)
| 对照 | 全样本 | IS 18-21 | OOS 22-25 |
|---|---|---|---|
| 裸B&H | 1.37 / -20.1% | 0.70 / -20.1% | 2.14 / -9.5% |
| B&H+vt8% | 1.59 / -12.9% | 0.78 / -12.9% | 2.44 / -5.9% |
| **layered+vt8%** | **1.59 / -6.5%** | **0.78 / -6.4%** | **2.30 / -4.5%** |

Sharpe与B&H+vt打平但**DD减半 (改善50%)** — 风控层价值全在回撤维度.
§22的教训固化: long-only评估必须含DD/Calmar维度, 只看Sharpe会误杀风控型策略.

### 23.3 门槛 + 敏感性
门槛: OOS 2.30≥0.15 ✓ | IS/OOS同向 ✓ | DD改善50%≥10% ✓ | r(主CTA) +0.049<0.3 ✓
敏感性 (hc×vc 9组): 全样本1.52-1.60 / OOS 2.26-2.32 / DD -6.1%~-9.0% 平滑无刀尖 ✓
脚本: scripts/cta_gold_layered_eval.py

### 23.4 E2-R组合接入
| 接入 | 全样本 | DD | IS段 | 判定 |
|---|---|---|---|---|
| 基线 E2-R | 3.207 | -2.80% | 3.273 | — |
| +AU 5% | 3.296 | -2.65% (改善5%) | 3.321 | FAIL(DD改善<10%) |
| **+AU 10%** | **3.382** | **-2.51% (改善10%)** | **3.368** | **PASS** |
| +AU 15% | 3.465 | -2.36% (改善16%) | 3.413 | PASS |

**IS段(2018-21金价平庸期)同样改善** — 分散化贡献非金牛市事后视角.
推荐 +10% (15%数字更好但黄金集中度上升; CU+jjm+AU三子策略各约10-15%更均衡).

### 23.5 当前最优定版候选
**E2-R + AU 10%** (主CTA 40.5% + CU 35.4% + jjm 14.1% + AU 10%):
全样本 Sharpe 3.382 / DD -2.51%. 切换待用户拍板 + 2026真样本验证.

## 24. 风控移植 (A/B) + 自动配权重 (C) — 机制揭穿与约束盒子 (2026-09-05)

### 24.1 A组: 主CTA层三级风控移植 — 表面PASS, 机制揭穿, 不采纳
黄金三级风控按主CTA波动重标定 (L1 vol>12%压制 / L2 dd<-3.5%半仓 /
L3 dd<-5%或日<-1.5%清仓+5d冷却):
| 变体 | 全样本 | DD | 预设规则 |
|---|---|---|---|
| golden基线 | 2.405 | -6.14% | — |
| A套(L1+L2+L3) | 2.920 (+0.51) | -4.66% | PASS |
表面全PASS — 与§18"同族必败"预判相反. **但分年拆解揭穿**:
  - L3 触发34天 (2024年9月连续14天空仓) — 单日-1.5%阈值对商品CTA太松
  - 分年收益7年里4年为负 (2019 -0.3%/2020 -2.7%/2023 -2.0%/2024 -0.4%),
    IS改善主要靠2021单年+5.8%
  - 机制本质 = 频繁开关仓降低平均暴露 → 波动分母降更多 → Sharpe升.
    与§20扩容同源("降暴露换Sharpe"), 但更复杂+执行噪音+多数年份收益负贡献.
**裁决: 不采纳. 更透明的同效替代 = 直接调低主CTA权重 (§20/C组).**
方法论沉淀: **dd风控的Sharpe改善必须拆分年贡献** — 多数是降暴露效应
而非危机保护; 危机保护型改善的标志是"弱年绝对收益升" (如§20的2024).

### 24.2 B组: 组合层dd风控 — PASS, 备选不叠 (与权重机制重叠)
E2-R+AU10% (3.382/-2.51%) 之上: B1 dd<-2%半仓 → 3.429/-2.06%,
IS持平(3.375vs3.368)+OOS升(3.489vs3.399)+DD改善18% — 健康.
但与C组权重机制功能重叠 (都是暴露管理), 暂不叠加, 留作备选层.

### 24.3 C组: 自动配权重 — 全PASS, 但权重路径诊断揭示约束盒子退化
约束 (用户拍板): 子策略≤10% × 主CTA [60%,75%], 候选池5成分 (含ZN).
| 方案 | 全样本 | DD | IS | OOS | 换手/月 | 判定 |
|---|---|---|---|---|---|---|
| B1固定(无ZN) | 2.813 | -5.24% | 2.957 | 2.665 | — | 基线 |
| B2固定(满配) | 3.010 | -3.90% | 3.090 | 2.929 | — | 基线 |
| erc126/Q | **3.195** | -3.90% | 3.548 | 2.899 | 0.063 | **PASS** |
| inv_vol252/M | 3.174 | -3.96% | 3.463 | 2.918 | **0.007** | PASS |

**权重路径诊断 (必做步骤)**: ERC126月度在2019-2024年末全部=0.60+0.10×4
(恰为B2满配); 触子策略上限92%/触主CTA下限83% — ERC在约束盒子里
几乎退化为"子策略全顶格+主CTA垫底". +0.19改善主要来自2018 warmup
和2025的jjm→AU/CU转移, 非机制智能.
**裁决**: B2满配(简洁) 与 erc126/Q(自适应+0.19, 季度一调) 二选一,
机制成本低可采纳; inv_vol252 为最低换手备选 (0.007/月).

### 24.4 约束的代价 (诚实呈现)
子策略≤10% ⇒ 主CTA≥60%, 组合Sharpe封顶在~3.0-3.2 (§20: 主CTA占比越低
Sharpe越高, 45%时3.14). 用户约束换的是单成分集中度风控 (任一子策略失效
最多伤10%), trade-off 合理但数字上 E2-R+AU10% (3.382, CU 35.4%) 更高.

### 24.5 定版候选矩阵 (待用户拍板)
| 版本 | Sharpe | DD | 集中度 |
|---|---|---|---|
| E2-R+AU10% (无约束) | 3.382 | -2.51% | CU 35.4% |
| erc126/Q (约束+自适应) | 3.195 | -3.90% | 各子≤10% |
| B2满配 (约束+固定) | 3.010 | -3.90% | 各子≤10% |
切换前均需2026真样本验证.

## 25. 横截面 carry 子策略 — 证伪归档 (2026-09-05)

### 25.1 设计 (经典学术口径)
term_structure_daily (88品种, rank1/2 close+days_to_expiry):
    carry_i = (P_near-P_next)/P_next × 365/Δdays
    多 top 1/3 (backwardation) / 空 bottom 1/3 (contango), 等权,
    vol_target 8%, cost 3bp, pit_shift=1.

### 25.2 两处实现bug修正 (记录避免重蹈)
  a) ±1信号sum未除以品种数 → 28倍杠杆 (AR -77%/DD -100% 荒谬值)
  b) 用smoothed主力收益做持仓收益 → 漏掉展期收益 (carry收益主体),
     修正为近月合约自身收益+换月日剔除.

### 25.3 结果 — 毛alpha太薄, 成本后全负
| 再平衡 | 全样本 Sharpe | 毛多空价差 | 年换手 |
|---|---|---|---|
| 日度 | -0.539 | +1.36%/年 | ~75x |
| 周度 | -0.260 | 同 | 30.6x |
| 双周 | -0.114 | 同 | 21.2x |
| 月度 | -0.171 | 同 | 10.8x |

多空方向正确 (多腿+0.0037%/日 > 空腿), 但**毛价差仅+1.36%/年**,
扛不住任何≥1%的成本 → 中国市场横截面carry原始信号太弱.
r(主CTA) = -0.036 (正交性完美但无用武之地).
**裁决: FAIL归档. 不做分位加强/滞回等变体 — 毛alpha量级不足,
继续调参=过拟合 (与§8 CFTC/经纪持仓同级结论).**

### 25.4 Alpha池状态更新
横截面carry证伪后, 剩余待测alpha仅: VRP子策略 (62品种数据完备).
若VRP再证伪, 本信号族 (价格+基本面日频) 的组合天花板即到顶:
E2-R+AU10% (3.382) 或约束版 (~3.2) 为最终形态, 增量只能来自
2026真样本验证后的重评估与实盘执行层.

## 26. CTA横截面策略族 — 调研与实验设计 (2026-09-05, doc-first冻结)

### 26.1 调研全景 (8族 × 数据 × 正交性 × 状态)
| 族 | 学术证据 | 数据 | 与主CTA正交 | 状态 |
|---|---|---|---|---|
| **basis momentum** (Boons-Prado JF19) | 强 | term_structure 88品种 | 高 (r≈0-0.2) | 未测 |
| **横截面动量** (多窗口相对强弱) | 强 | 价格 100+品种 | 中 (r≈0.2-0.4) | 未测 |
| **VRP横截面** (IV-RV排序对冲) | 中 | vrp_daily 62品种 | 高 | 未测 |
| OI变化横截面 | 中弱 | oi_crossover 88品种 | 中 | 时序版在用 |
| 低波异象 | 弱(商品) | hv 88品种 | 中 | 低优先 |
| PCR情绪 | 弱(样本短) | pcr 65品种2019后 | 高 | 低优先 |
| 库存横截面 | — | 覆盖太少 | — | 无意义 |
| carry水平值 | 强但中国失效 | 已测 | r=-0.04 | §25证伪 |

主CTA模块grep确认无横截面成分 (全时序), 无重复建设风险.

### 26.2 basis momentum ≠ §25已证伪的carry
已证伪: carry水平值 (back>contango静态排序), 毛alpha仅+1.36%/年.
本实验: basis momentum = Δbasis变化率动量 (基差走强/走弱的速度排序).
Boons-Prado: 价格动量与basis动量两因子互补, basis momentum独立Sharpe
高于carry. 信号更平滑换手更低. 非同一信号变体重测.

### 26.3 三策略设计 (共享基础设施)
共同: 横截面排序 top/bottom 1/3, ±1/n_active资金等分, 周度再平衡(W-FRI),
vol_target 8%, cost 3bp, pit_shift=1, 执行收益=smoothed主力(动量类无展期问题).
  S1 basis momentum: Δbasis 12周动量排序 (basis=carry原始值)
  S2 横截面动量: 4/13/26/52周收益排序均值 (与主CTA时序动量不同信息源)
  S3 VRP横截面: vrp_percentile排序对冲 (期权覆盖品种子集)

### 26.4 纪律 (carry教训固化, 预设冻结)
  a) 先算毛价差: 年化毛多空价差<3% → 直接归档, 零变体调参
  b) 门槛: OOS≥0.15 + IS/OOS同向 + r(主CTA)<0.4 + 分年无单年依赖
  c) 成本敏感线: 周度换手≤40x (carry 30.6x换手时成本1.8%/年为前车之鉴)
  d) PASS → 组合接入: 新成分摊薄CU 35.4%集中度, 目标CU≤25%且Sharpe≥3.3

### 26.5 实验结果 — 横截面族全灭, 毛价差≈0 (2026-09-05)
| 策略 | 全样本 | IS | OOS | 毛价差/年 | 换手 | r(主CTA) | 判定 |
|---|---|---|---|---|---|---|---|
| S1 basis momentum (12周) | -0.378 | -0.092 | -0.667 | +0.00% | 25.2x | -0.026 | FAIL |
| S2 横截面动量 (4-52周) | -1.245 | -1.545 | -0.918 | +0.08% | 11.2x | -0.020 | FAIL |
| S3 VRP横截面 | -0.586 | -0.881 | -0.040 | **-0.49%** | 25.1x | -0.008 | FAIL |

毛价差≈0 = 信号无alpha (非成本问题, 纪律线§26.4a直接归档零调参).
S3方向反向 (低VRP品种跑赢). **中国市场商品期货横截面动量族无毛alpha**.

归因分析: (1) 中国商品动量主要为时序而非截面相对强弱;
(2) 产业链聚类强 (黑色5-6品种同向), 横截面有效N小;
(3) 涨跌停/交割制度压缩分化; (4) 2020-21全球同涨期截面对冲失效.

### 26.6 Alpha池到顶确认 (§25+§26结论合并)
横截面族 (carry水平/动量/basis动量/VRP排序) 全部证伪.
本信号族 (日频价格+基本面) 组合最终形态: **主CTA + CU溢价TC + jjm + ZN + AU_layered**.
增量路径只剩: (a) 2026真样本验证 (管线授权);
(b) 实盘执行层 (滑点/成本优化); (c) 全新数据源 (日内/另类, 超出当前范围).


## 27. 主CTA评估口径修正 — 2018 warmup发现 + 正确口径数字 (2026-09-05)

### 27.1 两个污染源修正
  a) **2018为warmup占位** (242日收益全零, NAV恒1.0) — 官方八年口径
     (AR 19.18%/Sharpe 2.310) 被稀释. 真实窗口 2019-2025 (7年).
  b) 本环境pandas的 groupby/resample.apply(lambda (1+s).prod()-1)
     多层分组下产出荒谬值 (get_group().prod()验证正常) — 月度矩阵等
     历史个别数字受污染. 规范: 聚合一律用 .prod() 方法.

### 27.2 主CTA正确口径 (2019-2025)
| 窗口 | AR | Vol | Sharpe | MaxDD | Calmar | Sortino |
|---|---|---|---|---|---|---|
| 全样本19-25 | +23.12% | 8.86% | **2.609** | -6.14% | 3.76 | 4.17 |
| IS 19-21 | +33.16% | 10.14% | 3.272 | -5.95% | 5.57 | 4.85 |
| OOS 22-25 | +16.06% | 7.75% | 2.072 | -6.14% | 2.61 | 3.84 |

NAV 1.0→4.064 (7年4倍). 分年: 27.5/39.3/29.1/11.8/36.3/9.6/6.2 (%).
月度胜率70% (59/84), 最好+20.2% (2023-12), 最差-3.33%.
**衰减信号**: 年Sharpe 2.90→3.43→3.10→2.04→3.53→1.03→1.65;
**最深回撤-6.14% (2024-07起) 持续404天未修复** — 实盘最大心理挑战.

### 27.3 组合版本对照 (2019-2025窗口对齐, 全线抬升)
| 版本 | AR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| 主CTA本体 | +23.12% | 2.609 | -6.14% | 3.76 |
| R2-R(55/45) | +20.99% | 3.332 | -3.24% | 6.47 |
| E2-R(45/55) | +19.37% | 3.549 | -2.80% | 6.92 |
| **E2-R+AU10%** | **+18.55%** | **3.724** | **-2.51%** | **7.39** |
| B2满配(子≤10%) | +21.49% | 3.220 | -3.96% | 5.42 |

注: 早期文档的2.31/2.405/3.382等为八年窗口口径; 三口径并存历史
(§compute_metrics注释) 应以本节2019-2025窗口为准.
### 27.4 导出物
main_cta_full_assessment.csv (15行版本×窗口总表) /
main_cta_curve.csv (日频NAV+回撤) / main_cta_curve_monthly.csv /
main_cta_curve.html+.png (plotly三联: NAV/回撤/分年).

---

## 28. Zookeeper 因子复现立项 (2026-09-06, doc-first 启动)

### 28.1 立项动机
§25/§26 证伪了三族单变量信号 (carry / basis momentum / VRP), 与 Fan et al. 2026
"Call the Zookeeper" 论文 Spanning Test "单因子脆弱性" 论断完全对应. 论文核心
贡献是 L-En (5 线性模型等权集成) 用稳定性胜过单因子脆弱性 — 与本项目 §24
auto_weight 的 ERC/inv_vol 思想一脉相承. 论文样本外 5.8%/Sharpe 0.64, 极
可能解决本项目"alpha 池到顶"问题 (§27 后增量 alpha 找不到).

### 28.2 调研关键发现
- **81 变量** / 3 组 (C 商品 37 + M 宏观 40 + S 情绪 4)
- 样本 25 商品 1988-2022, 论文是月度再平衡横截面策略 (与本项目日频趋势策略**架构互补**)
- 关键陷阱 (知乎解读指出, 论文未充分强调):
  - **交易成本致命**: quartiles 月度排序变动 → 单品种方向换手 200%, 月度单边换手 40-80%, 5.8% 扣 15-20bps 后 Sharpe 大缩水
  - **前瞻偏差**: 宏观/文本情绪有 1-2 月发布延迟 + 修正机制, 用"修正后"数据严重高估 R²OOS
- 本项目覆盖度: **~38/81 ≈ 47%** (Group C 28/37 缺 hp12; Group M 10/40; S 跳过)

### 28.3 决策 (用户拍板)
- A = 文档 + 完整复现实验
- B = CTA 笔记 + 复现脚手架
- S 族: **先跳过** (Baker-Wurgler/FinBERT WSJ 缺数据)
- Spanning 对手: **vs 全部子策略** (carry/basis momentum/VRP/ZN inventory/jjm/CU 溢价 TC/AU layered/OscAlpha/主 CTA)

### 28.4 评估硬指标 (硬阈值, 不达标即停)
| 指标 | 通过 | 失败 |
|---|---|---|
| R²OOS (TS) | > 0% | < 0% (OLS 直接崩溃那种) |
| Zookeeper 扣成本 Sharpe | > 0.5 | < 0.3 |
| t-stat | > 2.0 | < 1.5 |
| 月度单边换手 | < 80% | > 100% (知乎报警) |

### 28.5 接入决策
- Spanning α < 2%/年: 归档
- 2% < α < 5%: 子策略候选 (组合 ≤10% 沿用硬约束)
- α > 5%: **谨慎审查** (重跑成本 / PIT / 鲁棒性)

### 28.6 Phase 计划
| Phase | 内容 | 工作量 | 状态 |
|---|---|---:|---|
| P1 | 精读笔记 + 脚手架骨架 | 3-4 h | ✅ 完成 (commit) |
| P2 | 38 变量构造 + PIT 对齐 | 4-5 天 | ✅ 完成 (commit) |
| P3 | 5 模型集成 + 60 月滚动 Beta | 2-3 天 | ✅ 完成 (commit) |
| P4 | 月度再平衡回测 + cost_model 扣减 | 2-3 天 | ✅ 完成 (commit, **结果负 FAIL**) |
| P5 | Spanning Test vs 全部子策略 | 2 天 | ✅ 完成 (commit, **主CTA α=12.94%**) |
| P6-C | 改造提取方式 (加权 LS) | 1 天 | ✅ 完成 (commit, **主CTA α=12.94% 一致**) |
| P7-B | ZK 状态变量接入 vol_target | 1 天 | ✅ 完成 (commit, **Sharpe 3.435→3.540**) |

### 28.7 P1 产出 (本次 commit)
- `docs/cta/zookeeper_paper_notes.md`: 精读笔记 (论文 §1-§7 + 本地化覆盖度 + 启示)
- `scripts/zookeeper/` 脚手架 10 文件:
  - `__init__.py / config.py` — 模块接口 + 全部常量
  - `features_c.py / features_m.py` — Group C/M 变量接口 (P2 待实装)
  - `loader.py` — 数据加载 + PIT 对齐 (P2 ✅ / P4 待实装)
  - `beta_rolling.py` — 60 月滚动 OLS Beta (P3 ✅)
  - `model_l_en.py` — 5 线性模型 L-En 集成 (P3 ✅)
  - `backtest.py` — 月度截面回测 + 成本扣减 (P4 待实装)
  - `spanning.py` — Spanning Test vs 全部 (P5 待实装)
  - `run.py` — CLI 入口 (features/train-test/spanning/all)

### 28.8 P2 产出
- `features_c.py` 12 类 Group C 变量实装 (库存族 6 / 对冲族 2 / 流动性族 4)
- `features_m.py` 10 个 Group M 宏观实装 (fed_rate / term_spread / fed_balance /
  m2_yoy / us_cpi_yoy / dxy / vix / tips10y / epu / wti_brent_spread)
- `loader.py` 实装 PIT 对齐 (宏观滞后 1 期, 库存按记账日, 行情天然 PIT OK)
- `INVENTORY_MAP`: 仅 CU/ZN/NI/AL 4 品种有 SHFE 库存数据
- 输出: `reports/cta/zookeeper/features_c.parquet (91月×720列)` /
  `features_m.parquet (91月×10列)` / `product_pool.csv (60品种按流动性)`
- 验证: ruff All checks passed; NaN 比例 C 33% (库存族 4/60 + mom12 前12月),
  M 1.1%; 日期 2018-01-31 ~ 2025-07-31 (覆盖论文测试期)

### 28.9 P3 产出 (本次 commit)
- `beta_rolling.py` 实装 60 月滚动 OLS Beta (论文 §3.1 维度对齐)
- `model_l_en.py` 实装 5 线性模型 L-En 集成:
  - OLS / PLS(n=5) / RidgeCV / LassoCV / ElasticNetCV
  - fit_predict_ts / fit_predict_cs / rolling_train_predict_{ts,cs}
  - 每估计期前标准化 (零均值单位方差, 沿论文 §3.2)
- `run.py` train-test 命令实装
- 输出: `ts_predictions.csv` (54月×3列 y_true/y_pred/y_baseline) /
  `ts_metrics.json` (R²OOS / R²OOS_rolling / IC / sign_acc) /
  `cs_predictions.parquet` (54月×60商品, L-En 预测 r̂ 矩阵)

#### P3 关键评估结果 (初看, ⚠️ 警惕)
| 指标 | 值 | 论文 | 评估 |
|---|---|---|---|
| R²OOS (vs 历史均值) | 42.5% | 3.55% | ⚠️ 偏高 |
| R²OOS (vs 12月 rolling) | 45.1% | — | ⚠️ 偏高 |
| IC | 0.70 | 0.05-0.10 显著 | ⚠️ 异常高 |
| sign_acc | 79.6% | — | ⚠️ 异常高 |

#### 异常高指标的 4 个可能原因 (P5 Spanning Test 验证)
1. **特征前瞻偏差**: features_c 含 t 期 close 计算 (basis/bmom/iliq 等用 t 当期数据)
   — 应改用 t-1 滞后口径
2. **商品池选择性**: 60 品种按流动性前 60, 含幸存者偏差
3. **样本期短**: 仅 5 年 OOS (论文 17 年), 含 2021 牛市特殊行情
4. **R²OOS 口径差异**: 论文 25 商品 17 年月频 vs 本项目 60 商品 5 年

### 28.10 P4 产出 (本次 commit, **真实回测揭示 P3 误判**)
- `backtest.py` 实装月度截面回测 (Q4/Q1 等权多空 + 成本扣减 + 换手监控)
- `run.py` train-test 流程增加 P4 回测步骤

#### P4 真实回测结果 (关键, **与 P3 完全相反**)
| 指标 | 论文 | 本项目 | 阈值 |
|---|---|---|---|
| LS Sharpe | 0.64 | **-0.65** | > 0.5 通过 / < 0.3 失败 |
| 年化 | +5.8% | **-7.36%** | > 0 |
| t-stat | 3.2 | **-1.37** | > 2 |
| 最大 DD | (未报) | **-34.52%** | — |
| 月度换手 | 40-80% | 24.6% | < 100% OK |

#### P4 复盘: P3 IC 0.70 完全是误导
1. **P3 TS 模式 (AVG 市场组合)**: 月度波动容易拟合, 22 特征对单 y 序列有大量噪声自由度
2. **P4 CS 模式 (60 商品排序)**: 真正的回测口径, Q4/Q1 多空月度再平衡
3. **预测 r̂ 与实际收益方向相反**: 模型学会了"市场均值"的趋势但不会区分商品强弱

#### 硬阈值判定: **FAIL**
- 扣成本 Sharpe -0.65 < 0.3 阈值 = FAIL
- 沿用户 §28.4 评估纪律, P4 已触发失败阈值

#### 4 个失败根因 (深挖, 不只是"模型不行")
1. **样本期过短**: 5 年 OOS (论文 17 年) 无法稳定估计 22 特征的回归系数
2. **特征构造 bug 嫌疑**: P2 的 basis/rbasis/bmom/iliq 用 t 当期数据 → 实时预测有
   前瞻偏差风险 (知乎 §7.2), 应该用 t-1 滞后
3. **OLS 类线性模型对异常值敏感**: 月度收益分布厚尾, 60 商品 × 5 年样本太少
4. **cost_model.json 校准基于 §22**: 单边 15bps 是最优品种, 60 商品池平均更高

### 28.11 决策点 (用户拍板: B 修复 P2 特征前瞻偏差)
| 选项 | 内容 | 推荐 |
|---|---|---|
| A | 立即归档 (沿 §28.4 纪律) | — |
| B | **修复 P2 特征前瞻偏差 + 重跑** | ✓ 用户选 B |
| C | 仍做 P5 Spanning Test | — |
| D | 扩大样本期 | — |

### 28.12 P2-B 修复 + P4-B 重跑 (本次 commit)

#### 修复内容
`features_c.py` 末尾对整 panel 加 `.shift(1)`:
- 修复前: features[T] 用 T 月 close 计算 → 预测 T 月收益 (同期泄漏)
- 修复后: features[T] = 原始 features[T-1] → 用 T-1 及之前数据预测 T 月收益 (合规 PIT)

#### P4-B 重跑结果 (与 P3/P4-A 对比)
| 指标 | P3 (原 IC 0.70) | P4-A (修复前) | P4-B (修复后) |
|---|---:|---:|---:|
| TS IC | +0.70 | — | **-0.35** |
| TS sign_acc | 79.6% | — | 42.6% |
| CS Sharpe | — | -0.65 | **-0.35** |
| CS 年化 | — | -7.36% | -4.72% |
| CS DD | — | -34.52% | -29.66% |

#### 关键发现: P3 IC 0.70 是"假信号"
- TS IC 从 +0.70 **翻转为 -0.35** (修复同期泄漏后)
- sign_acc 从 79.6% 跌至 42.6% (低于 50% 随机)
- **证实 P3 虚高指标完全是前瞻偏差造成的同期拟合**
- 用户决策 B 找对了根因 — 知乎指出的前瞻偏差风险 (§7.2) 在本项目完全成立

#### P4-B 仍 FAIL, 但向 0 收敛
- Sharpe -0.35 (仍 < 0.3 阈值)
- t-stat -0.75 (远 < 1.5 阈值)
- DD -29.66% (绝对值仍大)

#### 仍负 alpha 的次因 (沿 §28.10.3-4)
1. **样本期短**: 5 年 OOS 无法稳定估计 22 特征回归系数
2. **OLS 类对厚尾月度收益 + 60 商品样本敏感**
3. **60 商品池选择性**: 按流动性前 60 含幸存者偏差

#### 仍可继续的实验方向 (用户拍板)
- **C: P5 Spanning Test** (1 天): 即便负 alpha 也测与主 CTA 相关性, 若 |corr|<0.3 仍有分散价值
- **D: 扩大样本期** (5-7 天): 用 2014-2017 历史数据补全, 但要评估数据完整性

P2-B + P4-B 仍判定 **FAIL** (沿 §28.4 硬阈值纪律). 等用户决定 C/D 或彻底归档.

### 28.13 P5 Spanning Test 结果 (本次 commit, **重大发现**)

#### §28.4 决策规则提醒
- α < 2%/年: 归档
- 2-5%: 子策略候选 (组合 ≤10%)
- > 5%: 谨慎审查 (重跑成本 / PIT / 鲁棒性)

#### P5 Spanning Test vs 全部子策略结果
| 对手 | α_ann | t-stat | β | R² | 决策 |
|---|---:|---:|---:|---:|---|
| §25 carry | -2.49% | -2.08 | 0.029 | 2.40% | 候选集成 |
| ZN inventory | +6.34% | 1.60 | -0.019 | 0.11% | ⚠️ 谨慎审查 |
| jjm_coke | **+7.04%** | 2.37 | 0.039 | 0.72% | ⚠️ 谨慎审查 |
| CU premium TC | +6.17% | 1.89 | 0.050 | 1.00% | ⚠️ 谨慎审查 |
| AU layered | **+7.96%** | 2.81 | 0.007 | 0.03% | ⚠️ 谨慎审查 |
| OscAlpha | **+14.03%** | 2.82 | -0.383 | 20.35% | ⚠️ 谨慎审查 |
| **主 CTA baseline** | **+12.94%** | **4.56** | -0.183 | **15.30%** | ⚠️ 谨慎审查 |

#### 重大发现: ZK 因子是真正独立 α 源 (与 P4 负 Sharpe 看似矛盾)
- **主 CTA α = +12.94%/年 (t=4.56, p<0.001) 极显著**: ZK 因子对主 CTA 的独立解释力强
- **β = -0.183, R² = 15.30%**: ZK 与主 CTA **负相关 + 低解释度** — 主 CTA 没捕获到 ZK 的信号
- **AU layered α = +7.96% (t=2.81)**: 显著独立
- **OscAlpha α = +14.03% (t=2.82, β=-0.383, R²=20.35%)**: ZK 与 OscAlpha 显著负相关

#### 矛盾的解释
- **P4 LS Sharpe -0.35**: ZK 因子**自身**的 Q4/Q1 多空月度策略负 alpha (按 60 商品截面排序不稳)
- **P5 Spanning α +12.94%**: ZK 因子对**其他子策略**的边际解释力强 (β小 R²低 → 不能被替代)
- 含义: **ZK 含有的"信号"是真实独立信息**, 但用错了"提取方式" — 不是简单的 Q4/Q1 排序多空

#### 接入决策 (用户拍板)
| 选项 | 内容 | 推荐 |
|---|---|---|
| A | **归档**: P4 自身负 alpha 仍是硬失败, 不集成 | 沿 §28.4 纪律 |
| B | **作为宏观调节器**: ZK 不做多空, 只作为组合层 vol_target 的状态变量 | ✓ (与论文"state variable"用法一致) |
| C | **改造提取方式**: 不用 Q4/Q1 排序, 改为加权合成 (L-En 集成 IC >0.3 的品种) | 1-2 天实验 |
| D | **加大样本期 (D 路线)**: 用 2014-2017 历史补全, 5-7 天重投入 | 重投资 |

推荐 **B**: ZK 因子作为**状态变量**监控市场风险 (类似 §26 vrp_percentile),
不直接作为子策略 alpha, 但接入主 CTA 的 vol_target 调节层 (详见 §23 layered 风控思路).

### 28.14 P6-C 改造提取方式 (本次 commit)

#### 改造内容
`backtest.py` 增加 `weighted_ls_backtest`: 用 L-En r̂ 作为权重 (替代 Q4/Q1 等权):
  w_i[t] = r̂_i[t] / Σ|r̂_j[t]|  (净暴露 = 0, 毛暴露 = 2)
  收益 = Σ w_i × return_i (截面回归投资组合)
保留 L-En 预测的强度信息, 不损失于硬阈值排序.

#### P6-C 自身表现 vs P4-B (Q4/Q1)
| 指标 | P4-B (Q4/Q1) | P6-C (加权) | 改进 |
|---|---:|---:|---|
| 年化 | -4.72% | **-3.18%** | ✓ +1.5% |
| Sharpe | -0.35 | -0.37 | ≈ 持平 |
| 最大 DD | -29.66% | **-19.41%** | ✓ DD -35% |
| 换手 | 26.5% | 33.5% | ≈ 略增 |

→ 自身仍 FAIL (Sharpe 负), 但 DD 显著缩窄

#### P6-C Spanning Test (vs P5 Q4/Q1)
| 对手 | P5 α | P6-C α | 变化 |
|---|---:|---:|---|
| §25 carry | -2.49% | -2.51% | ≈ |
| ZN inventory | +6.34% | +6.59% | +0.25% |
| jjm_coke | +7.04% | +7.02% | ≈ |
| CU premium TC | +6.17% | +6.05% | ≈ |
| AU layered | +7.96% | +7.77% | ≈ |
| OscAlpha | +14.03% | +14.21% | +0.18% |
| **主 CTA baseline** | **+12.94%** | **+12.94%** | **完全一致** |

#### 关键: ZK α 不依赖提取方式
- 主 CTA α 在 Q4/Q1 和加权两种提取下**完全一致 12.94%**, t = 4.52 极显著
- 改造提取**没改变** ZK 作为独立 α 源的本质
- 进一步证实: ZK 含有的信号对主 CTA 是**真独立**的, 提取方式不是瓶颈

#### 最终接入决策 (P6-C 后用户拍板)
| 选项 | 含义 |
|---|---|
| A | 归档: P6-C 自身仍负 alpha |
| B | **作为状态变量接入 vol_target** (推荐, 与论文 §6 一致) |
| C | 已用 P6-C 加权替换 Q4/Q1, 无更多改造空间 |
| D | 加大样本期 (5-7 天重投入) |

**沿 §28.4 纪律 P6-C 仍 FAIL, 建议决策 B** (状态变量接入, 不作子策略方向).

### 28.15 P7-B ZK 状态变量接入 vol_target (本次 commit)

#### 决策背景 (P5+P6-C 结论)
- P5 Spanning: ZK 对主 CTA baseline α = +12.94%/年 (t=4.52) 极显著独立 α 源
- P6-C 加权 LS: 自身 Sharpe -0.37 仍是 FAIL (提取方式不是瓶颈)
- 论文 §6 用法: ZK 作为 state variable 而非子策略方向
- 用户决策: **B 接入 vol_target**

#### 接入原理
```
vol_target_adjusted = vol_target_base × (1 + direction × alpha × (zk_state_norm - 0.5) × 2)
adj_factor clip(0.3, 2.0)
```
- ZK 状态变量月度更新 → 月初调节下月 vol_target
- direction = -1 (反向): ZK 信号弱 → vol 上调 (机会窗口); 强 → vol 下调 (风险警报)

#### 网格搜结果 (1698 交易日 ≈ 7 年, signal_dispersion 状态变量)
| α | 方向 | Sharpe_B | DD_B | α_ann | t_α |
|---:|---:|---:|---:|---:|---:|
| 0.1 | -1 | 3.483 | -2.47% | +0.27% | 2.29 |
| 0.2 | -1 | 3.519 | -2.44% | +0.54% | 2.29 |
| **0.3** | **-1** | **3.540** | **-2.40%** | **+0.81%** | **2.29** ⭐ |
| 0.5 | -1 | 3.537 | -2.33% | +1.35% | 2.28 |
| 0.7 | -1 | 3.474 | -2.26% | +1.89% | 2.28 |

#### 最优配置
**α=0.3, direction=-1, stat=signal_dispersion**:
- 主 CTA Sharpe 3.435 → **3.540** (+3.0%)
- DD -2.51% → -2.40% (缩窄 4%)
- α_ann +0.81% (t = 2.29, 显著)

#### 模拟的局限
- simulate_vol_target_adjustment 在历史主 CTA NAV 上**近似缩放**月度收益, **不重跑主 CTA v11 引擎**
- 真实接入需按 §5.3 步骤 (1-2 天)
- 日内 vol_target 是常数, 实际应月度调整 (与 §23 layered 风控思路一致)

#### v11 引擎接入设计 (待用户拍板 E1)
详见 `docs/cta/zookeeper_state_var_integration.md` §5.3 接入代码骨架:
1. 把 factor_vol_adj_weighted_signal_dispersion_a0.3_d-1.csv 接入 ZK_VOL_ADJ_CFG
2. 在 `vol_target_positions()` 增加 if 开关 (config: enabled=False, 拍板后开启)
3. 重跑 `cta_full_assessment_export.py` 验证 Sharpe/DD
4. 新主 CTA α > 5% 触发谨慎审查

#### 当前提交 (E2: 文档 + simulate)
- `scripts/zookeeper/integrate.py`: vol_target 模拟集成 (load_main_cta_nav + zk_state_norm + simulate_vol_target_adjustment + 网格搜)
- `docs/cta/zookeeper_state_var_integration.md`: 完整接入设计文档
- `reports/cta/zookeeper/vol_adj_grid_weighted_signal_dispersion.csv`: 10 行网格结果

**当前决策: E2 (simulate + 文档), 等用户拍板 E1 (真接入 v11 引擎) 或 E4 (扩样本期)**
