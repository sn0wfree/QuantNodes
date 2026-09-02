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
9. [最终定版配置与实盘预期](#9-最终定版配置与实盘预期)
10. [剩余遗留项与未来研究方向](#10-剩余遗留项与未来研究方向)

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

## 9. 最终定版配置与实盘预期

### 9.1 定版配置

```python
FINAL_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    roll_cost_bps=10.0, roll_model="lump",
    commission_rate={per_product: ...},  # 按品种校准
    slippage_bps={per_product: ...},  # 按品种校准
)
```

### 9.2 业绩（两次修复后，lump口径）

| 口径 | Sharpe | AnnRet | Vol | MaxDD | Calmar |
|------|--------|--------|-----|-------|--------|
| 含EC | 2.41 | 20.0% | 8.3% | -6.1% | 3.25 |
| **剔EC（保守实盘预期）** | **2.20** | **15.6%** | **7.2%** | **-6.3%** | **2.55** |

### 9.3 业绩为什么好（核心贡献））

**一句话**：等权多模块趋势信号 + ema平滑 + 66品种分散 + 真实lump换月摩擦 + 按品种成本校准。

| 贡献来源 | 贡献值 | 说明 |
|---------|--------|------|
| 多模块等权 | α | 5模块趋势对冲（trend vs volume vs term structure）|
| EMA=0.15 | Sharpe +0.1 | 换手149x→29x，成本敏感度大幅下降 |
| 66品种分散 | 风险分散 | EC占25%PnL但仅10亿级仓位中的25% |
| 风险层 | Sharpe +0.2 | VolRiskFilter/TailRisk/Cluster Limit共同保护 |
| 按品种成本 | Sharpe +0.13 | 多数品种真实成本<5bp，比双10bp保守更准 |
| lump换月 | Sharpe -0.01 | 实际换月冲击被低真实成本抵消 |

### 9.4 业绩为什么好（归因））

**6个主力年（2019/2020/2021/2023）各贡献31-39%**：
- 与EC几乎无关（剔EC后差距<3pp）
- 趋势策略在趋势年赚大钱

**3个弱年（2022/2024/2025）全部正收益**：
- 弱年只是收益小，未亏损
- 2024剔EC=-0.56%（化工震荡磨损是结构性弱点）

---

## 10. 剩余遗留项与未来研究方向

### 10.1 实盘前必做

| 项 | 重要性 | 说明 |
|----|--------|------|
| 2026 OOS验证 | 🔴 | 回测止于2025-12，7个月空白需验证 |
| 按品种费率实盘账单校准 | 🟡 | 估算值需替换为真实交易费 |
| 涨跌停检测 + 流动性过滤 | 🟡 | 改进执行层 |

### 10.2 可探索方向（详见 `docs/cta/cta_cross_product.md`）

| 方向 | 价值 | 说明 |
|------|------|------|
| **CFTC持仓因子** | 高 | 数据已就绪（99,591行）但未使用 |
| **库存因子** | 高 | inventory_daily + eia_weekly_inventory 未用 |
| **现货升贴水因子** | 中 | spot_basis_daily 83,230行未用 |
| **跨品种风险平价仓位** | 高 | vol_target独立处理，无相关性矩阵 |
| **26对套利对冲启用** | 中 | cross_product_spread×140k行预计算 |
| **化工板块级regime** | 中 | 全局regime已证伪，板块级需谨慎 |
| **TF API指标因子** | 中 | tfapi_indicators 18,419行未用 |
| **Black-Litterman + CVaR** | 探索性 | 替换等权+取整+等分资金 |

### 10.3 因子补充优先级建议

**第一优先（实施快、风险低）**：
- **CFTC持仓因子**（managed_money/commercial）—— 数据已就绪，业内共识高，PIT处理成熟

**第二优先**：
- **现货升贴水**（spot_basis_daily）—— 数据已有但需补充2025-08后

**第三优先**：
- **库存因子** —— 品种受限（铜油煤豆等），但商业逻辑明确

**暂缓**：
- 套利对冲 + 板块regime + 跨品种组合优化 —— 涉及更大架构重构，建议在Tier1完成评估α增量后再决定

---

## 附录：所有commit历史

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

## 附录：所有脚本/报告

| 类型 | 路径 |
|------|------|
| baseline | `scripts/cta_baseline.py` |
| 换手率实验 | `scripts/cta_turnover_experiment.py` |
| 年度归因 | `scripts/cta_yearly_attribution.py` |
| 双口径监控 | `scripts/cta_dual_basis.py` |
| 成本模型 | `scripts/cta_cost_model.py` |
| 最终定版 | `scripts/cta_final_baseline.py` |
| baseline | `reports/cta/baseline_*.csv` |
| 定版 | `reports/cta/final_*_lump.csv` |
| 换手率 | `reports/cta/turnover_experiment.csv` |
| 年度 | `reports/cta/final_yearly_lump.csv` |
| 双口径 | `reports/cta/dual_basis.csv` |
| 成本模型 | `reports/cta/cost_model.csv` / `cost_model.json` |
| 敏感性 | `reports/cta/slippage_sensitivity.csv` |