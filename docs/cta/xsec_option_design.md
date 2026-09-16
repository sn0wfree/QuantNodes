# CTA 截面动量 × 期权持仓组合策略设计文档

> 最后更新: 2026-09-16
> 状态: 设计定稿（Stage 0 实现中）
> 模块路径: `scripts/cta_xsec_option/`
> 姊妹文档: `docs/cta/cusum_prealert_design.md`（突变点检测, 独立工作流）

---

## 一、策略核心思想

### 1.1 与传统截面策略的区别

传统截面动量：直接对期货价格截面排序，多强空弱。
**本策略**：CTA 先出一版截面信号 → 期权层利用该信号决定持仓方式与大小 → 双层叠加。

```
┌─────────────────────────────────────────────────────────────┐
│ L1: CTA 截面动量信号（§26 S2 改进版）                        │
│     输出: cta_rank ∈ {-1, 0, +1}（品种截面排名, 日频）        │
│     数据: 期货 close                                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ L2: 期权持仓执行层                                           │
│     输入: cta_rank + oi_confirm（期权 OI 确认系数）           │
│     方向: CTA 信号决定（rank>0 买 call, rank<0 买 put）       │
│     大小: OI 确认系数决定（0.5~1.5 倍缩放）                   │
│     类型: 低波 → OTM 买入 / 高波 → 价差组合（动态切换）       │
│     持期: 30 天                                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ L3: 组合层（主 CTA + 期权策略叠加）                          │
│     权重: 主 CTA 65% + 期权策略 35%                          │
│     vol_target 8%, 期权最大损失 ≤ 组合 2%                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 关键定位（避免误读）

| 误区 | 本策略实际做法 |
|---|---|
| ❌ 用期权持仓**代替**期货价格做截面 | 截面信号仍来自期货价格 |
| ❌ 期权 OI 作为独立 alpha 源 | OI 仅作**确认/仓位缩放**因子 |
| ❌ 重做 §26 全部三策略 | 仅改进 S2（截面动量），S1/S3 不动 |
| ❌ 裸卖期权 | 仅买方（OTM 买入 / 价差组合含一腿卖出但为价差结构） |

**依据**: §26 截面三策略全败（S1 -0.38 / S2 -1.25 / S3 -0.59），CFTC 与经纪持仓均已证伪为独立 alpha 源。本策略的假设是：**截面动量信号需要确认层过滤假信号，期权 OI 是可得的确认信息**。

---

## 二、全部决策记录（2026-09-16 用户拍板）

| # | 决策项 | 选择 | 含义 |
|---|---|---|---|
| 1 | 优先级 | **C 并行** | CTA 信号 + 期权 OI 同时推进 |
| 2 | 截面宽度 | **C 接受 22×3** | 22 品种 × 3 年 OOS 可接受 |
| 3 | 验证标准 | **B OOS 正即可** | OOS Sharpe > 0.15 |
| 4 | 替代含义 | **C 独立策略** | 与 §26 S2 并列，不替代 |
| 5 | 期权类型 | **B+C 动态** | 低波→OTM 买入；高波→价差组合 |
| 6 | 驱动方式 | **C 结合** | CTA 定方向 + OI 定仓位大小 |
| 7 | 持仓时间 | **A 30 天** | 快速迭代 |
| 8 | 与 §26 关系 | **B 仅 S2 改进** | 聚焦截面动量 |
| 9 | 数据补充 | **不用** | 接受 22 品种 × 3 年 |
| 10 | 品种池 | **预先列出** | 见 §三 |
| 11 | 波动率判定 | **更复杂** | HV 分位数 + 多窗口（非简单中位数） |
| 12 | 失败处理 | **继续讨论** | 任一 Stage 失败回到讨论, 不自动终止 |
| 13 | 推进方式 | **顺序推进** | Stage 0 → 1 → 2 → 3 |
| 14 | 工作流 | **git first doc first** | 提交先行, 文档先行 |

---

## 三、品种池（预先列出）

### 3.1 筛选规则

从 `options_pcr_daily`（`/home/ll/Public/DataCache/futures_options_daily.duckdb`）筛选：

1. 2023-01-01 起有连续 PCR/OI 数据的品种
2. 有效交易日 ≥ 500 天（保证回测样本量）
3. 日均总 OI ≥ 10 万手（流动性门槛）
4. 排除金融期权（IO/MO/HO 为 ETF 期权，标的物不同）

### 3.2 实测品种清单（2026-09-16 查询回填）

筛选规则（实测后调整）: 2023+ 有效交易日 ≥ 500 天 且 日均总 OI ≥ 8 万手 且 非金融期权 → **27 个品种**:

```text
农产品(10): M, C, CF, SR, RM, Y, P, OI, PK, SM
黑色(3):    I, RB, SI
化工(7):    TA, MA, SA, V, EB, EG, RU
工业品(4):  FG, SH, LC, AG
有色(3):    CU, AL, AU
```

**截面宽度实测（2018-2026 全窗口）**:

| 宽度门槛 | 覆盖交易日 | 评价 |
|---|---|---|
| ≥ 5 品种 | 87.4% | 保底 |
| ≥ 8 品种 | 77.1% | 稳健 |
| **≥ 10 品种（建仓线）** | **76.9%** | **G2 门槛达标** |
| ≥ 15 品种 | 42.1% | 宽裕日 |

**主 CTA 品种覆盖**: RB/I/AG/AU/CU/M/TA/SR/RU 全部入池 ✓

**IS 段覆盖补充**（重要修正）: 期权数据并非 2023 才开始 —
SR (2073d, 2018起), M (1988d, 2018起), CF (1812d, 2019), C (1727d, 2019), TA (1598d, 2019), MA (1598d, 2019), RM (1576d, 2020), I (1518d, 2019) 等 8-10 个品种在 IS 段（2018-2022）即有数据。IS 段截面宽度 8-10 品种（贴 G2 下限），OOS 段（2023+）27 品种。

### 3.3 截面动态入选规则

- 每日截面仅使用**当日有有效 OI 数据**的品种（自适应宽度）
- 单日截面宽度 < 10 品种 → 该日不建仓（空仓）
- G2 门槛: 有效截面天数 ≥ 60% 交易日

---

## 四、Stage 0: CTA 截面动量信号（§26 S2 改进）

### 4.1 §26 S2 失败原因与改进对照

| §26 S2 问题 | 本版改进 |
|---|---|
| 单一 lookback (26 周) | 多时间尺度加权: 4/13/26/52 周 |
| 全市场混排 | 品种池限定为有期权品种（与 L2 同池） |
| 无确认层 | 本 Stage 仅出信号, 确认层在 Stage 2 |
| 毛价差 < 3% 即归档 | 门槛提高至 > 5%/年 |
| IS/OOS 双负未设闸 | G1 门槛: OOS Sharpe > 0.15, 否则回讨论 |

### 4.2 信号定义

```python
# 多时间尺度动量 (周频收益累计)
mom_w = mean(
    ret_4w,          # 短: 抓拐点
    ret_13w,         # 中: 主力信号
    ret_26w,         # 长: 趋势过滤
    ret_52w,         # 长: regime 确认
)
# 截面排名 → 三值信号
cta_rank_i = +1 if mom_w_i 在 top 1/3
           = -1 if mom_w_i 在 bottom 1/3
           = 0  else
# pit_shift=1 (信号 T 日, T+1 日执行)
```

### 4.3 回测框架（复用 §26 纪律）

- 周频再平衡（W-FRI）, ±1/n_active 等权
- vol_target 8%, cost 3bp（期货腿）
- IS 2018-2022 / OOS 2023-2025
- **G1**: OOS Sharpe > 0.15 且 IS/OOS 同向
- **G2**: 有效截面 ≥ 60% 交易日
- **G2b**: 毛价差 > 5%/年

### 4.4 交付物

- `scripts/cta_xsec_option/cta_xsec_momentum.py`
- `reports/cta/xsec_option/cta_xsec_momentum_eval.csv`
- 门槛不达标 → 停止, 回到讨论（决策 12）

---

## 五、Stage 1: 期权 OI 信号 + IC 验证

### 5.1 五类信号定义

数据源: `options_pcr_daily`（聚合表）+ `v_all_options`（合约级）:

| # | 信号 | 定义 | 方向假设 |
|---|---|---|---|
| S1 | oi_5d_chg | PCR-oi 5 日变化率 | OI 确认趋势 |
| S2 | pcr_oi_zscore | PCR-oi 60 日 z-score | 情绪极值反转 |
| S3 | oi_atm_pct | ATM±2 档 OI 占比（`v_all_options` 合约级） | 信息含量集中度 |
| S4 | vol_oi_ratio | volume / OI 日频比 | 投机活跃度 |
| S5 | oi_skew | (call_oi − put_oi)/(call_oi + put_oi) | call/put 不对称 |

### 5.2 IC 验证纪律（避免 §26 同期负 IC 陷阱）

- **先单品种时序 IC**, 不直接截面回测
- 每品种: 未来 5 日收益 vs 当日信号 Spearman IC
- 滚动窗口: IS 2018-2022, OOS 2023-2025
- **G3 门槛**: |IC_t 统计量| > 1.0 且 IS/OOS 同向（宽松版, 期望仅 1-2 个信号通过）

### 5.3 交付物

- `scripts/cta_xsec_option/option_oi_signals.py`
- `reports/cta/xsec_option/option_oi_signals_ic.csv`

---

## 六、Stage 2: CTA 信号驱动期权持仓

### 6.1 决策公式（决策 5+6 组合）

```python
# Step 1: 方向 — CTA 定
direction_i = cta_rank_i            # {-1, 0, +1}

# Step 2: 大小 — OI 定（五信号标准化加权, 实现用 Stage 1 通过的信号）
oi_confirm_i = w1·z(S1) + w2·z(S2) + ... (仅通过 G3 的信号等权)
size_factor_i = clip(0.5 + 0.5·|oi_confirm_i|, 0.5, 1.5)

# Step 3: 类型 — 波动率分位动态切换（决策 11: 更复杂判定）
#   HV 判定: HV20 与 HV60 双窗口 + 252 日滚动分位
vol_pct = percentile_rank(HV20, window=252)
vol_trend = HV20 / HV60 - 1
if vol_pct < 0.5 and vol_trend < 0:      # 低波且波动收敛
    option_type = "OTM_buy"              # B: 买 ATM±5% 虚值, 30 天
elif vol_pct >= 0.5 or vol_trend >= 0:   # 高波或波动上升
    option_type = "spread_combo"         # C: 牛/熊市价差, 净费有限

# Step 4: 组合
option_pos_i = direction_i · size_factor_i   # {-1.5, ..., +1.5}
```

### 6.2 期权结构明细

| 状态 | 结构 | 持期 | 最大损失 |
|---|---|---|---|
| 低波 + CTA 多 | 买 ATM+5% call, 30d | 30 天 | 权利金 |
| 低波 + CTA 空 | 买 ATM−5% put, 30d | 30 天 | 权利金 |
| 高波 + CTA 多 | 牛市价差: 买 ATM call + 卖 ATM+10% call | 30 天 | 净权利金 |
| 高波 + CTA 空 | 熊市价差: 买 ATM put + 卖 ATM−10% put | 30 天 | 净权利金 |
| CTA rank=0 | 空仓 | — | 0 |

### 6.3 成本模型（期权腿）

- bid-ask: ATM 2 档内 1.5 tick, OTM 3 tick
- theta: 按剩余天数线性衰减（30 天持期）
- 换月: 到期前 5 天滚动换至次月
- **G4 门槛**: OOS Sharpe > 0.15
- **G5 门槛**: 与 L1 期货腿相关性 < 0.4
- **G5b**: 毛价差 > 5%/年（扣期权成本后）

### 6.4 交付物

- `scripts/cta_xsec_option/cta_xsec_option_strategy.py`
- `reports/cta/xsec_option/cta_xsec_option_eval.csv`

---

## 七、Stage 3: 组合层

### 7.1 组合配置

```python
PORTFOLIO = {
    "main_cta_weight": 0.65,     # L1 期货腿 (S2 改进版)
    "option_weight": 0.35,       # L2 期权腿
    "vol_target": 0.08,          # 组合级
    "option_max_loss": 0.02,     # 期权腿年化最大损失预算
    "rebalance": "weekly",       # W-FRI
}
```

### 7.2 Go/No-Go 门槛

| 门槛 | 标准 |
|---|---|
| G6 | 组合 OOS Sharpe > 两腿单独最大值 |
| G7 | 组合 MaxDD < 1.5 × 期货腿 MaxDD |
| G8 | 与 E2-R+AU10% 日收益相关性 < 0.3 |

### 7.3 交付物

- `scripts/cta_xsec_option/combined_xsec_portfolio.py`
- `reports/cta/xsec_option/combined_xsec_eval.csv`

---

## 八、风险边界与失败处理

### 8.1 红线（全程遵守）

| 禁止 | 原因 |
|---|---|
| 裸卖期权 | docs/99-111 十一篇专题反复证伪 |
| OI 当独立 alpha | CFTC/经纪双重证伪 |
| 截面 < 10 品种强行建仓 | 统计功效不足 |
| 跳过 IC 验证直接截面回测 | §26 同期负 IC 陷阱 |
| 修改 `scripts/cta_cross_sectional.py` | §26 冻结, 仅复用模板 |
| 用策略自身 NAV 做任何输入 | 反馈循环 |

### 8.2 失败处理（决策 12）

任一 Stage 门槛不过 → **停止推进, 输出失败分析报告, 回到与用户讨论**, 不自动跳过、不自动调参重试（防过拟合）。

### 8.3 回滚

各 Stage 模块独立, 主 CTA 与已有体系零依赖, 失败仅弃用新模块。

---

## 九、目录与文件

```
scripts/cta_xsec_option/
├── __init__.py
├── data_loaders.py              # 品种池查询 + 期货/期权数据加载
├── cta_xsec_momentum.py         # Stage 0
├── option_oi_signals.py         # Stage 1
├── cta_xsec_option_strategy.py  # Stage 2
├── combined_xsec_portfolio.py   # Stage 3
└── config.py                    # 参数集中

tests/cta/ (xsec_option 前缀)
├── test_xsec_momentum.py
├── test_option_oi_signals.py
└── ...

reports/cta/xsec_option/        # 各 Stage 评估输出
docs/cta/xsec_option_design.md  # 本文档
```

---

## 十、实施顺序与验证链

```
[git] 提交基线 ✓ (ec010590)
[doc] 本文档 → commit
[universe] 品种池实测 → 回填 §3.2 → commit
[Stage 0] cta_xsec_momentum → G1/G2/G2b → 过: commit / 挂: 讨论
[Stage 1] option_oi_signals + IC → G3 → 过: commit / 挂: 讨论
[Stage 2] cta_xsec_option_strategy → G4/G5/G5b → 过: commit / 挂: 讨论
[Stage 3] combined_xsec_portfolio → G6/G7/G8 → 过: commit / 挂: 讨论
```

每步: ruff check + 相关 pytest 通过后才 commit。
