# 01 — 数据准备链路全攻略

> **范围**：momentum_etf_rotation 策略**直接相关**的数据准备链路
> **起点**：V0 时期 Stage 8 的 OHLCV 价格面板
> **终点**：V10 时期的 36 因子 panel + 3 维 X[T,N,K] 标准化面板
> **关键演进**：月频 → 周频 → 日频，截面因子 → 截面+时序 → 36 因子统一 panel

---

## 一、数据资产演进的 5 个时代

```
时代 1 (V0-V3):    ETF close/OHLCV 单面板        [44 ETF × daily/ME]
时代 2 (V4-V6):    截面因子库 (Smart β + 量价 11)  [30 因子 × daily/ME]
时代 3 (V7.0-V7.5):宏观 + 量价 + 资产池扩展         [20 因子 × weekly + 41-56 ETF]
时代 4 (V7.6-V7.10):三维因子面板 + OOS             [X[T,N,K] + Y[T,N]] weekly
时代 5 (V10):      4 策略统一面板                    [X_macro + X_pv + X_micro] daily
```

---

## 二、5 大里程碑（动机 / 做法 / 结果 / 教训）

### 里程碑 1：ETF panel 真实化（OHLCV 前复权 + 起跑日对齐）

**动机**

早期回测同时存在原始 close、不同来源 NAV、未处理公司行为的 OHLCV 和前置 flat 段：
1. 除权日大跳变污染动量、波动率和 11 个量价因子
2. 策略上市时间不同 → NAV 指标从不一致的有效日期起算 → 跨策略不可比
3. v5 2024 收益虚假 +87%（因 9 只 ETF 拆合股未修正）掩盖真实 +35%

**做法**

#### 1.1 价格拉取（前复权 ETF close）

```python
# scripts/fetch_real_etf_panel.py
# Tencent API → per-ETF parquet → 宽面板
# 仅对短缺失做 ffill(limit=5)
```

- 缓存：`data/real/etf_nav_<start>_<end>.parquet`
- per-ETF 缓存：`data/real/per_etf/*.parquet`
- 日志：`data/real/fetch_log.json`

#### 1.2 公司行为检测与前复权修复

```python
# scripts/fix_ohlcv_adjust.py
# 检测: |日收益| > 50% → 视为公司行为跳变
# 修复: 对事件前 OHLCV 乘累积调整因子
```

**触发事件**（2026-07-09, `49d8420`）：
- 512200 房产：+170%
- 159941 创业板 -75% 等

**修复产物**：`data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet`

**修复效果**：
- v5 2024 收益：87.36% → 35.55%（去除虚假 spike）
- v5 Calmar：0.643 → 0.764（+19%）

#### 1.3 起跑日对齐（5 个连续 fix commit）

```
f0cd21b  align_navs()           同图内取"最晚首次交易日"为起跑, 削平 flat
c52b227  GLOBAL_ALIGN_START     全局强制 2019-04-30 统一对齐
bb5971d  trim_flat_prefix()     撤回全局对齐, 改为各策略独立削平
49d8420  OHLCV 前复权修复       9 只 ETF 拆合股检测与调整
c669455  52 ETF 统一池派生 NAV  v1-v5 公平对比基础设施
```

**结论**：
- **业务上推荐"各策略独立削平"**（`bb5971d` 撤回 `c52b227`）
- 公平比较应**另取共同有效区间**，而非污染原始策略 NAV

**结果**

| 产物 | 路径 |
|------|------|
| ETF NAV 主面板 | `data/real/etf_nav_2018-01-01_2026-06-30.parquet` |
| ETF OHLCV 主面板 | `data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet` |
| v1-v5 公平 NAV | `reports/momentum_etf_rotation/combo/unified_v1v5_navs.parquet` |

**教训**

1. **"接口声称 qfq" 不能替代事件级收益检查**：OHLCV 各字段必须同口径调整
2. 不能用 `bfill` 构造 ETF 上市前价格
3. volume 复权 ≠ 价格复权（价格连续性 vs 真实成交金额是两件事）
4. 跨交易所交易日不同 → 对齐后保留 NaN
5. 数据源差异（Tencent vs Wind/Choice）在除权日可能 0.1%-0.5% 偏差

---

### 里程碑 2：Proxy indices（晚上市 ETF 的历史建仓替代）

**动机**

56 资产扩展池包含大量 2018 后才上市的行业 ETF：
- 若仅用 ETF 本身 → 早期截面小、样本起点随资产变化
- 若直接填补上市前价格 → 产生不可交易虚假历史
- 用 Proxy 提供**历史风险与因子背景**，而非宣称 ETF 上市前可实盘买入

**做法**

#### 2.1 19+5 Proxy 指数拉取

```python
# scripts/fetch_proxy_indices.py
# iFinD: 抓取 19 个 CSI/SZ/HK/海外指数
# scripts/fetch_proxy_indices_wind.py
# Wind: 备选 17 个 ETF 对标指数
```

#### 2.2 对齐 v56 交易日

```python
# scripts/build_proxy_panel.py
# 将 proxy close 对齐至 v56_expanded_daily 交易日索引
# 不做前向填充
# 生成 ETF→proxy 映射 + 上市前后 NaN 统计
```

**关键约束**：
- **Proxy 可用于历史训练/风险中性化**
- **ETF 上市后才可进入真实交易池**
- v7.14 用动态资产池（`_get_valid_assets()`）实现

**结果**

| 产物 | 路径 |
|------|------|
| Proxy panel | `data/high_freq_macro/v56_proxy_indices_daily.parquet` |
| NaN audit | `data/high_freq_macro/_proxy_nan_table.csv` |
| ETF→Proxy 映射 | `data/high_freq_macro/_proxy_etf_map.csv` |

**教训**

1. **Proxy 只能提供信息历史**：不能替代真实可交易价格、流动性、跟踪误差和涨跌停约束
2. **iFinD token 起点（2021-07-01）** 无法覆盖 2018 起始训练 → 需 Wind 备选
3. **同一 ETF 可能有多个候选指数**：必须记录映射来源，避免指数定义漂移
4. **A 股/港股/美股交易日不同**：必须对齐后保留 NaN，不能随意跨市场 ffill

---

### 里程碑 3：因子库建设（截面 + 时序 + 宏观）

**动机**

简单动量在不同市场阶段衰减明显。研发需要：
- 行业横截面差异
- 全市场宏观状态
- 资产自身时序特征

为 TV-PR 提供统一 `X[T,N,K]` 输入。

**做法**

#### 3.1 v4 截面/Smart Beta 因子

```python
# QuantNodes/strategy/momentum_etf_rotation/v4/factor_ic.py
# 6 因子: momentum / reversal / value / low_vol / dividend / quality
# IC 测量: 截面 Spearman correlation
```

**实测**：
- `value`: mean IC +0.044, hit 60% ✅（唯一稳定）
- `reversal` / `dividend`: 稳定负 alpha ❌（A 股反转效应不存在）
- `low_vol`: IC -0.454 ❌（反指因子，已删除）

#### 3.2 v5 行业量价 11 因子（华西证券研报复刻）

```python
# QuantNodes/strategy/momentum_etf_rotation/v5/industry_factors.py
# 实施华西证券《行业有效量价因子与行业轮动策略》（23 页 PDF）
# 6 大类 / 11 月频因子（动量/交易波动/换手率/多空对比/量价背离/量幅同向）
```

**11 因子**：
| 因子 | 类型 | IC |
|------|------|-----|
| `f1_second_mom` | 二阶动量 | - |
| `f2_mom_term` | 动量期限差 | - |
| `f3_amt_vol` | 成交金额波动（反转）| **+0.0319** |
| `f4_vol_vol` | 成交量波动（反转）| **+0.0456** ✅ |
| `f5_turnover` | 换手率变化 | - |
| `f6_ls_total` | 多空对比总量（反转）| -0.0180 |
| `f7_ls_change` | 多空对比变化 | - |
| `f8_pv_rankcov` | 量价排序协方差（反转）| - |
| `f9_pv_corr` | 量价相关系数（反转）| - |
| `f10_first_div` | 一阶量价背离（反转）| +0.0182 |
| `f11_vol_range` | 量幅同向 | - |

#### 3.3 v7 增强量价 + 宏观因子

```python
# QuantNodes/strategy/momentum_etf_rotation/v7/enhanced_factors.py
# 新增微观结构 + 尾部风险 + 流动性
```

**f12-f17 增强量价**：
| 因子 | 含义 | IC |
|------|------|-----|
| `f12_amihud` | Amihud 非流动性 | **+0.0386** ✅ |
| `f13_rv` | 已实现波动率（偏态）| -0.0115 |
| `f14_rs` | 已实现偏度 | +0.0029 ❌ |
| `f15_max5` | 最大 5 日收益 | -0.0173 |
| `f16_52w_high` | 52 周高点距离 | **+0.0467** ⭐ |
| `f17_idio_vol` | 特质波动率 | **-0.0334** ✅ |

**v7.11 后续候选**：`f18_mom_short` / `f19_mom_mid` / `f20_mom_long` / `f21_reversal` / `f22_rsi`

**v7.9 去重 39→36**：f18 ↔ f21 (r=-1.000) 删 f21；f3 ↔ f4 (r=0.958) 删 f4；f8 ↔ f9 (r=0.938) 删 f9

**宏观因子库**（v7.3 基础组 + v7.6 增强组）：

| 因子 | IC |
|------|-----|
| VIX 综合分 | **-0.0831**（最强时序）/ **-0.1376** panel ⭐ |
| 生活端通胀 | +0.0471 |
| 信用利差 | -0.0396 |
| 股期限利差 | -0.0332 |
| DXY 对数收益 | -0.0220 |
| 中美利差 | - |
| 黄金—原油相关性 | - |
| 实际利率（FRED DFII10） | - |
| 无风险利率 / 生产端通胀 | < 0.02（应剔除）|

#### 3.4 IC 计算与评估（核心 4 维区分）

```python
# QuantNodes/strategy/momentum_etf_rotation/common/ic_utils.py
# 关键: 宏观因子 vs 截面因子用不同 IC 计算方式
```

**截面因子（PV, k≥17）**：
```python
# 截面 IC — 选股能力
ic = spearmanr(contrib[t,:], Y[t+1,:])
```

**时序因子（宏观, k≤16）**：
```python
# 时序 IC — 择时能力
ic = spearmanr(β_k[t]×X[t,0,k], mean(Y[t+1,:]))
```

**关键差异**：
- 宏观因子对所有资产截面值相同 → `contrib = β × X` 是常数向量
- 截面 IC 在宏观因子上会返回 NaN（`spearmanr` of constant vector）
- 必须分别用时序/截面 IC 计算方法

**IC 评估组件**：

```python
# scripts/calc_factor_ic.py
# 统一计算宏观时序/panel IC + 量价截面 IC

# scripts/factor_timing_diagnostic.py
# 诊断 IC 窗口衰减、自相关、regime 分解
```

**结果**

| 阶段 | 因子数 | 数据频率 | 工具 |
|------|--------|---------|------|
| v4 | 6 截面 | 日频 | `v4/factor_ic.py` |
| v5 | 11 量价 | 日频 | `v5/industry_factors.py` |
| v6.1/v6.2 | 11 + IC 加权 | 日频 | `v6_1/factor_weighting.py` |
| v7.3 | 9 宏观 | 周频 | `v7/macro_data.py` |
| v7.6 | 12 宏观 + 17 量价 = 29 | 周频 | `v7/enhanced_factors.py` |
| v7.9 | 17 宏观 + 19 量价，去重后 36 因子 | 周频 | `v7/data_loader_v7_6.py` |
| v7.10 | 36 因子 + 混合标准化 | 周频 | `v7/macro_substrategy_v7_6.py` |

**教训**

1. **宏观因子对所有资产相同**，不能直接按资产截面标准化 → 应沿时间维标准化
2. **上市前量价 NaN 用截面中位数填充**可保持矩阵维度，但隐含"平均因子暴露" → 需结合动态资产池
3. **特质波动率依赖沪深300基准**：基准缺失会显著增加 NaN
4. **因子越多 ≠ 信息越多**：v7.9 必须删除高度重复因子并做对数变换
5. **IC 均值必须与 IC 标准差、ICIR、命中率、显著性同时观察** —— IC 加权会放大噪声
6. **因子筛选必须在 OOS/Bootstrap 后完成** —— 不能用全样本 IC 排名直接决定生产因子

---

### 里程碑 4：TV-PR 数据面板（X[t] → Y[t+1] 严格训练）

**动机**

TV-PR 需要同时消费宏观时序和资产截面因子。初版存在：
- 月/周频混用
- 面板形状错误
- 同期目标
- full-sample 平滑器使用未来数据
- 周频信号映射到日频收益错位
- → 漂亮但虚假的 OOS

**做法**

#### 4.1 三维面板构造

```python
# QuantNodes/strategy/momentum_etf_rotation/v7/data_loader_v7_6.py
# 输入:
#   - 宏观列 (T, Km): 时序 DXY/VIX/实际利率/利差/相关性
#   - 量价列 (T, N, Kp): 截面 OHLCV 派生因子
# 处理:
#   1. 宏观沿资产复制 (T, N, Km)
#   2. 量价按资产填入
#   3. 当期截面中位数填量价 NaN
#   4. 动态资产池屏蔽未上市/历史不足资产
#   5. 偏态因子 Winsorize / log transform
#   6. 宏观时序 Z-score + 量价截面 Z-score
# 输出: X[T,N,K] 三维面板
```

#### 4.2 严格 X[t]→Y[t+1]

```python
# Y[T,N] 构造
# 1. ETF NAV 采样至周末（last）
# 2. Y[t] = (NAV[t] / NAV[t-1]) - 1   ← 必须是 t-1 到 t 的收益
# 3. 训练时: X[t] 预测 Y[t+1]          ← 严格下一期
```

**Look-ahead 修复链路**（4 个 commit）：
1. `0c1c6a4` 修复 6 个未来函数 Bug
2. `105b4b3` X[t]→Y[t+1] 信号重设计（仍 wip）
3. `4be2ba3` 彻底修复（信号-执行同期 + 日频 NAV 映射）
4. `9d56a0b` TV-PR ADMM 标准实现

#### 4.3 日频 NAV 映射

```python
# 日频 ETF 收益 + 周频权重 → 日频组合 NAV
# 独立处理交易成本、周一开盘 / 周五收盘
```

#### 4.4 expanding-window（OOS 无前视）

```python
# Walk-Forward 通用化（common/walk_forward.py 990 行）
# 原则: 训练只用 Y.iloc[:test_start]
# 默认: expanding window（覆盖训练集增长，OOS 稳定）
# 可选: rolling window 52w/104w/208w
```

**结果**

| 频率 | 阶段 | 报告样本 | 资产数 | 因子数 |
|------|------|----------|--------|--------|
| 月频 | v7.6 早期 | ~100 月 | 43 | 12+17 |
| 周频 | v7.6 修复后 | 430 周 | 43 | 12+17 |
| 周频（日频 NAV 映射）| v7.9 | 430 周 | 51+5 bond | 17+19=36 |
| 周频（v7.10 标准化）| v7.10 | 430 周 | 56 | 36 |
| v10 日频 | v10 | 2000+ 日 | 56 | 36 + 4 策略 |

**关键产物**：
- `v7_10_X_panel.npy`（3 维 X[T,N,K]）
- `v7_10_Y_weekly.parquet`
- factor names + codes CSV

**教训**

1. **`shift(-1)` 的方向、标签含义和执行周期必须用时间线测试**，不能凭数组位置推断
2. 周频索引用周日、行情用周五时必须显式平移两天；周一开盘到周五收盘需要独立对齐
3. **full-sample ADMM 平滑的 β[t] 天然会看到未来** → 不能称为 OOS
4. 先计算日收益再周度聚合 ≠ 先采样 NAV 再算周收益 → 遇到 NaN 和缺口时会不同
5. X 和 Y 维度匹配不代表资产顺序匹配 → 必须持久化 code 列表并在加载时校验
6. **expanding OOS 优于 full_sample**（v7.10 Sharpe 1.57 > 1.11）→ 暗示 full_sample 含前视偏差反致过拟合

---

### 里程碑 5：缺数据扰动（Sensitivity Phase 4）

**动机**

初始 TV-PR 对中位数填充和完整矩阵有强依赖：
- 扩展资产池包含大量晚上市 ETF
- 宏观数据跨中国/美国/香港交易日
- 需要测试在现实缺数、停牌、数据源延迟下能否保持表现

**做法**

#### 5.1 Sensitivity Phase 4 实施（`adb7cda`）

```python
# scripts/combo/sensitivity_test_phase4.py
# 按比例向输入注入缺失
# 比较基线与不同缺失率下的 OOS Calmar
```

#### 5.2 数据层防御

| 防护层 | 实现 |
|--------|------|
| 短价格缺口 | `ffill(limit=5)` |
| 量价 NaN | 当期截面中位数 |
| Proxy 上市前 | 保留 NaN |
| 跨缺口收益 | NaN-safe pct_change |
| 长缺口 | 显式 NaN（不填充）|
| 资产可交易 mask | `_get_valid_assets()` |

#### 5.3 NaN-safe 工程改进（`6ad3f88` + `680f6bc`）

```python
# QuantNodes/strategy/momentum_etf_rotation/v7/...
# - 日收益 NaN-safe 计算（daily_returns 可选参数）
# - v7 NaN weight 过滤: std() 返回 NaN 时跳过资产
# - 动态资产池（min_assets=10）
```

**结果**

| 扰动 | 退化 |
|------|------|
| Phase 3 Bootstrap | CV=165.95%, mean Calmar=0.0787 |
| Phase 4 20% 缺失 | -101% |
| Phase 5 构造层扰动 | 最大 -80.7% |
| Phase 6 β_path | 断点频率 3.26%, CV 9.91, ACF 0.31 |

**教训**

1. **缺失不是单一机制**：上市前缺失 / 停牌 / 跨交易所休市 / API 漏数 / 指标 warm-up 必须分别处理
2. **截面中位数填充可让算法运行，但会压低真实不确定性** → 应与可交易 mask 联动
3. **`pct_change` 默认填充行为可能产生伪零收益或跨缺口收益** → 显式禁用隐式填充并设置 gap rule
4. **缺失扰动应保持成块/连续缺失**，独立随机点缺失不足以模拟真实上市和停牌
5. 对缺失高度敏感的模型即使无缺失基线优秀，也不具备生产稳定性

---

## 三、数据 Pipeline 最后稳定状态

```text
Tencent ETF / 已有 OHLCV
    ↓
per-ETF parquet + fetch_log
    ↓
ETF NAV panel / OHLCV panel
    ↓
公司行为检测与前复权修复（50% 阈值 + 累积调整因子）
    ↓
adjusted OHLCV
    ├── 日频收益、滚动波动率、可交易 mask
    └── 11 基础量价 + 增强量价因子（f12-f22）
                    ↓
              周频 last 对齐
                    ↓
外部宏观 Excel/SQLite/FRED
    ↓
DXY / VIX / 实际利率（DFII10） / 利差 / 相关性
    ↓
按变量定义先做 log return / diff / rank
    ↓
周频对齐
                    ↓
Proxy indices (iFinD / Wind / 既有指数)
    ↓
对齐 v56 交易日，保留上市前 NaN
    ↓
ETF→proxy map + NaN audit
                    ↓
X_macro[T,Km] + X_pv[code][T,Kp]
    ↓
X_panel[T,N,K]
    ├── 宏观因子沿资产复制
    ├── 量价因子按资产填入
    ├── 当期截面中位数填量价 NaN
    ├── 动态资产池屏蔽未上市/历史不足资产（min_assets=10）
    ├── 偏态因子 Winsorize / log transform
    └── 宏观时序 Z-score + 量价截面 Z-score
                    ↓
ETF NAV 先周采样再 pct_change
                    ↓
Y[T,N]
                    ↓
严格训练 X[t] → Y[t+1]（无前视）
    ↓
expanding-window TV-PR / IC / 替代模型
    ↓
下一可执行周期周频权重
    ↓
映射至 NaN-safe 日频收益 + 交易成本（5bp+10bp）
    ↓
日频策略 NAV
    ↓
统一日历的多策略组合 NAV / OOS 指标 / HTML
```

---

## 四、8 条最后稳定原则

| # | 原则 | 解释 |
|---|------|------|
| 1 | **价格** | ETF 主链使用前复权/事件修复后的价格（`fix_ohlcv_adjust.py`）|
| 2 | **频率** | 周频特征 + TV-PR 训练；日频执行与绩效 |
| 3 | **时序** | 严格 `X[t]→Y[t+1]`，full-sample TV-PR 仅作诊断 |
| 4 | **缺失** | 短缺口有限填补（`ffill(limit=5)`）；长缺口保留 NaN；动态资产池 |
| 5 | **标准化** | 宏观按时间维 Z-score；量价按截面维 Z-score + Winsorize |
| 6 | **评估** | IC、滚动 IC、多个起点、Bootstrap、多段 hold-out、缺失扰动、expanding OOS 缺一不可 |
| 7 | **组合** | 所有派生 NAV 同日频指标口径，原始策略起跑日不被人为改写 |
| 8 | **模型** | v7.6 后向因子压缩、expanding、TV-PR/熵权混合演进（不依赖纯 TV-PR 最大权重）|

---

## 五、关键脚本清单（去重后）

| 路径 | 一句话 | 复用场景 |
|------|--------|---------|
| `scripts/fetch_real_etf_panel.py` | 从 Tencent 拉取前复权 ETF close，按 ETF 缓存构建宽面板 | 公开数据源 ETF 历史回补、增量重拉 |
| `scripts/fix_ohlcv_adjust.py` | 检测极端公司行为跳变（50% 阈值）并生成连续 adjusted OHLCV | 量价因子、波动率、技术指标前的数据清洗 |
| `scripts/fetch_proxy_indices.py` | 分块抓取 19 个 iFinD proxy 指数并生成 parquet 缓存 | 晚上市 ETF 的历史信息补充 |
| `scripts/fetch_proxy_indices_wind.py` | 从 Wind 获取 ETF 对标指数并写 SQLite | 有 Wind 权限时补齐 2018+ proxy 历史 |
| `scripts/build_proxy_panel.py` | 对齐 proxy 与 v56 交易日并生成 NaN 表和 ETF 映射 | 上市前缺失审计、动态资产池 |
| `scripts/extract_macro_data.py` | 从 gold SQLite 提取 DXY/VIX/实际利率到 parquet | 外部宏观数据库转本地缓存 |
| `scripts/calc_factor_ic.py` | 统一计算宏观时序/panel IC 与量价截面 IC | 新因子准入、OOS 前筛选 |
| `scripts/factor_timing_diagnostic.py` | 诊断 IC 窗口衰减、自相关、regime 和因子择时增益 | 决定 IC lookback、识别低信噪比因子 |
| `scripts/combo/regenerate_v7_10_nav_with_v56.py` | 用 v56 数据再生 v7.10 组合 NAV | 统一资产池后的策略组合输入 |
| `scripts/combo/regenerate_v8_v56_nolookahead.py` | 以 v56 数据无前视重算 v8 NAV | 校验策略组合是否混入未来信息 |
| `scripts/combo/standard_comparison.py` | 对多条 NAV 做统一指标比较 | 同日历、跨版本评估 |
| `scripts/combo/full_sample_metrics.py` | 计算组合 NAV 全样本指标 | 与独立 OOS 指标并列审计 |

---

## 六、因子库总表

### 截面/Smart Beta 因子（v4）

| 因子 | 含义 | IC | 用法 |
|------|------|------|------|
| `value` | 价值 | **+0.044** ✅ | 唯一稳定正 IC |
| `momentum` | 动量 | 接近 0 | 仅短期有效 |
| `reversal` | 反转 | 负 | A 股反转不存在 |
| `low_vol` | 低波 | -0.454 ❌ | 已删除 |
| `dividend` | 股息 | 负 | A 股股息效果不稳定 |
| `quality` | 质量 | regime-dependent | 仅趋势市有效 |

### v5 行业量价 11 因子

| 因子 | IC | 备注 |
|------|------|------|
| `f4_vol_vol` | +0.0456 ⭐ | 最强反转因子 |
| `f3_amt_vol` | +0.0319 | 次强 |
| `f10_first_div` | +0.0182 | 弱 |
| `f6_ls_total` | -0.0180 | 反向 |
| `f12_amihud` | +0.0386 | v7.6 增强 |
| `f16_52w_high` | +0.0467 ⭐ | v7.6 量价最强 |
| `f17_idio_vol` | -0.0334 | 低特质波动优 |

### v7 增强量价 + 宏观

| 类型 | 因子 | 关键 IC |
|------|------|---------|
| 增强量价 | `f12_amihud` / `f16_52w_high` / `f17_idio_vol` | > 0.03 |
| 宏观增强 | VIX / 生活端通胀 / 信用利差 / DXY | > 0.02 |
| 增强时序 | DCC 6 维 / HMM 距离先验 | regime 信号 |
| 增强距离 | 图谱距离（pagerank/local_clustering）| v7.13 |
| 增强截面 | 相关性距离（distance_to_centroid） | v7.14 |

---

## 七、数据审计清单（每次研究 / 报告前必跑）

1. ☐ **OHLCV 前复权**：跑 `fix_ohlcv_adjust.py`，确认无 50% 跳变未修复
2. ☐ **NAV 起跑日**：用 `trim_flat_prefix()` 削平 flat 段
3. ☐ **频率对齐**：周频用"先采样 NAV 再算收益"，日频映射写明 `W-FRI → W-MON`
4. ☐ **Proxy NaN**：上市前 NaN 保留，上市后短缺口用 `ffill(limit=5)`
5. ☐ **缺数据扰动**：Sensitivity Phase 4 用 10%/20% 缺失率测试
6. ☐ **频率一致性**：所有派生 NAV 用同一年化（√252 vs √52）
7. ☐ **代码顺序**：持久化资产列表（code list）+ 因子名（factor_names）随数据保存
8. ☐ **Look-ahead 验证**：OOS 用 expanding/rolling window，绝不用 full_sample
9. ☐ **CV% 测试**：起点依赖 < 25% PASS，25-50% PROMISING，> 50% DEPRECATED

---

## 八、关键文档产出（按主题）

### 通用方法 / 索引
- `reports/momentum_etf_rotation/README.md` — 策略研究入口
- `reports/momentum_etf_rotation/common/GAP_ANALYSIS.md` — Tencent vs Wind/Choice 复权 / 缺失 / 池差异
- `reports/momentum_etf_rotation/common/extended_metrics.md` — 统一指标口径
- `reports/momentum_etf_rotation/docs/STRATEGY_VERSIONS.md` — 月/周/日频 + 因子数 + OOS 演进
- `reports/momentum_etf_rotation/docs/CODE_REVIEW_CHECKLIST.md` — 提交前 checklist（含缺数据 fallback / 未来函数 / 频率检查）
- `reports/momentum_etf_rotation/docs/DEV_WORKFLOW.md` — 真实数据 / OOS / 缺失 fallback 流程

### V1–V3 早期数据验证
- `v1/validation_fix_report.md` — Stage 7 验证修复
- `v2/stage9_extended_comparison.md` — 17 指标 + 雷达 + DD vs Calmar
- `v2/stage9a-d_reports` × 4 — 52 周高点 / TF / VT / HMM
- `v3/stage16a_validation.md` — 多子策略 OOS 验证

### V4 截面因子 / IC
- `v4/FACTOR_TIMING_EFFECTIVENESS.md` — 6 因子 IC 分布 + 衰减 + 自相关
- `v4/SMART_BETA_ALPHA_DECAY.md` — Smart β alpha 衰减研究
- `v4/STYLE_ROTATION_RESEARCH.md` — 120d Top-1 Calmar 0.919
- `v4/STAGE17_RESEARCH_INDEX.md` — 4 份研究汇总
- `v4/STAGE18_V4_FINAL.md` — v4/v5 整合
- `v4/STAGE19_LW_INTEGRATION.md` — LW 集成

### V5 / V6 量价因子
- `v5/STAGE22_V5_REPORT.md` — 11 量价因子完整 SubStrategy + 详细统计
- `v6_2/STAGE29_PROMOTION.md` — 量价族 IC 加权 + 正交化 + 起点依赖评估

### V7–V10 TV-PR
- `v7_6_factor_ic_report.md` — 12 宏观 + 17 量价 IC
- `v7_6_sensitivity_report.md` — 10 阶段扰动
- `v7_6_oos_validation.md` — full-sample vs expanding 对照
- `v7_9_oos_validation.md` — v7.9 去重 + 正交化 36 因子 OOS
- `v7_10_oos_validation.md` — v7.10 标准化 expanding OOS
- `STAGE32_PLAN.md` / `STAGE33_PLAN.md` — v7.10 硬化 + 未来规划

---

## 九、教训（一句话版）

1. **OHLCV 前复权必须做**，否则量价因子被虚假跳变主导
2. **起跑日不要全局对齐**，各策略独立削平 + 共同区间比较更稳健
3. **Proxy 仅供历史信息**，不可替代真实可交易价格
4. **宏观 vs PV 用不同 IC 计算方法**（截面对宏观因子会返回 NaN）
5. **严格 `X[t]→Y[t+1]`**，缺一不可
6. **expanding OOS 优于 full_sample**（前视偏差反致过拟合）
7. **缺数据扰动是必要 P0**，20% 缺失退化 -101% 是严重信号
8. **8 条最后稳定原则** 是所有派生研究 + 报告的最低门槛

---
