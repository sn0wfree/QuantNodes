# 行业轮动策略研发全周期复盘（V0→V10）

> **本目录**：`docs/research_history/`
> **覆盖范围**：2026-07-07 ~ 2026-07-24，971 个 commit（`git log --all`），共 ~18 天
> **主题**：momentum_etf_rotation 策略构建链路（数据 → IC → 因子 → 选股 → 加权 → 风控 → 组合 → OOS → 硬化）
> **排除**：CTA、数据迁移脚本、项目基础架构、纯重构叙事（除影响回测正确性）
> **结论先行**：4 策略 Vol-parity（v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%）以 OOS Sharpe **1.991** 成为生产首选

---

## 一图概览

```
                    ┌─────────────────────────────────────────────┐
                    │  QuantNodes momentum_etf_rotation 策略研发链  │
                    │  2026-07-07 ~ 2026-07-24（18 天，971 commit）   │
                    └────────────────────────┬────────────────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
   数据基础设施                          V0→V3 策略期                          V7→V10 策略期
   (01_DATA_FOUNDATION)                 (02_V0_V3)                            (04_V7_V10)
   ────────────────                     ──────────                            ───────────
   • ETF NAV/OHLCV 真实化                • 单动量期 (Stage 8-12A)                • 5 Macro Dynamic (V7.0)
   • Proxy indices 19+5                • V2 重构 + slope_r²                    • Symmetry+Bootstrap-Lasso (V7.3)
   • 因子库:截面+时序+宏观               • Stage 16A 多策略架构 (V3)             • TV-PR 时变 LASSO (V7.6-V7.10)
   • IC 评估统一接口                    • 业绩曲线 HTML 体系                    • 统一引擎 + YAML 驱动
   • TV-PR 面板 X[T,N,K]               • 公平对比+前复权                       • 树模型失败 (V7.7)
   • 缺数据扰动                         • 1/N 等权失败教训                     • expanding-window 真实 OOS
        │                                    │                                    │
        └────────────────┬───────────────────┴──────────────────┬─────────────────┘
                         │                                       │
                    V4→V6 策略期                          沉淀：教训 + SOP + 索引
                    (03_V4_V6)                            ────────────────
                    ──────────                            • 05_LESSONS_LIBRARY
                    • 风格轮动+Smart β (V4, Stage 17)    • 06_RESEARCH_SOP
                    • 量价 11 因子 (V5/V5.1, Stage 22)   • 07_INVENTORY
                    • 单策略版 (V6)                       • 08_FUTURE_WORK
                    • IC 加权 + 正交化 (V6.1/V6.2)
                    • 5-fold walk-forward
                    • CV% 测试 → DEPRECATED

                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  V10 收敛：4 策略 Vol-parity 组合   │
                    │  v1.0(74%) + v9macro(12%) +        │
                    │  v7.10(9%) + DualMom(5%)            │
                    │  OOS Sharpe 1.991 ⭐⭐⭐            │
                    └─────────────────────────────────────┘
```

---

## 关键时间节点（研报级时间轴）

> 每个里程碑都给 commit hash + 一句话结论 + 教训。完整 commit 列表见各阶段文档。

### Phase 1：V0–V3 策略期（2026-07-07 ~ 2026-07-09）

| 日期 | Commit | 事件 |
|------|--------|------|
| 07-07 | `443f715` | Stage 9-A：52 周高点信号融合，开启多信号探索 |
| 07-07 | `9d2a603` | Stage 9-B：HS300 MA200 趋势过滤 |
| 07-07 | `9e52eb4` | **Stage 9-C：VolTargeting Calmar +28%（最大突破）** |
| 07-07 | `65ac981` | Stage 9-D：HMM 全协方差 NO-GO（Calmar -33%，需要 LW 收缩） |
| 07-07 | `0da2e2c` | Stage 10：ConcentrationCaps NO-GO（Calmar -22%，逆波动已自然分散） |
| 07-08 | `702907a` | Stage 11：协方差+RP（技术 OK，表现不佳） |
| 07-09 | `07956ca` | **Stage 12A：斜率×R² hybrid + VT = OOS Calmar 1.60** |
| 07-09 | `9a38fc3` | v1.0 锁定：策略版本系统（v0.0~v1.0） |
| 07-09 | `66a612a` | V2 重构：`common/ + v1/ + v2/` 拆分 |
| 07-09 | `c2374f8` | Stage 14：924 政策大涨分析（错过 18.87%）驱动 Stage 16A 多策略 |
| 07-09 | `5670b68` | V3 创建 + SubStrategy 抽象基类 |
| 07-09 | `f9e5563` | Stage 16A Step 2：均值反转子策略 |
| 07-09 | `aaf8aae` | Stage 16A Step 3：行业轮动子策略 |
| 07-09 | `5803e31` | Stage 16A Step 4：子策略权重分配 |
| 07-09 | `e12070c` | **Stage 16A Step 5：多策略主回测 v3 Calmar 0.504 < v2 0.892** |
| 07-09 | `49d8420` | **OHLCV 前复权修复（9 只 ETF 拆合股），v5 2024 虚假 +87% 修正为 +35%** |
| 07-09 | `c669455` | 52 ETF 统一池公平对比基础设施 |

### Phase 2：V4–V6 策略期（2026-07-09 ~ 2026-07-14）

| 日期 | Commit | 事件 |
|------|--------|------|
| 07-09 | `2c1de11` | V4 集成：风格轮动+Smart β+因子择时（Stage 17 ABC 雏形） |
| 07-09 | `64dcba4` | Stage 17 V4 完整：距离先验 HMM + 6 模式 + 43 测试 |
| 07-09 | `e983294` | **6 因子 IC 仅 value 稳定正 IC（mean +0.044, hit 60%）** |
| 07-09 | `1ba45a9` | **风格轮动 L120_T1 Calmar 0.919（4x 改善 L60_T3）** |
| 07-09 | `181bf5a` | Smart β 实为低 beta 工具（β=0.60），alpha +7.79%/y 但 regime-conditional |
| 07-09 | `9f4dfd9` | Stage 18 V5 诊断驱动 4+5 改进（实验性） |
| 07-09 | `e17098c` | **Stage 19 LW 增强：v4 IC² Calmar 0.613（OOS 0.581）** |
| 07-09 | `7c85cfd` | **Stage 22：华西证券 11 量价因子（v5 起点）** |
| 07-09 | `9157b31` | **V5.1 逆波动率加权 OOS Calmar 0.488→0.589 (+20.7%)** |
| 07-09 | `a2f07bc` | V5.1.1 消融 S1+S3+S4 选中 OOS Calmar 0.604 (+2.5%) |
| 07-10 | `00605c7` | **V6 = v5.1.1 选股 + v5.1.1 加权 + TF 风控（OOS 0.662）** |
| 07-10 | `a60589c` | **HTML OOS 显示 bug 修正：v6 0.748 → 真实 OOS 0.662** |
| 07-10 | `d7fac94` | V6.1 IC-IR 加权 + V6.2 Gram-Schmidt 正交化 |
| 07-14 | `223ef65` | **V6.2 过拟合审计：扣成本 Calmar 0.3310（-25.6%）, CV% 56.9% FAIL** |
| 07-14 | `8be00ae` | V6.2 状态从 PROMISING 降为**研究版本**（DEPRECATED）|

### Phase 3：V7–V10 策略期（2026-07-13 ~ 2026-07-24）

| 日期 | Commit | 事件 |
|------|--------|------|
| 07-13 | `ebff3cb` | V7.3 完整还原 source notebook（Symmetry + Bootstrap-Lasso 2000 + FRP） |
| 07-13 | `332aad3` | V7.2 TF 权益专属化（换手 -62%） |
| 07-13 | `43a7ba8` | **锁定 v7.3 完整版为 v7_macro_baseline（性能退化 > 5% 必更新）** |
| 07-13 | `ad2a9a3` | V7.4 扩大资产池（51 ETF + 5 bond = 56 assets） |
| 07-14 | `02f63ad` | V7.5 Step 1：硬止损（10% DD）OOS Calmar +20% |
| 07-14 | `f34e4ee` | V7.5 Step 2：连续 TF Score 实验失败（-68%）|
| 07-14 | `5f44fbf` | V7.5 Step 3：时变 LASSO（rolling 156w）实验失败（-66%）|
| 07-14 | `bf8256b` | **V7.6：TV-PR 时变 LASSO 估计器（ADMM 双辅助变量）** |
| 07-14 | `826db4d` | V7.6 Walk-Forward 框架（`common/walk_forward.py` 990 行通用化）|
| 07-15 | `0c1c6a4` | **修复 V7.6 未来函数 6 个 Bug + X[t]→Y[t+1] 严格重构** |
| 07-15 | `950420b` | V7.6 Sensitivity Phase 3：Bootstrap CV 165.95%（严重过拟合） |
| 07-15 | `adb7cda` | V7.6 Sensitivity Phase 4：20% 缺失退化 +101% |
| 07-16 | `df96963` | V7.6 因子 IC 评估统一脚本（截面 vs 时序） |
| 07-17 | `db4a852` | V7.9 NaN 修复 + 因子去重 39→36 + 对称正交化失败（Sharpe 1.40→0.15）|
| 07-18 | `a5de7f3` | V7.7 Phase 1：树模型 PyCaret 25 模型 → 修复 look-ahead 后 R²≈0 |
| 07-18 | `bbcaf86` | V7.10 Stage 32 硬化（stop_loss + CV% + v6.2 DEPRECATED + pandas 兼容） |
| 07-19 | `688862d` | **V7.10 验证过拟合：Calmar 0.671→0.241（-64%）** |
| 07-20 | `eaa6c9b` | **修复 off-by-one bug：Calmar 0.671→0.486（-28%）** |
| 07-20 | `ead005c` | **expanding-window 彻底消除 look-ahead → 真实 OOS** |
| 07-20 | `5c30172` | V7.11 区分截面/时序因子 IC：宏观择时 0.14-0.30 是 alpha 主力 |
| 07-20 | `2ac6b33` | V7.12 DCC 6 维时序特征 + regime overlay |
| 07-20 | `40e2d52` | V7.14 动态资产池（min_assets=10） |
| 07-20 | `53d6e5c` | **统一回测引擎（消除 v1-v7 8 个文件重复）** |
| 07-20 | `5f613f4` | BaseStrategy + StrategyEngine（最简策略引擎）|
| 07-20 | `c9eb84c` | **YAML 配置驱动 `run_from_yaml()`（6 个 YAML 模板）** |
| 07-20 | `fd61982` | **STAGE33 规划：新因子挖掘 + HMM Regime + 跨资产信号 + 代码清理** |
| 07-24 | `01b4f3c` | **V10 ETF 轮动策略（4 策略 + Vol-parity 组合）** |
| 07-24 | `cba81e4` | V10 统一日频 NAV + metrics 年化因子修复 |
| 07-24 | `7f401f6` | 全面重写 `nav_curves_html.py` + v10 加入所有图表 |
| 07-24 | `8646b25` → `c032612` | 24 → 16 → 14 → 9 策略精简版迭代 |
| 07-24 | `90bb853` | 添加 scripts 与 v8/v9/v10/cta 策略模块到 `strategy-modules` 分支 |

---

## 阶段成果总览

### 数据准备（详见 01_DATA_FOUNDATION.md）

- **数据源**：Tencent ETF + iFinD/Wind Proxy + 增强宏观（DXY/VIX/实际利率/FRED）
- **资产池演进**：7 ETF → 44 ETF → 51 ETF+5 bond → 56 assets（v7.4）→ 41 ETF（v7.0 B1 失败退回）
- **因子库（最终）**：12 宏观 + 24 量价 = 36 因子（v7.9 后）
- **数据 Pipeline 最后稳定原则**：
  1. 价格：ETF 主链使用前复权/事件修复后价格
  2. 频率：周频特征 + 日频执行
  3. 时序：严格 `X[t]→Y[t+1]`
  4. 缺失：动态资产池（min_assets=10）
  5. 标准化：宏观时序 Z-score + 量价截面 Z-score + Winsorize
  6. 评估：IC + 滚动 IC + Bootstrap + 多起点 + 缺失扰动 + expanding OOS
  7. 组合：所有派生 NAV 同日频指标口径
  8. 模型选择：v7.6 后向因子压缩、expanding、TV-PR/熵权混合演进

### 策略成果（详见 02_V0_V3.md / 03_V4_V6.md / 04_V7_V10.md）

| 阶段 | 关键里程碑 | 关键 OOS 指标 |
|------|------------|---------------|
| **V1.0 locked** | hybrid (price+slope_r2) + VT + Cost | **OOS Calmar 1.79 / Sharpe 1.51 / DD -1.94%** ⭐ |
| V2 baseline | Stage 12A hybrid + VT + Cost | OOS Calmar 0.892 |
| V3 equal | Stage 16A 多策略 1/3 等权 | OOS Calmar 0.504（架构先进业绩退化） |
| V4 IC² | 6 因子 IC² 加权（仅 value 稳定） | Calmar 0.613 |
| V4 LW | Stage 19 Ledoit-Wolf + λ 收缩 | Calmar 0.581（LW 不显著优于 IC²） |
| **V5.1** | 11 量价因子 + 逆波动率 | OOS Calmar 0.589（+20.7% vs V5） |
| V5.1.1 | 消融 S1+S3+S4 | OOS Calmar 0.604 |
| V6 (TF) | V5.1.1 + TF 风控（不加 VT/Cost） | **OOS Calmar 0.662** ⭐ |
| V6.1 IC12 | v5.1 + IC-IR expanding 12 月 | OOS Calmar 0.748 |
| V6.2 PROMISING | + Gram-Schmidt | OOS Calmar 0.901（→ DEPRECATED） |
| V6.2 (研究版本) | + 扣成本 + 起点 CV% | **Calmar 0.3310（-25.6%, CV% 56.9% FAIL）** |
| V7.0 C.Beta | 5 Macro Dynamic | 5-fold OOS Calmar 6.29 |
| V7.3 baseline | Symmetry+Bootstrap-Lasso+FRP | OOS Calmar 0.620 |
| V7.5 SL | + 硬止损 10% DD | OOS Calmar 0.597 (+20%) |
| V7.6 TV-PR | TV-PR 时变 LASSO（expanding）| **OOS Sharpe 1.57, Calmar 2.18** |
| **V7.10 final** | TV-PR expanding-window + 标准化 + Stop Loss | **CV% 16.6% PASS / OOS Calmar 1.121 (2023+)** |
| V10 DualMom | Antonacci GEM 模型 4 大类 | Sharpe 1.276 |
| **V10 4 策略 Vol-parity** | v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5% | **OOS Sharpe 1.991 / Sortino 2.842 / MaxDD -4.41%** ⭐⭐⭐ |

### 教训库总览（详见 05_LESSONS_LIBRARY.md）

| 类别 | 核心教训 |
|------|----------|
| **方法论类** | IC 信噪比低 / 简单规则常胜 / 多机制正交组合 > 单策略堆叠 / 反转效应在 A 股不存在 / Smart β 是风险溢价不是 alpha |
| **工程类** | Look-ahead bias 必须 4 步验证（严重程度→off-by-one→expanding→CV%）/ NaN-safe 计算 / 周频特征 vs 日频执行分频 / MACRO 沿时间复制 vs PV 沿截面填充 |
| **流程类** | "架构先进 ≠ 业绩进步"（V3 教训）/ "信号 + 风控不可独立可加"（slope_r2+VT 退步）/ "诚实回测必须严格无 look-ahead"（v5 0.712 含 bias）/ 起点 CV% < 25% 是硬性 P0 |

### SOP 总览（详见 06_RESEARCH_SOP.md）

研发链路 = **数据 → IC → 因子 → 选股 → 加权 → 风控 → 组合 → OOS → 硬化**

8 个阶段 + 5 道闸门：
1. 数据闸门：OHLCV 前复权、Proxy、动态资产池、缺数据审计
2. IC 闸门：单因子 IC、滚动 IC、去重、log 变换
3. 因子闸门：宏观时序 vs PV 截面、ADMM 正交化（慎用）
4. 选股闸门：sharp_signal + sub_strategy 单测
5. 加权闸门：等权/逆波动/IC/Vol-parity，单层验证
6. 风控闸门：VT/TF/Stop Loss/DCC overlay，正交叠加
7. OOS 闸门：single → 5-fold → walk-forward → expanding + CV%
8. 硬化闸门：起点依赖测试 / 死代码清理 / 文档 / 工厂函数

---

## 关键决策（用户视角）

1. **V3 多策略 1/N 等权失败**（`da7f588`）→ 转向粗粒度组合（v1.0 80% + v5 20%）
2. **V7.0 Phase B 41 ETF 池失败**（`59b0b64`）→ 退回 7/52 ETF 池
3. **V6.2 IC 加权 + 正交化从 PROMISING 降为研究版本**（`223ef65` + `8be00ae`）→ 起点 CV% < 25% 是硬门槛
4. **V7.7 树模型 ML 路线失败**（`a5de7f3` + `f22c407`）→ 信号质量问题，非模型问题
5. **V9 对称正交化失败**（`db4a852`，Sharpe 1.40→0.15）→ 只做因子去重 + log 变换，保留原始尺度
6. **V7.10 OOS 验证严格化**（`688862d`→`eaa6c9b`→`ead005c`）→ 4 步 OOS 验证流程标准化

---

## 最终生产首选

**4 策略 Vol-parity 组合**（`01b4f3c`）

| 子策略 | 权重 | 角色 |
|--------|------|------|
| **v1.0 locked** (hybrid + VT + Cost) | **74%** | 极致防御（Sharpe 1.285, MaxDD -1.94%）|
| **v9macro** (5 宏观 + 熵权 + 银风险控) | **12%** | 宏观择时（Sharpe 0.962）|
| **v7.10 TV-PR** (标准化 + expanding + SL) | **9%** | 激进 alpha（Sharpe 0.977）|
| **DualMom** (Antonacci GEM) | **5%** | 跨资产防御 |

- **OOS Sharpe 1.991** ⭐（单策略最高 v1.0 1.285 → 1.55 倍）
- **OOS Sortino 2.842**
- **OOS MaxDD -4.41%**
- **OOS Ann Ret 9.61%**
- target_vol = 0.08

策略生产化路径详见 `04_V7_V10.md §D.5/D.6` 和 `08_FUTURE_WORK.md`。

---

## 复盘文档导览

| 文档 | 内容 | 行数预估 |
|------|------|---------|
| `README.md` | 本目录索引 | 150 |
| `00_TIMELINE.md`（本文件）| V0→V10 全时间轴 + 一图概览 + 关键决策 | 350 |
| `01_DATA_FOUNDATION.md` | 数据准备全链路 | 1500+ |
| `02_V0_V3.md` | V0–V3 策略期 | 2000+ |
| `03_V4_V6.md` | V4–V6 策略期 | 2000+ |
| `04_V7_V10.md` | V7–V10 策略期 | 2500+ |
| `05_LESSONS_LIBRARY.md` | 教训库（方法论 / 工程 / 流程）| 1200+ |
| `06_RESEARCH_SOP.md` | 研发流程 SOP | 1000+ |
| `07_INVENTORY.md` | 关键资产索引 | 800+ |
| `08_FUTURE_WORK.md` | STAGE33 + 余下机会 | 600+ |
| **`09_QUICK_START.md`** ⭐ | **10 条金科玉律 + 决策树 + 提交清单（新人/自动研究必读）**| **400+** |

**核心素材源**：4 份 Subagent 调研报告（A 数据 / B V0-V3 / C V4-V6 / D V7-V10）+ `git log --all` 971 commit + 现实代码 / 报告 / 设计文档。

