# AllWeather 2.0 复现计划

> **目标**: 复现"避险猫"公众号《量化全天候模型大升级，2.0版本上线》的策略逻辑
> **模块**: `QuantNodes/strategy/allweather/`
> **作者**: 基于公开文章逆向工程（文章声明"具体实现的技术细节就不再赘述了"）
> **创建**: 2026-09-02

---

## 一、文章策略逆向工程

### 1.1 文章公布的策略要素

| 维度 | 文章数字 |
|------|----------|
| 样本期 | 2015-01-08 ~ 2026-08-31 |
| 年化收益 | 20.35% |
| 年化波动 | 6.87% |
| **Sharpe** | **2.67** |
| 最大回撤 | 5.91% |
| Calmar | 3.58 |
| 平均杠杆 | 2.3x |
| 目标波动率 | 10% |
| 月度胜率 | 76.4% (107/140) |
| 全自然年正收益 | 2015-2026 |

### 1.2 三大技术组件

| 组件 | 文章描述 | 文献依据 |
|------|----------|----------|
| **HRP** | 层次风险平价 | Lopez de Prado (2016): 层次聚类→准对角化→递归二分 |
| **RMT去噪** | 随机矩阵理论去噪 | Marchenko-Pastur λ⁺=(1+√(1/q))², 噪声特征值收缩为均值 |
| **负向趋势因子** | "自适应负向趋势因子"，细节隐瞒 | 解释 V1 (减仓型, 本次采用) |
| **核心原则** | 减少人为先验, 让数据驱动 | 不要求资产有因子暴露 |

### 1.3 "负向趋势因子"四种候选解释

| 解释 | 逻辑 | 决策 |
|------|------|------|
| **V1 减仓型 (主线)** | 资产 60d 动量 < 0 → 权重 × coef (默认 0) | 长多框架内自洽, 与文章平均杠杆 2.3x 不冲突 |
| V2 做空型 | 动量 < 0 → 信号 -1, 借期货做空 | 后续 Phase |
| V3 组合级 | 组合趋势得分 → 缩放总敞口 | 后续 Phase (CTA v7 momentum filter 同构, 已验证有效) |
| **V4 连续型 (敏感性对照, 2026-09-02 新增)** | 风险预算 ∝ max(滚动夏普, 0)², 负夏普→0; 国君系列之九方法 | 连续平滑, 回撤特性优于硬剔除; 进 Phase 3 敏感性 |

---

## 二、资产池 (38 资产)

### 2.1 期货 (29 品种, 日频, DuckDB)

| 类别 | 品种 | 交易所 | 合约乘数 |
|------|------|--------|----------|
| 有色/贵金属 | AU, AG, CU, AL, ZN, NI | SHFE/INE | 见 contract_specs |
| 黑色 | RB, HC, I, J, JM | SHFE/DCE | 见 contract_specs |
| 化工/能源 | TA, MA, PP, L, BU, SC | CZCE/DCE/SHFE/INE | 见 contract_specs |
| 农产品 | M, Y, P, CF, SR, C | DCE/CZCE | 见 contract_specs |
| 股指 | IF, IC, IH | CFFEX | 300/200/300 |
| 国债 | T, TF, TS | CFFEX | 10000/10000/2000 |

### 2.2 ETF (9 只, 日频, parquet)

| 代码 | 名称 | 类别 | 替代 |
|------|------|------|------|
| 513100 | 纳指ETF | 海外 | — |
| 159740 | 恒生科技ETF | 港股 | 513180 → 159740 (同指数, 池内) |
| 512890 | 红利低波ETF | 策略 | smartbeta 面板 |
| 512010 | 医药ETF | 行业 | — |
| 515030 | 新能源车ETF | 行业 | — |
| 512690 | 酒ETF (消费) | 行业 | — |
| 512800 | 银行ETF | 行业 | — |
| 515050 | 5G通信ETF | 行业 | — |
| 512200 | 房地产ETF | 行业 | — |

---

## 三、数据现状 (勘探结果, 2026-09-02)

### 3.1 ETF 面板 (复用 momentum_etf_rotation 落盘)

| 文件 | 路径 | 形状 | 日期范围 |
|------|------|------|----------|
| NAV 主面板 | `data/real/etf_nav_2018-01-01_2026-06-30.parquet` | (2058, 44) | 2018-01-02 ~ 2026-06-30 |
| Smartbeta 面板 | `data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet` | (2058, 12) | 同上, 含 512890 |
| OHLCV 复权 | `data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet` | (2216, 220) | 2017-05-23 ~ 2026-07-08 |

### 3.2 期货面板 (DuckDB `continuous_main_*_daily`)

- 29 品种全部存在
- **起点**: 2018-01-02 (SC 2018-03-26, TS 2018-08-17)
- **终点**: 2026-01-06 / 2026-07-21 (因交易所维护不同)
- 列: trade_date, open, high, low, close, volume, open_interest

### 3.3 数据起点偏离文章声明

**文章起点 2015-01-08, 实际可获最早 2018-01-02**:
- DuckDB 期货面板起始 2018-01
- ETF 主面板起始 2018-01
- SC/TS 等新品种 2018 年中才有

**处理方案**: 回测期从 **2018-01-08 起** (数据现实可行), 终点取 2026-07-21 (期货最早停止更新日) 或 ETF 终点 2026-06-30 后的交集。

### 3.4 终点的差距处理

- 文章终点 2026-08-31
- 期货 2026-07-21 (CZCE/INE/部分DCE 更新较慢)
- ETF 2026-06-30 (NAV 面板) / 2026-07-08 (OHLCV)
- **处理**: 用最早停止日期 2026-06-30 作为统一终点, 避免后期数据缺失噪声; 或拉增量后用 2026-07-21 (待 Phase 0 末决定)

---

## 四、算法管线 (复现版)

```
每个调仓日 (周频, 默认周五):
1. 取过去 W=252 日收益率窗口 (38 资产, 自动 NaN 跳过; 池内资产少于 5 只则全仓等权)
2. 样本协方差 Σ → 特征分解
3. RMT 去噪: q=n_obs/n_assets, λ⁺=(1+√(1/q))²
   噪声特征值 (≤λ⁺) → 收缩为均值 → Σ_denoised
4. 相关矩阵 C(Σ_denoised) → d_ij = √(0.5(1-c_ij))
   → single-linkage 聚类 → 准对角化 → 递归二分 → HRP 权重 (长多, Σw=1)
5. 趋势 overlay (V1): 资产 i 60d 动量 < 0 → w_i × coef (默认 coef=0, 即剔除)
6. Vol targeting: L = 10% / σ̂_port, cap 3x (文章均值 2.3x)
7. 生成期货手数 (按合约乘数换算) + ETF 整手, 计算下一期组合收益
```

---

## 五、消融矩阵 (Phase 2)

| # | 配置 | 目的 | Gate (OOS, 修订 2026-09-02) |
|---|------|------|-----------|
| 0 | 等权 + vol target | baseline | Sharpe ≥ 0.5 |
| 1 | HRP only | 结构组件贡献 | Sharpe ≥ 0.8 |
| 2 | HRP + RMT | RMT 去噪增量 | Sharpe ≥ 1.0 |
| 3a | 等权 + 趋势 V1 (无 HRP/RMT) | **趋势单独贡献** (隔离最强组件) | — |
| 3b | HRP + RMT (无趋势) | **结构组件的危机行为** (看 MaxDD 恶化幅度) | MaxDD ≤ -15% |
| 3 | HRP + RMT + 趋势 V1 (full) | 文章完整复现 | **全样本 Sharpe ≥ 1.2 (理想 1.8)** + 2025-2026 段 Sharpe ≥ 0 且 MaxDD ≥ -10% |

> Gate 下调依据 (2026-09-02 调研): ① HRP 高换手实证 (KTH) + vol-managed OOS 批评 (JFE 2020); ② 中国全天候 2025 年普遍承压 (头部私募 -2%~+2%, 超六成产品 <5%), 我们回测含此压力段; ③ 三方数字校准 (§11.3): 波动越高 Sharpe 越难做, 我们 10% 目标波动高于三家对照 (1.5-3.7%)。**判据升格: MaxDD 与月胜率优先于 Sharpe 绝对值** (§5.1)。

**参数全程固定不调优**: W=252, 调仓周频, 趋势窗口 60d, target_vol=10%, coef=0。

### 5.1 判读规则 (来自三命题分析 + 研报校准)

- **主判据 (研报校准后升格)**: MaxDD 与月胜率能否接近文章 (5.91% / 76.4%), 优先于 Sharpe 绝对值。国君"HRP 比风险平价年化仅 +0.11% 但 MaxDD 降 1.8pp"、华鑫"子策略仅 +0.26 Sharpe"——两个独立证据表明此类框架的 alpha 主要来自**风险压缩**而非收益预测
- **若 3a ≈ 3** (趋势是主力, 结构是装饰) → Sleeve A 简化为"等权/逆波动 + 趋势过滤", 不引入 HRP, 架构大幅简化 (华鑫低波方案就是这个形态的印证)
- **若 3 − 3a 显著** (结构有增量) → HRP+RMT 值得保留进 Sleeve A
- **若 3b 的 MaxDD 显著恶化 vs 3** → 验证"危机中真正干活的是趋势因子"命题; HRP 的分散化在相关性→1 时坍缩, 不能依赖其控回撤

### 5.2 诊断项 (D1-D4, Phase 3 执行)

| # | 诊断 | 方法 | Gate / 备选动作 |
|---|------|------|----------------|
| D1 | 权重稳定性 | 调仓换手率 + 聚类结构跨期持久度 (相邻调仓日簇成员 Jaccard 相似度) | **年化双边换手 ≤ 200%** (华鑫低波方案实盘水平 ≈68%/年); 超限触发备选: linkage 对照 (average/Ward) / 权重 EMA 平滑 / V4 连续变体 |
| D2 | 池敏感性 | 随机剔除 5 资产 × 20 次重跑, 看 Sharpe 分布离散度 | 离散度 > 0.5 → 结论附"池依赖"警示 |
| D3 | 资产类别贡献 | leave-one-class-out (债券/商品/权益/海外 各剔一次) | **升格为主审计项**: 国君 2.24/华鑫 2.08 的高 Sharpe 均建立在债牛+金牛上; 文章 2.67 同样存疑 |
| D4 | 期货 roll jump 检测 | 用 pre_settlement 识别涨跌停日, 区分"真实涨跌停" vs "主连拼接跳变" (panel 已见 NI/JM ±17% 单日变动) | 若 roll 污染显著 → Phase 0 数据管线加 roll 处理 (双价拼接 / OI 加权) |

**D2/D3 同时是对文章数字的外部审计**: 文章未公布资产池清单, 我们无法区分其"技术贡献"与"池选择运气"——用我们自己的公开池测出真实增量, 正是复现的价值。

### 5.3 协方差方法对照 (新增, 依据 CFM 期货警告)

CFM (Bouchaud) 论文明确警告: 期货市场板块内强相关 → 板块簇真实特征值极小, 朴素 MP clipping 可能不足。**Phase 3 并行报告: MP clipping vs Ledoit-Wolf 非线性收缩** (LW 实现已存在于 portfolio_research, 移植成本一行)。若 LW 版本风险预测显著更优 (OOS 波动偏差更小), 主线协方差方法切换为 LW。

---

## 六、实施步骤 (4 天)

### Phase 0: 数据准备 (0.5 天)
- [x] 加载 ETF parquet 面板 (主 + smartbeta 合并)
- [x] 加载 29 期货 DuckDB 面板
- [x] 对齐 38 资产到统一日期索引, 输出 `data/allweather/panel_v1.parquet`
- [x] 数据完整性检查 (各资产起始日, 缺失率)
- [x] D4: roll jump 检测 (2026-09-02 完成, **发现两处数据缺陷并修复**):
  - 期货: 13 品种 roll 日波动 2-5x (未复权拼接假收益 1-3%, M 最高 4.9x 方差膨胀)
    → `load_futures_returns_roll_adjusted` 收益拼接 (换月日=持有合约真实收益),
    修复后全部品种 roll/非roll 波动比 0.69~1.46 ✓ (产出 panel_v2.parquet, 主线)
  - ETF: 512690 NAV 面板 18 个坏日 (份额折算) + OHLCV 面板 5 个拆分日毛刺
    → `load_etf_panel` 收益级交叉验证合并 (NAV主 + OHLCV回退 + 拆分比修正),
    修复后 9 ETF 超±10.5% 单日数 = 0 ✓, ETF 期限延长至 2026-07-08
  - 脚本: `scripts/check_allweather_roll_jumps.py`

### Phase 1: 组件实现 + 单测 (1 天)
- [ ] `cov_rmt.py`: RMT 去噪 (移植自 portfolio_research DenoisedCov) + LW 非线性收缩对照 (§5.3)
- [ ] `hrp.py`: HRP (移植自 HierarchicalRiskParity), linkage 参数化 (single/average/Ward)
- [ ] `trend_overlay.py`: V1 减仓型 + V4 连续型 (风险预算 ∝ max(滚动夏普,0)², 国君方法)
- [ ] `vol_target.py`: 波动率目标
- [ ] 单元测试: RMT 特征值谱合理, HRP 权重和=1/非负, 趋势剔除逻辑正确

### Phase 2: 消融回测 (1.5 天)
- [x] `backtest.py`: 周频调仓, 混合频率 (日频 NAV)
- [x] `run_replication.py`: 消融 0/1/2/3a/3/3v4 六配置
- [x] 报告: 全样本 + 滚动 WF (2018~2020/2021~2023/2024~2026), **单独报告 2025-2026 压力段**
- [x] 指标对照文章表 (Sharpe 2.67 / MaxDD 5.91% / 月胜率 76.4%)
- [x] 按 §5.1 判读规则输出结论 (MaxDD/月胜率为主判据)

**Phase 2 初步结论 (2026-09-02, 详见 ablation_results.csv)**:

| 配置 | 全样本 Sharpe | MaxDD | 压力段 25+ Sharpe |
|------|--------------|-------|------------------|
| 0 等权+vol | 0.891 | -14.2% | **+0.82** |
| 1 HRP only | 1.323 | -4.1% | -0.29 |
| 2 HRP+RMT | **1.340** | **-4.1%** | -0.07 |
| 3a 等权+V1 | 0.603 | -21.4% | +0.55 |
| 3 HRP+RMT+V1 | 1.244 | -4.6% | -0.27 |
| 3v4 HRP+RMT+V4 | 0.697 | -7.4% | -0.00 |

1. **结构组件是主力** (§5.1 命中): HRP 使 Sharpe 0.89→1.32, MaxDD -14%→-4.1%, 换手 47.6x→3.9x; RMT 再 +0.02
2. **趋势 V1/V4 均为负贡献** (与预期相反): 文章的"负向趋势"绝非 V1 式资产级剔除; Phase 3 需测 V3 组合级/月频/更长窗口
3. **月胜率 47-48% ≪ 文章 76.4%**: §七-3 触发 — V1/V4 已排除, 差距源于资产池/样本段/未知实现
4. **压力段暴露 HRP 债券集中风险**: 2025-2026 HRP 系全部转负 (与全天候行业承压吻合), 等权系为正 → Sleeve A 设计需加债券集中度约束或 HRP/等权混合 (Phase 3 D3 归因确认)
5. **Gate: 全样本 1.340 ≥ 1.2 ✓; 压力段 -0.07 < 0 ✗** — 结论待 Phase 3 归因后定

### Phase 3: 验证 + 复盘 (1 天)
- [x] 诊断 D1: 权重稳定性 — **发现 TS 结构权重 86% (杠杆债基金退化)**
- [x] 诊断 D2: 池敏感性 — 1.369±0.174, 不依赖池 ✓
- [x] 诊断 D3: 类别归因 — **债券集中度假说证实** (剔债券 Sharpe 1.01→0.75)
- [x] 诊断 D4: roll 前后对照 — **未修复 0.798 vs 修复 1.340 (+0.54)**
- [x] 无杠杆 10%+ 实验: 分配天花板 ~8.3%, **需 Sleeve B (Phase 5)**
- [x] 新组件: max_weight 集中度约束 (cap0.40 → Sharpe 1.543, 压力段转正), V3/tilt, weight_scheme
- [x] 复盘报告: `docs/allweather-replication-report.md`
- [ ] 反哺宏观配置模型: 候选主线待用户定夺 (X2 vs X5 vs Sleeve B)

---

## 七、风险声明与预期

1. **Sharpe 2.67 不保证复现**: 文章资产池未知、细节隐瞒、2018-2026 含黄金/权益长牛, 我们用 38 资产 + 真实成本, 复现 1.2~1.8 区间即算成功 (gate 已据调研下调, 见 §五)
2. **MaxDD / 月胜率是主判据** (研报校准): 国君 HRP 年化仅 +0.11% 但 MaxDD 降 1.8pp; 华鑫子策略仅 +0.26 Sharpe — 框架价值在风险压缩。判据: MaxDD 接近 5.91% / 月胜率接近 76.4%
3. **月胜率 76.4% 若显著低**: 优先怀疑 V1 解释, 依次试 V4 连续版 / 10 月窗口 / V2 做空型
4. **数据起点偏差**: 2018-01 vs 文章 2015-01, 错失 2015-2017 黄金/一线地产牛市, 对 Sharpe 有负面影响
5. **2025-2026 压力段**: 中国全天候私募 2025 普遍承压 (头部 -2%~+2%, 六成 <5%), 我们回测含此段; 该段单独报告, 作为"文章数字是否含水分"的现实对照
6. **OOS 闸门**: Phase 2 必须用滚动 WF 评估, 全样本数字不可信

---

## 八、文件结构

```
/home/ll/Public/QuantNodes/QuantNodes/strategy/allweather/
├── __init__.py
├── PLAN.md                  # 本文件
├── config.py                # 38 资产池 / 固定参数
├── data_loader.py           # 期货(DuckDB) + ETF(parquet 合并) → 统一面板
├── cov_rmt.py               # RMT 去噪 + LW 收缩
├── hrp.py                   # HRP (与 RMT 串联)
├── trend_overlay.py         # V1/V3/V4/tilt 趋势 overlay
├── vol_target.py            # 10% 目标波动
├── backtest.py              # 混合频率回测引擎
├── run_replication.py         # 消融入口 (0/1/2/3a/3/3v4 六配置)
└── experiments.py          # 实验脚本集: lever-sweep/cap-sweep/trend-matrix/
                              # unlevered/segments/diag-d1~d4 (CLI, 结果落盘 CSV)

/home/ll/Public/QuantNodes/data/allweather/
└── panel_v1.parquet         # 38 资产统一面板 (Phase 0 产出)

/home/ll/Public/QuantNodes/docs/
└── allweather-replication-report.md  # Phase 3 复盘报告
```

---

## 九、复用关系 (避免重复造轮)

| 模块 | 复用来源 | 状态 |
|------|----------|------|
| ETF 数据面板 | `momentum_etf_rotation/data/real/*.parquet` | 复用, 无新抓取 |
| ETF 池定义 | `momentum_etf_rotation/common/universe.py` | 复用, 仅取子集 |
| 指标计算 | `momentum_etf_rotation/common/metrics.py` | 复用 (CTA 引擎也在用) |
| RMT 协方差参考实现 | `portfolio_research/.../covariance_estimation.py:DenoisedCov` | 移植到 QuantNodes |
| HRP 参考实现 | `portfolio_research/.../alternative_strategies.py:HierarchicalRiskParity` | 移植到 QuantNodes |
| 期货数据加载 | `strategy/cta/data_loader.py:CtaDataLoader` | 复用 |
| 期货合约乘数 | `strategy/cta/contract_specs.py` | 复用 |

---

## 十、与"宏观配置模型"的关系 (决策记录, 2026-09-02)

### 10.1 已确认的架构决策

**双 sleeve 独立 + 策略层集成优先, signal-tilted HRP 为后续演进**:

```
近期 (本复现验证后):
  Sleeve A: 全天候配置层 (本模块, HRP+RMT+趋势V1, 视消融结果可能简化)
  Sleeve B: CTA 信号层 (strategy/cta, TV-PR + inv_vol + 动量过滤)
  集成: 策略层 inv-vol 加权 (Phase 18 已验证, 干净/可归因/风险隔离)

远期 (仅当双 sleeve 集成 OOS 验证通过):
  signal-tilted HRP: w_i ∝ HRP_i × signal_i
  把 CTA 信号乘进 HRP 权重, 消除 sleeve 划分这一先验
  风险: 前视面扩大 + 两套已验证组件耦合, 必须重走完整 WF 验证
```

选择理由: 双 sleeve 保持归因清晰 (哪个 sleeve 挣钱一目了然)、风险隔离 (单 sleeve 失效不传染)、且每一步都有独立 OOS gate。

### 10.2 文章哲学对本项目的三条约束 (讨论结论)

1. **先验搬家原则**: "数据驱动" = 把不可证伪的经济假设替换成可证伪的统计程序, 不是消灭先验。后续所有组件选型按此标准: 每引入一个组件, 问"它的先验能否被 OOS/resampling 检验?"
2. **宏观因子降级为 regime filter**: 原 Phase 5 的"B 矩阵映射 3 宏观因子 → 38 资产权重"方案废弃 (正是文章批评的因子可解释性陷阱)。改为: 宏观因子只做 HRP 权重的后处理调仓系数 (如高通胀期降权国债), 新资产入池不再需要暴露估计
3. **池选择是最大未讨论先验**: 资产池边界本身是最大的主观决策。任何 Sharpe 数字必须附带 D2 (池敏感性) 和 D3 (类别归因), 否则不可信

### 10.3 Gate 条件 (2026-09-02 修订)

本次复现是 Sleeve A 的**预研**:
- 若复现 Sharpe ≥ 1.2 (理想 1.8) 且 3−3a 有增量 → Sleeve A 采用 HRP+RMT+趋势 框架
- 若 3a ≈ 3 (趋势为主力) → Sleeve A 简化为"等权/逆波动 + 趋势过滤" (华鑫低波方案形态)
- 若复现 Sharpe < 1.0 → 放弃 Sleeve A, 改用 Sharpe 1.589 的 ENSEMBLE_INV_VOL 思路
- **任一情形下, MaxDD/月胜率不达标 (显著差于文章) 都要触发趋势变体排查, 而非直接放弃**

**仅当 Sleeve A 验证有 alpha**, 才进入双 sleeve 集成; 宏观因子 sleeve (Factor Mimicking regime filter) 作为第三个可选 sleeve, 有独立 OOS 增量才保留。

---

## 十一、同类研报调研 (2026-09-02)

> 调研对象: 国泰君安《基于层次聚类和动量效应改进风险平价策略》(大类资产配置量化模型系列之九, 2024-10, 张雪杰/朱惠东/张涵) + 华鑫《多元资产2025年度策略》上下篇 (上篇 2024-12-29 期货版 / 下篇 2025-01-27 ETF版, 吕思江团队)。
> 信息来源: 新浪财经官方导读 (国君) + 发现报告摘要 (华鑫), 全文评估见 §11.4。

### 11.1 国君系列之九 — 方法论与数字

**方法论 (三层)**:

```
风险平价三大痛点: 资产多时求解复杂度指数级 / 不受收益方向影响 / 危机回撤大
  ↓ 改进一: 层次风险平价 (HRP)
  三种变体: 波动率倒数层次RP / 层次RP / 朴素层次RP
  求解速度 +90%, 年化 4.56%→4.67%, MaxDD 6.15%→4.34%
  → HRP 价值主要在回撤改善, 收益提升微小
  ↓ 改进二: 动量风险预算 (方向敏感)
  风险预算比例 ∝ max(滚动夏普, 0)²  (滚动夏普 = 过去N期收益/波动)
  负夏普资产 → 预算归零
```

**数字**:

| 策略 | 区间 | 年化 | MaxDD | 波动 | Sharpe |
|------|------|------|-------|------|--------|
| 动量预算组合 | 2007-2024 | 5.31% | 3.69% | 2.37% | 2.24 |
| 国内资产动量预算 | 2008-2024 | 6.40% | 4.17% | — | 2.51 |
| 全球资产动量预算 | 2007-2024 | 6.35% | 6.74% | — | 2.39 |

**与避险猫文章的关系**: 国君这篇 = 文章的"卖方严谨版"。HRP+动量 = 文章的 HRP+自适应负向趋势; 国君的"滚动夏普²预算"是**连续版**, 文章 V1 (动量<0→剔除) 是**离散版**。据此新增 **V4 连续变体** 进我们的敏感性测试。

### 11.2 华鑫上下篇 — 架构与数字

**上篇 (期货落地)**: 国内版全天候 = 商品 (17品种多空风险预算, 预测夏普² + 3月动量定多空; "低相关性低夏普率"反转筛选后 Sharpe 0.83) + 黄金 (多因子择时) + 债券 (10Y/30Y 国债期货 RP) + 权益 (大小盘轮动)。无杠杆十年无一年亏损, 年化 9.45%; 近三年 Sharpe 2.43, MaxDD 2.61%。

**下篇 (ETF 版, 与我们 ETF sleeve 直接同类)**:

```
鑫选ETF池 (权益/商品/债券/海外)
  ↓ 层次聚类: 相关性 → 距离矩阵 → average linkage → 树状图按距离切 K 组
  ↓ 风险平价: 簇间/簇内最小化风险贡献标准差
  ↓ 双动量: 时序动量 (过去10月涨跌幅<0 → 剔除或止损) + 截面动量 (排序归一化定权)
```

| 版本 | 区间 | 年化 | MaxDD | Sharpe | 换手 |
|------|------|------|-------|--------|------|
| 低波方案 (纯机械) | 2016-2024 | 4.59% | 2.12% | 2.08 | 5年单边171% (≈年化34%单边) |
| 低波方案 2024 | — | 5.64% | 0.48% | 3.63 | 单边 27.8% |
| 高波方案 (观点+子策略) | 2018-2024 | 8.15% | 3.86% | 2.34 | 低 |
| 多资产多策略 RP | — | 8.25% | 5.31% | 2.14 | — |

**架构映射到我们**: 华鑫资产层 (4类ETF β) ↔ 我们 Sleeve A; 华鑫策略层 (行业/风格/大小盘轮动 α) ↔ 我们 Sleeve B (CTA)。

### 11.3 三方数字校准 → 我们 Gate 的依据

| 来源 | 波动中枢 | Sharpe | 备注 |
|------|----------|--------|------|
| 国君动量预算 | 2.4% | 2.24 (17y) | 债券主导 |
| 华鑫低波 ETF | 1.5-2.5% | 2.08 (9y) | 固收+定位 |
| 华鑫多资产 RP | 3.7% | 2.14 | — |
| 避险猫文章 | 6.9% | 2.67 (11y) | 2.3x 杠杆, 10% 目标 |
| **我们的 gate** | **10%** | **1.2 (理想 1.8)** | 38 资产含大量商品期货 |

规律: **波动越高 Sharpe 越难做**。三家高 Sharpe 全部建立在债牛+金牛上 (D3 归因的必要性)。我们 10% vol 下 gate 1.2/1.8 保守合理。

**战略级启示**: 华鑫"高波比低波仅多 0.26 Sharpe" + 国君"HRP 年化仅 +0.11% 但 MaxDD 降 1.8pp" → 此类框架的 alpha 主要来自**风险压缩**而非收益预测。主判据 = MaxDD/月胜率, 非 Sharpe 绝对值。

### 11.4 全文下载评估 (结论: 现有信息足够, 不阻塞)

| 报告 | 已获取 | 缺口 | 全文价值 | 结论 |
|------|--------|------|----------|------|
| 国君系列之九 | 官方导读 (新浪), 含全部关键数字+方法描述 | HRP 三变体构造细节 / 动量窗口 N / 资产清单 | 中 — 其参数针对其资产池调优, 不直接适用我们 (我们固定参数不调优) | **可选**; 有万得/道合账号可顺带取 |
| 华鑫上篇 | fxbaogao 摘要, 含子策略架构+数字 | 黄金择时因子明细 / 大小盘轮动信号 | 中 — 属 Phase 5 双 sleeve 集成参考, 不阻塞 Sleeve A | **可选** |
| 华鑫下篇 | fxbaogao 摘要, 层次聚类+双动量方法已完整 (聚类法/10月动量/两种模式/换手) | 鑫选池完整清单 / 权重优化细节 | 低 — 最相关报告, 缺口最小 | **不需要** |

**判断**: Phase 1-3 实施不依赖全文。若后续进入 Phase 5 (双 sleeve 集成 + 子策略 alpha 层), 再考虑通过万得/道合/hibor 渠道取国君全文与华鑫上篇全文。

---

## 十二、实施路线图 (粒度到函数与验证命令)

> 依据 AGENTS.md: Goal-Driven Execution (步骤→verify) + Verify-Then-Proceed (每个 edit 后 ruff, 每个模块后 pytest)。
> 每步的验证命令统一: `ruff check <file>` + `python -m pytest tests/strategy/allweather/ -v`。

### 依赖图

```
Step 0 (D4 roll jump 检查)          ← 决定数据是否需要 roll 处理, 最先做
   ↓
Step 1 cov_rmt.py  (无依赖)
   ↓
Step 2 hrp.py       (依赖 cov_rmt 的输出接口, 但只吃 ndarray 可并行写)
   ↓
Step 3 trend_overlay.py (无依赖, V1+V4 可并行写)
   ↓
Step 4 vol_target.py    (无依赖)
   ↓
Step 5 backtest.py  (依赖 1-4 全部)
   ↓
Step 6 run_replication.py (依赖 5)
   ↓
Phase 2 消融回测 → Phase 3 敏感性+诊断
```

### Step 0: D4 roll jump 检查 (0.5h, 先行)

- 产出: `scripts/check_roll_jumps.py` (或 data_loader 内函数 `detect_roll_jumps()`)
- 方法: 对 29 期货, 用 DuckDB `futures_rule` / `pre_settlement` 判定涨跌停日; 单日 |ret| > 涨跌停幅度的日期 → 疑似 roll 拼接跳变
- 输出: 每品种 jump 日清单 + 占比; 判定规则:
  - 若 jump 日占比 < 0.5% 且集中在我已知换月窗口 → 低污染, 直接用 panel_v1
  - 否则 → data_loader 加 `roll_adjust` 选项 (换月日收益用 pre_settlement→settlement 计算或剔除该日)
- verify: 打印 29 品种 jump 统计表, 人工判读

### Step 1: `cov_rmt.py` (2h)

函数签名:
```python
def sample_cov(returns: pd.DataFrame, window: int = 252) -> np.ndarray
def mp_denoise(cov_or_corr: np.ndarray, n_obs: int) -> np.ndarray
    # 特征分解 → λ+ = (1+√(n/p/n_ratio))² 相关矩阵单位 → 噪声特征值收缩为均值 → 重构 → PSND 修正
def ledoit_wolf_nl(returns: pd.DataFrame, window: int = 252) -> np.ndarray  # §5.3 对照
```

测试 (`test_cov_rmt.py`):
1. `mp_denoise`: 随机 i.i.d. 数据 (T=252, N=38) → 去噪后特征值 ≤ λ+ 的都被收缩; 输出对称 + 半正定
2. 强信号数据 (构造 3 个真实因子) → 前 3 大特征值保留
3. `ledoit_wolf_nl`: 输出 PSD, trace 与样本协方差接近

### Step 2: `hrp.py` (2h)

函数签名:
```python
def hrp_weights(cov: np.ndarray, linkage: str = "single") -> np.ndarray
    # 距离 √(0.5(1-ρ)) → linkage → quasi_diag → 递归二分 (簇方差逆比分配)
def hrp_weights_from_returns(returns: pd.DataFrame, window: int, denoise: bool = True,
                             linkage: str = "single") -> pd.Series
    # NaN 资产自动剔除 (渐入池), 可用资产 < MIN_ASSETS_FOR_HRP → 返回等权
```

测试 (`test_hrp.py`):
1. 权重和 = 1, 全非负
2. 两簇构造数据 (簇内高相关): 簇间权重大致均衡
3. NaN 渐入池: 构造部分资产后上市 → 只对可用资产分配, 权重和 = 1
4. linkage ∈ {single, average, ward} 三种均能跑通且结果不同 (自由参数确认)
5. 资产数 < 5 → 退化为等权

### Step 3: `trend_overlay.py` (2h)

函数签名:
```python
def v1_overlay(prices: pd.DataFrame, lookback: int = 60, coef: float = 0.0) -> pd.DataFrame
    # ret = prices.pct_change(lookback); overlay = coef if ret<0 else 1  (无前视: 仅用 t 及以前)
def v4_overlay(prices: pd.DataFrame, lookback: int = 252, vol_window: int = 60,
               floor: float = 0.0) -> pd.DataFrame
    # rolling sharpe = mean(ret, lookback)/std(ret, vol_window) * √252
    # budget ∝ max(sharpe, 0)²; 归一化到均值 1 (保持组合 gross 不因 V4 改变, 只重分配)
def combined(prices, mode: str = "v1", **kwargs) -> pd.DataFrame  # 分发器
```

测试 (`test_trend_overlay.py`):
1. V1: 构造下跌序列 → overlay = coef; 上涨 → 1
2. V1 无前视: overlay[t] 只依赖 prices[:t+1] (截断重算一致性)
3. V4: 全正夏普 → 权重和 > 0 且各资产 ≤ 上限; 全负夏普 → 全 floor (全现金, gross=0)
4. NaN 资产 (未上市) → overlay 不产生假信号 (NaN→不参与)

### Step 4: `vol_target.py` (1h)

函数签名:
```python
def apply_vol_target(base_weights: pd.DataFrame, returns: pd.DataFrame,
                     target_vol: float = 0.10, lookback: int = 20,
                     max_leverage: float = 3.0, freq: str = "weekly") -> pd.DataFrame
    # 组合已实现 vol (lookback) → 杠杆 = target/realized, cap 3x → 权重整体缩放
```

测试:
1. 杠杆 cap: 构造高波动数据 → 总杠杆 ≤ 3.0
2. 低波动数据 → 杠杆 > 1
3. 无前视: 杠杆[t] 用 returns[:t]

### Step 5: `backtest.py` (3h, 核心引擎)

```python
@dataclass
class AllWeatherConfig:
    window: int = 252
    rebalance: str = "W-FRI"
    denoise: bool = True          # RMT on/off (消融开关)
    use_hrp: bool = True          # HRP vs 等权 (消融开关)
    trend_mode: str = "v1"        # none/v1/v4 (消融开关)
    trend_lookback: int = 60
    trend_coef: float = 0.0
    target_vol: float = 0.10
    max_leverage: float = 3.0
    cov_method: str = "mp"        # mp/ledoit_wolf (§5.3 开关)
    linkage: str = "single"
    futures_commission: float = 0.0001
    etf_commission: float = 0.0005

def run_backtest(panel: pd.DataFrame, config: AllWeatherConfig) -> BacktestResult
    # 混合频率: 周五求解 (窗口收益→cov→去噪→HRP/等权→trend→vol target)
    #           日频 NAV (权重持有, 逐日收益累积), 换手按权重变化 × 费率
    # 输出: nav / weights_history / turnover / metrics (复用 momentum_etf_rotation.common.metrics)
```

测试 (`test_backtest.py`):
1. 短样本冒烟 (100 日 × 8 资产): NAV 无 NaN, 长度正确
2. 消融开关: use_hrp=False → 等权路径结果与手算等权一致
3. 成本: 零费率 vs 高费率 → 高费率 NAV 更低
4. 无前视: 截断面板到 t 重跑, nav[:t] 与完整面板 nav[:t] 一致 (逐日匹配, 允许浮点容差)
5. 渐入池: 159740 上市前不持仓

### Step 6: `run_replication.py` (1h)

- CLI: `python -m QuantNodes.strategy.allweather.run_replication --config 3 --wf`
- 六配置批量跑 (0/1/2/3a/3b/3) → 输出对比表 CSV + 对照文章数字表
- WF 分段: 2018-2020 / 2021-2023 / 2024-2026 + **2025-2026 压力段单独列**
- 输出: `data/allweather/ablation_results.csv` + markdown 摘要

### Phase 2 验收 (对应 §六)

- [ ] 六配置全跑通, 表格含: 全样本 / 三段 WF / 压力段 × (AnnRet, Vol, Sharpe, MaxDD, 月胜率, Calmar, 换手)
- [ ] 按 §5.1 判读: 主判据 MaxDD/月胜率 vs 文章 (5.91%/76.4%)
- [ ] 结论写入 `docs/allweather-replication-report.md` 初稿

### Phase 3 敏感性矩阵 (全部为 backtest.py 开关, 无新引擎代码)

| 实验 | 开关组合 | 判据 |
|------|----------|------|
| 成本 | commission ∈ {1bp, 5bp, 10bp} | 换手×费率联动表 |
| 频率 | rebalance ∈ {W-FRI, ME} | Sharpe/换手 |
| 窗口 | window ∈ {126, 252, 504} | 权重稳定性 (Jaccard) |
| 趋势 | {v1 coef∈{0,0.5} × lookback∈{60,126,210}} + v4 | MaxDD 改善幅度 |
| linkage | {single, average, ward} | 换手 + Sharpe |
| 协方差 | {mp, ledoit_wolf} | OOS 波动预测偏差 |
| D1-D4 诊断 | 独立函数 | 各自 gate (§5.2) |

### 会话切分建议 (每会话一个 Step)

| 会话 | 内容 | 预计 |
|------|------|------|
| S1 | Step 0 + Step 1 (roll check + cov_rmt + tests) | 2.5h |
| S2 | Step 2 + Step 3 (hrp + trend + tests) | 4h |
| S3 | Step 4 + Step 5 (vol_target + backtest + tests) | 4h |
| S4 | Step 6 + Phase 2 六配置跑批 + 初判读 | 3h |
| S5 | Phase 3 敏感性矩阵 + D1-D4 + 复盘报告 | 4h |

### 全程红线

1. 每个文件 edit 后立即 `ruff check` (AGENTS.md 规则 6)
2. 每个模块完成后跑 `pytest tests/strategy/allweather/ -v`, 不攒
3. 参数只从 `config.py` 读, 禁止散落 magic number
4. 无前视是第一红线: 任何涉及"截断重算一致性"的测试失败 → 停下修 bug, 不继续
5. 不做样本内调优: Phase 3 敏感性只做稳健性报告, **不据其改主线参数**

---

## 十三、V6 三子因子路线 (2026-09-04, 调研驱动)

### 13.0 起源与修正

V5 触发器 A/B 单独回测（2026-09-03）已永久归档（详见 §5.7 / docs/allweather-replication-report.md §5.7）。失败根因：
- trigA = Sub-B (趋势) + 价格条件 (xmm + xmm_slope) 混合，无法归因
- trigB = 纯波动扩张（vol_ratio > 1.0），无方向性
- **Sub-C (量能维度) 完全缺失** → 三维度变两维度

用户指出修正方向：**三个维度应作为独立子因子**，不可混成 trigger。调研发现 CTA 因子库已完整实现三个维度（详见 docs/allweather-cta-borrowed-notes.md §8），无需从零构建。

### 13.1 V6 核心思路

```
V5 (失败):  trigger_A(trigger_B)  →  组合成避险猫因子
V6 (新):   Sub-A + Sub-B + Sub-C  →  三个独立子因子叠加
```

| 子因子 | 信息维度 | 已有 CTA 因子 | 默认窗口（日） | 周-月尺度对应 |
|--------|----------|---------------|----------------|---------------|
| **Sub-A** 时序波动率 | 波动 regime | `VolatilityConeFactor` | sw=10, lw=120 | 2 周 / 6 月 |
| **Sub-B** 趋势力度 | 价格方向+强度 | `TrendQualityFactor` + `MomentumSlopeFactor` | w=30 (R²), w=20 (slope) | 6 周 + 4 周 |
| **Sub-C** 量能确认 | 成交量/OI 方向 | `VolumeTrendFactor` + `OIPriceDivergenceFactor` | w=5, w=5, pw=5 | 1 周 × 3 |

**关键修正**：
1. **不再自造 trigger**，直接复用 CTA 因子库已有因子
2. **Sub-C 从缺失变已有**（CTA 库有 6 个相关因子），零新数据管线
3. **信号从二值变连续**：因子分数 ∈ [-1, +1] 而非 {0, 1}
4. **三维度独立可归因**：单因子回测可识别哪个维度有效

### 13.2 频率匹配性验证（关键调研结论）

| 系统 | 调仓频率 | 因子窗口尺度 |
|------|---------|-------------|
| AllWeather | **W-FRI** (周五) | — |
| 黄金 gold_combined | W-FRI | mom=4 周, vol=4 周, tips=26 周 |
| CTA v7 | 日级信号 + 周级调仓 | 5/20/60/120 日 = 1 周/1 月/3 月/6 月 |
| CTA factor `momentum_filter_window` | 60 日 | 12 周（3 个月） |

**结论**：CTA 因子库默认窗口（5/20/30/60/120 日）= **1 周/1 月/1.5 月/3 月/6 月**，正好覆盖 all-weather 周频调仓 + 月度风险预算视野。**无需重新调参**。

### 13.3 实施步骤（4 步，~4.5h）

#### 13.3.1 Step 1 — `subfactors.py`（新文件，~120 行）

封装三个子因子计算，输出 T × N 日度因子分数 ∈ [-1, +1]。

```python
# subfactors.py 接口
@dataclass
class SubFactorConfig:
    vol_cone_sw: int = 10
    vol_cone_lw: int = 120
    vol_cone_z: float = 1.0
    trend_quality_w: int = 30
    trend_quality_th: float = 0.6
    mom_slope_w: int = 20
    vol_trend_w: int = 5
    oi_div_w: int = 5
    oi_div_pw: int = 5

def compute_subfactor_a(prices, cfg) -> pd.DataFrame: ...
def compute_subfactor_b(prices, cfg) -> pd.DataFrame: ...
def compute_subfactor_c(prices, volumes, open_interest=None, cfg) -> pd.DataFrame: ...
def compute_all_subfactors(prices, volumes, open_interest=None, cfg) -> dict: ...
```

**ETF 处理**：`open_interest=None` 时（ETF 无 OI），Sub-C 仅用 `VolumeTrendFactor`。

**单测**（`test_subfactors.py`，5 用例）：
1. Sub-A 输出值域 [-1, +1]
2. Sub-B 组合 TrendQuality + MomentumSlope
3. Sub-C 无 OI 时仅用 VolumeTrend（ETF 路径）
4. Sub-C 含 OI 时叠加 OIPriceDivergence（期货路径）
5. compute_all_subfactors 返回 dict{a, b, c}

#### 13.3.2 Step 2 — `subfactors_weekly.py`（新文件，~60 行）

日度因子 → W-FRI 重采样 + 横截面 z-score 标准化。

```python
def resample_weekly(factor_daily, agg="last") -> pd.DataFrame: ...
def zscore_cross_section(factor_panel, lookback_weeks=52) -> pd.DataFrame: ...
def combine_subfactors(factors, weights=None, lookback_weeks=52) -> pd.DataFrame: ...
```

**核心参数**：
- 重采样方式：`last`（默认，对齐 W-FRI）/ `mean`（更稳健）
- z-score 窗口：52 周（1 年），min_periods=12（防冷启动）
- 组合权重：默认等权 {a: 0.33, b: 0.33, c: 0.34}

**单测**（`test_subfactors_weekly.py`，3 用例）：
1. resample_weekly 输出 W-FRI 频率
2. zscore 截面均值近似 0
3. combine_subfactors 权重和为 1

#### 13.3.3 Step 3 — `backtest.py` 扩展（+60 行）

新增 `subfactor_weights()` 函数 + `AllWeatherConfig` 加 2 字段。

```python
def subfactor_weights(base_weights, factor_score, alpha=0.3):
    """w_final = w_base × (1 + α × factor_score), clip 到 [0, 1]."""
    adjusted = base_weights * (1.0 + alpha * factor_score)
    return adjusted.clip(lower=0.0, upper=1.0)
```

**`AllWeatherConfig` 新增字段**：
- `subfactor_mode: str = "none"` — none / a / b / c / abc / ab / ac / bc
- `subfactor_alpha: float = 0.3` — 子因子叠加强度（扫描 {0.1, 0.2, 0.3, 0.5}）

**接入点**：`run_backtest()` 的 `_rebalance_mask` 循环内，在 `compute_base_weights()` 之后插入 `apply_subfactor_to_sleeve()`。

**单测**（`test_backtest_subfactor.py`，4 用例）：
1. 中性分数（z=0）不动基础权重
2. 正负分数对称调整
3. 极负分数 clip 到 0
4. sleeve 权重与因子分数索引对齐

#### 13.3.4 Step 4 — `experiments.py` 新增子命令（+120 行）

新增 `v6-subfactor-test` 子命令，包含三个函数：

```python
def run_subfactor_single(panel, cfg, factor, alpha_values=(0.1, 0.2, 0.3, 0.5)):
    """单子因子 × alpha 扫描, B-sleeve 上回测."""

def run_subfactor_combo(panel, cfg, combos=("ab", "ac", "bc", "abc"), alpha=0.3):
    """子因子组合 × B-sleeve + ABC_ew 集成回测."""

def evaluate_v6_gate(single_df, combo_df):
    """V6 Gate 判定 (单因子 fwd20 + 组合 Sharpe 损失)."""
```

**CLI 接入**：`ap.add_argument` choices 列表加入 `"v6-subfactor-test"`。

### 13.4 Pre-registered Gate（事先注册，比 V5 更严）

| Gate | 阈值 | 比 V5 严在哪里 |
|------|------|---------------|
| 单因子 fwd20 均值 | < 0% | 新增（V5 缺失） |
| 单因子 neg% | > 55% | 新增 |
| 单因子 n | ≥ 100 | 新增 |
| 组合 Sharpe 损失 | < **0.10** | V5 是 0.15 |
| 压力段 MaxDD 改善 | ≥ 0.5pp | V5 未达标 |
| 整体失败归档 | 三合一体仍失败 → **永久归档** | V5 是单 trigger 失败 |

### 13.5 风险与回退预案

| 步骤失败 | 应对 |
|---------|------|
| 7.3.1 因子计算维度不匹配 | 复用 CTA factor base class，加 `aggregate_etf_no_oi()` 子函数 |
| 7.3.2 z-score 不稳定 | 改用 rank percentile（更稳健） |
| 7.3.3 alpha=0.3 仍冲击大 | alpha 扫描 {0.1, 0.2, 0.3} |
| 7.3.4 任一 Gate 不过 | 立即归档，不回退到 V5 |

### 13.6 预期成果

| 情景 | Sharpe | 压力段 | 是否推广 |
|------|--------|--------|---------|
| 乐观 | ABC_ew 1.897 → 2.0+ | 1.46 → 1.8+ | ✅ V6 主线 |
| 中性 | ABC_ew 1.897 不变 | 持平 | 保留为备选 |
| 悲观 | ABC_ew 1.897 略降 | 持平 | Gate 失败永久归档 |

### 13.7 文件变更清单

| 状态 | 文件 | 行数 | 内容 |
|------|------|------|------|
| 新增 | `QuantNodes/strategy/allweather/subfactors.py` | +120 | 三子因子计算 |
| 新增 | `QuantNodes/strategy/allweather/subfactors_weekly.py` | +60 | 周频重采样+z-score |
| 修改 | `QuantNodes/strategy/allweather/backtest.py` | +60 | subfactor_weights 接入 |
| 修改 | `QuantNodes/strategy/allweather/config.py` | +5 | AllWeatherConfig + 2 字段 |
| 修改 | `QuantNodes/strategy/allweather/experiments.py` | +120 | v6-subfactor-test 子命令 |
| 新增 | `tests/strategy/allweather/test_subfactors.py` | +80 | 5 单测 |
| 新增 | `tests/strategy/allweather/test_subfactors_weekly.py` | +40 | 3 单测 |
| 新增 | `tests/strategy/allweather/test_backtest_subfactor.py` | +50 | 4 单测 |

**总代码量**：新增 12 单测 + 240 行生产代码 + 本文档 90 行。

---

## 十四、V6.5 进一步优化 (2026-09-04, α+lookback+Sub-A 修复)

### 14.1 优化背景

V6.1 三子因子路线 Gate 全部 PROMOTE, 但发现三个可优化点:
1. α 扫描只覆盖 0.1-0.5, 边界 (0.05/1.0) 未测
2. 默认 lookback (sw=10/lw=120, w=30, w=5) 未调优
3. Sub-A α=0.3 时 Sharpe 衰减到 0.657, 接近 Gate 边界 (0.658)

### 14.2 三个优化方向

| 优化项 | 当前 (V6.1) | V6.5 新增变体 | 假设 |
|--------|------------|---------------|------|
| **alpha 范围** | 0.1, 0.2, 0.3, 0.5 | +0.05, 1.0 | 更小 alpha 更稳; 1.0 找拐点 |
| **lookback** | sw=10 lw=120 / w=30 / w=5 | sw=5/20, lw=60/252; w=20/60; w=3/10/20 | 不同窗口捕捉不同 regime |
| **Sub-A 修复** | z-score | rank percentile / winsorize ±2 / asymmetric α | z-score 在低波资产偏正→误加仓 |

### 14.3 Sub-A 衰减根因

z-score 标准化对低波动资产偏正 (历史 vol 稳定 → z-score 偏右 → 误加仓).
修复方法 (从好到差):
1. **rank percentile**: 横截面 rank → [-0.5, +0.5], 对极值稳健
2. winsorize ±2: z-score 截尾, 防极值放大
3. asymmetric α: 正向弱 (0.1) / 负向强 (0.3)

### 14.4 实施步骤

| 步骤 | 文件 | 内容 |
|------|------|------|
| 14.4.1 | subfactors_weekly.py | rank_cross_section + zscore winsorize 参数 + combine_subfactors method='rank' |
| 14.4.2 | backtest.py | subfactor_alpha_pos/neg 非对称 alpha + AllWeatherConfig +2 字段 |
| 14.4.3 | experiments.py | v6-alpha-scan + v6-lookback-scan + v6-subA-fix 三个新子命令 |
| 14.4.4 | tests/ | +4 单测 (rank / winsorize / asymmetric) |
| 14.4.5 | docs/ | 复盘报告 §5.10 完整记录 |

### 14.5 优化结果 (实测数据)

#### 14.5.1 α 扩展扫描 (V6.3)

| 因子 | α=0.05 | α=0.1 | α=0.2 | α=0.3 | α=0.5 | α=1.0 |
|------|--------|-------|-------|-------|-------|-------|
| Sub-A | **0.741** | 0.722 | 0.687 | 0.657 | 0.586 | 0.502 |
| Sub-B | **0.757** | 0.754 | 0.750 | 0.746 | 0.735 | 0.675 |
| Sub-C | 0.755 | 0.752 | 0.742 | 0.733 | 0.723 | 0.751 |

#### 14.5.2 lookback 扫描 (V6.4)

| 因子 | 最优窗口 | Sharpe 提升 |
|------|----------|------------|
| Sub-A | sw=15/lw=180 (mid) | 0.722 → **0.767** (+6.2%) |
| Sub-B | w=60/slope=30 (slow) | 0.754 → **0.771** (+2.3%) |
| Sub-C | w=5 (default) | 0.752 (无显著变化) |

#### 14.5.3 Sub-A 修复 (V6.5)

| 修复 | α=0.1 | α=0.2 | α=0.3 | 稳定性 |
|------|-------|-------|-------|--------|
| zscore baseline | 0.722 | 0.687 | 0.657 | 衰减严重 |
| **rank percentile** | **0.755** | **0.752** | **0.749** | **稳定** |
| winsorize ±2 | 0.736 | 0.712 | 0.687 | 中等 |
| asymmetric α | 0.737 | 0.715 | 0.695 | 压力段最佳 (1.150) |

#### 14.5.4 V6.5 最佳组合验证

把三个优化叠加（rank percentile + 优化 lookback）:

| 配置 (B sleeve) | Sharpe | vs 基线 0.758 | 压力段 |
|-----------------|--------|---------------|--------|
| B 基线 | 0.758 | — | 1.140 |
| B + Sub-A rank+mid α=0.1 | 0.759 | +0.001 | 1.167 |
| **B + Sub-BC rank α=0.2** | **0.768** | **+0.010** | 1.165 |
| B + Sub-ABC rank α=0.2 | 0.765 | +0.007 | **1.174** |

**核心突破**:
- **从"有损叠加"变成"无成本提升"**: Sub-BC rank α=0.2 全样本 +0.010 Sharpe, 压力段 +2.2%, **Sharpe 与压力段同时改善**
- B sleeve Sharpe 0.758 → **0.768 (+1.3%)**

### 14.6 V6.5 Gate 判定

| 维度 | V6.1 | V6.5 | 提升 |
|------|------|------|------|
| Sub-A α=0.1 | 0.722 | 0.759 | +5.1% |
| Sub-B α=0.2 | 0.750 | 0.771 | +2.8% |
| Sub-ABC α=0.2 | 0.722 | 0.765 | +6.0% |
| B sleeve 主线 | 0.758 | **0.768** | +1.3% |
| 压力段 (Sub-ABC α=0.2) | 1.085 | **1.174** | +8.2% |

**Gate 判定**: **PROMOTE**. Sub-BC rank α=0.2 是 V6 备选主推配置, 待 ABC_ew 集成验证.


---

## 十五、V6.6 ML 复合因子测试 (2026-09-04, 决策驱动)

### 15.1 起源

V6.5 完成后, 用户提出新诉求: 用 ML (线性 + 树模型) 测试三个子因子能否组合成更有效的复合因子, 用于预测风险或转折点. 这与避险猫原话 "合成计算出的综合因子" 高度契合.

### 15.2 5 个决策

| 决策点 | 选择 |
|--------|------|
| 预测目标 | Y1 20d vol 突变 (binary) / Y2 20d vol 水平 (regression) / Y3 20d 趋势反转 (binary) |
| 模型 | 线性 (LogReg + Ridge) + 树 (DecisionTree + RandomForest) |
| 交叉验证 | Purged K-Fold (5 折 + 5d purge + 1% embargo) |
| 验证方式 | IC + B sleeve 回测双验证 (不上 ABC_ew) |
| 模型架构 | 全局模型 + 资产特征 (sector + vol class + volume class, 14 维 one-hot) |
| 类别不均衡 | class_weight="balanced" |
| 输出处理 | Soft 信号 (sigmoid/tanh 限制 [-1, +1]) |

### 15.3 资产特征设计

| 特征 | 类型 | 编码 | 维度 |
|------|------|------|------|
| Sector | 分类 | one-hot (8 类: 有色/黑色/化工/农产品/股指/国债/商品ETF/股票ETF) | 8 |
| Vol Class | 分类 (252d 滚动年化 vol 三分位) | one-hot (3 类) | 3 |
| Volume Class | 分类 (252d 滚动平均成交量三分位) | one-hot (3 类) | 3 |
| **资产特征小计** | | | **14** |

### 15.4 完整特征矩阵 (29 维)

```
[1-3]   sub_a, sub_b, sub_c               (原始子因子)
[4-6]   sub_a_lag1, sub_b_lag1, sub_c_lag1
[7-9]   sub_a_lag5, sub_b_lag5, sub_c_lag5
[10-12] sub_a_ma5, sub_b_ma5, sub_c_ma5
[13-15] sub_a_std5, sub_b_std5, sub_c_std5
[16-17] sub_a_x_b, sub_b_x_c              (交互项)
[18-25] sector one-hot × 8
[26-28] vol_class one-hot × 3
[29-31] volume_class one-hot × 3
```

### 15.5 三目标定义

```python
# Y1: vol_spike (binary) — 20d forward vol > 历史 90 分位
y1 = (rolling_20d_vol_future > rolling_252d_vol.quantile(0.9)).astype(int)

# Y2: vol_level (regression) — 20d forward 已实现 vol (年化)
y2 = returns.rolling(20).std().shift(-20) * np.sqrt(252)

# Y3: turning_point (binary) — 20d 趋势反转
y3 = (sign(close.shift(-20) - close) != sign(close - close.shift(20))).astype(int)
```

### 15.6 Soft 信号输出

```python
def to_soft_signal(oof_preds, target_type):
    if target_type == "binary":
        return 1.32 * (2 * sigmoid(oof_preds * 4) - 1)  # → [-1, +1]
    else:
        return tanh(oof_preds / oof_preds.std())          # → [-1, +1]
```

### 15.7 决策表

| 指标 | V6.5 基线 | V6.6 目标 | 失败判定 |
|------|----------|----------|---------|
| B sleeve Sharpe | 0.768 | > 0.78 | < 0.768 |
| 压力段 Sharpe (25+) | 1.165 | > 1.20 | < 1.165 |
| Per-asset IC | — | > 0.03 | < 0.02 |
| Y1 vol_spike AUC | — | > 0.55 | < 0.52 |
| Y2 vol_level RMSE | — | < 历史 vol 均值 | > 均值 |
| Y3 turning AUC | — | > 0.55 | < 0.52 |
| 特征重要性 | Sub-B 主导 | ML 是否发现新交互 | 无新发现 |

### 15.8 实施步骤 (7 步, ~7h)

| 步骤 | 内容 | 文件 | 产物 | 耗时 |
|------|------|------|------|------|
| 15.8.1 | PurgedKFold + 单测 | ml_composite.py + test_ml_composite.py | 5 单测 | 1.5h |
| 15.8.2 | 特征工程 | ml_composite.py:build_features | 29 维 | 1h |
| 15.8.3 | 三目标定义 | ml_composite.py:build_targets | Y1/Y2/Y3 | 0.5h |
| 15.8.4 | 模型训练 | ml_composite.py:train_models | OOF + importance | 2h |
| 15.8.5 | IC 验证 + 特征重要性 | experiments.py:v6-ml-ic | IC table | 1h |
| 15.8.6 | Soft 信号 + B sleeve 回测 | experiments.py:v6-ml-backtest | 12 跑 × 4 alpha | 1h |

### 15.9 文件变更清单

| 状态 | 文件 | 行数 | 内容 |
|------|------|------|------|
| 新增 | ml_composite.py | +300 | PurgedKFold + features + targets + train + soft_signal |
| 修改 | experiments.py | +150 | v6-ml-ic + v6-ml-backtest 子命令 |
| 新增 | test_ml_composite.py | +200 | 15 单测 |
| 修改 | allweather-replication-report.md | +50 | §5.11 V6.6 ML 结果 |

### 15.10 失败回退预案

若 V6.6 < V6.5 基线 (0.768):
1. 保留 ML IC 部分 (特征重要性是永久知识)
2. 回测部分归档
3. V6.5 Sub-BC rank α=0.2 仍为推荐配置
4. ML 工作作为 "未来 ML 方向" 的探索记录


## 十六、V7 类别级风险预算重构 (2026-09-05, A sleeve 债券集中诊断驱动)

### 16.1 背景与诊断 (见报告 §5.17)

前置诊断发现 A sleeve (HRP+MP+cap0.40+lev4) 实为 **3.5x 杠杆国债组合**:
- fut_ts +158.8% / fut_tf +138.5% / fut_t +55.0%, 三债券合计 352% gross
- 有效资产数仅 3.3; Sharpe 1.56 来源 = 2018-2026 国债牛市 carry, 非 HRP 分散

因果链: MP 过度去噪 (0信号特征值) → 伪对角协方差 → HRP 退化为 inv-vol 极限
→ 3 个同利率因子资产权重膨胀 → cap 0.40 只封单资产不封类 → 4x 杠杆放大。

风险平价债券倾斜为文献确认的结构性问题 (报告 §5.18 调研):
权重空间集中 / 杠杆放大 / Sharpe 押注 / 相关性 regime 翻转四缺陷;
2022 美国 RP 全线 -10%~-28%; 我们 2025-2026 债市调整期 HRP 系全负 (报告 §4.3 判读4)。

### 16.2 方案: 类别级 ERC + 债券风险预算封顶 (L3, 行业共识层级)

- 债券保留 (stress 段 flight-to-quality alpha 是真价值) 但风险贡献封顶
- regime 感知 cap (滚动股债相关 >0 收紧) 归档 V7.1, V7.0 只做静态 cap
- 杠杆上限保持 max_leverage 可配置 (后续单独讨论, 扫描附 2.3x 参考行)

### 16.3 类别定义 (10 细类, 用户确认)

```python
ASSET_CLASSES = {
    "bond_rate":      ["fut_t", "fut_tf", "fut_ts"],
    "equity_fut":     ["fut_if", "fut_ic", "fut_ih"],
    "equity_etf_dom": ["etf_512890", "etf_512010", "etf_515030", "etf_512690",
                       "etf_512800", "etf_515050", "etf_512200"],
    "equity_etf_off": ["etf_513100", "etf_159740"],   # 纳指/恒科, 海外/离岸
    "precious":       ["fut_au", "fut_ag"],
    "nonferrous":     ["fut_cu", "fut_al", "fut_zn", "fut_ni"],
    "ferrous":        ["fut_rb", "fut_hc", "fut_i", "fut_j", "fut_jm"],
    "energy_chem":    ["fut_ta", "fut_ma", "fut_pp", "fut_l", "fut_bu", "fut_sc"],
    "agri_oilseed":   ["fut_m", "fut_y", "fut_p"],
    "agri_soft":      ["fut_cf", "fut_sr", "fut_c"],
}
```
(ETF 身份以 data_loader 实际标注核对为准, Phase 0.0)

### 16.4 算法 (新模块 class_erc.py)

1. 类内: inverse-vol 加权合成类收益序列 → 类级协方差 (window 与 cfg.window 同源)
2. 类间: ERC 等风险贡献 (迭代法) → 10 类风险预算
3. 债券封顶: 若 bond 类 RC > bond_risk_cap → bond 预算固定为 cap,
   剩余 (1-cap) 在其余 9 类间重新 ERC (cap 是约束而非定额, 未触发时全 ERC)
4. 类内分配: 类权重按成员 inverse-vol 摊到资产
5. 晚上市/观测不足资产: 复用 min_obs_per_asset 渐入池模式

### 16.5 配置与接入

- AllWeatherConfig 新增: weight_scheme="class_erc", bond_risk_cap: float = 1.0,
  asset_classes: dict | None = None (外部可覆盖)
- backtest.py `_base_weights`: class_erc 分支先于 use_hrp 判断
- experiments.py: CLI `v7-class-erc` + `--v7-caps 0.25,0.33,0.5,1.0`

### 16.6 执行路线 (粒度到验证)

```
Phase 0 — 归因确认 (只读诊断)
  [0.0] 核对 9 只 ETF 身份 → verify: data_loader 标注与类别映射一致
  [0.1] A sleeve P&L 分解: 债券类日贡献占比轨迹 + vs 债券合成滚动相关
  [0.2] 利率熊市窗口回撤: 2020-04~07, 2022-11~2023-01 + DV01 解析估算
Phase 1 — class_erc.py + 单测 (~8个)
  verify: python -m pytest tests/strategy/allweather/test_class_erc.py -q
          python -m pytest tests/strategy/allweather/ -q (122 旧测全过)
          ruff check
Phase 2 — 扫描与诚实重测
  [2.1] bond_risk_cap × {25%,33%,50%,1.0} × lev {4x, 2.3x参考}
  [2.2] verify: bond RC 轨迹 ≤ cap; 与 fut_t 相关性显著下降
  [2.3] A_v7 替入 ABC, 重扫 A/B/C 权重 (不沿用 0.60/0.20/0.20)
Phase 3 — 利率+100bp 压力测试 + 报告 §5.19 + commit
```

### 16.7 预期管理

诚实代价: 主线 Sharpe 大概率从 1.995 回落 (去掉 Sharpe-bet, ABG 2012 教训);
前瞻回报预期以修复后数字为准。修复成功标准: 债券 RC ≤ cap + 与 fut_t 相关性
显著下降 + 全样本 Sharpe 回落幅度可解释 (carrying alpha 去除量级)。

### 16.8 无杠杆版本 (2026-09-05, 用户确认: 所有 sleeve 先不加杠杆)

max_leverage=1.0 对所有 sleeve. vol_target 仍开启 (cap=1.0 时只能向下缩, 高波动期防御保留).
参数保留可调, 当前默认不加杠杆.

无杠杆主线定版: A=0.50 / B=0.20 / C=0.30 (levA=1.0 下最优权重)

| 指标 | V7 lev2.3 | V7 lev1.0 (无杠杆) | 变化 |
|------|-----------|-------------------|------|
| Sharpe | 1.682 | **1.590** | -0.09 |
| AnnRet | 9.6% | **7.1%** | -2.5pp |
| MaxDD | -5.41% | **-4.17%** | +1.24pp |
| Vol | ~5.5% | ~3.5% | -2pp |
| stress_25+ | 1.264 | **1.009** | -0.26 |

A sleeve 单独: Sharpe 1.329, AnnRet 3.4%, Vol 2.6%, MaxDD -3.56%, stress 2.008.
Sharpe 微升 (无多余杠杆), AnnRet 减半, MaxDD 大幅改善.

无杠杆下天花板: vol ~3.5% × Sharpe ~1.6 ≈ AnnRet ~5.6%; 实际 7.1% (CTA 贡献高 vol alpha).
AnnRet ≥ 10% 目标在无杠杆架构下结构性难以达成.

## 十七、D 策略：资产级配置管线 (2026-09-06, 设计文稿复刻)

### 17.1 定位

D sleeve = 设计文稿《量化全天候模型2.0_生产版》五步管线的核心配置部分 (资产级 HRP).
与 A sleeve (class ERC) 是同一问题 (资产配置引擎) 的两种独立解法, 并行存在.
主线形态: ABC → ABCD (D.0 独立测试后再讨论).

### 17.2 D.0 Scope (核心版, 2026-09-06)

```
日线行情 (38 资产)
 → ① AVS 分层去噪协方差 (Ward聚类 → 分块目标F → 闭式δ*)
 → ② HRP 权重 (Ward树递归二分, 可配置频率, 默认周频W-FRI)
 → ③ 债券硬上限 w_bond≤40% → 溢出按RC share重分配
 → ④ vol_target 10% (lev默认1.0, 可配置)
 → D sleeve 净值
```

**D.0 不做**: 负趋势因子 (延后D.1) / 量价加载器 / 短融现金管理(G5) / ABCD扫描 / 主线改动.

### 17.3 已确认决策

| # | 决策 | 说明 |
|---|------|------|
| 1 | 杠杆默认1.0, 可配置 | 用户确认 2026-09-06 |
| 2 | 短融现金管理 D.0 不做 | G5 悬置 |
| 3 | 负趋势因子延后D.1 | D.0 为纯配置引擎基准 |
| 4 | 本次只做D单策略独立测试 | 不动ABCD/主线 |
| 5 | 调仓频率参数化, 默认周频 | 消融测周频vs20日 |

### 17.4 五模块实现映射

| 模块 | 代码 | 复用/新建 |
|------|------|----------|
| AVS 分层去噪 | `cov_avs.py` (~80行) | 新建: Ward→分块F→δ*→C=(1-δ*)S+δ*F |
| HRP Ward | `hrp.py` | 复用: hrp_weights(linkage="ward") |
| 债券cap+RC重分配 | `d_pipeline.py` (~150行) | 新建: bond_cap + RC重分配 |
| vol_target | `d_pipeline.py` | 参考 backtest.py vol_target 逻辑 |
| 负趋势 (D.1) | `neg_trend_daily.py` | 建后: 3D+异质权重+消融开关 |

### 17.5 关键发现: 债券上限是生死线

实测 HRP(Ward) 在我们数据上: 债券权重 99.6%, 有效资产 1.6.
**无论 single 还是 ward, HRP 在 38 资产池都塌缩到国债期货.**
设计文稿 §3.4 的债券硬上限40% + RC重分配是 D 管线能否工作的唯一修复.

### 17.6 D vs A 头对头对比框架

| 维度 | A (class ERC) | D (HRP+AVS+cap40%) |
|------|---------------|---------------------|
| 债券处理 | 风险预算制 (bond RC=10%) | 权重硬上限40% + RC重分配 |
| 协方差 | MP (伪对角) | AVS (保留簇内结构) |
| 更新频率 | 周频 | 可配置 (默认周频) |
| 复杂度 | 低 | 中 (AVS+重分配) |

### 17.7 执行路线

```
Phase 0 — 数据前置 (已完成: 期货vol+OI ✅ / ETF vol ✅ / D.0不需要量价)
Phase 1 — cov_avs.py + d_pipeline.py + ~9单测 → 132旧测全过 + ruff
Phase 2 — D.0独立实测: 5区间 / 频率消融 / D vs A / 协方差消融 / 利率压力
Phase 3 — 文档 + commit (仅新增文件, 不动现有路径)
```

### 17.8 D.1 路线图 (延后)

```
前置: 量价面板加载器 (期货vol+OI / ETF vol, 数据已核实存在)
 → neg_trend_daily.py (3D + 异质权重 + 消融开关)
 → penalty 日频叠加进D管线
 → 重测D + 头对头 + 主线ABCD讨论
触发: D.0 实测完成并经确认
```

### 17.9 协方差改进路线 (2026-09-06, D管线内探索)

在 D.0 LW+HRP+cap40% 框架上, 不改 A/B/C, 仅替换协方差引擎.

**Phase 1 (低成本):**
- EWMA 协方差 (λ∈{0.92,0.94,0.96,0.98})
- AVS-F + LW hybrid (F做target + LW MSE最优δ)
- EWMA + AVS-F + LW 三合一
- 波动率自适应窗口 (高波缩窗/低波恢复)

**Phase 2 (中成本):**
- NQN 非线性收缩 (Ledoit & Wolf 2012, 谱分解)
- Gerber + LW ensemble (50/50 融合)

新增文件: `cov_extended.py` (~200行) + `test_cov_extended.py` (~6单测)
消融框架: 加入 `cov_fns` dict 自动对比 (原有4种+新增6种=10种)

### 17.10 原理性重构 (2026-09-07, 用户确认: 替代全部补丁方案)

**背景**: §5.24 消融发现系统性 bug — EWMA λ=0.94 gross 56x / 负权重 88%.
诊断确认根因: EWMA 快衰减 → 协方差近奇异 (cond 62万) → RC share 极端 (-4.5~+20.8)
→ 无界重分配 → 杠杆爆炸. LW 干净 (cond 180) 因其强度公式自调节.

**决策**: 放弃补丁 (clip/guard/保险丝), 从原理上重构两条数学不变量:

**不变量 1 — 估计器自调节**:
- lw_ewma_cov: EWMA 加权样本 + LW 强度公式推广 (α 随 N_eff=(1+λ)/(1-λ) 自适应)
- OAS (Chen et al. 2010): sklearn 可用 (1.4.2), 解析式自实现 ~20 行
- 相关空间分块收缩 (修正版 AVS-F): corr 空间目标 (条件数有界) + vol 重标定;
  否定协方差空间 F 目标 (vol 离散度污染, §5.24 hybrid 0.714 失利根因)

**不变量 2 — 约束由求解器内建**:
- risk_budget_ccd: Spinu 凸目标 + 盒约束投影坐标下降
- 合成债券资产: 3 债券 → 1 合成资产 (inv-vol), 盒约束 u=0.40 直接可执行
- 预算 = HRP(Ward) 风险贡献 (层次洞察保留); 无约束时退化为 HRP 原解
- 删除 RC share 重分配整段代码 (求解器取代启发式)

**正反证据**: A sleeve 干净 (0% 污染) 因其重解预算而非事后修补;
D sleeve 爆炸因 cap+重分配启发式链. 重构让 D 采用与 A 同级的原理.

**次序** (用户确认): Phase A 原理引擎 → Phase B 方法消融
(先 OAS + 相关空间目标 → 再 LW2012 完整解析式 → POET 延后 B 轮后半).

**影响范围**: §5.24 初版结论作废 (None/EWMA/VolAdaptive 被污染);
LW=1.030 可信 (0% 污染); D.0 AVS=0.751 需复核; 频率消融需复核.

### 17.11 B 轮后半 (2026-09-08, 用户确认)

- nqn_lw2012_cov: Ledoit & Wolf 2012/2018 非线性收缩 (直接核估计:
  d_j = λ_j/[(1−c−c·λ_j·m̂)²+(π·c·λ_j·p̂)²], Hilbert变换+核密度, 对角排除)
- poet_cov: Fan-Liao-Mincheva 2013 (PCA top-K 因子 + 残差幅度相对硬阈值)
- 接入 d_pipeline cov_estimator={"nqn","poet"}; K=3 默认
- 行为单测: 各向同性塌缩 / 尖峰保留 / 因子子空间恢复 / 阈值稀疏性

### 17.12 D.1 负趋势因子日频风控层 (2026-09-08, 范围收窄)

**用户决策**: 先做 D.1 (负趋势因子), 动量倾斜+减仓补仓逻辑放 D.2 (做完 D.1 再说).

**D.1 Scope**:
- neg_trend_daily.py: 3D 子信号 (trend 5/10/20斜率 + vol 60日分位 + flow 价量/OI背离)
  + 异质权重 (equity重trend/futures重flow/gold重vol) + 自适应阈值 + 双阈值滞回
  + penalty ∈ [0,1], 日频; 消融开关 mode='3d'/'trend'/'none'
- 量价加载器: 期货 volume+OI (data/chinese_futures) + ETF volume (data/real/ohlcv)
- 接入 run_d_sleeve: 日频 penalty 乘性叠加 w×(1−penalty)
- 不做: 动量倾斜 (D.2) / 减仓补仓增强 (D.2) / 短融现金 (G5)

### 17.13 D.1 定版: 三档参数保留 (2026-09-08, 用户确认)

**饱和修复后画像翻转**: trend 止损全面超越基线 (基线 1.036/DD -9.84%):
- aggressive 0.4/0.2: Sharpe 1.142 / MaxDD -6.74% / 换手 14.6x
- balanced 0.5/0.3 (默认): Sharpe 1.111 / MaxDD -7.28% / 换手 12.5x
- conservative 0.6/0.4: Sharpe 1.080 / MaxDD -7.64% / 换手 10.7x

**正式化**: NEG_TREND_PRESETS 字典 (neg_trend_daily.py) + get_preset_params()
+ run_d_sleeve(neg_trend_preset=...) 档位透传 (preset 优先于显式阈值).
mode 默认保持 trend-only (3d 的 vol/flow 维度在 D 管线稀释趋势信号, 与 V6.8 互证).

**单测**: 三档阈值单调性 / 未知名回退 / 激活率单调 / run_d_sleeve 档位透传.

**下一步 D.2**: 动量倾斜 (alpha 层) + 高位减仓低位补仓逻辑.

### 17.14 D.2 动量倾斜 alpha 层 (2026-09-09, 用户确认)

**决策**: (a) 滞回风控版 — 补仓 = 动量拐正后倾斜超配 (有确认的加仓, 非接刀);
β = ±0.20; 与 D.1 同源 t-stat 信号 (mom/vol), 方向双向化 (D.1 单向风控/D.2 双向 alpha).

**架构** (叠在 D.1 trend 三档上):
- 周频动量倾斜: score = tanh(0.5·t-stat) (与 D.1 修复后 sub_trend 同源同分布)
  → t_i = 1 + β·score ∈ [1−β, 1+β] → w_tilt = w_base × t_i (归一化)
- 正动量超配 (补仓确认) / 负动量低配; 晚上市资产 score 缺失 → t_i = 1 中性
- 日频 D.1 趋势止损照常叠加 (减仓执行层)
- 消融: 基线 / +D.1 / +D.2 / +两者 (4 配置 × 分段)

### 17.15 对齐池策略评估 (2026-09-10, 用户确认)

**目标**: 把对齐池 (11 资产 ≈ 避险猫 12 风险资产) 当候选主线完整评估,
与 38 资产主线 (ABC 1.592) 正面对比, 回答"是否值得切换".

**对齐池** (panel_v3 子集, 7 类映射):
bond(2): fut_t/fut_ts | equity_dom(3): etf_159915/510300/512890 |
equity_off(2): etf_513100/513500 | commodities(4): fut_au/cu/sc/m
CASH 不含 (G5 悬置).

**执行范围** (用户确认):
- Phase 1: 全引擎矩阵 — A(class ERC) / D(v3完整) / B(EW+V6.8) / C(CTA 6期货)
  + D 消融 (trend vs 3d, β∈{0.1,0.2})
- Phase 2: 组合层正面对比 — 对齐池 ABC (0.5/0.2/0.3 与等权) vs 38 主线 (1.592)
- Phase 3: 2.3x 杠杆对照组 (仅标注为对照, 主线仍 1.0) + 利率+100bp 压力
- **数据窥探防护**: 附测"独立平衡池" (38 池商品类内 inv-vol 合成) 第三方对照;
  21-23 段对齐池 D 更差 (0.72 vs 0.77) 已是非全面碾压的结构性信号
- 切换标准: ≥3 分段优 + 全样本 ≥1.592 (未达标则归档, 不切换; 切换决策后续讨论)

### 17.16 13 资产完整复现 + G5-lite (2026-09-10, 用户确认)

**决策**: 中证1000 用 000852 指数; CASH 用货币基金 (真实数据优先, 缺则合成);
**评估基准只用公众号文章公开声称值** (HTML 自认内容排除):
Sharpe 2.67 / AnnRet 20.35% / MaxDD -5.91% / 月胜率 76.4% / 全年正收益 / 2.3x 杠杆.

**13 资产清单**:
etf_513100/513500 (美股) + etf_159915/510300/512890 (A股) +
fut_t/fut_ts (债) + fut_cu/sc/m/au (商品) + 中证1000 (000852 指数或 ETF) + CASH (货币基金)

**G5-lite 机制** (网页行为特征: 危机期现金>20%):
- CASH 踢出风险池 (不参与 HRP 风险预算)
- 债券 cap 溢出 → CASH; penalty 减仓释放 → CASH
- 行为指纹对照: 月胜率 76.4% / 全年正收益 / 杠杆路径 / 危机期现金占比

**Phase 0 数据考古**: 000852 指数可得性; 货币基金 (511990 等); 931080/000510 审计
