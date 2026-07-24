# v9 — 宏观周期择时与诊断

> **编号**: 49
> **状态**: 📋 设计中 → 文档阶段
> **日期**: 2026-07-22
> **作者**: QuantNodes
> **关联**: docs/46-v8_ml_design.md, docs/47-v8_optimization_record.md

---

## 1. 动机与背景

### 1.1 当前策略体系的天花板

| 版本 | 方法 | OOS Sharpe | 局限 |
|------|------|-----------|------|
| v7.14 | TV-PR (Time-Varying Penalized Regression) | 0.44 | 单尺度, 仅看因子 |
| v7.7 | PyCaret ML (失败) | N/A | R² ≈ 0 |
| v8 Jump Model | 单变量 Bear 状态检测 | 1.485 | 单变量, 忽略宏观周期 |
| **v9 (本文)** | **多周期 + 周期诊断** | 目标 ≥ 0.5 | **看大局观** |

### 1.2 v8 Jump Model 的边界

**v8 已做到的**:
- 单资产 2 状态 (bull/bear) 识别
- 仓位动态调节 (Bear% × v7.14 权重)
- OOS Sharpe 1.485, Calmar 1.467

**v8 未做到的**:
- 没有"我们当前在哪个周期"的全局视角
- 没有跨周期的耦合检测 (Hilbert, 双相干)
- 没有多周期叠加分析 (基钦 + 朱格拉 + 康波)
- 没有对接经典周期理论框架 (美林时钟, Pring 周期)

### 1.3 v9 的核心价值

**回答 3 个核心问题**:
1. **我们在哪里?** — 当前在哪个经济周期, 处于该周期的早/中/晚期?
2. **会怎么走?** — 多周期共振/背离的方向如何?
3. **该怎么做?** — 大盘择时信号 + 仓位建议

---

## 2. 设计哲学

### 2.1 双轨思路: 诊断 + 择时

```
                    ┌─────────────────────────┐
                    │  输出 A: 周期诊断 (CPD) │  ← 独立价值, 立即可用
                    │  ─ 当前位置              │
                    │  ─ 美林时钟 4 阶段       │
                    │  ─ 多周期叠加            │
                    └─────────────────────────┘
                                  ▲
┌────────────────────────────────────────────────────────────┐
│  v9 大盘择时层                                             │
│  ─ 评分 0-100 → 信号 0/1                                  │
│  ─ 与 v7.14 (选股) + v8 (降仓) 叠加                       │
└────────────────────────────────────────────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │  输出 B: 仓位权重       │  ← 战术价值
                    │  final = v9 × v7.14 × v8│
                    └─────────────────────────┘
```

**关键洞察**:
- **CPD (Cycle Position Diagnosis) 是独立产出**, 不依赖回测验证
- 即使择时部分失败, CPD 也有完整的诊断价值

### 2.2 多周期框架

**经济周期的四个尺度**:

| 周期 | 名称 | 时长 | 驱动因子 | 在股市中的角色 |
|------|------|------|----------|---------------|
| 超短波 | Minor Cycle | 9-12 月 | 库存, 短期信贷 | 短线择时 |
| 短波 | Kitchin | 3-5 年 | 库存周期 | A 股最强驱动力 |
| 中波 | Juglar | 7-11 年 | 设备投资 | 估值水位 |
| 长波 | Kuznets | 15-25 年 | 房地产/人口 | 金融地产 β |
| 超长波 | Kondratieff | 40-60 年 | 技术革命 | 长期趋势项 |

**A 股特殊性**:
- 1990 年至今 ~34 年数据, 不够 1 个完整 Kondratieff
- 政策驱动市场 (降准/降息), 周期被打断
- 美林/Pring 框架都是**西方成熟市场**, A 股适用性需自行验证

### 2.3 与传统美林/Pring 的区别

| 维度 | 传统美林/Pring | v9 |
|------|--------------|-----|
| 周期识别 | 主观经验 | Hilbert 客观提取 |
| 相位耦合 | 无 | 双相干系数 |
| 多周期叠加 | 简单叠加 | 客观分解 + 合成 |
| 输出形式 | 文字描述 | 数值评分 + 信号 |
| 验证方式 | 历史叙事 | 回测 + 鲁棒性测试 |

---

## 3. 五层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  第五层: 仓位合成 (Position Sizing)                                  │
│  ─ 输入: v9 信号 + v7.14 权重 + v8 Bear%                            │
│  ─ 输出: final_weight                                                │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  第四层: 周期耦合识别 (Cycle Coupling)                               │
│  ─ Hilbert 相位: φ_kitchin(t), φ_juglar(t), φ_yearly(t)              │
│  ─ 双相干系数: bic(ω1, ω2) → 0°/180° 锁定                          │
│  ─ 输出: cycle_score ∈ [0, 100]                                     │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  第三层: 多尺度周期分解 (Multi-scale Decomposition)                  │
│  ─ CEEMDAN 或 VMD 分解 (二选一)                                      │
│  ─ 输出: IMF1 (季度), IMF2 (年), IMF3 (基钦), IMF4 (朱格拉+)        │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  第二层: 数据预处理 (Preprocessing)                                  │
│  ─ HP 滤波 (λ=100) 分离趋势 + 周期残差                              │
│  ─ 对数差分平稳化                                                    │
│  ─ 缩尾 (1%-99%) 抑制极端值                                          │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  第一层: 数据加载 (Data Loading)                                     │
│  ─ 沪深300 等权 (43 ETF 合成)                                       │
│  ─ 39+ 宏观因子 (从 v7_14_X_panel.npy)                              │
│  ─ VIX, 利率, 利差                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 第一层: 数据加载

**输入**:
- 43 ETF 周频数据 (`data/high_freq_macro/v7_14_Y_weekly.parquet`)
- 39+ 宏观因子 (`data/high_freq_macro/v7_14_X_panel.npy`)
- VIX (`macro_vix_daily.parquet`)
- 中美利差 (`cn_us_spread_10y.parquet`)
- 美元指数 (`macro_dxy_daily.parquet`)
- 实际利率 (`macro_real_rate_daily.parquet`)

**输出**:
- 沪深300 等权指数
- 宏观因子面板

### 3.2 第二层: 预处理

**HP 滤波**:
```python
from statsmodels.tsa.filters.hp_filter import hpfilter

cycle, trend = hpfilter(log_price, lamb=100)
# 周频标准 λ=100, 月频 λ=14400, 日频 λ=1600
```

**对数差分**:
```python
returns = log_price.diff().dropna()
```

**缩尾**:
```python
from scipy.stats.mstats import winsorize

returns_w = winsorize(returns, limits=[0.01, 0.01])
```

### 3.3 第三层: 周期分解

**双算法对比** (用户决策: 两者都试):

| 算法 | 优点 | 缺点 | 包 |
|------|------|------|-----|
| **VMD** | 收敛快, 参数少, 抗混叠 | 对小样本敏感 | vmdpy |
| **CEEMDAN** | 模式分离彻底, 完备 | 计算慢 | PyEMD |

**默认参数**:
- IMF 数 = 4 (季度, 年, 基钦, 朱格拉+)
- VMD α = 1000
- CEEMDAN trials = 100

### 3.4 第四层: 周期耦合

**Hilbert 相位**:
```python
from scipy.signal import hilbert

analytic_signal = hilbert(imf)
phase = np.angle(analytic_signal)  # φ(t) ∈ [-π, π]
```

**双相干系数**:
```python
bic(ω1, ω2) = |E[F(ω1) F(ω2) F*(ω1+ω2)]| / sqrt(E[|F(ω1)F(ω2)|²] E[|F(ω1+ω2)|²])
```

### 3.5 第五层: 仓位合成

**分层叠加公式**:
```python
v9_signal ∈ {0, 1}      # 大盘信号
v8_factor ∈ [0, 1]      # v8 Bear% 转仓位因子
v7_weight ∈ [0, 1]      # v7.14 周权重

final_weight = v9_signal × v8_factor × v7_weight
```

---

## 4. 周期诊断层 (CPD)

### 4.1 美林时钟识别

**4 阶段**:

```
       GDP ↑
        │
        │  Phase II       Phase I
        │  Overheat       Recovery
        │  商品 > 股票    股票 > 债券
        │
────────┼─────────────────────────── GDP →
        │
        │  Phase III      Phase IV
        │  Stagflation    Recession
        │  现金 > 商品    债券 > 股票
        │
       GDP ↓
```

**数据映射**:
- GDP 同比: 工业增加值同比 或 PMI
- CPI 同比: PPI 同比 (高频代理)

### 4.2 Pring 10 年周期

**年份位置判定**:
- 第 1-3 年: 熊市主导
- 第 4-6 年: 牛市启动
- 第 7-9 年: 牛市顶部
- 第 10 年: 顶部/调整

### 4.3 综合定位输出

**CPD 报告核心字段**:
1. 当前美林时钟阶段
2. Pring 10 年周期位置
3. 多周期叠加状态 (Kitchin/Juglar/Kondratieff)
4. 评分 0-100
5. 大盘信号 (0/1)
6. 未来 12 个月情景

---

## 5. 与 v8/v7.14 集成

### 5.1 分层架构

```
v9 (顶层) ─→ 大盘信号 0/1, 决定是否参与市场
v7.14 (中层) ─→ ETF 周权重, 决定选什么
v8 (底层) ─→ Bear% 转仓位因子, 决定仓位多少
```

### 5.2 数学表达

$$
w_{final}(t, etf) = v9\_signal(t) \times v8\_factor(t, etf) \times v7\_weight(t, etf)
$$

其中:
- $v9\_signal(t) = \mathbb{1}[score(t) \geq 50]$
- $v8\_factor(t, etf) = \max\left(1 - \frac{bear\_pct(t, etf) - 0.3}{0.7}, 0\right)$ if $bear\_pct > 0.3$ else 1

### 5.3 与 v8 的差异化

| 维度 | v8 | v9 |
|------|----|----|
| 信号类型 | 单变量 bear 状态 | 多元宏观周期 |
| 检测粒度 | 日频 | 周频-月频 |
| 解释性 | 单一阈值 | 多维评分 |
| 与美林时钟 | 无对接 | 直接对接 |
| 与 Pring 周期 | 无对接 | 直接对接 |

---

## 6. 实施计划

### 6.1 三阶段实施

| Phase | 时间 | 内容 | 必交付 |
|-------|------|------|--------|
| **Phase 0: 文档** | Day 1-3 | 8 份文档 | ✅ |
| **Phase 1: CPD** | Day 4-6 | 美林+Pring+CPD+HTML | ✅ |
| **Phase 2: 验证** | Day 7-11 | 择时信号验证 | ⏸ 可选 |

### 6.2 决策点 (Day 11)

**信号清晰标准** (任一满足即算"清晰"):
- 2014/2019/2024 三次大底中 ≥ 2 次在底部前 4 周发出 Δφ<30° 锁定
- OOS 期 Sharpe 0.44 → 0.5 显著提升 (p<0.1)
- 双相干系数在 ≥ 2 对 IMF 上 >0.6

**如果失败** → 仅交付 Phase 0 + Phase 1。

---

## 7. 验证标准

### 7.1 CPD 验证 (Phase 1 必交付)

| 项 | 标准 |
|-----|------|
| 美林时钟识别准确度 | 历史回溯 ≥ 80% 命中 |
| Pring 周期位置 | 与已知 10 年周期阶段一致 |
| HTML 仪表盘渲染 | 8 个核心面板 |
| 当前周期报告 | docs/50 填写完整 |

### 7.2 择时验证 (Phase 2 可选)

| 指标 | v7.14 baseline | v9 目标 |
|------|----------------|---------|
| OOS Sharpe | 0.438 | ≥ 0.5 |
| OOS Calmar | - | ≥ 0.5 |
| OOS MaxDD | -22.7% | ≤ -20% |
| OOS AnnRet | 7% | ≥ 8% |

**目标含义**: 不追求超越 v8 (1.485), 只求显著超过 v7.14 baseline。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 周期不稳定 (Kitchin 28-60 月) | CEEMDAN 自适应分解 |
| Hilbert 端点效应 | 丢弃首尾 60 天 |
| 双相干计算量大 | numpy 向量化 |
| 美林/Pring 不适用 A 股 | 仅作参考, 主信号来自客观分解 |
| 评分过拟合 | 权重用学术默认值 (40+40+20) |
| Phase 2 失败 | 接受部分完成, 仅交付 CPD |

---

## 9. 关键决策汇总

| 决策项 | 选定 |
|--------|------|
| 项目命名 | v9 |
| 目录 | `QuantNodes/strategy/momentum_etf_rotation/v9/` |
| 主目标 | 周期诊断 (CPD) + 大盘择时 (双轨) |
| 起步路径 | 全部 8 份文档优先 → CPD → 验证 |
| 标的范围 | 分层: v9 大盘 + v7.14 ETF |
| 验证标准 | OOS Sharpe ≥ 0.5 |
| 分解方法 | CEEMDAN vs VMD 双算法 |
| v9/v8 集成 | 分层叠加 |
| 输出形式 | 连续仓位 + HTML 仪表盘 |
| 数据源 | 复用现有 (39+ 宏观因子) |
| Pring 代理 | 真实指标 + 市场指数 |
| 失败应对 | 接受部分完成 |

---

## 10. 文件清单

### 10.1 文档 (8 份)

```
docs/49-v9_cycle_timing.md           (本文档)
docs/49a-v9_cycle_diagnosis.md       (CPD 框架)
docs/49b-v9_decomposition.md         (分解算法)
docs/49c-v9_coupling.md              (耦合识别)
docs/49d-v9_scoring.md               (评分合成)
docs/49e-v9_backtest.md              (回测验证)
docs/49f-v9_v8_integration.md        (v8 集成)
docs/50-v9_current_cycle_state.md    (当前状态)
```

### 10.2 主代码 (11 文件)

```
QuantNodes/strategy/momentum_etf_rotation/v9/
├── __init__.py
├── config.py
├── data_loader.py
├── preprocess.py
├── decompose.py
├── coupling.py
├── scoring.py
├── position.py
├── backtest.py
├── metrics.py
└── cpd/
    ├── __init__.py
    ├── merrill_clock.py
    ├── pring_cycles.py
    ├── cycle_position.py
    └── diagnose.py
```

### 10.3 脚本与报告

```
scripts/v9/
├── v9_cycle_timing_main.py
├── v9_imf_visualize.py
├── v9_phase_coupling.py
├── v9_decomposition_comparison.py
└── v9_cpd_diagnose.py

reports/momentum_etf_rotation/v9/
├── dashboard.html
├── dashboard_data.json
├── summary.md
├── backtest_results.csv
└── current_cycle_state.md
```

---

## 11. 参考文献

1. **Schumpeter, J. A. (1939)**. *Business Cycles: A Theoretical, Historical, and Statistical Analysis of the Capitalist Process*.
2. **Kitchin, J. (1923)**. "Trade Cycles and the Effort to Anticipate". *Economic Journal*.
3. **Juglar, C. (1862)**. *Des crises commerciales et de leur retour périodique en France, en Angleterre et aux États-Unis*.
4. **Kuznets, S. (1930)**. *Secular Movements in Production and Prices*.
5. **Kondratieff, N. D. (1925)**. "The Major Economic Cycles". *Voprosy Konjunktury*.
6. **Pring, M. (2002)**. *The All-Season Investor*. McGraw-Hill.
7. **Merrill Lynch (2004)**. *The Investment Clock*. Research Report.
8. **Huang, N. E. et al. (1998)**. "The Empirical Mode Decomposition and the Hilbert Spectrum". *Proc. R. Soc. Lond. A*.
9. **Dragomiretskiy, K., Zosso, D. (2014)**. "Variational Mode Decomposition". *IEEE Trans. Signal Processing*.
10. **Shu, D. et al. (2024)**. "Statistical Jump Model for Financial Time Series". SSRN.

---

## 12. 状态与里程碑

| 日期 | 事件 |
|------|------|
| 2026-07-22 | 文档启动 |
| Day 1-3 | 8 份文档完成 |
| Day 4-6 | CPD 模块完成 |
| Day 7-11 | 择时验证 (可选) |
| Day 11 | 决策点: 信号清晰? |
| Day 12-15 | 完整集成 (条件性) |

---

**最后更新**: 2026-07-22
**状态**: 📋 文档阶段