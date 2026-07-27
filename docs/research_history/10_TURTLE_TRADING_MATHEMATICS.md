# 10 — 海龟交易法则的数学结构：可借鉴点与 V10 优化映射

> **来源**：Polson & Sokolov (2026) *The Mathematics of Turtle Trading*（工作论文），微信翻译版「Alpha搬运工」
> **原始 URL**：https://mp.weixin.qq.com/s/hyA0KOqtn0uKLy_z19CbPQ
> **目的**：梳理该文章的可借鉴点，映射到我们的 V10 策略系统（`strategy-modules` 分支），识别可以直接拿走的优化方向
> **前提**：已读 [00_TIMELINE](./00_TIMELINE.md) / [05_LESSONS_LIBRARY](./05_LESSONS_LIBRARY.md) / [09_QUICK_START](./09_QUICK_START.md)
> **本文结构**：
> 1. 文章核心论点概述
> 2. 文章的 8 个"直接可用的收获"（Section 15 原文）
> 3. 每个收获与我们代码库的映射
> 4. 6 个具体行动项（p0-p2 优先级）
> 2 个绝对不该照搬的事项（caveat）
> 5. 附录：文章核心公式速查表

---

## 一、文章核心论点（一句话）

> 海龟的六条规则（ATR / 通道突破 / 2% 风险 / 金字塔 / 止损 / 回撤减仓）不是六个互不相干的经验法则，而是 **同一个优化问题的六个手算近似** —— 贝叶斯增长最优资本配置。

用现代量化的语言重述：

> 海龟法则是一台跑在纸笔上的、被硬阈值化的 **隐马尔可夫滤波 + 风险平价 sizing + 回撤约束控制器**。

这个视角的价值在于：今天大型 CTA（Winton, Man AHL, Systematic Alpha）做的事情在架构上完全一样，只是每个组件从"手算近似"换成了"数值精确"。**分辨率提高了，骨架没变**。

---

## 二、文章的 8 个直接可用收获（原文 Section 15）

### A1: ATR → 波动率的换算常数

> `σ_annual ≈ ATR × 10.5`

**原文意义**：任何用 ATR 做 sizing 的人，可以直接把 ATR 翻译成年化波动率，不必再拟合系数。

**我们代码现状**：
- `common/config_runner.py:183` 计算 `max_drawdown`，但未发现显式的 ATR → σ 换算
- `QuantNodes/operators/talib.py:403` 有 `atr()` 算子可用
- `QuantNodes/operators/composite_dag_ops.py:214` 有 `realized_vol()` = `returns.rolling_std(window)`

**可借鉴度**：中。我们已经用 close-close σ（`realized_vol`），但若改用 ATR 体系做 sizing，这个换算常数可以省掉一次回归拟合。

---

### A2: 振幅估计量的效率优势

> **20 日振幅窗口 ≈ 100 日收盘价窗口的信息量**。五倍响应速度的免费午餐。

**原文意义**：Parkinson / Garman-Klass / Rogers-Satchell 用 OHLC 四价，同等信息量下窗口缩短 5 倍。对需要快速响应 regime 切换的场景（vol-targeting、止损触发）有直接价值。

**我们代码现状**：
- `composite_dag_ops.py:336` 已注册了：
  - `parkinson_vol`（基于 high-low 范围）
  - `garman_klass_vol`（基于 OHLC）
  - `yang_zhang_vol`（基于 OHLC 四价）
  - `realized_vol`（基于 close-close std，**当前默认**）
- `backtest_service.py:272` 的 volatility regime 检测用的是 `ts_std(close, 20) / ts_std(close, 60)` —— **正是 close-close std**，响应最慢

**可借鉴度**：**高**。Parkinson 已经写好，只等调用。

---

### A3: 漂移会污染振幅估计（**对动量策略极其关键**）

> 趋势策略恰恰在漂移最大时满仓。ATR 被趋势撑大，sizing 公式将仓位系统性地在你最该重仓时压小。

**原文意义**：Parkinson / GK 估计量是在**无漂移**假设下推导的。带漂移的布朗运动会把一部分确定性趋势误判为噪声，导致 σ 高估。**对动量策略这不是二阶项——它是系统性低估收益**。

**我们代码现状**：
- V10 的全部 4 个子策略（v1.0 locked / v9macro / v7.10 TV-PR / DualMom）都依赖趋势/动量
- **当前 sizing 走 close-close std（`realized_vol`）**，漂移污染最大
- 改用 Rogers-Satchell（基于 OHLC，交叉乘积湮灭线性趋势项）可消除此偏差
- `yang_zhang_vol` 内含 Rogers-Satchell 的一部分（漂移无关项），可以部分修正

**可借鉴度**：**最高**。这是文章 4.4 节重点强调的对趋势策略的坑，直接命中我们的架构。

---

### A4: 金字塔加仓的最坏情况有闭式界

> N 个金字塔单位最坏 P/L = `(1 - N/5) × 1.0 × 单位风险`（当 `K=2`）
> - N=1 → 最坏 = 1.0×（单仓风险）
> - N=2 → 最坏 = 1.5×（**比单仓只多 50%**）
> - N=3 → 最坏 = 1.5×（同上）
> - N=4 → 最坏 = 1.0×（早期锁定利润抵消）
> - N=5 → 最坏 = 0.0×（恰好打平）

**原文意义**：Eckhardt 说"金字塔加仓不会实质性增加灾难性损失"，这是形式化证明。

**我们代码现状**：
- V10 没有 scaling-in 逻辑：所有策略都是横截面排序后一次性调整
- 文章 14 明确指出：4/6/10/12 上限规则在独立模拟中从未触发（相关性被构造性地消除）
- **我们不该硬套金字塔**——横截面动量不存在"同一个资产连续加仓"的几何空间

**可借鉴度**：**低**。但闭式公式可用于量化 "如果未来加了 scaling-in，最坏会怎样"。

---

### A5: 回撤减仓曲线可按资金方容忍度反解

> 每回撤 10% 应削减比例 `β = (1/floor) - 1`，其中 `floor = 1 - max_tolerance`

| DD 容忍 | 每 10% DD 应削减 | 海龟原值 |
|---------|-----------------|---------|
| 40% | 14% | |
| **31%（海龟隐含）** | **20%** | 20% |
| 25% | 26% | |
| 20% | **33%** | |
| 15% | 43% | |

**原文意义**：如果你的资金方只能忍 20% 回撤，那么"每 10% 减 20%"远远不够，你需要的是每 10% 减 33%。**五分钟就能改进现有风控参数**。

**我们代码现状**：
- 当前没有显式的"回撤触发减仓"控制器（只有 `max_drawdown` 作为目标上限）
- v7.6 有止损逻辑（`test_v7_6_stop_loss.py`），但那是**全仓止损**，不是**渐进减仓**
- 海龟的 `每 10% DD 减 20%` 是一个离散化的 Grossman-Zhou（1993）解

**可借鉴度**：**高**。可以直接添加一个 `drawdown_controller.py`，输入 (peak, equity) 输出 size 乘数。

---

### A6: 分数 Kelly 的增长-安全权衡

> half-Kelly 保留 75% 最大增长率、方差减半。
> 增长率对 β 是二次的、方差是线性的 → 往下调 β 时，增长损失远慢于风险下降。

**原文意义**：MAR-ratio 对 β 是一个**宽而浅的平台**，精确取值不重要，关键是量级。你不需要精确求解 Kelly 分数，只需确保站在峰值的左边。

**我们代码现状**：
- V10 用 Vol-parity（4 策略等权风险预算），隐含的 sizing 系数由各策略 OOS Calmar 间接决定
- 当前 4 个子策略权重：v1.0 74% / v9macro 12% / v7.10 9% / DualMom 5%
- 没有做"MAR 平台扫描"来验证当前站位是否最优

**可借鉴度**：**中**。MAR 平台扫描可验证当前 sizing 是否在峰值附近。

---

### A7: 最大对数增长率 = 半个 Sharpe 平方

> `G*_max = σ²/2`

**原文意义**：审计任何"高收益 + 低 Sharpe"宣称最快的一把尺子。Sharpe 1.0 → 理论满 Kelly CAGR ≈ 65%/年。

**我们代码现状**：
- V10 OOS Sharpe = 1.991（4 策略 Vol-parity 组合）
- 理论满 Kelly CAGR = 1.991²/2 ≈ 198%/年
- 我们的实际 OOS CAGR 远低于此 → 已经在 half-Kelly 或更低区间（安全）
- **这个审计公式可以直接集成到评估流程**

**可借鉴度**：**中**。已在 `common/extended_metrics.py` 做类似分析，可加一行审计。

---

### A8: 无记忆性是正确性条件，不只是纪律问题

> 任何往策略里注入入场价、当前浮盈浮亏、或"最近手感"的改动，都在破坏它最优性所依赖的前提。

**原文意义**：Eckhardt & Polson (2000) 证明，任何最大化 Sharpe 等标准绩效统计量的规则**必须是无记忆的**。海龟规则在构造上就是无记忆的：仓位只依赖当前权益、当前价格、当前 ATR。

**我们代码现状**：
- v9macro：`factor_score↑ → 降仓防御; factor_score↓ → 加仓进攻`（`v9_factor_galaxy.py:212`）—— **factor_score 是路径依赖的**（取决于当前因子值的水平和变化方向）
- `combine_d_3source_avg.py:81`：`f3 = -beta_abs_sum`（β 大就反向）—— **β 是近期历史的函数**
- `v9/cpd/diagnose.py:60`：`50-70 | 偏多 | 适度加仓` —— 路径依赖的诊断逻辑

**可借鉴度**：**高但需谨慎**。无记忆性审计可识别路径依赖逻辑，但不一定需要全部删除（某些路径依赖可能就是 edge）。

---

## 三、与我们代码库的完整映射

### 3.1 波动率估计路径（A1 + A2 + A3）

```
当前路径（慢响应 + 漂移污染）：
  close → rolling_std(20) → realized_vol → vol_target sizing

改进路径（快响应 + 无漂移偏差）：
  OHLC → yang_zhang_vol（或 parkinson_vol）→ vol_target sizing
```

**算子就绪情况**：
- `yang_zhang_vol`：`composite_dag_ops.py:204` 已实现，用 `log(high/low)` 的 rolling_std
- `parkinson_vol`：`composite_dag_ops.py:210` 已实现
- `garman_klass_vol`：`composite_dag_ops.py:214` 已实现

**切换成本**：零（只需改一个函数调用）。

---

### 3.2 回撤控制路径（A5）

**当前缺失**：没有显式的 "DD 触发减仓" 控制器。

**海龟实现**：
```python
if cumulative_drawdown >= 10%:
    risk_allocation *= 0.8  # 每 10% DD 减 20%
```

**Grossman-Zhou (1993) 精确形式**：
```python
def drawdown_controller(equity, high_watermark, max_tolerance=0.25):
    dd = (high_watermark - equity) / high_watermark
    return max(0, 1 - (dd / max_tolerance))  # DD → 0 时全仓，DD → tolerance 时归零
```

**我们应采用**：Grossman-Zhou 形式（连续渐进）而非海龟离散形式（阶梯式）。

---

### 3.3 无记忆性审计路径（A8）

**当前潜在违反点**：
1. v9macro `factor_score` 方向性加减仓（`v9_factor_galaxy.py:212`）
2. `combine_d_3source_avg.py:81` 用近期 β 反向

**文章 11 关键结论**：
- 纯技术派（用突破）= 平坦先验 + 只靠似然 → 天然无记忆
- 基本面派 + 价格 = 先验 × 似然 → 有记忆但仍是贝叶斯正确组合
- **v9macro 的 factor_score** 是贝叶斯先验（宏观状态），属于"慢变状态"——不是记忆，是贝叶斯正确组合
- **combine_d 的 β 反向**是路径依赖（近期历史），是**真正的记忆违反**

---

### 3.4 Sizing 审计路径（A6 + A7）

**当前实现**：Vol-parity（等权风险预算）
**潜在改进**：按 MAR 平台扫描结果微调权重

**审计公式**：
```python
def kelly_audit(sharpe, actual_cagr):
    max_log_growth = sharpe**2 / 2
    actual_log_growth = np.log(1 + actual_cagr)
    kelly_fraction = actual_log_growth / max_log_growth
    return f"当前 sizing ≈ {kelly_fraction:.0%} Kelly（<50% 为安全区间）"
```

---

## 四、6 个具体行动项

### p0：立即可做（0.5 天）

#### ACT-1: 用 YZ 替换 V10 的 realized_vol 做 sizing

**目标**：消除漂移污染，同时获得 5x 响应速度提升。

**实现**：
```python
# 改前
vol = realized_vol(returns, window=20)
# 改后
vol = yang_zhang_vol(high, low, open, close, window=20)
```

**验证**：对 V10 四个子策略分别用 YZ 重跑 backtest，对比：
- Calmar 变化（期望上升）
- MaxDD 变化（期望下降）
- Sharpe 变化（期望上升或持平）

---

#### ACT-2: 添加 kelly_audit() 到评估流程

**目标**：每次 OOS 测试自动报告当前 sizing 位置。

**实现**：
```python
# common/extended_metrics.py 添加
def kelly_audit(sharpe: float, cagr: float) -> dict:
    max_log_growth = sharpe**2 / 2
    actual_log_growth = np.log(1 + cagr) if cagr > -1 else float('-inf')
    fraction = actual_log_growth / max_log_growth if max_log_growth > 0 else 0
    return {
        "max_log_growth": max_log_growth,
        "actual_log_growth": actual_log_growth,
        "kelly_fraction": fraction,
        "status": "SAFE" if fraction < 0.5 else "CAUTION" if fraction < 0.8 else "OVER-KELLY"
    }
```

---

### p1：1-2 周（设计 + 实现 + 测试）

#### ACT-3: 添加 drawdown_controller.py

**目标**：实现 Grossman-Zhou (1993) 连续回撤控制器，叠加在 Vol-parity 之上。

**输入**：(equity, peak, max_tolerance) → 输出：size_multiplier ∈ [0, 1]

**与 04_V7_V10 中的 4 步 OOS 验证结合**：在 Gate 5（实盘韧性）中加入 DD 控制器的压力测试。

---

#### ACT-4: 对 v9macro 的 factor_score 做无记忆性审计

**目标**：验证 `factor_score↑ → 降仓` 是否确实是贝叶斯先验（慢变状态），还是伪装成先验的记忆依赖。

**方法**：画出 factor_score 的自相关函数（ACF）。如果 ACF 在 lag-5+ 仍然显著 → 慢变状态（贝叶斯 OK）；如果 ACF 在 lag-1 就快速衰减 → 近期历史（记忆依赖，有问题）。

---

#### ACT-5: MAR 平台扫描

**目标**：验证 V10 4 策略当前的 vol-target 是否在 MAR 平台的峰值附近。

**实现**：对每个子策略，扫 `vol_target ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]`，画 CAGR / MaxDD / MAR 三条曲线，找平台。

---

### p2：研究层面（2-4 周）

#### ACT-6: 写 `docs/research_history/11_BAYESIAN_INTERPRETATION.md`

**目标**：把 V10 重新放在贝叶斯增长最优配置的框架里，写一份正式的理论解读文档。

**结构**：
- V10 各部件对应的贝叶斯对象（见映射表）
- "动量排序 = 漂移项截面后验均值" 的数学推导
- "Cross-sectional dispersion filter = regime 检测" 的等价性
- TV-PR + OOS gate = 收缩因子 κ

---

## 五、绝对不该照搬的 2 项（Caveat）

### ❌ 金字塔加仓

**原因**：
1. V10 是横截面动量——所有"行业"用同一规则同时进出，不存在 scaling-in 的几何空间
2. 硬加路径依赖会违反无记忆性原则（#8）
3. 文章 14 明确指出：4/6/10/12 上限规则在独立模拟中从未触发（相关性被构造性地消除），实盘中相关性会让该规则失效

### ❌ 破产指数公式直接套用

**原因**：
1. 公式假设**布朗 + 轻尾**
2. 趋势跟踪实际交易流：17% 胜率、右尾 Hill 指数约 2.5（**二阶矩存在但三阶矩不存在**）
3. 同一套规则在 sub-Kelly 下实测 DD 22.8%，与"深度回撤基本消除"的公式推算对不上
4. **公式只能量级判断，不能做风险预算**

---

## 六、文章核心公式速查表

| 编号 | 公式 | 用途 | 我们怎么用 |
|------|------|------|-----------|
| F-1 | `σ_annual = ATR × √252 / 1.5 ≈ ATR × 10.5` | ATR → 波动率换算 | ACT-1 的系数校准 |
| F-2 | `σ̂² = 0.5·ln²(H/L) + (2ln2-1)·ln²(C/O)` | Garman-Klass 一日方差 | `composite_dag_ops.py:214` 已实现 |
| F-3 | `σ̂²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` | Rogers-Satchell（漂移无关） | `yang_zhang_vol` 内含 |
| F-4 | `ΔP/L_max = (1 - N/5) × K × 单位风险` | 金字塔最坏情况 | ACT-3 的 DD 控制器校准 |
| F-5 | `β = (1/floor) - 1`，floor = 1 - DD容忍 | 每 10% DD 应减仓比例 | ACT-3 的核心公式 |
| F-6 | `G*_max = Sharpe²/2` | 满 Kelly 理论增长率 | ACT-2 kelly_audit() |
| F-7 | `P(DD > d) ≈ (1-d)^(1/β)` | 破产概率量级判断 | ⚠️ 仅量级，不作预算 |
| F-8 | `P(DD > d) ≈ (1-d)^(2β⁻¹-1)`（half-Kelly） | half-Kelly 下破产概率 | ⚠️ 仅量级，不作预算 |

---

## 七、与我们现有教训的交叉验证

| 文章结论 | 我们的教训 | 交叉结论 |
|---------|-----------|---------|
| 漂移污染振幅 | L-109: 动量信号在高波阶段失效 | 部分可解释：高波 + 高漂移 → σ 高估 → sizing 被压低，叠加动量失效 |
| 无记忆性 | L-111: look-ahead 是最常见 bug | 无记忆性 = 没有 look-ahead；L-111 本质是"人为注入了未来信息" |
| 分数 Kelly | L-310: v6.2 过拟合 | 过拟合 = 对历史样本的"全 Kelly" → 在新样本上破产 |
| MAR 平台宽浅 | L-307: 风险归因不准 | sizing 不需要精确到小数点后两位，量级对就行 |

---

## 八、开放问题（留给 STAGE 33 或 Agent 试错）

1. **v1.0 locked 的"锁定"逻辑是否符合无记忆性？**
   - v1.0 是静态持仓（锁定不交易），是否等价于"永远不满足出场条件"的无记忆策略？
   - 如果锁定是基于某个历史状态（如"当前盈利就锁"），那就是记忆依赖

2. **V10 的 4 策略等权风险预算（Vol-parity）是否是增长最优的？**
   - 理论上应该按各策略的 edge / 协方差矩阵做 Kelly，而不是等权
   - 但我们 edge 估不准（L-307），等权是现实选择

3. **Darvas 箱体的"基本面先验 × 技术面似然"框架是否适用于 A 股行业轮动？**
   - A 股有行业基本面（利润增速、ROE 等），可以构造先验
   - 然后叠加动量突破作为似然
   - 这是 v9macro 的延伸方向

---

*文档版本: 1.0*
*日期: 2026-07-27*
*来源: Polson & Sokolov (2026) + V10 codebase 映射*
