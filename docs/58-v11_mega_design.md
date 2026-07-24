# v11_mega 统一大策略 — 设计文档

> **编号**: 58
> **日期**: 2026-07-23
> **状态**: 设计完成, 等待实施
> **基于**: v1-v10 全版本 + Brinson 归因 + 9 策略分析 + TV-PR 调优

---

## 一、设计动机

### 1.1 现有策略体系问题

经过 v1-v10 十个版本的迭代, 现有体系存在 3 个核心问题:

| 问题 | 具体表现 | 解决方案 |
|------|----------|----------|
| **碎片化** | 9 个独立策略散落在 v9/, v10/ 等目录, 无法统一调用 | 集成到 v11_mega Layer 2D ensemble |
| **缺乏自适应** | 每个策略只适用于特定行情 (熊/震/牛) | 加 Layer 0 多层 regime 检测 + 动态切换 |
| **风控单层** | v9/v10 仅依赖动态仓位, 无组合级约束 | Layer 3 三层风控 (仓位+组合+择时) |

### 1.2 设计目标

v11_mega 统一大策略要解决:

1. **多市场环境自适应**: bull/neutral/bear/crisis 四状态自动切换
2. **策略集成**: 9 个子策略 Ensemble 集成, 加权输出
3. **多层风控**: 仓位级 + 组合级 + 择时级 三层防护
4. **可调度框架**: 既可作为统一策略直接实盘, 也可作为调度框架扩展

### 1.3 性能预期

| 指标 | v9 银河方案 | v10 (W) | v11_mega (W) 预期 |
|------|-------------|---------|-------------------|
| Sharpe | 1.230 | 1.030 | **1.4 - 1.6** |
| Calmar | 1.196 | 0.815 | **1.0 - 1.3** |
| MaxDD | -13.72% | -9.09% | **≤ -12%** |
| 年化 | 16.41% | 7.41% | **12% - 18%** |

---

## 二、7 层架构设计

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 0: 调度层 (Scheduler)                                       │
│  ── 多层信号合成 regime = macro_signal ⊕ Jump Model ⊕ HMM          │
│  ── 输出: regime_state ∈ {bull, neutral, bear, crisis}            │
│  ── 用途: 驱动 Layer 2D ensemble 权重 + Layer 5 组合构建           │
└────────────────────────────────────────────────────────────────────┘
                                 ▲
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1: 宏观择时 (Macro) ── 复用 v10                             │
│  ── 5 宏观因子 + 熵权 + TV-PR (默认开启)                          │
│  ── 输出: macro_score ∈ [-1, +1]                                   │
└────────────────────────────────────────────────────────────────────┘
                                 ▲
┌────────────────────────────────────────────────────────────────────┐
│  Layer 2: 子策略集成 (Sub-Strategy Ensemble) ← 核心新增            │
│  ── 2A: 行业轮动 (v9 中信行业, regime 条件)                       │
│  ── 2B: 风格轮动 (v4 IC 驱动 + v9 中信多因子)                     │
│  ── 2C: 横截面选股 (v9 citic_multifactor, K 自适应 5/10/15)      │
│  ── 2D: Ensemble 权重融合 (★ 9 个子策略加权)                      │
│       └─ A1 银河方案  A2 中信多因子  A3 银河因子配置              │
│       └─ A4 中信里昂全天候  A5 中信大类资产  A6 中信行业轮动       │
│       └─ A7 基础风险平价  A8 60/40股债  A9 等权基准               │
└────────────────────────────────────────────────────────────────────┘
                                 ▲
┌────────────────────────────────────────────────────────────────────┐
│  Layer 3: 多层风险控制 (Risk) ★ 加强                              │
│  ── 3A 仓位级: pos 公式 + bear_prob 调整                          │
│  ── 3B 组合级: RP 底仓 + 单 ETF 上下限 (15%/0.5%) + 行业暴露约束 │
│  ── 3C 择时级: 宏观切换 + 极端熔断 (crisis_flag)                  │
└────────────────────────────────────────────────────────────────────┘
                                 ▲
┌────────────────────────────────────────────────────────────────────┐
│  Layer 4: 动态仓位 (Position) ── 复用 v10                        │
│  ── pos_t = (0.7 - 0.5·z_score).clip(0.2, 1.0)                  │
│  ── pos_t *= (1 - bear_prob × 0.5)                              │
│  ── z_score 合成: macro(0.4) + sector(0.2) + style(0.2) + IC(0.2)│
└────────────────────────────────────────────────────────────────────┘
                                 ▲
┌────────────────────────────────────────────────────────────────────┐
│  Layer 5: 组合构建 (Portfolio) ★ 重组                            │
│  ── w_final = pos × Σ(α_k^regime × w_k)                          │
│  ── α_k 由 Layer 0 regime + Layer 2D ensemble 决定               │
│  ── w_k 由各子策略单独计算 (每个子策略输出一组完整权重)          │
└────────────────────────────────────────────────────────────────────┘
```

### 2.1 三种融合方式的落实

| 融合方式 | 在 MegaStrategy 中的应用 | 落地位置 |
|----------|--------------------------|----------|
| **Stacked (分层叠加)** | Layer 0→1→2→3→4→5 的宏观→选股→仓位链路 | 主架构 |
| **Ensemble (加权集成)** | Layer 2D 9 个子策略按权重集成 | 子策略融合 |
| **Regime Switch (条件分支)** | Layer 0 检测 regime, Layer 5 切换 α_k 权重 | 调度机制 |

---

## 三、9 子策略 Ensemble 加权表

### 3.1 分 regime 加权表

| regime | 银河方案 | 中信多因子 | 银河因子 | 中信里昂 | 中信大类 | 中信行业 | 基础RP | 60/40 | 等权 | cash |
|--------|----------|------------|----------|----------|----------|----------|--------|-------|------|------|
| **bull** | 0.20 | 0.25 | 0.10 | 0.05 | 0.05 | 0.15 | 0.10 | 0.00 | 0.10 | 0.00 |
| **neutral** | 0.15 | 0.20 | 0.15 | 0.10 | 0.10 | 0.10 | 0.10 | 0.00 | 0.10 | 0.00 |
| **bear** | 0.05 | 0.10 | 0.10 | 0.15 | 0.10 | 0.05 | 0.30 | 0.00 | 0.15 | 0.00 |
| **crisis** | 0.00 | 0.05 | 0.05 | 0.10 | 0.05 | 0.00 | 0.40 | 0.00 | 0.10 | 0.25 |

### 3.2 加权设计原理

- **bull**: 进攻为主, 中信多因子 (0.25) + 银河方案 (0.20) + 行业轮动 (0.15) 合计 60%
- **neutral**: 平衡配置, 选股与防御各半, 中信多因子 (0.20) + 风险平价 (0.10) + 等权 (0.10)
- **bear**: 防御为主, 风险平价 (0.30) + 中信里昂 (0.15) + 等权 (0.15) 合计 60%
- **crisis**: 极保守, 风险平价 (0.40) + cash (0.25) + 等权 (0.10) 合计 75%
- **60/40 一律 0.00**: A 股相关性失效, 禁用

---

## 四、Layer 0 多层 Regime 检测

### 4.1 信号源 (4 个)

| 信号 | 来源 | 输出 | 权重 |
|------|------|------|------|
| macro_score | Layer 1 (v10) | 连续 [-1, 1] | 0.40 |
| jump_states | Layer 3 (v8) | bull/bear | 0.30 |
| hmm_state | v4 HMM | bull/bear/transition | 0.20 |
| vol_percentile | 新增 | 滚动 vol 分位 | 0.10 |

### 4.2 合成规则

```python
def detect_regime(macro, jump, hmm, vol_pct):
    """多层信号合成 regime.

    信号源:
    - macro_score (Layer 1, v10 复用): 连续 [-1, 1]
    - jump_states (Layer 3, v8 复用): bull/bear
    - hmm_state (v4 HMM, 新接入): bull/bear/transition
    - vol_percentile (新增): 滚动 vol 分位数 → crisis 触发

    合成规则:
    - 任一信号 strong bear → bear
    - 3/4 信号 neutral → neutral
    - ≥2 信号 bull → bull
    - vol_percentile > 0.95 + bear_prob > 0.5 → crisis
    """
    # Crisis: vol 极高 + bear 确认 (双条件 AND)
    if vol_pct > 0.95 and (jump == 'bear' or hmm == 'bear'):
        return 'crisis'

    # Bear: 多数信号指向 bear
    bear_count = sum([macro < -0.3, jump == 'bear', hmm == 'bear'])
    if bear_count >= 2:
        return 'bear'

    # Bull: 多数信号指向 bull
    bull_count = sum([macro > 0.3, jump == 'bull', hmm == 'bull'])
    if bull_count >= 2:
        return 'bull'

    return 'neutral'
```

### 4.3 Hysteresis 抗抖动

regime 切换需连续 4 周确认, 避免频繁切换:

```python
class RegimeStateMachine:
    def __init__(self, hold_weeks=4):
        self.hold_weeks = hold_weeks
        self.current_state = 'neutral'
        self.pending_state = None
        self.pending_count = 0

    def update(self, new_state):
        if new_state == self.current_state:
            self.pending_state = None
            self.pending_count = 0
            return self.current_state

        if new_state == self.pending_state:
            self.pending_count += 1
            if self.pending_count >= self.hold_weeks:
                self.current_state = new_state
                self.pending_state = None
                self.pending_count = 0
        else:
            self.pending_state = new_state
            self.pending_count = 1

        return self.current_state
```

---

## 五、Layer 3 多层风控

### 5.1 三层级风控设计

| 层级 | 控制类型 | 参数 | 触发条件 |
|------|----------|------|----------|
| **3A 仓位** | 总仓位暴露 | pos ∈ [0.2, 1.0] | z_score 异常 |
| **3A 仓位** | bear_prob 调整 | pos ×= (1-bear×0.5) | bear_prob > 0.3 |
| **3B 组合** | 风险平价底仓 | inv_vol 加权 | 高 Vol 持续 |
| **3B 组合** | 单 ETF 上限 | cap=15% | 单 ETF 突破 |
| **3B 组合** | 单 ETF 下限 | floor=0.5% | 单 ETF 跌破 |
| **3B 组合** | 行业暴露 | ≤30% | 行业集中 |
| **3C 择时** | 宏观切换 | regime_state 切换 | macro_signal 跨阈值 |
| **3C 择时** | 极端熔断 | pos → 0.2 | crisis_flag |
| **3C 择时** | cash 比例 | 25% | crisis 模式 |

### 5.2 crisis mode 规则

crisis 触发条件 (双条件 AND):

- vol_percentile > 0.95 (历史 5% 极端波动)
- (jump == 'bear' OR hmm == 'bear')

crisis 应对:

- pos 强制降到 0.2
- cash 比例 25%
- 风险平价底仓权重提到 0.40
- 选股权重全部 ≤ 0.10

### 5.3 行业暴露约束

```python
INDUSTRY_CAPS = {
    '金融': 0.30,    # 银行+非银合计上限
    '消费': 0.30,
    '医药': 0.25,
    '科技': 0.30,
    '周期': 0.25,
    '海外': 0.20,
    '黄金': 0.15,
}

def apply_industry_constraint(weights, classification):
    for industry, cap in INDUSTRY_CAPS.items():
        industry_mask = classification == industry
        if industry_mask.sum() > 0:
            industry_weight = weights[industry_mask].sum()
            if industry_weight > cap:
                scale = cap / industry_weight
                weights[industry_mask] *= scale
    return weights
```

---

## 六、与现有体系关系

```
v1-v3 (历史)         ─── 已归档
v4 (Stage 30 完成)   ── 复用 factor_timing_v4.py (Layer 2B IC 驱动)
v5/v6 (历史)         ─── 已归档
v7 (TV-PR)           ── 复用 tvpr_estimator.py (Layer 1)
v8 (Jump Model)      ── 复用 jump_model.py (Layer 0 + 3A)
v9 (9 子策略)        ── 全部 9 个子策略作为 Layer 2D ensemble 输入
v10 (5 层框架)       ── 保留独立, Layer 1/4 直接复用
v11_mega (新建)      ── ★ 整合所有版本, 7 层 + ensemble + 多层风控
```

### 6.1 复用 vs 新增

| 组件 | 复用 | 修改 | 新增 |
|------|------|------|------|
| Layer 0 scheduler | — | — | ★ 新增 |
| Layer 1 macro_layer | v10 完全复用 | — | — |
| Layer 2A 行业轮动 | v10 复用 | regime 调节细化 | — |
| Layer 2B 风格轮动 | v10 复用 | IC 加权替换 v9 中信多因子 | — |
| Layer 2C 横截面选股 | v9 citic_multifactor | K 自适应 (regime 切换 5/10/15) | — |
| Layer 2D Ensemble | — | — | ★ 新增核心 |
| Layer 3 风险控制 | v8/v10 复用 | 加组合级约束 + 极端熔断 | ★ 部分新增 |
| Layer 4 动态仓位 | v10 完全复用 | z_score 合成权重重调 | — |
| Layer 5 组合构建 | v10 部分复用 | ensemble 加权替换单一 RP | ★ 重写 |

---

## 七、文件结构

```
QuantNodes/strategy/momentum_etf_rotation/v11_mega/
├── __init__.py
├── config_mega.py            # 配置中心 (5 层 + ensemble + 风控)
├── scheduler.py              # Layer 0: 调度/Regime Detection
├── macro_layer.py            # Layer 1: 复用 v10
├── sub_strategy_ensemble.py  # Layer 2: ★ 新增 5 子策略融合
│   ├── class SubStrategyEnsemble:
│   │     ├── industry_rotation()    # 2A
│   │     ├── style_rotation()       # 2B
│   │     ├── multifactor_picking()  # 2C
│   │     └── ensemble_weights()     # 2D ★ 核心
├── risk_layer.py             # Layer 3: 多层风控
│   ├── position_risk()       # 3A
│   ├── portfolio_risk()      # 3B
│   └── timing_risk()         # 3C
├── position_layer.py         # Layer 4: 复用 v10
├── portfolio_layer.py        # Layer 5: 重组
├── mega_strategy.py          # 主入口 (7 层串联)
└── backtest_mega.py          # 回测引擎 (复用 v9 metrics)

scripts/v11_mega/
├── mega_backtest.py          # 单策略回测
├── mega_compare.py           # 对比 v9 银河 + v10
├── mega_regime_test.py       # ★ 分阶段回测 (熊/震/牛 3 段)
├── mega_sensitivity.py       # ensemble 权重敏感性
└── mega_crisis_test.py       # ★ 模拟 crisis 场景
```

---

## 八、实施步骤

```
Step 1: docs/58-v11_mega_design.md (本文档)  ← 当前
Step 2: v11_mega/config_mega.py (配置中心)
Step 3: v11_mega/scheduler.py (Layer 0 调度)
Step 4: v11_mega/sub_strategy_ensemble.py (Layer 2 ★ 核心)
Step 5: v11_mega/risk_layer.py (Layer 3 多层风控)
Step 6: v11_mega/portfolio_layer.py (Layer 5 重组)
Step 7: v11_mega/mega_strategy.py (主入口串联)
Step 8: v11_mega/backtest_mega.py (回测引擎)
Step 9: scripts/v11_mega/mega_backtest.py
Step 10: scripts/v11_mega/mega_compare.py
Step 11: scripts/v11_mega/mega_regime_test.py
Step 12: docs/59-v11_mega_implementation.md + reports/
```

---

## 九、预期性能

### 9.1 三窗口性能预期

| 窗口 | v9 银河 | v10 (W) | v11_mega (W) 预期 |
|------|---------|---------|-------------------|
| v9 窗口 (2021-2026) | 1.230 | 1.030 | **1.4 - 1.6** |
| 完整窗口 (2018-2026) | — | 0.823 | **0.9 - 1.1** |
| 熊市段 (2021-2022) | — | — | **> 0.5** |
| 震荡段 (2022-2024) | — | — | **> 1.5** |
| 牛市段 (2024-2026) | — | — | **> 1.2** |

### 9.2 关键验证指标

- **Sharpe**: ≥ 1.3 (v9 银河 1.23, v10 1.03)
- **Calmar**: ≥ 1.0 (改善 MaxDD)
- **MaxDD**: ≤ -12% (v9 -13.7%)
- **分阶段稳定性**: 三段 Sharpe 都 > 0.5 (避免单段过拟合)
- **Ensemble 贡献**: 9 子策略中至少 5 个 Sharpe > 0.3

---

## 十、风险与回滚

| 风险 | 缓解措施 |
|------|----------|
| Ensemble 权重过拟合 | 加权表来自历史分阶段归因, 不做 OOS 调参 |
| 9 子策略计算开销 | v10/v9 模块化复用, 单次回测 < 30s |
| regime 切换频繁 | 加 hysteresis (state 切换需连续 4 周确认) |
| crisis 误触发 | 用 vol_percentile 0.95 + bear_prob > 0.5 双条件 |
| 与 v10 冲突 | v10 完全独立, 互不影响 |
| HMM 缺失 | v4 HMM 已有代码, 直接复用, 失败则降级为只用 macro+jump |

---

## 十一、产出清单

### 文档

- `docs/58-v11_mega_design.md` (本文档)
- `docs/59-v11_mega_implementation.md` (实施记录)
- `docs/60-v11_mega_results.md` (最终结果报告)

### 代码

- `QuantNodes/strategy/momentum_etf_rotation/v11_mega/` (10 文件)
- `scripts/v11_mega/` (5 个脚本)

### 报告

- `reports/momentum_etf_rotation/v11_mega/v11_mega_backtest_results.csv`
- `reports/momentum_etf_rotation/v11_mega/v11_mega_backtest.png`
- `reports/momentum_etf_rotation/v11_mega/v11_mega_compare.csv`
- `reports/momentum_etf_rotation/v11_mega/v11_mega_compare.png`
- `reports/momentum_etf_rotation/v11_mega/v11_mega_regime_decomposition.md`
- `reports/momentum_etf_rotation/v11_mega/v11_mega_sensitivity.csv`

**总产出**: 19 项

---

## 十二、下一步

待用户审核本文档后, 按 Step 2 → Step 12 依次实施。