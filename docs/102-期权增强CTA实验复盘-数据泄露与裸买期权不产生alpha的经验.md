# 102. 期权增强 CTA 实验复盘：数据泄露修复与"裸买期权加仓不产生 alpha"的经验

> 状态: 定稿 (2026-08-15)
> 关联: docs/98 (期权基础设施), docs/99 (文献), docs/100 (双轨设计), docs/101 (因子库)
> 代码: `QuantNodes/strategy/cta/options/options_boosted_fut_v1.py`, `scripts/options_boosted_fut_v1_bt.py`

## 一、实验目的

在 fut_v1 (72 因子, Sharpe 2.006) 基础上,尝试用期权数据增强 CTA:
1. 期权因子 (IV/VRP/Skew) 过滤 CTA 仓位
2. 裸买 Call/Put 加仓 (利用 gamma 凸性)
3. 多品种 + 全历史 (2018-2025) 验证

## 二、关键发现：三个数据泄露问题

### 2.1 回测跳过 T+1 滞后 (最严重)

**现象**: 修复前报告 Sharpe 2.72 (baseline 2.09 → +0.63),修复后 2.10 (+0.01)。

**根因**: fut_v1 原版正确路径是
```
_run_signal(sig) → _exec_sig(sig)  # T+1: T日信号 → T+1日成交
→ _signal_to_lots()
```
而自定义回测脚本直接
```
_signals() → _signal_to_lots()  # 跳过了 _exec_sig → T+0 当日成交!
```

**影响**: 用 T 日收盘信号在 T 日成交 = 前视偏差,夸大 alpha。

**教训**: 继承/复用策略类时,必须走原版的 `_run_signal` 完整路径,不能手动拆开调用链。

### 2.2 IV/VRP 分位计算用了全样本 (含未来)

**现象**: `_get_iv_rank`/`_get_vrp_percentile`/`_get_iv_zscore` 里
```python
hist = self._iv_panel[product].dropna()  # 用全部历史, 含未来!
return (hist < iv_val).sum() / len(hist)
```

**修复**: 改为只用截至 date 的历史
```python
hist = self._iv_panel[product].loc[:date].dropna()  # 无前视
```

**教训**: 任何 rolling/分位/z-score 计算,必须显式用 `loc[:date]` 或 `.rolling()`,禁止全样本 dropna。

### 2.3 回测范围错误 (混入无期权数据期)

**现象**: `load_panel()` 默认加载全量 2018-2026 (1942天/7.71年),实际想测 2023-2025 (727天/2.88年)。

**修复**: 创建策略时传 `start='2023-01-01', end='2025-12-31'`。

**教训**: `_ModuleStrategy.__init__` 有 `start`/`end` 参数,`load_panel` 用它限制数据范围。自定义脚本必须显式传参。

## 三、核心结论：裸买期权加仓不产生 alpha

### 3.1 修复后真实业绩

| 策略 | Sharpe | AnnRet | MaxDD |
|------|--------|--------|-------|
| fut_v1 基线 | 2.09 | 23.5% | -9.1% |
| + 期权因子过滤 | 2.10 | 23.4% | -9.2% |
| + 裸买期权加仓 | 2.10 | 23.4% | -9.2% |

### 3.2 期权加仓的独立 P&L

| 指标 | 值 |
|------|-----|
| 加仓天数 | 309 / 1942 |
| 期权目标 | 563 个 |
| 买入次数 | 292 次 |
| 累计权利金 | 10.8M (216.9% 资金) |
| 胜率 | 31.8% (gamma 凸性: 少数大盈利) |
| 平均年贡献 | ~1.4% |

**结论**: 即使放大到 100% 敞口 + 多持仓,gamma 凸性收益 ≈ theta 成本,对组合 Sharpe 贡献 +0.01,可忽略。

### 3.3 为什么裸买期权加仓无效

1. **gamma 凸性收益 ≈ theta 成本**: 买入 ATM 期权,大波动收益被时间价值衰减抵消
2. **顺向加仓只是放大已有敞口**: 不产生独立 alpha,只增加风险暴露
3. **VRP 逻辑反了**: 裸买期权时 VRP 低 (IV<HV, 期权便宜) 应加仓,修复前逻辑写反了
4. **T+1 + 真实成本下无优势**: 之前看到的 +0.63 Sharpe 纯属数据泄露假象

## 四、过程中积累的正确做法

### 4.1 期权加仓参数设计

```python
OptionsBoostConfig(
    boost_notional_pct=1.0,       # 覆盖 100% 期货方向敞口
    boost_max_lots=60,
    max_options_per_product=2,    # 每品种多期权
    max_premium_pct=0.10,         # 组合权利金预算 (必须! 防过度加仓)
    iv_high_threshold=0.7,        # IV 太高不加仓
    iv_zscore_max=2.0,
    min_dte=30, max_dte=60,       # 长 DTE 减 theta
    roll_dte=10,                  # 到期前换月
)
```

### 4.2 关键: 裸买期权时 VRP 的正确方向

```
裸买期权 (long gamma):
    VRP 低 (IV < HV, 期权便宜) → 加仓 (×1.3)
    VRP 高 (IV > HV, 期权贵)   → 减仓/不买

卖期权 (short gamma):
    VRP 高 (IV > HV, 权利金贵) → 加仓
    VRP 低 (IV < HV)           → 减仓
```

### 4.3 性能优化: 内存版 ATM 选择器

预计算期权目标时,不要每次 `select_call`/`select_put` 查 DuckDB:
```python
def _select_atm_from_cache(self, product, date, spot, direction):
    """从 self._option_data 内存缓存选 ATM 期权."""
    # 1. 过滤当日/最近交易日行情
    # 2. 过滤流动性 (volume/OI > 0)
    # 3. 选 OTM 方向 (call: strike>=spot, put: strike<=spot) 最近 strike
    # 4. expiry 从 underlying 推断 (rb2307 → 2023-07-15)
```
效果: 预计算 1942 天从分钟级降到秒级。

### 4.4 期权加仓的 Hold/Roll 管理

```
每持仓独立检查:
    days_left <= roll_dte (10) → 平仓 (target=0)
    days_held > 30             → 平仓
持有日: 无 target → broker 持仓不变
```

## 五、方法论经验

### 5.1 回测可信度检查清单

1. **T+1 滞后**: 信号 T 日算, T+1 成交 (用 `_exec_sig`)
2. **rolling 只用历史**: 分位/z-score 用 `loc[:date]`
3. **日期范围**: 明确 start/end, 不混入无数据期
4. **成本计入**: 确认 broker 扣了手续费/保证金
5. **OOS 验证**: 固定参数后 in-sample 训练, out-of-sample 测试

### 5.2 期权增强 CTA 的诚实验证

| 方法 | 结论 |
|------|------|
| 裸买期权顺向加仓 | ❌ 无效 (theta≈gamma) |
| 期权因子过滤 CTA | ⚠️ 效果微弱 |
| 期权卖方向加仓 (收 theta) | 待验证 |
| 期权尾部保护 (hedge) | 待验证 |

### 5.3 关键教训

1. **"看起来能赚钱"的策略, 先审计数据泄露再相信**
2. **任何继承/复用, 走原版完整调用链**
3. **期权加仓必须先设权利金预算** (max_premium_pct)
4. **顺向期权加仓 = 放大风险敞口, 不是 alpha 来源**
5. **gamma 凸性的收益和 theta 成本天然对冲** — 不能指望裸买期权白赚

## 六、后续方向

1. **卖方向**: 期权因子高 VRP 时卖 put spread 收 theta (正 carry)
2. **尾部保护**: 期权做 CTA 的 hedge (降低 MaxDD, 而非增收益)
3. **skew 因子**: 利用期权特有的 skew 信号 (非顺向加仓)
4. **OOS 严格验证**: 2023-2024 训练, 2025 测试
