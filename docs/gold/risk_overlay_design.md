# 风控覆盖层 (Risk Overlay) 设计文档

## 一、问题背景

现状所有候选策略/子策略在OOS 2010-2019窗口的MaxDD普遍在-30%以上, 远超生产要求的-15%上限:

| 策略 | OOS MaxDD | 距-15%硬约束 |
|---|---|---|
| macro_regime_signal (生产) | -29.23% | -14.23pp |
| gold_real_rate: tips_ma | -17.73% | -2.73pp |
| gold_combined: use_timing | -34.92% | -19.92pp |
| gold_multi_factor | -30.57% | -15.57pp |
| Buy & Hold | -43.82% | -28.82pp |

**单纯依靠信号层/子策略组合无法解决根本问题**, 必须加入独立的主动风控层。

## 二、设计目标

1. **首要目标**: 把MaxDD压到-15%以内(硬约束)
2. **次要目标**: 在硬约束下最大化收益(Pareto前沿)
3. **风控优先**: 任何时候风控 > 信号, 信号被风控否决时服从风控

## 三、三级风控架构

```
原始仓位信号 (signal layer)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Level 1: 波动率压制 (continuous)              │
│  当realized_vol > vol_cap时, 仓位×vol_cap/实际│
│  默认vol_cap=20%年化                         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Level 2: 软减仓 (discrete)                    │
│  当滚动MaxDD < soft_cap时, 仓位×0.5            │
│  默认soft_cap=-12%                            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Level 3: 硬止损+智能再入场 (discrete)          │
│  触发条件:                                    │
│   - 滚动MaxDD < hard_cap (-15%)              │
│   - OR 单日ret < daily_cap (-3%)              │
│  触发动作:                                    │
│   - 立即清仓 (position = 0)                   │
│   - 启动冷却期 (default 5天)                  │
│   - 冷却期内强制position=0                    │
│  冷却期结束后再入场条件:                       │
│   - vix.rolling(252).rank(pct) < 0.6 (低恐慌) │
│   - AND gvz_zscore < 1.0 (黄金低波动)         │
└─────────────────────────────────────────────┘
    │
    ▼
最终仓位 (position_final)
```

## 四、参数设计

### 4.1 默认参数(决策2 = -15%)

| 参数 | 默认 | 候选扫描范围 | 说明 |
|---|---|---|---|
| `hard_cap` | -0.15 | [-0.20, -0.10] | MaxDD硬阈值(决策2) |
| `soft_cap` | hard+0.03 | hard+0.02~+0.05 | MaxDD软阈值 |
| `daily_cap` | -0.03 | [-0.04, -0.02] | 单日DD阈值 |
| `vol_cap` | 0.20 | [0.15, 0.25] | 年化波动率上限 |
| `cooldown_days` | 5 | [3, 10] | 触发后冷却期 |

### 4.2 智能再入场逻辑(决策3 = C)

```python
# 冷却期结束后检查
vix_pct = vix.rolling(252).rank(pct= True).iloc[t]
gvz_z = ((gvz - gvz.rolling(90).mean()) / gvz.rolling(90).std()).iloc[t]

reentry_ok = (vix_pct < 0.6) and (gvz_z < 1.0)
```

**双确认语义**:
- VIX低百分位 → 市场恐慌平息
- GVZ低zscore → 黄金自身波动正常
- 任一不满足 → 继续等待(避免立即重入陷阱)

### 4.3 参数化函数签名

```python
def compute_risk_overlay(
    close: pd.Series,           # 收盘价
    position: pd.Series,         # 原始仓位信号 (0~1)
    vix: pd.Series = None,       # VIX (用于再入场确认)
    gvz: pd.Series = None,       # GVZ (用于再入场确认)
    hard_cap: float = -0.15,
    soft_cap: float = -0.12,
    daily_cap: float = -0.03,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    vol_window: int = 20,        # 波动率窗口
    reentry_vix_pct: float = 0.6,
    reentry_gvz_z: float = 1.0,
) -> pd.Series:
    """
    三级风控覆盖层 (position_final).

    返回: 调整后的仓位序列, 与输入position同index
    """
```

## 五、实现细节

### 5.1 滚动DD计算

```python
def _rolling_drawdown(close: pd.Series, position: pd.Series, window: int = 252) -> pd.Series:
    """基于历史仓位的滚动MaxDD."""
    nav = (1 + (position.shift(1) * close.pct_change())).cumprod()
    rolling_max = nav.rolling(window, min_periods=20).max()
    dd = nav / rolling_max - 1
    return dd
```

### 5.2 实现策略

```python
def compute_risk_overlay(close, position, vix=None, gvz=None, ...):
    # 准备输入
    pos = position.reindex(close.index).fillna(method='ffill').fillna(0)
    
    # 计算原始nav和rolling_dd
    nav = (1 + (pos.shift(1) * close.pct_change()).fillna(0)).cumprod()
    daily_ret = close.pct_change()
    rolling_dd = compute_rolling_drawdown(close, pos)
    
    # 1. 波动率压制 (Level 1)
    realized_vol = daily_ret.rolling(vol_window).std() * sqrt(252)
    vol_mask = realized_vol > vol_cap
    pos = pos.where(~vol_mask, pos * (vol_cap / realized_vol))
    
    # 2. 软减仓 (Level 2)
    soft_mask = rolling_dd < soft_cap
    pos = pos.where(~soft_mask, pos * 0.5)
    
    # 3. 硬止损 (Level 3)
    hard_mask = (rolling_dd < hard_cap) | (daily_ret < daily_cap)
    pos = pos.where(~hard_mask, 0.0)
    
    # 4. 冷却期 + 智能再入场
    in_cooldown = pd.Series(False, index=close.index)
    reentry_allowed = pd.Series(True, index=close.index)
    
    # 检测硬触发时刻
    trigger = hard_mask & ~hard_mask.shift(1).fillna(False)
    
    for i in range(len(close)):
        if trigger.iloc[i]:
            # 启动冷却期
            end_idx = min(i + cooldown_days, len(close) - 1)
            in_cooldown.iloc[i:end_idx+1] = True
        if in_cooldown.iloc[i]:
            pos.iloc[i] = 0.0
            # 检查是否可再入场 (冷却期最后一天)
            if i == min(i + cooldown_days, len(close) - 1) or (i > 0 and in_cooldown.iloc[i-1] and not in_cooldown.iloc[i]):
                if vix is not None and gvz is not None:
                    vix_pct = vix.iloc[i]
                    gvz_z = (gvz.iloc[i] - gvz.iloc[max(0,i-90):i].mean()) / gvz.iloc[max(0,i-90):i].std()
                    if vix_pct < reentry_vix_pct and gvz_z < reentry_gvz_z:
                        reentry_allowed.iloc[i] = True
                    else:
                        pos.iloc[i] = 0.0
                        # 延长冷却期
    
    return pos.clip(0, 1)
```

### 5.3 边界处理

```python
# warmup期: 滚动指标NaN, 视为"无风险"
nav.iloc[:20] = 1.0  # 前20天不应用风控
# 冷却期延长: 信号不满足时, 继续延长直到满足
max_cooldown_extend = 30  # 最多延长30天
```

## 六、集成位置

```python
# gold_trend_bnh.py compute_macro_regime_signal 末尾
position = (base * f_gvz * f_rate).clip(0, 1.2)

# 新增: 风控覆盖
position_final = compute_risk_overlay(
    close=close,
    position=position,
    vix=vix,
    gvz=gvz,
    hard_cap=hard_cap,  # 新参数
    ...
)
return pd.DataFrame({"close": close, "position": position_final})
```

## 七、评估指标

### 7.1 双重Pareto目标

```python
def evaluate(close, position):
    """Pareto评估: 双窗口收益 vs MaxDD."""
    results = {}
    for label, s, e in [("OOS", ...), ("IS", ...)]:
        m = calc_metrics(position, close, s, e)
        results[label] = {
            "ann_ret": m["ann_ret"],
            "max_dd": m["max_dd"],
            "calmar": m["calmar"],
        }
    # 主指标: 几何均值G (稳健)
    g = sqrt(results["OOS"]["calmar"] * results["IS"]["calmar"])
    # 副指标: 平均年化收益
    avg_ret = mean([results["OOS"]["ann_ret"], results["IS"]["ann_ret"]])
    # 副指标: 最差MaxDD
    worst_dd = min([results["OOS"]["max_dd"], results["IS"]["max_dd"]])
    return {"g": g, "avg_ret": avg_ret, "worst_dd": worst_dd, ...}
```

### 7.2 Pareto支配判定

```python
def is_dominated(point_a, point_b):
    """point_a是否被point_b支配."""
    # 在收益和MaxDD两个维度上, B都不差于A, 且至少一项更好
    # 注意MaxDD是负值, 所以"更好"是数值更大(接近0)
    return (
        (point_b.ann_ret_oos >= point_a.ann_ret_oos) and
        (point_b.max_dd_oos >= point_a.max_dd_oos) and  # 更接近0
        (point_b.ann_ret_is >= point_a.ann_ret_is) and
        (point_b.max_dd_is >= point_a.max_dd_is) and
        (point_b.ann_ret_oos > point_a.ann_ret_oos or
         point_b.max_dd_oos > point_a.max_dd_oos or
         point_b.ann_ret_is > point_a.ann_ret_is or
         point_b.max_dd_is > point_a.max_dd_is)
    )
```

## 八、扫描计划 (决策决策4 = B)

### 8.1 粗扫矩阵 (27组合)

```python
for hard_cap in [-0.10, -0.15, -0.20]:
    for daily_cap in [-0.02, -0.03, -0.04]:
        for vol_cap in [0.15, 0.20, 0.25]:
            # soft_cap = hard_cap + 0.03 (固定)
            # 测试3个信号候选
            for signal in [macro_regime, real_rate_tips_ma, momentum_long_only]:
                evaluate(signal, risk_overlay(...))
```

### 8.2 精扫Top3 (27组合)

```python
for top_combo in top3_results:
    for hard_cap in [top.hard_cap*0.8, top.hard_cap, top.hard_cap*1.2]:
        for daily_cap in [top.daily_cap*0.8, top.daily_cap, top.daily_cap*1.2]:
            for vol_cap in [top.vol_cap*0.8, top.vol_cap, top.vol_cap*1.2]:
                evaluate(top.signal, risk_overlay(...))
```

## 九、风险与诚实声明

### 9.1 预期结果

- MaxDD从-30%级别压到-15%以内
- 年化收益损失20-40%
- G指标期望改善(回撤降低的边际 > 收益损失)
- 可能无Pareto最优解(OOS/IS很难同时最优)

### 9.2 失败模式

1. 风控太严 → 错过大级别反弹
2. 冷却期太长 → 错过趋势重启
3. 波动率压制过度 → 低波期过度减仓
4. 双窗口同时Pareto最优不存在

### 9.3 诚实预期

"最安全"点可能在G≈0.5-0.8区间, 远低于生产IS窗口的G=0.65(单窗口)。
**降低期望, 接受G≈0.5已是胜利。**

## 十、决策点汇总

| 决策 | 选择 | 实施 |
|---|---|---|
| 1. Pareto目标 | A+B | G主指标 + max(ann_ret)在约束下 |
| 2. 硬风控阈值 | -15% | 默认hard_cap=-0.15 |
| 3. 冷却机制 | C | VIX<60%百分位 且 GVZ-z<1.0 |
| 4. 扫描范围 | B | 粗扫27 + 精扫27 = 54组合 |
| 5. 策略选择 | E | 输出Pareto前沿点, 你再选 |

## 十一、执行顺序

1. **Phase 1**: 实现 `compute_risk_overlay()` + 单元测试
2. **Phase 2-3**: 粗扫+精扫 (3信号 × 36风控 = 108组合)
3. **Phase 4-5**: Pareto筛选 + 输出图
4. **Phase 6**: 集成到生产 + 回归测试

预计总时间: ~2.5小时