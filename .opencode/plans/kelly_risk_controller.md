# Kelly-Based Risk Controller

## 1. 核心思想

Kelly公式衡量"边际风险收益比"——当预期收益为正时，Kelly分数为正，策略应加仓；当预期收益为负时，Kelly分数为负，策略应减仓。

```
kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
```

映射到min_defensive:
- Kelly > 0 → 进攻 (min_defensive低)
- Kelly < 0 → 防御 (min_defensive高)
- trend_strength调制: 强趋势 → 低防御, 弱趋势 → 高防御

## 2. 实现

```python
def kelly_based_defensive(win_rate, avg_win, avg_loss, trend_strength):
    b = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / b
    
    # Kelly negative → 更防御, Kelly positive → 更进攻
    risk_factor = max(0, -kelly)  # 取负号, 亏损越多防御越高
    
    # 趋势调制
    trend_factor = 1 - trend_strength
    
    combined = 0.6 * risk_factor + 0.4 * trend_factor
    return 0.25 + combined * (0.90 - 0.25)
```

## 3. 参数说明

| 参数 | 含义 | 范围 |
|------|------|------|
| win_rate | 滚动60d胜率 | [0,1] |
| avg_win | 平均盈利幅度 | >0 |
| avg_loss | 平均亏损幅度 | >0 |
| trend_strength | 趋势强度 (归一化) | [0,1] |
| risk_factor | max(0, -kelly), 亏损越多越高 | [0,∞) |
| trend_factor | 1 - trend_strength, 趋势越强越低 | [0,1] |
| combined | 0.6*risk + 0.4*trend | [0,∞) |
| min_defensive | 0.25 + combined * 0.65 | [0.25,0.90] |

## 4. 行为

| 场景 | win_rate | avg_win/loss | trend | kelly | min_def | 含义 |
|------|----------|-------------|-------|-------|---------|------|
| 连续盈利 | 0.6 | 1.5 | 0.8 | +0.2 | 0.34 | 激进 |
| 盈亏持平 | 0.5 | 1.0 | 0.5 | 0.0 | 0.55 | 中性 |
| 连续亏损 | 0.4 | 0.8 | 0.2 | -0.25 | 0.81 | 强防御 |

## 5. 风险

- 需要足够的历史数据计算win_rate和avg_win/avg_loss
- Kelly对极端值敏感(avg_loss=0或win_rate=0)
- 参数(0.6/0.4)需要调优
