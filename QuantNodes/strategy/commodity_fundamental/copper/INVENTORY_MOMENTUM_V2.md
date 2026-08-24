# 库存动量v2 - 三因子确认

## 1. 背景

v1版本(纯库存变化)效果一般(Sharpe 0.31)，需要加入其他信号确认。

## 2. 三因子确认方案

### 2.1 因子定义

| 因子 | 数据源 | 逻辑 | 方向 |
|---|---|---|---|
| 铜库存变化 | inventory_daily (铜-库存) | 4周变化率 | 下降=利好 |
| 铜注册仓单 | inventory_daily (铜-注册仓单) | 4周变化率 | 下降=利好 |
| 铝库存变化 | inventory_daily (铝-库存) | 4周变化率 | 下降=利好 |

### 2.2 信号合成

```
信号 = (铜库存下降) + (注册仓单下降) + (铝库存下降)
     = 0, 1, 2, 或 3

做多: 信号 >= 2 (至少2个因子确认)
空仓: 信号 < 2
```

### 2.3 为什么用这三个因子？

1. **铜库存**: 直接反映铜供需
2. **注册仓单**: 反映真实需求(可交割库存)
3. **铝库存**: 工业金属整体趋势(跨品种确认)

## 3. 实现

```python
def compute_inventory_momentum_v2(con, change_window=4, z_threshold=0.3):
    # 主信号: 铜库存变化
    inv_cu = load_inventory("铜-库存")
    inv_change = inv_cu.pct_change(change_window)
    
    # 确认1: 注册仓单
    warrant = load_inventory("铜-注册仓单")
    warrant_change = warrant.pct_change(change_window)
    
    # 确认2: 铝库存
    inv_al = load_inventory("铝-库存")
    al_change = inv_al.pct_change(change_window)
    
    # 三因子投票
    signal = (inv_change < 0).astype(int) + 
             (warrant_change < 0).astype(int) + 
             (al_change < 0).astype(int)
    
    # 信号 >= 2 → 做多
    position = (signal >= 2).astype(float)
    
    return position
```

## 4. 预期效果

| 指标 | v1 (纯变化) | v2 (三因子确认) |
|---|---|---|
| Sharpe | 0.31 | 0.4+ |
| MaxDD | -20.7% | -15% |
| 胜率 | 37% | 50%+ |

## 5. 测试计划

1. 实现三因子确认信号
2. 单独回测
3. 与v6.2+carry组合
4. 参数优化 (change_window, z_threshold)
