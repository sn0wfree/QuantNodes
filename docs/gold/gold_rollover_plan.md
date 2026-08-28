# 黄金策略 — 渐进式换月方案

## 背景

### 问题

当前策略使用 `ak.futures_main_sina(symbol="AU0")` 获取SHFE黄金主力连续合约数据。
AU0在换月时存在**价格跳空**（3-8%），被O2O回测错误计算为策略收益/亏损。

### 换月跳空统计（2010年后）

- 检测到 51 个换月事件
- 跳空范围: -5.53% ~ +7.67%
- 标准差: 3.64%
- 2017年前多为两日换月（旧合约OI暴跌→新合约上线），gap较大
- 2019年后多为单日换月，gap较小但仍显著

### 关键换月日

| 日期 | 跳空 | 说明 |
|------|------|------|
| 2011-08-25 | -4.57% | 旧合约下线 |
| 2013-04-15 | -4.93% | 实际金价暴跌+换月叠加 |
| 2025-04-07 | -5.04% | 近期最大 |
| 2026-03-24 | +6.20% | 近期最大正向 |

## 方案：渐进式换月（4天线性插值）

### 核心逻辑

**不使用AU0自动换月**，改为：
1. 获取所有AU合约的日度数据（价格+OI）
2. 每天选择OI最大的合约作为主力
3. 当新合约OI超过旧合约OI时，进入换月期
4. 换月期4天，每天线性切换25%

### 权重分配

| 天数 | 旧合约权重 | 新合约权重 | 说明 |
|------|-----------|-----------|------|
| Day 0 | 100% | 0% | 换月前 |
| Day 1 | 75% | 25% | 开始换月 |
| Day 2 | 50% | 50% | 中间点 |
| Day 3 | 25% | 75% | 接近完成 |
| Day 4 | 0% | 100% | 换月完成 |

### 价格计算

```python
effective_close[T] = w_old[T] * old_close[T] + w_new[T] * new_close[T]
effective_open[T]  = w_old[T] * old_open[T]  + w_new[T] * new_open[T]
```

换月期内每天的价格是新旧合约的加权平均，平滑过渡，无跳空。

## 实现步骤

### Step 1: 获取所有AU合约数据

**脚本**: `scripts/cmf/fetch_gold_contracts.py`

- 从akshare获取所有AU合约历史数据
- `ak.futures_zh_daily_sina(symbol='AU2412')` → 具体合约
- 存储到DuckDB: `futures_all_contracts_au`
- 字段: trade_date, contract, open, close, volume, oi
- 时间范围: 2010年至今

### Step 2: 主力合约识别与换月检测

**脚本**: `scripts/cmf/detect_gold_main.py`

- 每天选择OI最大的合约作为主力
- 检测换月起点: 新合约OI > 旧合约OI
- 输出切换记录: switch_date, old_contract, new_contract

### Step 3: 渐进式价格构建

**脚本**: `scripts/cmf/build_gold_smooth.py`

- 换月期外: 直接使用主力合约价格
- 换月期内: 4天线性插值
- 输出: `futures_main_close_au_adj`
- 字段: trade_date, open, close, high, low, volume, oi, is_rollover

### Step 4: 数据比对验证

- 对比调整前后收益率序列
- 计算Sharpe/CAGR/MaxDD差异
- 可视化换月日附近收益分布

### Step 5: 更新策略回测

- 修改 `prepare_gold_overview_data.py` 使用调整后价格
- 重新生成 `overview_data.pkl`
- 重新生成HTML报告

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/cmf/fetch_gold_contracts.py` | 新增 | 获取所有AU合约数据 |
| `scripts/cmf/detect_gold_main.py` | 新增 | 主力合约识别与换月检测 |
| `scripts/cmf/build_gold_smooth.py` | 新增 | 渐进式换月价格构建 |
| `data/cmf/fundamental.duckdb` | 修改 | 新增表 |
| `scripts/cmf/prepare_gold_overview_data.py` | 修改 | 使用调整后价格 |
| `reports/gold/overview_data.pkl` | 重新生成 | |
| `reports/gold/report.html` | 重新生成 | |

## 风险与注意事项

1. **akshare数据完整性**: 已到期合约可能无法获取历史数据，需要验证
2. **换月起点判断**: 需要准确判断何时开始换月，避免过早或过晚
3. **流动性考虑**: 换月早期新合约流动性可能不足，价格可能不准确
4. **回测偏差**: 如果新合约在换月期尚未充分交易，价格可能有偏差

## 预期效果

- 消除换月跳空（3-8% → 平滑过渡）
- 业绩指标更准确（不受换月噪声影响）
- 回测更贴近真实交易（实际交易也是渐进换月）

## 状态

- [ ] Step 1: 获取所有AU合约数据
- [ ] Step 2: 主力合约识别与换月检测
- [ ] Step 3: 渐进式价格构建
- [ ] Step 4: 数据比对验证
- [ ] Step 5: 更新策略回测
