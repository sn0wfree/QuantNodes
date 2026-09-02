# CTA换月建模现状与盲区（2026-09-02）

## 核心发现

**当前回测完全未建模换月冲击（实证确认）**：
- `roll_cost_bps` 配置项存在（默认0）
- `SlippageEngine.load_roll_dates` 从 `main_contract_mapping` 加载换月日
- **但该表数据严重不全** —— 实测只有IF/IC/IM（金融期货），黑色/有色/化工等所有实物品种均空
- 启用roll_cost_bps=5/10bp 实测Sharpe 几乎无变化（2.432/2.430 vs 无2.433），因为换月日清单空 → 0天×cost=0

## 为什么这是盲区：smoothed数据的"未来数据幻觉"

CTA策略回测用 `smoothed_*_daily` 表（back-adjusted连续合约）：
- 原始主合约换月日有跳价（RB实证：73天>2%，部分15%+）
- smoothed已用历史gap**反向修正**未来价格 → 序列平滑、跳价消失
- 策略PnL用 `pos.shift(1) × smoothed_ret` 计算 → **看起来无换月影响**
- 但实盘真实执行：近月平仓 + 远月开仓，承受近远价差（与smoothed修正方向相反）
- **回测PnL与真实PnL的差异 = roll_cost_bps，默认0 = 系统性低估**

## 实证：RB换月跳价

| 指标 | 值 |
|------|-----|
| 主合约大跳价日(>2%) | 73天 (1年内) |
| 跳价幅度 | 2%-17%（多在15日/17日附近，合约到期日） |
| smoothed价格 | 平滑（被回测调整吸收） |

## 当前机制（代码完整，但缺数据）

```python
# cta_slippage.py
def compute(self, product, date, notional, price):
    base = self.model.compute(...)  # 基础滑点
    # 换月成本: 在换月日起 roll_days 天内分摊
    if self.roll_cost_bps > 0 and product in self._roll_dates:
        for roll_date in self._roll_dates[product]:
            if 0 <= (date - roll_date).days < self.roll_days:
                base += abs(notional) * self.roll_cost_bps / 10000 / self.roll_days
                break
    return base
```

- 引擎完整
- 数据缺失：`main_contract_mapping`表只有金融期货

## 对品种PnL贡献的影响评估

回测低估换月成本 → **真实PnL系统性高估**：
- 单边roll成本典型 5-15bp（远月升水/贴水幅度）
- 对换月频繁的品种影响大：黑色/有色（每月换月）
- 农产品（黄大豆、玉米）季月合约集中，影响较小
- EC（上市晚、数据不全）影响最严重

## 修复路径（待做）
1. 拉全品种换月日期数据（按月合约编号变化检测，26对已有cross_product_spread可参考）
2. 写入 `main_contract_mapping` 表
3. 启用 `roll_cost_bps` 重跑校准回测，看实际影响
4. 如Sharpe显著下降，调整滑点预算/调仓频率

## 对EC双口径的二次校正

EC全期+26.07的25%PnL占比已识别为依赖。叠加换月未建模：
- EC上市2023-08，跨月数据可能极不完整
- 换月跳价+真实执行差距 = 实际EC贡献可能远低于25%
- **剔除EC口径预期更接近真实**

## 结论

换月建模是实盘前必须补的最后一环。代码已具备，加数据即可生效。