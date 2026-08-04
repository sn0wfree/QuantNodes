# v10.3 Risk-to-Trading: 三层防御方案

## 1. 问题

当前 H 方案 (`scheme_h_risk_aware`) 只用 equity 风险概率调节 v1.0 权重，
但风险预警系统有 7 个资产的风险概率，4 个策略权重，信息利用率极低。

## 2. 三层防御架构

```
Layer 1: 权益风险 → 防御底仓
  equity_prob → min_defensive → v1.0 权重
  (当前 H 方案已实现)

Layer 2: 商品/黄金风险 → 子策略调节
  commodity_ex_gold prob → v7.10(动量) 权重
  gold prob → v9macro(宏观因子) 权重
  silver prob → DualMom(双动量) 权重

Layer 3: 系统性风险 → 全防御
  3+ 资产同时高风险 → 强制 v1.0 = 0.90
```

## 3. 风险-策略映射关系

| 资产风险 | 关联策略 | 理由 | 调节方式 |
|---------|---------|------|---------|
| equity (沪深300) | v1.0 (锁仓轮动) | 权益高波动时，锁仓策略天然防御 | min_defensive 提升 |
| commodity_ex_gold (除金商品) | v7.10 (动量趋势) | 商品高波动→趋势策略信号不稳定 | prob>0.7 时减半 |
| gold (黄金) | v9macro (宏观因子) | 黄金避险→宏观因子受益 | prob>0.7 时加成 20% |
| silver (白银) | DualMom (双动量) | 白银波动大→双动量易反转 | prob>0.7 时减半 |

## 4. 系统性风险检测

```python
high_risk_count = sum(
    p > 0.8 for p in [equity_prob, comm_prob, gold_prob, silver_prob, ex_gold_prob]
)
if high_risk_count >= 3:
    # 3+ 资产同时高风险 → 系统性风险
    w['v1.0'] = 0.90
```

## 5. 实现改动

文件: `QuantNodes/strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py`

### 5.1 新增辅助函数

```python
def _get_all_risk_probs(date: pd.Timestamp) -> dict:
    """获取所有7个资产的风险概率."""
    from QuantNodes.strategy.momentum_etf_rotation.v10_3.risk_warning import (
        EnsembleRiskWarning, load_etf_data, CONFIGS)
    etf_data = load_etf_data()
    return {
        asset: EnsembleRiskWarning(etf_data, asset).predict(date)
        for asset in ["equity", "commodity", "gold", "soybean",
                       "silver", "energy_chem", "commodity_ex_gold"]
    }

def _apply_risk_layers(target: dict, prev_w: dict, probs: dict) -> dict:
    """三层防御: 先调节子策略, 再检查系统性风险, 最后应用约束."""
    w = dict(target)

    # Layer 1: 权益风险 → min_defensive (由调用方传入)
    # 已在 _apply_constraints 中处理

    # Layer 2: 子策略调节
    if probs.get("commodity_ex_gold", 0) > 0.7:
        w["v7.10"] = w.get("v7.10", 0) * 0.5
    if probs.get("gold", 0) > 0.7:
        w["v9macro"] = w.get("v9macro", 0) * 1.2
    if probs.get("silver", 0) > 0.7:
        w["DualMom"] = w.get("DualMom", 0) * 0.5

    # Layer 3: 系统性风险
    high_risk = sum(
        probs.get(a, 0) > 0.8
        for a in ["equity", "commodity", "gold", "silver", "commodity_ex_gold"]
    )
    if high_risk >= 3:
        w = {k: 0.0 for k in w}
        w["v1.0"] = 0.90
        return w

    # 归一化
    for k in w:
        w[k] = max(0.0, w[k])
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w
```

### 5.2 修改 scheme_h_risk_aware

```python
def scheme_h_risk_aware(prices: pd.DataFrame) -> pd.DataFrame:
    """v10.3 H: Risk-Aware - 三层防御."""
    ...
    risk_model = None

    for i in range(252, len(prices)):
        ...
        if in_d_mode:
            ...  # 同 G 方案
        elif date in monthly_dates:
            if risk_model is None:
                risk_model = _get_risk_model()
            equity_prob = risk_model.predict(date)
            min_def = _risk_prob_to_min_defensive(equity_prob)
            target = _compute_target_weights(prices, i)
            # 先应用子策略调节 + 系统性风险检测
            all_probs = _get_all_risk_probs(date)
            adjusted = _apply_risk_layers(target, prev_w, all_probs)
            # 再应用防御底仓约束
            prev_w = _apply_constraints(adjusted, prev_w, min_defensive=min_def)
        ...
```

## 6. 预期收益

| 场景 | 当前 G 方案 | H 方案三层防御 | 改进 |
|------|-----------|-------------|------|
| 2024-01 熔断式下跌 | v1.0=25% | v1.0=90% | 更早进入全防御 |
| 2024-09 924行情 | D模式(信号加权) | D模式+商品/能化减仓 | 动量策略减仓 |
| 2025-04 关税冲击 | 不确定 | equity=90%+商品减半 | 多资产协同防御 |
| 2023 低波动期 | v1.0=25% | v1.0=25% | 不变(prob<0.3) |
| 正常波动 | v1.0=25% | v1.0=25% | 不变 |

## 7. 回测验证

```bash
cd /home/ll/Public/QuantNodes && python3 -c "
from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
    load_navs, compute_nav, scheme_g_trend_adaptive, scheme_h_risk_aware,
    compute_metrics)
prices = load_navs()
g_w = scheme_g_trend_adaptive(prices)
h_w = scheme_h_risk_aware(prices)
g_nav = compute_nav(prices, g_w)
h_nav = compute_nav(prices, h_w)
# 对比 Sharpe, Calmar, MaxDD
"
```

## 8. 风险

- Layer 2 的调节系数（0.5/1.2）是启发式的，需要回测验证
- 系统风险检测阈值 3/5 需调优
- 多层防御叠加可能过度紧缩，错过反弹