# 73 — v10 研发起点

> **基于 v1-v9 + Vol-parity 的完整经验, 确定 v10 的方向、约束和起点**

---

## 1. 已知约束 (不可违反)

### 1.1 已验证的失败路径 (v10 不应重复)

| 尝试 | 结果 | 原因 | v10 指导 |
|------|------|------|---------|
| **v8 Jump Model per-asset** | Sharpe 0.871 | 样本内过拟合 | 不做 per-asset 二元分类 |
| **v8 + risk_scalar** | Sharpe 0.691 | 仓位调整有害 | 不做动态仓位缩放 |
| **v7.10 + v9 macro 叠加** | Sharpe 1.305 | 信息冗余 (v7.10 已涵盖宏观) | 不做同源信号叠加 |
| **P_bear/LEVEL/FLOW 动态权重** | Sharpe 1.407 | 策略同源 (相关性 0.85-0.94) | 不做宏观信号驱动权重切换 |
| **v9 macro per-asset** | 过拟合 | 43 ETF 同时训练 | 不做 per-asset 宏观信号 |

### 1.2 已验证的有效路径 (v10 应继承)

| 方法 | 效果 | 原因 | v10 指导 |
|------|------|------|---------|
| **静态 Vol-parity 组合** | **Sharpe 1.535** | 风险多样性 (低 vol 锚定) | 继承组合思路 |
| **v1.0 locked (固定权重)** | Sharpe 1.596 | 无参数, 无过拟合 | 继承简单性 |
| **v7.10 TV-PR (expanding window)** | Sharpe 1.238 | 无前视 + 动量分有效 | 继承 TV-PR 机制 |
| **v9 macro risk_scalar** | Sharpe 1.014 | 宏观信息有效 (但有限) | 继承宏观因子 |

### 1.3 数字约束

| 约束 | 值 | 来源 |
|------|-----|------|
| 最低 Sharpe 目标 | ≥ 1.20 (单策略) / ≥ 1.50 (组合) | Vol-parity baseline |
| 最大 MaxDD | ≤ -15% (单策略) / ≤ -5% (组合) | 风控要求 |
| 最大 MaxDDDays | ≤ 150 天 | 客户要求 |
| 最小 AnnRet | ≥ 15% (单策略) / ≥ 8% (组合) | 收益要求 |
| 成本假设 | 5bp commission + 5bp slippage | v7.10 默认 |

---

## 2. v10 方向探索

### 2.1 可行方向 (基于已知约束)

#### 方向 A: 独立策略组合 (最稳妥)

**思路**: 寻找与 v7.10/v1.0 **低相关** 的新策略, 加入 Vol-parity 组合

| 新策略候选 | 预期相关性 | 复杂度 | 风险 |
|-----------|-----------|--------|------|
| 均值回归 (短周期) | 低 (与动量负相关) | 中 | 过拟合 |
| 波动率择时 (VIX-based) | 低 | 低 | 信号稀疏 |
| 宏观 regime 切换 | 中 | 高 | 过拟合 |
| 多因子 alpha (基本面) | 低 | 高 | 数据不足 |

**优势**: 不改变现有 Vol-parity 框架, 只增加新组件
**风险**: 新策略可能过拟合

#### 方向 B: v7.10 内部优化 (最直接)

**思路**: 在 v7.10 框架内优化, 不改变组合结构

| 优化点 | 描述 | 预期提升 |
|--------|------|---------|
| TV-PR 窗口优化 | 测试不同 expanding window 起点 | +0.05 Sharpe |
| 因子权重优化 | 测试不同因子集/权重 | +0.03 Sharpe |
| 换仓频率优化 | 周频 vs 双周 vs 月频 | +0.02 Sharpe |
| 成本优化 | 降低换手率 | +0.05 Sharpe |

**优势**: 最小改动, 最低风险
**风险**: 提升有限 (v7.10 已接近上限)

#### 方向 C: 信号层创新 (最激进)

**思路**: 开发全新信号源, 与现有 momentum 互补

| 信号候选 | 描述 | 独立性 | 复杂度 |
|---------|------|--------|--------|
| 情绪因子 (新闻/社交) | NLP 情绪分 | 高 | 高 |
| 资金流因子 (北向/融资) | 资金流向 | 中 | 中 |
| 微观结构因子 (买卖盘) | 订单流 | 高 | 高 |
| 另类数据 (卫星/电商) | 非传统数据 | 高 | 极高 |

**优势**: 可能带来真正独立的 alpha
**风险**: 最高, 需要新数据源

### 2.2 推荐优先级

```
1. 方向 B (v7.10 内部优化) — 立即可做, 低风险
2. 方向 A (独立策略组合) — 中期, 中风险
3. 方向 C (信号层创新) — 长期, 高风险
```

---

## 3. v10 架构约束

### 3.1 必须保持

| 约束 | 原因 |
|------|------|
| expanding window (无前视) | v7.10 验证有效 |
| 月度调仓 | 避免过度交易 |
| 5bp+5bp 成本假设 | 真实场景 |
| OOS 验证 (2022-2026) | 防止过拟合 |
| 9 指标 × 9 区间评估 | 标准化对比 |

### 3.2 可以改变

| 参数 | 当前值 | 可测试范围 |
|------|--------|-----------|
| 调仓频率 | 月度 | 周频/双周/季频 |
| 因子集 | 5 因子 | 3-10 因子 |
| 组合方式 | Vol-parity | 等权/风险平价/优化 |
| 成本模型 | 固定 10bp | 动态/分档 |

### 3.3 禁止

| 操作 | 原因 |
|------|------|
| 全样本训练 | 过拟合 (v8 教训) |
| per-asset 二元分类 | 过拟合 (v8 教训) |
| 宏观信号动态权重 | 无效 (docs/71) |
| 同源信号叠加 | 冗余 (docs/69) |

---

## 4. v10 起点代码

### 4.1 基线: Vol-parity 3 组合

```python
# v10 起点: 基于 Vol-parity 框架
# 新策略只需提供 NAV Series, 即可加入组合

import pandas as pd

def vol_parity_weights(navs: dict, target_vol: float = 0.08) -> dict:
    """Vol-parity 权重计算."""
    vols = {}
    for name, nav in navs.items():
        rets = nav.pct_change().dropna()
        vols[name] = float(rets.std() * np.sqrt(252))

    weights = {}
    for name in navs:
        weights[name] = (target_vol / len(navs)) / vols[name]

    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}


def combine_navs_monthly(navs: dict, weights: dict) -> pd.Series:
    """月末加权组合 NAV."""
    common = navs[list(navs.keys())[0]].index
    for nav in navs.values():
        common = common.intersection(nav.index)
    common = common.sort_values()

    nav_combined = pd.Series(1.0, index=common, dtype=float)
    last_month = None
    month_start = {}

    for i, d in enumerate(common):
        if d.month != last_month and last_month is not None:
            # 月末: 计算月度收益
            port_ret = 0
            for name in navs:
                if name in month_start and month_start[name] > 0:
                    month_ret = navs[name].loc[d] / month_start[name] - 1
                    port_ret += weights[name] * month_ret
            nav_combined.iloc[i] = nav_combined.iloc[i-1] * (1 + port_ret)

        # 更新月度起始值
        if d.month != last_month:
            for name in navs:
                month_start[name] = navs[name].loc[d]
        last_month = d.month

    return nav_combined
```

### 4.2 新策略接入模板

```python
# 新策略只需实现这个接口
def compute_new_strategy_nav(data: pd.DataFrame) -> pd.Series:
    """计算新策略 NAV.

    输入: data (日频 OHLCV + 因子)
    输出: nav (日频净值 Series, index=日期)

    约束:
    - expanding window (无前视)
    - 月度调仓
    - 包含 10bp 成本
    """
    # TODO: 实现新策略
    pass

# 接入 Vol-parity
navs = {
    'v1.0': v1_nav,
    'v9macro': v9_nav,
    'v7.10': v7_nav,
    'new': compute_new_strategy_nav(data),  # 新策略
}
weights = vol_parity_weights(navs)
combined = combine_navs_monthly(navs, weights)
```

---

## 5. 验证清单

### 5.1 新策略必须通过

| 检查项 | 标准 | 失败 = |
|--------|------|--------|
| OOS Sharpe | ≥ 1.00 | 不加入组合 |
| OOS MaxDD | ≤ -20% | 不加入组合 |
| 与 v7.10 相关性 | < 0.70 | 信息冗余, 不加入 |
| 与 v1.0 相关性 | < 0.50 | 同源, 不加入 |
| 全样本 vs OOS Sharpe 差 | < 30% | 过拟合嫌疑 |

### 5.2 组合必须通过

| 检查项 | 标准 | 失败 = |
|--------|------|--------|
| OOS Sharpe | ≥ 1.50 | 不如 Vol-parity |
| OOS MaxDD | ≤ -6% | 风险过高 |
| OOS MaxDDDays | ≤ 150 | 恢复太慢 |
| OOS AnnRet | ≥ 8% | 收益不足 |

---

## 6. 下一步行动

### 立即 (今天)
- [ ] 确认 v10 方向 (A/B/C)
- [ ] 创建 v10 策略模板

### 本周
- [ ] 实现方向 B (v7.10 内部优化)
- [ ] 测试 3-5 个优化点

### 下周
- [ ] 评估方向 A (独立策略候选)
- [ ] 开始方向 C 调研 (如选择)

---

## 7. 参考文档

| 文档 | 关键内容 |
|------|---------|
| docs/72-vol_parity_method_record.md | Vol-parity 完整方法 |
| docs/70-three_strategy_combination.md | 3 策略组合实验 |
| docs/71-pbear_dynamic_weighting.md | 动态加权失败原因 |
| docs/69-v7_10_v9_macro_combination.md | 信息冗余验证 |
| docs/68-standard_comparison.md | 25 策略标准对比 |
| docs/67-v8_dynamic_position_master.md | v8 完整历程 |
