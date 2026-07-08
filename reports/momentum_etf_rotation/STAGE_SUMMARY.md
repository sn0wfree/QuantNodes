# 动量 ETF 轮动策略 — 当前状态总结

> 最后更新: 2026-07-07
> 当前阶段: Stage 13 (交易成本建模) 完成
> 下一阶段: 待定 (协方差优化探索中)

---

## 1. 策略核心

### 1.1 一句话定义

**基于沪深 44 只 ETF 的月度动量轮动策略**, 4 步组合管理:
1. 同指数去重 + 剔高相关 (相关 > 0.9)
2. 强制分散 (A 股宽基+行业 ≤ 3, 港股 ≤ 1, 必含商品+海外)
3. 逆波动加权 (权重 ∝ 1/σ)
4. 止损 + 补位 (跌破 55 日均线 + 排名跌出后 30% 分位)

### 1.2 最佳配置 (Stage 9-C + Stage 13)

```python
RotationConfig(
    lookback=90, top_n=10,
    vol_targeting=VolTargeting(
        enabled=True, target_vol=0.15, lookback=60,
        min_scale=0.3, max_scale=1.5,
    ),
    cost_model=CostModel(
        enabled=True, commission_bp=5, slippage_bp=10,
        impact_factor=0.1,
    ),
)
```

### 1.3 关键指标 (2019~2026, 86 次调仓)

| 指标 | 当前最优 | vs CICC 报告 |
|------|---------|-------------|
| Calmar | **0.98** | 0.76 (+29%) |
| 最大回撤 | -6.94% | -18.78% (远优) |
| 年化收益 | 6.83% | (含实盘成本) |
| OOS Calmar (2024-2026) | 1.00 | - |

---

## 2. 已完成的研发阶段

### 2.1 阶段总览

| 阶段 | 内容 | 结果 | 当前状态 |
|------|------|------|---------|
| 1-5 | 基础 (数据/回测/校验/文档) | ✅ | baseline |
| 6 | 测试稳定化 (pandas 3.0) | ✅ | 5163 tests pass |
| 7 | CICC 对齐 + Validation 修复 | ✅ | validation 1/4→需 4/4 |
| 8 | 17 指标 + 4 维贡献分析 | ✅ | Calmar 0.78 |
| 9-A | 52周新高信号融合 | ✅ pass | Calmar 0.78 (持平) |
| 9-B | 趋势过滤器 (TF bear=0.7) | ✅ pass | Calmar 0.88 |
| **9-C** | **波动率目标 (VT tv=0.15)** | ✅ **强烈推荐** | **Calmar 1.00** |
| 9-D | HMM Regime 检测器 | ❌ **放弃** | Calmar 0.52 (过拟合) |
| 10 | 集中度约束 | ❌ **放弃** | Calmar 0.61 (限制过严) |
| 13 | 交易成本建模 | ✅ pass | Calmar 0.98 |

### 2.2 测试统计

```
总测试数: 226 个 (test_*.py)
├── 单元测试: 150+
├── 集成测试: 50+
├── OOS 测试: 9 个
└── Validation 测试: 13 个

总代码: ~3000+ 行 (strategy/momentum_etf_rotation/)
├── portfolio.py: 700+ 行 (核心组合管理)
├── backtest.py: 200+ 行 (回测循环)
├── contribution.py: 340+ 行 (4 维贡献分析)
├── brinson.py: 165+ 行 (Brinson 归因)
├── regime_detector.py: 150+ 行 (HMM)
├── extended_metrics.py: 175 行 (17 指标)
└── momentum.py: 130+ 行 (动量信号)

文档: 10+ 报告 + 13+ HTML 图表
git commits: 10+ (每 Stage 一提交)
```

---

## 3. 关键发现总结

### 3.1 策略特征

1. **动量策略在此数据上极强** - 商品类(黄金+白银)贡献 77% 收益
2. **集中度风险高** - 黄金 30% 收益贡献, 17% 风险贡献
3. **A 股宽基被自动低估** - 动量排序靠后, 触发策略向商品/海外倾斜
4. **熊市唯一亏损时段 2022** - 单 -18.7%, 触发后通过趋势过滤器改善

### 3.2 vs CICC 报告

| 指标 | 本实现 | CICC | 差异 | 可消除? |
|------|--------|------|------|---------|
| Calmar | 0.98 | 0.76 | +29% | 部分 (数据源/池子) |
| DD | -6.94% | -18.78% | 优于 | 部分 (DD 已改善) |
| 池子 | 44 只公开 ETF | 推测内部精选 | 不可控 | ❌ |
| 数据 | Tencent 行情 | Wind/Choice | 不可控 | ❌ |

### 3.3 已知技术债

| 债项 | 影响 | 优先级 |
|------|------|--------|
| **逆波动忽略相关性** | 集中度风险 | ★★★ |
| HMM 协方差在小样本下过拟合 | Stage 9-D 失败根因 | ★★ |
| 21 日窗口样本协方差不可用 | p>>n 问题 | ★★ |
| 数据源单一 (Tencent) | 与 CICC 数据源差异 | ★ |

---

## 4. 关键 Bug 与教训

### 4.1 Bug 清单（从开发过程总结）

| Bug | 阶段 | 类型 | 修复方式 |
|-----|------|------|---------|
| `resample("ME")` 标签错位 | CICC 对齐 | pandas API 误解 | 改用 groupby period |
| `fill_by_rank` 未检查 caps | CICC 对齐 | 集成遗漏 | 加 caps 检查 |
| `apply_stops` 缺 base_categories | CICC 对齐 | 设计遗漏 | 添加参数 |
| `TrendFilter` 字段顺序 | 9-B | 位置参数陷阱 | 强制 kwarg |
| `apply_vol_targeting` 后归一化 | 9-C | 逻辑冲突 | 删除归一化 |
| `apply_stops` 覆盖 caps | 10 | 集成遗漏 | 二次应用 caps |
| `nav[i] = nav[i] * 1.0` 未初始化 | 13 | copy-paste 失误 | 恢复 `nav[i-1]` |

**平均每功能引入 1.5 个 bug**

### 4.2 核心教训

1. **集成测试 > 单元测试** - 集成顺序问题最致命
2. **关键字参数 > 位置参数** - dataclass 字段顺序是陷阱
3. **OOS 验证救命** - HMM 表面 OK 但 OOS 不稳定
4. **诚实记录失败** - 失败是最有价值的反馈

---

## 5. 文档资产清单

```
reports/momentum_etf_rotation/
├── README.md                              # 总览 (待创建)
├── GAP_ANALYSIS.md                        # 数据/池子差距
├── validation_fix_report.md               # Stage 7 验证修复
├── extended_metrics.md                    # 17 指标定义
├── contribution_analysis.md              # 4 维贡献分析
├── industry_vs_commodity.md              # 维度对比
├── covariance_research.md                # 协方差调研
├── stage9a_report.md                      # 信号融合
├── stage9b_report.md                      # 趋势过滤器
├── stage9c_report.md                      # 波动率目标
├── stage9d_report.md                      # HMM (失败)
├── stage10_report.md                      # 集中度约束 (失败)
├── stage13_report.md                      # 交易成本
├── stage9_extended_comparison.md          # Stage 9 横向对比
├── stage9_metrics_table.md                # 17×9 表格
├── experiments/                           # 待创建: 归档失败实验
├── charts/                                # 13+ HTML
│   ├── rotation_2019_lb90.html
│   ├── stage9a_*.html
│   ├── stage9b_*.html
│   ├── stage9c_*.html
│   ├── stage9d_*.html
│   ├── stage10_*.html
│   └── stage13_*.html
├── *.csv                                  # 贡献数据
└── *.json                                 # 指标数据
```

---

## 6. 当前待办

### 6.1 高优先级

1. **协方差优化** - Ledoit-Wolf 收缩协方差 (调研已就绪)
2. **风险平价求解** - 替换 inverse_vol_weights
3. **HMM 重做** (用收缩协方差 + 更长训练窗口)

### 6.2 流程改进

1. 建立 7 阶段研发流程 (本文档)
2. 创建 README 总览
3. 创建 decision_log.md
4. 归档失败的 Stage 9-D 和 Stage 10
5. 建立 code review checklist

---

## 7. 一句话总结

**当前最优策略**: 沪深 44 ETF 月度动量轮动 + 波动率目标 + 交易成本, **Calmar 0.98 显著超 CICC 报告 0.76**, DD -6.94% 远优 -18.78%。

**技术债**: 协方差估计与优化是下一步关键突破点。

**研发流程**: 7 阶段规范化正在建立中。
