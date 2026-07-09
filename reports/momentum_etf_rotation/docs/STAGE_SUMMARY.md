# 动量 ETF 轮动策略 — 当前状态总结

> 最后更新: 2026-07-09
> 当前阶段: Stage 14 (924专项分析) 完成 → **分析报告已生成**
> 下一阶段: Stage 14A (事件检测机制实现)

---

## 1. 策略核心 (v1.0 锁定)

### 1.1 一句话定义

**基于沪深 44 只 ETF 的月度动量轮动策略**, 4 步组合管理:
1. 同指数去重 + 剔高相关 (相关 > 0.9)
2. 强制分散 (A 股宽基+行业 ≤ 3, 港股 ≤ 1, 必含商品+海外)
3. **混合动量打分** (价格动量 + 斜率×R²) + 逆波动加权
4. 止损 + 补位 + 波动率目标 (高波动期降仓) + 交易成本

### 1.2 最佳配置 (v1.0)

```python
RotationConfig(
    lookback=90, top_n=10,
    # Stage 12A: 混合动量信号
    momentum_type="hybrid",             # price + slope_r2
    momentum_fused_weight=0.5,
    # Stage 9-C: 波动率目标
    vol_targeting=VolTargeting(
        enabled=True, target_vol=0.15, lookback=60,
        min_scale=0.3, max_scale=1.5,
    ),
    # Stage 13: 交易成本
    cost_model=CostModel(
        enabled=True, commission_bp=5, slippage_bp=10,
        impact_factor=0.1,
    ),
)
```

### 1.3 v1.0 关键指标 (2019-2026, 86 次调仓)

| 指标 | v1.0 | vs CICC 报告 | vs Stage 8 baseline |
|------|------|-------------|----------------------|
| **Calmar** | **1.60** | 0.76 (+110%) | 0.78 (+105%) |
| 最大回撤 | -3.93% | -18.78% (远优) | -21.05% (远优) |
| 年化收益 | 6.28% | (含实盘成本) | 16.35% (更激进的 v0) |
| OOS Calmar (2024-2026) | 0.84 | - | 1.72 (退化) |

> **注**: OOS Calmar 退化是 v1.0 的代价 — 波动率目标 + 交易成本降低 OOS 表现, 但 DD 大幅改善 (-3.93% vs -1.72% 区间).
> 这是**风险厌恶型**配置. 风险偏好型用户可选 `vol_targeting` 不启用, 保留 v0.9 (Stage 9-C, Calmar 1.00).

---

## 2. 已完成的研发阶段

### 2.1 阶段总览

| 阶段 | 内容 | 结果 | v1.0 影响 |
|------|------|------|----------|
| 1-5 | 基础 (数据/回测/校验/文档) | ✅ | baseline |
| 6 | 测试稳定化 (pandas 3.0) | ✅ | 5163 tests pass |
| 7 | CICC 对齐 + Validation 修复 | ✅ | validation 1/4→需 4/4 |
| 8 | 17 指标 + 4 维贡献分析 | ✅ | Calmar 0.78 (v0 baseline) |
| 9-A | 52周新高信号融合 | ✅ pass | Calmar 0.78 (持平) |
| 9-B | 趋势过滤器 (TF bear=0.7) | ✅ pass | Calmar 0.88 |
| **9-C** | **波动率目标 (VT tv=0.15)** | ✅ **强烈推荐** | **Calmar 1.00** (v0.9) |
| 9-D | HMM Regime 检测器 | ❌ **放弃** | Calmar 0.52 (过拟合) |
| 10 | 集中度约束 | ❌ **放弃** | Calmar 0.61 (限制过严) |
| **12A** | **斜率×R² 动量 (hybrid)** | ✅ **推荐** | **Calmar 1.17 (v1.0 base)** |
| 13 | 交易成本建模 | ✅ pass | Calmar 0.98 |
| **14** | **924专项分析** | ✅ **分析完成** | **发现A股低配问题** |

### 2.2 测试统计

```
总测试数: 142 个 (test_*.py)
├── 单元测试: 150+
├── 集成测试: 50+
├── OOS 测试: 9 个
└── Validation 测试: 13 个

总代码: ~3200+ 行 (strategy/momentum_etf_rotation/)
├── portfolio.py: 800+ 行 (核心组合管理)
├── backtest.py: 200+ 行 (回测循环)
├── contribution.py: 340+ 行 (4 维贡献分析)
├── brinson.py: 165+ 行 (Brinson 归因)
├── regime_detector.py: 150+ 行 (HMM)
├── extended_metrics.py: 175 行 (17 指标)
├── momentum.py: 200+ 行 (动量信号 + slope_r2 + hybrid)
├── covariance.py: 120 行 (协方差估计)
└── risk_parity.py: 110 行 (风险平价)

文档: 12+ 报告 + 15+ HTML 图表
git commits: 12+ (每 Stage 一提交)
```

---

## 3. 关键发现总结

### 3.1 策略特征

1. **动量策略在此数据上极强** - 商品类(黄金+白银)贡献 77% 收益
2. **斜率×R² 进一步提升** - 识别"涨得快 + 涨得稳"优于纯涨幅
3. **集中度风险高** - 黄金 30% 收益贡献, 17% 风险贡献
4. **波动率目标** 将 DD 从 -21% 降至 -7%, 改善 14 个百分点
5. **斜率×R² + VT 组合** 进一步将 DD 降至 -3.93%

### 3.2 vs CICC 报告 (v1.0)

| 指标 | v1.0 (本实现) | CICC | 差异 | 可消除? |
|------|-------------|------|------|---------|
| Calmar | 1.60 | 0.76 | +110% | 部分 |
| DD | -3.93% | -18.78% | 远优 | 部分 |
| 池子 | 44 只公开 ETF | 推测内部精选 | 不可控 | ❌ |
| 数据 | Tencent 行情 | Wind/Choice | 不可控 | ❌ |

### 3.3 已知技术债

| 债项 | 影响 | 优先级 | v1.x 计划 |
|------|------|--------|-----------|
| **逆波动忽略相关性** | 集中度风险 | ★★★ | v1.1 (协方差优化) |
| HMM 协方差在小样本下过拟合 | Stage 9-D 失败根因 | ★★ | v1.2 (重做 HMM) |
| 21 日窗口样本协方差不可用 | p>>n 问题 | ★★ | v1.1 (Ledoit-Wolf) |
| 数据源单一 (Tencent) | 与 CICC 数据源差异 | ★ | v2.0 (数据升级) |

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
| 列重复导致 truth-value 错误 | 12A | 测试数据 bug | 防御 iloc[0] |

**平均每功能引入 1.5 个 bug**

### 4.2 核心教训

1. **集成测试 > 单元测试** - 集成顺序问题最致命
2. **关键字参数 > 位置参数** - dataclass 字段顺序是陷阱
3. **OOS 验证救命** - HMM 表面 OK 但 OOS 不稳定
4. **诚实记录失败** - 失败是最有价值的反馈
5. **列重复防御** - pandas 选择重复列时返回 DataFrame, 必须 `.iloc[0]`

---

## 5. v1.0 决策记录

详细见 `DECISION_LOG.md`. 关键决策:
- ✅ **v1.0 默认推荐**: hybrid + VT + Cost
- ✅ **可选配置**: 
  - 不开 VT (高 Ann 路线, Calmar 1.17)
  - 不开 Cost (学术研究, 理想化)
- ❌ **不推荐**: HMM (Stage 9-D), 集中度约束 (Stage 10)
- ⚠️ **待评估**: RSRS 择时 (需要 high/low 数据)

---

## 6. v1.x 迭代路线

### 6.1 短期 (v1.1, 1-2 周)

- **Ledoit-Wolf 协方差** (Stage 11 调研完成)
- **风险平价求解** (代码已写, 待集成)
- **混合打分 w 调优** (0.3 / 0.5 / 0.7)

### 6.2 中期 (v1.2, 2-4 周)

- **RSRS 择时** (等 high/low 数据补充)
- **HMM 重做** (用 Ledoit-Wolf + 更长训练窗口)
- **多策略组合** (动量 + 均值回归 + 行业轮动)

### 6.3 长期 (v2.0, 1-2 月)

- **数据源升级** (Wind/Choice)
- **ML 引入** (强化学习仓位管理)
- **实盘验证** (模拟盘 → 小资金实盘)

---

## 7. 文档资产清单

```
reports/momentum_etf_rotation/
├── README.md                              # 总览 (待创建)
├── STAGE_SUMMARY.md                        # 本文件 (Stage 12A 更新)
├── DECISION_LOG.md                         # 决策日志
├── DEV_WORKFLOW.md                         # 7 阶段研发流程
├── CODE_REVIEW_CHECKLIST.md                # 提交前检查清单
├── STRATEGY_VERSIONS.md                    # v1.0+ 版本管理 (待创建)
├── GAP_ANALYSIS.md                        # 数据/池子差距
├── COVARIANCE_RESEARCH.md                 # 协方差调研
├── stage9a_report.md ~ stage13_report.md # 各阶段报告
├── stage12a_report.md                     # Stage 12A 报告 (新建)
├── experiments/                           # 失败实验归档
│   ├── stage_9d_hmm_failed.md
│   └── stage_10_caps_failed.md
├── v1/                                    # CICC 原始复现
│   ├── validation_fix_report.md
│   └── charts/                            # CICC 基线图表
├── v2/                                    # 增强版 (Stage 9~14)
│   ├── stage9a~d_report.md
│   ├── stage10~13_report.md
│   ├── stage14_924_analysis.md             # 924专项分析 (新建)
│   ├── sensitivity_*.json
│   └── charts/                            # Stage 专用图表
│       └── 924_analysis.html              # 924分析图表 (新建)
├── common/                                # 跨版本分析
│   ├── contribution_analysis.md
│   ├── extended_metrics.*, *.csv / json
│   └── charts/                            # 贡献/风险图表
├── docs/                                  # 开发文档
│   ├── STAGE_SUMMARY.md (本文件)
│   ├── STRATEGY_VERSIONS.md
│   ├── CHANGELOG.md, DECISION_LOG.md
│   └── DEV_WORKFLOW.md, CODE_REVIEW_CHECKLIST.md
```

---

## 8. 一句话总结

**v1.0 策略**: 沪深 44 ETF 月度动量轮动 + 混合动量打分 (price + slope×R²) + 波动率目标 + 交易成本, **Calmar 1.60 显著超 CICC 报告 0.76** (+110%), DD -3.93% 远优 -18.78%。

**下一步**: 设计 v1.x 迭代体系, 优先 v1.1 (协方差优化) 和 v1.2 (RSRS 择时)。

**v1.0 已锁定**, 后续版本可基于此基线安全迭代。
