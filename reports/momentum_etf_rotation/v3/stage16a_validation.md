# Stage 16A 验证: 多策略组合 (v3) vs 单策略 (v2)

> **验证日期**: 2026-07-09
> **数据**: 2018-01-02 ~ 2026-06-30 (2058 天 × 44 ETF)
> **状态**: ✅ 实现完成, 验证中
> **结论**: v3 多策略在 924 期间改善 +0.7%, 但全周期 Calmar 退化 (待优化)

---

## 1. 测试矩阵

| 配置 | 描述 | 子策略 |
|------|------|--------|
| v2 | Stage 12A baseline | 单策略 (动量) |
| v3_equal | v3 多策略 + 等权 | 动量 + 反转 + 行业轮动 (1/3 each) |
| v3_signal | v3 多策略 + 信号加权 | 同上, 按 signal_strength 动态加权 |

---

## 2. 全周期业绩 (2018-01-02 ~ 2026-06-30)

| 指标 | v2 | v3_equal | v3_signal |
|------|------|----------|-----------|
| 年化收益 | **10.88%** | 7.04% | 5.70% |
| 年化波动 | 9.53% | 7.76% | 6.48% |
| Sharpe | **1.14** | 0.91 | 0.88 |
| 最大回撤 | -12.20% | -13.97% | -13.00% |
| Calmar | **0.892** | 0.504 | 0.439 |

**关键观察**:
- v2 全周期 Calmar 0.892, 是 v3 的 1.7-2 倍
- v3 波动率更低 (7.76% vs 9.53%), 体现多策略分散
- 但 v3 收益也低 (7.04% vs 10.88%), 净效应负

**原因分析**:
1. v3 用 1/3 配重给反转+行业轮动, 但它们收益贡献低
2. 反转策略在趋势市(2019-2020)反向, 拖累
3. 行业轮动 (A 股) 在 2018-2020 牛市权重过低
4. 子策略权重等分 (1/3 each) 浪费了动量策略的优势

---

## 3. 924 专项 (2024-09-23 ~ 2024-10-31)

| 指标 | v2 | v3_equal | v3_signal |
|------|------|----------|-----------|
| 期间收益 | 3.16% | **3.89%** | 3.35% |
| 峰值日 | 2024-10-10 | 2024-10-08 | **2024-09-26** |
| 峰值日收益 | **+1.83%** | +1.06% | +0.87% |

**关键观察**:
- v3_equal 改善 924 期间收益 **+0.73%** (vs v2)
- v3_signal 峰值日提前到 9/26 (更早捕捉到政策红利)
- 但 v3 在峰值日的单日收益反而低 (反转策略反向)

**结论**: v3 部分改善 924 失分, 但远未达到 +5~10% 预期目标

---

## 4. 子策略 NAV 分析 (v3_equal)

| 子策略 | 最终 NAV | 贡献 |
|--------|----------|------|
| combined | 1.50 | - |
| industry_rotation | 1.04 | 低 (A 股熊市损失) |
| reversion | (未跟踪) | - |
| momentum | (未跟踪) | 主导 |

**问题**: 跟踪不完整, 需在 multi_strategy_v3.py 修复

---

## 5. 已知问题与改进

### 5.1 当前 v3 的问题

1. **子策略权重等分** (1/3 each) 浪费动量优势
2. **反转策略与动量方向冲突** (趋势市反向)
3. **行业轮动 (A 股) 在熊市拖累**
4. **NAV 跟踪不完整** (reversion/momentum 缺失)
5. **未应用 max_weight cap** (子策略内部允许 0.15)

### 5.2 改进方向

1. **动态子策略权重**: 用历史 Sharpe 决定权重 (动量应得更多)
2. **关闭反转在趋势市**: 用 200 日均线判断市场状态
3. **限制行业轮动在熊市权重**: 0.1 上限
4. **修复 NAV 跟踪**
5. **应用主回测 max_weight cap**

### 5.3 短期优化 (1-2 天)

- 修改 `MultiStrategyConfig.weight_method="signal"` 用历史 Sharpe
- 修复 `sub_navs` 跟踪 (确保所有子策略都有 NAV)
- 重新跑验证

---

## 6. 测试结果

| 测试 | 数量 | 通过 | 失败 |
|------|------|------|------|
| SubStrategy 抽象 | 1 | 1 | 0 |
| Reversion | 12 | 12 | 0 |
| IndustryRotation | 12 | 12 | 0 |
| SubWeighting | 14 | 14 | 0 |
| MultiStrategy | 12 | 12 | 0 |
| **小计** | **51** | **51** | **0** |
| 全量回归 | 181 | 181 | 0 |

---

## 7. 文件清单

### v3 模块
- `QuantNodes/strategy/momentum_etf_rotation/v3/__init__.py`
- `QuantNodes/strategy/momentum_etf_rotation/v3/sub_strategy_v3.py` (抽象基类)
- `QuantNodes/strategy/momentum_etf_rotation/v3/reversion_v3.py` (反转)
- `QuantNodes/strategy/momentum_etf_rotation/v3/industry_rotation_v3.py` (行业轮动)
- `QuantNodes/strategy/momentum_etf_rotation/v3/sub_weighting_v3.py` (权重)
- `QuantNodes/strategy/momentum_etf_rotation/v3/multi_strategy_v3.py` (主回测)

### 测试
- `tests/strategy/momentum_etf_rotation/test_reversion_v3.py` (12 测试)
- `tests/strategy/momentum_etf_rotation/test_industry_rotation_v3.py` (12 测试)
- `tests/strategy/momentum_etf_rotation/test_sub_weighting_v3.py` (14 测试)
- `tests/strategy/momentum_etf_rotation/test_multi_strategy_v3.py` (12 测试)

### 数据
- `reports/momentum_etf_rotation/v3/stage16a_navs.parquet`
- `reports/momentum_etf_rotation/v3/stage16a_summary.json`

---

## 8. 结论

✅ **架构目标达成**: 多策略框架完整实现, 单元测试 100% 通过

⚠️ **业绩目标未达成**: v3 多策略在 924 期间改善 +0.7% (vs 预期 +5~10%), 全周期 Calmar 退化 0.39 (vs 预期 +0.10+)

📋 **下一步**:
1. 短期: 修复子策略权重 (动态 Sharpe 加权)
2. 短期: 修复 NAV 跟踪
3. 中期: 实施趋势市自适应 (反转策略关闭)
4. 长期: 继续 Stage 16B (RSRS) + 16C (RL)

---

## 9. 提交记录

- `5670b68` refactor(v3): 创建 v3/ 目录, 冻结 v2
- `f9e5563` feat(v3): Stage 16A Step 2 - 反转子策略
- `aaf8aae` feat(v3): Stage 16A Step 3 - 行业轮动子策略
- `5803e31` feat(v3): Stage 16A Step 4 - 子策略权重
- `e12070c` feat(v3): Stage 16A Step 5 - 多策略主回测
