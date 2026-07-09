# Stage 17 v4 完整验证报告 (风格轮动 + Smart β + 因子择时)

> **日期**: 2026-07-09
> **数据**: 2018-01-02 ~ 2026-06-30, 12 只 Smart β ETF
> **状态**: ✅ 全 6 模式可跑; 因子择时效果被 cap 吸收, 待整合时改进
> **关键结论**: v4 在 924/2025H2/2026H1 表现远好, 但全周期 DD 太大; v3 抗回撤更强

---

## 1. 6 模式对比 (全周期 2018-2026)

| 模式 | Sharpe | Calmar | AnnRet | DD | Final Nav |
|------|--------|--------|--------|----|-----------|
| **v3_baseline** | **0.916** | **0.504** | 7.04% | **-13.97%** | 1.743 |
| v4A_style | 0.311 | 0.092 | 4.54% | -49.31% | 1.437 |
| v4B_smartbeta | 0.387 | 0.140 | 5.69% | -40.56% | 1.572 |
| v4C_combo | 0.311 | 0.097 | 4.43% | -45.70% | 1.424 |
| v4D_ic | 0.311 | 0.097 | 4.43% | -45.70% | 1.424 |
| v4E_hmm | 0.311 | 0.097 | 4.43% | -45.70% | 1.424 |
| v4F_fusion | 0.311 | 0.097 | 4.43% | -45.70% | 1.424 |

**核心观察**:
- v3 baseline 全周期最稳健 (Calmar 0.50, DD -14%)
- v4 模式 DD 都在 -40%~-49%, 子策略过于集中
- v4D/E/F ≈ v4C — 因子择时效果被 max_weight=0.20 吸收

## 2. 关键区间表现

### 2.1 924 反弹 (2024-09-23 ~ 2024-10-31)
| 模式 | 收益 | 排名 |
|------|------|------|
| v3_baseline | +3.89% | 7 |
| v4A_style | +13.81% | 4 |
| v4B_smartbeta | **+20.63%** | 🥇 |
| v4C_combo | +16.58% | 2-6 (tie) |
| v4D/E/F | +16.58% | 2-6 (tie) |

**v4 抓反弹能力 4-5x 强于 v3** (Smart β 在政策红利期反应最快)

### 2.2 2025 H2 牛市 (2025-07-01 ~ 2025-12-31)
| 模式 | 收益 | 排名 |
|------|------|------|
| v3_baseline | +17.53% | 4 |
| v4A_style | **+27.81%** | 🥇 |
| v4B_smartbeta | +7.29% | 6 |
| v4C_combo | +19.25% | 3 |
| v4D/E/F | +19.25% | 2-3 (tie) |

**v4A 风格轮动 牛市最强, v4B 反向 (Smart β 不在牛市) 较弱**

### 2.3 2026 H1 (2026-01-01 ~ 2026-06-30)
| 模式 | 收益 | 排名 |
|------|------|------|
| v3_baseline | -1.32% | 5 |
| v4A_style | **+18.66%** | 🥇 |
| v4B_smartbeta | -0.51% | 4 |
| v4C_combo | +10.72% | 2-3 (tie) |
| v4D/E/F | +10.72% | 2-3 (tie) |

**v4A 在 2026 H1 远超 v3 (+18.66% vs -1.32%), 抓反弹能力突出**

## 3. IC 因子表现

| 因子 | Mean IC | ICIR | Hit Rate |
|------|---------|------|----------|
| momentum | -0.0145 | -0.064 | 47.2% |
| reversal | -0.0568 | -0.441 | 32.0% |
| **value** | **+0.0437** | **+0.171** | **59.6%** ⭐ |
| low_vol | -0.0146 | -0.065 | 51.5% |
| dividend | -0.0348 | -0.271 | 32.4% |
| quality | -0.0116 | -0.077 | 46.6% |

**唯一显著正 IC: value (+0.044, Hit 60%)**
**IC_short_fwd (10d) 最佳: Calmar 0.48 vs 等权 0.46 (+4%)**

## 4. HMM Regime 检测 (距离先验)

### 4.1 距离先验矩阵 (alpha=1.5, gamma=0.3)
```
                to
from     bear   transition   bull
bear     0.83   0.15         0.03
trans    0.19   0.69         0.12
bull     0.06   0.21         0.74
```

**特性**: bull ↔ bear 直接跳转仅 ~3% (符合金融常识)

### 4.2 实际 Regime 分布
- bull: 23 samples
- bear: 369 samples (主导, A 股长期震荡/下跌)
- transition: 8 samples

**HMM 在 A 股 2018-2026 数据上, 主要识别为"熊"状态**, 与 A 股 7 跌 3 涨长期表现一致.

## 5. 核心问题诊断

### 5.1 v4 DD 过大原因
- max_weight=0.20 cap 后, 每个子策略 5-7 只 ETF 都接近 20%
- 实际持仓 5-7 只, 集中度过高
- 单一 ETF 暴跌 (-20% 单日) 直接拖累组合 -4% ~ -5%
- 修复方向: 增 top_n + 降低 max_weight

### 5.2 因子择时效果被 cap 吸收
- v4D 的 sub_weights 实际计算为 0.33/0.67 (vs v4C 的 0.5/0.5)
- 但 5 个 ETF × 20% cap 强制 equal weight
- 修复方向: 因子择时调整个股权重而非子策略权重, 或降 cap

### 5.3 v3 全面胜出原因
- 44 ETF universe, 充分分散
- 6 类别 cap 强制行业/类别分散
- 14 只持仓 (top_n=10 + 增项), 降低单 ETF 风险

## 6. 整合建议 (v3 + v4 互补)

| 维度 | v3 (现有) | v4 (新) | 整合方向 |
|------|----------|---------|----------|
| Universe | 44 ETF | 12 Smart β | **保留两者**, 风格/Smart β 用 v4, 其他用 v3 |
| DD 控制 | 类别 cap | 集中 (待修) | 借鉴 v3 cap 到 v4 |
| 抓反弹 | 弱 | 强 (924 +20%) | 整合 v4 子策略 |
| 风格轮动 | 无 | v4A (+27% 牛市) | 加入 v3 |
| 因子择时 | 无 | 弱 (cap 吸收) | 重新设计: 影响个股权重 |

**Stage 18 计划**:
1. v3 多策略中加入 v4 风格轮动作为新子策略
2. v3 universe 扩展到 56 ETF (44 + 12 Smart β)
3. 因子择时改为个股层面 (而非子策略层面)
4. cap 调整: 14 只 × 0.15 + 子策略 cap

## 7. 文件清单

### 7.1 v4 代码 (10 个文件)
- `v4/__init__.py` - 导出
- `v4/universe_v4.py` - 12 ETF 池 + 5 风格组 + 7 Smart β
- `v4/sub_strategy_v4.py` - 子策略基类
- `v4/style_rotation_v4.py` - 风格轮动子策略
- `v4/smart_beta_v4.py` - Smart β 工具子策略
- `v4/factor_ic.py` - 6 因子 IC 计算
- `v4/regime_detector_v4.py` - HMM (距离先验)
- `v4/regime_transitions.py` - 距离矩阵 (距离 → 转移概率)
- `v4/factor_timing_v4.py` - 因子择时 (IC + HMM 融合)
- `v4/multi_strategy_v4.py` - 6 模式回测主入口

### 7.2 验证脚本
- `scripts/validate_stage17_ic.py` - IC 验证
- `scripts/validate_stage17.py` - 6 模式对比

### 7.3 数据
- `data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet` (12 ETF)
- `data/real/per_etf_smartbeta/*.parquet` (12 个)

### 7.4 输出
- `reports/momentum_etf_rotation/v4/stage17_navs.parquet` (7 模式 NAV)
- `reports/momentum_etf_rotation/v4/stage17_summary.json` (指标汇总)
- `reports/momentum_etf_rotation/v4/IC_PERFORMANCE_REPORT.md` (IC 详细)
- `reports/momentum_etf_rotation/v4/hmm_regime_history.csv` (HMM 时序)

## 8. 后续工作

### 8.1 必须做
1. **写单元测试** (37 个) - 提高代码可信度
2. **写图表脚本** - 6 模式 NAV 对比图
3. **更新 index.html** - 链接 Stage 17

### 8.2 建议做
1. **v4 子策略调优**: top_n=5, max_weight=0.15
2. **因子择时改个股**: 而非子策略
3. **HMM 多窗口融合**: 5d/10d/20d 滑动

### 8.3 后续阶段
1. **Stage 18**: v3 + v4 整合 (扩展 universe, 加入风格轮动子策略)
2. **Stage 19**: 因子择时 v2 (个股层面)
3. **Stage 20**: 完整回测 + 实盘模拟
