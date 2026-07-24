# 74 — v10 研发计划: 三个独立策略

> **目标: 基于现有数据构建 3 个独立策略, 单独验证后集成到 Vol-parity 组合**
> **Sharpe 目标: 1.70 (组合)**
> **时间: 4 周**

---

## 1. 策略总览

| # | 策略 | 核心逻辑 | 预期 Sharpe | 复杂度 | 优先级 |
|---|------|---------|-----------|--------|--------|
| 1 | **Dual Momentum** | 绝对动量+相对动量, 5大类ETF轮动 | 0.8-1.0 | 低 | ⭐ 最先 |
| 2 | **EPO Optimization** | slope×R²动量 + shrinkage cov优化 | 1.5-1.8 | 中 | 第二 |
| 3 | **RRG Rotation** | RS-Ratio×RS-Momentum 四象限行业轮动 | 0.8-1.0 | 中高 | 第三 |

---

## 2. Strategy 1: Dual Momentum

### 2.1 来源
- Gary Antonacci, *Dual Momentum Investing* (2014)
- GEM (Global Equity Momentum) 模型

### 2.2 资产池

| 类别 | ETF | 代码 | 用途 |
|------|-----|------|------|
| A股 | 沪深300ETF | 510300 | 权益 |
| 港股 | 恒生科技ETF | 159740 | 权益 |
| 美股 | 纳指ETF | 513100 | 权益 |
| 商品 | 黄金ETF | 518880 | 避险/通胀 |
| 债券 | 10Y国债ETF | 511260 | 防御/无风险 |

### 2.3 信号规则
```
1. 计算每个风险资产过去 12 个月收益率
2. 绝对动量过滤: 收益率 > 0? → 保留, 否则剔除
3. 相对动量排序: 在通过筛选的资产中选收益最高者
4. 全部未通过 → 持有 511260 (国债)
5. 月末调仓
```

### 2.4 实现文件
- `strategy/momentum_etf_rotation/v10/dual_momentum.py`
- 输出: `reports/momentum_etf_rotation/v10/dual_momentum_nav.parquet`

### 2.5 验证标准
| 检查项 | 标准 |
|--------|------|
| OOS Sharpe | ≥ 0.80 |
| OOS MaxDD | ≤ -15% |
| 与 v7.10 相关性 | < 0.50 |
| 与 v1.0 相关性 | < 0.30 |

---

## 3. Strategy 2: EPO Optimization

### 3.1 来源
- 聚宽社区 openhe 策略 + 东方财富财富号实盘优化 (2026)
- EPO (Enhanced Portfolio Optimization)

### 3.2 资产池
- 43 ETF (与 v7.10 相同)
- 数据源: `v7_6_Y_weekly.parquet` (43 ETF 周频收益率)

### 3.3 信号规则
```
1. 动量打分:
   for each ETF:
     x = log(close[-34:])
     slope, R² = linregress(range(34), x)
     score = annualized_return × R²

2. 权重优化:
   a. 收缩相关矩阵: shrunk_corr = (1-w)×corr + w×I, w=0.2
   b. 重建协方差矩阵 Σ
   c. 锚定逆方差权重: w_iv = 1/diag(Σ)
   d. EPO权重: w_epo = argmin(w'Σw) s.t. w'×signal ≥ threshold
   e. 负权重置零, 归一化

3. 混合: 60% EPO + 40% Risk Parity
```

### 3.4 实现文件
- `strategy/momentum_etf_rotation/v10/epo_momentum.py`
- 输出: `reports/momentum_etf_rotation/v10/epo_momentum_nav.parquet`

### 3.5 验证标准
| 检查项 | 标准 |
|--------|------|
| OOS Sharpe | ≥ 1.20 |
| OOS MaxDD | ≤ -20% |
| 与 v7.10 相关性 | < 0.70 |
| 全样本 vs OOS Sharpe 差 | < 30% |

---

## 4. Strategy 3: RRG Rotation

### 4.1 来源
- 西部证券《RRG框架下行业与ETF轮动策略构建》(2026)
- RRG (Relative Rotation Graph)

### 4.2 资产池
- 20 行业 ETF (已有价格数据)

| 行业 | ETF | 代码 |
|------|-----|------|
| 半导体 | 半导体ETF | 512760 |
| 新能源车 | 新能车ETF | 515030 |
| 光伏 | 光伏ETF | 515790 |
| 酒 | 酒ETF | 512690 |
| 医药 | 医疗ETF | 512170 |
| 5G通信 | 5G通信ETF | 515050 |
| 消费 | 主要消费ETF | 159928 |
| 证券 | 证券ETF | 512880 |
| 券商 | 券商ETF | 512000 |
| 银行 | 银行ETF | 512800 |
| 煤炭 | 煤炭ETF | 515220 |
| 地产 | 地产ETF | 512200 |
| 有色金属 | 有色ETF | 512400 |
| 军工 | 军工ETF | 512660 |
| 传媒 | 传媒ETF | 512980 |
| 通信 | 通信ETF | 515880 |
| 家电 | 家电ETF | 159996 |
| 化工 | 化工ETF | 512120 |

### 4.3 信号规则
```
1. RS-Ratio (220日相对强度比):
   RS = ETF_close / benchmark_close (沪深300)
   RS_ratio = RS / MA(RS, 220) × 100

2. RS-Momentum (60日动量):
   RS_mom = MA(RS, 60) / MA(RS, 220) × 100

3. 四象限分类:
   领先 (RS_ratio > 100, RS_mom > 100) → 买入
   改善 (RS_ratio < 100, RS_mom > 100) → 观望
   滞后 (RS_ratio < 100, RS_mom < 100) → 卖出
   疲软 (RS_ratio > 100, RS_mom < 100) → 卖出

4. 扩散指标 (简化):
   涨幅220日排名前60% → 行业内广度

5. 综合: 领先+扩散高 → 最高权重
```

### 4.4 实现文件
- `strategy/momentum_etf_rotation/v10/rrg_rotation.py`
- 输出: `reports/momentum_etf_rotation/v10/rrg_rotation_nav.parquet`

### 4.5 验证标准
| 检查项 | 标准 |
|--------|------|
| OOS Sharpe | ≥ 0.70 |
| OOS MaxDD | ≤ -30% |
| 与 v7.10 相关性 | < 0.60 |
| 年度胜率 | ≥ 60% |

---

## 5. 集成到 Vol-parity

### 5.1 相关性矩阵检查

```
                v1.0   v7.10  v9m    DM     EPO    RRG
v1.0             1.00
v7.10            0.85   1.00
v9macro          0.82   0.94   1.00
DualMom          ?      ?      ?      1.00
EPO              ?      ?      ?      ?      1.00
RRG              ?      ?      ?      ?      ?      1.00
```

### 5.2 加入条件
- 与 v7.10 相关性 < 0.70 → 加入
- 与 v7.10 相关性 ≥ 0.70 → 不加入 (信息冗余)

### 5.3 Vol-parity 权重重算
```python
# 新组合候选
navs = {
    'v1.0': v1_nav,
    'v7.10': v7_nav,
    'v9macro': v9_nav,
    'DualMom': dm_nav,      # 如果相关性合格
    'EPO': epo_nav,          # 如果相关性合格
    'RRG': rrg_nav,          # 如果相关性合格
}
weights = vol_parity_weights(navs, target_vol=0.08)
```

### 5.4 预期提升

| 组合 | Sharpe | AnnRet | MaxDD |
|------|--------|--------|-------|
| 当前 Vol-parity (3策略) | 1.535 | 9.72% | -4.72% |
| + Dual Momentum | 1.55-1.60 | 9-10% | -4.5% |
| + EPO | 1.60-1.70 | 10-12% | -5.0% |
| + RRG | 1.55-1.65 | 10-11% | -5.5% |
| 全部 (如果相关性合格) | 1.65-1.75 | 10-12% | -5.0% |

---

## 6. 时间计划

| 周 | 任务 | 产出 |
|---|------|------|
| Week 1 | Strategy 1: Dual Momentum | `dual_momentum.py` + NAV |
| Week 2 | Strategy 2: EPO Optimization | `epo_momentum.py` + NAV |
| Week 3 | Strategy 3: RRG Rotation | `rrg_rotation.py` + NAV |
| Week 4 | 集成 + 文档 + INDEX 更新 | `combine_f_extended.py` + docs/75 |

---

## 7. 数据依赖

| 数据 | 文件 | 状态 |
|------|------|------|
| 5 大类 ETF close | `data/real/per_etf/{510300,159740,513100,518880,511260}.parquet` | ✅ |
| 43 ETF 周收益 | `data/high_freq_macro/v7_6_Y_weekly.parquet` | ✅ |
| 20 行业 ETF close | `data/real/per_etf/{512760,...}.parquet` (20个) | ✅ |
| 17 宏观因子 | `data/high_freq_macro/v7_6_X_macro_weekly.parquet` | ✅ |
| 沪深300基准 | `data/high_freq_macro/v9_benchmark_沪深300.parquet` | ✅ |

**所有数据已就绪, 无需额外 iFinD 下载.**

---

## 8. 文件结构

```
strategy/momentum_etf_rotation/v10/
├── dual_momentum.py      # Strategy 1
├── epo_momentum.py       # Strategy 2
├── rrg_rotation.py       # Strategy 3
└── __init__.py

reports/momentum_etf_rotation/v10/
├── dual_momentum_nav.parquet
├── epo_momentum_nav.parquet
├── rrg_rotation_nav.parquet
└── v10_strategy_comparison.csv

scripts/combo/
└── combine_f_extended.py  # 集成到 Vol-parity

docs/
├── 74-v10_research_plan.md     # 本文档
└── 75-v10_results.md           # 实验结果
```
