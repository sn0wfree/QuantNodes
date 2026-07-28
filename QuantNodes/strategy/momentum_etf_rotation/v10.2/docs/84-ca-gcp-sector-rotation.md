# 84 · CA-GCP 板块轮动 (Sector CA-GCP, 场景 C)

## 摘要

板块 ETF 之间相关性弱（科技 vs 消费 corr ≈ 0.3），跨板块借用校准分数会引入噪声。本文实现**板块内独立 CA-GCP**：每个板块训练独立的 pipeline，只在板块内借数据。配合 v10 的板块轮动架构（如 5 宏观因子 → 板块 ETF），可避免"科技暴跌污染医药覆盖率"。

## 1. 问题定义

**设定**：38 ETF 池，分入 6-8 个板块（科技 / 医药 / 金融 / 消费 / 商品 / 债券 / 宽基 / 周期）。

**对比方法**：

| 方法 | 校准策略 | 跨板块借用？ |
|------|---------|------------|
| Global CA-GCP | 跨所有 ETF 借数据 | ✅ 全部 |
| **Sector CA-GCP**（本文）| 板块内独立借数据 | ❌ 仅板块内 |
| Sector + Correlation Hybrid | 板块内 + 板块间弱相关（corr < 0.5）借用 | ⚖️ 部分 |

**评估指标**：
- 板块内覆盖率（每个板块单独看 95% ± 5%）
- 板块间独立性（科技暴跌不影响医药覆盖率）
- 极端日覆盖率（取各板块的均值）

## 2. ETF → 板块映射

基于 ETF 代码命名规则**自动推断**（无需手动标注）：

| 代码前缀/关键字 | 推断板块 |
|----------------|---------|
| 510300, 510500, 510050, 159915, 510880 | 宽基 |
| 512760 (芯片), 515030 (AI), 515050, 159941 (纳指), 512480 | 科技/海外 |
| 512170 (医疗), 512010 (医药), 512290 | 医药 |
| 512880 (证券), 512000 (券商), 510200 | 金融 |
| 518880 (黄金), 518800 (豆粕), 159980 (有色), 162411 | 商品 |
| 511010 (短债), 511260, 511030 | 债券 |
| 512400, 159920, 512660, 159928 | 周期/资源 |
| 512690, 512170, 512980, 512800, 159928, 159740, 159766, 159786 | 其他板块 |

**注**：映射在 `data/etf_sector_map.csv`，后续可由用户 review 修正。

## 3. Sector CA-GCP 算法

```python
def fit_sector_ca_gcp(returns_train, sectors, config):
    pipelines = {}
    for sector, codes in sectors.items():
        sector_returns = returns_train[codes]
        pipe = CAGCPipeline(config)
        pipe.fit(sector_returns)
        pipelines[sector] = pipe
    return pipelines

def predict_sector_ca_gcp(pipelines, returns_calib, returns_test, sectors):
    out = {"lower": {}, "upper": {}, "half_width": {}, "stress": {}}
    for sector, pipe in pipelines.items():
        sector_calib = returns_calib[pipe.codes]
        sector_test = returns_test[pipe.codes]
        res = pipe.predict(sector_calib, sector_test)
        out["lower"][sector] = res["lower"]
        out["upper"][sector] = res["upper"]
        out["half_width"][sector] = res["half_width"]
        out["stress"][sector] = res["stress"]
    return out
```

**关键性质**：
- 每个板块独立的 KNN 图（避免跨板块污染）
- 每个板块独立的 σ 估计（板块波动率结构不同）
- 每个板块独立的 systemic stress（科技股灾 vs 债市波动完全不同）

## 4. 预期结果

| 板块 | Method | Cov (%) | PA-Std | Width (bps) |
|------|--------|---------|--------|-------------|
| 科技 | Global | 99.8 | 0.2% | 920 |
| 科技 | **Sector** | **94.5** | 1.5% | **780** |
| 医药 | Global | 99.8 | 0.2% | 880 |
| 医药 | **Sector** | **94.8** | 1.4% | **750** |

**预期 trade-off**：
- Sector CA-GCP 覆盖率更接近 95%（更"准"，不再过保守）
- Sector CA-GCP 宽度更窄（板块内池小，邻居更紧）
- Sector CA-GCP 板块间独立性更强（科技股灾不影响医药信号）

## 5. 文件清单

| 文件 | 角色 |
|------|------|
| `data/etf_sector_map.csv` | ETF → 板块映射 |
| `ca_gcp/cluster.py` | Sector CA-GCP API |
| `experiments/10_sector_rotation.py` | 板块对比实验 |
| `tests/test_cluster.py` | 单元测试 |
| `docs/84-ca-gcp-sector-rotation.md` | 本文档 |
| `data/results/sector_comparison.csv` | 实验输出 |

## 6. 局限

- 板块映射基于代码规则推断，可能不准确
- 板块内池小（5-10 只），邻居可能重复
- Hybrid 策略的"板块间借用"阈值 0.5 未校准
- 未考虑板块 ETF 的轮动节奏（周频 vs 日频）

## 7. 论文参考

Parker & Zhang (2026) Sec. 5.5:
> "In a highly liquid, strongly correlated large-cap universe, volatility normalization homogenizes scores to the point where topology barely matters. We expect the graph to carry more weight in settings with stronger idiosyncratic or local structure, such as small-capitalization or cross-market universes..."

→ ETF 板块属于"较强 idiosyncratic"场景，板块内独立 CA-GCP 应当更适合。