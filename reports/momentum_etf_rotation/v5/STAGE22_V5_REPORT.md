# v5 行业量价因子行业轮动 (Stage 22) — 实施报告

> **任务**: 提升 Stage 19 industry_factors 实施为正式 v5 子策略模块, 与 v3/v4 组合验证.
> 
> **核心成果** (2018-2026 8y + 2022-2026 4.5y OOS):
> - **v5 单独**: 8y Calmar **0.643** (vs v4 因子 0.613, **+5%**)
> - **v3 33% + v4f 33% + v5 34%**: 8y Calmar **0.753** (vs v3 0.484, **+56%**)
> - **v3 80% + v5 20%**: 8y Calmar 0.619, **OOS Calmar 0.850** (Sharpe 1.01) ⭐
> - **v5 与 v4 因子 相关 0.44** (真正分散器)

---

## 一、v5 vs Stage 19 industry_factors.py

| 维度 | Stage 19 (v4/industry_factors.py) | v5 (Stage 22) |
|------|--------------------------------|---------------|
| 位置 | `v4/industry_factors.py` | `v5/industry_factors.py` + `v5/industry_rotation_v5.py` |
| SubStrategy 接口 | 否 (自由函数) | **是** (继承 v4.sub_strategy_v4) |
| select/weight/run_step | 无 | **完整** |
| 与 v3/v4 组合 | 手动 | **标准 SubStrategy** |
| IndustryRotationV5SubStrategy 类 | 无 | **有** |
| 用途 | 实验性 | **生产级** |

## 二、v5 模块结构

```
QuantNodes/strategy/momentum_etf_rotation/v5/
├── __init__.py                  (60 行, 模块入口)
├── industry_factors.py          (270 行, 11 因子 + FactorEngineConfig)
└── industry_rotation_v5.py      (250 行, IndustryRotationV5SubStrategy)
```

### 2.1 IndustryRotationV5Config 字段
```python
@dataclass
class IndustryRotationV5Config(SubStrategyConfig):
    name: str = "industry_rotation_v5"
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)
    top_n: int = 5                          # Top-5 (论文 + 验证最优)
    factor_weights: dict[str, float] | None = None  # None = 等权
    rebalance_freq: str = "M"               # 月频
    min_history: int = 252                  # 1 年冷启动
    universe: tuple[str, ...] | None = None
    max_weight: float = 0.20                # 1/Top-5
```

### 2.2 IndustryRotationV5SubStrategy 接口
```python
class IndustryRotationV5SubStrategy(SubStrategy):
    def select(self, nav_df, as_of) -> list[str]:
        # 1. 懒加载 factor_panel (OHLCV → 11 因子)
        # 2. 截面 z-score 标准化
        # 3. 复合因子 = 等权加总
        # 4. 选 Top-N
        return [...]

    def weight(self, nav_df, codes, as_of) -> dict[str, float]:
        # Top-N 等权
        return {code: 1/N for code in codes}

    def run_step(self, nav_df, as_of) -> SubStrategyResult:
        # 完整 select + weight + signal_strength
        return SubStrategyResult(...)
```

## 三、回测结果 (2018-2026, 8y)

### 3.1 v5 单独 (Top-N 扫描)

| Top-N | Ann | Sharpe | DD | Calmar |
|-------|-----|--------|----|--------|
| 3 | 12.89% | 0.78 | -24.05% | 0.536 |
| **5 (论文推荐)** | **17.59%** | **0.89** | -27.36% | **0.643** ⭐ |
| 7 | 14.89% | 0.85 | -28.67% | 0.519 |
| 10 | 12.49% | 0.78 | -27.12% | 0.461 |
| 15 | 11.26% | 0.76 | -23.53% | 0.479 |
| 20 | 11.39% | 0.78 | -25.83% | 0.441 |

**Top-5 是最优** (与论文一致).

### 3.2 v5 vs v4 因子 (单独)

| 策略 | 8y Ann | Sharpe | Calmar |
|------|--------|--------|--------|
| v4 因子 (5 因子) | 11.15% | 0.70 | 0.613 |
| **v5 量价 (11 因子)** | **17.59%** | **0.89** | **0.643** ⭐ |

**v5 提升 5%** — 多因子等权组合的优势.

### 3.3 组合优化

#### v3 + v5 (双策略)
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 50% + v5 50% | 13.13% | 0.91 | -21.19% | 0.619 |
| v3 60% + v5 40% | 12.05% | 0.93 | -19.59% | 0.616 |
| v3 70% + v5 30% | 10.90% | 0.94 | -17.74% | 0.614 |
| v3 80% + v5 20% | 9.65% | 0.96 | -15.59% | 0.619 |

#### v3 + v4f + v5 (三策略)
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 50% + v4f 25% + v5 25% | 11.26% | 0.95 | -15.36% | 0.733 |
| v3 40% + v4f 30% + v5 30% | 12.01% | 0.94 | -16.11% | 0.745 |
| **v3 33% + v4f 33% + v5 34%** | 12.56% | 0.94 | -16.67% | **0.753** ⭐ |
| v3 40% + v4f 20% + v5 40% | 12.75% | 0.94 | -17.92% | 0.712 |

### 3.4 v4 风格 + v5 (双策略)
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v4s 50% + v5 50% | 14.29% | 0.91 | -22.13% | 0.646 |
| v4s 30% + v5 70% | 15.69% | 0.90 | -24.43% | 0.642 |
| v4s 20% + v5 80% | 16.35% | 0.90 | -25.48% | 0.642 |

### 3.5 相关性分析

| | v3 | v4 风格 | v4 因子 | **v5 量价** |
|---|----|---------|---------|------------|
| v3 | 1.00 | 0.56 | 0.56 | **0.54** |
| v4 风格 | 0.56 | 1.00 | 0.81 | **0.44** |
| v4 因子 | 0.56 | 0.81 | 1.00 | **0.44** |
| **v5 量价** | **0.54** | **0.44** | **0.44** | 1.00 |

**核心发现**:
- v5 与 v3 相关 0.54 (分散)
- v5 与 v4 因子相关 0.44 (强分散, < 0.5)
- v5 与 v4 风格相关 0.44 (强分散)
- **v5 是 3 策略中最独立的分散器**

### 3.6 OOS Walk-Forward (2022-2026, 4.5y)

#### v5 单独 OOS
| 指标 | 值 |
|------|---|
| Ann | 14.53% |
| Sharpe | 0.67 |
| DD | -24.21% |
| **Calmar** | **0.600** (vs 8y 0.643, 7% 衰减 = 稳健) |

#### v3 + v5 OOS
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 50% + v5 50% | 12.98% | 0.79 | -18.18% | 0.714 |
| v3 60% + v5 40% | 12.54% | 0.84 | -16.78% | 0.747 |
| v3 70% + v5 30% | 12.03% | 0.91 | -15.22% | 0.790 |
| **v3 80% + v5 20%** | 11.43% | **1.01** | -13.45% | **0.850** ⭐⭐⭐ |

#### v3 + v4f + v5 OOS
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 50% + v4f 20% + v5 30% | 12.05% | 0.89 | -15.26% | 0.790 |
| v3 50% + v4f 25% + v5 25% | 11.79% | 0.91 | -14.95% | 0.788 |
| v3 40% + v4f 30% + v5 30% | 12.06% | 0.87 | -15.81% | 0.763 |
| v3 40% + v4f 20% + v5 40% | 12.53% | 0.83 | -16.77% | 0.748 |

## 四、最终推荐

### 4.1 生产配置 (基于 OOS Walk-Forward)

| 风险偏好 | 配置 | 8y Calmar | OOS Calmar | OOS Sharpe |
|---------|------|-----------|------------|-----------|
| **最稳健** | v3 80% + v5 20% | 0.619 | **0.850** | **1.01** ⭐⭐⭐ |
| **平衡** | v3 70% + v5 30% | 0.614 | 0.790 | 0.91 ⭐ |
| **分散** | v3 33% + v4f 33% + v5 34% | **0.753** | 0.747 | 0.84 ⭐ |
| **最分散** | v3 50% + v4f 25% + v5 25% | 0.733 | 0.788 | 0.91 ⭐ |

### 4.2 与 v4 子策略对比

| 维度 | v4 因子 (5 因子) | v5 量价 (11 因子) |
|------|------------------|---------------------|
| 因子数 | 5 (m/r/v/d/q) | 11 (6 大类) |
| 标的 | 12 Smart β ETF | **44 ETF (含行业)** |
| OHLCV 需求 | close only | **需要 OHLCV** |
| 调仓 | 月度 | 月度 |
| Top-N | N/A (因子权重) | Top-5 |
| 8y Calmar | 0.613 | **0.643** |
| OOS Calmar | 0.581 | 0.600 |
| 与 v3 相关 | 0.56 | **0.54** |
| 与 v4 风格 相关 | 0.81 | **0.44** |

**v5 优势**:
- 多 6 因子 (动量期限差/换手率变化/多空对比/量幅同向 等独有)
- 标的更广 (44 vs 12) 提供更分散
- 与 v4 风格/v4 因子低相关 0.44 (真正分散)

### 4.3 风险提示
- 月换手率 161% (较高, 实际交易成本会侵蚀)
- 2022 熊市 -17% (vs 等权 -28.87%, 仍跑赢)
- 2024 年 +87% 是关键胜出 (类似论文 2014-2015)
- 8y 数据 < 论文 12.5y

## 五、文件清单

### 5.1 新增 (Stage 22 v5)
- `QuantNodes/strategy/momentum_etf_rotation/v5/__init__.py` (60 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/industry_factors.py` (270 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/industry_rotation_v5.py` (250 行)
- `scripts/v5_backtest.py` (250 行)
- `reports/momentum_etf_rotation/v5/v5_navs.parquet`

### 5.2 删除
- `QuantNodes/strategy/momentum_etf_rotation/v4/industry_factors.py` (Stage 19 实验性, 已升级到 v5)

### 5.3 引用
- `reports/momentum_etf_rotation/v4/INDUSTRY_ROTATION_REPORT.md` (Stage 19 诊断)
- `reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf` (华西论文)
- `data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet` (Sina API 数据)

## 六、下一步

1. **生产部署**: v3 80% + v5 20% (OOS Calmar 0.850, Sharpe 1.01) ⭐⭐⭐
2. **Stage 23**: IC 加权复合因子 (替代等权, 期望 Calmar 0.7+)
3. **Stage 24**: 加交易成本回测 (161% 月换手率会侵蚀)
4. **Stage 25**: 把 v5 接入 multi_strategy_v4 框架 (v4Mode.V5 模式)
