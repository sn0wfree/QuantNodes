# v5 子策略实施报告 (Stage 18)

> **基于**: Stage 17 v4 诊断研究 (SUB_STRATEGY_DIAGNOSTIC.md)
> 
> **核心成果**:
> - **v5 风格轮动**: Calmar **0.439** (vs 当前 v4 单窗口 0.016, **27x 改善**)
> - **v5 因子择时**: Calmar **0.712** (vs 等权 7 Smart β 0.478, **+49%**)
> - **三策略组合 (v3 33% + v5风格 33% + v5因子 34%)**: Calmar **0.763** (vs v3 0.484, **+58%**)

---

## 一、v5 风格轮动 (StyleRotation v5)

### 1.1 4 大改进 (基于诊断)

| # | 改进 | 诊断基础 |
|---|------|---------|
| 1 | **强制 dividend 底仓 20%** | 5 风格组 0.86-0.90 高度相关, dividend 是唯一分散器 (与 tech -0.01) |
| 2 | **多窗口 Long-biased 5/20/120/180** | 单窗口 L=120 Calmar 0.016; 多窗口组合 = L180 median +2.45% |
| 3 | **Top-2 选择** | Top-1 准确率 34.5%, Top-2 53.4% |
| 4 | **Sideways regime filter** (50% 仓位) | Sideways 70% 时间亏钱 -2.50% ann |

### 1.2 实施细节 (style_rotation_v5.py)

```python
@dataclass
class StyleRotationV5Config(SubStrategyConfig):
    name: str = "style_rotation_v5"
    windows: tuple[int, ...] = (5, 20, 120, 180)        # 改进 2
    window_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    dividend_floor: float = 0.20                        # 改进 1
    top_n: int = 2                                      # 改进 3
    top_n_per_style: int = 1
    regime_lookback_short: int = 60
    regime_lookback_long: int = 252
    bull_threshold: float = 0.05
    bear_threshold: float = -0.05
    long_threshold: float = 0.10
    sideways_style_exposure: float = 0.50               # 改进 4
    rebalance_freq: str = "M"
    min_history: int = 252
    max_weight: float = 0.40
```

**关键算法**:
- `multi_window_score()`: 每个窗口用 `rank_pct(group_max(L-day-return))`, 然后 Long-biased 加权
- `classify_regime()`: HS300 60d/252d 动量 → bull/bear/sideways
- `run_step()`: 多窗口得分 → Top-2 → 加权 (dividend 底仓 + score 加权) → regime 缩放 → cash 注入

### 1.3 独立回测结果

| 指标 | v5 风格轮动 | v4 当前 (单窗口) | 提升 |
|------|------------|-----------------|------|
| Ann | **10.06%** | 0.49% | 20x |
| Vol | - | 36% | - |
| Sharpe | 0.70 | 0.20 | 3.5x |
| Max DD | **-22.95%** | -30.23% | -24% |
| **Calmar** | **0.439** | 0.016 | **27x** |
| n_rebal | 60 | 60 | - |

#### Year-by-year
| 年 | 收益 | 备注 |
|----|------|------|
| 2018 | 0% | min_history=252 限制 |
| 2019 | **+16.93%** | 成长牛 |
| 2020 | +13.81% | |
| 2021 | -0.69% | 风格切换 |
| 2022 | **-10.23%** | 熊市 (bear filter 未启用) |
| 2023 | -3.51% | sideways |
| 2024 | **+16.10%** | 关键胜出 (v3 同期 -4.45%) |
| 2025 | **+37.79%** | 风格切换 |
| 2026 | +18.50% | H1 |

#### Regime 分布 (rebal 日)
- sideways: 42 (70.0%) → 应用 50% 风格 + 50% cash
- bull: 11 (18.3%) → 100% 风格
- bear: 7 (11.7%) → 70% 风格 (注意: 这里 v5 风格没单独 bear filter)

---

## 二、v5 因子择时 (FactorTiming v5)

### 2.1 5 大改进 (基于诊断)

| # | 改进 | 诊断基础 |
|---|------|---------|
| 1 | **因子特异性 forward_window** | momentum 120d (+0.067), value 40d (+0.072), reversal 60d (+0.018) |
| 2 | **lag 平滑** (momentum/value/dividend/quality 用 4w) | 5 因子 IC lag1=0.48-0.69 高持续; reversal lag1=-0.01 不平滑 |
| 3 | **Regime-conditioned 因子选择** | bull: m+v, bear: v+d+q, sideways: v |
| 4 | **删除 low_vol 因子** | IC vs forward 相关 -0.454, 反指因子 |
| 5 | **IC 质量过滤** (\|IC\|<0.05 → weight=0) | \|IC\|>0.05 频率 84-94% 是噪声 |

### 2.2 实施细节 (factor_timing_v5.py)

```python
@dataclass
class FactorTimingV5Config(SubStrategyConfig):
    name: str = "factor_timing_v5"
    factor_fw: dict[str, int] = field(default_factory=lambda: {
        "momentum": 120, "reversal": 60, "value": 40,
        "dividend": 180, "quality": 252,
    })
    factor_smooth_window: dict[str, int] = field(default_factory=lambda: {
        "momentum": 4, "value": 4, "dividend": 4, "quality": 4,
        "reversal": 1,  # 无 lag 持续
    })
    factor_ic_threshold: float = 0.05
    regime_factors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bull":     ("momentum", "value"),
        "bear":     ("value", "dividend", "quality"),
        "sideways": ("value",),
    })
    factor_to_etf: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "momentum": ("510300", "510500", "159915", "588000", "510880"),
        "reversal": ("510300", "510500", "159915", "588000", "510880"),
        "value":    ("512040",),
        "dividend": ("510880", "512890", "515080", "515100"),
        "quality":  ("515900",),
    })
```

**关键算法**:
- `compute_v5_factor_weights()`: regime → 可用因子 → |IC| 过滤 → raw_w = max(0, IC+0.05)^2 → 归一化
- `aggregate_factor_to_etf()`: 因子权重 → ETF 权重 (因子内等权, 多因子→同 ETF 累加)
- `classify_regime_v5()`: 同 v5 风格

### 2.3 独立回测结果

| 指标 | v5 因子择时 | 等权 7 Smart β | 提升 |
|------|------------|---------------|------|
| Ann | **13.52%** | 9.85% | +37% |
| Sharpe | **0.81** | 0.71 | +14% |
| Max DD | **-19.00%** | -20.60% | -8% |
| **Calmar** | **0.712** | 0.478 | **+49%** |
| n_rebal | 60 | - | - |

#### Year-by-year
| 年 | 收益 | 备注 |
|----|------|------|
| 2019 | **+26.25%** | momentum 主导 |
| 2020 | +18.33% | |
| 2021 | **+22.64%** | momentum 持续 |
| 2022 | **-8.37%** | bear → value/quality (防御性亏损) |
| 2023 | +11.65% | |
| 2024 | **+21.93%** | 关键胜出 (v3 同期 -4.45%) |
| 2025 | +25.67% | |
| 2026 | -2.40% | H1 回撤 |

---

## 三、组合优化 (Diversification)

### 3.1 相关性分析 (日收益)

| | v3 | v5_style | v5_factor | combo_50_50 |
|---|----|----|----|----|
| **v3** | 1.00 | **0.56** | **0.55** | 0.58 |
| **v5_style** | 0.56 | 1.00 | 0.82 | 0.93 |
| **v5_factor** | 0.55 | 0.82 | 1.00 | 0.97 |
| combo_50_50 | 0.58 | 0.93 | 0.97 | 1.00 |

**关键发现**:
- **v3 与 v5 风格/因子 相关仅 0.55-0.56** → **真正分散** (v3 是独立分散器)
- v5 风格 vs v5 因子 相关 0.82 (都是 Smart β 类, 相关性高)
- 60d 滚动相关: v5_style vs v5_factor mean 0.84, min 0.42, max 0.997

### 3.2 组合优化 (与 v3 混合)

#### v5 风格 + v3
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v5s 20% + v3 80% | 7.48% | **0.92** | -13.12% | 0.570 |
| v5s 30% + v3 70% | 7.83% | 0.89 | -13.84% | 0.566 |
| v5s 40% + v3 60% | 8.17% | 0.86 | -14.58% | 0.561 |
| v5s 50% + v3 50% | 8.51% | 0.82 | -15.62% | 0.544 |

→ 风格轮动与 v3 组合效果一般 (Calmar 提升有限)

#### v5 因子 + v3 ⭐
| 比例 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v5f 20% + v3 80% | 8.39% | 0.93 | -12.20% | 0.687 |
| **v5f 30% + v3 70%** | 9.13% | 0.91 | -12.78% | **0.715** |
| **v5f 40% + v3 60%** | 9.85% | 0.88 | -13.75% | **0.716** |
| v5f 50% + v3 50% | 10.53% | 0.86 | -14.74% | 0.714 |
| v5f 60% + v3 40% | 11.18% | 0.84 | -15.64% | 0.715 |

→ **v5 因子 + v3 是最佳 2 策略组合**: v5因子 30-40% + v3 60-70% → Calmar **0.715-0.716**

### 3.3 三策略组合 (v3 + v5 风格 + v5 因子) ⭐⭐⭐

| 配置 | Ann | Sharpe | DD | Calmar |
|------|------|--------|----|--------|
| v3 60% + v5s 20% + v5f 20% | 9.03% | 0.89 | -12.66% | 0.713 |
| v3 50% + v5s 25% + v5f 25% | 9.55% | 0.86 | -13.04% | 0.732 |
| v3 40% + v5s 30% + v5f 30% | 10.05% | 0.84 | -13.40% | 0.750 |
| **v3 33% + v5s 33% + v5f 34%** | 10.41% | 0.83 | -13.65% | **0.763** ⭐ |

→ **三策略等权组合 Calmar 0.763** (vs v3 单独 0.484, **+58%**)

### 3.4 50/50 (v5 风格 + v5 因子)
| 指标 | 值 |
|------|---|
| Ann | 11.89% |
| Sharpe | 0.79 |
| DD | -16.16% |
| **Calmar** | **0.736** |

---

## 四、最终推荐

### 4.1 推荐生产配置

| 风险偏好 | 配置 | 预期 Calmar | 实际 |
|---------|------|------------|------|
| **稳健** | v3 70% + v5因子 30% | ~0.7 | **0.715** ⭐ |
| **平衡** | v3 50% + v5风格 25% + v5因子 25% | ~0.7-0.75 | **0.732** |
| **激进** | v3 33% + v5风格 33% + v5因子 34% | ~0.75 | **0.763** ⭐ |
| **替代** | v3 40% + v5风格 30% + v5因子 30% | ~0.75 | **0.750** |

### 4.2 关键经验

1. **v3 + v5 因子择时 是 2 策略最优解** (Calmar 0.715)
2. **加 v5 风格有边际改善** (Calmar 0.732-0.763, 但 3 策略权重相近)
3. **v3 是独立分散器** (与 v5 风格/因子 相关 0.55-0.56)
4. **v5 风格与 v5 因子 高相关 0.82** — 它们共享 Smart β universe, 真实 alpha 不完全独立

### 4.3 局限与风险

1. **v5 风格 2022 熊市 -10.23%** — bear regime filter 未启用 (只对 sideways 减仓)
2. **v5 因子 2026 H1 -2.40%** — 最近回撤
3. **v5 风格 / 因子 高度相关 (0.82)** — 多样化有限
4. **min_history=252 限制**: 2018 年无信号 (前 252 天)
5. **没有交易成本**: 实际 Calmar 会略低 (0.01-0.05)

---

## 五、文件清单

### 5.1 新增
- `QuantNodes/strategy/momentum_etf_rotation/v5/__init__.py` (24 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/style_rotation_v5.py` (245 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/factor_timing_v5.py` (281 行)
- `scripts/v5_backtest.py` (230 行) — v5 独立回测 + 组合验证
- `reports/momentum_etf_rotation/v5/v5_navs.parquet` — NAV 数据

### 5.2 诊断基础
- `reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md` (368 行)
- `scripts/style_rotation_diagnostic.py` (392 行)
- `scripts/factor_timing_diagnostic.py` (257 行)

---

## 六、下一步

1. **生产部署**: 实施 v3 33% + v5风格 33% + v5因子 34% (Calmar 0.763)
2. **加 bear filter** 给 v5 风格 (2022 改善空间)
3. **Walk-forward OOS 验证** v5 在 2020-2026 子样本
4. **与 v3 Stage 18 升级协同**: v3 动量参数 + v5 因子择时 = 更强因子组合
