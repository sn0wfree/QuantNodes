# Stage 16B: RSRS 择时 (Resistance Support Relative Strength)

> **创建日期**: 2026-07-09
> **优先级**: P2 战略级
> **前置**: Stage 16A (多策略组合) - 因 RSRS 仅作为子策略的辅助信号
> **状态**: 规划中

---

## 1. 目标与动机

### 1.1 核心问题

v1.0 策略在 A 股宽基 ETF 上的权重调整依赖:
- 动量信号 (滞后, 144 天窗口)
- 动量加速器 (Stage 14B 待实现)
- 反转信号 (Stage 16A 反转子策略)

**缺失**: 实时市场情绪指标

### 1.2 解决思路: RSRS 阻力支撑相对强度

**RSRS 定义** (光大证券 2017):
- 用 high/low 数据对最近 N 日 (默认 18 日) 做线性回归
- 斜率 β 反映市场动能: β > 1 = 强势突破, β < 0 = 弱势下破
- 标准化: z-score = (β - β_ma) / β_std

**用法**:
- z > 0.7: 极强多头 → A股宽基权重 +20%
- z > 0: 多头 → 正常权重
- z < 0: 空头 → A股宽基权重 -50%
- z < -0.7: 极弱 → A股宽基权重 = 0

### 1.3 预期收益

| 应用场景 | 预期改善 |
|---------|----------|
| 924 案例 | A股宽基在政策发布日 (z > 0.7) 权重 +20%, 抓住 +5~8% 反弹 |
| 2018 熊市 | 极弱时 (z < -0.7) 减仓, 减少 -3~5% 损失 |
| 2022 熊市 | 极弱时 A股宽基减仓, 减少 -2~3% 损失 |
| 综合 | Calmar +5~10%, DD 进一步降低 |

### 1.4 风险

- **数据缺失**: 现有数据只有 close, **无 high/low**
- **需新数据源**: akshare 或其他行情 API
- **单一标的依赖**: z 计算依赖沪深300 指数, 标的选择敏感

---

## 2. 数据需求 (关键瓶颈)

### 2.1 现有数据

| 字段 | 来源 | 覆盖 | 用途 |
|------|------|------|------|
| ETF close | Tencent | 2018-01 ~ 2026-06 | 动量/波动率 |

### 2.2 缺失数据

| 字段 | 必需 ETF | 时间范围 | 来源 |
|------|----------|----------|------|
| ETF high/low | 沪深300 (510300) | 2018-01 ~ 2026-06 | **需新接口** |
| 指数 high/low | 沪深300指数 (000300.SH) | 同上 | akshare 可用 |

### 2.3 数据源调研

| 数据源 | 价格 | high/low | 速率 | 限制 |
|--------|------|----------|------|------|
| **akshare** (`ak.stock_zh_index_daily`) | 免费 | ✅ | 中 | 需 pip install |
| **Tencent** (现用) | 免费 | ❌ (仅 close) | 快 | API 失效 |
| **East Money** (`ak.fund_etf_hist_em`) | 免费 | ✅ (high/low) | 中 | 需 pip install |
| **Wind/Choice** | 贵 | ✅ | 快 | 需付费 |

**推荐**: akshare (`pip install akshare`)
- 免费
- high/low/close 全有
- 接口成熟 (社区项目)

### 2.4 备用方案

**A. 用 close 估算 high/low (粗糙)**
- 假设 daily_range = close × 2% (波动率代理)
- 优点: 无新数据
- 缺点: 精度差, 信号弱

**B. 改用 ATR (Average True Range) 替代**
- 已有数据可计算
- 信号: ATR 突破历史均值 = 突破
- 精度有限, 但可行

**推荐**: 先用 akshare 补充沪深300指数 high/low, 同时用 ATR 作为 fallback.

---

## 3. 技术方案

### 3.1 RSRS 计算

**输入**:
- `high`: 沪深300 ETF (510300) 每日最高价
- `low`: 沪深300 ETF (510300) 每日最低价
- `lookback`: 回归窗口 (默认 18)

**输出**:
- `beta`: 斜率
- `r_squared`: 决定系数
- `z_score`: (β - μ_β) / σ_β, μ/σ 用 600 日滚动

**实现**:
```python
def compute_rsrs(high: pd.Series, low: pd.Series, lookback: int = 18) -> pd.DataFrame:
    """计算 RSRS 指标 (斜率 + R² + z-score)."""
    betas = []
    r2s = []
    for i in range(lookback, len(high) + 1):
        window_high = high.iloc[i-lookback:i].values
        window_low = low.iloc[i-lookback:i].values
        slope, intercept, r, _, _ = stats.linregress(window_low, window_high)
        betas.append(slope)
        r2s.append(r**2)

    beta_series = pd.Series(betas, index=high.index[lookback-1:])
    r2_series = pd.Series(r2s, index=high.index[lookback-1:])

    # z-score: 600 日滚动
    mu = beta_series.rolling(600).mean()
    sigma = beta_series.rolling(600).std()
    z_score = (beta_series - mu) / sigma.replace(0, np.nan)

    # 综合 RSRS 信号
    rsrs = z_score * r2_series  # 引入 R² 加权

    return pd.DataFrame({
        'beta': beta_series,
        'r2': r2_series,
        'z': z_score,
        'rsrs': rsrs,
    })
```

### 3.2 集成点

#### 3.2.1 A股宽基权重调整

```python
def adjust_a_share_weight_by_rsrs(
    base_weights: dict[str, float],
    rsrs_value: float,
    a_share_broad_codes: list[str] = ['510300', '510500', '510050', '159915'],
) -> dict[str, float]:
    """根据 RSRS 调整 A 股宽基权重.

    z > 0.7:  +20%  (极强突破)
    z > 0:    +10%  (多头)
    z < 0:    -10%  (空头)
    z < -0.7: -50%  (极弱)
    """
    if rsrs_value > 0.7:
        adj = 0.20
    elif rsrs_value > 0:
        adj = 0.10
    elif rsrs_value > -0.7:
        adj = -0.10
    else:
        adj = -0.50

    new_weights = dict(base_weights)
    a_total = sum(new_weights.get(c, 0) for c in a_share_broad_codes)
    if a_total > 0:
        scale = (1 + adj)
        for c in a_share_broad_codes:
            new_weights[c] = new_weights.get(c, 0) * scale
        # 归一化
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
    return new_weights
```

#### 3.2.2 集成到 select_and_weight_v2

```python
def select_and_weight_v2(...):
    # ... 现有逻辑 (cap 过滤, 加权) ...

    # Stage 16B: RSRS 调整 A 股宽基权重
    if cfg.rsrs.enabled:
        rsrs_df = compute_rsrs(high_series, low_series, cfg.rsrs.lookback)
        current_z = rsrs_df['rsrs'].loc[:as_of].iloc[-1]
        state.weights = adjust_a_share_weight_by_rsrs(
            state.weights, current_z, cfg.rsrs.a_share_broad_codes
        )

    # ... 后续逻辑 ...
```

### 3.3 回测兼容性

**问题**: RSRS 改变权重后, NAV 计算会变化
**方案**: 在 backtest_v2.py 的 `if date in rebal_dates:` 分支中, 在 `state.weights = ...` 之后插入 RSRS 调整

```python
# backtest_v2.py 现有:
state = apply_stops(etf_norm, pool, rot_eff, prev_weights, date)
# 之后:
if rot.rsrs.enabled and i > 0:
    state.weights = apply_rsrs_adjustment(etf_norm, state.weights, date, rot.rsrs)
```

---

## 4. 文件结构

```
v2/
├── (现有文件不动)
└── rsrs_v2.py                 # 新增: RSRS 计算 + 集成

common/
├── (现有文件不动)
└── rsrs.py                    # 新增: 通用 RSRS 计算 (可独立测试)

scripts/
└── fetch_etf_highlow.py       # 新增: 拉取 high/low 数据 (akshare)

data/real/
├── per_etf_hl/                # 新增: high/low per ETF
│   ├── 510300.parquet         # columns: high, low, close
│   └── ...
└── etf_highlow_2018_2026.parquet  # 新增: 沪深300 high/low
```

---

## 5. 实施步骤 (建议 7-10 天)

### 步骤 1: 数据补充 (2 天)
- [ ] `pip install akshare` (用户需确认)
- [ ] 创建 `scripts/fetch_etf_highlow.py`
- [ ] 拉取 510300 (沪深300) high/low/close
- [ ] 验证数据完整性 (2018-01 ~ 2026-06)
- [ ] 存储到 `data/real/per_etf_hl/`

**风险**: akshare 拉数据不稳定, 需要重试机制

### 步骤 2: RSRS 计算 (1.5 天)
- [ ] 创建 `common/rsrs.py`
- [ ] 实现 `compute_rsrs(high, low, lookback)` (向量化)
- [ ] 单元测试: tests/strategy/momentum_etf_rotation/test_rsrs.py

### 步骤 3: A股宽基权重调整 (1 天)
- [ ] 创建 `v2/rsrs_v2.py`
- [ ] 实现 `adjust_a_share_weight_by_rsrs()`
- [ ] 单元测试

### 步骤 4: 集成到 select_and_weight (1 天)
- [ ] 在 `v2/portfolio_v2.py` 集成 RSRS
- [ ] 添加 `RotationConfig.rsrs: RSRSConfig` 字段
- [ ] 集成测试

### 步骤 5: 回测验证 (1.5 天)
- [ ] 全周期: Calmar ≥ 1.65 (不退化)
- [ ] 924 专项: A股宽基权重 ≥ 15%
- [ ] 2018/2022 熊市: DD 改善 ≥ 1pp

### 步骤 6: 文档与提交 (0.5 天)
- [ ] `reports/momentum_etf_rotation/v2/stage16b_rsrs.md`
- [ ] `reports/momentum_etf_rotation/charts/v2/stage16b_*.html`
- [ ] 更新 STAGE_SUMMARY.md
- [ ] git commit

---

## 6. RSRSConfig 设计

```python
@dataclass
class RSRSConfig:
    """RSRS 阻力支撑相对强度配置 (Stage 16B)."""
    enabled: bool = False
    lookback: int = 18          # 回归窗口 (光大默认 18)
    z_window: int = 600         # z-score 滚动窗口
    benchmark_code: str = "510300"  # 沪深300 ETF (作为标的)
    # 阈值
    strong_threshold: float = 0.7   # 极强
    weak_threshold: float = -0.7   # 极弱
    # 调整幅度
    strong_adj: float = 0.20       # 极强时 +20%
    normal_pos_adj: float = 0.10   # 多头时 +10%
    normal_neg_adj: float = -0.10  # 空头时 -10%
    weak_adj: float = -0.50        # 极弱时 -50%
```

---

## 7. 测试计划

### 7.1 单元测试
- `test_rsrs.py` (新): 斜率, R², z-score, 边界条件
- `test_rsrs_v2.py` (新): 权重调整 4 个区间
- `test_fetch_etf_highlow.py` (新): 数据拉取

### 7.2 集成测试
- 全周期: Calmar ≥ 1.65, DD ≤ -4.5%
- 924: A股宽基权重 ≥ 15%
- 2018-12: A股宽基权重 ≤ 0% (z < -0.7)

### 7.3 鲁棒性测试
- 不同 lookback (10, 18, 30) 敏感性
- 不同 z_window (300, 600, 1200) 稳定性

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| akshare 不可用 | 中 | 高 | Fallback: 用 ATR 近似 |
| 数据缺失 | 中 | 高 | 多数据源 + 缓存 |
| 过拟合 (lookback/z_window) | 中 | 中 | 多窗口鲁棒性测试 |
| z 极端值 | 低 | 中 | clip 到 [-3, 3] |

---

## 9. 验收标准

| 指标 | 阈值 |
|------|------|
| 全周期 Calmar | ≥ 1.65 |
| 全周期 DD | ≤ -4.5% |
| 924 期间 A股宽基权重 | ≥ 15% |
| 2018-12 熊市 A股宽基权重 | ≤ 0% |
| RSRS 数据完整 | 2018-01-02 ~ 2026-06-30 |
| 测试通过率 | 100% |

---

## 10. 后续联动

- Stage 16C (RL): RSRS z-score 可作为 RL 状态特征
- Stage 16A (多策略): RSRS 作为子策略的辅助信号
- Stage 16D (实时): RSRS 需要 daily 拉取数据

---

## 11. 文档与资产

完成后产出:
1. `reports/momentum_etf_rotation/v2/stage16b_rsrs.md` (详细报告)
2. `reports/momentum_etf_rotation/charts/v2/stage16b_*.html` (3-4 个图表)
3. `data/real/per_etf_hl/510300.parquet` (high/low 数据)
4. `scripts/fetch_etf_highlow.py` (数据拉取脚本)
5. 更新 STAGE_SUMMARY.md
