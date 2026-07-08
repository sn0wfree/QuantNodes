# 实验失败: HMM Regime 检测器 (Stage 9-D)

> 阶段: Stage 9-D (2026-07-07)
> 结论: ❌ NO-GO

## 假设

- H1: HMM 隐马尔可夫模型识别 3 种市场状态 (牛/震荡/熊), 动态调整参数可降低 DD
- H2: 504 天训练窗口足够估计 44×44 协方差矩阵

## 实现

### 改动文件
- `QuantNodes/strategy/momentum_etf_rotation/regime_detector.py` (新增, 150 行)
- `QuantNodes/strategy/momentum_etf_rotation/portfolio.py` (修改, +5 行)
- `QuantNodes/strategy/momentum_etf_rotation/backtest.py` (修改, +20 行)
- `tests/strategy/momentum_etf_rotation/test_regime_detector.py` (新增, 115 行)
- `reports/momentum_etf_rotation/charts/stage9d_*.html` (2 个图表)
- `reports/momentum_etf_rotation/stage9d_report.md` (详细报告)

### 代码量
- 代码: ~175 行
- 测试: 10 个 (10/10 PASS)
- 文档: 5 个文件

## 失败原因

### 主因: 高维协方差在小样本下过拟合

- 504 天训练 / 3 regime = **每个 regime 只有 ~168 天**
- 44 ETF × (44 + 1) / 2 = **990 个参数**
- 样本/参数比 = 168 / 990 = **0.17** (远低于警戒线 1.0)

**数学上的后果**: 协方差矩阵估计的均方误差爆炸, HMM 学到的是**噪声而非信号**。

### 次因: OOS 验证显示 regime 不稳定

- 训练 (2019-2021) 学到的 regime 模式
- OOS (2024-2026) 显示 regime 频繁切换
- **数据分布漂移**, HMM 无法泛化

### 第三因: bear regime 默认 lookback=144 在 OOS 段数据不足

修复: 改为 90, 但仍表现差

## 证据

| 指标 | 实验版本 (HMM) | baseline | 退化 |
|------|--------------|---------|------|
| Calmar | 0.52 | 0.78 | -33% |
| DD | -27.94% | -21.05% | -33% |
| OOS Calmar | 1.07 | 1.72 | -38% |

## 教训

1. **p>>n 问题**: 低样本高维是统计学上的硬约束, 不是实现 bug
2. **ML 算法的隐性陷阱**: 单元测试 10/10 PASS 看似正常, 但 OOS 暴露真问题
3. **HMM 复杂度高**: 三个参数 (lookback, rank_cutoff, retrain_freq) 互相影响, 难以调试

## 可能的复苏方向

### 条件
- [ ] 使用 **Ledoit-Wolf 收缩协方差** (解决 p>>n)
- [ ] **更长的训练窗口** (至少 1500 天)
- [ ] **减少 regime 数量** (3→2)
- [ ] **更简单的状态机** (基于趋势而非全 HMM)

### 修改方案
```python
@dataclass
class RegimeDetector:
    enabled: bool = False
    n_regimes: int = 2  # 减少到 2 (bull/bear)
    lookback_train: int = 1500  # 加长
    cov_estimator: str = "ledoit_wolf"  # 改用收缩
```

### 验证要求
- 训练期和 OOS 期的 regime 切换模式必须**相似** (regime stability test)
- Calmar 必须 >= 0.78
- OOS Calmar 必须 > 0.5

## 相关文档

- 详细报告: `../stage9d_report.md`
- 图表: `../charts/stage9d_hmm_regime.html`, `../charts/stage9d_regime_timeline.html`
- 协方差讨论: `../COVARIANCE_RESEARCH.md`

---

**失败归档完成** - 2026-07-07
