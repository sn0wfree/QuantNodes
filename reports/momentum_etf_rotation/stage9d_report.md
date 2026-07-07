# Stage 9-D 报告 — HMM Regime 检测器

> Stage 9-D: 添加 HMM 隐马尔可夫模型, 识别 3 种市场状态, 动态调整参数
> 完成日期: 2026-07-07
> 状态: ✅ 完成 (但表现**不如预期**, 见下文)

## 1. 改动概览

### 1.1 新增配置 (`regime_detector.py`)

```python
@dataclass
class RegimeParams:
    bull:   {"lookback": 60,  "rank_cutoff": 0.50}
    neutral: {"lookback": 90,  "rank_cutoff": 0.30}
    bear:   {"lookback": 90,  "rank_cutoff": 0.10}  # 原 144, 缩短避免数据不足

@dataclass
class RegimeDetector:
    enabled: bool = False
    n_regimes: int = 3
    lookback_train: int = 504       # 2 年训练窗口
    retrain_freq: int = 60
    regime_params: RegimeParams = field(default_factory=RegimeParams)
    benchmark_code: str = "510300"

class HMMRegimeDetector:
    """3-regime HMM (GaussianHMM).
    
    特征: 日收益率 + 21 日波动率
    标签: 0=熊, 1=震荡, 2=牛 (按均值排序)
    """
```

### 1.2 集成点 (`backtest.py`)

```python
# 回测前训练 HMM
if rot.regime_detector.enabled:
    detector = HMMRegimeDetector(...)
    detector.fit(train_nav)  # 前 lookback_train 天

# 每次调仓, 用 HMM 预测当前 regime, 覆盖参数
if detector is not None and i > 0:
    regime = detector.predict(...)
    overrides = get_regime_params(regime)
    rot_eff = replace(rot, **overrides)
```

## 2. 真实数据回测结果 (2019~2026)

### 2.1 性能对比

| 配置 | Calmar | DD | Ann |
|------|--------|-----|-----|
| **Baseline** | **0.78** | **-21.05%** | **16.35%** |
| HMM Regime (default) | 0.52 | -27.94% | 14.49% |
| HMM + TF bear=0.7 | 0.47 | -27.94% | 13.03% |

### 2.2 OOS 段 (2024-2026)

| 配置 | Calmar | DD | Ann |
|------|--------|-----|-----|
| Baseline | **1.72** | -12.96% | 22.31% |
| HMM Regime | 1.07 | -12.81% | 13.72% |

## 3. 关键发现: HMM 在此数据集表现不佳

### 3.1 表现劣于 Baseline
- Calmar 从 0.78 降至 0.52 (-33%)
- DD 从 -21.05% 恶化至 -27.94%
- Ann 从 16.35% 降至 14.49%

### 3.2 原因分析

1. **频繁切换**: HMM 在不同 regime 间频繁切换, 导致策略参数不稳定
2. **过拟合**: HMM 在训练窗口(2018-2019)学到的模式不适用于 2020-2026
3. **Regime 划分不准确**: 3-regime 模型对本数据不够细粒度
4. **Regime 参数不当**: bear regime 的 lookback=90 + rank_cutoff=0.10 过于保守

### 3.3 验证计划中的预测

> 修订后计划退出条件: "HMM 仅在 Phase 2 全部成功后考虑"
> "HMM 容易过拟合, 必须严格 OOS 测试"

✅ 计划正确识别了 HMM 的高风险
✅ 实际表现验证了"高过拟合风险"警告
✅ 没有让 HMM 影响生产配置

## 4. 决策: HMM 不进入默认配置

### 不推荐启用 HMM
- ❌ Calmar 0.52 远低于 Baseline 0.78
- ❌ DD -27.94% 恶化
- ❌ 频繁切换导致不稳定的交易信号

### 退出条件 (计划要求)

| 条件 | 实际 | 结果 |
|------|------|------|
| OOS Calmar > 0.5 | 1.07 | ✅ (通过) |
| OOS 段 regime 切换稳定 | 频繁切换 | ❌ |
| Calmar 提升 | 下降 | ❌ |

**结论**: HMM 未达到计划退出条件, 不进入默认配置。

## 5. 教训与启示

### 5.1 状态机策略的固有风险
- HMM/RL 等状态机模型对 regime 划分敏感
- 在趋势市容易过拟合到历史模式
- 不同 regime 的参数需要谨慎调优

### 5.2 已验证可靠的升级 (Stage 9-B, 9-C)
- **Stage 9-B (趋势过滤器)**: Calmar 0.88, DD -17%
- **Stage 9-C (波动率目标)**: Calmar 1.00, DD -7%
- **Stage 9-D (HMM Regime)**: Calmar 0.52, DD -28% ❌

### 5.3 最优配置: Stage 9-C
- 启用 `VolTargeting(enabled=True, target_vol=0.15)`
- Calmar 1.00 > CICC 0.76 (+32%)
- DD -6.89% << CICC -18.78%

## 6. Stage 9 全阶段对比

| Stage | Calmar | DD | Ann | 推荐度 |
|-------|--------|-----|-----|--------|
| Baseline | 0.78 | -21.05% | 16.35% | ★★ |
| 9-A (fused w=0.6) | 0.78 | -20.38% | 15.89% | ★★ |
| 9-B (TF bear=0.7) | **0.88** | **-17.05%** | 14.98% | ★★★ |
| 9-C (VT tv=0.15) | **1.00** | **-6.89%** | 6.87% | **★★★★** |
| 9-D (HMM) | 0.52 | -27.94% | 14.49% | ❌ |

## 7. 测试覆盖

```bash
python3.11 -m pytest tests/strategy/momentum_etf_rotation/test_regime_detector.py -v
# 10/10 PASS
```

测试类:
- `TestHMMRegimeDetector`: 4 个 (核心算法)
- `TestGetRegimeParams`: 3 个 (参数映射)
- `TestBacktestRegime`: 2 个 (回测集成)

总测试数: 157/157 PASS (+10 from Stage 9-D)

## 8. 文件清单

| 文件 | 类型 | 改动 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/regime_detector.py` | 新增 | 150 行 (HMM 检测器 + RegimeParams + RegimeDetector) |
| `QuantNodes/strategy/momentum_etf_rotation/portfolio.py` | 修改 | +5 行 (regime_detector 配置占位) |
| `QuantNodes/strategy/momentum_etf_rotation/backtest.py` | 修改 | +20 行 (HMM 训练 + 动态参数覆盖) |
| `QuantNodes/strategy/momentum_etf_rotation/__init__.py` | 修改 | 导出 RegimeDetector 等 |
| `tests/strategy/momentum_etf_rotation/test_regime_detector.py` | 新增 | 115 行 (10 个测试) |
| `reports/momentum_etf_rotation/charts/stage9d_hmm_regime.html` | 新增 | 净值对比 |
| `reports/momentum_etf_rotation/charts/stage9d_regime_timeline.html` | 新增 | Regime 时间线 |
| `reports/momentum_etf_rotation/stage9d_report.md` | 新增 | 本报告 |

## 9. 退出条件检查

| 检查项 | 阈值 | 实际 | 结果 |
|--------|------|------|------|
| 测试通过率 | ≥ 95% | 100% (10/10) | ✅ |
| Calmar 提升 | > baseline | 0.52 < 0.78 | ❌ |
| DD 改善 | < baseline | -27.94% > -21.05% | ❌ |
| OOS 稳定 | 切换 < 3次 | 频繁切换 | ❌ |

**最终决定**: ❌ HMM 不进入生产配置, 保留代码作为实验性功能。

## 10. Stage 9 总结

| 阶段 | 状态 | 推荐配置 | 最终 Calmar |
|------|------|---------|-------------|
| **9-A** | ✅ 通过 | fused w=0.6 | 0.78 (DD -20.38%) |
| **9-B** | ✅ 通过 | TF bear=0.7 | **0.88** (DD -17.05%) |
| **9-C** | ✅ 通过 | VT tv=0.15 | **1.00** (DD -6.89%) |
| **9-D** | ❌ 失败 | 不推荐 | 0.52 (DD -27.94%) |

**最优升级方案**:
- 风险厌恶: **Stage 9-C** (Calmar 1.00, DD -6.89%)
- 平衡型: **Stage 9-B** (Calmar 0.88, DD -17.05%)
- 追求收益: Baseline + Stage 9-A (Calmar 0.78, DD -20.38%)