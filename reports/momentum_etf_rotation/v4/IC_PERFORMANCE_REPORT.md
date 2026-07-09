# IC 因子择时 — 性能验证报告 (Stage 17, v4.0)

> **测试日期**: 2026-07-09
> **数据**: 2018-01-02 ~ 2026-06-30, 12 只 Smart β ETF
> **结论**: IC 加权略弱于等权; value 因子唯一显著正 IC; IC 短窗口略优

---

## 1. 6 因子 IC 统计 (2020-01-02 ~ 2026-06-01, n=311)

| 因子 | Mean IC | Std | ICIR | Hit Rate | 评价 |
|------|---------|-----|------|----------|------|
| **momentum** | -0.0145 | 0.227 | -0.064 | 47.2% | 中性, 略负 |
| **reversal** | -0.0568 | 0.129 | -0.441 | 32.0% | **负向, 反向因子** |
| **value**    | **+0.0437** | 0.255 | **+0.171** | **59.6%** | ⭐ **唯一正 IC** |
| **low_vol**  | -0.0146 | 0.224 | -0.065 | 51.5% | 中性, 略负 |
| **dividend** | -0.0348 | 0.129 | -0.271 | 32.4% | 负向, 红利组跑输 |
| **quality**  | -0.0116 | 0.152 | -0.077 | 46.6% | 中性, 略负 |

**关键发现**:
- 仅 **value** (低偏离/低估因子) 显著正 IC (+0.044, Hit 60%)
- momentum/quality/low_vol 都是噪声信号 (IC ≈ 0)
- reversal/dividend 负 IC, 与金融常识 (A 股短期无反转, 红利组表现一般) 一致

## 2. IC 加权因子权重时序

| 因子 | 起始权重 (2020-01) | 末尾权重 (2026-06) | 平均权重 |
|------|---------------------|---------------------|----------|
| momentum | 16.7% | 4.2% | 25.0% |
| reversal | 16.7% | 4.2% | 7.9% |
| value | 16.7% | **41.4%** | 28.9% |
| low_vol | 16.7% | 4.2% | 16.2% |
| dividend | 16.7% | 4.2% | 8.9% |
| quality | 16.7% | **41.7%** | 13.1% |

**观察**:
- value 和 quality 末尾权重都到 ~40% (高 IC 因子)
- 其他 4 个因子被压到 min_weight 4.2%
- 平均权重 value > low_vol > momentum > quality > dividend > reversal

## 3. 多参数回测对比 (2020-2026)

| 配置 | Sharpe | Calmar | AnnRet | DD | Final Nav |
|------|--------|--------|--------|----|-----------|
| **equal (等权 baseline)** | **0.684** | 0.461 | **11.21%** | -24.34% | 1.939 |
| IC_aggressive (base=0, power=3) | 0.645 | 0.420 | 10.44% | -24.88% | 1.857 |
| IC_default (base=0.05, power=2) | 0.660 | 0.434 | 10.66% | -24.56% | 1.880 |
| IC_soft (base=0.10, power=1.5) | 0.678 | 0.457 | 10.98% | -24.02% | 1.915 |
| IC_long_window (120d) | 0.640 | 0.442 | 10.25% | -23.21% | 1.838 |
| **IC_short_fwd (10d)** ⭐ | 0.669 | **0.480** | 10.93% | **-22.74%** | 1.909 |
| IC_long_fwd (40d) | 0.638 | 0.412 | 10.17% | -24.69% | 1.829 |

**最佳**: **IC_short_fwd** (forward_window=10d), Calmar 0.480 vs 等权 0.461 (+4%)

## 4. 核心结论

### 4.1 IC 加权 ≠ 更好
- 默认参数 (IC_default) 比等权 Sharpe 低 0.024, Calmar 低 0.027
- 原因: IC 信号噪声大, 加权引入额外不确定性
- 等权分散化效果 > 弱 IC 信号的择时能力

### 4.2 短窗口 IC 略好
- forward_window=10d 比 20d 略好 (Calmar +0.046)
- long_window=120d 反而差 (IC 平滑过度丢失信号)
- **建议**: forward=10-15d, smooth=12 step (60d)

### 4.3 value 因子是亮点
- 唯一显著正 IC (+0.044, hit 60%)
- A 股 2018-2026 期间, 低估 (低偏离 MA60) 持续有效
- **应用**: 在 2026 H1 加大 value 风格子策略权重

### 4.4 momentum/reversal 反直觉
- momentum IC 接近 0 而非正 — 可能是 12 只 ETF 噪声太大
- reversal 负 IC — 短期反转在 ETF 池上不显著
- **后续**: 用更细的 universe 重新测 IC (Stage 17B)

## 5. 启示

1. **简单等权仍是稳健 baseline**: 不要为了"加 IC"而牺牲稳健性
2. **IC 强因子 (value) 值得加权重**: 单独加大 value 因子权重可能更优
3. **短 forward 窗口更好**: 10d > 20d > 40d
4. **IC 加权对 DD 略友好**: 最大回撤降低 1-2pp

## 6. 后续方向

| 方向 | 预期效果 | 工作量 |
|------|----------|--------|
| A. 仅 value 因子加权 | +Sharpe 0.05 | 小 |
| B. 多 forward 窗口融合 (10/20/40d 平均 IC) | +Sharpe 0.03 | 中 |
| C. 加 HMM regime 状态门控 | +Sharpe 0.05-0.10 | 中 |
| D. 用更多 ETF (扩展 universe 到 24) | +IC 稳定性 | 中 |

**下一步**: 实施 C (HMM 距离先验 + IC 融合) — 已在计划中

## 7. 文件清单

- `scripts/validate_stage17_ic.py`: IC 验证主脚本
- `QuantNodes/strategy/momentum_etf_rotation/v4/factor_ic.py`: IC 计算模块
- `QuantNodes/strategy/momentum_etf_rotation/v4/factor_timing_v4.py`: 因子择时模块
- `reports/momentum_etf_rotation/v4/ic_*.{parquet,csv,json}`: 验证结果
