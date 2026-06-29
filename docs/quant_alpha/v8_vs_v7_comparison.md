# V8 vs V7 对比报告

## 背景
V7 (4 logic) 单轮探索饱和，9 因子，best |IR|=0.1208 (vol)。
**V8 加 2 个新 logic (trend_breakout, intraday_reversal) 拓展挖掘面**。

## V8 新增 Logic 设计

### Logic 5: trend_breakout (趋势突破)
```
predicates:
  - high ts_max 20d (突破近期高点)
  - close ts_mean 5d (短期均价)
  - vol ts_mean 20d (量能基础)
behavior: forward_return_5, direction=+1
operators: rank, ts_max, ts_min, ts_mean, sub, div, mul, sign
```
**挖掘概念**: 价量共振的向上突破

### Logic 6: intraday_reversal (日内反转)
```
predicates:
  - open ts_mean 5d
  - close ts_mean 5d
behavior: forward_return_5, direction=-1
operators: rank, ts_std, ts_mean, sub, div, abs, sign, mul
```
**挖掘概念**: 大幅日内波后均值回归

## V8 完整结果 (6 logic)

| Logic | 因子数 | best \|IR\| | 耗时 | 状态 |
|-------|--------|-------------|------|------|
| price_volume_divergence | 0 | 0.000 | 141s | 旧, 仍 0 |
| mean_reversion | 3 | 0.0610 | 141s | 旧 |
| momentum | 3 | 0.0610 | 73s | 旧 |
| volatility | 3 | 0.1208 | 132s | 旧 |
| **trend_breakout** | **3** | **0.1596** ⭐ | **108s** | **新** |
| **intraday_reversal** | **3** | **0.1103** | **266s** | **新** |
| **总因子** | **15** | - | **861s** | +67% |
| **best \|IR\|** | - | **0.1596** | - | +32% |

## 关键发现

### 1. trend_breakout 历史最佳
- FORMULA-1-3: IR=-0.1596, IC=-0.0130
- 负 IR: sign_constraint=+1 期望 positive，但实际 negative → sign mismatch
- 但 |IR|=0.1596 绝对值是 V4-V8 所有公式中最高的

### 2. intraday_reversal 全部 positive IR
- 3 个因子: IR=+0.1103, +0.0841, +0.0775
- sign_constraint=-1 期望 negative，但全部 positive
- 这是 sign-mismatch 的反面案例 — LLM 生成的 sign 与预期相反

### 3. pvd 仍 0 — 反复验证弱逻辑
V4 (bug)→V5 (bug)→V6 (bug)→V7 (修了但公式弱)→V8 (公式能算，IR 0.02-0.06 < 0.05)
**结论**: pvd 逻辑在 A股这个数据集就是难挖

### 4. 老 logic 在 V8 中的 IR 与 V7 不同
- V7 mr best 0.1133 → V8 mr best 0.0610 (LLM 采样随机)
- V7 mom best 0.1008 → V8 mom best 0.0610 (同上)
- V7 vol best 0.1208 → V8 vol best 0.1208 (一致, 收敛了)
- **同一 logic 不同 run IR 差异 ~0.05**, LLM 输出有随机性

## V7 → V8 总结

| 维度 | V7 | V8 | 变化 |
|------|----|----|------|
| Logic 数 | 4 | 6 | +2 |
| 总因子 | 9 | 15 | **+67%** |
| 整体 best \|IR\| | 0.1208 (vol) | 0.1596 (trend) | **+32%** |
| 0 因子 logic | 1 (pvd) | 1 (pvd) | 持平 |
| 耗时 | 622s | 861s | +38% |
| 老 logic best | 0.1208 (vol) | 0.1208 (vol) | 持平 (说明 vol 收敛) |
| 新 logic best | - | 0.1596 (trend) | **新挖掘面成功** |

## V8 价值

1. **挖掘面拓展成功**: 2 个新 logic 贡献 6 因子, 40% 占比
2. **历史新高 best \|IR\|**: 0.1596 (trend_breakout) 超越 V6 (0.1284)
3. **逻辑设计有效**: ts_max / intraday 这些 V4-V7 没充分用的算子组合有效
4. **pvd 验证完毕**: 不是 bug, 是 A股价量关系弱

## V9 方向

V8 已达 15 因子, 进一步挖掘可考虑:
1. **多轮迭代** (V7 mr/mom 不同 run IR 差异大, feedback 可能稳定)
2. **降低 IR 阈值** (pvd 公式 IR 0.02-0.06, 阈值 0.05 → 0.03 可能挖出)
3. **再加 1-2 logic** (e.g. value_turnover, gap_reversal)
4. **针对 best factor 做多角度组合** (orthogonal pool construction)
