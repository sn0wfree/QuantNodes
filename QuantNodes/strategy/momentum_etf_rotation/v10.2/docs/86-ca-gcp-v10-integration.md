# 86 · v10.2 真 v10 主体接入 (Real v10 Integration)

## 摘要

前 3 个 phase 用 mock momentum 信号验证 CA-GCP 框架可行。本 phase 用**真 v10 主体**（`scheme_e_hybrid`）替换 mock，对比 v10 vs v10.2 的真实差距。

## 1. 集成架构

```
真 v10 主体（不改）
   ↓ scheme_e_hybrid(): 5 宏观因子 → 目标权重
v10_weights (pd.Series)
   ↓
ca_gcp_risk_filter(v10_weights, intervals, rules)
   ↓ 黄/红灯触发减仓
adjusted_weights
```

## 2. v10 调研

`QuantNodes/strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py`:
- `scheme_e_hybrid()`: 简化版 v10，混合方案（A/B/C/D），输出 daily weights
- `metrics()`: 计算 Sharpe / MaxDD / Calmar

注意：v10 用 NAV 而非 returns，需要数据格式转换。

## 3. 数据流

| 步骤 | 输入 | 输出 |
|------|------|------|
| 1. 加载 ETF NAV | `data/real/etf_nav_*.parquet` | (T, N) NAV |
| 2. 计算 returns | NAV → pct_change | (T-1, N) returns |
| 3. 跑 v10 | returns → scheme_e_hybrid | daily weights (T, N) |
| 4. 跑 CA-GCP | returns → predict_fast | intervals, stress |
| 5. 风控 hook | v10 weights + intervals | adjusted weights |
| 6. 计算 metrics | adjusted weights + realized | Sharpe, MaxDD |

## 4. 实验设计

**时间切分**：
- Train: 2018-01 ~ 2021-04 (v10 训练 + CA-GCP 拟合)
- Calib: 2021-04 ~ 2022-04 (CA-GCP 校准)
- Test: 2022-04 ~ 2023-04 (对比期)

**对比**：
| 策略 | Sharpe | MaxDD | Calmar | 备注 |
|------|--------|-------|--------|------|
| v10 (scheme_e_hybrid) | ? | ? | ? | 真主体 |
| v10.2 (v10 + CA-GCP) | ? | ? | ? | 加风控层 |

**预期**：v10.2 的 MaxDD 减少（风控层在 2022-10/11 减仓），Sharpe 类似或更好。

## 5. 关键风险

| 风险 | 缓解 |
|------|------|
| v10 接口复杂（CLI 而非函数） | 用 `scheme_e_hybrid` 函数而非 `main()` |
| v10 需要 NAV 而 CA-GCP 用 returns | 中间转换层 |
| scheme_e_hybrid 可能很慢 | 仅在 test 期调用（252 天） |
| CA-GCP predict 仍 ~3s/天 | 用 `predict_fast` |

## 6. 输出

| 文件 | 角色 |
|------|------|
| `experiments/08_v10_2_backtest.py` | 重写（真 v10） |
| `data/results/v10_2_real_comparison.csv` | 真 v10 vs v10.2 对比 |
| `data/results/v10_real_nav.csv` | v10 净值曲线 |
| `data/results/v10_2_real_nav.csv` | v10.2 净值曲线 |
| `docs/82-ca-gcp-pool-size-test.md` | 更新（加真 v10 结果） |

## 7. 局限

- 仅 1 个 test 期（2022-04 ~ 2023-04），未 walk-forward
- scheme_e_hybrid 是 v10 简化版，非完整 5 策略加权
- CA-GCP fit 用 train+calib 联合，预测 test 期（一次性）

## 8. 论文参考

无（这是工程集成，非论文内容）。