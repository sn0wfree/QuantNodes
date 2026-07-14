# v7.6 TV-PR 参数敏感性测试 — 设计文档

> **目的**: 通过系统性参数扰动, 量化 v7.6 TV-PR 对超参数的敏感性, 判断过拟合嫌疑的严重程度
> **创建时间**: 2026-07-14
> **相关文件**: `reports/momentum_etf_rotation/v7_6_sensitivity_*.csv`, `v7_6_sensitivity_report.md`

## 一、背景

### 1.1 当前 v7.6 TV-PR 状态

**主结论**:
- OOS Calmar 1.685 (日频含成本)
- OOS Sharpe 1.46
- OOS 年化 25.86%
- **起点 CV% 50.2% → FAIL** (阈值 25%)

**核心疑虑**: OOS 表现是否真实, 还是参数过拟合?

| 参数 | 当前值 | 来源 |
|---|---|---|
| `lambda_tv` | 0.01 | **CV 校验调出** |
| `lambda_l1` | 0.001 | **CV 校验调出** |
| `window_size` | 52 周 | 经验选择 |
| `rho` | 1.0 | ADMM 默认值 |
| `min_history` | 52 周 | 经验选择 |

**疑点**: `lambda_tv=0.01` 是论文推荐范围 (0.05-0.10) 的下限之 5-10×, 表明我们的 CV 可能在训练集上选过。

### 1.2 Cui 2025 论文推荐范围

| 参数 | 论文 Table 4 | 我们的选择 | 偏离倍数 |
|---|---|---|---|
| `lambda_tv` | 0.05-0.10 | 0.01 | **5-10× 过小** |
| `lambda_l1` | 0.001-0.005 | 0.001 | 边缘一致 |
| `window_size` | 全样本 expanding | 52 周 rolling | 差异较大 |

## 二、过拟合判据

### 2.1 过拟合评级

| 等级 | 表现 | 结论 |
|---|---|---|
| **绿色 (低)** | 单一参数偏离 50%, CV% 退化 < 30%; 多段 hold-out Calmar 接近 | 真实可靠, 可锁定 |
| **黄色 (中)** | 偏离 30-50% 退化; 部分起点退化 | 需缩小 λ 范围重测 |
| **红色 (高)** | 偏离 > 50% 退化; 起点 2022 Calmar < 0; bootstrap std/mean > 50% | 严重过拟合, 重新设计 |

### 2.2 五个判据维度

| 维度 | 测试 | 判据 (红色) |
|---|---|---|
| A. 参数敏感度 | Phase 1 | lambda_tv 0.01 → 0.05, OOS Calmar 退化 > 50% |
| B. 起点稳定性 | 现有数据 | CV% > 35% |
| C. Hold-out 时间衰减 | Phase 2 | 2024-2026 与 2022-2024 Calmar 比 < 0.5 |
| D. Bootstrap 稳定性 | Phase 3 | Calmar std/mean > 50% |
| E. 数据鲁棒性 | Phase 4 | 20% X mask 下 Calmar 退化 > 50% |

## 三、测试设计 (10 阶段)

### Phase 0: 论文默认 λ 快速验证 (5 min, 优先级最高)

**目的**: 5 分钟内确认是否过拟合

```python
# scripts/v7_6_paper_default_check.py
lambda_tv = 0.05  # 论文默认 (Table 4)
lambda_l1 = 0.001  # 论文默认
window_size = 52
# 跑 OOS 2022-2026 + 起点依赖
```

**判据**:
- 论文默认 OOS Calmar ≥ 1.0 → 低过拟合
- 0.5-1.0 → 中度过拟合
- < 0.5 → **严重过拟合**

### Phase 1: 单参数敏感性 (30 min)

```python
# scripts/v7_6_sensitivity_single.py
# 14 组实验 (单参数轮换)
```

| 参数 | 测试值 |
|---|---|
| `lambda_tv` | {0.005, 0.02, 0.05, 0.10} |
| `lambda_l1` | {0.0001, 0.0005, 0.002, 0.005} |
| `window_size` | {26, 78, 104, 156} |
| `rho` | {0.5, 2.0, 5.0} |

**输出**: `reports/momentum_etf_rotation/v7_6_sensitivity_single.csv`

**列**: `param | value | full_calmar | oos_calmar | oos_sharpe | oos_dd | start_cv | 退化%`

### Phase 2: Hold-out 多段测试 (15 min)

```python
# scripts/v7_6_holdout_test.py
# 冻结当前 λ, 测试 3 段:
```

| 测试段 | 时间范围 | 长度 |
|---|---|---|
| 段 A | 2022-01-01 ~ 2024-06-30 | 2.5 年 |
| 段 B | 2024-07-01 ~ 2026-06-30 | 2 年 |
| 段 C | 2023-01-01 ~ 2024-12-31 | 2 年 |

**判据**: 三段 Calmar 接近 (差异 < 30%), 最近段退化 < 50% → 稳定

### Phase 3: Bootstrap 稳定性 (45 min)

```python
# scripts/v7_6_sensitivity_bootstrap.py
# 100 次 Y 重采样 (random_state=42-141)
# 固定 λ 为当前值
```

**3 个实验**:
1. **Y bootstrap** (100 次重采样)
2. **subset bootstrap** (随机抽 80% 资产 × 50 次)
3. **残差 bootstrap** (Y = pred + residual, 重采样 residual)

**判据**: bootstrap Calmar std/mean < 30% → 稳定

### Phase 4: 缺失数据扰动 (30 min)

```python
# scripts/v7_6_sensitivity_missing.py
# rates = [0.05, 0.10, 0.20]
# 每个 rate × 5 次 (random_state=42)
```

**判据**: 20% 缺失下 Calmar 退化 < 40% → 鲁棒

### Phase 5: 构造层扰动 (20 min)

```python
# scripts/v7_6_sensitivity_construction.py
# 11 组实验
```

| 参数 | 测试值 |
|---|---|
| `top_n` | {5, 8, 15, 20} |
| `max_weight` | {0.15, 0.20, 0.30} |
| `vol_window` | {13, 52} |

### Phase 6: β_path 断点分析 (10 min)

```python
# scripts/v7_6_beta_stability.py
# 用现有 rolling_tvpr 输出
```

**输出**:
- `\|β[t] - β[t-1]\|` 分布 (均值, 分位数)
- 断点频率 (TV 罚项内部)
- β_t 各维度 std (跨时间)

**判据**: β_std 各维 CV > 100% → 部分维度拟合不稳

### Phase 7: 综合报告 (10 min)

```python
# scripts/v7_6_sensitivity_report.py
# 汇总所有 CSV → v7_6_sensitivity_report.md
```

**报告结构**:
1. 执行摘要 (绿色/黄色/红色评级)
2. 每个 Phase 关键数据表
3. 5 维度判据汇总
4. 过拟合嫌疑评级
5. 建议下一步

### Phase 8: λ_tv=0.05 默认值验证 (5 min)

与 Phase 0 相同 (双重确认)。

### Phase 9: 决策讨论

依据敏感性测试评级讨论下一步。

## 四、产出文件清单

```
reports/momentum_etf_rotation/
├── v7_6_sensitivity_paper_default.csv       # Phase 0 + 8
├── v7_6_sensitivity_single.csv              # Phase 1
├── v7_6_holdout_test.csv                    # Phase 2
├── v7_6_sensitivity_bootstrap.csv           # Phase 3
├── v7_6_sensitivity_missing.csv             # Phase 4
├── v7_6_sensitivity_construction.csv        # Phase 5
├── v7_6_beta_stability.csv                  # Phase 6
└── v7_6_sensitivity_report.md               # Phase 7 (主报告)
```

## 五、执行模式

**分批执行**, 每 Phase 单独跑, 跑完汇报结果:

1. **Phase 0 + 8** (5 min, 优先级最高) - 最快拿到核心证据
2. **Phase 1** (30 min) - 单参数敏感性
3. **Phase 2** (15 min) - 多段 hold-out
4. **Phase 3-7** (115 min) - 综合分析

总耗时 ~3 小时。

## 六、风险评估

| 风险 | 概率 | 缓解 |
|---|---|---|
| Phase 3 bootstrap 慢 | 中 | 100 次替代 500 次 |
| Phase 4 mask 引入 NaN bug | 低 | 用固定 random_state |
| 全红色 (不回退, 用户决策) | 中 | 进入"修改设计"讨论 |

## 七、过拟合应对 (如果需要)

**若 Phase 0 + 1 + 2 都红色**:

| 修改 | 内容 |
|---|---|
| 简化 v7.6 | 去掉 λ_cv, 用论文默认 λ + expanding window |
| λ 范围约束 | λ_tv 限定 [0.05, 0.10] |
| 降低 K | 用 5-K 模型而非 20-K |
| 保留 ensemble | v7.6 作为子信号, 与 v1.0 配对 |

## 八、版本控制

本测试每完成一个 Phase 单独 git commit, 便于回溯。

```bash
git add scripts/v7_6_sensitivity_*.py reports/momentum_etf_rotation/v7_6_sensitivity_*.csv
git commit -m "test: v7.6 sensitivity - Phase N - <描述>"
```

---

**生成时间**: 2026-07-14
**责任人**: QuantNodes Agent
**关联**: docs/39-v7_6_tvpr.md, reports/momentum_etf_rotation/v7_6_validation.md
