# 策略迭代体系 (Strategy Versioning System)

> 制定日期: 2026-07-08
> 最后更新: 2026-07-14
> 适用: QuantNodes 动量 ETF 轮动策略
> 当前版本: **v1.0** (Stage 12A 锁定) / **v6.2** (研究版本) / **v7.3** (宏观子策略)

---

## 1. 版本号体系 (SemVer for Strategies)

采用类似 SemVer 但适配策略迭代的三段式:

```
v{Major}.{Minor}.{Patch}
│       │       │
│       │       └─ Patch: 微调, 不改变默认行为
│       │           例: v1.0.1 - 修复 VT 计算边界 bug
│       │
│       └─ Minor: 新增可选配置 / 新指标
│           例: v1.1 - 新增 momentum_fused_weight 选项
│           例: v1.2 - 新增 Ledoit-Wolf 协方差估计
│
└─ Major: 架构变更 / 破坏性接口 / 核心算法重写
    例: v2.0 - 从月度调仓改为日度调仓
    例: v2.0 - 加入 ML 仓位管理
```

---

## 2. 版本历史

### v0.x (v1.0 之前的研究阶段)

| 版本 | 内容 | Calmar | 状态 |
|------|------|--------|------|
| v0.0 | Stage 8 baseline | 0.78 | 已弃用 |
| v0.1 | + Stage 9-C (VT) | 1.00 | 备选 (激进) |
| v0.2 | + Stage 9-B (TF) | 0.88 | 备选 (仅趋势过滤) |
| v0.3 | + Stage 13 (Cost) | 0.98 | 备选 (含交易成本) |
| v0.4 | + Stage 12A (hybrid) | 1.17 | 备选 (无 VT) |

### v1.0 (当前锁定)

| 属性 | 值 |
|------|---|
| Calmar | **1.60** |
| DD | -3.93% |
| Ann | 6.28% |
| OOS Calmar | 0.84 |
| 测试 | 142 PASS |
| Git commit | 见 `git log` |

**特征**:
- 混合动量打分 (price + slope_r2)
- 波动率目标 (TV tv=0.15)
- 交易成本 (5bp+10bp)
- 风险厌恶型配置

### v1.x 计划 (Roadmap)

| 版本 | 内容 | 预期 Calmar | 状态 |
|------|------|-------------|------|
| v1.1 | Ledoit-Wolf 协方差估计 | ~1.55-1.65 | 调研完成 |
| v1.2 | 风险平价 (RP) 加权 | ~1.50-1.70 | 调研完成 |
| v1.3 | RSRS 择时 (需 high/low 数据) | ~1.65-1.75 | 等待数据 |
| v1.4 | HMM 重做 (用 Ledoit-Wolf) | ~1.60-1.80 | 待启动 |
| v1.5 | 多策略组合 (动量 + 均值回归) | 待评估 | 待启动 |

### v6.2 (量价族 + IC 加权 + 因子正交化)

| 属性 | 值 |
|------|---|
| Calmar (不扣成本) | 0.4449 |
| Calmar (扣成本) | **0.3310** |
| OOS Calmar (5-fold mean) | 1.512 (历史, 未扣成本) |
| 起点 CV% | **56.9%** (阈值 25%, FAIL) |
| 状态 | ⚠️ **研究版本** |

**特征**:
- 11 量价因子 + IC 加权 + Gram-Schmidt 正交化
- 逆波动率加权 (v5.1.1)
- 交易成本 (5bp+10bp) — Stage 30 新增

**过拟合问题**:
- 交易成本影响: -25.6%
- 起点依赖: CV% 56.9% 远超 25% 阈值
- Fold 4 (2023.7-2024) 严重退化: v6.2 Calmar 0.636 vs v6.1 1.893

### v7.3 (宏观子策略)

| 属性 | 值 |
|------|---|
| Calmar (OOS 2023+) | 0.620 |
| Calmar (combo 50/50) | **1.210** |
| 状态 | ⭐ **LOCKED** |

**特征**:
- Symmetry + Bootstrap-Lasso + FactorRiskParity
- 13 INDEX_COLS + 9 macro factors
- 交易成本 (5bp+5bp)

### v7.3 + v6.2 combo

| 属性 | 值 |
|------|---|
| Calmar (OOS 2023+) | **1.210** |
| 状态 | ⭐ **current best** |

**特征**:
- v6.2 行业轮动 + v7.3 宏观配置
- 权重 50/50 固定
- 依赖 v6.2 扣成本后 Calmar 0.3310, 需重新评估

### v2.0 计划 (远期)

| 版本 | 内容 |
|------|------|
| v2.0 | 数据源升级 (Wind/Choice) + 完整实盘 |
| v2.1 | ML 仓位管理 (强化学习) |
| v2.2 | 多周期组合 (日频 + 月频) |

---

## 3. 版本升级流程 (Promotion Criteria)

### 3.1 Minor 版本 (v1.0 → v1.1)

**Go 条件 (全部满足)**:
- [ ] 全段 Calmar ≥ 当前版本 (不退化)
- [ ] 或 DD 改善 ≥ 5% (即使 Calmar 略降)
- [ ] OOS Calmar > 0.5
- [ ] 所有原有测试通过
- [ ] 至少 1 个新测试覆盖新功能
- [ ] CHANGELOG.md 更新
- [ ] 1 次 git commit
- [ ] v1.x 推荐配置文档更新

**No-Go 条件 (任一触发)**:
- [ ] 全段 Calmar 退化 > 5%
- [ ] OOS Calmar 退化 > 20%
- [ ] 测试失败率 > 5%
- [ ] 任何原有功能回归

### 3.2 Major 版本 (v1.x → v2.0)

**Go 条件 (全部满足)**:
- [ ] 满足 Minor 版本所有 Go 条件
- [ ] 至少 1 个新模块 (>200 行)
- [ ] 新模块测试覆盖率 > 80%
- [ ] 与 v1.x 可在 config 中切换 (向后兼容)
- [ ] ARCHITECTURE.md 更新
- [ ] 完整 README 重写
- [ ] 旧 v1.x 测试仍 100% 通过

---

## 4. 版本锁定机制 (Version Pinning)

### 4.1 配置文件 `strategy_versions.py`

```python
# QuantNodes/strategy/momentum_etf_rotation/strategy_versions.py
"""策略版本锁定 - 用户可显式选择版本."""

from .portfolio import (
    RotationConfig, DiversificationCaps, TrendFilter, VolTargeting,
    CostModel, CovEstimator,
)


def v1_0() -> RotationConfig:
    """v1.0 锁定配置 (Stage 12A 验证)."""
    return RotationConfig(
        lookback=90, top_n=10,
        momentum_type="hybrid",
        momentum_fused_weight=0.5,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
        cost_model=CostModel(
            enabled=True, commission_bp=5, slippage_bp=10,
            impact_factor=0.1,
        ),
    )


def v0_1_vt_only() -> RotationConfig:
    """v0.1 备选 (激进路线, 仅波动率目标)."""
    return RotationConfig(
        lookback=90, top_n=10,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
    )


def v0_0_baseline() -> RotationConfig:
    """v0.0 baseline (Stage 8, 无任何增强)."""
    return RotationConfig(lookback=90, top_n=10)


VERSIONS = {
    "1.0": v1_0,
    "0.1": v0_1_vt_only,
    "0.0": v0_0_baseline,
}
```

### 4.2 使用方式

```python
# 方式 1: 直接调用版本函数
from QuantNodes.strategy.momentum_etf_rotation.strategy_versions import v1_0
cfg = v1_0()

# 方式 2: 通过版本字符串
from QuantNodes.strategy.momentum_etf_rotation.strategy_versions import VERSIONS
cfg = VERSIONS["1.0"]()

# 方式 3: 在回测中使用
from QuantNodes.strategy.momentum_etf_rotation import (
    run_rotation_backtest, BacktestConfig, DEFAULT_POOL,
)
from QuantNodes.strategy.momentum_etf_rotation.strategy_versions import v1_0

result = run_rotation_backtest(
    panel, DEFAULT_POOL, BacktestConfig(rotation=v1_0())
)
```

---

## 5. 版本对比表 (v0.0 → v1.0 → 计划 v2.0)

| 版本 | Calmar | DD | Ann | OOS Calmar | 推荐场景 |
|------|--------|-----|-----|-----------|----------|
| v0.0 baseline | 0.78 | -21.05% | 16.35% | 1.72 | 学术研究 / 理想化 |
| v0.1 +VT | 1.00 | -6.89% | 6.87% | 1.00 | 风险厌恶 (单 VT) |
| v0.3 +Cost | 0.98 | -6.94% | 6.83% | 1.00 | 含实盘成本 |
| **v1.0 hybrid+VT+Cost** | **1.60** | **-3.93%** | **6.28%** | 0.84 | **风险厌恶 (推荐)** |
| 计划 v1.1 +LW协方差 | ~1.55 | ~-4% | ~7% | ~0.85 | v1.0 改进 |
| 计划 v1.3 +RSRS | ~1.65 | ~-4% | ~7% | ~0.90 | v1.1 改进 |
| 计划 v2.0 +Wind | ~1.70 | ~-4% | ~7% | ~1.20 | 数据升级 |

---

## 6. 版本实验流程 (Minor 版本开发)

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: 实验设计 (基于 v1.0 baseline)                         │
│  - 确定改动点 (例: 新增协方差估计方法)                          │
│  - 设定成功标准 (Calmar ≥ v1.0, DD ≤ v1.0)                   │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│  Step 2: 沙盒实验 (sandbox/, 1-2 小时)                       │
│  - 合成数据验证算法正确性                                     │
│  - 单元测试 (覆盖率 > 80%)                                   │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│  Step 3: 集成 (向 RotationConfig 添加新配置项)                │
│  - 默认值保持向后兼容 (新选项默认关闭)                         │
│  - 数据流图 (画集成点, 验证顺序)                             │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│  Step 4: 真实数据验证                                         │
│  - 全段回测 (Calmar, DD, Ann)                                │
│  - OOS 段 (2024-2026)                                       │
│  - 横向对比 (与 v1.0 baseline)                                │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│  Step 5: 决策 (Go / No-Go)                                   │
│  - 满足 promotion criteria? → Go (升 v1.x)                    │
│  - 不满足 → 归档到 experiments/, v1.0 不变                    │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│  Step 6: 文档化与提交                                         │
│  - 更新 STRATEGY_VERSIONS.md (新版本)                          │
│  - 更新 strategy_versions.py (新版本函数)                     │
│  - 更新 CHANGELOG.md                                        │
│  - 更新 STAGE_SUMMARY.md                                    │
│  - git commit + tag (v1.x)                                  │
└────────────────────────────────────────────────────────────┘
```

---

## 7. 与现有研发流程 (DEV_WORKFLOW.md) 的关系

| DEV_WORKFLOW 阶段 | 对应版本开发 |
|-------------------|-------------|
| Stage 0: Idea Brief | 决定 v1.x 内容 |
| Stage 1: Research | 调研论文 / 库 |
| Stage 2: Sandbox | 合成数据 PoC |
| Stage 3: Implementation | 集成到 RotationConfig |
| Stage 4: Validation | 真实数据 + OOS |
| Stage 5: Decision | Go / No-Go (v1.x 升级?) |
| Stage 6: Documentation | 更新版本文档 |

**每个 v1.x 升级 = 一次完整 DEV_WORKFLOW 流程**

---

## 8. 实验归档规范

任何 No-Go 的实验必须归档到 `experiments/`, 命名规范:
```
experiments/
├── v1_1_xxx_failed.md         # v1.1 候选失败
├── v1_2_xxx_failed.md         # v1.2 候选失败
└── README.md                   # 索引
```

每个归档文件包含:
- 实验假设
- 实现方法
- 失败原因 (技术 / 数据 / 集成)
- 证据 (指标对比)
- 教训
- 可能的复苏条件

---

## 9. 立即可执行 (v1.0 落地)

1. ✅ 创建 `strategy_versions.py` (本文已设计)
2. ✅ 创建 `STRATEGY_VERSIONS.md` (本文)
3. ⏳ 创建 `CHANGELOG.md` (记录 v0.x → v1.0 的所有变化)
4. ⏳ 创建 `experiments/README.md` (索引)
5. ⏳ 在 `__init__.py` 导出 v1_0()
6. ⏳ 添加 v1.0 测试 (`test_v1_0_regression.py`)

---

## 10. 长期愿景 (v2.0+)

### 10.1 多策略框架

```
v2.0: 单一策略 → 策略组合
├── Strategy A: 动量 (v1.x)
├── Strategy B: 均值回归
├── Strategy C: 行业轮动
└── Meta-strategy: 风险平价 + 动态权重
```

### 10.2 数据源矩阵

```
v2.0 数据策略:
├── Wind (主力)
├── Choice (备份)
├── 自建 (验证)
└── 实时数据流
```

### 10.3 实盘验证

```
v2.0 实盘路线:
├── 模拟盘 (3 个月)
├── 小资金实盘 (3 个月, 10 万)
├── 中等实盘 (6 个月, 100 万)
└── 全量实盘 (持续)
```

---

**v1.0 已锁定. v1.1+ 迭代体系已设计. 等待 v1.1 启动.**