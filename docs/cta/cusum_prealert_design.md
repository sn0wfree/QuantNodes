# CTA 突变点检测 / 风险因子设计文档

> 最后更新: 2026-09-16
> 状态: 设计阶段（待 Phase 1 实现）
> 模块路径: `scripts/cta_changepoint/`

---

## 一、问题背景

### 1.1 当前 CTA 体系的缺口

CTA 主策略（E2-R+AU10% 组合）当前的风控体系以**被动响应**为主：

| 风控层 | 触发条件 | 类型 |
|---|---|---|
| 三级风控 | 已实现 DD / Vol | 反应型 |
| vol_target | realized_vol 上升 | 被动调节 |
| daily_cap | 单日 -3% | 硬止损 |
| emergency | 单日 -5% | 应急 |

**核心缺口**：
- 没有"预警型"机制（基于市场状态统计偏移提前 1-3 天提示）
- vol_target 与波动率突变功能重叠但缺乏离散阶跃
- 缺乏系统性风险（跨品种相关性）的独立监测
- 持仓集中度（HHI）的变化未被跟踪

### 1.2 文章 2 框架的指引

《从时序动量 Alpha 到可落地 PnL，还有几步？》的核心论点：

> "预测方向关注的是'价格要去哪'，测量状态是问：现在是什么情况？"

```
可交易Alpha = 统计Alpha × 信号效率 × 交易成本 × 拥挤度
             ────────  ────────  ────────  ────────
             回测覆盖   实盘捕获   成本留存   拥挤度扣减
```

**任一项归零，实盘归零**。当前 CTA 主策略对"拥挤度"和"系统性风险状态"缺乏量化监控。

### 1.3 与已有研究的关系

| 已有实现 | 关系 |
|---|---|
| `DCCRegimeOverlay` (v7) | **直接复用** DCC zscore 计算 |
| `_rapid_move_score` (AllWeather) | 设计参考（单变量 CUSUM 替代品） |
| `monitor/drift.py` (KS+Sharpe+DD) | **直接复用** DriftAlert 框架 |
| `ml_anomaly_v3` 设计 | **未来结合**（ML 是补充层） |
| `v7_6_beta_stability.py` (β 断点) | 设计参考（断点检测思想） |

**代码库无任何 CUSUM/BOCPD/PELT 实现**（`grep` 0 命中）。这是新空白。

---

## 二、设计原则

### 2.1 解耦测量与决策

**风险因子 = 测量工具，不是决策工具**：

```
市场状态 → 风险因子（CUSUM）→ risk_factor ∈ [0, 1]
                                    ↓
                            仓位管理器（消费风险因子）
                                    ↓
策略仓位 × 风险调整 → 最终仓位
```

**MVP 阶段**：风险因子仅输出**信号**，**不动仓位**。下游消费方决定如何使用。

### 2.2 无反馈循环

**禁止**用策略自身 NAV 作为输入（会导致触发-减仓-触发的循环）。

**正确做法**：监测**外部市场状态变量**：
- 跨品种相关性（DCC zscore）— 系统性风险
- 持仓集中度（HHI）— 策略脆弱性
- VIX 百分位 — 外部冲击代理（可选）

### 2.3 不替代现有体系

风险因子**补充而非替代**：
- 三级风控：兜底（已实现 DD/vol）
- vol_target：常态调节（连续缩放）
- **风险因子**：预警（离散阶跃，1-3 天提前）

三者**正交并行**，互不干扰。

---

## 三、核心设计

### 3.1 监测对象（MVP: DCC + HHI）

#### 第一维度：跨品种相关性（DCC zscore_mean）

**物理含义**：所有品种同时下跌时，相关性急剧上升 → 系统性危机预警

**数据基础**：
- 已有研究：`QuantNodes/strategy/momentum_etf_rotation/v7/dcc_regime_overlay.py`
- 阈值经验：dcc_zscore_mean > 1.5 触发（`dcc_regime_overlay.py:90`）
- 关键发现（`docs/research_history/04_V7_V10.md:669`）：
  > "相关性突变（dcc_zscore_mean > 1.5）是危机预警的最稳定信号"

**CUSUM 应用**：
```python
S_t^DCC = max(0, S_{t-1}^DCC + dcc_zscore_t - k_dcc)
trigger when S_t^DCC > h_dcc
```

**推荐参数**：
- `k_dcc = 0.5`（reference: 期望最小可检测偏移）
- `h_dcc = 3.0`（threshold: ARL ≈ 100 天）
- ARL（Average Run Length）= 触发之间的平均天数

#### 第二维度：持仓集中度（HHI 指数）

**物理含义**：策略仓位过度集中 → 脆弱性上升

**HHI 定义**：
```python
HHI_t = Σ(weight_i_t)^2
```
- HHI = 1 表示全部押注单一品种
- HHI = 1/N 表示完全分散

**数据来源**：每日各品种持仓权重（需新建数据提取）

**CUSUM 应用**：
```python
S_t^HHI = max(0, S_{t-1}^HHI + ΔHHI_t - k_hhi)
trigger when S_t^HHI > h_hhi
```

**推荐参数**：
- `k_hhi = 0.01`（日变化 1% HHI 作为最小可检测偏移）
- `h_hhi = 0.05`（累积 5% 触发）

### 3.2 综合风险分数合成

```
risk_factor = max(f(S_t^DCC), f(S_t^HHI))
其中 f(S) = clip(S / h, 0, 1)  # 归一化到 [0, 1]
```

**采用 max 而非平均的原因**：
- 任一维度触发即代表风险上升
- 平均会掩盖单一维度的极端信号
- 与"组合级风险 = 各项最大风险"的直觉一致

### 3.3 风险等级映射

| risk_level | 含义 | risk_factor 范围 | 建议响应（MVP 不执行） |
|---|---|---|---|
| 0 | 正常 | [0.0, 0.3] | 无 |
| 1 | 关注 | [0.3, 0.6] | 告警 |
| 2 | 警告 | [0.6, 0.8] | 告警 + 人工审视 |
| 3 | 危险 | [0.8, 1.0] | 强制告警 + 建议减仓 |

---

## 四、模块结构

### 4.1 目录布局

```
scripts/cta_changepoint/
├── __init__.py
├── cusum_detector.py            # 核心 CUSUM 算法（纯函数）
├── risk_factor.py               # 风险因子计算（主入口）
├── data_loaders.py              # 数据加载（DCC, HHI, VIX）
├── risk_config.py               # 参数配置 dataclass
└── README.md                    # 模块文档

tests/cta/
├── test_cusum_detector.py       # CUSUM 单元测试
├── test_risk_factor.py          # 风险因子集成测试
└── test_risk_factor_backtest.py # 历史回测验证

docs/cta/
├── cusum_prealert_design.md     # 本文档
└── cusum_prealert_results.md    # Phase 2 验证报告

reports/cta/
└── risk_factor_YYYY-MM-DD.csv   # 运行时输出
```

### 4.2 核心 API 设计

#### `cusum_detector.py`：纯函数层

```python
def cusum_detector(
    x: pd.Series,
    mu: float = 0.0,
    k: float = 0.5,
    h: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """标准 CUSUM 单变量检测.

    Args:
        x: 输入时间序列
        mu: in-control 均值
        k: reference value（最小可检测偏移）
        h: trigger threshold

    Returns:
        (S_t, triggered): 累计统计量 + 触发布尔序列
    """
```

**算法**：
```python
S_t = max(0, S_{t-1} + (x_t - mu - k))
triggered_t = S_t > h
```

#### `risk_config.py`：参数配置

```python
@dataclass
class CUSUMConfig:
    """CUSUM 风险因子配置."""

    enabled: bool = True

    # DCC CUSUM 参数
    k_dcc: float = 0.5
    h_dcc: float = 3.0
    mu_dcc: float = 0.0

    # HHI CUSUM 参数
    k_hhi: float = 0.01
    h_hhi: float = 0.05
    mu_hhi: float = 0.0

    # 风险等级阈值
    threshold_level1: float = 0.3
    threshold_level2: float = 0.6
    threshold_level3: float = 0.8
```

#### `risk_factor.py`：主入口

```python
@dataclass
class RiskFactor:
    """单日风险因子输出."""

    timestamp: pd.Timestamp
    dcc_zscore: float
    dcc_cusum: float
    dcc_triggered: bool
    hhi: float
    hhi_cusum: float
    hhi_triggered: bool
    risk_factor: float       # [0, 1]
    risk_level: int          # {0, 1, 2, 3}


def compute_risk_factor(
    dcc_zscore: pd.Series,
    hhi: pd.Series,
    config: CUSUMConfig,
) -> pd.DataFrame:
    """计算每日风险因子.

    Args:
        dcc_zscore: 跨品种相关性 zscore 时间序列
        hhi: 持仓集中度时间序列
        config: CUSUM 配置

    Returns:
        DataFrame with columns:
        - timestamp, dcc_zscore, dcc_cusum, dcc_triggered,
          hhi, hhi_cusum, hhi_triggered,
          risk_factor, risk_level
    """
```

### 4.3 数据流

```
┌─────────────────────────────────────────────┐
│ 数据源                                       │
│  ├── DCC 因子（复用 v7 实现）                 │
│  ├── 持仓快照（新建提取）                     │
│  └── VIX（已有 parquet, 可选）                │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ scripts/cta_changepoint/                    │
│  data_loaders.py → 加载并对齐数据            │
│  cusum_detector.py → 计算 S_t 和触发         │
│  risk_factor.py → 合成 risk_factor 和等级    │
└────────┬────────────────────────────────────┘
         │ 输出: RiskFactor DataFrame
         ▼
┌─────────────────────────────────────────────┐
│ 下游消费方                                   │
│  ├── DriftAlert（monitor/drift.py）          │
│  ├── 跟踪报告（reports/cta/risk_factor_*）   │
│  └── 未来：仓位管理器 / 风险预算              │
└─────────────────────────────────────────────┘
```

---

## 五、参数设计依据

### 5.1 DCC CUSUM 参数

| 参数 | 推荐值 | 依据 |
|---|---|---|
| `k_dcc` | 0.5 | reference value：日均相关性偏移 0.5σ |
| `h_dcc` | 3.0 | threshold：对应 ARL(in-control) ≈ 100 天 |

**依据**：
- 已有研究阈值 1.5（`DCCRegimeOverlay`）用于**单日触发**
- CUSUM 用累积偏移，所以单日阈值需要乘以 √N（N ≈ 4，即 4 天累积 = 2 × 1.5 = 3.0）
- ARL 100 天意味着正常情况下每月误报 ≤ 0.3 次

### 5.2 HHI CUSUM 参数

| 参数 | 推荐值 | 依据 |
|---|---|---|
| `k_hhi` | 0.01 | reference value：日变化 1% HHI |
| `h_hhi` | 0.05 | threshold：累积 5% HHI 变化触发 |

**依据**：
- 完全分散组合 HHI = 1/N ≈ 0.014（72 品种）
- 完全集中组合 HHI = 1.0
- 5% HHI 变化 = 从分散（HHI=0.014）到中等集中（HHI=0.064）

### 5.3 风险等级阈值

| 等级 | 阈值 | 设计理念 |
|---|---|---|
| 0 → 1 | 0.3 | 轻度关注（30% 触发线） |
| 1 → 2 | 0.6 | 中度警告（60% 触发线） |
| 2 → 3 | 0.8 | 严重危险（80% 触发线） |

---

## 六、实施路径

### Phase 1: 原型（1-2 天）

**任务清单**：
- [ ] 实现 `cusum_detector.py`（纯函数）
- [ ] 实现 `data_loaders.py`（DCC + HHI 加载）
- [ ] 实现 `risk_factor.py`（综合风险分数）
- [ ] 实现 `risk_config.py`（参数配置）
- [ ] 单元测试（5+ 用例）
- [ ] ruff + pytest 通过

**验收标准**：
- 单元测试覆盖率 > 90%
- ruff 零错误
- 性能测试 < 100ms / 1000 天

### Phase 2: 历史验证（3-5 天）

**任务清单**：
- [ ] 历史回测：2018-2025 全样本
- [ ] 危机切片测试：
  - 2020-03 新冠暴跌
  - 2022-02 俄乌冲突
  - 2024-09 政策转折
- [ ] 报告指标：
  - 风险因子分布（0/1/2/3 等级占比）
  - 触发准确率（危机前 vs 真实危机）
  - 误报率（正常时段）
- [ ] Go/No-Go 评估

**Go/No-Go 门槛**：
| 门槛 | 通过标准 | 重要性 |
|---|---|---|
| G1: 预警价值 | 风险因子在历史危机前 1-3 天进入 2 级以上 | **必过** |
| G2: 误报率 | 正常时段 95% 以上为 0 级 | **必过** |
| G3: 危机命中 | 已知危机时段命中率 ≥ 70% | 必须 |
| G4: 性能 | 计算 < 100ms / 1000 天 | 重要 |

3/4 通过即集成，G1+G2 必过。

### Phase 3: 监控集成（1-2 天）

**任务清单**：
- [ ] DriftAlert 接入（monitor/drift.py）
- [ ] SQLite 持久化（risk_factor_history 表）
- [ ] 跟踪报告 `reports/cta/risk_factor_YYYY-MM-DD.csv`
- [ ] 验证报告 `docs/cta/cusum_prealert_results.md`
- [ ] **不集成到仓位决策**（仅监控）

**纪律**：MVP 只做风险因子输出，**不动现有仓位逻辑**。

---

## 七、风险与边界

### 7.1 必须遵守的边界

| 不要做 | 原因 |
|---|---|
| ❌ 引入 `ruptures` 依赖 | MVP 不需要 |
| ❌ 实现 BOCPD/PELT | 留 P1 研究 |
| ❌ 用策略自身 NAV 作为输入 | 反馈循环 |
| ❌ 直接控制仓位 | MVP 仅监控 |
| ❌ 实时推送（Slack/邮件） | 走日频管道 |
| ❌ 修改 `compute_risk_overlay` | 仅新增，不改 |

### 7.2 失败回滚方案

若 Phase 2 不通过 Go/No-Go：
- 风险因子模块**保留代码**但 `enabled=False`
- DriftAlert 接入可独立关闭
- 失败仅损失预警能力，不损失策略本身

### 7.3 与 ZK_VOL_ADJ_CFG 的对照

ZK_VOL_ADJ_CFG 处理方式（`QuantNodes/strategy/cta/strategies/base.py:36`）：
```python
ZK_VOL_ADJ_CFG = {
    "enabled": False,  # 回滚关闭
    ...
}
```

CUSUM 风险因子采用**相同的处理模式**：
```python
@dataclass
class CUSUMConfig:
    enabled: bool = True  # 验证通过后启用
    ...
```

---

## 八、与现有体系的关系

### 8.1 三层架构

```
┌─────────────────────────────────────────┐
│ L1: 风险因子（本模块新增）              │
│     - 输入: DCC zscore, HHI             │
│     - 输出: risk_factor ∈ [0, 1]       │
│     - 类型: 预警型（1-3 天提前）        │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ L2: vol_target + 三级风控（已有）       │
│     - vol_target: 连续缩放              │
│     - 三级风控: 离散减仓                │
│     - 类型: 被动响应                    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ L3: daily_cap + emergency（已有）       │
│     - 硬止损                            │
│     - 类型: 紧急熔断                    │
└─────────────────────────────────────────┘
```

### 8.2 与 v3 ML 异常检测的关系

| 信号源 | 类型 | 决策权重 |
|---|---|---|
| 三级风控 | 反应型 | 主力 |
| **CUSUM 风险因子（本模块）** | **预警型** | **辅助** |
| ML v3 R 分数 | 异常型 | 辅助 |
| vol_target | 被动调节 | 主力 |

**三者互补不替代**：
- 三级风控兜底（已实现 DD/vol）
- CUSUM 给"快信号"（市场状态统计偏移）
- ML v3 给"准信号"（分布偏离）

---

## 九、未来扩展（P1+ 暂不实施）

### 9.1 监测对象扩展

- 单品种 CUSUM（如黄金、铜、jmm_coke）
- 跨资产相关性突变（DCC regime overlay 完整集成）
- VIX/GVZ 突变（外部信号）

### 9.2 下游使用扩展

- 风险预算：risk_factor > 0.6 时降低总仓位
- 动态杠杆：risk_factor 与 vol_target 合成
- 跨策略比较：不同策略的风险因子横向对比
- 实时通知：Slack/钉钉/邮件 webhook

### 9.3 算法升级

- BOCPD 多变量 CUSUM
- PELT 离线归因
- ML 异常检测 v3 实现

---

## 十、决策记录

### 10.1 已决策

| 决策项 | 选择 | 依据 |
|---|---|---|
| 监测对象 | DCC + HHI（暂不加 VIX） | 系统性 + 策略性互补，无反馈循环 |
| 算法 | CUSUM 单变量 | 简单、可解释、低延迟 |
| 输出格式 | risk_factor ∈ [0,1] + risk_level ∈ {0,1,2,3} | 易消费 |
| 下游使用 | **仅监控 + 告警**，不动仓位 | MVP 验证基础可行性 |
| 与 B3 关系 | 独立集成（接当前生产链 -2/-3/-15） | 解耦 |
| 与 ML v3 关系 | 仅 CUSUM，先验证再启动 ML | Sequential 验证 |

### 10.2 待决策（Phase 1 前需确认）

| 待决策项 | 影响 |
|---|---|
| DCC zscore 数据来源 | 是否需重算 / 复用 v7 产物 |
| HHI 持仓快照来源 | 新建数据提取脚本 |
| 风险因子持久化方式 | SQLite / CSV / parquet |

---

## 十一、参考依据

### 11.1 文章

- 《CTA 因子挖掘的两条路，选时序还是截面？》— 波动率相位差框架
- 《从时序动量 Alpha 到可落地 PnL，还有几步？》— 三层六段 + 四要素

### 11.2 代码库内已有研究

- `QuantNodes/strategy/momentum_etf_rotation/v7/dcc_regime_overlay.py` — DCC 实现
- `QuantNodes/strategy/allweather/neg_trend_daily.py:146` — `_rapid_move_score` 设计参考
- `QuantNodes/monitor/monitor/drift.py` — DriftAlert 框架
- `docs/research_history/04_V7_V10.md:669` — DCC 危机预警实证
- `docs/gold/ml_anomaly_v3.md` — ML 异常检测（未来结合）

### 11.3 外部文献（文章引用）

- Moskowitz, Ooi, Pedersen (2012) — 时间序列动量
- Erb, Harvey (2006) — 商品截面基差
- Goulding et al. (2023) — 趋势生命周期
- Kelly et al. (2019) — 因子剥离方法论
- Antulov-Fantulin et al. (2024) — 状态识别策略
- Gârleanu, Pedersen (2013) — 钝化处理
- Kyle, Obizhaeva (2016) — 市场深度约束
