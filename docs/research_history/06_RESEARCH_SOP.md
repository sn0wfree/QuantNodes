# 06 — 研发流程 SOP（Standard Operating Procedure）

> **范围**：从 V0-V10 提炼出的**可复用研发流程**
> **适用对象**：momentum_etf_rotation 后续策略 / cta / 期权 / 其他类策略
> **核心原则**：8 个阶段 + 5 道闸门 + 4 步 OOS 验证

---

## 一、SOP 总览（一张图）

```
数据 → IC → 因子 → 选股 → 加权 → 风控 → 组合 → OOS → 硬化
 │    │     │      │      │      │      │      │     │
 [1]  [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]
 │    │     │      │      │      │      │      │     │
OHLCV IC   宏观   动量   等权   VT    Vol-  5-fold Stage
Proxy 滚动 vs    + 反转  逆波   TF    parity walk- 32
池  IC    PV    + 行业  动率   SL    4策略 forward 硬化
缺  自相关 截面  ABC   IC    DCC   组合   expan- CV%
数  阈值   时序  子策  IC-IR  regime        ding  死代码
据  |IC|>   对数  略单  正交化 overlay        wind   清理
扰  0.05   变换  测                            ow    工厂
 动

5 道闸门：
1. 数据闸门   2. IC 闸门   3. 因子闸门   4. OOS 闸门   5. 硬化闸门

4 步 OOS 流程：
① 验证过拟合严重程度  ② 修复 off-by-one bug
③ expanding-window 消除 look-ahead  ④ CV% < 25% PASS
```

---

## 二、8 个阶段详细 SOP

### 阶段 1：数据准备（[详细文档](./01_DATA_FOUNDATION.md)）

#### 1.1 数据源选择

| 数据类型 | 推荐源 | 备选 |
|---------|--------|------|
| ETF close | Tencent | Wind/Choice |
| OHLCV | Sina | Eastmoney/Tencent |
| Proxy | iFinD | Wind |
| 宏观 | gold SQLite + Excel | FRED |

#### 1.2 数据审计清单（必跑）

```python
# 1. OHLCV 前复权
python scripts/fix_ohlcv_adjust.py --threshold 0.5

# 2. NAV 起跑日
from combo.trim_flat_prefix import trim_flat_prefix
navs_trimmed = trim_flat_prefix(navs, min_history=252)

# 3. 频率对齐
assert all(navs.index.freq == 'B')  # 工作日
assert all(weekly_xts.index.dayofweek == 4)  # 周频在周五

# 4. Proxy NaN
proxy_panel = build_proxy_panel(...).where(build_proxy_panel(...).notna())
# 保留上市前 NaN

# 5. 缺数据扰动（粗）
test_with_20pct_missing(panel, seed=42)
```

#### 1.3 数据 Pipeline 最后稳定原则（8 条）

| # | 原则 | 检查点 |
|---|------|--------|
| 1 | 价格 = 前复权/事件修复后 | `etf_ohlcv_*_adjusted.parquet` 存在 |
| 2 | 周频特征 + 日频执行 | `data_loader_v7_6.py` 输出 `X[T,N,K]` |
| 3 | 严格 `X[t]→Y[t+1]` | 不变量：`Y[t]` 是 t-1→t 收益，训练 `(X[t-1], Y[t])` |
| 4 | 短缺口填补 + 长缺口 NaN + 动态资产池 | `_get_valid_assets(min_assets=10)` |
| 5 | 宏观时序 Z-score + PV 截面 Z-score + Winsorize | `X_panel[T,N,K]` 输出 |
| 6 | 评估 7 件套：IC + 滚动 IC + 多起点 + Bootstrap + 多段 hold-out + 缺失扰动 + expanding OOS | 每个都跑一遍 |
| 7 | 所有派生 NAV 同日频指标 | `combo/standard_comparison.py` |
| 8 | 模型选择向"压缩 + expanding + 混合"演进 | 不依赖纯 TV-PR 最大权重 |

### 阶段 1 闸门：数据闸门（P0）

- ☐ OHLCV 前复权已完成
- ☐ Proxy NaN 表存在
- ☐ 缺数据扰动报告有结论
- ☐ 起跑日对齐方式已声明
- ☐ 频率一致性已 verify

**通过标准**：5 个 ☐ 全打，方可进入阶段 2

---

### 阶段 2：IC 评估（[详细文档](./01_DATA_FOUNDATION.md §3)）

#### 2.1 IC 计算

```python
# 截面因子 (PV, k≥17)
from common.ic_utils import compute_cross_sectional_ic
ic_panel = compute_cross_sectional_ic(factor_panel, fwd_returns_panel, horizon=21)

# 时序因子 (Macro, k≤16)
from common.ic_utils import compute_time_series_ic
ic_timeseries = compute_time_series_ic(macro_X, mean_Y_panel, beta_t_panel)
```

#### 2.2 IC 诊断清单

| 维度 | 工具 | 阈值 |
|------|------|------|
| **单因子 IC mean** | `compute_ic_summary()` | \|IC\| > 0.05 |
| **IC 标准差** | `compute_ic_summary()` | < 0.1（ICIR > 0.5）|
| **命中率** | `ic > 0 占比` | > 50% |
| **持续性 / 半衰期** | `factor_timing_diagnostic.py` | 4-13 周 |
| **跨年稳定性** | `yearly_ic` std | < mean |
| **跨窗口对比** | `rolling_ic(window=52)` | 1m/3m/6m/12m 显著 |

### 阶段 2 闸门：IC 闸门

- ☐ **截面 vs 时序**用正确 IC 方法（L-102）
- ☐ **\|IC\| > 0.05** 的因子入选
- ☐ **ICIR > 0.5** 的因子保留
- ☐ **持续性 4-13 周**的因子可用（短期为佳）
- ☐ **跨年稳定**的因子保留

**通过标准**：至少 5 个因子（>=5）+ 全部诊断完成

---

### 阶段 3：因子构建

#### 3.1 因子 3 大类型

| 类型 | 例 | 处理方式 |
|------|------|---------|
| **宏观（时序）** | VIX、DXY、实际利率 | 沿时间 Z-score |
| **量价 / PV（截面）** | 11 量价因子 | 沿截面 Z-score |
| **复合 / DCC** | dcc_zscore_mean | 全市场时序 |

#### 3.2 因子工程 4 步法

```
(1) 候选 → (2) 评估 → (3) 去重 → (4) 标准化
```

##### 3.2.1 候选
- 从 industry/research papers 找 (e.g., 华西证券 11 量价)
- 从关联项目找 (e.g., comovement 40 维特征)
- 从 user insight 找 (e.g., "92 错过 18.87%" → 反转/行业轮动)

##### 3.2.2 评估（见阶段 2）

##### 3.2.3 去重

```python
# 相关矩阵 + 散点图
import seaborn as sns
corr = factor_panel.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')

# r > 0.93 必删其一
high_corr = corr.stack()[corr.stack() > 0.93]
```

##### 3.2.4 标准化

```python
# 宏观: 时序 Z-score
macro_X_zscore = (macro_X - macro_X.mean(axis=0)) / macro_X.std(axis=0)

# PV: 截面 Z-score + Winsorize
def cross_section_zscore_with_winsorize(panel, winsorize_pct=(0.01, 0.99)):
    panel_winsorized = panel.clip(
        lower=panel.quantile(winsorize_pct[0]),
        upper=panel.quantile(winsorize_pct[1])
    )
    zscore = (panel_winsorized - panel_winsorized.mean(axis=1).values.reshape(-1,1)) / \
             panel_winsorized.std(axis=1).values.reshape(-1,1)
    return zscore
```

#### 3.3 因子正交化（慎用！）

| 方法 | 推荐度 | 理由 |
|------|-------|------|
| **不处理** | ⭐⭐⭐ | 适合多数场景 |
| **Gram-Schmidt 残差化**（金融预定义顺序）| ⭐⭐ | 保留金融意义（L-104）|
| **去重 + log 变换** | ⭐⭐ | 处理偏态量纲 |
| **QR 对称正交** | ❌ | OOS 0.056 失败 |
| **Symmetry 正交化** | ❌❌ | Sharpe 1.40→0.15 失败 |

### 阶段 3 闸门：因子闸门

- ☐ 因子分类正确（宏观 vs PV）
- ☐ 去重完成（r > 0.93 必删）
- ☐ 标准化方式正确
- ☐ 不使用 Symmetry / QR 对称正交化
- ☐ 所有 factor_names + codes 持久化

---

### 阶段 4：选股（SubStrategy）

#### 4.1 SubStrategy 接口

```python
class SubStrategy(ABC):
    @abstractmethod
    def select(self, nav_df, as_of, pool_codes) -> list[str]:
        """返回选中的资产 codes"""
        ...

    @abstractmethod
    def weight(self, nav_df, codes, as_of) -> dict[str, float]:
        """返回各 code 的权重"""
        ...

    def run_step(self, nav_df, as_of, pool_codes, max_weight=0.15):
        chosen = self.select(nav_df, as_of, pool_codes)
        weights = self.weight(nav_df, chosen, as_of)
        weights = self._apply_max_weight(weights, max_weight)
        return SubStrategyResult(date=as_of, chosen=chosen, weights=weights, ...)
```

#### 4.2 选股 3 大方法

| 方法 | 用例 | 实证 |
|------|------|------|
| **动量 rank** | V1.0 locked | OOS Calmar 1.79 ⭐ |
| **量价 Top-N** | V5 (华西) | OOS 0.643 |
| **IC 驱动** | V6.1 + V6.2 (DEPRECATED) | OOS 0.748 / 0.901→0.331 |

### 阶段 4 闸门：选股闸门

- ☐ SubStrategy 接口完整（select / weight）
- ☐ 单独测试 ≥ 12 个用例
- ☐ Max_weight 边界 bug 已修复
- ☐ SubStrategyResult 校验通过

---

### 阶段 5：加权

#### 5.1 加权方式选择

| 方法 | 用例 | 推荐 |
|------|------|------|
| **等权** | baseline | ⭐ |
| **逆波动率** (`w ∝ 1/σ`) | v5.1 / v6 | ⭐⭐ 最简单稳健 |
| **IC 加权** | v6.1 | ⭐⭐ |
| **Gram-Schmidt IC + 正交** | v6.2 | ⭐ (有 CV% 风险) |
| **风险平价** | v9 RP | ⭐⭐ 高维场景 |
| **Vol-parity** | V10 组合 | ⭐⭐⭐ 生产首选 |

#### 5.2 加权 3 大原则

1. **粗粒度 > 细粒度**（L-123）
2. **加权层验证单一变量**（先用 v1 baseline，加权后再做对比）
3. **T+1 lag 防 look-ahead**（L-204）

### 阶段 5 闸门：加权闸门

- ☐ 单层加权验证（先不动其他层）
- ☐ 权重和=1、非负、code 在 pool 内
- ☐ T+1 lag 已应用
- ☐ max_weight 上限检查

---

### 阶段 6：风控

#### 6.1 风控叠加原则（L-122）

> **"信号 + 风控不可独立可加，必须成对测试"**

**7 档消融**：
| # | 模式 | 推荐 |
|---|------|------|
| 1 | 无风控 | baseline |
| 2 | 只 VT | ❌ 进攻型策略不适用 |
| 3 | 只 TF | ✅ 大多数场景 |
| 4 | 只 Cost | 中性/微负 |
| 5 | TF + Cost | 二选一 |
| 6 | VT + TF + Cost | 全风控（重）|
| 7 | 动态风控（DCC overlay） | ✅ 长期 |

#### 6.2 风控层权重

```python
VT = VolTargeting(target=0.15, scale_clip=(0.3, 1.5))  # 默认开
TF = TrendFilter(ma=200, bear=0.7, bond="511260")     # 必备
SL = StopLoss(threshold=-0.15, cooldown=5)            # 推荐
DCC = DCCRegimeOverlay(threshold=1.5, reduce=0.5, cooldown=4)  # 长期
```

### 阶段 6 闸门：风控闸门

- ☐ 7 档消融完成
- ☐ 推荐档有数字支持（不能凭空选）
- ☐ 风控层优先级记录（策略回调 > 引擎配置）
- ☐ 风控辅助函数单测 ≥ 18 个

---

### 阶段 7：组合

#### 7.1 组合方式选择

| 方式 | 推荐度 | 用例 |
|------|-------|------|
| **单策略** | ⭐ | 一开始 |
| **粗粒度 v3 80% + v5 20%** | ⭐⭐ | 真实有效路径 |
| **细粒度多策略（1/N 等权）** | ❌ | V3 教训 |
| **Vol-parity (V10 4 策略)** | ⭐⭐⭐ | **生产首选** |

#### 7.2 组合代码模板

```python
# 4 策略 Vol-parity
strategies = {
    'v1.0_locked': 0.74,
    'v9macro': 0.12,
    'v7.10': 0.09,
    'DualMom': 0.05
}
target_vol = 0.08

# 每子策略 vol-parity 子权重
for name, base_w in strategies.items():
    sub_vol = strategies_vol[name].std() * np.sqrt(252)
    sub_target = target_vol / 3  # 每子策略 ≈ target/3
    parity_weight = sub_target / sub_vol
    final_weights[name] = base_w * parity_weight

# 归一化
final_weights /= sum(final_weights.values())
```

### 阶段 7 闸门：组合闸门

- ☐ 粗粒度 vs 细粒度选择有依据
- ☐ 组合 sub-nav 已跟踪完整
- ☐ Vol-parity 权重有 OOS 验证
- ☐ 组合档数 ≤ 5（避免 N 个策略对齐难）

---

### 阶段 8：OOS 验证（4 步标准化流程）

> **L-201 / L-322：4 步 OOS 流程** ⭐⭐⭐

#### 8.1 Step ① 验证过拟合严重程度

```python
# scripts/test_overfitting_signals.py
# (a) Bootstrap CV%
result = bootstrap_cv(returns, n_bootstrap=200, seed=42)
print(f"Bootstrap CV: {result.cv:.2%}")  # > 100% 红灯

# (b) 单起点
for start_date in [2018-01, 2019-01, 2020-01, 2021-01, 2022-01]:
    oos = run_strategy(data.loc[start_date:])
    print(f"{start_date}: Calmar={oos.calmar:.3f}")

# (c) 敏感性测试（10 阶段，按需启动）
#   Phase 0: 论文默认参数对比
#   Phase 1: 单参数扰动 (19 组)
#   Phase 2: Hold-out 多段
#   Phase 3: Bootstrap 稳定性
#   Phase 4: 缺失数据扰动
#   Phase 5: 构造层扰动
#   Phase 6: β_path 断点
#   Phase 7: 综合报告
#   Phase 8: 默认参数对比
```

**判定**：
- Bootstrap CV < 50% 可继续
- 多起点 Calmar CV < 25% 可继续
- 任何 🔴 红灯 → 回到阶段 3（因子去重 / 数据质量）

#### 8.2 Step ② 修复 off-by-one bug

```python
# 验证索引对齐
assert nav.index.freq is not None
assert all(weights.index <= next_rebal_date)

# 验证标签含义
# Y[t] 必须是 t-1 到 t 的收益
# 训练时 (X[t], Y[t+1]) OR (X[t-1], Y[t])
```

**经验**：
- 周频索引用周日，行情用周五 → **必须显式平移 2 天**
- 周一开盘到周五收盘需要**独立对齐**

#### 8.3 Step ③ expanding-window 消除 look-ahead

```python
# common/walk_forward.py
from common.walk_forward import WalkForwardConfig, walk_forward

config = WalkForwardConfig(
    method="expanding",  # 不允许 rolling / full_sample
    train_window=None,   # expanding 模式
    val_window=252,      # 1y OOS
    lookback=21          # 因子 lookback
)
result = walk_forward(strategy_fn, data, config)

# expand 模式报告 OOS 优于 full_sample 是真理信号
```

**判定**：
- expanding OOS Sharpe > 1.0 可继续
- expanding OOS 优于 full_sample 是强信号

#### 8.4 Step ④ 起点依赖 CV% < 25% PASS

```python
# scripts/test_starting_points.py
calmars = []
for start_date in starts_10:
    oos = run_strategy(data.loc[start_date:])
    calmars.append(oos.calmar)

cv = np.std(calmars) / np.mean(calmars)
print(f"起点依赖 CV%: {cv:.2%}")

if cv < 0.25:
    print("✅ PASS")
elif cv < 0.50:
    print("⚠️ PROMISING")
else:
    print("❌ DEPRECATED - 回到阶段 1")
```

#### 8.5 5-fold walk-forward（gold standard）

```python
# scripts/generalization_test_*.py
folds = [
    ('2020', '2018-01-01', '2020-01-01'),
    ('2021', '2018-01-01', '2021-01-01'),
    ('2022-H1', '2018-01-01', '2022-07-01'),
    ('2022-H2', '2018-01-01', '2024-01-01'),
    ('2023-2024', '2018-01-01', '2025-01-01'),
]

# 每 fold 重新算 IC weights + 正交化 IR 排序
# 阈值: ≥ 3/5 胜前一版本
```

### 阶段 8 闸门：OOS 闸门

- ☐ Step ① Bootstrap CV < 50%
- ☐ Step ① 单起点 ≥ 3 个起点
- ☐ Step ② 索引/标签对齐通过
- ☐ Step ③ expanding OOS 优于 full_sample
- ☐ Step ④ CV% < 25% PASS
- ☐ 5-fold walk-forward ≥ 3/5 胜

**通过标准**：6 个 ☐ 全打，方可进入阶段 9

---

### 阶段 9：硬化（生产化）

#### 9.1 Stage 32 硬化 5 P0 任务（来自 `bbcaf86`）

| 任务 | 内容 |
|------|------|
| **Task 1** | `stop_loss()` 实现（不再是 stub）|
| **Task 2** | 3 起点 CV% 测试 |
| **Task 3** | `strategy_versions.py` 接入（`v7_10_std_newλ(**overrides)` 工厂）|
| **Task 4** | 数据生成自动化（CLI）|
| **Task 5** | v6.2 DEPRECATED 标记（DeprecationWarning + 文档）|

#### 9.2 代码清理 + 补测试（1-2 天）

```bash
# 死代码
- pycaret_estimator.py → ARCHIVED
- macro_substrategy_v7_7.py → ARCHIVED
- adaptive_factor_selector.py → ARCHIVED

# 移除未使用导入
# tvpr_estimator.py __all__ 移到文件末尾

# 归档研究脚本
- scripts/research/
```

#### 9.3 工厂函数 + YAML

```python
# 工厂函数
def v7_10_std_newλ(**overrides):
    cfg = V7_10Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

# YAML
# strategies/v7.10.yaml
strategy: v7_10
v7_10:
  lambda_tv: 0.15
  lambda_l1: 0.05
  method: expanding  # 强制 expanding 而非 admm
  stop_loss:
    threshold: -0.15
    cooldown: 5
```

#### 9.4 文档

- 更新 `STRATEGY_VERSIONS.md`
- 更新 `STAGE_*_PLAN.md` (加上此版本)
- 更新 `UNIFIED_V1V5_REPORT.md` 加新版本对比
- 重新生成 `STRATEGY_ITERATION_RECORD.html`

### 阶段 9 闸门：硬化闸门

- ☐ Stop_loss 不再是 stub
- ☐ CV% 验证在文档中可见
- ☐ 工厂函数可被外部调用
- ☐ YAML 配置可被 `run_from_yaml()` 加载
- ☐ 死代码已 ARCHIVED
- ☐ 文档 + HTML 已更新

**通过标准**：6 个 ☐ 全打，可以进 `STRATEGY_VERSIONS` 的 LATEST

---

## 三、SOP 应用模板（适用于新策略）

### 3.1 新策略研发清单

```
□ 阶段 1: 数据准备
    □ 选择数据源（Tencent/iFinD/Wind）
    □ 跑 OHLCV 前复权
    □ 跑 Proxy NaN 表
    □ 跑 缺数据扰动
    □ 跑 起跑日对齐
    □ 形成 X_panel[T,N,K] + Y[T,N]

□ 阶段 2: IC 评估
    □ 截面/时序因子分类
    □ 单因子 IC
    □ IC 标准差 + ICIR + 命中率
    □ 持续性诊断（4-13 周最佳）
    □ 跨年稳定性

□ 阶段 3: 因子构建
    □ 因子去重（r > 0.93 必删）
    □ 标准化（宏观时序 Z / PV 截面 Z）
    □ 不使用 Symmetry / QR 对称正交

□ 阶段 4: 选股
    □ SubStrategy 接口
    □ 单独测试 ≥ 12 个
    □ Max_weight 边界修复

□ 阶段 5: 加权
    □ 单层验证
    □ 权重和=1
    □ T+1 lag

□ 阶段 6: 风控
    □ 7 档消融
    □ TF 必备
    □ Stop Loss 推荐

□ 阶段 7: 组合
    □ 粗粒度首选
    □ Vol-parity 4 策略

□ 阶段 8: OOS 验证 (4 步)
    □ ① Bootstrap CV < 50%
    □ ② 索引/标签对齐
    □ ③ expanding 优于 full_sample
    □ ④ CV% < 25% PASS
    □ 5-fold walk-forward

□ 阶段 9: 硬化
    □ Stop_loss 实现
    □ 工厂函数
    □ YAML 配置
    □ 死代码 ARCHIVED
    □ 文档 + HTML
```

### 3.2 修改现有策略清单（更快版本）

```
□ 跑数据闸门（5 项）
□ 跑因子诊断（去重 + 标准化）
□ 跑 7 档风控消融（如有风控改动）
□ 跑 4 步 OOS 验证
□ 更新 STRATEGY_VERSIONS
□ 更新 HTML
```

---

## 四、SOP 与教训库结合

| 步骤 | 关联教训 |
|------|---------|
| 阶段 1 | L-211(OHLCV) / L-212(起跑日) / L-213(NaN-safe) / L-214(动态资产池) / L-215(成块缺失) / L-221(双频) / L-222(标准化方式) |
| 阶段 2 | L-101(简单规则胜) / L-102(截面 vs 时序) |
| 阶段 3 | L-103(去重基于实际) / L-104(Gram-Schmidt 而非 QR/Symmetry) / L-109(Symmetry 失败) |
| 阶段 4 | L-105(A 股反转) / L-106(Smart β 不是 alpha) / L-107(A 股动量低配) |
| 阶段 5 | L-121(逆波动基线) / L-123(粗粒度) |
| 阶段 6 | L-122(风控不可独立可加) / L-110(DCC overlay) / L-133(连续 TF 失败) |
| 阶段 7 | L-124(Vol-parity 1.55x) / L-125(5 机制 Sharpe 区间) |
| 阶段 8 | **L-201/L-322 (4 步)** ⭐⭐⭐ / L-202(expanding 优于 full) / L-203(CV%<25%) / L-204(X[t]→Y[t+1]) / L-205(Y 重叠) |
| 阶段 9 | L-301(架构≠业绩) / L-302(高 Ann ≠ 高 Calmar) / L-303(诚实归因) / L-323(工程债) |

---

## 五、SOP 的执行节奏

### 5.1 单次 P0 任务（1-2 周）

| Day | 任务 |
|-----|------|
| Day 1-2 | 阶段 1-2（数据 + IC 评估）|
| Day 3-4 | 阶段 3（因子构建）|
| Day 5-7 | 阶段 4-6（选股 + 加权 + 风控）|
| Day 8-10 | 阶段 7-8（组合 + OOS 4 步）|
| Day 11-14 | 阶段 9（硬化）|

### 5.2 调整现有策略（3-5 天）

| Day | 任务 |
|-----|------|
| Day 1 | 数据闸门复检 |
| Day 2 | 因子诊断（去重 + 标准化）|
| Day 3 | 风控消融（如有改动）|
| Day 4 | 4 步 OOS |
| Day 5 | 文档 + HTML 更新 |

---

## 六、SOP 失败模式（FAQ）

### Q1: Bootstrap CV = 165% 怎么办？

**A**: 这是 v7.6 的实际失败模式（`950420b`）。**红牌**。建议：
1. 检查样本量（200 bootstrap 重采样是否够）
2. 检查缺数据（Phase 4 同步）
3. **回到阶段 1 / 3**（数据 / 因子）

### Q2: expanding OOS 没有优于 full_sample 怎么办？

**A**: 这意味着**有可能 full_sample 不含前视，反而是 expanding 过度惩罚**。建议：
1. 检查 expanding 长度（不够 100 期会有问题）
2. 检查 expanding 时的 min_history 设定
3. **若仍不优于，应回退 full_sample 但加 walk-forward 验证**

### Q3: CV% 50% 怎么办？

**A**: **DEPRECATED**。必须回退。建议：
1. 加宽训练样本
2. 减少参数数量
3. 增加训练样本多样性

### Q4: 树模型 R² ≈ 0 怎么办？

**A**: **信号质量问题，非模型问题**（L-108）。不要纠结模型：
- 回到 IC 闸门，看因子 ICIR
- 增加宏观择时层

### Q5: Vol-parity 不work 怎么办？

**A**: 检查子策略之间的相关性：
- 子策略高相关 (r > 0.7) → Vol-parity 无效
- 需要更"独立"的策略组合

---

## 七、SOP 文档维护

### 7.1 文档位置

- 当前：`docs/research_history/06_RESEARCH_SOP.md`
- 引用：`docs/research_history/05_LESSONS_LIBRARY.md`

### 7.2 每次新教训应同时更新

1. `05_LESSONS_LIBRARY.md`（教训登记 L-NNN）
2. `06_RESEARCH_SOP.md`（SOP 对应阶段）
3. `00_TIMELINE.md`（如果是新事件）

### 7.3 阶段报告产出

每个 P0 任务完成后应：
- 在 `reports/` 下写一份 `STAGE*_NEW.md`
- 更新 `STRATEGY_VERSIONS.md`
- 更新 `STRATEGY_ITERATION_RECORD.html`

---
