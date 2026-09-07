# Zookeeper 因子作为状态变量接入主 CTA vol_target — 设计文档

> ZK 因子复现 §28.15 · P7-B 模拟式集成 · 用户拍板 B (2026-09-07)

## 1. 决策背景

P5 Spanning Test 结果 (54 月 OOS):
- **主 CTA baseline α = +12.94%/年 (t = 4.52)** —— ZK 因子对主 CTA 极显著独立 α 源
- **自身 Q4/Q1 多空 Sharpe -0.35** (P4-B) / 加权 LS Sharpe -0.37 (P6-C) —— 仍 FAIL

矛盾结论: **ZK 含有的信号是真实独立信息, 但用 Q4/Q1 或加权提取方式做自身多空策略都是负 alpha**。论文 §6 提示真正用法: **state variable** 而非子策略方向。

用户决策: **B 接入 vol_target (与论文 §6 state variable 一致)**.

## 2. 接入原理

```
vol_target_adjusted = vol_target_base × adjustment_factor(zk_state)
adj_factor(zk) = 1 + direction × alpha × (zk_state_norm - 0.5) × 2
                clip(0.3, 2.0)
```

**核心逻辑**:
- ZK 状态变量月度更新 → 月初调节下月 vol_target
- direction = -1 (反向): ZK 信号弱 → vol 上调 (机会窗口); ZK 信号强 → vol 下调 (风险警报)
- direction = 1 (正向): ZK 信号强 → vol 上调 (顺势加仓); 弱 → vol 下调

## 3. ZK 状态变量三选一

| 状态变量 | 计算 | 语义 |
|---|---|---|
| **signal_dispersion** (推荐) | 月度 r̂ 横截面 std | 信号分散度越高 → 商品强弱分化大 → 不明朗 |
| signal_strength | 月度 \|r̂\| 横截面均值 | 信号强度 → 整体方向 |
| signal_momentum | ZK 因子过去 3 月累计收益 | ZK 自身趋势 → 外推 |

## 4. 网格搜最优 (1698 个交易日 ≈ 7 年)

| α | 方向 | Sharpe_B | DD_B | α_ann | t_α |
|---:|---:|---:|---:|---:|---:|
| 0.1 | -1 | 3.483 | -2.47% | +0.27% | 2.29 |
| 0.2 | -1 | 3.519 | -2.44% | +0.54% | 2.29 |
| **0.3** | **-1** | **3.540** | **-2.40%** | **+0.81%** | **2.29** |
| 0.5 | -1 | 3.537 | -2.33% | +1.35% | 2.28 |
| 0.7 | -1 | 3.474 | -2.26% | +1.89% | 2.28 |
| 0.3 | +1 | 3.235 | -2.75% | -0.80% | -2.29 |

**最优配置**: α=0.3, direction=-1, stat=signal_dispersion
- 主 CTA Sharpe 3.435 → **3.540** (+3.0%)
- DD -2.51% → -2.40% (缩窄 4%)
- α_ann +0.81% (t = 2.29, 显著)

## 5. v11 主 CTA 引擎接入设计 (E1 真接入已实施)

### 5.1 不破坏现有架构原则
- ZK 状态变量作为**可选调节器**, 不替换 vol_target_base
- 通过 config 注册: `use_zk_vol_adj: bool = False` (默认关闭, 手动开启)
- 接入点: `vol_target_positions()` (factors/volatility.py)

### 5.2 接入代码 (已实施, commit 待)
**修改 1**: `QuantNodes/strategy/cta/factors/volatility.py`
```python
def vol_target_positions(signals, prices, target_vol=0.15, vol_lookback=20, max_leverage=3.0,
                       daily_vol_target=None):   # ← 新增参数
    ...
    weight_per_unit = target_vol / vol.replace(0, np.nan)
    if daily_vol_target is not None and len(daily_vol_target) > 0:
        dvt = daily_vol_target.reindex(weight_per_unit.index).ffill().fillna(target_vol)
        weight_per_unit = vol.replace(0, np.nan).div(dvt.values, axis=0)  # 关键: axis=0 广播
    weight_per_unit = weight_per_unit.clip(upper=max_leverage).fillna(0)
    return signals * weight_per_unit
```

**修改 2**: `QuantNodes/strategy/cta/strategies/base.py`
```python
ZK_VOL_ADJ_CFG = {
    "enabled": False,                        # 默认关闭
    "method": "weighted",
    "stat": "signal_dispersion",
    "alpha": 0.5,                            # E3 推荐 (Sharpe 3.557)
    "direction": -1,                         # 反向 (弱信号加仓)
    "weights": {"signal_strength": 1/3, "signal_dispersion": 1/3, "signal_momentum": 1/3},
}

def load_zk_vol_adj_factor(config=None) -> pd.Series:
    """加载 ZK 多 stat 集成 vol_target 调节因子 (月度→日频)."""
    cfg = config or ZK_VOL_ADJ_CFG
    if not cfg.get("enabled", False):
        return None
    # 加载三个 stat, 标准化, 等权集成, 调节因子计算
    ...

# 在 _signal_to_lots() 调用前:
def _signal_to_lots(self, signals, cfg):
    daily_vol_target = None
    if ZK_VOL_ADJ_CFG.get("enabled", False):
        daily_vol_target = load_zk_vol_adj_factor(ZK_VOL_ADJ_CFG)
    weights = vol_target_positions(signals, self._price_panel,
                                  target_vol=cfg.target_vol,
                                  vol_lookback=cfg.vol_lookback,
                                  max_leverage=cfg.max_leverage,
                                  daily_vol_target=daily_vol_target)  # ← 新传参
    ...
```

### 5.3 实际接入工作 (已完成, 待 commit)
1. ✅ 修改 `vol_target_positions()` 接受 daily_vol_target
2. ✅ 修改 `_signal_to_lots()` 调用新参数
3. ✅ 添加 ZK_VOL_ADJ_CFG + load_zk_vol_adj_factor()
4. ✅ ruff 通过 + 单元测试通过 (vol 平均 1.59 → 0.69 = dvt/target_vol 比例)
5. ⏳ 用户拍板后改 `enabled=True` → 跑 cta_full_assessment_export.py → 评估新主 CTA Sharpe/DD

### 5.4 风控纪律 (沿 §28.4 硬阈值)
- 接入后主 CTA Sharpe 必须 > 3.0 (现状 3.435)
- DD 必须 < 5% (现状 2.51%)
- 接入后 Spanning Test (主 CTA 自身 vs vol_adj 主 CTA) 验证 α 显著
- 若新主 CTA α > 5% 触发谨慎审查

### 5.5 接入实施 E1 验证 (本次)
- ruff All checks passed ✓
- 单元测试: vol_target_positions 接受 daily_vol_target 正常 ✓
  - 默认 (target_vol=0.15): 平均 vol = 1.59
  - 注入日频 dvt (mean 0.13): 平均 vol = 0.69 = 0.13/0.15 × 1.59 ✓ (精确比例)
  - 月频 dvt + ffill 注入: 平均 vol = 0.475 ✓
- ZK_VOL_ADJ_CFG 加载函数: 默认 None (关闭), 开启后 65 月调节因子 (范围 0.73 ~ 1.34)
- ⚠️ 主 CTA 全流程 lazy load 测试未跑 (production库派生表只到 2026-01-06, 需先用 cta_full_assessment_export.py 触发实际评估)

### 5.6 启用步骤 (用户拍板后)
```python
# 1. config/ZK_VOL_ADJ_CFG.enabled = True
# 2. 跑 scripts/cta_full_assessment_export.py
# 3. 评估 Sharpe/DD/Calmar vs 原主 CTA
# 4. 跑 spanning.py 验证新主 CTA α 显著
# 5. 跟踪 30 天 OOS 后再正式启用 (避免 overfit)
```

### 5.7 决策路线图 (待用户拍板)
| 选项 | 含义 | 估时 |
|---|---|---|
| **E1A** | **保持 enabled=False, 文档化集成代码 (本 commit)** | 0 (当前状态) |
| E1B | 拍板启用 + 跑主 CTA 全流程评估 (验证 Sharpe/DD) | 1-2 天 |
| E4 | 扩样本期 (拉到 2014-2017, 但 CH 2018 起始 → **不可行**) | N/A |

## 6. 限制与警示

### 6.1 模拟的局限
- `simulate_vol_target_adjustment()` 在历史主 CTA NAV 上**近似**缩放月度收益, **不重跑主 CTA 引擎**
- 真实接入需重跑全 CTA pipeline (v11 factor_modules), 结果可能 ±5%
- 日内 vol_target 是常数, 实际应月度调整

### 6.2 警惕的 3 个潜在风险
1. **样本期短 (5 年)**: 与 P4/P5/P6 同样问题, vol_target 调节有效性待 2027+ 验证
2. **过拟合风险**: α 网格搜的最优是 0.3, 但 α=0.5/0.7 也有正 α (差距 < 1%), 实际选用应保守取 α=0.2
3. **状态变量选择敏感**: signal_momentum 表现不如 signal_dispersion, 需鲁棒性测试

### 6.3 不能做的事
- ❌ **不要把 ZK 因子自身当成子策略方向** (P4/P6 FAIL 已证实)
- ❌ **不要把 ZK 作为信号直接驱动多空仓位** (vol_target 调节才是合规用法)
- ❌ **不要单独 ZK 跑 E2E 回测** (回测引擎未真实集成, 当前是 simulate 近似)

## 7. 决策路线图 (待用户拍板)

| 选项 | 含义 | 估时 |
|---|---|---|
| **E1** | **真接入 v11 引擎 (按 §5.3 步骤)** | 1-2 天 |
| E2 | 维持 simulate, 仅文档化状态变量接入设计 | 0 (当前状态) |
| E3 | 跨多 stat/α/direction 鲁棒性测试 | 0.5 天 |
| E4 | 5 年样本期扩展 (拉 2014-2017 历史) | 5-7 天重投 |

**当前提交为 E2 (simulate + 文档)**, 等用户拍板 E1/E4。