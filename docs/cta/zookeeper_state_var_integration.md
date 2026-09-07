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

## 5. v11 主 CTA 引擎接入设计 (待用户拍板)

### 5.1 不破坏现有架构原则
- ZK 状态变量作为**可选调节器**, 不替换 vol_target_base
- 通过 config 注册: `use_zk_vol_adj: bool = False` (默认关闭, 手动开启)
- 接入点: `vol_target_positions()` (factor_modules/base.py)

### 5.2 接入代码骨架 (示意)
```python
# config.py (新增)
ZK_VOL_ADJ_CFG = {
    "enabled": False,           # 拍板后开启
    "alpha": 0.3,
    "direction": -1,
    "stat": "signal_dispersion",
    "src_path": "reports/cta/zookeeper/factor_vol_adj_weighted_signal_dispersion_a0.3_d-1.csv",
}

# factor_modules/base.py (新增vol函数)
def vol_target_positions_with_zk(self, ...):
    """vol_target_adjusted = vol_target_base × zk_adj_factor."""
    if ZK_VOL_ADJ_CFG["enabled"]:
        adj = load_zk_factor(...)
        target_vol *= adj.loc[date]
    return vol_target_positions(...)
```

### 5.3 实际接入工作清单 (5 步, 估 1-2 天)
1. 把 `factor_vol_adj_weighted_signal_dispersion_a0.3_d-1.csv` 接入 ZK_VOL_ADJ_CFG
2. 在 `vol_target_positions()` 增加 if 开关
3. 跑 `scripts/cta_full_assessment_export.py` 重新生成主 CTA 评估曲线
4. 对比 vol_target 调节前后 (Sharpe/DD/Calmar)
5. 更新 tuning_log §28.15 commit

### 5.4 风控纪律 (沿 §28.4 硬阈值)
- 接入后主 CTA Sharpe 必须 > 3.0 (现状 3.435)
- DD 必须 < 5% (现状 2.51%)
- 接入后 Spanning Test (主 CTA 自身 vs vol_adj 主 CTA) 验证 α 显著
- 若新主 CTA α > 5% 触发谨慎审查

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