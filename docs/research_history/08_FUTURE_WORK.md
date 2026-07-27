# 08 — 未来工作（STAGE33 + 余下机会）

> **范围**：V10 之后的策略演进方向
> **核心依据**：`fd61982` STAGE33_PLAN（V7.10 收尾时一并规划）+ V9/V10 沉淀的经验
> **优先级**：可立即执行 ≥ 1-2 周可完成 ≥ 1 月可完成 ≥ 长期

---

## 一、STAGE33 规划原文回顾

`fd61982` STAGE33_PLAN.md（v7.10 收尾时一致通过）规划 4 大方向：

```
A. 代码清理 + 补测试（1-2 天）
B. HMM Regime 集成（1 周）
C. 新因子挖掘：40 维特征（1 周）
D. 跨资产信号集成（1 周）
```

下面逐一展开 + 加入我们从 V8/V9/V10 沉淀出来的"余下机会"。

---

## 二、A. 代码清理 + 补测试（1-2 天，可立即开始）

### A1. `stop_loss` 测试（≥8 tests）

**当前缺口**：`v7_6_with_stop_loss()` 在 Stage 32 P0 Task 1 实现，但测试不完整。

```python
# tests/strategy/momentum_etf_rotation/test_v7_6_stop_loss.py
# 测试用例:
# - 触发条件：NAV DD > -15%
# - cooldown：5 周内不重开
# - 滑点保护：避免震荡市来回止损
# - 多场景回测：2018 / 2022 / 2024 牛熊市
# - 与 TF + VT 叠加的行为
# - 空仓 + cooldown 期不触发
```

### A2. v7.10 工厂函数测试（≥5 tests）

```python
# tests/strategy/momentum_etf_rotation/test_strategy_versions_v7_10.py
# 测试用例:
# - 默认参数生成
# - override 参数生效
# - YAML 加载 + strategy_versions 转换
# - get_version("7.10") 工作
# - 版本不存在的错误处理
```

### A3. 清理死代码（已部分执行 `252fabe`）

```bash
# ARCHIVED（移到 archive/）
- pycaret_estimator.py          → ARCHIVED  # v7.7 失败
- macro_substrategy_v7_7.py     → ARCHIVED  # v7.7 失败
- adaptive_factor_selector.py   → ARCHIVED  # v7 早期实验

# 移除未使用导入
- 检查 tests/strategy/...
- 检查 QuantNodes/strategy/...

# __all__ 整理
- tvpr_estimator.py __all__ 移到文件末尾
- 哪些函数是公开 API？
```

### A4. 归档研究脚本（`scripts/research/`）

- 已经在 `scripts/research/`，检查是否有死代码
- 重复的 `aggregate_alpha_results.py` / `analyze_alpha_results.py` / `merge_alpha_data.py` 是否需要精简

---

## 三、B. HMM Regime 集成（1 周）

### 来源

`~/Public/comovement/hmm_regime/`（独立模块，可复用观测向量）

### 思路

HMM 5 状态叠加 v7.10，regime-aware 权重调整：

```python
# v7/hmm_regime_overlay.py（已存在）
# 复用 9 维观测向量 (VIX, TED, 信用利差, GPR, FSI, 油金相关性等)

class HMMRegimeOverlay:
    def __init__(self, n_regimes=5, transition="distance_prior"):
        self.hmm = GaussianHMM(n_components=n_regimes)
        self.observations = [...]  # 9 维

    def fit(self, data):
        self.hmm.fit(data)
        # 用 distance_prior 初始化 transmat_prior
        self.hmm.transmat_prior = build_distance_transmat(...)

    def predict_regime(self, current_data):
        return self.hmm.predict(current_data)

    def overlay(self, base_weights, regime):
        """状态×权重乘数调整 v7.10 组合"""
        multipliers = self.regime_multiplier(regime)
        return {k: v * multipliers[asset_class(k)] for k, v in base_weights.items()}
```

### 与现有组件集成

- 现有 DCC overlay 在 V7.12 实现 (dcc_zscore_mean > 1.5) — HMM 提供更宏观视角
- Jump Model 在 V8 实现（DD_10/Sortino_20/60 bull/bear）— HMM 提供更多状态（5 vs 2）
- **三者正交叠加**：趋势过滤 + 微观结构（DCC）+ Regime（HMM）

### 验收标准

- HMM + v7.10 OOS Calmar > 0.662（v7.10 当前 expanding OOS 真实值）
- HMM state 转换频率 vs 市场事件（手动 sanity check）
- regime-conditioned 收益分布对比 unconditional

---

## 四、C. 新因子挖掘：40 维特征（1 周）

### 来源

`~/Public/comovement/resonance_warning/data/features.py`（只读）

### 10 个新因子

不与现有 36 因子重叠：

| # | 因子 | 来源 | 优先 |
|---|------|------|------|
| 1 | `skewness_60d` | 60 日偏度 | 中 |
| 2 | `kurtosis_60d` | 60 日峰度 | 中 |
| 3 | `max_dd_60d` | 60 日最大回撤 | **高** |
| 4 | `macd` | MACD 信号 | 低 |
| 5 | `bollinger_pct` | 布林带位置 | 中 |
| 6 | `atr` | 平均真实波幅 | 中 |
| 7 | `market_beta` | 市场 beta | 中 |
| 8 | `dispersion` | 截面离散度 | **高** |
| 9 | `tail_co_occurrence` | 尾部共现 | 高 |
| 10 | `vix_corr` | 与 VIX 相关性 | **高** |

### 实施步骤

1. 在 `v7/enhanced_factors_v7_11.py` 实现 10 个新因子（已完成 `be4ad1e` 部分）
2. 生成 v7.11 数据（36 + 10 = 46 因子）
3. 计算每个新因子 IC（`corr(X[t], Y[t+1])`）
4. 筛选 IC > 0.03 的因子
5. expanding-window OOS 验证

### 验收标准

- 新因子组合 OOS Calmar > 0.662（v7.10 真实 OOS）
- 至少 3 个新因子入最终因子库
- 新因子排名后重新做因子去重

---

## 五、D. 跨资产信号集成（1 周）

### 来源

`~/Public/comovement/resonance_warning/data/`（只读）

### 4 大信号

```python
# v7/cross_asset_signals.py（新增）
# 1. DY 溢出指数 (Diebold-Yilmaz 跨资产波动溢出)
# 2. TENET 尾部网络（风险传导方向）
# 3. DCC regime（相关性突变预警，已实现）
# 4. 尾部依赖 spike（GPD 危机预警）
```

### 与 v10 集成

```python
# v10/macro_layer.py 加跨资产信号
cross_asset_signals = compute_cross_asset_signals(etf_returns, macro_data)

# V10 Layer 1 加权:
# TV-PR 50% + 熵权法 30% + 跨资产信号 20%
tvpr_weight = 0.50
entropy_weight = 0.30
cross_asset_weight = 0.20
```

### 验收标准

- 跨资产信号叠加后 Calmar > baseline
- 信号 vs 历史危机（2008 / 2015 / 2018 / 2020 / 2022）的"触发率"
- 信号 IC 在 OOS 时段的可预测性

---

## 六、余下机会（来自 V8/V9/V10 沉淀）

### O1. FI+ 真实化（V2 stub，Stage 15D 遗留）

```python
# v2/fi_plus_v2.py 当前只有 64 行占位
# Stage 15D 标"80% 债券 + 20% v2 动量真实回测"

# 应改为:
class FIPlusV2Strategy(V1Strategy):
    """80% 债券 + 20% v1.0 动量"""
    bond_weight = 0.80
    momentum_weight = 0.20

    def compute_weights(self, date, price_panel, nav_history):
        mom_weights = super().compute_weights(date, price_panel, nav_history)
        return {k: v * 0.20 for k, v in mom_weights.items()}.update({
            "511260": 0.80  # 30 年国债 ETF
        })
```

**应用**：作为 FI+ 风格的"股债混合"防御资产。

### O2. 树模型方向（V7.7 失败 → 转向半监督 / 自适应）

```python
# 不能直接用树模型预测 R² ≈ 0
# 但可以用"自适应权重"或者"半监督"路线

# 选项 1: 把树的特征重要性作为"先验"
#   先用 LightGBM 评估特征重要性
#   再用重要特征进入 TV-PR / IC² 加权

# 选项 2: 半监督学习（结合未标注宏观数据）
#   用 SQLite 里未充分使用的指标作为补充
```

**不要重启 LightGBM 主路线**，因为因子质量问题。

### O3. v9 cycle timing 落地（V10 Layer 1 实现）

```python
# docs/49-v9_cycle_timing.md (457 行)
# v9 宏观周期择时:
# - 周期检测（HMM / 拐点）
# - 周期相位 + 资产轮动对应
# - 49a-f 6 个分阶段诊断

# 已部分集成到 v10/macro_layer.py
# 进一步:
# - 实现 cycle signal 作为宏观择时输入
# - v10 4 策略组合里加 cycle timing overlay
```

### O4. v8 probabilistic model 完善

```python
# v8/probabilistic_*.py 已存在
# 但 v8 大部分时间花在 Jump Model
# probabilistic 可以作为"概率分布式"风险控制
```

### O5. Stage 32 P0 完整化：CLAUDE.md / AGENTS.md 强化

- 把所有教训 L-NNN 整合到 AGENTS.md（不只是项目级，还要策略级别）
- 让新加入者看 AGENTS.md 就知道踩过哪些坑

### O6. docstring + type hints 完整化

- 当前大量代码 docstring 不完整
- Python type hints 不一致
- 在不破接口的前提下改进 docstring / 添加类型注解

### O7. CI / 测试覆盖率 / 自动化

```python
# 当前测试覆盖率未明确报告
# 应该加:
# - pytest --cov=QuantNodes/strategy
# - 覆盖率门槛 (e.g., ≥ 80% for common/)
# - 提交前自动 lint (ruff)
```

### O8. StageBacktest 阶段的简化统一接口

```python
# 现在 v1-v10 各种接口混合
# 应该统一:
# class BacktestRunner:
#     def run(self, strategy, data, config) -> BacktestResult
#     def walk_forward(self, strategy, data, config) -> list[WalkForwardResult]
#     def grid_search(self, strategy, data, param_space) -> GridSearchResult
```

---

## 七、长期机会（按"是否能带来超额 Sharpe"排序）

| 机会 | 来源 | 影响 | 难度 | 优先级 |
|------|------|------|------|--------|
| HMM Regime 5 状态 | comovement | 中 (regime-aware 收益 +0.2-0.4 Sharpe) | 中 | **B 类** |
| 跨资产信号 (DY/TENET) | comovement | 中 (危机预警 -5%~-10% DD) | 中 | **D 类** |
| 新因子 (40 维) | comovement | 低-中 (新增 3-5 因子) | 低 | **C 类** |
| LW 启用（8+ 因子场景）| v4 LW | 低 (v7.10 已 36 因子) | 中 | 长期 |
| FI+ 真实化 (Stage 15D 债股混合) | v2 stub | 低 (新防御资产) | 低 | O1 |
| Cycle timing 落地 | v9 | 中 (宏观择时 IC +0.05) | 高 | O3 |
| v8 probabilistic 完善 | v8 | 低 (风控维度扩展) | 中 | O4 |
| AGENTS.md / CLAUDE.md 强化 | 当前目录 | 高 (经验沉淀) | 低 | O5 |
| docstring + type hints | 项目级 | 中 (代码质量) | 低 | O6 |
| CI / 覆盖率 | 当前没有 | 高 (生产保障) | 中 | O7 |
| BacktestRunner 统一接口 | 当前混合 | 中 (新策略接入成本) | 中 | O8 |
| 树模型方向（不直接预测）| V7.7 | 低 (特征重要性作为先验) | 高 | O2 |

---

## 八、Stage 33 之后的演进路线（5 个里程碑）

| # | 里程碑 | 时间 | 描述 |
|---|--------|------|------|
| M1 | **代码清理完成** | 1-2 天 | A1-A4 全部完成 |
| M2 | **死代码归档** | 1 周 | A3 完成 + 报告更新 |
| M3 | **新因子挖掘 10 个** | 1-2 周 | C 全套 |
| M4 | **HMM Regime overlay** | 1-2 周 | B 全套 |
| M5 | **跨资产信号 + FI+** | 2-3 周 | D + O1 |
| M6 (可选) | **统一 BacktestRunner + CI** | 1 月 | O7/O8 |
| M7 (可选) | **AGENTS.md / CLAUDE.md 强化** | 持续 | O5 |

---

## 九、给"未来本团队"的金句

1. **"STAGE33 4 类方向中,跨资产信号是潜力最大的"** —— 因为 v10 Vol-parity 4 策略组合仍有提升空间
2. **"树模型不是出路"** —— 因子本身 R² ≈ 0,不是因为模型简单
3. **"v2/fi_plus_v2.py 是 8 月遗留债"** —— 该还了
4. **"Stage 32 5 P0 是模板"** —— 任何新策略落地都先做 5 P0
5. **"8 道工序 + 5 道闸门 + 4 步 OOS"** —— 这是 V0-V10 18 天的最大沉淀

---

## 十、Stage 33 任务清单（可立即执行）

如果你打开这个目录是为了"下一步做什么"，下面是 7 天工作周建议：

### 周一（1-2 天）: 代码清理

- ☐ A1. `stop_loss` 测试 ≥ 8 用例（写入 `tests/strategy/momentum_etf_rotation/test_v7_6_stop_loss.py`）
- ☐ A2. v7.10 工厂函数测试 ≥ 5 用例
- ☐ A3. 归档研究脚本（`scripts/research/`）
- ☐ A4. 整理 `QuantNodes/strategy/momentum_etf_rotation/common/` 的 `__all__`

### 周二-周三（1 周）: HMM Regime

- ☐ B1. 从 `comovement/hmm_regime/` 抽取观测向量
- ☐ B2. 在 `v7/hmm_regime_overlay.py` 实现 `HMMRegimeOverlay` 类
- ☐ B3. 9 维观测向量拟合（VIX/TED/信用利差/GPR/FSI/油金相关性等）
- ☐ B4. distance_prior 初始化 transmat
- ☐ B5. regime-conditioned 权重乘数
- ☐ B6. OOS 测试（Calmar > 0.662）

### 周四（1 周）: 新因子 10 个

- ☐ C1. 从 `comovement/resonance_warning/data/features.py` 抽取可用函数
- ☐ C2. `v7/enhanced_factors_v7_11.py` 实现 10 个新因子
- ☐ C3. 生成 v7.11 数据（36 + 10 = 46 因子）
- ☐ C4. 每个新因子 IC 计算
- ☐ C5. 筛选 IC > 0.03
- ☐ C6. expanding OOS 验证
- ☐ C7. 入选因子库（至少 3 个）+ 更新 STRATEGY_VERSIONS

### 周五（1 周）: 跨资产信号

- ☐ D1. DY 溢出指数实现（v7/cross_asset_signals.py）
- ☐ D2. TENET 尾部网络
- ☐ D3. 尾部依赖 spike (GPD)
- ☐ D4. 与 v10 macro_layer 集成
- ☐ D5. V10 Layer 1 加权: TV-PR 50% + 熵权 30% + 跨资产 20%
- ☐ D6. OOS 测试

### 周末（可选）

- ☐ O5: AGENTS.md / CLAUDE.md 强化（整合所有 L-NNN 教训）
- ☐ O6: docstring + type hints 完整化
- ☐ O7: pytest --cov 配置

---

## 十一、最终结论

V0-V10 18 天 971 commits 的最大沉淀：

1. **3+1 个生产候选策略**（v1.0 locked, V7.10 TV-PR, DualMom, V10 Vol-parity 4 策略组合）
2. **统一回测引擎 + YAML 驱动 + walk_forward 工具**
3. **60+ 条教训库**（方法论 / 工程 / 流程 三大类）
4. **8 道工序 + 5 道闸门 + 4 步 OOS 验证的 SOP**
5. **5 个 P0 任务清单模板**（Stage 32 硬化）

未来已来：**STAGE33 的 4 大方向 + 余下机会**，均可直接基于现有 `docs/research_history/06_RESEARCH_SOP.md` 的 8 阶段 SOP 推进。

谢谢团队所有人 18 天的辛苦付出！

---
