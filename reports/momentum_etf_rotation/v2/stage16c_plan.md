# Stage 16C: 强化学习仓位管理 (RL Position Sizing)

> **创建日期**: 2026-07-09
> **优先级**: P2 战略级 (最高复杂度, 最大风险)
> **前置**: Stage 16A (多策略) + Stage 16B (RSRS) - 都需要作为特征
> **状态**: 规划中

---

## 1. 目标与动机

### 1.1 核心问题

v1.0 策略的仓位管理是**规则式**:
- `inverse_vol_weights` 静态规则 (权重 ∝ 1/σ)
- `vol_targeting` 静态目标 (tv=0.15)
- `cost_model` 固定费率 (5bp+10bp)

**缺失**: 动态自适应仓位管理
- 不同市场状态应有不同仓位
- 例: 牛尾 (2019-07) 应减仓, 熊底 (2018-12) 应加仓
- 例: 高 VIX 时降低股票暴露, 加现金

### 1.2 解决思路: 强化学习

**状态空间 (State)**: 当日市场观察值
**动作空间 (Action)**: 总仓位 0~150% (允许 1.5x 杠杆, 加 50% 杠杆)
**奖励函数 (Reward)**: 调仓周期间的夏普比率 (差分)
**算法**: PPO (Stable-Baselines3) 或 SAC

### 1.3 预期收益

| 指标 | v1.0 | 16C 目标 | 改善 |
|------|------|----------|------|
| Calmar | 1.60 | 1.85+ | +15% |
| DD | -3.93% | -2.5% | +36% |
| 月度胜率 | 58% | 70%+ | +12pp |
| 熊市减仓 | 无 | 自动 | -3~5% DD |

**关键风险**:
- **过拟合**: RL 在金融数据上极易过拟合
- **OOS 不稳定**: 训练集表现好, 测试集差
- **训练成本**: 单次训练 30+ 分钟
- **回测偏差**: live trading 与 backtest 滑点差异

---

## 2. 技术方案

### 2.1 强化学习基本框架

```
Environment (市场) ──observation──> Agent
       ^                              │
       │                              │
       └──────────────reward──────────┘
```

**关键设计决策**:
1. **离策略 (off-policy) vs 在策略 (on-policy)**: PPO (on-policy, 稳定) > SAC (off-policy, 灵活)
2. **离散 vs 连续动作**: 连续 (仓位 0~1.5), 简单
3. **状态-动作频率**: 调仓日决策 (避免高频交易)
4. **奖励函数**: 风险调整收益 (Sharpe 差分)
5. **训练-测试分离**: 2 年滚动 OOS

### 2.2 状态空间 (Observation)

#### 2.2.1 基础市场特征 (12 维)

| # | 特征 | 维度 | 计算 |
|---|------|------|------|
| 1 | 沪深300 20d return | 1 | pct_change(20) |
| 2 | 沪深300 60d return | 1 | pct_change(60) |
| 3 | 沪深300 120d return | 1 | pct_change(120) |
| 4 | 沪深300 20d vol (年化) | 1 | std × sqrt(252) |
| 5 | 沪深300 60d vol | 1 | 同上 |
| 6 | 沪深300 ma_ratio (close/ma60) | 1 | ma60 distance |
| 7 | 最大回撤 (近 1 年) | 1 | max DD |
| 8 | 距离 52w 高点 | 1 | (close - 52w_high) / 52w_high |
| 9 | 距离 52w 低点 | 1 | (close - 52w_low) / 52w_low |

#### 2.2.2 动量/反转特征 (8 维)

| # | 特征 | 维度 | 来源 |
|---|------|------|------|
| 10 | 动量信号均值 | 1 | hybrid_momentum_score_v2 |
| 11 | 动量信号 p75-p25 spread | 1 | 分位差 |
| 12 | RSRS z-score | 1 | Stage 16B |
| 13 | 动量信号 60d 变化 | 1 | rolling diff |

#### 2.2.3 策略内部状态 (4 维)

| # | 特征 | 维度 | 来源 |
|---|------|------|------|
| 14 | 当前现金比例 | 1 | 1 - Σ weights |
| 15 | 当前 A股宽基权重 | 1 | sum A broad weights |
| 16 | 当前组合年化 vol | 1 | realized |
| 17 | 当前组合 1m Sharpe | 1 | rolling |

**总维度**: 17 维 (合理)

### 2.3 动作空间 (Action)

```
Action: scalar ∈ [0, 1.5]
- 0: 全现金
- 1: 满仓 (无杠杆)
- 1.5: 1.5x 杠杆 (可加融资, 假设融资成本 5%/年)
```

**应用**: 选定的 ETF 权重按比例缩放
```python
def apply_position_sizing(base_weights: dict[str, float], scale: float) -> dict[str, float]:
    """应用仓位缩放."""
    new_w = {k: v * scale for k, v in base_weights.items()}
    total = sum(new_w.values())
    if total > 1.0:
        # 超过 1.0 的部分假设是现金 + 融资
        return {k: v / total for k, v in new_w.items()}  # 归一化到 1
    return new_w
```

### 2.4 奖励函数 (Reward)

**基础**: 调仓周期间 Sharpe 差分
```python
def compute_reward(nav_before, nav_after, vol_window=21) -> float:
    rets = (nav_after / nav_before - 1).pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    excess_return = rets.mean() - 0.0  # 无风险利率 0
    vol = rets.std()
    if vol == 0:
        return 0.0
    sharpe = excess_return / vol * np.sqrt(252)
    return sharpe  # 周度 Sharpe
```

**惩罚项**:
- 换手惩罚: -0.1 × turnover_rate
- 杠杆惩罚: -0.05 × max(0, scale - 1.0)
- DD 惩罚: -0.5 × max(0, current_dd - 0.10)

### 2.5 训练设置

```python
# PPO config (推荐)
algo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,  # 鼓励探索
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

# 训练周期
TRAIN_YEARS = 2         # 训练窗口
TEST_YEARS = 1          # 测试窗口 (滚动 OOS)
TOTAL_OOS_PERIODS = 5   # 5 个 OOS 周期
TOTAL_ITERATIONS = 100_000  # 总训练步数
```

### 2.6 训练流程

```
1. 数据切分:
   - 训练: 2018-01 ~ 2019-12 (24 个月)
   - 测试: 2020-01 ~ 2020-12 (12 个月)
   - OOS-2 训练: 2019-01 ~ 2020-12
   - OOS-2 测试: 2021-01 ~ 2021-12
   - ...
   - OOS-5 训练: 2022-01 ~ 2023-12
   - OOS-5 测试: 2024-01 ~ 2024-12

2. 对每个 OOS:
   - 训练 PPO
   - 在测试集上评估
   - 记录 Calmar, DD, Sharpe

3. 汇总 OOS 表现:
   - 平均 Calmar, DD, Sharpe
   - 与 v1.0 baseline 对比
   - 通过测试 → 进入下一 Stage
```

### 2.7 集成点

```python
# v2/rl_position_v2.py
class RLPositionAgent:
    """强化学习仓位管理 (Stage 16C)."""
    def __init__(self, model_path: str = None):
        self.model = PPO.load(model_path) if model_path else None

    def predict(self, observation: np.ndarray) -> float:
        """预测仓位缩放 (0~1.5)."""
        if self.model is None:
            return 1.0  # 满仓 fallback
        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])

# v2/backtest_v2.py 修改
def run_rotation_backtest(...):
    # ... 现有逻辑 ...
    if rot.rl_agent.enabled:
        # 计算 observation
        obs = build_observation(etf_norm, state, prev_weights, i, rot)
        # 预测仓位
        scale = rl_agent.predict(obs)
        # 应用缩放
        state.weights = apply_position_sizing(state.weights, scale)
    # ... 后续 ...
```

---

## 3. 数据需求

| 数据 | 必需? | 来源 |
|------|--------|------|
| ETF 日线 (close) | ✅ 已有 | - |
| ETF 日线 (high/low) | ⚠️ 可选 (RSRS 特征需) | Stage 16B 补充 |
| 训练 OOS 划分 | ✅ 已有 | - |
| 无风险利率序列 | ⚠️ 可选 (奖励用 0 代替) | - |

**最大瓶颈**: 计算资源
- PPO 训练 1 次: ~30 min (CPU) / ~5 min (GPU)
- 5 个 OOS 周期: ~2.5 h (CPU) / ~30 min (GPU)
- 推荐 GPU 环境 (本地或云)

---

## 4. 文件结构

```
v2/
├── (现有文件不动)
├── rl_position_v2.py          # 新增: RL 仓位管理
├── rl_train_v2.py             # 新增: 训练入口
├── rl_env_v2.py               # 新增: Gymnasium 环境
└── rl_obs_v2.py               # 新增: Observation 构造

common/
├── (现有文件不动)
└── rl_common.py               # 新增: 通用工具 (callback, eval)

models/
├── rl_v1.0/                   # 新增: 训练好的模型
│   ├── oos1.zip
│   ├── oos2.zip
│   └── ...

scripts/
├── train_rl_v2.py             # 新增: 训练脚本 (命令行)
└── eval_rl_v2.py              # 新增: 评估脚本
```

---

## 5. 依赖

```bash
# 新增依赖
pip install stable-baselines3>=2.0
pip install gymnasium>=0.29
pip install torch>=2.0  # 自动随 sb3 装
```

**安装验证**:
```python
import stable_baselines3
print(stable_baselines3.__version__)  # >= 2.0
import gymnasium
print(gymnasium.__version__)  # >= 0.29
```

---

## 6. 实施步骤 (建议 10-14 天)

### 步骤 1: 依赖安装 (0.5 天)
- [ ] pip install stable-baselines3 gymnasium
- [ ] 验证 torch / cuda
- [ ] 单元测试: 环境创建 + 随机动作

### 步骤 2: Gymnasium 环境 (2 天)
- [ ] 创建 `v2/rl_env_v2.py`
- [ ] 实现 `MomentumEnv(gym.Env)`:
  - `__init__`: 加载数据
  - `reset()`: 重置到训练起点
  - `step(action)`: 应用仓位, 计算奖励, 返回新状态
- [ ] 单元测试: tests/strategy/momentum_etf_rotation/test_rl_env.py

### 步骤 3: Observation 构造 (1 天)
- [ ] 创建 `v2/rl_obs_v2.py`
- [ ] 实现 `build_observation()`: 17 维向量
- [ ] 单元测试

### 步骤 4: PPO 训练 (2 天)
- [ ] 创建 `v2/rl_train_v2.py`
- [ ] 5 折 OOS 训练脚本
- [ ] 评估 + 记录指标
- [ ] 训练超参调优

### 步骤 5: 集成到 backtest (1.5 天)
- [ ] 创建 `v2/rl_position_v2.py`
- [ ] 集成到 `v2/backtest_v2.py`
- [ ] 添加 `RotationConfig.rl_agent: RLConfig`
- [ ] 集成测试

### 步骤 6: 鲁棒性测试 (1.5 天)
- [ ] 多随机种子 (5 seeds)
- [ ] 不同超参对比
- [ ] 与 v1.0 (无 RL) 对比
- [ ] 失败模式分析

### 步骤 7: 文档与提交 (0.5 天)
- [ ] `reports/momentum_etf_rotation/v2/stage16c_rl_position.md`
- [ ] `reports/momentum_etf_rotation/charts/v2/stage16c_*.html`
- [ ] `models/README.md` (模型说明)
- [ ] git commit (大文件用 LFS 或外链)

---

## 7. RLConfig 设计

```python
@dataclass
class RLConfig:
    """强化学习仓位管理配置 (Stage 16C)."""
    enabled: bool = False
    model_path: str = ""           # 预训练模型路径
    obs_window: int = 252          # 观察窗口 (1 年)
    action_low: float = 0.0        # 仓位下限
    action_high: float = 1.5       # 仓位上限
    action_smooth: float = 0.3     # 动作平滑 (避免剧烈调仓)
    # 奖励
    reward_sharpe_window: int = 21  # Sharpe 计算窗口
    turnover_penalty: float = 0.1  # 换手惩罚
    leverage_penalty: float = 0.05 # 杠杆惩罚
    dd_threshold: float = 0.10    # DD 阈值 (超过则惩罚)
    # 训练
    train_years: int = 2
    test_years: int = 1
    total_iter: int = 100_000
    seed: int = 42
```

---

## 8. 测试计划

### 8.1 单元测试
- `test_rl_env.py`: 环境 step, reset, reward
- `test_rl_obs.py`: 17 维向量构造
- `test_rl_position.py`: 仓位应用

### 8.2 训练测试
- 5 seeds 训练 5 OOS
- 平均 Calmar ≥ 1.65 (不退化)
- DD ≤ -4.5%
- 4/5 OOS 优于 v1.0

### 8.3 鲁棒性测试
- 不同 action 范围 (0~1.0 vs 0~1.5)
- 不同 reward 函数 (Sharpe vs Sortino vs IR)
- 不同 obs 维度 (10/17/25 维)

### 8.4 失败模式测试
- 极端事件 (2018-12, 2020-03, 2022-04)
- 数据缺失场景
- 模型推理失败 fallback

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 过拟合 | **高** | 高 | 多 OOS, 严格 Calmar 阈值, drop 模型 if OOS < baseline |
| 训练不稳定 | 高 | 中 | 多 seeds, 早停, 监控 |
| 模型失效 (数据漂移) | 中 | 高 | 月度重训, fallback 满仓 |
| 杠杆风险 | 中 | 中 | action_high=1.5 限制 |
| 换手率高 | 高 | 中 | action_smooth, cost penalty |
| 计算资源 | 中 | 低 | 本地 GPU 或云端训练 |

---

## 10. 验收标准 (严格)

| 指标 | 阈值 | 备注 |
|------|------|------|
| OOS Calmar (5 折平均) | ≥ 1.70 | vs v1.0 1.60 |
| OOS DD (5 折平均) | ≤ -4.0% | vs v1.0 -3.93% |
| OOS 优于 v1.0 比例 | ≥ 4/5 | 至少 4 个 OOS 优于 baseline |
| 训练-测试相关 | < 0.5 | 防过拟合 |
| 最大日回撤 | ≤ -8% | 极端情况 |
| 测试通过率 | 100% | |

**如果未达标**: 不合并到 v2 主分支, 保留为实验性 feature

---

## 11. 后续联动

- Stage 16A: RL 可作为子策略权重学习器
- Stage 16B: RSRS z 作为 RL 状态特征
- Stage 16D: RL 模型在 live trading 部署

---

## 12. 文档与资产

完成后产出:
1. `reports/momentum_etf_rotation/v2/stage16c_rl_position.md` (详细报告)
2. `reports/momentum_etf_rotation/charts/v2/stage16c_*.html` (5-6 个图表)
3. `models/rl_v1.0/` (训练好的模型, 5 个 OOS)
4. `scripts/train_rl_v2.py` (训练脚本)
5. `scripts/eval_rl_v2.py` (评估脚本)
6. `QuantNodes/strategy/momentum_etf_rotation/v2/rl_*.py` (实现代码)
7. 更新 STAGE_SUMMARY.md

---

## 13. 关键决策 (待用户确认)

- [ ] 是否引入 stable-baselines3 依赖? (会增加 ~200MB)
- [ ] 是否允许杠杆 (action_high=1.5)?
- [ ] 是否使用 GPU 训练? (本地/CPU 也能跑, 慢一些)
- [ ] OOS 折数 (推荐 5 折, 可调)
- [ ] 模型是否纳入 v2 主分支? (推荐: 通过验证后纳入, 否则实验)
