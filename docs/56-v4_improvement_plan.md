# v4 风格轮动/因子轮动/行业轮动 — 改进计划

> 日期: 2026-07-22
> 状态: 📋 实施中
> 基于: v4 现有代码 + v5/v6 行业轮动经验

---

## 一、v4 现状

### 已实现

| 模块 | 功能 | 状态 |
|------|------|------|
| **Style Rotation** | 5 风格组 (大盘/中盘/成长/科创/红利) 动量轮动 | ✅ 完整 |
| **Smart Beta** | 7 个 Smart Beta ETF 选优 | ✅ 完整 |
| **Factor Timing** | 6 因子 IC 驱动 + regime 条件 | ✅ 完整 (v4D) |
| **Regime Detection** | HMM 3 状态 (bull/bear/transition) | ✅ 代码存在 |
| **LW Factor Timing** | Ledoit-Wolf 协方差收缩 + MVO | ✅ 代码存在 (默认关闭) |

### 未完成 / 待实施

| 模块 | 功能 | 状态 |
|------|------|------|
| **v4E** | HMM-only 因子择时 | ❌ 标记"待实施" |
| **v4F** | IC + HMM 融合择时 | ❌ 标记"待实施" |
| **Industry Rotation** | 行业轮动 (v5 才有) | ❌ v4 不包含 |
| **Valuation Factor** | 估值因子 | ❌ 未实现 |
| **Fundamental Factor** | 基本面因子 | ❌ 未实现 |
| **Correlation Constraint** | 相关性约束 | ❌ 未实现 |

### 已知问题

1. **low_vol 因子已禁用**: IC vs forward 相关 -0.454 (反指因子)
2. **Regime 双重实现**: Style Rotation 用简单动量阈值, HMM 用复杂模型, 未统一
3. **因子→ETF 映射硬编码**: Momentum/Reversal 映射到所有 5 个风格 ETF
4. **Smart Beta 评分静态**: 不随 regime 或 factor timing 调整
5. **红利底线 20% 无条件**: 即使红利因子 IC 为负也强制配置
6. **LW 模块完整但默认关闭**: 需要对比研究

---

## 二、改进计划

### 阶段 1: 完善 v4 核心 (风格轮动 + 因子轮动)

#### 1.1 实现 v4E (HMM 因子择时)

**文件**: `v4/factor_timing_v4.py` (修改)

**逻辑**:
```python
# 当前 v4D: IC 驱动
weight[name] = max(0, IC[name] + base)^power

# v4E 新增: HMM regime 调整
if regime == 'bull':
    weight['momentum'] *= 1.5  # 牛市加动量
    weight['value'] *= 0.7     # 牛市减价值
elif regime == 'bear':
    weight['dividend'] *= 1.5  # 熊市加红利
    weight['quality'] *= 1.3   # 熊市加质量
    weight['momentum'] *= 0.5  # 熊市减动量
```

**参数**:
- `use_hmm: bool = True` (启用 HMM)
- `hmm_regime_factors: dict` (regime→因子权重调整)

#### 1.2 实现 v4F (IC + HMM 融合)

**文件**: `v4/factor_timing_v4.py` (修改)

**逻辑**:
```python
# IC 权重
ic_weight[name] = max(0, IC[name] + base)^power

# HMM 权重
hmm_weight[name] = regime_factor_weight[regime][name]

# 融合
final_weight = alpha * ic_weight + (1-alpha) * hmm_weight
```

**参数**:
- `fusion_alpha: float = 0.7` (IC 权重占 70%)

#### 1.3 统一 Regime 检测

**文件**: `v4/style_rotation_v4.py` (修改)

**逻辑**:
```python
# 当前: 简单动量阈值
if short_mom > 0.05 and long_mom > 0.10:
    regime = 'bull'

# 改进: 用 HMM
hmm_detector = RegimeDetector()
regime = hmm_detector.predict(current_features)
```

#### 1.4 添加估值因子

**文件**: `v4/factor_ic.py` (修改)

**逻辑**:
```python
# 估值代理: 过去 52 周累计收益的反向 (越跌越便宜)
def value_proxy(returns, window=52):
    cum_ret = (1 + returns).rolling(window).apply(np.prod) - 1
    return -cum_ret  # 反向: 跌得多 = 估值低
```

#### 1.5 添加基本面因子

**文件**: `v4/factor_ic.py` (修改)

**逻辑**:
```python
# 基本面代理: Sharpe ratio (高质量 ≈ 高 ROE)
def quality_proxy(returns, window=26):
    mean_ret = returns.rolling(window).mean()
    std_ret = returns.rolling(window).std()
    return mean_ret / (std_ret + 1e-10)
```

### 阶段 2: 添加行业轮动

#### 2.1 集成 v5 的 11 因子

**文件**: `v4/industry_rotation_v4.py` (新建)

**逻辑**:
```python
class IndustryRotationV4(SubStrategy):
    """v4 行业轮动: 继承 v5 的 11 因子 + 添加 regime 条件"""

    def select(self, date, pp):
        # 1. 计算 11 因子 (复用 v5 的 FactorEngine)
        factor_panel = self.compute_factors(date)

        # 2. 横截面 z-score
        z_scores = cross_section_zscore(factor_panel)

        # 3. 综合得分
        composite = z_scores.mean(axis=1)

        # 4. Regime 条件
        regime = self.get_regime(date)
        if regime == 'bear':
            # 熊市: 偏防御型行业 (低波动、高分红)
            defensive_mask = self.get_defensive_industries()
            composite[~defensive_mask] *= 0.5

        # 5. Top-N 选优
        top_n = composite.nlargest(self.config.top_n)

        return top_n.index.tolist()
```

#### 2.2 添加估值/基本面到行业轮动

**文件**: `v4/industry_rotation_v4.py` (修改)

**逻辑**:
```python
# 在 11 因子基础上, 添加:
# 12. 估值因子 (收益率反向)
# 13. 基本面因子 (Sharpe)

factors['f12_value'] = -cumulative_return(returns, window=52)
factors['f13_quality'] = sharpe_ratio(returns, window=26)

# 综合得分 = 13 因子等权
composite = factors.mean(axis=1)
```

#### 2.3 添加 Regime 条件

**文件**: `v4/industry_rotation_v4.py` (修改)

**逻辑**:
```python
# 行业分类
DEFENSIVE_INDUSTRIES = ['银行', '公用事业', '食品饮料']  # 防御型
GROWTH_INDUSTRIES = ['电子', '计算机', '新能源']  # 进攻型

# Regime 条件
if regime == 'bull':
    # 牛市: 进攻型行业权重 1.5x
    for ind in GROWTH_INDUSTRIES:
        scores[ind] *= 1.5
elif regime == 'bear':
    # 熊市: 防御型行业权重 1.5x
    for ind in DEFENSIVE_INDUSTRIES:
        scores[ind] *= 1.5
```

#### 2.4 添加相关性约束

**文件**: `v4/industry_rotation_v4.py` (修改)

**逻辑**:
```python
# 计算行业间相关系数
corr_matrix = returns[sector_codes].rolling(52).corr()

# 剔除冗余
selected = []
for code in top_n_sorted:
    if not selected:
        selected.append(code)
    else:
        # 检查与已选行业的相关性
        max_corr = max([corr_matrix.loc[code, s] for s in selected])
        if max_corr < 0.7:  # 相关性阈值
            selected.append(code)
```

---

## 三、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `v4/factor_ic.py` | 修改 | 添加估值/基本面因子 |
| `v4/factor_timing_v4.py` | 修改 | 添加 v4E/v4F 模式 |
| `v4/style_rotation_v4.py` | 修改 | 用 HMM 替代简单阈值 |
| `v4/multi_strategy_v4.py` | 修改 | 启用 v4E/v4F 模式 |
| `v4/industry_rotation_v4.py` | **新建** | 行业轮动子策略 |
| `v4/__init__.py` | 修改 | 导出新模块 |

---

## 四、预期性能

| 版本 | 当前 OOS Sharpe | 改进后预期 |
|------|----------------|-----------|
| v4 (style+smart beta) | ~0.3-0.5 | **0.6-0.8** |
| v4+industry rotation | N/A | **0.7-0.9** |
| v4+v5 融合 | N/A | **0.8-1.1** |

---

## 五、实施顺序

1. **阶段 1.1-1.2**: 实现 v4E/v4F (因子择时增强)
2. **阶段 1.3**: 统一 Regime 检测
3. **阶段 1.4-1.5**: 添加估值/基本面因子
4. **阶段 2.1**: 集成 v5 的 11 因子
5. **阶段 2.2-2.4**: 添加 Regime 条件 + 相关性约束
6. **回测验证**: 对比改进前后性能
