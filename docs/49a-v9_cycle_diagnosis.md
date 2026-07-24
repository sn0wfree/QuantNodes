# v9a — 周期诊断框架 (CPD: Cycle Position Diagnosis)

> **编号**: 49a
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49-v9_cycle_timing.md, docs/50-v9_current_cycle_state.md

---

## 1. CPD 框架概述

### 1.1 目标

**Cycle Position Diagnosis (CPD)** 是 v9 的核心独立产出, 回答:

> "当前经济/市场处于哪个周期, 在该周期的什么位置, 状态如何?"

### 1.2 三层诊断

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: 综合定位                                            │
│  ─ 整合 美林 + Pring + 多周期 → 综合判定                      │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Pring 多周期定位                                     │
│  ─ 10 年周期位置                                               │
│  ─ 多周期叠加 (Kitchin/Juglar/Kuznets/Kondratieff)            │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: 美林时钟 4 阶段识别                                  │
│  ─ GDP 同比 + CPI 同比 → Recovery/Overheat/Stagflation/Recession │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: 美林时钟识别

### 2.1 原始美林时钟定义

**Merrill Lynch Investment Clock (2004)** — 4 阶段:

| 阶段 | GDP | CPI | 推荐资产 |
|------|-----|-----|---------|
| **I. Recovery (复苏)** | ↑ | ↓ | 股票 > 债券 |
| **II. Overheat (过热)** | ↑ | ↑ | 商品 > 股票 |
| **III. Stagflation (滞胀)** | ↓ | ↑ | 现金 > 商品 |
| **IV. Recession (衰退)** | ↓ | ↓ | 债券 > 股票 |

### 2.2 中国数据映射

由于 GDP/CPI 月度数据可能滞后, 用**高频代理**:

| 美林指标 | 中国代理 | 频率 | 数据源 |
|---------|---------|------|--------|
| GDP 同比 | 工业增加值同比 或 PMI 同比 | 月度 | v7_14_X_panel |
| CPI 同比 | PPI 同比 (高频代理) | 月度 | v7_14_X_panel |
| 经济增长 | 社融同比 (领先指标) | 月度 | v7_14_X_panel |
| 通胀压力 | CPI 同比 | 月度 | v7_14_X_panel |

### 2.3 阶段识别算法

```python
def detect_merrill_phase(gdp_yoy: pd.Series, cpi_yoy: pd.Series, 
                          window: int = 6) -> pd.Series:
    """
    美林时钟 4 阶段识别.
    
    参数:
        gdp_yoy: GDP 同比 (3 月移动平均)
        cpi_yoy: CPI 同比 (3 月移动平均)
        window: 平滑窗口 (默认 6 月)
    
    返回:
        phase: 0=Recovery, 1=Overheat, 2=Stagflation, 3=Recession
    """
    # 平滑处理 (避免单月波动)
    gdp_smooth = gdp_yoy.rolling(window).mean()
    cpi_smooth = cpi_yoy.rolling(window).mean()
    
    # 中位数判定 (相对水平, 而非绝对水平)
    gdp_threshold = gdp_yoy.rolling(36).median()
    cpi_threshold = cpi_yoy.rolling(36).median()
    
    # 4 阶段判定
    phase = pd.Series(0, index=gdp_yoy.index)
    phase[(gdp_smooth > gdp_threshold) & (cpi_smooth < cpi_threshold)] = 0  # Recovery
    phase[(gdp_smooth > gdp_threshold) & (cpi_smooth >= cpi_threshold)] = 1  # Overheat
    phase[(gdp_smooth <= gdp_threshold) & (cpi_smooth >= cpi_threshold)] = 2  # Stagflation
    phase[(gdp_smooth <= gdp_threshold) & (cpi_smooth < cpi_threshold)] = 3  # Recession
    
    return phase
```

### 2.4 历史回测验证

**回溯 2010-2026 美林时钟定位**:

| 时期 | 实际 | 模型判定 | 一致性 |
|------|------|---------|--------|
| 2010-2011 | Overheat (4万亿后遗症) | ? | 待验证 |
| 2012-2015 | Recession/Recovery | ? | 待验证 |
| 2016-2017 | Recovery | ? | 待验证 |
| 2018-2019 | Overheat (贸易战) | ? | 待验证 |
| 2020 H1 | Recession (新冠) | ? | 待验证 |
| 2020 H2-2021 | Overheat (放水) | ? | 待验证 |
| 2022 | Stagflation (俄乌) | ? | 待验证 |
| 2023-2024 | Recovery | ? | 待验证 |
| 2025-2026 | ? | ? | 待验证 |

**目标**: ≥ 80% 时期与历史叙事一致。

---

## 3. Layer 2: Pring 多周期定位

### 3.1 Pring 周期理论概述

**Martin Pring (1931-)** 的多周期理论:

```
超长波: Kondratieff (50-54 年)
   ↓ 嵌套
长波: Kuznets (15-25 年)
   ↓ 嵌套
中波: Juglar (7-11 年)
   ↓ 嵌套
短波: Kitchin (3-5 年)
   ↓ 嵌套
超短波: Minor (9-12 月)
```

### 3.2 Pring 10 年周期 (Decennial Pattern)

**核心洞察**: 10 年周期中, 不同年份有季节性偏好:

| 年份位置 | 季节性 | 历史特征 |
|---------|--------|---------|
| 第 1-3 年 | 熊市主导 | 估值压缩, 利润下行 |
| 第 4-6 年 | 牛市启动 | 估值修复, 利润改善 |
| 第 7-9 年 | 牛市顶部 | 高估值, 政策紧缩 |
| 第 10 年 | 顶部/调整 | 泡沫破裂或盘整 |

**判定逻辑**:
```python
def pring_decennial_position(year: int, base_year: int = 2015) -> int:
    """
    返回当前年份在 10 年周期中的位置 (1-10).
    
    默认基准 2015 年为第 6 年 (熊市末/牛市启动).
    """
    # 10 年周期起点 = base_year - 5 (即 2010 年为第 1 年)
    cycle_year = ((year - (base_year - 5)) % 10) + 1
    return cycle_year
```

### 3.3 中国市场 10 年周期校准

**基准年选择**: 用 2005 年 (股权分置改革启动) 作为第 1 年起点:

| 年份 | 位置 | 实际 | 季节性 |
|------|------|------|--------|
| 2005 | 1 | 熊市末 (998) | 熊市主导 ✓ |
| 2006 | 2 | 大牛市启动 | 熊市主导 ✗ |
| 2007 | 3 | 大牛市顶 (6124) | 熊市主导 ✗ |

**问题**: 中国 2005-2007 是大牛市, 与 Pring 季节性不符。

**修正**: 用 **2015 年牛市顶** 作为第 7 年基准:
- 2015 = 第 7 年 (牛市顶部)
- 2016 = 第 8 年 (顶部/调整)
- 2017 = 第 9 年
- 2018 = 第 10 年 (贸易战, 调整)
- 2019 = 第 1 年 (新一轮启动)
- 2020 = 第 2 年 (新冠, 但快速反弹)
- 2021 = 第 3 年 (高点)
- 2022 = 第 4 年 (调整)
- 2023 = 第 5 年
- 2024 = 第 6 年 (9.24 反弹)
- 2025 = 第 7 年 (新一轮牛市?)
- 2026 = 第 8 年

### 3.4 多周期叠加定位

```python
def multi_cycle_position(date: pd.Timestamp, 
                          kitchin_phase: str,
                          juglar_phase: str,
                          kuznets_phase: str,
                          kontr_phase: str) -> str:
    """
    多周期相位综合判定.
    
    返回: 综合相位描述 + 建议配置
    """
    # 简化判定 (4 状态)
    if kitchin_phase == 'up' and juglar_phase == 'up':
        return 'STRONG_BULL'  # 双周期向上
    elif kitchin_phase == 'down' and juglar_phase == 'down':
        return 'STRONG_BEAR'  # 双周期向下
    elif kitchin_phase == 'up' and juglar_phase == 'down':
        return 'MIXED_RECOVERY'  # 短周期复苏, 中周期下行
    else:
        return 'TRANSITION'  # 转换期
```

---

## 4. Layer 3: 综合定位

### 4.1 综合评分

**总评分公式**:

$$
score_{total} = 0.4 \cdot score_{cycle} + 0.4 \cdot score_{coupling} + 0.2 \cdot score_{vix}
$$

| 评分维度 | 权重 | 来源 | 范围 |
|---------|------|------|------|
| 周期趋势 (cycle) | 40% | 多周期方向综合 | 0-40 |
| 共振分 (coupling) | 40% | Hilbert 相位差 | 0-40 |
| VIX 分 (vix) | 20% | VIX 倒数百分位 | 0-20 |

### 4.2 大盘信号生成

```python
def v9_signal(score: float, upper: int = 50, lower: int = 30) -> int:
    """
    评分 → 大盘信号.
    
    score >= upper → 1 (满仓)
    score <= lower → 0 (空仓)
    其他 → 保持上一状态 (迟滞, 防止抖动)
    """
    ...
```

### 4.3 状态输出结构

```python
@dataclass
class CycleState:
    """周期状态数据结构."""
    
    # 美林时钟
    merrill_phase: str           # 'Recovery'/'Overheat'/'Stagflation'/'Recession'
    merrill_phase_num: int       # 0/1/2/3
    gdp_yoy: float
    cpi_yoy: float
    
    # Pring 10 年周期
    pring_year: int              # 1-10
    pring_seasonality: str       # 'bear_dominant'/'bull_start'/'bull_top'/'adjustment'
    
    # 多周期定位
    kitchin_phase: str           # 'up'/'down'/'transition'
    juglar_phase: str
    kuznets_phase: str
    kontr_phase: str
    composite_phase: str         # 综合
    
    # 评分
    cycle_score: float           # 0-40
    coupling_score: float        # 0-40
    vix_score: float             # 0-20
    total_score: float           # 0-100
    
    # 信号
    v9_signal: int               # 0/1
    
    # 元数据
    report_date: pd.Timestamp
    data_through: pd.Timestamp
```

---

## 5. CPD 报告模板

### 5.1 报告结构

```
# v9 当前周期状态诊断报告
> 报告日期: YYYY-MM-DD
> 数据截至: YYYY-MM-DD

## 一、美林时钟定位
[当前阶段 + 驱动因素 + 历史对比]

## 二、Pring 周期定位
### 2.1 10 年周期位置
[当前年份 + 季节性 + 历史同期表现]

### 2.2 多周期叠加
[Kitchin + Juglar + Kuznets + Kondratieff]

## 三、周期耦合状态
[Hilbert 相位差 + 双相干系数]

## 四、当前评分
[总分 0-100 + 三维分解]

## 五、择时建议
[大盘信号 + 仓位建议]

## 六、未来 12 个月情景
[3 种情景概率与预期收益]
```

### 5.2 HTML 仪表盘布局

```
┌──────────────────────────────────────────────────────────────┐
│  v9 周期择时诊断仪表盘                       [日期]          │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 美林时钟      │  │ Pring 10年   │  │ 多周期定位    │      │
│  │ Phase: ??    │  │ 第 X 年      │  │ Kitchin 后段 │      │
│  │ GDP: 5.2% ↑  │  │ 季节: 启动   │  │ Juglar 中段   │      │
│  │ CPI: 0.5% ↓  │  │              │  │ Kontr 中段   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├──────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 评分时序 + 信号 + 历史大底/大顶标注                     │ │
│  └────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 周期分解      │  │ 相位耦合      │  │ 资产配置建议  │      │
│  │ 4 IMF 时序   │  │ Δφ(t)        │  │ 股票/债券/   │      │
│  │              │  │              │  │ 商品/现金     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 实施清单

### 6.1 CPD 模块文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/cpd/
├── __init__.py              # 模块入口, 暴露公共 API
├── merrill_clock.py         # 美林时钟识别
├── pring_cycles.py          # Pring 多周期定位
├── cycle_position.py        # 综合定位
└── diagnose.py              # 报告生成 (含 HTML 仪表盘)
```

### 6.2 接口设计

```python
# 公共 API
from QuantNodes.strategy.momentum_etf_rotation.v9.cpd import (
    detect_merrill_phase,
    pring_decennial_position,
    multi_cycle_position,
    diagnose_current_state,
    generate_html_dashboard,
)

# 主入口
state = diagnose_current_state(
    data_through='2026-05-31',
    macro_factors=macro_panel,
    index_data=hs300_index,
    vix_data=vix_series,
)
# 返回 CycleState dataclass

# HTML 输出
generate_html_dashboard(state, output_path='reports/.../dashboard.html')
```

---

## 7. 验证方案

### 7.1 美林时钟验证

| 时期 | 实际 (历史叙事) | 期望判定 | 通过标准 |
|------|----------------|---------|----------|
| 2010 | Overheat | Overheat | ✓ |
| 2012-2015 | Recession | Recession | ✓ |
| 2016-2017 | Recovery | Recovery | ✓ |
| 2020 H1 | Recession | Recession | ✓ |
| 2021 | Overheat | Overheat | ✓ |
| 2022 | Stagflation | Stagflation | ✓ |

**目标**: 至少 80% (6/7) 一致。

### 7.2 Pring 周期验证

**与已知顶部/底部对比**:
- 2015 年 6 月 (6124): Pring 第 7 年, 应判定"顶部"
- 2018 年底 (2440): Pring 第 10 年, 应判定"调整"
- 2019 年 (2440-3288): Pring 第 1 年, 应判定"启动"
- 2024 年 9 月: Pring 第 6 年, 应判定"启动"

**目标**: 至少 75% 一致。

### 7.3 HTML 仪表盘验证

| 检查项 | 标准 |
|--------|------|
| 美林时钟面板 | 显示阶段 + GDP + CPI + 推荐资产 |
| Pring 10 年面板 | 显示当前位置 + 季节性 |
| 多周期面板 | 显示 4 周期相位 |
| 评分时序 | 含历史大底/大顶标注 |
| 资产配置建议 | 输出权重百分比 |
| 响应式 | 支持桌面 + 移动 |
| 加载速度 | < 3 秒 |

---

## 8. 不做的事 (Out of Scope)

- ❌ 不做实时监控 (一次性诊断报告)
- ❌ 不做情景预测 (仅给出 3 种情景概率, 不做详细建模)
- ❌ 不做仓位寻优 (用学术默认权重 40+40+20)
- ❌ 不做跨市场对比 (仅 A 股)

---

## 9. 风险

| 风险 | 缓解 |
|------|------|
| 美林/Pring 框架不适用 A 股 | 仅作参考, 标注置信度 |
| 数据长度不够 | 在报告中明示数据长度限制 |
| 主观判断混入 | 用客观指标判定, 减少人为干预 |
| 模型不确定性 | 提供多情景概率, 而非单一判定 |

---

## 10. 参考文献

1. **Merrill Lynch (2004)**. *The Investment Clock: Making Money from Macro*.
2. **Pring, M. (2002)**. *The All-Season Investor*. McGraw-Hill.
3. **Pring, M. (2014)**. *The Pring Turner Market Rhythm*.

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中