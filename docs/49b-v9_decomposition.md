# v9b — 多尺度周期分解算法

> **编号**: 49b
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49-v9_cycle_timing.md, docs/49c-v9_coupling.md

---

## 1. 问题定义

### 1.1 输入与输出

**输入**:
- 时序信号: $x(t)$, $t = 1, 2, \ldots, T$
- 目标周期数: $K = 4$ (季度, 年, 基钦, 朱格拉+)
- 期望中心频率: $\{f_k\}_{k=1}^{K}$

**输出**:
- $K$ 个本征模态函数 (IMF): $x(t) = \sum_{k=1}^{K} \text{IMF}_k(t) + r(t)$
- 每个 IMF 的中心频率, 带宽, 能量

### 1.2 为什么不用 FFT?

**FFT 的局限** (来自学术共识):

| 问题 | 后果 |
|------|------|
| 全局平稳假设 | 周期漂移被抹平 |
| 固定分辨率 | 无法同时看长期+短期 |
| 频谱泄漏 | 真实周期被噪声淹没 |
| 无时频定位 | 不知道"何时"是某种周期 |

**v9 替代方案**: 自适应分解算法 (CEEMDAN, VMD), 时频局部化。

---

## 2. VMD (Variational Mode Decomposition)

### 2.1 算法原理

**Dragomiretskiy & Zosso (2014)** 提出的变分模态分解。

**核心思想**: 求解以下约束变分问题:

$$
\min_{\{u_k\}, \{\omega_k\}} \sum_{k=1}^{K} \left\| \partial_t \left[ \delta(t) + \frac{j}{\pi t} \right] u_k(t) e^{-j\omega_k t} \right\|_2^2
$$

$$
\text{s.t.} \quad \sum_{k=1}^{K} u_k(t) = x(t)
$$

其中:
- $u_k(t)$: 第 $k$ 个模态 (复信号)
- $\omega_k$: 第 $k$ 个模态的中心频率
- $\delta(t)$: Dirac 脉冲
- $\partial_t$: 时间偏导

**直觉**: 每个 IMF 应该是"窄带"信号 (带宽最小化)。

### 2.2 算法步骤

```
1. 初始化: {u_k}, {ω_k}, λ (拉格朗日乘子)
2. 迭代直至收敛:
   for k = 1..K:
       u_k ← FFT-domain Wiener filter update
       ω_k ← update based on current u_k
   λ ← update with augmented Lagrangian
3. 输出: {u_k} (取实部), {ω_k}
```

### 2.3 Python 实现

```python
import numpy as np
from vmdpy import VMD

def vmd_decompose(signal: np.ndarray, 
                  alpha: float = 1000,
                  K: int = 4,
                  DC: int = 0,
                  init: int = 1,
                  tol: float = 1e-7) -> tuple:
    """
    VMD 多尺度分解.
    
    参数:
        signal: 1-D 输入信号
        alpha: 带宽约束 (大=窄带, 小=宽带)
        K: 模态数
        DC: 是否包含 DC 分量
        init: 初始化方式 (1=均匀, 2=随机)
        tol: 收敛阈值
    
    返回:
        imfs: (K, T) 模态数组
        omega: (K,) 中心频率 (归一化, ∈ [0, 0.5])
    """
    imfs, omega, _ = VMD(signal, alpha, K, DC, init, tol)
    return imfs, omega
```

### 2.4 参数选择

| 参数 | 默认值 | 调参建议 |
|------|--------|---------|
| alpha | 1000 | 调小=带宽更宽, 调大=更窄 |
| K | 4 | 太少丢周期, 太多过拟合 |
| init | 1 | 1=频率均匀初始化 (推荐) |
| tol | 1e-7 | 1e-6 足够, 不必太严 |

### 2.5 优缺点

**优点**:
- ✅ 收敛快 (< 1 秒, 430 周数据)
- ✅ 抗模态混叠
- ✅ 频率分辨率高
- ✅ 易于控制模态数

**缺点**:
- ❌ 需预设模态数 K
- ❌ 对噪声敏感
- ❌ alpha 难调

---

## 3. CEEMDAN (Complete EEMD with Adaptive Noise)

### 3.1 算法原理

**Torres et al. (2011)** 提出的完备集成经验模态分解。

**核心思想**: 在 EMD 基础上, 添加**自适应白噪声**, 通过多次试验求平均, 消除模态混叠。

**算法**:
```
1. 添加自适应白噪声到原始信号
2. 对每个噪声版本做 EMD 分解
3. 求所有分解结果的平均
4. 第一个 IMF = 平均后的 IMF1
5. 残差 = 原信号 - IMF1
6. 重复 1-5, 得到 IMF2, IMF3, ...
```

### 3.2 Python 实现

```python
from PyEMD import CEEMDAN

def ceemdan_decompose(signal: np.ndarray, 
                      trials: int = 100,
                      max_imfs: int = 4) -> np.ndarray:
    """
    CEEMDAN 多尺度分解.
    
    参数:
        signal: 1-D 输入信号
        trials: 噪声试验次数 (默认 100)
        max_imfs: 最多模态数
    
    返回:
        imfs: (K, T) 模态数组, K <= max_imfs
    """
    ceemdan = CEEMDAN(trials=trials)
    imfs = ceemdan(signal, max_imfs)
    return imfs
```

### 3.3 参数选择

| 参数 | 默认值 | 调参建议 |
|------|--------|---------|
| trials | 100 | 太少噪声残留, 太多计算慢 |
| max_imfs | 4 | 与 VMD 保持一致 |
| epsilon | 0.05 | 白噪声标准差 (相对信号) |

### 3.4 优缺点

**优点**:
- ✅ 自适应, 无需预设基函数
- ✅ 模式分离彻底
- ✅ 完备性: 残差 = 趋势 + 噪声

**缺点**:
- ❌ 计算慢 (100 trials × 多次 EMD)
- ❌ 模式数不确定 (依赖停止准则)
- ❌ 调试困难

---

## 4. 双算法对比

### 4.1 对比维度

| 维度 | VMD | CEEMDAN |
|------|-----|---------|
| 计算速度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 模式分离 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 完备性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 参数敏感度 | 中 (alpha) | 中 (trials) |
| 调试友好度 | ⭐⭐⭐⭐ | ⭐⭐ |

### 4.2 选择规则

```python
def select_decomposition(imfs_vmd: np.ndarray, 
                         omega_vmd: np.ndarray,
                         imfs_ceemdan: np.ndarray) -> str:
    """
    选择分解算法.
    
    返回: 'vmd' 或 'ceemdan'
    """
    # 指标 1: 频谱集中度 (峰度)
    kurt_vmd = np.mean([kurtosis(imf) for imf in imfs_vmd])
    kurt_ceemdan = np.mean([kurtosis(imf) for imf in imfs_ceemdan])
    
    # 指标 2: 相邻 IMF 中心频率比
    ratio_vmd = np.min(np.diff(omega_vmd)) / np.max(np.diff(omega_vmd))
    
    # 指标 3: 重构误差
    recon_vmd = imfs_vmd.sum(axis=0)
    recon_ceemdan = imfs_ceemdan.sum(axis=0)
    
    # 加权评分
    score_vmd = kurt_vmd + ratio_vmd + 1 / np.std(recon_vmd)
    score_ceemdan = kurt_ceemdan + 1 / np.std(recon_ceemdan)
    
    return 'vmd' if score_vmd > score_ceemdan else 'ceemdan'
```

---

## 5. 周期验证

### 5.1 IMF 周期验证

**目标**: 每个 IMF 的平均周期应接近目标:

| IMF | 目标周期 | 容差 |
|-----|---------|------|
| IMF1 (季度) | 12 周 | 8-16 周 |
| IMF2 (年) | 48 周 | 40-60 周 |
| IMF3 (基钦) | 160 周 (3.2 年) | 120-240 周 |
| IMF4 (朱格拉+) | 480 周 (10 年) | 320-960 周 |

### 5.2 周期计算

```python
def compute_period(omega: float, sample_rate: float = 1) -> float:
    """
    由归一化频率计算周期.
    
    参数:
        omega: 归一化频率 ∈ [0, 0.5]
        sample_rate: 采样率 (周频=1)
    
    返回:
        period: 周期 (单位与 sample_rate 倒数一致)
    """
    if omega < 1e-10:
        return np.inf
    return sample_rate / (omega * 2)
```

---

## 6. 输入信号选择

### 6.1 用什么数据做分解?

**方案 A: 价格对数 (推荐用于周期识别)**

```python
log_price = np.log(hs300_index)
imfs = vmd_decompose(log_price)
```

**优点**: 直观, 周期可视化清晰
**缺点**: 包含趋势项, IMF4 可能包含长期趋势

**方案 B: 对数收益率 (推荐用于择时)**

```python
log_returns = np.log(hs300_index).diff().dropna()
imfs = vmd_decompose(log_returns)
```

**优点**: 平稳, 适合统计分解
**缺点**: 周期绝对幅度小, 信噪比低

**方案 C: HP 滤波残差 (推荐) ← v9 默认**

```python
from statsmodels.tsa.filters.hp_filter import hpfilter

cycle, trend = hpfilter(np.log(hs300_index), lamb=100)
imfs = vmd_decompose(cycle)
```

**优点**: 趋势被剔除, 周期信号更纯
**缺点**: λ 选择影响结果

### 6.2 v9 默认方案

```
HP 滤波残差 → VMD (默认) / CEEMDAN (备选) → 4 IMF
```

---

## 7. 实施清单

### 7.1 文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/
└── decompose.py             # 本模块
```

### 7.2 接口

```python
from QuantNodes.strategy.momentum_etf_rotation.v9.decompose import (
    decompose_signal,         # 主入口
    vmd_decompose,            # VMD 实现
    ceemdan_decompose,        # CEEMDAN 实现
    select_best_decomposition,
    compute_imf_periods,
)

# 主入口
imfs, omega, method = decompose_signal(
    signal=hp_cycle_residual,
    K=4,
    method='both',  # 'vmd' | 'ceemdan' | 'both'
)
```

### 7.3 测试用例

```python
def test_vmd_decompose():
    """测试 VMD 分解."""
    # 构造合成信号: 2 个正弦波 + 噪声
    t = np.linspace(0, 10, 500)
    signal = np.sin(2 * np.pi * t / 2) + 0.5 * np.sin(2 * np.pi * t / 5) + 0.1 * np.random.randn(500)
    
    imfs, omega = vmd_decompose(signal, K=2)
    
    assert imfs.shape == (2, 500)
    assert np.all(np.diff(omega) > 0)  # 频率递增
```

---

## 8. 风险与对冲

| 风险 | 缓解 |
|------|------|
| 模式数选择错误 | K=4 基于领域知识, 不在数据上寻优 |
| 频率混叠 | 双算法对比, 选最优 |
| 端点效应 | 丢弃首尾 60 周 |
| 噪声敏感 | 先做 HP 滤波 |
| 计算时间长 | CEEMDAN trials=100, VMD 默认参数 |

---

## 9. 参考文献

1. **Huang, N. E. et al. (1998)**. "The Empirical Mode Decomposition and the Hilbert Spectrum". *Proc. R. Soc. Lond. A* 454, 903-995.
2. **Wu, Z., Huang, N. E. (2009)**. "Ensemble Empirical Mode Decomposition". *Advances in Adaptive Data Analysis* 1(1), 1-41.
3. **Torres, M. E. et al. (2011)**. "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise". *IEEE ICASSP*, 4144-4147.
4. **Dragomiretskiy, K., Zosso, D. (2014)**. "Variational Mode Decomposition". *IEEE Trans. Signal Processing* 62(3), 531-544.

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中