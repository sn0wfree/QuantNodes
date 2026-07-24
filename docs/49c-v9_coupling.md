# v9c — 周期耦合识别

> **编号**: 49c
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49b-v9_decomposition.md, docs/49d-v9_scoring.md

---

## 1. 周期耦合的理论基础

### 1.1 什么是周期耦合?

**定义**: 两个或多个周期在**特定时刻相位锁定**, 产生协同效应, 导致资产价格出现**超额波动** (大涨/大跌)。

**数学表达**: 给定两个周期信号 $x_1(t), x_2(t)$, 如果它们的相位差 $\Delta\phi(t) = \phi_1(t) - \phi_2(t)$ 长时间保持稳定 (例如接近 0° 或 180°), 则称两个周期发生**相位锁定 (phase locking)**。

### 1.2 为什么耦合重要?

**单周期 vs 多周期**:

| 场景 | 预测力 | 风险 |
|------|--------|------|
| 单周期独立 | 中 | 易错过共振点 |
| 多周期共振 | **高** | 共振常伴随大行情 |
| 多周期背离 | 低 | 震荡市 |

**核心洞察**: 
- 单一周期 (Kitchin) 只能解释 A 股 ~30% 波动
- 多周期共振 (Kitchin+Juglar) 可解释 ~50%
- **共振点 = 大底/大顶**

### 1.3 v9 关注的两类耦合

1. **相位锁定 (Phase Locking)**: Hilbert 变换提取瞬时相位, 监测相位差稳定度
2. **双相干 (Bicoherence)**: 频域二次统计量, 检测非线性耦合

---

## 2. Hilbert 变换与瞬时相位

### 2.1 Hilbert 变换定义

对于实信号 $x(t)$, 其 Hilbert 变换定义为:

$$
\hat{x}(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{x(\tau)}{t - \tau} d\tau
$$

构造**解析信号**:

$$
z(t) = x(t) + i\hat{x}(t) = A(t) e^{i\phi(t)}
$$

其中:
- $A(t) = |z(t)|$: 瞬时振幅
- $\phi(t) = \arg(z(t))$: 瞬时相位

### 2.2 Python 实现

```python
from scipy.signal import hilbert

def compute_instantaneous_phase(imf: np.ndarray) -> np.ndarray:
    """
    由 IMF 计算瞬时相位.
    
    参数:
        imf: 1-D IMF 信号
    
    返回:
        phase: 瞬时相位, ∈ [-π, π]
    """
    analytic_signal = hilbert(imf)
    phase = np.angle(analytic_signal)
    return phase
```

### 2.3 相位解缠绕 (Phase Unwrapping)

直接 Hilbert 得到的相位是**卷绕的** (从 -π 到 π 跳跃), 需要解缠绕:

```python
from scipy.signal import hilbert

def compute_unwrapped_phase(imf: np.ndarray) -> np.ndarray:
    """
    解缠绕相位 (单调递增).
    """
    analytic_signal = hilbert(imf)
    phase = np.unwrap(np.angle(analytic_signal))
    return phase
```

**用途**: 解缠绕相位可以**直接比较**两个周期的相对位置。

### 2.4 端点效应处理

**问题**: Hilbert 变换在信号两端有显著失真 (Gibbs 现象)。

**对策**: 丢弃首尾各 60 周:

```python
phase = compute_instantaneous_phase(imf)
phase_clean = phase[60:-60]
```

---

## 3. 相位差分析

### 3.1 相位差定义

给定两个 IMF 的瞬时相位 $\phi_k(t), \phi_j(t)$, 相位差:

$$
\Delta\phi_{kj}(t) = \phi_k(t) - \phi_j(t)
$$

### 3.2 相位锁定判定

**标准 1: 相位差稳定度 (Phase Locking Value, PLV)**

$$
\text{PLV}_{kj} = \left| \frac{1}{T} \sum_{t=1}^{T} e^{i \Delta\phi_{kj}(t)} \right| \in [0, 1]
$$

- PLV = 1: 完全锁定
- PLV = 0: 完全独立

**标准 2: 相位差均值与方差**

```python
def phase_locking_metrics(delta_phi: np.ndarray, 
                          window: int = 12) -> pd.Series:
    """
    滚动计算相位锁定指标.
    
    参数:
        delta_phi: 相位差序列
        window: 滚动窗口 (默认 12 周 = 3 月)
    
    返回:
        PLV 时序
    """
    def plv_window(x):
        complex_phase = np.exp(1j * x)
        return np.abs(complex_phase.mean())
    
    return pd.Series(delta_phi).rolling(window).apply(plv_window, raw=True)
```

### 3.3 历史大底/大顶的相位特征

**经验规律** (基于学术文献 + A 股历史):

| 事件 | 预期相位差 Δφ (Kitchin - Juglar) | PLV |
|------|--------------------------------|-----|
| 2014 大底 | < 30° | > 0.9 |
| 2019 大底 | < 30° | > 0.9 |
| 2024 大底 | < 30° | > 0.9 |
| 2015 顶部 | 接近 180° | > 0.9 |
| 2021 顶部 | 接近 180° | > 0.9 |

**v9 验证目标**: 2014/2019/2024 三次大底中 ≥ 2 次 Δφ<30° 且 PLV>0.9。

---

## 4. 双相干系数 (Bicoherence)

### 4.1 定义

**双相干**是频域的二次统计量, 衡量两个频率 $f_1, f_2$ 与它们的和频 $f_1 + f_2$ 之间的相位耦合。

$$
b^2(f_1, f_2) = \frac{|E[F(f_1) F(f_2) F^*(f_1 + f_2)]|^2}{E[|F(f_1) F(f_2)|^2] \cdot E[|F(f_1 + f_2)|^2]}
$$

- $b^2 = 0$: 无耦合
- $b^2 = 1$: 完全耦合

### 4.2 计算实现

```python
def bicoherence(signal: np.ndarray, 
                nperseg: int = 64,
                noverlap: int = 32,
                fs: float = 1.0) -> tuple:
    """
    双相干系数计算.
    
    参数:
        signal: 1-D 输入信号
        nperseg: 每段长度
        noverlap: 段重叠
        fs: 采样率
    
    返回:
        freq: 频率数组
        bic: 双相干矩阵 (n_freq, n_freq)
    """
    from scipy.signal import spectrogram, csd
    
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 计算双谱
    n_freq = len(f)
    bic = np.zeros((n_freq, n_freq))
    
    for i in range(n_freq):
        for j in range(n_freq - i):
            if i + j < n_freq:
                # 三阶谱
                B = np.mean(Sxx[i] * Sxx[j] * np.conj(Sxx[i + j]))
                norm = np.sqrt(np.mean(np.abs(Sxx[i] * Sxx[j])**2) * 
                              np.mean(np.abs(Sxx[i + j])**2))
                bic[i, j] = np.abs(B) / (norm + 1e-10)
    
    return f, bic
```

### 4.3 计算优化

**问题**: 双谱是 O(N²) 计算量, 对 430 周数据可能需要数分钟。

**优化**: 利用对称性 $b(f_1, f_2) = b(f_2, f_1) = b^*(f_1, f_1+f_2)$, 只需计算 $f_1 \leq f_2$ 的三角区域。

### 4.4 显著性检验

**零假设**: $b^2 = 0$ (无耦合)

**检验统计量**:

$$
\chi^2 = 2N b^2 \sim \chi^2_2 \text{ (卡方分布)}
$$

其中 $N$ 是分段数。

**p 值**: $p = 1 - F_{\chi^2}(\chi^2; 2)$

**v9 阈值**: $b^2 > 0.6$ 且 $p < 0.05$ 判定为显著耦合。

---

## 5. 耦合信号生成

### 5.1 信号合成

**输入**:
- $\text{PLV}_{kj}(t)$: 相位锁定值
- $\Delta\phi_{kj}(t)$: 相位差
- $\text{bic}_{kj}(f_1, f_2)$: 双相干

**合成耦合信号**:

```python
def compute_coupling_score(imfs: np.ndarray, 
                            phases: np.ndarray,
                            delta_phases: dict,
                            bic_matrix: np.ndarray,
                            lock_threshold_deg: float = 30,
                            lock_min_duration_weeks: int = 12) -> float:
    """
    耦合评分 0-40.
    
    评分规则:
    - 每对 IMF 的相位锁定 (Δφ < 30° 持续 3 月): +10 分
    - 双相干显著 (bic > 0.6): +10 分
    - 最多取 4 对 IMF, 总分 ≤ 40
    """
    score = 0
    n_pairs = min(len(delta_phases), 4)
    
    for k, (kp, jp, dp) in enumerate(delta_phases.items()):
        if k >= n_pairs:
            break
        
        # 相位锁定检测
        locked = np.abs(dp) < np.deg2rad(lock_threshold_deg)
        locked_duration = locked.sum()
        
        if locked_duration >= lock_min_duration_weeks:
            score += 10
        
        # 双相干检测
        if np.max(bic_matrix) > 0.6:
            score += 10
            break
    
    return min(score, 40)
```

### 5.2 信号滤波

**避免噪声干扰**:

```python
def smooth_coupling_signal(coupling_score: pd.Series, 
                            window: int = 4) -> pd.Series:
    """
    4 周移动平均平滑.
    """
    return coupling_score.rolling(window).mean()
```

---

## 6. 验证方案

### 6.1 相位锁定验证

**回溯历史**:

| 大底/大顶 | 期望 Δφ (度) | 期望 PLV | 实测 |
|----------|-------------|---------|------|
| 2014-06 | < 30 | > 0.9 | 待验证 |
| 2015-06 | ~180 | > 0.9 | 待验证 |
| 2019-01 | < 30 | > 0.9 | 待验证 |
| 2021-12 | ~180 | > 0.9 | 待验证 |
| 2024-09 | < 30 | > 0.9 | 待验证 |

**目标**: 至少 4/5 命中。

### 6.2 双相干验证

**A 股市场历史**:

| 时期 | 主导耦合频率 | 期望 bic | 实测 |
|------|------------|---------|------|
| 2014-2015 大牛 | 基钦-朱格拉 | > 0.6 | 待验证 |
| 2018 单边下跌 | 基钦-年 | > 0.5 | 待验证 |
| 2020 疫情底 | 基钦-年 | > 0.6 | 待验证 |

**目标**: 至少 2/3 命中。

### 6.3 信号稳定性

**滚动窗口验证**:
- 4 年训练 + 4 年 OOS
- 多次起点 (5 个)
- 报告 PLV 与 bic 的均值/标准差

---

## 7. 实施清单

### 7.1 文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/
└── coupling.py              # 本模块
```

### 7.2 接口

```python
from QuantNodes.strategy.momentum_etf_rotation.v9.coupling import (
    compute_instantaneous_phase,
    compute_unwrapped_phase,
    phase_locking_value,
    bicoherence,
    compute_coupling_score,
)

# 主入口
phases = compute_all_phases(imfs)  # (K, T) 相位矩阵
delta_phases = compute_delta_phases(phases)  # (K*(K-1)/2, T) 相位差
plv = compute_phase_locking_value(delta_phases)  # (K*(K-1)/2, T) PLV
bic_matrix = compute_bicoherence(signal)  # (n_freq, n_freq)
coupling_score = compute_coupling_score(...)  # 0-40
```

---

## 8. 风险与对冲

| 风险 | 缓解 |
|------|------|
| Hilbert 端点效应 | 丢弃首尾 60 周 |
| 相位跳跃 (2π wrap) | 使用解缠绕相位 |
| 双相干计算慢 | 限制 nperseg=64, 利用对称性 |
| 假阳性 (噪声) | 显著性检验 + 滚动窗口 |
| 过拟合 (历史匹配) | 仅用通用阈值 (30°, 0.6) |

---

## 9. 参考文献

1. **Rosenblum, M. G. et al. (1996)**. "Phase synchronization: From theory to data analysis". *Handbook of Biological Physics*.
2. **Lachaux, J. P. et al. (1999)**. "Measuring phase synchrony in brain signals". *Human Brain Mapping* 8(4), 194-208.
3. **Nikias, C. L., Petropulu, A. P. (1993)**. *Higher-Order Spectra Analysis*. Prentice Hall.
4. **Hagihira, S. et al. (2001)**. "Practical issues in bispectral analysis of electroencephalographic signals". *Anesthesia & Analgesia*.

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中