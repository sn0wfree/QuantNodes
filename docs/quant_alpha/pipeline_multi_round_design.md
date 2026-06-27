# Alpha Pipeline 多轮迭代设计文档

> **版本**: v1.0  
> **日期**: 2026-06-27  
> **状态**: 设计完成，待实现

## 目录

- [1. 概述](#1-概述)
- [2. 流程架构](#2-流程架构)
- [3. 评估标准](#3-评估标准)
- [4. 终止条件](#4-终止条件)
- [5. 反馈机制](#5-反馈机制)
- [6. 结果保存](#6-结果保存)
- [7. 配置参数](#7-配置参数)
- [8. 实现计划](#8-实现计划)

---

## 1. 概述

### 1.1 背景

当前 Alpha Pipeline 采用单轮流程：`Alpha-GPT → MCTS → 去重 → Wiki`。这种方式存在以下问题：

1. **单向流程**：只有 Alpha-GPT → MCTS，没有反向反馈
2. **无法迭代优化**：因子质量受限于单轮生成
3. **缺乏终止机制**：无法根据收敛情况自动停止

### 1.2 目标

实现多轮迭代机制，通过反馈循环持续优化因子质量：

1. **多轮迭代**：支持 N 轮迭代，每轮生成新因子
2. **反馈闭环**：将 MCTS 结果反馈给 Alpha-GPT，指导下一轮生成
3. **自动终止**：基于收敛条件自动停止，避免无效迭代
4. **完整记录**：保存所有中间结果，便于分析和追溯

### 1.3 参考框架

| 框架 | 迭代机制 | 终止条件 | 反馈机制 |
|------|----------|----------|----------|
| **Alpha-GPT** | 5 轮固定迭代 | 固定轮次 | Reflector → IdeaGenerator |
| **Qlib** | 无自动化迭代 | 手动停止 | 无 |
| **EvolutionLoop** | 遗传算法迭代 | 早停（patience） | 适应度反馈 |
| **MCTS** | 树搜索迭代 | 固定迭代次数 | 5 通道反馈 |

---

## 2. 流程架构

### 2.1 主流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Alpha Pipeline 多轮迭代流程                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        初始化阶段                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ 预计算       │  │ 初始化       │  │ 初始化       │              │   │
│  │  │ 前瞻收益     │  │ 早停机制     │  │ 反馈为空     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        主循环 (Round 1 ~ N)                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Round i 开始                              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    ↓                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Stage 1: Alpha-GPT                                          │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │ IdeaGenerator│→│ FormulaTrans  │→│  Evaluator   │      │   │   │
│  │  │  │ (想法生成)   │  │ (公式翻译)   │  │ (评估器)     │      │   │   │
│  │  │  │ temp=0.8     │  │ temp=0.4     │  │ IC/IR计算    │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  │         ↑                                      ↓            │   │   │
│  │  │  ┌──────────────┐                        ┌──────────────┐  │   │   │
│  │  │  │ Reflector    │←───────────────────────│ 评估结果     │  │   │   │
│  │  │  │ (反思器)     │                        │              │  │   │   │
│  │  │  │ temp=0.6     │                        └──────────────┘  │   │   │
│  │  │  └──────────────┘                                          │   │   │
│  │  │         ↓                                                   │   │   │
│  │  │  ┌──────────────┐                                          │   │   │
│  │  │  │ Feedback     │  ← 第2轮起注入上轮反馈                    │   │   │
│  │  │  │ (反馈注入)   │                                          │   │   │
│  │  │  └──────────────┘                                          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    ↓                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Stage 2: MCTS 搜索                                         │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │ Select       │→│ Expand       │→│ Evaluate     │      │   │   │
│  │  │  │ (UCB1选择)   │  │ (生成子公式) │  │ (5通道反馈)  │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  │                        ↑                    ↓               │   │   │
│  │  │                        │           ┌──────────────┐         │   │   │
│  │  │                        └───────────│ Backpropagate│         │   │   │
│  │  │                                    │ (回传评分)   │         │   │   │
│  │  │                                    └──────────────┘         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    ↓                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Stage 3: 合并去重                                           │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │ 收集         │→│ Mutual IC    │→│ 排序取       │      │   │   │
│  │  │  │ Alpha-GPT    │  │ 去重         │  │ Top-K        │      │   │   │
│  │  │  │ + MCTS       │  │ (阈值=0.7)   │  │              │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    ↓                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Stage 4: 评估与反馈                                         │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │ 计算         │  │ 生成         │  │ 保存         │      │   │   │
│  │  │  │ 综合评分     │→│ 反馈报告     │→│ 单轮结果     │      │   │   │
│  │  │  │              │  │              │  │              │      │   │   │
│  │  │  │ score =      │  │ best_formulas│  │ round_i/     │      │   │   │
│  │  │  │ 0.5*|IR|     │  │ failed       │  │  alphagpt/   │      │   │   │
│  │  │  │ +0.2*decay   │  │ suggestions  │  │  mcts/       │      │   │   │
│  │  │  │ +0.2*divers  │  │ stats        │  │  dedup/      │      │   │   │
│  │  │  │ +0.1*turnover│  │              │  │  feedback    │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    ↓                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Stage 5: 终止条件检查                                       │   │   │
│  │  │                                                              │   │   │
│  │  │  ┌─────────────────────────────────────────────────────┐   │   │   │
│  │  │  │ 条件1: len(final_pool) >= target_factors?           │   │   │   │
│  │  │  │ 条件2: early_stopping.should_stop(best_ir)?         │   │   │   │
│  │  │  │ 条件3: elapsed > timeout_seconds?                   │   │   │   │
│  │  │  └─────────────────────────────────────────────────────┘   │   │   │
│  │  │                          ↓                                  │   │   │
│  │  │              ┌───────────────────────┐                     │   │   │
│  │  │              │  满足任一条件?         │                     │   │   │
│  │  │              └───────────────────────┘                     │   │   │
│  │  │                    ↓           ↓                           │   │   │
│  │  │                  Yes          No                           │   │   │
│  │  │                    ↓           ↓                           │   │   │
│  │  │              ┌─────────┐  ┌─────────┐                     │   │   │
│  │  │              │ 退出循环│  │继续下轮 │                     │   │   │
│  │  │              └─────────┘  └─────────┘                     │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        最终处理阶段                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ 选择         │→│ Wiki         │→│ 生成         │              │   │
│  │  │ 最终因子池   │  │ 持久化       │  │ 报告         │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        输出                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Wiki 页面    │  │ JSON 结果    │  │ 报告文档     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

| 阶段 | 输入 | 输出 | 数量 |
|------|------|------|------|
| Alpha-GPT | 目标 + 数据 + 反馈 | 种子公式 | 4-10 个 |
| MCTS | 种子公式 + 数据 | 优化公式 | 10-20 个 |
| 合并去重 | Alpha-GPT + MCTS | 最终因子池 | Top-K 个 |
| Wiki | 最终因子池 | Wiki 页面 | Top-K 个 |

---

## 3. 评估标准

### 3.1 核心指标

| 指标 | 定义 | 阈值 | 权重 | 说明 |
|------|------|------|------|------|
| **IR** | IC_mean / IC_std | ≥ 0.5 | 50% | 信息比率，综合衡量预测能力和稳定性 |
| **IC Decay** | 5d IC / 1d IC | ≥ 0.3 | 20% | IC衰减率，衡量因子信号的持续性 |
| **Mutual IC** | 与已有因子的相关性 | ≤ 0.7 | 20% | 互相关系数，用于去冗余 |
| **Turnover** | 换手率 | ≤ 200% | 10% | 换手率，衡量交易成本 |

### 3.2 综合评分公式

```python
score = (
    0.50 * normalized(|IR|) +
    0.20 * normalized(ic_decay_ratio) +
    0.20 * normalized(1 - avg_mutual_ic) +
    0.10 * normalized(1 - turnover_ratio)
)
```

### 3.3 硬性过滤门（Hard Gates）

```python
def is_valid_factor(factor: FactorMetrics) -> bool:
    """判断因子是否有效"""
    return (
        factor.status == "success" and           # 执行成功
        abs(factor.ir) >= 0.5 and               # IR >= 0.5
        factor.ic_decay_ratio >= 0.3 and        # IC 衰减率 >= 30%
        factor.max_mutual_ic <= 0.7 and         # 最大互相关 <= 0.7
        factor.turnover <= 2.0                  # 换手率 <= 200%
    )
```

### 3.4 评估指标详细说明

#### 3.4.1 IR（Information Ratio）

- **定义**：IC_mean / IC_std
- **范围**：通常在 -1.0 到 1.0 之间
- **阈值**：|IR| >= 0.5
- **说明**：IR 综合衡量因子的预测能力和稳定性。IR 越高，因子越有效。

#### 3.4.2 IC Decay（IC衰减率）

- **定义**：5d IC / 1d IC
- **范围**：0.0 到 1.0
- **阈值**：>= 0.3
- **说明**：衡量因子信号的持续性。衰减率越高，因子信号越持久。

#### 3.4.3 Mutual IC（互相关系数）

- **定义**：与已有因子的最大互相关系数
- **范围**：0.0 到 1.0
- **阈值**：<= 0.7
- **说明**：用于去冗余。互相关越高，因子越冗余。

#### 3.4.4 Turnover（换手率）

- **定义**：因子值变化导致的持仓变化率
- **范围**：0.0 到无穷大
- **阈值**：<= 200%
- **说明**：衡量交易成本。换手率越高，交易成本越高。

---

## 4. 终止条件

### 4.1 主要终止条件

满足任一条件即停止：

| 条件 | 说明 | 默认值 | 可配置 |
|------|------|--------|--------|
| **max_rounds** | 最大轮次 | 5 | ✅ |
| **target_factors** | 目标因子数量 | 10 | ✅ |
| **min_improvement** | 最小 IR 提升 | 0.01 | ✅ |

### 4.2 早停机制（Early Stopping）

```
┌─────────────────────────────────────────────────────────────────┐
│                        早停机制 (Early Stopping)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  初始化                                                   │   │
│  │  best_ir = 0.0                                          │   │
│  │  counter = 0                                            │   │
│  │  patience = 3                                           │   │
│  │  min_improvement = 0.01                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  每轮结束后                                               │   │
│  │  current_ir = round_result.best_ir                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  判断: current_ir > best_ir + min_improvement?          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                     ↓           ↓                               │
│                   Yes          No                               │
│                     ↓           ↓                               │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ best_ir = current_ir │  │ counter += 1         │            │
│  │ counter = 0          │  │                      │            │
│  │ 继续下一轮           │  │                      │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                         ↓                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  判断: counter >= patience?                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                     ↓           ↓                               │
│                   Yes          No                               │
│                     ↓           ↓                               │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ 触发早停             │  │ 继续下一轮           │            │
│  │ 退出主循环           │  │                      │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 超时配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| **timeout_seconds** | 总超时时间 | 3600 秒（1 小时） |
| **round_timeout_seconds** | 单轮超时时间 | 600 秒（10 分钟） |

### 4.4 终止条件代码实现

```python
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_ir = 0.0
        self.counter = 0
    
    def should_stop(self, current_ir: float) -> bool:
        """判断是否应该停止"""
        if current_ir > self.best_ir + self.min_improvement:
            self.best_ir = current_ir
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

---

## 5. 反馈机制

### 5.1 反馈内容

```python
@dataclass
class RoundFeedback:
    """单轮反馈"""
    
    round_num: int
    best_ir: float
    avg_ir: float
    valid_count: int
    
    # 最佳因子（用于下一轮种子）
    best_formulas: List[str]
    
    # 失败模式（避免重复）
    failed_patterns: List[Dict[str, str]]
    
    # 改进建议
    suggestions: List[str]
    
    # 统计信息
    stats: Dict[str, Any]
```

### 5.2 反馈生成流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        反馈生成流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输入: round_result, history                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 1: 提取最佳因子                                    │   │
│  │  - 筛选 IR >= 0.5 的因子                                │   │
│  │  - 取 Top-5 作为种子                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 2: 提取失败模式                                    │   │
│  │  - 收集被拒绝的公式                                      │   │
│  │  - 记录拒绝原因                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 3: 生成改进建议                                    │   │
│  │  - 分析成功因子的算子组合                                │   │
│  │  - 分析失败模式的共同特征                                │   │
│  │  - 生成多样性建议                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 4: 计算统计信息                                    │   │
│  │  - best_ir, avg_ir, valid_count                         │   │
│  │  - improvement_vs_prev (与上轮对比)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输出: RoundFeedback                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 反馈注入 Alpha-GPT

```python
def inject_feedback_to_alphagpt(
    feedback: RoundFeedback,
    alphagpt_config: AlphaGptConfig,
) -> AlphaGptConfig:
    """将反馈注入 Alpha-GPT 配置"""
    
    # 构建反馈文本
    feedback_text = f"""
## Round {feedback.round_num} 反馈

### 最佳因子（IR >= 0.5）
{format_best_formulas(feedback.best_formulas)}

### 失败模式（避免重复）
{format_failed_patterns(feedback.failed_patterns)}

### 改进建议
{format_suggestions(feedback.suggestions)}

### 统计信息
- 最佳 IR: {feedback.best_ir:.4f}
- 平均 IR: {feedback.avg_ir:.4f}
- 有效因子: {feedback.valid_count} 个
"""
    
    # 注入到 Alpha-GPT 配置
    alphagpt_config.custom_feedback = feedback_text
    
    return alphagpt_config
```

### 5.4 反馈示例

```markdown
## Round 2 反馈

### 最佳因子（IR >= 0.5）
1. `rank(-ts_mean(returns, 20))` (IR=0.82)
2. `rank(ts_mean(close, 10) / ts_mean(close, 20) - 1)` (IR=0.65)
3. `rank(ts_std(returns, 20))` (IR=-0.58)

### 失败模式（避免重复）
1. `rank(vol / ts_mean(vol, 5))` - 失败原因: 换手率过高 (150%)
2. `rank(close / ts_mean(close, 5))` - 失败原因: IC衰减过快 (0.15)

### 改进建议
- 尝试使用 `ts_mean`, `ts_std`, `rank` 的组合
- 避免使用 `vol / ts_mean(vol, 5)` 的模式（换手率过高）
- 增加因子多样性，避免重复

### 统计信息
- 最佳 IR: 0.82
- 平均 IR: 0.45
- 有效因子: 8 个
- 与上轮对比: IR 提升 0.15
```

---

## 6. 结果保存

### 6.1 保存结构

```
pipeline_output/
├── round_1/
│   ├── alphagpt/
│   │   ├── ideas.json
│   │   ├── formulas.json
│   │   ├── evaluations.json
│   │   └── final_pool.json
│   ├── mcts/
│   │   ├── tree.json          # 所有节点
│   │   ├── feedback.json      # 详细评估
│   │   └── correlation.json   # 相关性矩阵
│   ├── dedup/
│   │   ├── mutual_ic.json
│   │   └── final_pool.json
│   ├── feedback.json          # 反馈信息
│   └── summary.json
├── round_2/
│   └── ...
├── final/
│   ├── factors.json
│   ├── correlation_matrix.csv
│   └── report.md
└── wiki/
    └── Factor/
        ├── FORMULA-1.md
        └── ...
```

### 6.2 保存内容

| 文件 | 内容 | 格式 |
|------|------|------|
| `alphagpt/ideas.json` | Alpha-GPT 生成的想法 | JSON |
| `alphagpt/formulas.json` | Alpha-GPT 生成的公式 | JSON |
| `alphagpt/evaluations.json` | 评估结果 | JSON |
| `alphagpt/final_pool.json` | Alpha-GPT 最终因子池 | JSON |
| `mcts/tree.json` | MCTS 所有节点 | JSON |
| `mcts/feedback.json` | MCTS 详细评估 | JSON |
| `mcts/correlation.json` | 相关性矩阵 | JSON |
| `dedup/mutual_ic.json` | Mutual IC 矩阵 | JSON |
| `dedup/final_pool.json` | 去重后因子池 | JSON |
| `feedback.json` | 单轮反馈 | JSON |
| `summary.json` | 单轮摘要 | JSON |
| `final/factors.json` | 最终因子列表 | JSON |
| `final/correlation_matrix.csv` | 相关性矩阵 | CSV |
| `final/report.md` | 最终报告 | Markdown |

### 6.3 保存函数

```python
def save_pipeline_result(result: PipelineResult, output_dir: Path):
    """保存流水线结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存每轮结果
    for round_result in result.rounds:
        save_round_result(round_result, output_dir)
    
    # 2. 保存最终结果
    save_final_result(result, output_dir)
    
    # 3. 保存报告
    save_report(result, output_dir)

def save_round_result(round_result: RoundResult, output_dir: Path):
    """保存单轮结果"""
    round_dir = output_dir / f"round_{round_result.round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 Alpha-GPT 结果
    save_alphagpt_result(round_result.alphagpt_result, round_dir / "alphagpt")
    
    # 保存 MCTS 结果（所有节点）
    save_mcts_result(round_result.mcts_result, round_dir / "mcts")
    
    # 保存详细评估
    save_feedback_details(round_result.feedback_details, round_dir / "mcts")
    
    # 保存相关性矩阵
    save_correlation_matrix(round_result.mutual_ic_matrix, round_dir / "mcts")
    
    # 保存去重结果
    save_dedup_result(round_result.dedup_result, round_dir / "dedup")
    
    # 保存反馈信息
    save_feedback(round_result.feedback, round_dir)
    
    # 保存摘要
    save_summary(round_result, round_dir)
```

---

## 7. 配置参数

### 7.1 PipelineConfig

```python
@dataclass
class PipelineConfig:
    """流水线配置"""
    
    # 研究目标
    objective: str
    
    # 终止条件配置
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    
    # Alpha-GPT 配置
    alphagpt_iterations: int = 3
    alphagpt_pool_size: int = 10
    alphagpt_top_k: int = 10
    
    # MCTS 配置
    mcts_iterations: int = 50
    mcts_max_depth: int = 5
    mcts_dedup_threshold: float = 0.7
    
    # 去重配置
    max_mutual_ic: float = 0.7
    
    # 评估配置
    min_ir_threshold: float = 0.5
    min_ic_decay_ratio: float = 0.3
    max_turnover: float = 2.0
    
    # 通用配置
    top_k: int = 10
    date_column: str = "date"
    code_column: str = "code"
    forward_returns: Tuple[int, ...] = (1, 5, 20)
    
    # LLM 配置
    llm_provider: str = "minimax"
    llm_model: Optional[str] = None
    temperature: float = 0.7
    
    # 各阶段温度参数
    temperature_idea_gen: float = 0.8
    temperature_formula: float = 0.4
    temperature_reflector: float = 0.6
    temperature_critic: float = 0.3
    
    # 输出配置
    output_dir: str = "pipeline_output"
```

### 7.2 TerminationConfig

```python
@dataclass
class TerminationConfig:
    """终止条件配置"""
    
    # 主要终止条件
    max_rounds: int = 5                     # 最大轮次
    target_factors: int = 10                # 目标因子数量
    min_improvement: float = 0.01           # 最小 IR 提升
    
    # 早停配置
    early_stopping: bool = True             # 是否启用早停
    patience: int = 3                       # 连续 N 轮无改善则停止
    
    # 超时配置
    timeout_seconds: int = 3600             # 总超时时间（秒）
    round_timeout_seconds: int = 600        # 单轮超时时间（秒）
```

### 7.3 CLI 参数

```bash
quantnodes alpha-pipeline \
    --objective "捕捉 A 股反转效应" \
    --max-rounds 5 \
    --target-factors 10 \
    --min-improvement 0.01 \
    --patience 3 \
    --no-early-stopping \
    --timeout 3600 \
    --round-timeout 600 \
    --alphagpt-iterations 3 \
    --alphagpt-pool-size 10 \
    --mcts-iterations 50 \
    --mcts-max-depth 5 \
    --min-ir-threshold 0.5 \
    --min-ic-decay-ratio 0.3 \
    --max-turnover 2.0 \
    --llm minimax \
    --temperature-idea-gen 0.8 \
    --temperature-formula 0.4 \
    --temperature-reflector 0.6 \
    --temperature-critic 0.3 \
    --output-dir pipeline_output/
```

---

## 8. 实现计划

### 8.1 实现步骤

| 步骤 | 任务 | 文件 | 状态 |
|------|------|------|------|
| 1 | 添加 `TerminationConfig` | `pipeline.py` | 待实现 |
| 2 | 添加 `RoundFeedback` | `pipeline.py` | 待实现 |
| 3 | 实现 `generate_feedback()` | `pipeline.py` | 待实现 |
| 4 | 实现 `EarlyStopping` | `pipeline.py` | 待实现 |
| 5 | 修改 `run()` 支持多轮迭代 | `pipeline.py` | 待实现 |
| 6 | 修改 `_run_alphagpt()` 支持反馈注入 | `pipeline.py` | 待实现 |
| 7 | 实现 `save_pipeline_result()` | `pipeline.py` | 待实现 |
| 8 | 修改 CLI 添加新参数 | `cli/commands/alpha.py` | 待实现 |
| 9 | 测试 | `tests/quant_alpha/test_pipeline.py` | 待实现 |

### 8.2 预期效果

| 指标 | 当前 | 优化后 |
|------|------|--------|
| 迭代轮数 | 1 轮 | 3-5 轮 |
| 因子质量 | IR 0.08-0.20 | IR 0.15-0.30 |
| 因子数量 | 10 个 | 10-20 个 |
| 可追溯性 | 低 | 高（完整审计轨迹） |
| 自动化程度 | 低 | 高（自动终止） |

### 8.3 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM API 限流 | 迭代变慢 | 指数退避重试 |
| 公式评估失败 | 因子数量减少 | 容错处理，跳过失败公式 |
| 内存占用过高 | 系统崩溃 | 分批保存，定期清理缓存 |
| 相关性计算耗时 | 迭代变慢 | 使用缓存，增量计算 |

---

## 附录

### A. 参考框架

1. **Alpha-GPT**: arXiv:2308.00016
2. **Qlib**: https://qlib.readthedocs.io/
3. **WorldQuant Alpha101**: 内部文档
4. **EvolutionLoop**: QuantNodes 内置模块

### B. 相关文件

| 文件 | 说明 |
|------|------|
| `QuantNodes/research/quant_alpha/pipeline.py` | 流水线主模块 |
| `QuantNodes/research/quant_alpha/workflow/alpha_gpt.py` | Alpha-GPT 工作流 |
| `QuantNodes/research/quant_alpha/mcts/search.py` | MCTS 搜索 |
| `QuantNodes/cli/commands/alpha.py` | CLI 命令 |
| `tests/quant_alpha/test_pipeline.py` | 测试文件 |

### C. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-27 | 初始设计文档 |
