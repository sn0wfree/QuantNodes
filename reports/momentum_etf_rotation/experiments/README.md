# Experiments (失败实验归档)

> 用途: 归档所有**失败的实验**, 防止未来重复调研
> 维护: 每次 NO-GO 决策时追加

---

## 归档规则

### 何时归档

- ❌ NO-GO 决策的实验 (未通过 Stage 5 Decision)
- ⚠️ 部分失败的实验 (有可复用部分)
- 🔬 探索性实验 (尝试后发现方向错误)

### 不归档

- ✅ 成功的 Stage (写入 `stage<N>_<feature>_report.md`)
- Bug 修复 (在 `DEV_WORKFLOW.md` 教训章节)
- 临时实验 (Code Review 等)

### 命名规范

```
experiments/
├── README.md                       # 本文件
├── stage_9d_hmm_failed.md         # 失败案例 1
├── stage_10_caps_failed.md        # 失败案例 2
└── <new_failure>.md
```

---

## 当前归档

| 实验 | 结论 | 文档 |
|------|------|------|
| Stage 9-D HMM Regime 检测 | Calmar 退化 33% | [stage_9d_hmm_failed.md](./stage_9d_hmm_failed.md) |
| Stage 10 集中度约束 | Calmar 退化 22% | [stage_10_caps_failed.md](./stage_10_caps_failed.md) |

---

## 失败模式分类 (供未来参考)

### 1. 过度优化型失败

**特征**: 算法理论上完美, 但实际表现远差于预期

**例子**: 
- Stage 9-D HMM (高维 + 小样本 = 过拟合)
- 任何带 ML 的复杂算法

**预防**:
- Stage 1 中严格评估样本量 / 参数数
- Stage 2 沙盒用合成数据验证
- Stage 4 必有 OOS 测试

### 2. 风控叠加型失败

**特征**: 单一风险控制措施看起来合理, 但与现有架构叠加后过度

**例子**:
- Stage 10 集中度约束 (逆波动 + caps + 趋势 + 波动率 = 过度)

**预防**:
- Stage 5 必须测**与其他现有功能的组合**
- 不仅看单一功能, 要看叠加效果

### 3. 集成顺序型失败

**特征**: 单元测试通过, 集成后失败

**例子**:
- Stage 9-C vol_targeting 被归一化
- Stage 10 caps 被 apply_stops 覆盖

**预防**:
- Code Review Checklist B.1-B.4 必查
- 集成测试覆盖集成点
- 画数据流图

---

## 实验归档模板

```markdown
# 实验失败: <标题>

> 阶段: Stage <N> (<日期>)
> 结论: ❌ NO-GO

## 假设
- H1: ...

## 实现
- 改动文件: ...
- 代码量: ~XXX 行

## 失败原因
- <技术原因>
- <数据原因>
- <集成原因>

## 证据
| 指标 | 实验版本 | baseline | 退化 |
|------|---------|---------|------|
| Calmar | X | X | X% |
| DD | X | X | X% |
| OOS | X | X | X% |

## 教训
- <可复用经验>

## 可能的复苏方向
- 如果 <条件 1> 和 <条件 2>, 可重新尝试
- 修改方案: ...
```

---

**目录结束**

未来失败实验请按模板添加。
