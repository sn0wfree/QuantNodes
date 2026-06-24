---
name: factor-research
description: 单因子研究工作流 — 因子生成、有效性测试、相关性分析、Wiki 沉淀。
---

# Factor Research

驱动 QuantNodes 单因子研究全流程：候选因子生成 → IC 测试 → 分组回测 → 相关性分析 → 写入 Wiki。

## 工作流

1. **生成因子候选** — 调用 `strategy_generate` 让 LLM 基于研究主题（动量/反转/质量/价值/波动率/...）生成 3-5 个因子公式
2. **代码验证** — 调用 `pipeline_validate` 验证因子代码语法与数据依赖
3. **单因子测试** — 调用 `factor_test` 跑 IC/ICIR/分组回测，得到 IC 序列、ICIR、分组收益表
4. **相关性分析** — 调用 `factor_test` 配 `mode="correlation"` 检查新因子与已有因子库的相关系数（|r|>0.7 视为冗余）
5. **沉淀 Wiki** — 调用 `wiki_write` 写入 Wiki 页面，附 IC 摘要 + 分组收益图 + 相关性矩阵
6. **Dream 触发** — 若因子表现异常好（ICIR>3 或年化>30%），调用 `quant_dream` 工具追加洞察

## 工具集

| 工具 | 用途 |
|------|------|
| `strategy_generate` | NL → 因子公式代码 |
| `pipeline_validate` | 语法/依赖验证 |
| `factor_test` | IC/ICIR/分组回测/相关性 |
| `wiki_write` | 写入 Wiki 页面 |
| `quant_dream` | 追加 quant dream 洞察 |

## 验收标准

- IC 绝对值均值 > 0.03
- ICIR > 0.5
- 分组收益单调（top-bottom > 5%/年）
- 与现有因子库 |r| < 0.7

## 反模式

- 不要跳过代码验证直接回测（浪费时间）
- 不要在同一日期截面用未来数据（未来函数）
- 不要忽略相关性分析（冗余因子）
