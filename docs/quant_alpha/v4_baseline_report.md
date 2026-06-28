# V4 Baseline 报告

**版本**: V4
**日期**: 2026-06-28
**作者**: LLM Pipeline
**目标**: 思维链改造前的 4 逻辑 E2E baseline
**状态**: ✅ 完成

---

## 概述

V4 是在 `feature/thinking-chain` 分支上的 baseline 实验，使用**当前的 prompt（已内联 JSON schema）+ `_complete_direct` 直接 OpenAI API 路径**，跑 4 个逻辑（volatility / momentum / mean_reversion / price_volume_divergence）各 3 个 idea × 1 round。

V4 不含任何思维链利用，仅验证：
1. 4 逻辑 pipeline 全部跑通
2. 当前 prompt + 直接 API 路径稳定
3. 拿到 V5 对比用的基准数据

---

## 实施环境

- **数据**: `data/cache/full_a_2019_2024.parquet`（1,258,502 行 × 8 列）
- **LLM**: MiniMax M3 via direct OpenAI API（绕过 nanobot agent 路径）
- **max_tokens**: 16384
- **timeout**: 300s
- **MCTS iterations**: 20
- **每个逻辑 pool_size**: 3
- **top_k**: 3
- **total_elapsed**: 522.2s

## 配置

```python
logics = {
    "volatility": WikiLogicStructured(...),
    "momentum": WikiLogicStructured(...),
    "mean_reversion": WikiLogicStructured(...),
    "price_volume_divergence": WikiLogicStructured(...),
}
```

---

## 结果

| 逻辑 | 有效因子数 | 最佳 IR | 最佳 \|IR\| | 平均 IR | 耗时 |
|------|-----------|---------|---------|---------|------|
| volatility | 3 | -0.0502 | **0.1208** | -0.0506 | 135.9s |
| mean_reversion | 3 | 0.1133 | **0.1133** | 0.0273 | 82.6s |
| momentum | 3 | 0.0193 | **0.0387** | -0.0002 | 113.7s |
| price_volume_divergence | 0 | — | 0.0000 | — | 190.0s |
| **合计** | **9** | — | — | — | **522.2s** |

### 详情

**volatility** (3 因子):
- FORMULA-1-1: IR=-0.0502, IC=-0.0053
- FORMULA-1-2: IR=0.0193, IC=0.0014
- FORMULA-1-3: IR=-0.1208, IC=-0.0144 ← 最佳

**mean_reversion** (3 因子):
- IR=0.1133 (最佳)
- IR=0.05~-0.05

**momentum** (3 因子):
- IR=0.0193~0.0387

**price_volume_divergence** (0 因子):
- LLM 未生成有效公式

## 关键观察

1. **3/4 逻辑产出因子**，仅 price_volume_divergence 失败
2. **最佳 |IR|=0.1208** (volatility FORMULA-1-3)
3. **总计 9 因子** in 522.2s
4. **LLM 思维链** 100% 产生但完全被丢弃（gateway 清理 <think> 标签）

## 后续

V4 baseline 用于 V5（`feature/thinking-chain` 完成版）对比，验证思维链利用是否真的能加快因子生产。
