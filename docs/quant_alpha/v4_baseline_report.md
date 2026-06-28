# V4 Baseline 报告

**版本**: V4
**日期**: 2026-06-28
**作者**: LLM Pipeline
**目标**: 思维链改造前的 4 逻辑 E2E baseline

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
- **MCTS iterations**: 50
- **每个逻辑 pool_size**: 3
- **top_k**: 3

## 配置

```python
logics = {
    "volatility": WikiLogicStructured(
        predicates=[LogicCondition(variable='close', op='ts_std', threshold=0, window=20)],
        behavior=LogicBehavior(target='forward_return_5', direction=-1, horizon=5),
        operator_whitelist=['rank', 'ts_std', 'ts_mean', 'div'],
        parameter_ranges={'ts_std': (5, 60), 'ts_mean': (5, 60)},
        sign_constraint=-1,
    ),
    "momentum": ...,
    "mean_reversion": ...,
    "price_volume_divergence": ...,
}
```

---

## 结果

（待 V4 跑完填入）

| 逻辑 | 有效因子数 | 最佳 IR | 平均 IR | 公式示例 |
|------|-----------|---------|---------|----------|
| volatility | TBD | TBD | TBD | TBD |
| momentum | TBD | TBD | TBD | TBD |
| mean_reversion | TBD | TBD | TBD | TBD |
| price_volume_divergence | TBD | TBD | TBD | TBD |
| **合计** | TBD | TBD | TBD | — |

## 文件位置

- 脚本: `tests/quant_alpha/run_4_logic_v4.py`
- 输出: `pipeline_output_v4/`
- Summary: `pipeline_output_v4/v4_summary.json`

## 后续

V4 baseline 用于 V5（`feature/thinking-chain` 完成版）对比，验证思维链利用是否真的能加快因子生产。
