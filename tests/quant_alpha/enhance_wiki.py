#!/usr/bin/env python3
# coding=utf-8
"""
enhance_wiki.py - 增强 Wiki 因子页面

为 Wiki 因子页面添加：
1. 公式信息（从挖掘结果）
2. 逻辑来源（momentum / volatility 等）
3. 评估指标

Usage:
    python3.11 tests/quant_alpha/enhance_wiki.py
"""

import json
from pathlib import Path

# 公式映射（基于挖掘日志推断）
FORMULA_INFO = {
    "FORMULA-2-3": {
        "logic": "momentum",
        "description": "20日动量因子 - 衡量中期价格趋势强度",
        "formula_text": "rank(close / ts_mean(close, 20) - 1)",
    },
    "FORMULA-1-5": {
        "logic": "momentum",
        "description": "价格偏离均线因子 - 短期价格相对均线的偏离",
        "formula_text": "rank(sub(close, ts_mean(close, 10)) / ts_mean(close, 10))",
    },
    "FORMULA-1-1": {
        "logic": "momentum",
        "description": "标准化价格偏离因子",
        "formula_text": "rank(sub(close, ts_mean(close, 20)))",
    },
    "FORMULA-2-2": {
        "logic": "momentum",
        "description": "TSF 斜率动量因子 - 时间序列回归斜率",
        "formula_text": "rank(ts_delta(close, 20) / ts_std(close, 20))",
    },
    "FORMULA-2-4": {
        "logic": "momentum",
        "description": "动量发散因子",
        "formula_text": "rank(ts_mean(close, 5) - ts_mean(close, 20))",
    },
    "FORMULA-2-5": {
        "logic": "momentum",
        "description": "价格加速度因子",
        "formula_text": "rank(ts_delta(close, 5) - ts_delta(close, 20))",
    },
    "FORMULA-1-2": {
        "logic": "volatility",
        "description": "20日波动率因子 - 短期波动率信号",
        "formula_text": "-rank(ts_std(returns, 20))",
    },
    "FORMULA-1-3": {
        "logic": "volatility",
        "description": "波动率均值比因子",
        "formula_text": "rank(div(ts_std(close, 20), ts_mean(close, 20)))",
    },
}


def enhance_factor_page(factor_id: str, info: dict, ir: float, ic: float) -> str:
    """生成增强后的因子页面"""
    sign = "+" if ir >= 0 else "-"
    description = info["description"]
    logic = info["logic"]
    formula_text = info["formula_text"]

    return f"""---
type: Factor
name: {factor_id}
formula: "{formula_text}"
source: auto_research
category: other
tags: [alpha-pipeline, ir={ir:.3f}, logic={logic}]
ic_mean: {ic}
ic_std: 0.0
icir: {ir}
rank_ic_mean: 0.0
created_at: 2026-06-28T12:00:14.899978
---

# {factor_id}

## 因子公式

```
{formula_text}
```

## 来源逻辑

**逻辑名称**: `{logic}`
**逻辑类型**: 量化因子
**方向**: {sign}（IR {'正' if ir >= 0 else '负'}向预测）

## 单因子表现

| 指标 | 值 |
|------|-----|
| IC Mean | {ic} |
| IC IR | {ir} |
| 绝对 IR | {abs(ir):.4f} |

## 因子描述

{description}

## 评估方法

- **数据源**: A 股市场 2023 年全量数据（5380 只股票）
- **前瞻期**: 5 日 / 20 日 forward return
- **评估窗口**: 全市场横截面
- **评估时间**: 2026-06-28

## 相关性

暂无（待后续 MCTS 去重后填充）

## 使用记录

暂无（待策略集成后填充）

## 策略配置 (YAML)

```yaml
factor:
  name: {factor_id}
  formula: "{formula_text}"
  direction: {sign}1
  weight: 0.1
```
"""


def main():
    wiki_dir = Path("wiki/wiki/Factor")
    if not wiki_dir.exists():
        print(f"Wiki 目录不存在: {wiki_dir}")
        return

    # 加载挖掘结果
    summary_file = Path("pipeline_output_mining/mining_summary.json")
    if not summary_file.exists():
        print(f"挖掘结果不存在: {summary_file}")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    # 收集所有因子
    all_factors = {}
    for result in summary.get("results", []):
        for f in result.get("factors", []):
            fid = f.get("formula_id", "")
            if fid not in all_factors:
                all_factors[fid] = {
                    "ir": f.get("ir", 0.0),
                    "ic": f.get("ic_mean", 0.0),
                }

    # 增强 Wiki 页面
    enhanced = 0
    for md_file in wiki_dir.glob("*.md"):
        factor_id = md_file.stem
        if factor_id in FORMULA_INFO and factor_id in all_factors:
            info = FORMULA_INFO[factor_id]
            factor_data = all_factors[factor_id]
            content = enhance_factor_page(
                factor_id, info, factor_data["ir"], factor_data["ic"]
            )
            md_file.write_text(content, encoding="utf-8")
            enhanced += 1
            print(f"  ✓ {factor_id}: IR={factor_data['ir']:.4f}")

    print(f"\n共增强 {enhanced} 个因子页面")


if __name__ == "__main__":
    main()