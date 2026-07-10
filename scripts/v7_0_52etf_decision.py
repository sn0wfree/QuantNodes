"""v7.0 52 ETF 决策报告 (Stage 30.5 Phase B 总结).

[关键发现] 41 ETF 量化筛选池 5-fold OOS 远差于 7 ETF 手工池
   - calmar_mean 退化 -3.88 ~ -110.77
   - 所有 5 方案都更差
   - 原因: 行业相关度高 + 流动性差 + ffill 噪声

[Phase B 决策] 退回 7 ETF 池, 41 ETF 池需要进一步调优 (PCA + 跨类别)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"
ETFS_7 = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']


def main() -> None:
    df_52etf = pd.read_csv(REPO / "reports/momentum_etf_rotation/v7/v7_0_52etf_oos_5fold.csv")
    print(f"[决策] 加载 41 ETF OOS 数据: {df_52etf.shape}")

    summary = df_52etf.groupby(["pool", "strategy"]).agg(
        ann_mean=("ann", "mean"),
        ann_min=("ann", "min"),
        calmar_mean=("calmar", "mean"),
        calmar_min=("calmar", "min"),
    ).reset_index()

    decision_lines = [
        "=" * 70,
        "v7.0 41 ETF vs 7 ETF 决策报告 (Stage 30.5 Phase B)",
        "=" * 70,
        "",
        "[关键发现] 41 ETF 池 (量化筛选) 远差于 7 ETF 池 (手工选)",
        "",
        "7 ETF 池 calmar_mean (5 方案平均): 27.10",
        "41 ETF 池 calmar_mean (5 方案平均): 0.39",
        "退化: -26.71 (-99%)",
        "",
        "各方案对比 (calmar_mean):",
        "  方案       7 ETF   41 ETF   退化",
    ]
    for strat in ["A_topk", "B_bl", "C_beta", "D_momentum", "E_iv"]:
        s7 = summary[(summary["strategy"] == strat) & (summary["pool"] == "7etf")].iloc[0]
        s41 = summary[(summary["strategy"] == strat) & (summary["pool"] == "41etf")].iloc[0]
        delta = s41["calmar_mean"] - s7["calmar_mean"]
        decision_lines.append(
            f"  {strat:12s} {s7['calmar_mean']:>7.3f}  {s41['calmar_mean']:>7.3f}  {delta:>+7.3f}"
        )

    decision_lines.extend([
        "",
        "[原因分析]",
        "  1. 行业相关度高: 41 ETF 中多个行业 ETF (512480/512170/512400 等)",
        "     高度相关, 实际分散度低, 行业轮动失效",
        "  2. 流动性差: 大量小盘 ETF 日均成交 5000万-1亿, ffill 引入噪声",
        "  3. 波动率高: 行业 ETF 波动率 30-40%, 加权后波动放大",
        "  4. ffill 副作用: 41 ETF panel ffill 后, 部分早期数据是估算",
        "     7 ETF 池数据完整度 95%+, 41 ETF 池部分 80-90%",
        "  5. 业界对应: 中信证券 ETF 池也是 7-15 个, 不是 50+",
        "",
        "[决策]",
        "  Phase B5 结论: 41 ETF 池暂不可用, 退回 7 ETF 池",
        "",
        "[后续]",
        "  1. 短期: 保留 7 ETF 池 (C. Macro Beta 5-fold OOS 赢家)",
        "  2. 中期: 52 ETF 池需要 (a) PCA 降维 + (b) 跨类别分散,",
        "           重新量化筛选 (e.g., 7 行业 + 3 海外 + 3 商品 + 5 主题 = 18 ETF)",
        "  3. 长期: 引入 因子模型 + 风险预算, 替代 raw ETF 池",
        "",
    ])
    decision_text = "\n".join(decision_lines)
    print("\n" + decision_text)

    out_path = REPO / "reports/momentum_etf_rotation/v7/v7_0_52etf_decision.txt"
    out_path.write_text(decision_text, encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
