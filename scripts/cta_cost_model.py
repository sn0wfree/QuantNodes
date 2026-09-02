# coding=utf-8
"""cta_cost_model.py — 按品种交易成本模型 (手续费 + 滑点).

手续费: 交易所公开费率近似 (典型价格换算bp), 待实盘账单替换
滑点:   交易所标准tick ÷ 典型价格 (市价单吃1档, 1 tick保守估计)

输出: reports/cta/cost_model.json {product: {commission_bps, slippage_bps}}

⚠️ 两表均为估算值 (交易所公开标准+典型价格), 上实盘前用真实账单校准.
   不依赖DB的tick/价格 (contract_specs有坏数据: LC tick错误/I价格异常).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORT_DIR = Path("reports/cta")

# 手续费估算表 (单边, bp of notional)
# 来源: 各交易所公开费率 ÷ 典型价格×乘数; 分级近似, 待实盘账单校准
COMMISSION_BPS = {
    # CFFEX 金融
    "IF": 0.023, "IH": 0.023, "IC": 0.023, "IM": 0.023,
    "T": 0.003, "TF": 0.003, "TS": 0.003, "TL": 0.003,
    # 贵金属 (SHFE)
    "AU": 0.022, "AG": 0.05,
    # 基本金属 (SHFE/GFEX)
    "CU": 0.05, "AL": 0.03, "ZN": 0.05, "PB": 0.05, "NI": 0.05, "SN": 0.05,
    "SI": 0.10, "LC": 0.06, "AO": 0.10,
    # 黑色 (SHFE/DCE/CZCE)
    "RB": 0.10, "HC": 0.10, "I": 0.10, "J": 0.12, "JM": 0.12,
    "SF": 0.06, "SM": 0.06, "SS": 0.10,
    # 能源 (INE)
    "SC": 0.40, "LU": 0.15, "NR": 0.10, "EC": 0.02,
    # 化工 (DCE/CZCE/INE)
    "L": 0.10, "V": 0.10, "PP": 0.10, "EG": 0.10, "EB": 0.075,
    "PG": 0.10, "BU": 0.09, "RU": 0.02, "BR": 0.10,
    "TA": 0.12, "MA": 0.08, "FG": 0.10, "SA": 0.15, "UR": 0.14,
    "PF": 0.10, "PX": 0.10, "SH": 0.10, "SP": 0.05,
    # 农产品 (DCE/CZCE)
    "M": 0.05, "Y": 0.05, "A": 0.05, "B": 0.04, "C": 0.04, "CS": 0.05,
    "P": 0.06, "RM": 0.12, "OI": 0.07, "SR": 0.05, "CF": 0.06,
    "JD": 0.15, "LH": 0.20, "AP": 0.06, "CJ": 0.05, "PK": 0.09,
}

# 滑点 = 1 tick ÷ 典型价格 (bp); tick为交易所标准, 价格为典型值
# 来源: 交易所合约标准 (长期稳定), 价格2025-26典型值
SLIPPAGE_BPS = {
    # 贵金属
    "AU": 0.25, "AG": 1.25,
    # 基本金属
    "CU": 1.33, "AL": 2.44, "ZN": 2.17, "PB": 2.94, "NI": 0.80,
    "SN": 0.40, "SI": 4.55, "LC": 6.25, "AO": 2.86,
    # 黑色
    "RB": 3.03, "HC": 2.94, "I": 6.25, "J": 2.78, "JM": 3.85,
    "SF": 5.88, "SM": 7.41, "SS": 3.70,
    # 能源/航运
    "SC": 1.82, "LU": 2.38, "NR": 4.00, "EC": 0.83,
    # 化工
    "L": 1.28, "V": 1.67, "PP": 1.33, "EG": 2.17, "EB": 1.22,
    "PG": 2.13, "BU": 5.26, "RU": 3.45, "BR": 4.35,
    "TA": 4.00, "MA": 4.00, "FG": 8.33, "SA": 6.67, "UR": 5.71,
    "PF": 2.86, "PX": 2.90, "SH": 3.70, "SP": 3.70,
    # 农产品
    "M": 3.45, "Y": 2.53, "A": 2.50, "B": 2.86, "C": 4.35, "CS": 3.57,
    "P": 2.41, "RM": 4.00, "OI": 1.09, "SR": 1.69, "CF": 3.70,
    "JD": 3.03, "LH": 3.57, "AP": 1.32, "CJ": 4.17, "PK": 2.35,
    # 金融
    "IF": 0.51, "IH": 0.74, "IC": 0.36, "IM": 0.32,
    "T": 0.46, "TF": 0.48, "TS": 0.80, "TL": 1.69,
}


def main():
    products = sorted(set(COMMISSION_BPS) & set(SLIPPAGE_BPS))
    rows = []
    for p in products:
        comm = COMMISSION_BPS[p]
        slip = SLIPPAGE_BPS[p]
        rows.append({
            "product": p,
            "commission_bps": comm,
            "slippage_bps": slip,
            "total_bps": round(comm + slip, 2),
        })
    df = pd.DataFrame(rows).set_index("product").sort_values("total_bps")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = {
        p: {"commission_bps": float(r.commission_bps),
            "slippage_bps": float(r.slippage_bps)}
        for p, r in df.iterrows()
    }
    (REPORT_DIR / "cost_model.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False)
    )
    df.to_csv(REPORT_DIR / "cost_model.csv")

    print(f"品种数: {len(df)}")
    print("\n成本分布 (total_bps = 手续费+滑点, 单边):")
    print(df["total_bps"].describe().round(2).to_string())
    print("\n最贵10品种:")
    print(df.tail(10).to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n最便宜10品种:")
    print(df.head(10).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\n已保存: {REPORT_DIR}/cost_model.{{json,csv}}")
    print("⚠️ 估算值 (交易所公开标准+典型价格), 上实盘前用真实账单校准")


if __name__ == "__main__":
    main()
