# coding: utf-8
"""export_data_dictionary.py — 生成数据字典 (MD + duckdb内嵌元数据表).

对 export/futures_market_full.duckdb 与 export/fundamentals_full.duckdb:
  1. table_catalog:  表级目录 (类别/说明/行数/日期范围/品种数)
  2. data_dictionary: 列级字典 (类型/说明)
  3. DATA_DICTIONARY.md: 人读版
语义说明基于一手审计知识 (smoothed拼接口径/rank语义/vrp口径/PIT要点).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

EXPORT = Path("export")
MD_PATH = EXPORT / "DATA_DICTIONARY.md"

DBS = {
    "futures_market_full.duckdb": "期货行情数据库 (全量)",
    "fundamentals_full.duckdb": "基本面数据库 (全量)",
}

# ── 表级语义映射 (前缀/名称规则 → 类别 + 中文说明) ─────────────────────────

def classify(table: str) -> tuple[str, str]:
    if table.startswith("smoothed_"):
        p = table.replace("smoothed_", "").replace("_daily", "").upper()
        return ("行情-主力连续", f"{p} 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径)")
    if table.startswith("gradual_"):
        p = table.replace("gradual_", "").replace("_daily", "").upper()
        return ("行情-连续(中间)", f"{p} 渐进切换连续 (平滑的中间产物)")
    if table.startswith("continuous_main_"):
        p = table.replace("continuous_main_", "").replace("_daily", "").upper()
        return ("行情-连续(中间)", f"{p} 主力连续原始拼接 (未平滑)")
    if table.endswith("_futures_daily"):
        exch = table.split("_")[0].upper()
        prod = table.replace("_futures_daily", "").upper()
        return ("行情-原始合约", f"{exch} {prod} 全部合约日行情 (交易所原始)")
    if table.endswith("_options_daily"):
        exch = table.split("_")[0].upper()
        prod = table.replace("_options_daily", "").upper()
        return ("期权-原始合约", f"{exch} {prod} 全部期权合约日行情")
    m = {
        "term_structure_daily": ("衍生-期限结构",
            "88品种×多到期日合约结构. rank=1近月/2次近月...; days_to_expiry为年化carry分母; "
            "§25横截面carry与basis动量的数据源"),
        "vrp_daily": ("衍生-波动率溢价",
            "62品种 VRP = ATM隐含波动率 - 实现波动率(HV); 含 vrp/iv/hv 的滚动分位数与 regime 标签. "
            "PIT注意: IV为EOD口径"),
        "volatility_hv_daily": ("衍生-实现波动率", "88品种 HV(20/60/120日) 实现波动率"),
        "option_iv_daily": ("衍生-隐含波动率",
            "65品种 ATM IV + iv_percentile. 部分品种数据滞后 (RB/SC等2019-2022才上市期权)"),
        "options_pcr_daily": ("衍生-期权情绪", "65品种 Put/Call 比率 (成交量与持仓两个口径)"),
        "oi_crossover": ("衍生-移仓结构",
            "88品种 主力/次主力 OI与价格 — 移仓换月进程监测 (main_symbol/next_symbol)"),
        "spot_basis_daily": ("衍生-基差", "现货价-近月-主力 三口径基差 (symbol维度)"),
        "cross_product_spread": ("衍生-跨品种价差", "跨品种价差序列 (研究用)"),
        "inter_contract_spread": ("衍生-跨期价差", "同品种跨期价差序列 (研究用)"),
        "trading_calendar": ("参考-交易日历", "交易日历"),
        "main_contract_mapping": ("参考-主力映射", "日期×品种 → 主力合约映射"),
        "contract_specs": ("参考-合约要素", "合约乘数/保证金等要素表"),
        "futures_rule_snapshot": ("参考-规则快照", "交易所规则快照"),
        "v_all_futures": ("视图-全量期货", "全品种期货宽表视图"),
        "v_all_options": ("视图-全量期权", "全品种期权宽表视图"),
        "default_ip_futures_daily": ("行情-IP连续", "默认IP口径连续合约 (备用口径)"),
        # 基本面库
        "fundamentals_daily": ("基本面-宏观/全球",
            "长表: source(数据源)×trade_date×value_col(指标名)×value. 29个指标含 "
            "dxy(美元)/gvz/vix(恐慌)/tips与tips10y(实际利率)/us_10y/us_2y/us_10y_2y_spread/"
            "fed_rate/fed_rate_daily/fed_total_assets/m2_money_supply/us_cpi_yoy/us_ip/"
            "pmi_mfg(⚠️记账日在每月1号,语义未验证,审计§26建议+35自然日)/us_oil_rig(钻机数,"
            "⚠️周五13:00ET=北京周六发布)/epu(经济政策不确定性)/wti/brent/gold_spot/"
            "comex_inv/etf_tonnes(黄金ETF持仓)/cb_reserves(央行储备)/gprd/cftc_net/cftc_oi/"
            "hog_index/rir 等"),
        "inventory_daily": ("基本面-库存",
            "长表: source(如'LME铜库存')×trade_date×value_col(如'锌-库存')×value. "
            "LME/EIA等库存序列; ZN库存子策略信号源 (注意: 日期为数据方记账日)"),
        "tfapi_indicators": ("基本面-产业指标",
            "长表: index_name×trade_date×value. 含洋山铜溢价/铜粗炼费TC (CU溢价TC子策略"
            "信号源, 天风期货API, 2026-09刷新) 等产业高频指标"),
        "tfapi_cftc": ("基本面-CFTC", "CFTC COT 长短持仓 (⚠️§8已证伪为alpha源, 仅留档)"),
        "tfapi_holding": ("基本面-经纪持仓", "经纪公司持仓排名 (⚠️PIT后-0.26, §8已证伪, 仅留档)"),
        "eia_weekly_inventory": ("基本面-EIA库存", "美国EIA周度原油/成品油库存 (周频)"),
        "term_structure_cu": ("基本面-铜期限结构", "铜期限结构 (copper策略族数据源, 与行情库"
            "term_structure_daily中CU部分同源不同期)"),
        "futures_main_close_au": ("行情-黄金主力", "黄金主力合约日收盘 (周频策略用)"),
        "futures_main_close_au_adj": ("行情-黄金主力(换月调整)", "黄金主力换月比率调整收盘"),
        "futures_main_close_cu": ("行情-铜主力", "铜主力合约日收盘"),
        "futures_main_close_m": ("行情-豆粕主力", "豆粕主力合约日收盘"),
        "futures_main_close_sc": ("行情-原油主力", "原油主力合约日收盘"),
        "futures_all_contracts_au": ("行情-黄金全合约", "黄金全部合约日行情"),
        "gold_main_contracts": ("行情-黄金主力合约表", "黄金主力合约代码序列"),
        "gold_main_switches": ("行情-黄金换月事件", "黄金主力切换事件 (31次)"),
    }
    if table in m:
        return m[table]
    return ("其他", table)


# ── 列级标准说明 ────────────────────────────────────────────────────────────

COL_DESC = {
    "trade_date": "交易日", "date": "日期", "symbol": "合约代码", "product": "品种代码",
    "exchange": "交易所", "open": "开盘价", "high": "最高价", "low": "最低价",
    "close": "收盘价", "settle": "结算价", "volume": "成交量(手)", "open_interest": "持仓量(手)",
    "oi": "持仓量(手)", "amount": "成交额", "prev_settle": "昨结算价",
    "rank": "到期序 (1=近月, 2=次近月, ...)", "expiry_year": "到期年", "expiry_month": "到期月",
    "days_to_expiry": "距到期日天数",
    "atm_iv": "ATM隐含波动率(年化)",
    "hv_20": "20日实现波动率(年化)",
    "hv_60": "60日实现波动率(年化)", "hv_120": "120日实现波动率(年化)",
    "vrp": "波动率风险溢价 = atm_iv - hv_20", "vrp_percentile": "VRP滚动分位数",
    "iv_percentile": "IV滚动分位数", "regime": "波动率状态标签 (low/neutral/high)",
    "call_volume": "看涨期权成交量", "put_volume": "看跌期权成交量",
    "call_oi": "看涨持仓", "put_oi": "看跌持仓", "pcr_volume": "PCR(成交量)",
    "pcr_oi": "PCR(持仓)", "main_symbol": "主力合约", "main_oi": "主力持仓",
    "main_close": "主力收盘", "main_open": "主力开盘", "next_symbol": "次主力合约",
    "next_oi": "次主力持仓", "next_close": "次主力收盘",
    "spot_price": "现货价", "near_contract": "近月合约", "near_contract_price": "近月价格",
    "dominant_contract": "主力合约", "dominant_contract_price": "主力价格",
    "source": "数据源", "value": "数值", "value_col": "指标/字段名 (取值见MD附录)",
    "index_name": "指标名 (取值见MD附录)", "n_contracts": "合约数",
    "iv_percentile_252d": "IV 252日分位",
}


def build_catalog_and_dict(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(db_path), read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()
              if t[0] not in ("table_catalog", "data_dictionary")]
    cat_rows, dict_rows = [], []
    for t in sorted(tables):
        cat, desc = classify(t)
        cols = con.execute(f"PRAGMA table_info('{t}')").fetchall()
        colnames = [c[1] for c in cols]
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        dmin = dmax = None
        if "trade_date" in colnames:
            dmin, dmax = con.execute(
                f"SELECT MIN(TRY_CAST(trade_date AS VARCHAR)), "
                f"MAX(TRY_CAST(trade_date AS VARCHAR)) FROM {t}").fetchone()
        nprod = None
        if "product" in colnames:
            nprod = con.execute(f"SELECT COUNT(DISTINCT product) FROM {t}").fetchone()[0]
        cat_rows.append({"table_name": t, "category": cat, "rows": n,
                         "date_min": dmin, "date_max": dmax,
                         "n_products": nprod, "description": desc})
        for c in cols:
            dict_rows.append({"table_name": t, "column_name": c[1],
                              "data_type": c[2],
                              "description": COL_DESC.get(c[1], "")})
    con.close()
    return pd.DataFrame(cat_rows), pd.DataFrame(dict_rows)


def write_meta_tables(db_path: Path, cat: pd.DataFrame, dct: pd.DataFrame) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS table_catalog")
    con.execute("DROP TABLE IF EXISTS data_dictionary")
    con.register("cat_v", cat)
    con.register("dict_v", dct)
    con.execute("CREATE TABLE table_catalog AS SELECT * FROM cat_v")
    con.execute("CREATE TABLE data_dictionary AS SELECT * FROM dict_v")
    con.unregister("cat_v")
    con.unregister("dict_v")
    con.close()


def write_md(db: str, cat: pd.DataFrame, dct: pd.DataFrame, out: list[str]) -> None:
    out.append(f"\n\n# {DBS[db]} — `{db}`\n")
    out.append(f"共 {len(cat)} 张表 / {cat['rows'].sum():,} 行。\n")
    for category, grp in cat.groupby("category"):
        out.append(f"\n## {category} ({len(grp)}表)\n")
        out.append("| 表名 | 行数 | 日期范围 | 品种数 | 说明 |")
        out.append("|---|---:|---|---:|---|")
        for _, r in grp.iterrows():
            rng = f"{str(r['date_min'])[:10]}~{str(r['date_max'])[:10]}" if r["date_min"] else "—"
            out.append(f"| `{r['table_name']}` | {r['rows']:,} | {rng} | "
                       f"{r['n_products'] if pd.notna(r['n_products']) else '—'} | "
                       f"{r['description']} |")
    # 列级字典 (仅衍生/基本面/参考类详细表, 原始表列结构标准)
    special = cat[cat["category"].str.startswith(("衍生", "基本面", "参考", "期权-原始"))]
    out.append("\n## 关键表列级字典\n")
    for _, trow in special.iterrows():
        cols = dct[dct["table_name"] == trow["table_name"]]
        out.append(f"\n### `{trow['table_name']}`\n")
        out.append("| 列 | 类型 | 说明 |")
        out.append("|---|---|---|")
        for _, c in cols.iterrows():
            out.append(f"| `{c['column_name']}` | {c['data_type']} | {c['description']} |")


def main() -> None:
    md = ["# CTA 数据字典",
          "",
          "导出自本项目生产数据 (只读快照), 两库结构对账一致.",
          "",
          "## 使用注意",
          "",
          "1. **smoothed_*_daily 是本项目标准回测口径** (真主连平滑连续);",
          "   gradual_/continuous_ 为中间产物; 原始交易所表为上游原始数据.",
          "2. **2026 数据截止**: 行情库 2026-07-21, 基本面库部分指标 2026-09-04",
          "   (洋山铜溢价/TC). smoothed/gradual 仅到 2026-01 (管线滞后, 见 tuning_log).",
          "3. **2018 为 warmup 年**: 主CTA NAV 在 2018 全年零收益 (§27),",
          "   真实业绩窗口 2019-2025.",
          "4. **PIT 要点**: pmi_mfg 记账日在每月1号 (建议使用时+35自然日);",
          "   us_oil_rig 周五发布 (北京周六凌晨); 库存/溢价为数据方记账日口径.",
          "5. 每个库内嵌元数据表: `table_catalog` (表级) + `data_dictionary` (列级),",
          "   可直接 SQL 查询: `SELECT * FROM table_catalog WHERE category LIKE '衍生%'`.",
          "6. ⚠️ 标记的字段存在已审计的 PIT 风险 (tuning_log §26).",
          ]
    for db in DBS:
        p = EXPORT / db
        cat, dct = build_catalog_and_dict(p)
        write_meta_tables(p, cat, dct)
        write_md(db, cat, dct, md)
        print(f"{db}: {len(cat)}表, 字典 {len(dct)}列条目 — 内嵌表已写入")
    MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"MD: {MD_PATH} ({MD_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
