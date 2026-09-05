# coding: utf-8
"""build_2026_market_p3.py — 临时库 → export/futures_market_2026.duckdb 切片导出.

全部含 trade_date 的表: WHERE trade_date >= '2026-01-01'
无日期表 (contract_specs/futures_rule_snapshot): 全拷
重建 v_all_* 视图 + 内嵌 table_catalog/data_dictionary + DATA_DICTIONARY_2026.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_data_dictionary import build_catalog_and_dict, write_meta_tables  # noqa: E402

TMP_DB = PROJECT_ROOT / "export" / "_tmp_2026.duckdb"
OUT_DB = PROJECT_ROOT / "export" / "futures_market_2026.duckdb"
OUT_MD = PROJECT_ROOT / "export" / "DATA_DICTIONARY_2026.md"
CUTOFF = "2026-01-01"
META_TABLES = {"table_catalog", "data_dictionary"}


def main() -> int:
    src = duckdb.connect(str(TMP_DB), read_only=True)
    tables = [t[0] for t in src.execute("SHOW TABLES").fetchall()]
    tables = [t for t in tables if not t.startswith("v_all_")]
    nodate = []
    sliced = []
    for t in sorted(tables):
        if t in META_TABLES:
            continue
        cols = [c[1] for c in src.execute(f"PRAGMA table_info('{t}')").fetchall()]
        if "trade_date" in cols:
            sliced.append(t)
        else:
            nodate.append(t)
    src.close()
    print(f"切片 {len(sliced)} 表 + 全拷 {len(nodate)} 表: {nodate}")

    if OUT_DB.exists():
        OUT_DB.unlink()
    dst = duckdb.connect(str(OUT_DB))
    dst.execute(f"ATTACH '{TMP_DB}' AS srcdb (READ_ONLY)")
    for t in sliced:
        dst.execute(
            f'CREATE TABLE "{t}" AS SELECT * FROM srcdb."{t}" '
            f"WHERE trade_date >= '{CUTOFF}'")
    for t in nodate:
        dst.execute(f'CREATE TABLE "{t}" AS SELECT * FROM srcdb."{t}"')

    # 视图 (切片后重建)
    fut = [t for t in sliced + nodate if t.endswith("_futures_daily")]
    opt = [t for t in sliced + nodate if t.endswith("_options_daily")]
    dst.execute("DROP VIEW IF EXISTS v_all_futures")
    dst.execute("CREATE VIEW v_all_futures AS "
                + " UNION ALL ".join(f'SELECT * FROM "{t}"' for t in sorted(fut)))
    dst.execute("DROP VIEW IF EXISTS v_all_options")
    dst.execute("CREATE VIEW v_all_options AS "
                + " UNION ALL ".join(f'SELECT * FROM "{t}"' for t in sorted(opt)))
    dst.execute("CHECKPOINT")
    dst.close()

    # 元数据表 + MD
    cat, dct = build_catalog_and_dict(OUT_DB)
    write_meta_tables(OUT_DB, cat, dct)
    write_md_2026(cat, dct)
    print(f"完成: {OUT_DB} ({OUT_DB.stat().st_size / 1e6:.0f}MB), "
          f"{len(cat) - 2}数据表, 字典 {len(dct)}条目")
    return 0


def write_md_2026(cat, dct) -> None:
    md = [
        "# 2026 年行情数据库数据字典 (futures_market_2026.duckdb)",
        "",
        f"切片口径: trade_date >= {CUTOFF} (行情至 2026-09-04, CH tick_quote 聚合).",
        f"共 {len(cat) - 2} 张数据表 + 2 张元数据表.",
        "",
        "## 与 full 库的差异",
        "",
        "1. **内容为 2026 年切片** (管线 warmup 自 2025-10 起, 拼接/平滑边界正确).",
        "2. **期权日线改由 CH tick_quote 聚合生成** (src_file='ch_tick_quote'),",
        "   非交易所爬虫源; 剔除了套利合约(SP a&b / MSC / MS-C) / EFP / TAS /",
        "   期转现(F) 等非标准合约.",
        "3. **spot_basis_daily 由 akshare 重拉** (2025-07-22 起, 54 品种).",
        "4. 滚动分位列 (vrp_percentile / iv_percentile / hv) 在 2026 年前段",
        "   因窗口不足存在 NaN (切片库固有局限, warmup 自 2025-10).",
        "5. 表级/列级说明同 DATA_DICTIONARY.md (full 库); 期权表结构:",
        "   exchange/product/symbol/underlying/option_type/strike_price/",
        "   trade_date/OHLCV/pre_*/src_file.",
        "",
        "## 内嵌元数据",
        "",
        "```sql",
        "SELECT * FROM table_catalog;   -- 表级目录",
        "SELECT * FROM data_dictionary; -- 列级字典",
        "```",
        "",
        "## 表级目录",
        "",
        "| 表名 | 行数 | 日期范围 | 说明 |",
        "|---|---:|---|---|",
    ]
    for _, r in cat.sort_values(["category", "table_name"]).iterrows():
        rng = f"{str(r['date_min'])[:10]}~{str(r['date_max'])[:10]}" if r["date_min"] else "—"
        md.append(f"| `{r['table_name']}` | {r['rows']:,} | {rng} | {r['description']} |")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"MD: {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
