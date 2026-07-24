#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""init_contract_specs.py — 初始化 DuckDB contract_specs 表.

从硬编码常量生成 92 行合约参数 (91 期货产品 + 1 IP 跨品种组合).
SQL: CREATE TABLE IF NOT EXISTS contract_specs.

Schema:
    exchange            VARCHAR        # CFFEX / SHFE / DCE / CZCE / INE / GFEX / IPS
    product             VARCHAR        # IF / IC / AU / RB / IP / ...
    contract_multiplier DOUBLE         # 元/手
    price_unit          VARCHAR        # 元/吨 / 元/克 / 元/点 / ...
    tick_size           DOUBLE         # 最小变动价位
    has_futures         BOOLEAN        # HO/IO/MO = False (无期货合约)

Usage:
    python scripts/init_contract_specs.py              # 写入 DuckDB
    python scripts/init_contract_specs.py --dry-run    # 打印预览, 不写
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)

DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()


SPECS = [
    # ════════════════════════════════════════════════════════════════
    # CFFEX (中国金融期货交易所) — 11 products
    # ════════════════════════════════════════════════════════════════
    # 股指期货: 乘数 300/200/100, 价格 1 点 = 300/200/100 元
    ("CFFEX", "IF",  300.0,   0.2,  "元/点",  True),
    ("CFFEX", "IC",  200.0,   0.2,  "元/点",  True),
    ("CFFEX", "IH",  300.0,   0.2,  "元/点",  True),
    ("CFFEX", "IM",  200.0,   0.2,  "元/点",  True),
    # 国债期货: 面值 100 万元 (T/TF/TS) 或 100 万元 (TL), 价格 = 元 / 100 元面值
    ("CFFEX", "T",   10000.0, 0.005,"元/元",  True),  # 10年国债
    ("CFFEX", "TF",  10000.0, 0.005,"元/元",  True),  # 5年国债
    ("CFFEX", "TS",  10000.0, 0.005,"元/元",  True),  # 2年国债
    ("CFFEX", "TL",  10000.0, 0.01, "元/元",  True),  # 30年国债
    # 股指期权 (无期货合约, 仅有期权)
    ("CFFEX", "HO",  100.0,   0.2,  "元/点",  False),  # 沪深300股指期权
    ("CFFEX", "IO",  100.0,   0.2,  "元/点",  False),  # 中证1000股指期权
    ("CFFEX", "MO",  100.0,   0.2,  "元/点",  False),  # 上证50股指期权

    # ════════════════════════════════════════════════════════════════
    # CZCE (郑州商品交易所) — 27 products, 大宗 + 农产品
    # ════════════════════════════════════════════════════════════════
    # 农产品 (元/吨)
    ("CZCE", "AP",   10.0,    1.0,  "元/吨",  True),   # 苹果
    ("CZCE", "CF",   5.0,     5.0,  "元/吨",  True),   # 棉花
    ("CZCE", "CJ",   5.0,     2.0,  "元/吨",  True),   # 红枣
    ("CZCE", "CY",   5.0,     5.0,  "元/吨",  True),   # 棉纱
    ("CZCE", "FG",   20.0,    1.0,  "元/吨",  True),   # 玻璃
    ("CZCE", "JR",   20.0,    1.0,  "元/吨",  True),   # 粳稻
    ("CZCE", "LR",   20.0,    1.0,  "元/吨",  True),   # 晚籼稻
    ("CZCE", "MA",   10.0,    1.0,  "元/吨",  True),   # 甲醇
    ("CZCE", "OI",   10.0,    1.0,  "元/吨",  True),   # 菜油
    ("CZCE", "PF",   5.0,     2.0,  "元/吨",  True),   # 涤纶短纤
    ("CZCE", "PK",   5.0,     2.0,  "元/吨",  True),   # 花生
    ("CZCE", "PL",   5.0,     1.0,  "元/吨",  True),   # 涤纶长丝
    ("CZCE", "PM",   50.0,    1.0,  "元/吨",  True),   # 普麦
    ("CZCE", "PR",   10.0,    1.0,  "元/吨",  True),   # 瓶片
    ("CZCE", "PX",   5.0,     2.0,  "元/吨",  True),   # 对二甲苯
    ("CZCE", "RI",   20.0,    1.0,  "元/吨",  True),   # 早籼稻
    ("CZCE", "RM",   10.0,    1.0,  "元/吨",  True),   # 菜粕
    ("CZCE", "RS",   10.0,    1.0,  "元/吨",  True),   # 油菜籽
    ("CZCE", "SA",   20.0,    1.0,  "元/吨",  True),   # 纯碱
    ("CZCE", "SF",   5.0,     2.0,  "元/吨",  True),   # 硅铁
    ("CZCE", "SH",   5.0,     2.0,  "元/吨",  True),   # 烧碱
    ("CZCE", "SM",   5.0,     2.0,  "元/吨",  True),   # 锰硅
    ("CZCE", "SR",   10.0,    1.0,  "元/吨",  True),   # 白糖
    ("CZCE", "TA",   5.0,     2.0,  "元/吨",  True),   # PTA
    ("CZCE", "UR",   20.0,    1.0,  "元/吨",  True),   # 尿素
    ("CZCE", "WH",   20.0,    1.0,  "元/吨",  True),   # 强麦
    ("CZCE", "ZC",   100.0,   0.2,  "元/吨",  True),   # 动力煤 (改名 BM?)

    # ════════════════════════════════════════════════════════════════
    # DCE (大连商品交易所) — 23 products
    # ════════════════════════════════════════════════════════════════
    # 农产品 (元/吨)
    ("DCE", "A",    10.0,    1.0,  "元/吨",  True),    # 豆一
    ("DCE", "B",    10.0,    1.0,  "元/吨",  True),    # 豆二
    ("DCE", "BB",   10.0,    1.0,  "元/吨",  True),    # 胶合板
    ("DCE", "C",    10.0,    1.0,  "元/吨",  True),    # 玉米
    ("DCE", "CS",   10.0,    1.0,  "元/吨",  True),    # 玉米淀粉
    ("DCE", "JD",   10.0,    1.0,  "元/500千克", True),  # 鸡蛋
    ("DCE", "FB",   500.0,   0.05, "元/立方米",  True),  # 纤维板
    ("DCE", "L",    5.0,     5.0,  "元/吨",  True),    # 聚乙烯
    ("DCE", "LG",   20.0,    1.0,  "元/吨",  True),    # 焦煤
    ("DCE", "LH",   16.0,    5.0,  "元/吨",  True),    # 生猪
    ("DCE", "M",    10.0,    1.0,  "元/吨",  True),    # 豆粕
    ("DCE", "P",    10.0,    2.0,  "元/吨",  True),    # 棕榈油
    ("DCE", "Y",    10.0,    2.0,  "元/吨",  True),    # 豆油
    ("DCE", "BZ",   10.0,    1.0,  "元/吨",  True),    # 苯乙烯
    # 工业品 (元/吨)
    ("DCE", "EB",   5.0,     1.0,  "元/吨",  True),    # 乙二醇
    ("DCE", "EG",   10.0,    2.0,  "元/吨",  True),    # 乙二醇 (旧)
    ("DCE", "I",    100.0,   0.5,  "元/吨",  True),    # 铁矿石
    ("DCE", "J",    100.0,   0.5,  "元/吨",  True),    # 焦炭
    ("DCE", "JM",   60.0,    0.5,  "元/吨",  True),    # 焦煤 (旧, 现 LG)
    ("DCE", "PG",   20.0,    1.0,  "元/吨",  True),    # 液化石油气
    ("DCE", "PP",   5.0,     1.0,  "元/吨",  True),    # 聚丙烯
    ("DCE", "RR",   10.0,    1.0,  "元/吨",  True),    # 粳米
    ("DCE", "V",    5.0,     5.0,  "元/吨",  True),    # PVC

    # ════════════════════════════════════════════════════════════════
    # GFEX (广州期货交易所) — 5 products, 贵金属 + 工业硅
    # ════════════════════════════════════════════════════════════════
    ("GFEX", "LC",  1.0,   0.01, "元/克",  True),       # 碳酸锂
    ("GFEX", "PD",  1000.0, 0.02,"元/克",  True),       # 钯
    ("GFEX", "PS",  1000.0, 0.02,"元/克",  True),       # 铂
    ("GFEX", "PT",  1000.0, 0.02,"元/克",  True),       # 铂 (新)
    ("GFEX", "SI",  5.0,    5.0, "元/吨",  True),       # 工业硅

    # ════════════════════════════════════════════════════════════════
    # INE (上海国际能源交易中心) — 5 products, 国际化品种
    # ════════════════════════════════════════════════════════════════
    ("INE", "BC",  5.0,    10.0,  "元/吨",  True),       # 国际铜
    ("INE", "EC",  50.0,   0.05,  "元/吨",  True),       # 集运指数 (欧线)
    ("INE", "LU",  10.0,   1.0,   "元/吨",  True),       # 低硫燃料油
    ("INE", "NR",  10.0,   5.0,   "元/吨",  True),       # 20号胶
    ("INE", "SC",  1000.0, 0.1,   "元/桶",  True),       # 原油

    # ════════════════════════════════════════════════════════════════
    # SHFE (上海期货交易所) — 20 products
    # ════════════════════════════════════════════════════════════════
    # 贵金属 (元/克)
    ("SHFE", "AG",  15.0,    1.0,    "元/克",  True),     # 白银
    ("SHFE", "AU",  1000.0,  0.02,   "元/克",  True),     # 黄金
    # 有色金属 (元/吨)
    ("SHFE", "AL",  5.0,     5.0,    "元/吨",  True),     # 铝
    ("SHFE", "CU",  5.0,     10.0,   "元/吨",  True),     # 铜
    ("SHFE", "NI",  1.0,     10.0,   "元/吨",  True),     # 镍
    ("SHFE", "PB",  5.0,     5.0,    "元/吨",  True),     # 铅
    ("SHFE", "SN",  1.0,     10.0,   "元/吨",  True),     # 锡
    ("SHFE", "ZN",  5.0,     5.0,    "元/吨",  True),     # 锌
    ("SHFE", "AO",  20.0,    1.0,    "元/吨",  True),     # 氧化铝
    ("SHFE", "BR",  10.0,    1.0,    "元/吨",  True),     # 丁二烯橡胶
    # 黑色 (元/吨)
    ("SHFE", "RB",  10.0,    1.0,    "元/吨",  True),     # 螺纹钢
    ("SHFE", "HC",  10.0,    1.0,    "元/吨",  True),     # 热卷
    ("SHFE", "SS",  5.0,     5.0,    "元/吨",  True),     # 不锈钢
    ("SHFE", "WR",  10.0,    1.0,    "元/吨",  True),     # 线材
    # 能化 (元/吨)
    ("SHFE", "BU",  10.0,    1.0,    "元/吨",  True),     # 沥青
    ("SHFE", "FU",  10.0,    1.0,    "元/吨",  True),     # 燃料油
    ("SHFE", "RU",  10.0,    5.0,    "元/吨",  True),     # 天然橡胶
    ("SHFE", "SP",  10.0,    2.0,    "元/吨",  True),     # 纸浆
    ("SHFE", "AD",  10.0,    1.0,    "元/吨",  True),     # 铝合金
    ("SHFE", "OP",  20.0,    1.0,    "元/吨",  True),     # 胶版印刷纸

    # ════════════════════════════════════════════════════════════════
    # IPS (跨品种组合) — 1 product
    # ════════════════════════════════════════════════════════════════
    ("IPS",  "IP",  None,    None,   "元/组合", True),     # CZCE 跨品种套利
]


def create_table_if_not_exists(con) -> None:
    """创建 contract_specs 表 (IF NOT EXISTS)."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS contract_specs (
            exchange            VARCHAR NOT NULL,
            product             VARCHAR NOT NULL,
            contract_multiplier DOUBLE,
            price_unit          VARCHAR,
            tick_size           DOUBLE,
            has_futures         BOOLEAN DEFAULT TRUE,
            PRIMARY KEY (exchange, product)
        )
    """)


def insert_specs(con) -> int:
    """清空表并插入全部 SPECS. 返回写入行数."""
    con.execute("DELETE FROM contract_specs")
    con.executemany(
        """
        INSERT INTO contract_specs
            (exchange, product, contract_multiplier, tick_size, price_unit, has_futures)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(ex, prod, mult, tick, unit, has_f) for ex, prod, mult, tick, unit, has_f in SPECS],
    )
    return len(SPECS)


def verify(con) -> None:
    """验证写入结果."""
    rows = con.execute("""
        SELECT
            exchange,
            COUNT(*) AS n,
            SUM(CASE WHEN has_futures THEN 1 ELSE 0 END) AS n_futures,
            SUM(CASE WHEN NOT has_futures THEN 1 ELSE 0 END) AS n_options_only
        FROM contract_specs
        GROUP BY exchange
        ORDER BY exchange
    """).fetchall()
    print("\n=== contract_specs 内容 ===")
    print(f"{'Exchange':<8} {'Total':>6} {'Futures':>8} {'OptOnly':>8}")
    print("-" * 32)
    total = 0
    for ex, n, n_f, n_o in rows:
        print(f"{ex:<8} {n:>6} {n_f:>8} {n_o:>8}")
        total += n
    print(f"{'TOTAL':<8} {total:>6}")


def create_v_all_views(con) -> tuple[int, int]:
    """创建 v_all_futures / v_all_options 视图.

    跨所有 *_futures_daily / *_options_daily 单产品表 UNION ALL.
    跳过 default_ip_futures_daily (exchange='IPS', 不在 KNOWN_EXCHANGES).
    """
    tables = con.execute("SHOW TABLES").fetchall()
    futures_tables = sorted(
        t[0] for t in tables
        if "futures" in t[0]
        and "continuous" not in t[0]
        and "v_all" not in t[0]
        and not t[0].startswith("default_")
    )
    options_tables = sorted(
        t[0] for t in tables
        if "options" in t[0]
        and "continuous" not in t[0]
        and "v_all" not in t[0]
        and not t[0].startswith("default_")
    )

    fut_sql = " UNION ALL ".join(f'SELECT * FROM "{t}"' for t in futures_tables)
    con.execute(f"CREATE OR REPLACE VIEW v_all_futures AS {fut_sql}")

    opt_sql = " UNION ALL ".join(f'SELECT * FROM "{t}"' for t in options_tables)
    con.execute(f"CREATE OR REPLACE VIEW v_all_options AS {opt_sql}")

    fut_rows = con.execute("SELECT COUNT(*) FROM v_all_futures").fetchone()[0]
    opt_rows = con.execute("SELECT COUNT(*) FROM v_all_options").fetchone()[0]
    return fut_rows, opt_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 DuckDB contract_specs + v_all 视图")
    parser.add_argument("--dry-run", action="store_true", help="仅打印预览, 不写入")
    parser.add_argument("--duckdb", type=str, default=None, help="DuckDB 路径")
    parser.add_argument(
        "--skip-views", action="store_true",
        help="跳过创建 v_all_futures / v_all_options 视图",
    )
    args = parser.parse_args()

    duckdb_path = Path(args.duckdb).expanduser() if args.duckdb else DUCKDB_PATH
    if not duckdb_path.exists():
        LOG.error(f"DuckDB not found: {duckdb_path}")
        return 1

    LOG.info(f"Total specs to load: {len(SPECS)}")
    LOG.info(f"  - 91 期货产品 (CFFEX=11, CZCE=27, DCE=23, GFEX=5, INE=5, SHFE=20)")
    LOG.info(f"  - 3 仅有期权 (HO, IO, MO)")
    LOG.info(f"  - 1 跨品种组合 (IPS.IP)")
    LOG.info(f"Target DuckDB: {duckdb_path}")

    if args.dry_run:
        LOG.info("[DRY-RUN] 打印全部 SPECS (前 10 行):")
        for ex, prod, mult, tick, unit, has_f in SPECS[:10]:
            print(f"  {ex:<6} {prod:<4} mult={mult!s:<6} tick={tick!s:<6} unit={unit:<10} has_fut={has_f}")
        print(f"  ... +{len(SPECS)-10} more")
        return 0

    con = duckdb.connect(str(duckdb_path))
    try:
        create_table_if_not_exists(con)
        LOG.info("CREATE TABLE IF NOT EXISTS contract_specs ... OK")

        n = insert_specs(con)
        LOG.info(f"INSERTED {n} rows")

        verify(con)

        if not args.skip_views:
            fut_rows, opt_rows = create_v_all_views(con)
            LOG.info(f"v_all_futures: {fut_rows:,} 行")
            LOG.info(f"v_all_options: {opt_rows:,} 行")
    finally:
        con.close()

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())