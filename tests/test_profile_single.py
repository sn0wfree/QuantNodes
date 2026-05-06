#!/usr/bin/env python3
# coding=utf-8
"""
单策略全市场性能剖析

只跑一个策略，分阶段计时，找到瓶颈。
"""

import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import configparser
from pathlib import Path
from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
from QuantNodes.agent.config.executor import ConfigExecutor
from QuantNodes.database_node import ClickHouseNode


def main():
    print("=" * 60)
    print("全市场性能剖析 - dual_ma 策略")
    print("=" * 60)

    # 1. ClickHouse 查询
    print("\n[1] ClickHouse 查询...")
    ini_path = Path("conn.ini")
    cp = configparser.ConfigParser()
    cp.read(str(ini_path))
    section = "ClickHouse"
    host = cp.get(section, "host", fallback="localhost")
    port = int(cp.get(section, "port", fallback="8123"))
    user = cp.get(section, "user", fallback="default")
    passwd = cp.get(section, "passwd", fallback="")
    database = cp.get(section, "db", fallback="default")

    node = ClickHouseNode(host=host, port=port, user=user, passwd=passwd, database=database)
    node.connect()

    sql = "SELECT ts_code, trade_date, open, high, low, close, vol FROM quote.cn_stock WHERE trade_date >= toDateTime('2023-07-01') ORDER BY ts_code, trade_date"

    t0 = time.time()
    df = node.query(sql)
    t1 = time.time()
    print(f"    查询耗时: {t1-t0:.2f}s")
    print(f"    行数: {len(df):,}")
    print(f"    列: {list(df.columns)}")
    print(f"    股票数: {df['ts_code'].nunique():,}")
    print(f"    日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    node.disconnect()

    # 2. 列名映射
    print("\n[2] 列名映射 + 类型转换...")
    t0 = time.time()
    df = df.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"})
    t1 = time.time()
    print(f"    rename 耗时: {t1-t0:.4f}s")

    t0 = time.time()
    pdf = pl.from_pandas(df)
    if pdf.schema.get("date") == pl.Datetime:
        pdf = pdf.with_columns(pl.col("date").cast(pl.Date))
    lf = pdf.lazy()
    t1 = time.time()
    print(f"    to_polars 耗时: {t1-t0:.4f}s")
    print(f"    Polars schema: {pdf.schema}")

    # 3. 因子计算
    print("\n[3] 因子计算...")
    import yaml
    with open("QuantNodes/agent/config/templates/dual_ma.yaml", "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config_yaml = yaml.dump(config_dict, allow_unicode=True, default_flow_style=False)

    tool = ConfigBacktestTool()
    config = tool._load_config(config_yaml=config_yaml)

    executor = ConfigExecutor()
    gc.collect()
    t0 = time.time()
    result = executor.run(config, lf)
    t1 = time.time()
    print(f"    因子计算耗时: {t1-t0:.4f}s")

    if result.data is not None:
        schema = result.data.collect_schema()
        print(f"    输出列: {list(schema.keys())}")

    # 4. Collect to pandas (这一步可能很慢)
    print("\n[4] Polars collect → Pandas...")
    t0 = time.time()
    collected = result.data.collect()
    t1 = time.time()
    print(f"    collect 耗时: {t1-t0:.4f}s")
    print(f"    Shape: {collected.shape}")

    t0 = time.time()
    pandas_df = collected.to_pandas()
    t1 = time.time()
    print(f"    to_pandas 耗时: {t1-t0:.4f}s")

    # 5. 回测
    print("\n[5] 回测...")
    from QuantNodes.backtest.config_runner import ConfigBacktestRunner
    runner = ConfigBacktestRunner()
    gc.collect()
    t0 = time.time()
    bt_result = runner.run(config, lf)
    t1 = time.time()
    print(f"    回测耗时: {t1-t0:.4f}s")
    print(f"    交易数: {bt_result.statistics.get('total_trades', 0)}")
    print(f"    总收益: {bt_result.total_return:.4f}")
    print(f"    夏普: {bt_result.sharpe_ratio:.4f}")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)


if __name__ == "__main__":
    main()
