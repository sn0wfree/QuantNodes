#!/usr/bin/env python3
# coding=utf-8
"""
全市场策略性能测试 (ClickHouse)

全市场 ~5535 只股票 x 1 年数据，测试各阶段耗时。
用法: python3 tests/test_full_market_performance.py
"""

import asyncio
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
from QuantNodes.agent.config.loader import ConfigLoader
from QuantNodes.agent.config.executor import ConfigExecutor
from QuantNodes.backtest.config_runner import ConfigBacktestRunner


STRATEGIES = [
    ("dual_ma_crossover", "QuantNodes/agent/config/templates/dual_ma.yaml"),
    ("rsi_reversal", "QuantNodes/agent/config/templates/rsi_strategy.yaml"),
    ("bollinger_breakout", "QuantNodes/agent/config/templates/bollinger_bands.yaml"),
    ("volume_price_divergence", "QuantNodes/agent/config/templates/volume_price.yaml"),
    ("momentum_breakout", "QuantNodes/agent/config/templates/momentum_breakout.yaml"),
    ("mean_reversion_zscore", "QuantNodes/agent/config/templates/mean_reversion_zscore.yaml"),
]


def profile_load_data(config):
    """分阶段 profile 数据加载"""
    timings = {}

    # 1. 构建 SQL
    tool = ConfigBacktestTool()
    t0 = time.time()
    sql = tool._build_query(config.data)
    timings["build_sql"] = time.time() - t0

    # 2. ClickHouse 查询
    from QuantNodes.database_node import ClickHouseNode
    import configparser
    from pathlib import Path

    ini_path = Path(config.data.conn_ini)
    cp = configparser.ConfigParser()
    cp.read(str(ini_path))
    section = config.data.conn_section
    host = cp.get(section, "host", fallback="localhost")
    port = int(cp.get(section, "port", fallback="8123"))
    user = cp.get(section, "user", fallback="default")
    passwd = cp.get(section, "passwd", fallback="")
    database = cp.get(section, "db", fallback="default")

    node = ClickHouseNode(host=host, port=port, user=user, passwd=passwd, database=database)
    node.connect()

    t0 = time.time()
    df = node.query(sql)
    timings["clickhouse_query"] = time.time() - t0
    timings["rows"] = len(df)
    timings["columns"] = list(df.columns)

    # 3. 列名映射
    t0 = time.time()
    if config.data.column_mapping:
        df = df.rename(columns=config.data.column_mapping)
    timings["rename_columns"] = time.time() - t0

    # 4. DateTime 转换 + to Polars LazyFrame
    t0 = time.time()
    date_col = config.data.date_column
    if date_col in df.columns:
        try:
            pdf = pl.from_pandas(df)
            if pdf.schema.get(date_col) == pl.Datetime:
                pdf = pdf.with_columns(pl.col(date_col).cast(pl.Date))
            lf = pdf.lazy()
        except Exception:
            lf = pl.from_pandas(df).lazy()
    else:
        lf = pl.from_pandas(df).lazy()
    timings["to_polars_lazy"] = time.time() - t0

    node.disconnect()
    return lf, timings


def profile_executor(config, lf):
    """Profile 因子计算"""
    timings = {}
    executor = ConfigExecutor()

    t0 = time.time()
    result = executor.run(config, lf)
    timings["executor_run"] = time.time() - t0

    if result.data is not None:
        t0 = time.time()
        schema = result.data.collect_schema()
        timings["schema_check"] = time.time() - t0

    return result, timings


def profile_backtest(config, lf):
    """Profile 完整回测"""
    timings = {}

    t0 = time.time()
    runner = ConfigBacktestRunner()
    bt_result = runner.run(config, lf)
    timings["total_backtest"] = time.time() - t0

    return bt_result, timings


async def run_strategy_profiled(name, yaml_path):
    """Profile 单个策略"""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config_yaml = yaml.dump(config_dict, allow_unicode=True, default_flow_style=False)

    tool = ConfigBacktestTool()
    config = tool._load_config(config_yaml=config_yaml)

    all_timings = {}
    total_start = time.time()

    # Stage 1: 数据加载
    gc.collect()
    lf, load_timings = profile_load_data(config)
    all_timings.update({f"load_{k}": v for k, v in load_timings.items()})

    # Stage 2: 因子计算
    gc.collect()
    result, exec_timings = profile_executor(config, lf)
    all_timings.update({f"factor_{k}": v for k, v in exec_timings.items()})

    # Stage 3: 回测
    gc.collect()
    bt_result, bt_timings = profile_backtest(config, lf)
    all_timings.update(bt_timings)

    all_timings["total"] = time.time() - total_start

    return {
        "name": name,
        "status": "success" if bt_result.statistics.get("total_trades", 0) > 0 else "no_trades",
        "timings": all_timings,
        "rows": load_timings.get("rows", 0),
        "summary": {
            "total_trades": bt_result.statistics.get("total_trades", 0),
            "total_return": bt_result.total_return,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown": bt_result.max_drawdown,
            "win_rate": bt_result.win_rate,
        }
    }


async def main():
    print("=" * 70)
    print("QuantNodes 全市场性能测试 (ClickHouse)")
    print("=" * 70)

    # 验证连接
    print("\n[0] 验证 ClickHouse 连接...")
    try:
        from QuantNodes.database_node import ClickHouseNode
        node = ClickHouseNode(host='localhost', port=8123, user='data', passwd='123456', database='quote')
        node.connect()
        count = node.query("SELECT count(*) as cnt FROM quote.cn_stock")
        stocks = node.query("SELECT count(DISTINCT ts_code) as cnt FROM quote.cn_stock")
        node.disconnect()
        print(f"    总数据量: {count.iloc[0]['cnt']:,} 行, {stocks.iloc[0]['cnt']:,} 只股票")
    except Exception as e:
        print(f"    ClickHouse 连接失败: {e}")
        return 1

    # 先用一个简单查询测试全市场数据量
    print("\n[1] 预估全市场数据量...")
    try:
        node = ClickHouseNode(host='localhost', port=8123, user='data', passwd='123456', database='quote')
        node.connect()
        q = "SELECT count(*) as cnt FROM quote.cn_stock WHERE trade_date >= toDateTime('2023-07-01')"
        result = node.query(q)
        node.disconnect()
        print(f"    2023-07-01 至今: {result.iloc[0]['cnt']:,} 行")
    except Exception as e:
        print(f"    查询失败: {e}")

    print(f"\n[2] 运行 {len(STRATEGIES)} 个策略 (全市场):")
    print("-" * 70)

    all_results = []
    total_start = time.time()

    for name, yaml_path in STRATEGIES:
        print(f"\n  策略: {name}")
        try:
            result = await run_strategy_profiled(name, yaml_path)
            all_results.append(result)

            print(f"    数据行数: {result['rows']:,}")
            print(f"    状态: {result['status']}")
            for k, v in result["timings"].items():
                print(f"    {k}: {v:.3f}s")
            s = result["summary"]
            print(f"    交易: {s['total_trades']}, 收益: {s['total_return']:.4f}, "
                  f"夏普: {s['sharpe_ratio']:.4f}, 回撤: {s['max_drawdown']:.4f}")

        except Exception as e:
            print(f"    EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": name, "status": "error", "error": str(e)})

    total_end = time.time()

    # 汇总
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    print(f"  总耗时: {total_end - total_start:.2f}s")
    print()

    for r in all_results:
        if r["status"] == "error":
            print(f"  [{r['name']}] ERROR: {r.get('error', '')[:80]}")
            continue
        timings = r["timings"]
        print(f"  [{r['name']}]")
        print(f"    CH查询: {timings.get('load_clickhouse_query', 0):.2f}s | "
              f"因子计算: {timings.get('factor_executor_run', 0):.2f}s | "
              f"回测: {timings.get('total_backtest', 0):.2f}s | "
              f"总计: {timings.get('total', 0):.2f}s")
        print(f"    数据行: {r['rows']:,}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
