#!/usr/bin/env python3
# coding=utf-8
"""
示例策略测试脚本 (ClickHouse 真实数据)

从 ClickHouse 加载真实股票行情数据，运行所有示例策略。
用法: python3 tests/test_example_strategies.py

性能参考 (20只股票 x 1年):
  ClickHouse 查询: ~3s
  数据转换: ~0.3s
  因子计算+回测: <0.1s
"""

import asyncio
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool


STRATEGIES = [
    ("dual_ma_crossover", "QuantNodes/agent/config/templates/dual_ma.yaml"),
    ("rsi_reversal", "QuantNodes/agent/config/templates/rsi_strategy.yaml"),
    ("bollinger_breakout", "QuantNodes/agent/config/templates/bollinger_bands.yaml"),
    ("volume_price_divergence", "QuantNodes/agent/config/templates/volume_price.yaml"),
    ("momentum_breakout", "QuantNodes/agent/config/templates/momentum_breakout.yaml"),
    ("mean_reversion_zscore", "QuantNodes/agent/config/templates/mean_reversion_zscore.yaml"),
]


async def run_strategy(name: str, yaml_path: str):
    """运行单个策略"""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config_yaml = yaml.dump(config, allow_unicode=True, default_flow_style=False)

    tool = ConfigBacktestTool()
    result = await tool.execute(config_yaml=config_yaml)
    return result


async def main():
    """主函数"""
    print("=" * 60)
    print("QuantNodes 示例策略测试 (ClickHouse 真实数据)")
    print("=" * 60)

    # 验证 ClickHouse 连接
    print("\n[0] 验证 ClickHouse 连接...")
    try:
        from QuantNodes.database_node import ClickHouseNode
        node = ClickHouseNode(host='localhost', port=8123, user='data', passwd='123456', database='quote')
        node.connect()
        count = node.query("SELECT count(*) as cnt FROM quote.cn_stock")
        node.disconnect()
        print(f"    ClickHouse 连接成功，总数据量: {count.iloc[0]['cnt']:,} 行")
    except Exception as e:
        print(f"    ClickHouse 连接失败: {e}")
        print("    请确保 ClickHouse 服务已启动且 conn.ini 配置正确")
        return 1

    print(f"\n[1] 运行 {len(STRATEGIES)} 个示例策略:")
    print("-" * 60)

    results = {}
    total_start = time.time()

    for name, yaml_path in STRATEGIES:
        print(f"\n  策略: {name}")
        print(f"  配置: {yaml_path}")

        try:
            t0 = time.time()
            result = await run_strategy(name, yaml_path)
            t1 = time.time()
            results[name] = result

            if result["status"] == "success":
                summary = result["summary"]
                print(f"  状态: SUCCESS ({t1-t0:.2f}s)")
                print(f"  交易次数: {summary.get('total_trades', 0)}")
                print(f"  最终资金: {summary.get('final_cash', 0):,.2f}")
                print(f"  总收益率: {summary.get('total_return', 0):.4f}")
                print(f"  夏普比率: {summary.get('sharpe_ratio', 0):.4f}")
                print(f"  最大回撤: {summary.get('max_drawdown', 0):.4f}")
                print(f"  年化收益: {summary.get('annualized_return', 0):.4f}")
            else:
                print(f"  状态: ERROR ({t1-t0:.2f}s)")
                errors = result.get('errors', ['Unknown error'])
                for err in errors[:3]:
                    print(f"  错误: {err[:200]}")

        except Exception as e:
            print("  状态: EXCEPTION")
            print(f"  异常: {e}")
            results[name] = {"status": "error", "errors": [str(e)]}

    total_end = time.time()

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  总耗时: {total_end - total_start:.2f}s")
    print()

    for name, result in results.items():
        status = "PASS" if result.get("status") == "success" else "FAIL"
        print(f"  [{status}] {name}")

    print()
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
