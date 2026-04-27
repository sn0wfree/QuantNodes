#!/usr/bin/env python3
# coding=utf-8
"""
QuantNodes 快速开始示例

演示如何使用：
1. BaseNode 自定义节点
2. Pipeline 线性管道
3. Parallel 并行分叉
4. Join 聚合组合
5. IfNode 条件分支
6. MapNode 分组映射
7. WhileNode 条件循环
"""

from QuantNodes.core import (
    BaseNode,
    Pipeline,
    Parallel,
    Join,
    IfNode,
    MapNode,
    WhileNode,
)
import pandas as pd
import numpy as np


# ============================================================================
# 1. 自定义节点示例
# ============================================================================

class LogReturnNode(BaseNode):
    """计算对数收益率"""
    def _execute(self, price_df, **kwargs):
        return np.log(price_df / price_df.shift(1)).dropna()


class VolatilityNode(BaseNode):
    """计算波动率（滚动标准差）"""

    def __init__(self, window: int = 20, name=None):
        super().__init__(name=name or "Volatility")
        self.window = window

    def _execute(self, return_df, **kwargs):
        return return_df.rolling(self.window).std()


class SharpeRatioNode(BaseNode):
    """计算夏普比率"""
    def _execute(self, return_df, **kwargs):
        return np.sqrt(252) * return_df.mean() / return_df.std()


# ============================================================================
# 2. 基本 Pipeline 示例
# ============================================================================

def example_simple_pipeline():
    """简单的线性管道"""
    print("=" * 60)
    print("示例 1: 简单 Pipeline 线性管道")
    print("=" * 60)

    # 创建模拟价格数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'AAPL': 100 + np.cumsum(np.random.randn(100)),
        'GOOG': 150 + np.cumsum(np.random.randn(100)),
    }, index=dates)

    # 构建管道
    pipeline = (
        LogReturnNode()
        >> VolatilityNode(window=20)
        >> SharpeRatioNode()
    )

    # 执行
    sharpe_ratios = pipeline.execute(prices)

    print(f"夏普比率计算结果:")
    print(sharpe_ratios)
    print()


# ============================================================================
# 3. Parallel + Join 组合示例
# ============================================================================

def example_parallel_join():
    """并行计算多个因子然后聚合"""
    print("=" * 60)
    print("示例 2: Parallel + Join 因子计算")
    print("=" * 60)

    # 创建模拟数据
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50)),
        'volume': np.random.randint(1000, 5000, 50),
    })

    # 并行计算多个技术指标
    factors = Parallel({
        'ret_1d': LogReturnNode(),
        'ret_5d': LogReturnNode() >> Pipeline([lambda x: x.rolling(5).mean()]),
        'volatility': VolatilityNode(window=10),
    }, parallel=False)  # 串行方便调试

    # 组合成一个因子
    combine = Join(lambda ret_1d, ret_5d, volatility: (ret_1d + ret_5d) / (volatility + 1e-6))

    # 完整管道
    pipeline = factors >> combine

    result = pipeline.execute(data['close'])
    print(f"组合因子形状: {result.shape}")
    print(f"前 5 个值: {result.head()}")
    print()


# ============================================================================
# 4. IfNode 条件分支示例
# ============================================================================

def example_if_node():
    """条件分支，波动率高低时使用不同策略"""
    print("=" * 60)
    print("示例 3: IfNode 条件分支策略")
    print("=" * 60)

    # 高波动率使用 10 天窗口
    high_vol_strategy = VolatilityNode(window=10)
    # 低波动率使用 30 天窗口
    low_vol_strategy = VolatilityNode(window=30)

    # 条件分支节点
    strategy_selector = IfNode(
        condition=lambda df: df.std().mean() > 0.02,  # 波动率 > 2% 算高波动
        true_branch=high_vol_strategy,
        false_branch=low_vol_strategy,
    )

    # 测试两组数据
    for name, volatility_level in [('高波动', 0.05), ('低波动', 0.01)]:
        returns = pd.DataFrame({
            'asset': np.random.randn(100) * volatility_level,
        })
        result = strategy_selector.execute(returns)
        print(f"{name} - 使用窗口大小: {result.columns[0] if isinstance(result.columns[0], int) else 'unknown'}")
        print(f"  输出形状: {result.shape}")
    print()


# ============================================================================
# 5. MapNode 分组回表示例
# ============================================================================

def example_map_node():
    """按年份分组计算"""
    print("=" * 60)
    print("示例 4: MapNode 分组处理")
    print("=" * 60)

    # 创建多年数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'price': 100 + np.cumsum(np.random.randn(len(dates))),
    })

    class YearlyReturnNode(BaseNode):
        def _execute(self, group_df, **kwargs):
            start_price = group_df['price'].iloc[0]
            end_price = group_df['price'].iloc[-1]
            year = group_df['year'].iloc[0]
            return (year, (end_price / start_price - 1) * 100)

    # 按年份分组计算
    mapper = MapNode(
        node=YearlyReturnNode(),
        group_by='year',
        parallel=False,
    )

    results = mapper.execute(data)
    print("各年度收益率:")
    for year, ret in sorted(results):
        print(f"  {year}: {ret:.1f}%")
    print()


# ============================================================================
# 6. WhileNode 循环优化示例
# ============================================================================

def example_while_node():
    """循环迭代直到夏普比率达标"""
    print("=" * 60)
    print("示例 5: WhileNode 参数优化")
    print("=" * 60)

    class OptimizerState:
        """优化状态"""
        def __init__(self, sharpe, iteration):
            self.metrics = type('Metrics', (), {'sharpe': sharpe})()
            self.iteration = iteration

    class SharpeImprover(BaseNode):
        """每次执行改进一点夏普比率"""
        def _execute(self, state, **kwargs):
            new_sharpe = state.metrics.sharpe + 0.15
            print(f"  迭代 {state.iteration + 1}: 夏普 = {new_sharpe:.2f}")
            return OptimizerState(new_sharpe, state.iteration + 1)

    # 目标：夏普 >= 1.5，最多迭代 10 次
    optimizer_loop = WhileNode(
        condition=lambda state: state.metrics.sharpe < 1.5,
        body=SharpeImprover(),
        max_iterations=10,
    )

    initial_state = OptimizerState(0.8, 0)
    final_state = optimizer_loop.execute(initial_state)

    print(f"优化完成!")
    print(f"最终夏普比率: {final_state.metrics.sharpe:.2f}")
    print(f"迭代次数: {optimizer_loop.iteration_count}")
    print()


# ============================================================================
# 7. 完整组合示例
# ============================================================================

def example_complete_pipeline():
    """完整的量化策略 Pipeline"""
    print("=" * 60)
    print("示例 6: 完整量化策略 Pipeline")
    print("=" * 60)

    # 模拟价格数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates))),
    }, index=dates)

    # 完整 Pipeline
    strategy = (
        LogReturnNode()
        >> Parallel({
            'short_vol': VolatilityNode(window=10),
            'long_vol': VolatilityNode(window=60),
            'raw_returns': lambda x, ctx: x,  # 直接透传
        })
        >> Join(lambda short_vol, long_vol, raw_returns: {
            'vol_ratio': short_vol / (long_vol + 1e-6),
            'returns': raw_returns,
        })
    )

    result = strategy.execute(prices)
    print(f"波动率比率形状: {result['vol_ratio'].shape}")
    print(f"波动率比率均值: {result['vol_ratio'].mean():.3f}")
    print(f"收益率均值: {result['returns'].mean() * 100:.4f}%")
    print()


if __name__ == '__main__':
    example_simple_pipeline()
    example_parallel_join()
    example_if_node()
    example_map_node()
    example_while_node()
    example_complete_pipeline()

    print("=" * 60)
    print("✅ 所有示例运行完成!")
    print("=" * 60)
