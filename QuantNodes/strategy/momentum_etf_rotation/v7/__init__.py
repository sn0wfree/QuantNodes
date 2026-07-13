# coding=utf-8
"""V7 子包入口 — 暴露核心 API + 端到端入口.

公开 API:
    V7_3Config, V7_4Config
    V7_3SubStrategy
    run_v7_3_backtest (端到端)
    v7_macro_baseline (锁定 baseline, 2026-07-13)
    v7_macro_baseline_v2_tf (趋势过滤)
    v7_macro_baseline_v3_momentum (动量叠加)
    v7_macro_baseline_v4_expanded (扩大资产池)
    v7_macro_baseline_v5_stop_loss (硬止损)
    RollingSymmetry
    BootstrapLassoMapping
    FactorRiskParityOptimizer
"""
from .bootstrap_lasso import BootstrapLassoMapping
from .data_loader import (
    FACTOR_COLS,
    INDEX_COLS,
    EXPANDED_COLS,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
    EXPANDED_BOND_INDICES,
    load_benchmark_price,
    load_factor_returns,
    load_index_panel,
    load_index_prices,
    load_expanded_panel,
    load_macro_factors,
)
from .factor_risk_parity import FactorRiskParityOptimizer
from .macro_substrategy_v7_3 import (
    V7_3Config,
    V7_4Config,
    V7_5Config,
    V7_3SubStrategy,
    run_v7_3_backtest,
    apply_trend_filter,
    apply_trend_score_filter,
    compute_trend_score,
)
from .symmetry import RollingSymmetry

__all__ = [
    "V7_3Config",
    "V7_4Config",
    "V7_5Config",
    "V7_3SubStrategy",
    "run_v7_3_backtest",
    "apply_trend_filter",
    "apply_trend_score_filter",
    "compute_trend_score",
    "v7_macro_baseline",
    "v7_macro_baseline_v2_tf",
    "v7_macro_baseline_v3_momentum",
    "v7_macro_baseline_v4_expanded",
    "v7_macro_baseline_v5_stop_loss",
    "v7_macro_baseline_v5_tf_score",
    "RollingSymmetry",
    "BootstrapLassoMapping",
    "FactorRiskParityOptimizer",
    "load_macro_factors",
    "load_factor_returns",
    "load_index_panel",
    "load_index_prices",
    "load_expanded_panel",
    "load_benchmark_price",
    "FACTOR_COLS",
    "INDEX_COLS",
    "EXPANDED_COLS",
    "EQUITY_ETF_COLS",
    "COMMODITY_ETF_COLS",
    "EXPANDED_BOND_INDICES",
]


def v7_macro_baseline() -> V7_3Config:
    """v7 宏观子策略 baseline 锁定 (Stage 30.4 完整版, 2026-07-13).

    算法: Symmetry (Klein 2013) + Bootstrap-Lasso × 500 + 源 FactorRiskParity
    数据: 13 指数 (含 中债1-3年国债财富指数) + 9 宏观因子 (8 实际使用)
    调仓: 季度, 8 quarter 滚动窗口
    成本: 5bp 佣金 + 5bp 滑点

    业绩 (3 个 random_state [42, 7, 123] 平均, OOS 2023-至今):
        Ann 5.24%, Vol 6.74%, Sharpe 0.778, DD -8.45%, Calmar 0.620
    OOS 2022-至今:  Ann 3.37%, Vol 6.99%, Calmar 0.371
    全期 2010-2026: Ann 2.94%, Vol 7.20%, Calmar 0.145

    用途: 宏观子策略 baseline, 与 v6.2 (行业轮动) 配合用
    锁定目的: 未来 v7.x 变更 (新因子 / 新池 / 新算法) 必对照此 baseline,
              退化 > 5% 需更新 baseline 并加 migration note.
              详见 tests/.../v7/test_v7_macro_baseline.py
    """
    return V7_3Config(
        bootstrap_times=500,           # 收敛点 (敏感性分析确认)
        bootstrap_resample_min=78,    # 1.5 年
        bootstrap_resample_max=104,   # 2 年
        bootstrap_random_state=42,
        bootstrap_cache_alpha=True,   # 30x 加速
        quarter_window=8,             # 2 年回看
        max_weight=0.5,               # 源 cell 99
        sum_lower=0.9,                # 源 cell 94
        sum_upper=1.0,
        commission_bp=5.0,
        slippage_bp=5.0,
    )


def v7_macro_baseline_v2_tf() -> V7_3Config:
    """v7 宏观子策略 baseline v2: 加 趋势过滤 (TF, 2026-07-13).

    相对 v7_macro_baseline 改动 (单点修复, ROI 最高):
    - trend_filter_enabled: True
    - trend_filter_ma: 200 日 (沪深300)
    - trend_filter_bear: 0.5 (熊市半仓)
    - 防御资产: 中债10年期国债指数 (池内最稳)

    TF 逻辑 (每个调仓日检查):
        if 沪深300 < 200 日 MA:
            w = w * 0.5
            w['中债10年期国债指数'] += 0.5
        else:
            w 保持不变

    预期性能 (OOS 2018-至今, 修复 v7_macro_baseline 缺 #1 root cause):
        Ann 3.0-3.5%, Vol 5.5-6.5%, Sharpe 0.6-0.8
        DD -5% ~ -7% (vs v1 -8.98%), Calmar 0.5-0.8 (vs v1 0.387)
    OOS 2023-至今预估: Calmar 0.7-0.9 (vs v1 0.620)

    用途: 探索版 baseline, 验证 TF 修复 ROI. 用户 2 选 1 (vs v7_macro_baseline).
    注: v7_macro_baseline 锁定不动, 严格遵守 v7 baseline 锁定规则.
    """
    cfg = v7_macro_baseline()
    cfg.trend_filter_enabled = True
    cfg.trend_filter_benchmark = "沪深300指数"
    cfg.trend_filter_ma = 200
    cfg.trend_filter_bear = 0.5
    return cfg


def v7_macro_baseline_v4_expanded(**overrides) -> V7_4Config:
    """v7 宏观子策略 v4: 扩大资产池 (51 ETFs + 5 bond indices = 56 assets).

    相对 v7+v2 TF 改动:
    - asset_pool: "expanded" (56 assets vs 13 indices)
    - 51 ETFs: A股宽基6 + 行业20 + 港股5 + 商品6 + 海外6 + SmartBeta8
    - 5 bond indices: 中债10年/3-5年/1-3年/国开/企业债
    - TF: equity ETFs → bond indices (flight to safety)

    预期优势:
    - 更多资产 = 更好分散化 + 更多轮动机会
    - ETF 可直接交易 (vs 指数不可交易)
    - 商品 ETF 提供通胀对冲

    用途: 探索扩大资产池对 v7 框架的影响.
    """
    base = v7_macro_baseline_v2_tf()
    return V7_4Config(
        asset_pool="expanded",
        equity_cols=EQUITY_ETF_COLS,
        commodity_cols=COMMODITY_ETF_COLS,
        bond_cols=EXPANDED_BOND_INDICES,
        # 继承 v7+v2 TF 设置
        trend_filter_enabled=base.trend_filter_enabled,
        trend_filter_benchmark=base.trend_filter_benchmark,
        trend_filter_ma=base.trend_filter_ma,
        trend_filter_bear=base.trend_filter_bear,
        # 继承 baseline 设置
        bootstrap_times=base.bootstrap_times,
        bootstrap_resample_min=base.bootstrap_resample_min,
        bootstrap_resample_max=base.bootstrap_resample_max,
        bootstrap_random_state=base.bootstrap_random_state,
        bootstrap_cache_alpha=base.bootstrap_cache_alpha,
        quarter_window=base.quarter_window,
        max_weight=base.max_weight,
        sum_lower=base.sum_lower,
        sum_upper=base.sum_upper,
        commission_bp=base.commission_bp,
        slippage_bp=base.slippage_bp,
        **overrides,
    )


def v7_macro_baseline_v5_stop_loss(**overrides) -> V7_4Config:
    """v7 宏观子策略 v5.0: 硬止损 (Stop Loss, 2026-07-13).

    相对 v7+v4 expanded 改动 (单点修复, 风控基础):
    - stop_loss_enabled: True
    - stop_loss_threshold: -0.10 (10% DD 触发)
    - stop_loss_bond_alloc: 1.0 (止损后 100% 债券)
    - 触发后: equity 仓位清零, 释放权重分配给债券 (flight to safety)

    设计动机 (用户深度讨论 2026-07-13):
    "金融预测存在极限, 这条止损线是所有逻辑假设失效时的最后实盘生存保障"
    — 简单有效, 不依赖任何宏观/市场信号, 防止系统性风险

    与传统风控区别:
    - 不基于 VaR / ES (不依赖分布假设)
    - 不基于宏观象限 (不依赖预测)
    - 纯粹 NAV-based 硬线, 触发后 100% 债券

    预期效果 (OOS 2022-2026):
    - DD: -11.6% → ~-12% (止损会限制部分反弹, 但截断大幅下跌)
    - Calmar: 0.499 → ~0.55+ (改善风险调整)
    - 适用场景: 系统性大跌 (如 2022/2024 熊市)

    用途: 在 v4 expanded 基础上增加硬止损, 防止极端行情.
    """
    base = v7_macro_baseline_v4_expanded()
    cfg = V7_4Config(
        asset_pool=base.asset_pool,
        equity_cols=base.equity_cols,
        commodity_cols=base.commodity_cols,
        bond_cols=base.bond_cols,
        # 继承 v4 expanded TF
        trend_filter_enabled=base.trend_filter_enabled,
        trend_filter_benchmark=base.trend_filter_benchmark,
        trend_filter_ma=base.trend_filter_ma,
        trend_filter_bear=base.trend_filter_bear,
        # 继承 baseline 设置
        bootstrap_times=base.bootstrap_times,
        bootstrap_resample_min=base.bootstrap_resample_min,
        bootstrap_resample_max=base.bootstrap_resample_max,
        bootstrap_random_state=base.bootstrap_random_state,
        bootstrap_cache_alpha=base.bootstrap_cache_alpha,
        quarter_window=base.quarter_window,
        max_weight=base.max_weight,
        sum_lower=base.sum_lower,
        sum_upper=base.sum_upper,
        commission_bp=base.commission_bp,
        slippage_bp=base.slippage_bp,
        # [v5 硬止损] 开启止损
        stop_loss_enabled=True,
        stop_loss_threshold=-0.10,
        stop_loss_bond_alloc=1.0,
        **overrides,
    )
    return cfg


def v7_macro_baseline_v5_tf_score(**overrides) -> V7_5Config:
    """v7 宏观子策略 v5.1: 连续 TF Score (替代二值 MA200, 2026-07-13).

    相对 v7+v2 TF (二值) 改动:
    - trend_filter_enabled: False (关闭二值)
    - tf_score_enabled: True (开启连续 score)
    - score = 0.5 × MA200距离 + 0.3 × 60日动量 + 0.2 × 波动率比率
    - 仓位: 强熊 (-0.3) → 30%, 强牛 (+0.3) → 120%, 中间线性插值

    关键改进 (相对 v2 二值 MA200):
    1. **信息保留**: 距 MA200 5% 和 20% 触发不同减仓, 不再二值化
    2. **多因子**: MA200 + 60日动量 + 波动率, 解决 "只靠 MA200 反应慢"
    3. **平滑过渡**: 介于 bear/bull 之间线性插值, 避免信号突变
    4. **可超配**: 牛市时权益 > 100% (杠杆效果, 受 max_weight 约束)

    权重设计 (用户决策):
    - MA200 距离 0.5 (主信号, 反映长期趋势)
    - 60 日动量 0.3 (中期确认, 避免 MA200 滞后)
    - 波动率比率 0.2 (反向, 高 vol = 恐慌 → 减仓)

    用途: 替代 v2 二值 TF, 提供更平滑/及时的趋势信号.
    """
    base = v7_macro_baseline_v4_expanded()
    return V7_5Config(
        asset_pool=base.asset_pool,
        equity_cols=base.equity_cols,
        commodity_cols=base.commodity_cols,
        bond_cols=base.bond_cols,
        # [v5.1 关键] 关闭二值 TF, 开启连续 TF Score
        trend_filter_enabled=False,  # 关闭二值 MA200
        # 继承 baseline 设置
        bootstrap_times=base.bootstrap_times,
        bootstrap_resample_min=base.bootstrap_resample_min,
        bootstrap_resample_max=base.bootstrap_resample_max,
        bootstrap_random_state=base.bootstrap_random_state,
        bootstrap_cache_alpha=base.bootstrap_cache_alpha,
        quarter_window=base.quarter_window,
        max_weight=base.max_weight,
        sum_lower=base.sum_lower,
        sum_upper=base.sum_upper,
        commission_bp=base.commission_bp,
        slippage_bp=base.slippage_bp,
        # [v5 硬止损] 默认关闭 (用户可单独开启)
        stop_loss_enabled=False,
        # [v5.1 连续 TF Score]
        tf_score_enabled=True,
        # 继承 expanded pool 设置
        **overrides,
    )
