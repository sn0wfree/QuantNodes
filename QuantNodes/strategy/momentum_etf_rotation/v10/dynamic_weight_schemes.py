"""v10 动态权重分配方案对比.

5 个方案:
  A: 市场状态切换 (Regime-Based)
  B: 波动率目标 (Vol-Targeting)
  C: 回撤控制 (Drawdown Control)
  D: 信号强度加权 (Signal-Weighted)
  E: 混合方案 (Hybrid)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能导入公共模块
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 策略 NAV (日频)
STRATS = {
    'v1.0': 'unified_v1v5_navs_calA.parquet',
    'v7.10': 'v7_10_v56_5bp.parquet',
    'v9macro': 'v9_macro_best_C5.parquet',
    'DualMom': 'v10/dual_momentum_nav.parquet',
}

# 基础权重 (静态 Vol-parity)
BASE_WEIGHTS = {'v1.0': 0.74, 'v7.10': 0.09, 'v9macro': 0.12, 'DualMom': 0.05}


def load_navs() -> pd.DataFrame:
    """加载所有策略 NAV, 统一到日频."""
    navs = {}
    navs['v1.0'] = pd.read_parquet(REPO / 'reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet')['v1.0 locked']
    navs['v7.10'] = pd.read_parquet(REPO / 'reports/momentum_etf_rotation/combo/v7_10_v56_5bp.parquet').iloc[:, 0]
    navs['v9macro'] = pd.read_parquet(REPO / 'reports/momentum_etf_rotation/combo/v9_macro_best_C5.parquet')['nav']
    navs['DualMom'] = pd.read_parquet(REPO / 'reports/momentum_etf_rotation/v10/dual_momentum_nav.parquet').iloc[:, 0]

    # 对齐到公共交易日 (所有源已经是交易日频率)
    df = pd.DataFrame(navs)
    df = df.dropna()
    return df


def compute_nav(prices: pd.DataFrame, weights_history: pd.DataFrame, cost_bp: int = 10) -> pd.Series:
    """根据权重历史计算 NAV."""
    rets = prices.pct_change()
    nav = pd.Series(1.0, index=prices.index, dtype=float)

    for i in range(1, len(prices)):
        date = prices.index[i]
        w = weights_history.loc[date] if date in weights_history.index else weights_history.iloc[i-1]

        # 组合收益
        port_ret = (w * rets.iloc[i]).sum()

        # 换手成本
        if i > 1:
            prev_w = weights_history.loc[prices.index[i-1]] if prices.index[i-1] in weights_history.index else w
            turnover = (w - prev_w).abs().sum()
            cost = turnover * cost_bp / 10000
        else:
            cost = 0

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)

    return nav


def scheme_a_regime(prices: pd.DataFrame) -> pd.DataFrame:
    """方案 A: 市场状态切换.

    信号: 44 ETF 平均 12M 收益 (用 v7.10 近似)
    牛市: DualMom 15% + v7.10 15% + v9macro 15% + v1.0 55%
    熊市: DualMom 0% + v7.10 5% + v9macro 5% + v1.0 90%
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(252, len(prices)):
        date = prices.index[i]
        # 12M 收益 (用 v7.10 近似市场状态)
        ret_12m = prices['v7.10'].iloc[i] / prices['v7.10'].iloc[i-252] - 1

        if ret_12m > 0:
            # 牛市
            weights.loc[date] = {'v1.0': 0.55, 'v7.10': 0.15, 'v9macro': 0.15, 'DualMom': 0.15}
        else:
            # 熊市
            weights.loc[date] = {'v1.0': 0.90, 'v7.10': 0.05, 'v9macro': 0.05, 'DualMom': 0.00}

    return weights


def scheme_b_vol_target(prices: pd.DataFrame) -> pd.DataFrame:
    """方案 B: 波动率目标.

    信号: 组合近 20 日波动率
    高波动 (>15%): 全部切 v1.0
    中波动 (8-15%): 正常 vol-parity
    低波动 (<8%): 增加 DualMom/v7.10
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(60, len(prices)):
        date = prices.index[i]
        # 近 20 日波动率 (用各策略日收益的加权)
        w_mid = pd.Series(BASE_WEIGHTS)
        rets_20d = prices.pct_change().iloc[i-20:i]
        port_rets = (rets_20d * w_mid).sum(axis=1)
        vol_20d = port_rets.std() * np.sqrt(252)

        if vol_20d > 0.15:
            # 高波动 → 防御
            weights.loc[date] = {'v1.0': 0.95, 'v7.10': 0.02, 'v9macro': 0.03, 'DualMom': 0.00}
        elif vol_20d > 0.08:
            # 中波动 → 正常
            weights.loc[date] = BASE_WEIGHTS
        else:
            # 低波动 → 进攻
            weights.loc[date] = {'v1.0': 0.55, 'v7.10': 0.15, 'v9macro': 0.15, 'DualMom': 0.15}

    return weights


def scheme_c_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """方案 C: 回撤控制.

    信号: 组合从峰值回撤
    < 3%: 正常
    3-5%: DualMom 减半, v1.0 加倍
    5-8%: DualMom 清仓, v1.0 90%
    > 8%: 全部切 v1.0
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 先计算基准 NAV
    rets = prices.pct_change()
    port_ret_base = sum(BASE_WEIGHTS[s] * rets[s] for s in prices.columns)
    nav_base = (1 + port_ret_base).cumprod()
    nav_base.iloc[0] = 1.0

    for i in range(1, len(prices)):
        date = prices.index[i]
        peak = nav_base.iloc[:i+1].max()
        dd = nav_base.iloc[i] / peak - 1

        if dd < -0.08:
            # 严重回撤 → 全防御
            weights.loc[date] = {'v1.0': 1.00, 'v7.10': 0.00, 'v9macro': 0.00, 'DualMom': 0.00}
        elif dd < -0.05:
            # 中等回撤 → DualMom 清仓
            weights.loc[date] = {'v1.0': 0.90, 'v7.10': 0.05, 'v9macro': 0.05, 'DualMom': 0.00}
        elif dd < -0.03:
            # 轻微回撤 → DualMom 减半
            weights.loc[date] = {'v1.0': 0.80, 'v7.10': 0.10, 'v9macro': 0.10, 'DualMom': 0.00}
        else:
            # 正常
            weights.loc[date] = BASE_WEIGHTS

    return weights


def scheme_d_signal_weighted(prices: pd.DataFrame) -> pd.DataFrame:
    """方案 D: 信号强度加权.

    信号: 各策略近 12M Sharpe
    权重 = Sharpe_i / Σ Sharpe_j × base_weight_i
    月度调整, 限制单策略权重上下限
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 月末调仓日
    rebal_dates = prices.resample('M').last().index

    prev_w = pd.Series(BASE_WEIGHTS)
    for i in range(252, len(prices)):
        date = prices.index[i]
        if date not in rebal_dates:
            weights.loc[date] = prev_w
            continue

        # 计算各策略近 12M Sharpe
        sharpes = {}
        for col in prices.columns:
            rets_12m = prices[col].iloc[i-252:i].pct_change().dropna()
            if len(rets_12m) > 50:
                ann_ret = (prices[col].iloc[i] / prices[col].iloc[i-252] - 1)
                vol = rets_12m.std() * np.sqrt(252)
                sharpes[col] = ann_ret / vol if vol > 0 else 0
            else:
                sharpes[col] = 0

        # 归一化 Sharpe (转为正数)
        sharpes_pos = {k: max(v, 0) + 0.1 for k, v in sharpes.items()}
        total_s = sum(sharpes_pos.values())

        # 加权
        w = {}
        for col in prices.columns:
            w[col] = (sharpes_pos[col] / total_s) * BASE_WEIGHTS[col]

        # 归一化
        total_w = sum(w.values())
        w = {k: v / total_w for k, v in w.items()}

        # 限制上下限
        for col in w:
            w[col] = max(0.0, min(0.50, w[col]))
        total_w = sum(w.values())
        w = {k: v / total_w for k, v in w.items()}

        weights.loc[date] = w
        prev_w = pd.Series(w)

    return weights


def scheme_e_hybrid(prices: pd.DataFrame) -> pd.DataFrame:
    """方案 E: 混合方案.

    1. 基础权重: v1.0 60% + v7.10 15% + v9macro 15% + DualMom 10%
    2. 市场状态调整: 牛市 DualMom +5%, 熊市 v1.0 +10%
    3. 回撤控制: 回撤 > 5% 时 DualMom 减半
    4. 波动率缩放: 组合vol > 10% 时整体降仓
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 基础权重
    base = {'v1.0': 0.60, 'v7.10': 0.15, 'v9macro': 0.15, 'DualMom': 0.10}

    # 基准 NAV
    rets = prices.pct_change()
    port_ret_base = sum(base[s] * rets[s] for s in prices.columns)
    nav_base = (1 + port_ret_base).cumprod()
    nav_base.iloc[0] = 1.0

    for i in range(252, len(prices)):
        date = prices.index[i]
        w = base.copy()

        # 1. 市场状态调整
        ret_12m = prices['v7.10'].iloc[i] / prices['v7.10'].iloc[i-252] - 1
        if ret_12m > 0:
            w['DualMom'] += 0.05
            w['v1.0'] -= 0.05
        else:
            w['v1.0'] += 0.10
            w['DualMom'] -= 0.05
            w['v7.10'] -= 0.03
            w['v9macro'] -= 0.02

        # 2. 回撤控制
        peak = nav_base.iloc[:i+1].max()
        dd = nav_base.iloc[i] / peak - 1
        if dd < -0.05:
            w['DualMom'] *= 0.5
            w['v1.0'] += w['DualMom'] * 0.5

        # 3. 波动率缩放
        rets_20d = prices.pct_change().iloc[i-20:i]
        vol_20d = (sum(w[s] * rets_20d[s] for s in prices.columns)).std() * np.sqrt(252)
        if vol_20d > 0.10:
            scale = 0.10 / vol_20d
            for col in w:
                if col != 'v1.0':
                    w[col] *= scale
            w['v1.0'] = 1.0 - sum(v for k, v in w.items() if k != 'v1.0')

        # 归一化
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        else:
            w = base.copy()

        # 限制
        for col in w:
            w[col] = max(0.0, min(0.90, w[col]))
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}

        weights.loc[date] = w

    return weights


def metrics(nav: pd.Series, ps: str = '2022-01-01', pe: str = '2026-05-29') -> dict:
    """计算标准指标 (委托给公共模块)."""
    seg = nav.loc[ps:pe].dropna()
    if len(seg) < 20:
        return None
    result = compute_metrics(seg)
    # 转换为旧接口格式 (兼容调用方)
    return {
        'Sharpe': result['Sharpe'], 'Sortino': result['Sortino'],
        'Calmar': result['Calmar'], 'AnnRet': result['AnnRet'],
        'Vol': result['Vol'], 'MaxDD': result['MaxDD'],
        'MaxDDDays': result['MaxDDDays'], 'WinRate': result['WinRate'],
    }


def main():
    log('=' * 60)
    log('v10 动态权重分配方案对比')
    log('=' * 60)

    # 加载数据
    log('\n[1] 加载数据...')
    prices = load_navs()
    log(f'  {len(prices)} days, {prices.index[0].date()} ~ {prices.index[-1].date()}')

    # 生成各方案权重
    log('\n[2] 生成权重...')
    schemes = {
        'A_regime': scheme_a_regime,
        'B_vol_target': scheme_b_vol_target,
        'C_drawdown': scheme_c_drawdown,
        'D_signal_weighted': scheme_d_signal_weighted,
        'E_hybrid': scheme_e_hybrid,
    }

    # 静态 baseline
    static_w = pd.DataFrame(BASE_WEIGHTS, index=prices.index, columns=prices.columns)

    results = {}
    results['Static'] = compute_nav(prices, static_w)

    for name, func in schemes.items():
        log(f'  计算 {name}...')
        w = func(prices)
        nav = compute_nav(prices, w)
        results[name] = nav
        # 保存权重
        w.to_parquet(OUT_DIR / f'dynamic_weights_{name}.parquet')

    # 计算指标
    log('\n[3] 计算指标...')
    rows = []
    for name, nav in results.items():
        m = metrics(nav)
        if m:
            m['Scheme'] = name
            rows.append(m)

    df = pd.DataFrame(rows)
    df = df.sort_values('Sharpe', ascending=False)

    log('\n[4] 结果:')
    cols = ['Scheme', 'Sharpe', 'Sortino', 'Calmar', 'AnnRet', 'MaxDD', 'MaxDDDays', 'WinRate']
    log(df[cols].to_string(index=False))

    # 保存
    df.to_csv(OUT_DIR / 'dynamic_schemes_comparison.csv', index=False)
    log(f'\n保存: {OUT_DIR}/dynamic_schemes_comparison.csv')

    # 各方案保存 NAV
    for name, nav in results.items():
        nav.to_frame('nav').to_parquet(OUT_DIR / f'dynamic_nav_{name}.parquet')

    log('\n[完成]')


if __name__ == '__main__':
    main()
