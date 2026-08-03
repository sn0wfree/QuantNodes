"""v10 动态权重分配方案对比.

保留方案:
  A: 市场状态切换 (Regime-Based) — 独立模型, 用于市场择时研究
  D: 信号强度加权 (Signal-Weighted) — 基准方案
  F: Conservative - D + 防御底仓 + 换手上限 + hysteresis
  G: Trend-Adaptive - F + v10.2 TrendFilter (生产推荐)
   H: Risk-Aware - G + EnsembleRiskWarning 动态防御底仓

Baseline:
  Static: 固定权重 (74/9/12/5)

已删除: B(波动率目标), C(回撤控制), E(混合) — 被 D/F/G 覆盖

注: v10.3c (CA-GCP-Bounded) 已跳过 - v6 信号 AUC ≈ 0.5 太弱
     (详见 docs/83-ca_gcp_v6_utility_aligned.md 第 5.4 节)
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

# v10.3 Conservative (F) 参数
V10_3F_MIN_DEFENSIVE = 0.25   # v1.0 最低权重 25%
V10_3F_MAX_TURNOVER = 0.30    # 单次调仓最大换手 30%
V10_3F_HYSTERESIS = 0.05      # 权重变化 < 5% 不调仓
V10_3F_DD_GUARD = 0.05        # NAV 回撤 > 5% 时强制 v1.0 90%
V10_3F_REBAL_FREQ = 25        # 调仓频率 25 天 (月频)

# v10.3 Trend-Adaptive (G) 参数 (继承 F + TrendFilter)
V10_3G_TREND_WINDOW = 5       # 5 日 trend
V10_3G_TREND_THRESH = 0.02    # 5 日收益 > 2% → 顺势 (用 base scheme)
V10_3G_HYSTERESIS = 0.005     # 趋势过滤 hysteresis buffer (0.5%)
V10_3G_WEEKLY_MIN_CHANGE = 0.05  # 周频最小变化率, < 5% 不调仓

# v10.3 Risk-Aware (H) 参数
V10_3H_V1_NORMAL = 0.25       # 低风险时 v1.0 最低权重
V10_3H_V1_HIGH = 0.90         # 高风险时 v1.0 最高权重
V10_3H_RISK_MODEL = None      # 延迟初始化: EnsembleRiskWarning 实例


def _risk_prob_to_min_defensive(prob: float) -> float:
    """Map risk probability [0,1] to min_defensive [0.25, 0.90].

    Piecewise linear:
      prob <= 0.3 → 0.25
      0.3 < prob <= 0.5 → 0.25 → 0.45
      0.5 < prob <= 0.7 → 0.45 → 0.65
      0.7 < prob <= 0.9 → 0.65 → 0.85
      prob > 0.9 → 0.90
    """
    if prob <= 0.3:
        return 0.25
    if prob <= 0.5:
        return 0.25 + (prob - 0.3) * (0.45 - 0.25) / 0.2
    if prob <= 0.7:
        return 0.45 + (prob - 0.5) * (0.65 - 0.45) / 0.2
    if prob <= 0.9:
        return 0.65 + (prob - 0.7) * (0.85 - 0.65) / 0.2
    return 0.90


def _get_risk_model() -> object:
    """Lazy-load the risk warning model."""
    global V10_3H_RISK_MODEL
    if V10_3H_RISK_MODEL is None:
        from QuantNodes.strategy.momentum_etf_rotation.v10_3.risk_warning import (
            EnsembleRiskWarning, load_etf_data)
        etf_data = load_etf_data()
        V10_3H_RISK_MODEL = EnsembleRiskWarning(etf_data, "equity")
    return V10_3H_RISK_MODEL


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
    """方案 A: 市场状态切换 (Regime-Based).

    信号: 44 ETF 平均 12M 收益 (用 v7.10 近似)
    牛市: DualMom 15% + v7.10 15% + v9macro 15% + v1.0 55%
    熊市: DualMom 0% + v7.10 5% + v9macro 5% + v1.0 90%

    注: 本方案保留作为独立市场择时模型, 详见 docs/86
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 前 252 天 (warmup) 全部用 BASE_WEIGHTS (与 D/F/G 一致)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = BASE_WEIGHTS

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
    # 前 252 天 (warmup) 全部用 BASE_WEIGHTS (与 F/G 一致)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w

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


def _apply_constraints(
    w_target: dict,
    prev_w: dict,
    min_defensive: float = V10_3F_MIN_DEFENSIVE,
    max_turnover: float = V10_3F_MAX_TURNOVER,
    hysteresis: float = V10_3F_HYSTERESIS,
) -> dict:
    """v10.3 共享约束: 防御底仓 + 换手上限 + hysteresis.

    1. min_defensive: v1.0 最低权重 (防御底仓)
    2. max_turnover: 换手 (Σ |Δw|) 上限, 超过则缩放
    3. hysteresis: 权重变化 < hysteresis 不调仓
    """
    w = dict(w_target)

    # 1. min defensive
    if w.get('v1.0', 0) < min_defensive:
        deficit = min_defensive - w['v1.0']
        w['v1.0'] = min_defensive
        # 从其他策略等比例扣减
        other = sum(v for k, v in w.items() if k != 'v1.0')
        if other > 0:
            for k in w:
                if k != 'v1.0':
                    w[k] -= deficit * w[k] / other

    # 2. hysteresis
    if prev_w:
        drift = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in w)
        if drift < hysteresis:
            return prev_w

    # 3. turnover cap
    if prev_w:
        turnover = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in w)
        if turnover > max_turnover:
            scale = max_turnover / turnover
            for k in w:
                delta = w[k] - prev_w.get(k, 0)
                w[k] = prev_w[k] + delta * scale

    # 4. 归一化 (总和为 1, 移除负值)
    for k in w:
        w[k] = max(0.0, w[k])
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


def scheme_f_conservative(prices: pd.DataFrame) -> pd.DataFrame:
    """v10.3 F: Conservative - 在 D 基础上加防御底仓 + 换手上限 + hysteresis.

    约束:
      - min(v1.0) = 25% (防御底仓)
      - max turnover = 30% per rebalance
      - hysteresis: weight drift < 5% 不调仓
      - DD guard: NAV drawdown > 5% 强制 v1.0 = 90%
      - rebalance freq = 25 days (月频)
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_dates = prices.resample('M').last().index

    prev_w = dict(BASE_WEIGHTS)
    # 前 252 天 (warmup) 全部用 BASE_WEIGHTS
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w

    for i in range(252, len(prices)):
        date = prices.index[i]
        if date not in rebal_dates:
            weights.loc[date] = prev_w
            continue

        # 计算各策略近 12M Sharpe (同 D)
        sharpes = {}
        for col in prices.columns:
            rets_12m = prices[col].iloc[i-252:i].pct_change().dropna()
            if len(rets_12m) > 50:
                ann_ret = prices[col].iloc[i] / prices[col].iloc[i-252] - 1
                vol = rets_12m.std() * np.sqrt(252)
                sharpes[col] = ann_ret / vol if vol > 0 else 0
            else:
                sharpes[col] = 0

        sharpes_pos = {k: max(v, 0) + 0.1 for k, v in sharpes.items()}
        total_s = sum(sharpes_pos.values())
        w_target = {
            col: (sharpes_pos[col] / total_s) * BASE_WEIGHTS[col]
            for col in prices.columns
        }
        total_w = sum(w_target.values())
        w_target = {k: v / total_w for k, v in w_target.items()}

        # 应用 v10.3 约束
        prev_w = _apply_constraints(w_target, prev_w)

        # DD guard: NAV drawdown > 5% 强制 v1.0 = 90%
        rets = prices.pct_change()
        port_ret = sum(prev_w.get(s, 0) * rets[s].iloc[:i+1] for s in prices.columns)
        nav_so_far = (1 + port_ret).cumprod()
        if len(nav_so_far) > 0:
            peak = nav_so_far.max()
            dd = nav_so_far.iloc[-1] / peak - 1
            if dd < -V10_3F_DD_GUARD:
                prev_w = {'v1.0': 0.90, 'v7.10': 0.05, 'v9macro': 0.05, 'DualMom': 0.0}

        weights.loc[date] = prev_w

    return weights


def scheme_g_trend_adaptive(prices: pd.DataFrame) -> pd.DataFrame:
    """v10.3 G: Trend-Adaptive - 在 F 基础上加 v10.2 TrendFilter.

    逻辑:
      - 5 日市场收益 > 2.5% → 顺势: 用 D scheme (signal-weighted)
      - 5 日市场收益 < 1.5% → 逆势: 用 F scheme (conservative)
      - 1.5%~2.5% 之间 → 维持当前模式 (hysteresis)

    调仓规则:
      - 月度: 强制调仓 (所有模式)
      - 周度: D 模式可选, 变化率 < 5% 不调仓
      - 日度: 无
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    monthly_dates = set(prices.resample('M').last().index)
    weekly_dates = set(prices.index[4::5])

    market_ret_5d = prices['v7.10'].pct_change(V10_3G_TREND_WINDOW)

    prev_w = dict(BASE_WEIGHTS)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w

    last_trade_w = dict(BASE_WEIGHTS)
    in_d_mode = False
    for i in range(252, len(prices)):
        date = prices.index[i]
        trend_up = (
            i >= V10_3G_TREND_WINDOW
            and market_ret_5d.iloc[i] > V10_3G_TREND_THRESH + V10_3G_HYSTERESIS
        )
        trend_down = (
            i >= V10_3G_TREND_WINDOW
            and market_ret_5d.iloc[i] < V10_3G_TREND_THRESH - V10_3G_HYSTERESIS
        )

        if trend_up:
            in_d_mode = True
        elif trend_down:
            in_d_mode = False

        if in_d_mode:
            target = _compute_target_weights(prices, i)
            change = sum(abs(target.get(k, 0) - last_trade_w.get(k, 0)) for k in target)
            if date in monthly_dates or (date in weekly_dates and change > V10_3G_WEEKLY_MIN_CHANGE):
                prev_w = target
                last_trade_w = dict(target)
        elif date in monthly_dates:
            target = _compute_target_weights(prices, i)
            prev_w = _apply_constraints(target, prev_w)

        weights.loc[date] = prev_w

    return weights


def scheme_h_risk_aware(prices: pd.DataFrame) -> pd.DataFrame:
    """v10.3 H: Risk-Aware - G + 风险预警动态防御底仓.

    在 G 基础上叠加 EnsembleRiskWarning 风险概率:
      - 5 日市场收益 > 2.5% → D 模式 (signal-weighted)
      - 5 日市场收益 < 1.5% → F 模式, 但 min_defensive 由风险概率动态决定
        prob=0.3 → 0.25, prob=0.5 → 0.40, prob=0.7 → 0.60, prob=0.9 → 0.85

    调仓规则同 G:
      - 月度: 强制调仓 (所有模式)
      - 周度: D 模式可选, 变化率 < 5% 不调仓
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    monthly_dates = set(prices.resample('M').last().index)
    weekly_dates = set(prices.index[4::5])

    market_ret_5d = prices['v7.10'].pct_change(V10_3G_TREND_WINDOW)

    risk_model = None

    prev_w = dict(BASE_WEIGHTS)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w

    last_trade_w = dict(BASE_WEIGHTS)
    in_d_mode = False
    for i in range(252, len(prices)):
        date = prices.index[i]
        trend_up = (
            i >= V10_3G_TREND_WINDOW
            and market_ret_5d.iloc[i] > V10_3G_TREND_THRESH + V10_3G_HYSTERESIS
        )
        trend_down = (
            i >= V10_3G_TREND_WINDOW
            and market_ret_5d.iloc[i] < V10_3G_TREND_THRESH - V10_3G_HYSTERESIS
        )

        if trend_up:
            in_d_mode = True
        elif trend_down:
            in_d_mode = False

        if in_d_mode:
            target = _compute_target_weights(prices, i)
            change = sum(abs(target.get(k, 0) - last_trade_w.get(k, 0)) for k in target)
            if date in monthly_dates or (date in weekly_dates and change > V10_3G_WEEKLY_MIN_CHANGE):
                prev_w = target
                last_trade_w = dict(target)
        elif date in monthly_dates:
            if risk_model is None:
                risk_model = _get_risk_model()
            risk_prob = risk_model.predict(date)
            min_def = _risk_prob_to_min_defensive(risk_prob)
            target = _compute_target_weights(prices, i)
            prev_w = _apply_constraints(target, prev_w, min_defensive=min_def)

        weights.loc[date] = prev_w

    return weights


def _compute_sharpes(prices: pd.DataFrame, i: int) -> dict:
    """计算各策略近 12M Sharpe."""
    sharpes = {}
    for col in prices.columns:
        rets_12m = prices[col].iloc[i-252:i].pct_change().dropna()
        if len(rets_12m) > 50:
            ann_ret = prices[col].iloc[i] / prices[col].iloc[i-252] - 1
            vol = rets_12m.std() * np.sqrt(252)
            sharpes[col] = ann_ret / vol if vol > 0 else 0
        else:
            sharpes[col] = 0
    return sharpes


def _compute_target_weights(prices: pd.DataFrame, i: int) -> dict:
    """计算 D 模式目标权重 (基于 Sharpe 信号)."""
    sharpes = _compute_sharpes(prices, i)
    sharpes_pos = {k: max(v, 0) + 0.1 for k, v in sharpes.items()}
    total_s = sum(sharpes_pos.values())
    w_target = {
        col: (sharpes_pos[col] / total_s) * BASE_WEIGHTS[col]
        for col in prices.columns
    }
    total_w = sum(w_target.values())
    return {k: v / total_w for k, v in w_target.items()}


def main():
    log('=' * 60)
    log('v10 动态权重分配方案对比 (含 v10.3 F/G)')
    log('=' * 60)

    # 加载数据
    log('\n[1] 加载数据...')
    prices = load_navs()
    log(f'  {len(prices)} days, {prices.index[0].date()} ~ {prices.index[-1].date()}')

    # 生成各方案权重
    log('\n[2] 生成权重...')
    schemes = {
        'A_regime': scheme_a_regime,
        'D_signal_weighted': scheme_d_signal_weighted,
        'F_v10_3_conservative': scheme_f_conservative,
        'G_v10_3_trend_adaptive': scheme_g_trend_adaptive,
        'H_v10_3_risk_aware': scheme_h_risk_aware,
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
