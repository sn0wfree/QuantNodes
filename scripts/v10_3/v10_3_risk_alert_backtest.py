"""v10.3 Risk Warning System — Phase 3: Backtest Verification.

Integrate risk alert with G scheme, compare OOS performance.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'QuantNodes'))

from strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
    load_navs, compute_nav, metrics, BASE_WEIGHTS, STRATS,
    scheme_d_signal_weighted, scheme_f_conservative, scheme_g_trend_adaptive,
    _apply_constraints, _compute_target_weights, _compute_sharpes,
)

OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10_3'
OUT_DIR.mkdir(exist_ok=True)
HF_DIR = REPO / 'data' / 'high_freq_macro'

OOS_START = '2022-01-01'
OOS_END = '2026-05-29'

log = print


def load_model():
    import joblib
    model_path = OUT_DIR / 'risk_alert_model.pkl'
    if not model_path.exists():
        log(f'  [WARN] 模型不存在: {model_path}')
        return None
    model = joblib.load(model_path)
    log(f'  加载模型: {len(model["features"])} features')
    return model


def compute_features_for_prediction(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    port_rets = rets.mean(axis=1)
    idx = port_rets.index
    features = pd.DataFrame(index=idx)

    vix = pd.read_parquet(HF_DIR / 'macro_vix_daily.parquet')['vix']
    vix.index = pd.to_datetime(vix.index)
    vix = vix.reindex(idx).ffill()
    features['vix'] = vix
    features['vix_logret'] = np.log(vix / vix.shift(1))
    features['vix_zscore'] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    features['vix_ratio_ma20'] = vix / vix.rolling(20).mean()
    features['vix_ratio_ma60'] = vix / vix.rolling(60).mean()
    features['vix_above_20'] = (vix > 20).astype(float)
    features['vix_above_25'] = (vix > 25).astype(float)
    features['vix_above_30'] = (vix > 30).astype(float)
    features['vix_jump_5d'] = (vix.pct_change(5) > 0.20).astype(float)

    v9_weekly = pd.read_parquet(HF_DIR / 'v9_factors_weekly.parquet')
    v9_weekly.index = pd.to_datetime(v9_weekly.index)
    v9_weekly = v9_weekly.resample('D').ffill()
    for col in v9_weekly.columns:
        s = v9_weekly[col].reindex(idx).ffill()
        features[f'macro_{col}'] = s
        features[f'macro_{col}_zscore'] = (s - s.rolling(13 * 5).mean()) / s.rolling(13 * 5).std()
        features[f'macro_{col}_chg'] = s.diff(5)

    dxy = pd.read_parquet(HF_DIR / 'macro_dxy_daily_v2.parquet')['dxy']
    dxy.index = pd.to_datetime(dxy.index)
    dxy = dxy.reindex(idx).ffill()
    features['dxy'] = dxy
    features['dxy_zscore'] = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    features['dxy_logret'] = np.log(dxy / dxy.shift(1))

    rr = pd.read_parquet(HF_DIR / 'macro_real_rate_daily.parquet')['real_rate']
    rr.index = pd.to_datetime(rr.index)
    rr = rr.reindex(idx).ffill()
    features['real_rate'] = rr
    features['real_rate_zscore'] = (rr - rr.rolling(252).mean()) / rr.rolling(252).std()

    go = pd.read_parquet(HF_DIR / 'gold_oil_corr.parquet')
    go['date'] = pd.to_datetime(go['date'])
    go = go.set_index('date')['gold_oil_corr'].reindex(idx).ffill()
    features['gold_oil_corr'] = go

    cs = pd.read_parquet(HF_DIR / 'cn_us_spread_10y.parquet')
    cs['date'] = pd.to_datetime(cs['date'])
    cs = cs.set_index('date')['cn_us_spread'].reindex(idx).ffill()
    features['cn_us_spread'] = cs
    features['cn_us_spread_zscore'] = (cs - cs.rolling(252).mean()) / cs.rolling(252).std()

    for col in rets.columns:
        r = rets[col]
        features[f'skew_60d_{col}'] = r.rolling(60).skew()
        features[f'kurt_60d_{col}'] = r.rolling(60).kurt()
        features[f'max_dd_60d_{col}'] = r.rolling(60).apply(
            lambda x: (1 + x).cumprod().min() - 1, raw=True
        )
        features[f'vol_20d_{col}'] = r.rolling(20).std() * np.sqrt(252)
        features[f'ret_12m_{col}'] = (1 + r).rolling(252).apply(
            lambda x: x.prod() - 1, raw=True
        )
        features[f'ret_5d_{col}'] = r.rolling(5).sum()
        features[f'ret_20d_{col}'] = r.rolling(20).sum()

    features['ret_dispersion_20d'] = rets.std(axis=1).rolling(20).mean()
    features['port_vol_20d'] = port_rets.rolling(20).std() * np.sqrt(252)
    features['port_ret_12m'] = (1 + port_rets).rolling(252).apply(
        lambda x: x.prod() - 1, raw=True
    )
    features['port_drawdown'] = (1 + port_rets).cumprod() / (1 + port_rets).cumprod().cummax() - 1
    features['vix_x_momentum'] = features['vix'] * features['port_ret_12m']
    vol_high = (features['port_vol_20d'] > features['port_vol_20d'].rolling(252).median()).astype(float)
    features['vix_x_vol_high'] = features['vix'] * vol_high
    features['credit_x_momentum'] = features.get('macro_信用利差因子_zscore', 0) * features['port_ret_12m']

    return features


def get_risk_level(y_prob: float) -> int:
    if y_prob < 0.25:
        return 0
    elif y_prob < 0.50:
        return 1
    elif y_prob < 0.75:
        return 2
    else:
        return 3


def scheme_g_with_risk_alert(prices: pd.DataFrame, features: pd.DataFrame,
                              model_data: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    monthly_dates = set(prices.resample('M').last().index)
    weekly_dates = set(prices.index[4::5])
    market_ret_5d = prices['v7.10'].pct_change(5)
    V10_3G_TREND_THRESH = 0.02
    V10_3G_HYSTERESIS = 0.005
    V10_3G_WEEKLY_MIN_CHANGE = 0.05
    V10_3G_TREND_WINDOW = 5

    prev_w = dict(BASE_WEIGHTS)
    for i in range(min(252, len(prices))):
        weights.iloc[i] = prev_w

    last_trade_w = dict(BASE_WEIGHTS)
    in_d_mode = False
    model = model_data['model']
    model_features = model_data['features']

    for i in range(252, len(prices)):
        date = prices.index[i]
        trend_up = (i >= V10_3G_TREND_WINDOW and market_ret_5d.iloc[i] > V10_3G_TREND_THRESH + V10_3G_HYSTERESIS)
        trend_down = (i >= V10_3G_TREND_WINDOW and market_ret_5d.iloc[i] < V10_3G_TREND_THRESH - V10_3G_HYSTERESIS)

        if trend_up:
            in_d_mode = True
        elif trend_down:
            in_d_mode = False

        risk_level = 0
        if date in features.index:
            feat_row = features.loc[date:date]
            avail = [c for c in model_features if c in feat_row.columns]
            if len(avail) == len(model_features):
                X = feat_row[model_features].fillna(0)
                y_prob = model.predict_proba(X)[0, 1]
                risk_level = get_risk_level(y_prob)

        min_def = {0: 0.25, 1: 0.40, 2: 0.60, 3: 0.90}[risk_level]

        if in_d_mode:
            target = _compute_target_weights(prices, i)
            change = sum(abs(target.get(k, 0) - last_trade_w.get(k, 0)) for k in target)
            if date in monthly_dates or (date in weekly_dates and change > V10_3G_WEEKLY_MIN_CHANGE):
                prev_w = target
                last_trade_w = dict(target)
        elif date in monthly_dates:
            target = _compute_target_weights(prices, i)
            prev_w = _apply_constraints(target, prev_w, min_defensive=min_def)

        weights.loc[date] = prev_w

    return weights


def main():
    log('=' * 60)
    log('v10.3 Risk Warning — Backtest Verification')
    log('=' * 60)

    log('\n[1] 加载数据...')
    prices = load_navs()
    log(f'  {len(prices)} days')

    log('\n[2] 加载模型...')
    model_data = load_model()
    if model_data is None:
        log('  [ABORT] 模型不存在, 跳过回测')
        return

    log('\n[3] 构建特征...')
    features = compute_features_for_prediction(prices)
    log(f'  {features.shape[0]} rows x {features.shape[1]} cols')

    common_idx = prices.index.intersection(features.index)
    prices = prices.loc[common_idx]
    features = features.loc[common_idx]

    log('\n[4] 生成权重...')
    w_g = scheme_g_trend_adaptive(prices)
    log('  G scheme (baseline)...')

    w_g_ra = scheme_g_with_risk_alert(prices, features, model_data)
    log('  G + RiskAlert...')

    static_w = pd.DataFrame(BASE_WEIGHTS, index=prices.index, columns=prices.columns)

    log('\n[5] 回测验证...')
    results = {}
    for name, w in [('Static', static_w), ('G_baseline', w_g), ('G_RiskAlert', w_g_ra)]:
        nav = compute_nav(prices, w, cost_bp=10)
        results[name] = nav

    log('\n--- 5.1 OOS 整体对比 ---')
    rows = []
    for name, nav in results.items():
        m = metrics(nav)
        m['Scheme'] = name
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = ['Scheme', 'Sharpe', 'Sortino', 'Calmar', 'AnnRet', 'MaxDD', 'MaxDDDays', 'WinRate']
    log(df[cols].to_string(index=False))

    log('\n--- 5.2 成本敏感性 ---')
    for cost_bp in [0, 5, 10, 20]:
        rows_cost = []
        for name, w in [('G_baseline', w_g), ('G_RiskAlert', w_g_ra)]:
            nav = compute_nav(prices, w, cost_bp=cost_bp)
            m = metrics(nav)
            m['Scheme'] = name
            m['CostBP'] = cost_bp
            rows_cost.append(m)
        cost_df = pd.DataFrame(rows_cost)
        g_sharpe = cost_df[cost_df['Scheme'] == 'G_baseline']['Sharpe'].values[0]
        ra_sharpe = cost_df[cost_df['Scheme'] == 'G_RiskAlert']['Sharpe'].values[0]
        delta = ra_sharpe - g_sharpe
        log(f'  cost={cost_bp}bp: G={g_sharpe:.3f} G+RA={ra_sharpe:.3f} delta={delta:+.3f}')

    log('\n--- 5.3 假警报分析 ---')
    y_prob = pd.Series(index=features.index, dtype=float)
    for date in features.index:
        feat_row = features.loc[date:date]
        avail = [c for c in model_data['features'] if c in feat_row.columns]
        if len(avail) == len(model_data['features']):
            X = feat_row[model_data['features']].fillna(0)
            y_prob[date] = model_data['model'].predict_proba(X)[0, 1]

    oos_idx = features.index[(features.index >= OOS_START) & (features.index <= OOS_END)]
    y_prob_oos = y_prob.loc[oos_idx]

    rets = prices.pct_change().mean(axis=1)
    fwd_ret_10d = rets.rolling(10).sum().shift(-10)
    fwd_ret_10d = fwd_ret_10d.loc[oos_idx]

    alerts = []
    for date in oos_idx:
        rl = get_risk_level(y_prob_oos[date])
        if rl >= 2:
            actual = fwd_ret_10d.get(date, 0)
            alerts.append({'date': date, 'risk_level': rl, 'prob': y_prob_oos[date],
                           'fwd_10d_ret': actual, 'correct': actual < -0.02})

    if alerts:
        alert_df = pd.DataFrame(alerts)
        n_correct = alert_df['correct'].sum()
        n_alerts = len(alert_df)
        log(f'  总预警次数: {n_alerts}')
        log(f'  正确预警 (fwd_10d < -2%): {n_correct} ({n_correct / n_alerts:.1%})')
        log(f'  假警报率: {1 - n_correct / n_alerts:.1%}')
        alert_df.to_csv(OUT_DIR / 'risk_alert_events.csv', index=False)
        log(f'  保存: {OUT_DIR / "risk_alert_events.csv"}')

    log('\n--- 5.4 极端事件命中 ---')
    events = [
        ('2022-10-01', '2022-10-31', 'A-share bottom 2022-10'),
        ('2022-11-01', '2022-11-30', 'A-share bottom 2022-11'),
        ('2023-03-01', '2023-03-31', 'SVB crisis 2023-03'),
        ('2023-04-01', '2023-04-30', 'AI rotation 2023-04'),
    ]
    for start, end, label in events:
        idx = y_prob_oos.loc[start:end].index
        if len(idx) > 0:
            max_prob = y_prob_oos.loc[idx].max()
            max_rl = get_risk_level(max_prob)
            avg_prob = y_prob_oos.loc[idx].mean()
            log(f'  {label}: avg_prob={avg_prob:.2%} max_prob={max_prob:.2%} max_risk_level={max_rl}')

    log('\n[完成] 回测验证结束')


if __name__ == '__main__':
    main()