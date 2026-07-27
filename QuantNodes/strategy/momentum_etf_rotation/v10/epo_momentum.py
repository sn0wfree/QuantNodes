"""v10 Strategy 2: Simplified Momentum + Risk Parity.

简化版: 动量排名 + 等权 + 绝对动量过滤 + Risk Parity 混合.

信号:
  1. 动量排名: 34周收益率排名
  2. 绝对动量: 12个月收益 > 0? → 保留, 否则剔除
  3. 等权持有前 5 个通过筛选的资产
  4. 混合: 60% 等权动量 + 40% Risk Parity

调仓: 月末
成本: 10bp
数据: 信号用周频收益, NAV 用日频收益 × 周权重
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_BP = 10
MOM_LOOKBACK = 34  # 动量回看 (周)
ABS_LOOKBACK = 52  # 绝对动量回看 (周, 12个月)
TOP_N = 5  # 持有前 N 个
MOM_WEIGHT = 0.60  # 动量组合权重
RP_WEIGHT = 0.40  # Risk Parity 权重
MAX_WEIGHT = 0.25  # 单资产最大权重


def load_weekly_returns() -> pd.DataFrame:
    """加载 v7.6 周频收益率 (44 ETF), 用于信号计算."""
    path = REPO / 'data' / 'high_freq_macro' / 'v7_6_Y_weekly.parquet'
    df = pd.read_parquet(path)
    log(f'  周频: {path.name}, shape={df.shape}, range={df.index[0]} ~ {df.index[-1]}')
    return df


def load_daily_returns() -> pd.DataFrame:
    """加载日频收益率, 用于 NAV 计算."""
    # 43 ETF from v56_expanded_daily (日频收益)
    path56 = REPO / 'data' / 'high_freq_macro' / 'v56_expanded_daily.parquet'
    df56 = pd.read_parquet(path56)
    log(f'  日频(v56): {path56.name}, shape={df56.shape}')

    # 511260 from per_etf (日频价格 → 日频收益)
    bond_path = REPO / 'data' / 'real' / 'per_etf' / '511260.parquet'
    bond_df = pd.read_parquet(bond_path)
    col = 'close' if 'close' in bond_df.columns else bond_df.columns[0]
    bond_ret = bond_df[col].pct_change()

    # 合并
    df56['511260'] = bond_ret
    df56 = df56.dropna()
    log(f'  合并后日频: {df56.shape}, range={df56.index[0]} ~ {df56.index[-1]}')
    return df56


def momentum_score(returns: pd.DataFrame, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """动量打分: 过去 N 周收益率."""
    return returns.iloc[-lookback:].sum()


def risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Risk Parity 权重: 逆波动率."""
    vol = returns.std() * np.sqrt(52)
    inv_vol = 1.0 / (vol + 1e-10)
    return inv_vol / inv_vol.sum()


def compute_nav(
    daily_returns: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cost_bp: int = COST_BP,
) -> pd.Series:
    """根据月末信号计算 NAV (日频).

    信号用周频收益计算, NAV 用日频收益 × 周权重累乘.
    """
    nav = pd.Series(1.0, index=daily_returns.index, dtype=float)
    prev_weights = pd.Series(0, index=daily_returns.columns, dtype=float)

    for i in range(1, len(daily_returns)):
        date = daily_returns.index[i]

        # 是否调仓日 (月末)
        if date in rebal_dates:
            # 用截至当天的周频数据算信号
            wk = weekly_returns.loc[:date]
            if len(wk) >= ABS_LOOKBACK:
                # 1. 动量排名
                mom = momentum_score(wk, MOM_LOOKBACK)

                # 2. 绝对动量过滤: 12个月收益 > 0
                abs_ret = wk.iloc[-ABS_LOOKBACK:].sum()
                positive = abs_ret[abs_ret > 0].index

                # 3. 在通过筛选的资产中选动量最强的 TOP_N
                valid_mom = mom[positive]
                if len(valid_mom) >= TOP_N:
                    top_assets = valid_mom.nlargest(TOP_N).index
                elif len(valid_mom) > 0:
                    top_assets = valid_mom.index
                else:
                    top_assets = pd.Index([])

                # 4. 构建组合
                if len(top_assets) > 0:
                    n = len(top_assets)
                    mom_weights = pd.Series(0, index=weekly_returns.columns)
                    mom_weights[top_assets] = 1.0 / n

                    rp_weights = risk_parity_weights(wk.iloc[-52:])

                    combined = MOM_WEIGHT * mom_weights + RP_WEIGHT * rp_weights
                    combined = combined.clip(upper=MAX_WEIGHT)
                    combined = combined / combined.sum()

                    # 映射到日频收益的列 (可能列名不完全一致)
                    curr_weights = pd.Series(0, index=daily_returns.columns)
                    for c in combined.index:
                        if c in curr_weights.index:
                            curr_weights[c] = combined[c]
                else:
                    curr_weights = pd.Series(0, index=daily_returns.columns)
            else:
                curr_weights = prev_weights.copy()
        else:
            curr_weights = prev_weights.copy()

        # 日收益
        day_ret = daily_returns.iloc[i]

        # 组合收益
        port_ret = (curr_weights * day_ret).sum()

        # 换手成本 (仅调仓日)
        if date in rebal_dates:
            turnover = (curr_weights - prev_weights).abs().sum()
            cost = turnover * cost_bp / 10000
        else:
            cost = 0.0

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)
        prev_weights = curr_weights

    return nav


def metrics(nav: pd.Series, ps: str = '2022-01-01', pe: str = '2026-05-29') -> dict:
    """计算标准指标 (日频数据)."""
    seg = nav.loc[ps:pe].dropna()
    if len(seg) < 10:
        return {'Sharpe': 0, 'AnnRet': 0, 'MaxDD': 0, 'MaxDDDays': 0, 'Vol': 0}

    rets = seg.pct_change().dropna()
    total = seg.iloc[-1] / seg.iloc[0] - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_ret / vol) if vol > 0 else 0.0

    peak = seg.cummax()
    max_dd = float((seg / peak - 1).min())
    underwater = (seg < peak).astype(int)
    max_dd_days = int(
        underwater.groupby((underwater != underwater.shift()).cumsum()).sum().max()
    ) if underwater.any() else 0

    return {
        'Sharpe': sharpe,
        'AnnRet': ann_ret,
        'MaxDD': max_dd,
        'MaxDDDays': max_dd_days,
        'Vol': vol,
    }


def main():
    log('=' * 60)
    log('v10 Strategy 2: EPO Optimization')
    log('=' * 60)

    # 加载数据
    log('\n[1] 加载数据...')
    weekly_returns = load_weekly_returns()
    daily_returns = load_daily_returns()

    # 对齐列: 只保留两个数据源共有的列
    common_cols = weekly_returns.columns.intersection(daily_returns.columns)
    weekly_returns = weekly_returns[common_cols]
    daily_returns = daily_returns[common_cols]
    log(f'  共有列: {len(common_cols)} 个')

    # 月末调仓日 (在日频数据上找月末)
    rebal_dates = daily_returns.resample('M').last().index
    log(f'  月末调仓日: {len(rebal_dates)} 个')

    # 计算 NAV (日频)
    log('\n[2] 计算 EPO NAV (日频)...')
    nav = compute_nav(daily_returns, weekly_returns, rebal_dates, cost_bp=COST_BP)

    # 保存
    nav_path = OUT_DIR / 'epo_momentum_nav.parquet'
    nav.to_frame('nav').to_parquet(nav_path)
    log(f'  保存: {nav_path}')

    # 指标
    log('\n[3] 指标...')
    for period, ps, pe in [
        ('Full', '2018-01-01', '2026-06-30'),
        ('OOS', '2022-01-01', '2026-05-29'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
    ]:
        m = metrics(nav, ps, pe)
        log(f'  {period:6s}: Sharpe={m["Sharpe"]:.3f} AnnRet={m["AnnRet"]:.2%} '
            f'MaxDD={m["MaxDD"]:.2%} MaxDDDays={m["MaxDDDays"]:.0f}')

    # vs v7.10
    log('\n[4] vs v7.10...')
    try:
        v710 = pd.read_parquet(
            REPO / 'reports' / 'momentum_etf_rotation' / 'combo' / 'v7_10_v56_5bp.parquet'
        ).iloc[:, 0]
        m_v710 = metrics(v710, '2022-01-01', '2026-05-29')
        m_epo = metrics(nav, '2022-01-01', '2026-05-29')
        log(f'  v7.10:  Sharpe={m_v710["Sharpe"]:.3f} AnnRet={m_v710["AnnRet"]:.2%} MaxDD={m_v710["MaxDD"]:.2%}')
        log(f'  EPO:    Sharpe={m_epo["Sharpe"]:.3f} AnnRet={m_epo["AnnRet"]:.2%} MaxDD={m_epo["MaxDD"]:.2%}')

        # 相关性
        common = nav.index.intersection(v710.index)
        corr = nav.reindex(common).pct_change().corr(v710.reindex(common).pct_change())
        log(f'  相关性: {corr:.3f}')
    except Exception as e:
        log(f'  ⚠️ v7.10 对比失败: {e}')

    log('\n[完成]')


if __name__ == '__main__':
    main()
