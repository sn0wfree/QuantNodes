"""v10 Strategy 1: Dual Momentum (全球宏观配置, 5大类资产轮动).

基于 Gary Antonacci Dual Momentum Investing (2014) 的 GEM 模型.
4 大类 ETF 轮动: A股 / 美股 / 黄金 / 国债.

信号规则:
  1. 绝对动量: 12个月收益 > 0? → 保留, 否则剔除
  2. 相对动量: 在通过筛选的资产中选收益最高者
  3. 全部未通过 → 持有 511260 (国债)

调仓: 月末
成本: 10bp (5bp commission + 5bp slippage)
数据: per_etf 日频 close; 信号用周频, NAV 用日频收益 × 周权重
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info

REPO = Path(__file__).resolve().parents[4]
DATA_DIR = REPO / 'data' / 'real' / 'per_etf'
OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4 大类资产 (去掉港股 159740, 数据太短)
ASSETS = {
    'A股': '510300',    # 沪深300ETF
    '美股': '513100',    # 纳指ETF
    '黄金': '518880',    # 黄金ETF
    '国债': '511260',    # 10Y国债ETF (防御)
}

BOND_CODE = '511260'
COST_BP = 10
LOOKBACK_MONTHS = 12
LOOKBACK_WEEKS = 52


def load_etf_daily(code: str) -> pd.Series:
    """加载单个 ETF close price (日频)."""
    path = DATA_DIR / f'{code}.parquet'
    df = pd.read_parquet(path)
    col = 'close' if 'close' in df.columns else df.columns[0]
    s = df[col].dropna()
    s.name = code
    return s


def load_all_assets_daily() -> pd.DataFrame:
    """加载所有资产 close price (日频)."""
    series_list = []
    for name, code in ASSETS.items():
        try:
            s = load_etf_daily(code)
            series_list.append(s)
            log(f'  {name} ({code}): {len(s)} days, {s.index[0].date()} ~ {s.index[-1].date()}')
        except Exception as e:
            log(f'  ❌ {name} ({code}) 加载失败: {e}')

    df = pd.concat(series_list, axis=1).dropna()
    log(f'  合并后: {len(df)} days, {df.shape[1]} assets')
    return df


def load_all_assets_weekly() -> pd.DataFrame:
    """加载所有资产 close price, resample 到周频 (用于信号计算)."""
    daily = load_all_assets_daily()
    weekly = daily.resample('W-SUN').last()
    return weekly


def dual_momentum_signal(
    prices: pd.DataFrame,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> pd.Series:
    """计算 Dual Momentum 信号.

    返回: Series, index=日期, columns=资产名, value=权重 (0 或 1)
    """
    # 过去 N 周收益率
    returns_lookback = prices.pct_change(lookback_weeks).iloc[-1:]

    # 1. 绝对动量过滤: 收益率 > 0
    total_ret = returns_lookback.iloc[0]
    positive_mask = total_ret > 0

    # 2. 相对动量: 在通过筛选的资产中选最高
    risk_assets = [c for c in prices.columns if c != BOND_CODE]

    weights = pd.Series(0.0, index=prices.columns)

    if positive_mask[risk_assets].any():
        valid_rets = total_ret[risk_assets][positive_mask[risk_assets]]
        best = valid_rets.idxmax()
        weights[best] = 1.0
    else:
        weights[BOND_CODE] = 1.0

    return weights


def compute_nav(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cost_bp: int = COST_BP,
) -> pd.Series:
    """根据月末信号计算 NAV (日频).

    信号用周频价格计算, NAV 用日频收益 × 周权重累乘.
    """
    nav = pd.Series(1.0, index=daily_prices.index, dtype=float)
    prev_weights = pd.Series(0, index=daily_prices.columns, dtype=float)

    for i in range(1, len(daily_prices)):
        date = daily_prices.index[i]

        # 是否调仓日 (月末)
        if date in rebal_dates:
            # 用截至当天的周频数据算信号
            wk = weekly_prices.loc[:date]
            if len(wk) >= LOOKBACK_WEEKS:
                curr_weights = dual_momentum_signal(wk, LOOKBACK_WEEKS)
            else:
                # 预热期: 1/4 等权分配给 4 资产, 剩 3/4 买 511260 国债
                n_assets = len(daily_prices.columns)
                curr_weights = pd.Series(0.0, index=daily_prices.columns)
                curr_weights[BOND_CODE] = 0.75
                for col in daily_prices.columns:
                    if col != BOND_CODE:
                        curr_weights[col] = 0.25 / (n_assets - 1)
        else:
            curr_weights = prev_weights.copy()

        # 日收益
        day_ret = daily_prices.iloc[i] / daily_prices.iloc[i - 1] - 1

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
    log('v10 Strategy 1: Dual Momentum (全球宏观配置)')
    log('=' * 60)

    # 加载数据
    log('\n[1] 加载数据...')
    daily_prices = load_all_assets_daily()
    weekly_prices = daily_prices.resample('W-SUN').last().dropna()

    # 月末调仓日 (在日频数据上找月末)
    rebal_dates = daily_prices.resample('M').last().index
    log(f'  月末调仓日: {len(rebal_dates)} 个')

    # 计算 NAV (日频)
    log('\n[2] 计算 NAV (日频)...')
    nav = compute_nav(daily_prices, weekly_prices, rebal_dates, cost_bp=COST_BP)

    # 保存
    nav_path = OUT_DIR / 'dual_momentum_nav.parquet'
    nav.to_frame('nav').to_parquet(nav_path)
    log(f'  保存: {nav_path}')

    # 持仓统计
    log('\n[3] 持仓统计...')
    signals_log = []
    code_to_name = {v: k for k, v in ASSETS.items()}
    for date in rebal_dates:
        wk = weekly_prices.loc[:date]
        if len(wk) >= LOOKBACK_WEEKS:
            w = dual_momentum_signal(wk, LOOKBACK_WEEKS)
            held = [code_to_name.get(c, c) for c in w[w > 0].index.tolist()]
            signals_log.append({'date': date, 'held': held})

    # 统计各资产被持有次数
    hold_count = {}
    for s in signals_log:
        for a in s['held']:
            hold_count[a] = hold_count.get(a, 0) + 1
    total = len(signals_log)
    for a, c in sorted(hold_count.items(), key=lambda x: -x[1]):
        log(f'  {a}: {c}/{total} ({c/total:.0%})')

    # 指标
    log('\n[4] 指标...')
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

    log('\n[完成]')


if __name__ == '__main__':
    main()
