"""v10 Strategy 1: Dual Momentum (全球宏观配置, 5大类资产轮动).

基于 Gary Antonacci Dual Momentum Investing (2014) 的 GEM 模型.
4 大类 ETF 轮动: A股 / 美股 / 黄金 / 国债.

信号规则:
  1. 绝对动量: 12个月收益 > 0? → 保留, 否则剔除
  2. 相对动量: 在通过筛选的资产中选收益最高者
  3. 全部未通过 → 持有 511260 (国债)

调仓: 月末
成本: 10bp (5bp commission + 5bp slippage)
数据: per_etf 日频 close, resample 到周频
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info

REPO = Path(__file__).resolve().parents[3]
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


def load_etf_weekly(code: str) -> pd.Series:
    """加载单个 ETF close price, resample 到周频."""
    path = DATA_DIR / f'{code}.parquet'
    df = pd.read_parquet(path)
    col = 'close' if 'close' in df.columns else df.columns[0]
    daily = df[col].dropna()
    weekly = daily.resample('W-SUN').last()
    return weekly


def load_all_assets() -> pd.DataFrame:
    """加载所有资产 close price, resample 到周频."""
    series_list = []
    for name, code in ASSETS.items():
        try:
            s = load_etf_weekly(code)
            series_list.append(s)
            log(f'  {name} ({code}): {len(s)} weeks, {s.index[0].date()} ~ {s.index[-1].date()}')
        except Exception as e:
            log(f'  ❌ {name} ({code}) 加载失败: {e}')

    df = pd.concat(series_list, axis=1).dropna()
    log(f'  合并后: {len(df)} weeks, {df.shape[1]} assets')
    return df


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
    prices: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    cost_bp: int = COST_BP,
) -> pd.Series:
    """根据月末信号计算 NAV (周频)."""
    nav = pd.Series(1.0, index=prices.index, dtype=float)
    prev_weights = pd.Series(0, index=prices.columns, dtype=float)

    for i in range(1, len(prices)):
        date = prices.index[i]

        # 是否调仓日
        if date in rebal_dates:
            # 用截至当天的数据
            hist = prices.loc[:date]
            if len(hist) >= LOOKBACK_WEEKS:
                curr_weights = dual_momentum_signal(hist, LOOKBACK_WEEKS)
            else:
                curr_weights = prev_weights.copy()
        else:
            curr_weights = prev_weights.copy()

        # 周收益
        week_ret = prices.iloc[i] / prices.iloc[i - 1] - 1

        # 组合收益
        port_ret = (curr_weights * week_ret).sum()

        # 换手成本
        turnover = (curr_weights - prev_weights).abs().sum()
        cost = turnover * cost_bp / 10000

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)
        prev_weights = curr_weights

    return nav


def metrics(nav: pd.Series, ps: str = '2022-01-01', pe: str = '2026-05-29') -> dict:
    """计算标准指标 (周频数据)."""
    seg = nav.loc[ps:pe].dropna()
    if len(seg) < 10:
        return {'Sharpe': 0, 'AnnRet': 0, 'MaxDD': 0, 'MaxDDDays': 0, 'Vol': 0}

    rets = seg.pct_change().dropna()
    total = seg.iloc[-1] / seg.iloc[0] - 1
    n_years = len(rets) / 52  # ← 周频: 52 周/年
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
    vol = float(rets.std() * np.sqrt(52))  # ← 周频: sqrt(52)
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
    prices = load_all_assets()

    # 月末调仓日
    rebal_dates = prices.resample('M').last().index
    log(f'  月末调仓日: {len(rebal_dates)} 个')

    # 计算 NAV
    log('\n[2] 计算 NAV...')
    nav = compute_nav(prices, rebal_dates, cost_bp=COST_BP)

    # 保存
    nav_path = OUT_DIR / 'dual_momentum_nav.parquet'
    nav.to_frame('nav').to_parquet(nav_path)
    log(f'  保存: {nav_path}')

    # 持仓统计
    log('\n[3] 持仓统计...')
    signals_log = []
    for date in rebal_dates:
        hist = prices.loc[:date]
        if len(hist) >= LOOKBACK_WEEKS:
            w = dual_momentum_signal(hist, LOOKBACK_WEEKS)
            held = w[w > 0].index.tolist()
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
