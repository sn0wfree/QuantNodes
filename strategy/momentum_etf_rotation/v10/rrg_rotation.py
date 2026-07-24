"""v10 Strategy 3: RRG Rotation (相对旋转图行业轮动).

基于西部证券《RRG框架下行业与ETF轮动策略构建》(2026).

信号:
  1. RS-Ratio: 220日相对强度比 (相对沪深300)
  2. RS-Momentum: 60日相对动量变化
  3. 四象限分类: 领先/改善/滞后/疲软
  4. 扩散指标: 220日涨幅排名 (简化版)

调仓: 月末
成本: 10bp
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

COST_BP = 10
RS_RATIO_LOOKBACK = 220  # RS-Ratio 回看 (交易日)
RS_MOM_LOOKBACK = 60  # RS-Momentum 回看 (交易日)
DIFFUSION_LOOKBACK = 220  # 扩散指标回看
TOP_N = 6  # 持有前 N 个行业

# 20 个行业 ETF
SECTOR_ETFS = {
    '半导体': '512760',
    '新能车': '515030',
    '光伏': '515790',
    '酒': '512690',
    '医药': '512170',
    '5G': '515050',
    '消费': '159928',
    '证券': '512880',
    '券商': '512000',
    '银行': '512800',
    '煤炭': '515220',
    '地产': '512200',
    '有色': '512400',
    '军工': '512660',
    '传媒': '512980',
    '通信': '515880',
    '家电': '159996',
    '化工': '512120',
}

BENCHMARK = '510300'  # 沪深300


def load_etf_close(code: str) -> pd.Series:
    """加载单个 ETF close price."""
    path = DATA_DIR / f'{code}.parquet'
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    col = 'close' if 'close' in df.columns else df.columns[0]
    return df[col].rename(code).dropna()


def load_all_sectors() -> pd.DataFrame:
    """加载所有行业 ETF + 基准."""
    series_list = []

    # 基准
    bench = load_etf_close(BENCHMARK)
    if len(bench) > 0:
        series_list.append(bench)
        log(f'  基准 {BENCHMARK}: {len(bench)} days')

    # 行业 ETF
    for name, code in SECTOR_ETFS.items():
        s = load_etf_close(code)
        if len(s) > 0:
            series_list.append(s)
            log(f'  {name} ({code}): {len(s)} days')
        else:
            log(f'  ⚠️ {name} ({code}): 无数据')

    df = pd.concat(series_list, axis=1).dropna()
    log(f'  合并后: {len(df)} days, {df.shape[1]} assets')
    return df


def compute_rrg_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """计算 RRG 信号.

    返回: DataFrame, index=日期, columns=[RS-Ratio, RS-Mom, Quadrant, Score]
    """
    benchmark = prices[BENCHMARK]
    results = {}

    for col in prices.columns:
        if col == BENCHMARK:
            continue

        # 相对强度
        rs = prices[col] / benchmark

        # RS-Ratio: 220日相对强度比
        rs_ma_long = rs.rolling(RS_RATIO_LOOKBACK).mean()
        rs_ratio = rs / rs_ma_long * 100

        # RS-Momentum: 60日动量
        rs_ma_short = rs.rolling(RS_MOM_LOOKBACK).mean()
        rs_mom = rs_ma_short / rs_ma_long * 100

        # 扩散指标 (简化: 220日涨幅)
        diffusion = prices[col].pct_change(DIFFUSION_LOOKBACK) * 100

        # 四象限分类
        quadrant = pd.Series('', index=prices.index)
        for i in range(len(prices)):
            if pd.isna(rs_ratio.iloc[i]) or pd.isna(rs_mom.iloc[i]):
                quadrant.iloc[i] = '未知'
            elif rs_ratio.iloc[i] > 100 and rs_mom.iloc[i] > 100:
                quadrant.iloc[i] = '领先'
            elif rs_ratio.iloc[i] <= 100 and rs_mom.iloc[i] > 100:
                quadrant.iloc[i] = '改善'
            elif rs_ratio.iloc[i] <= 100 and rs_mom.iloc[i] <= 100:
                quadrant.iloc[i] = '滞后'
            else:
                quadrant.iloc[i] = '疲软'

        # 综合评分: RS-Ratio × RS-Mom × diffusion
        score = rs_ratio * rs_mom / 100  # 标准化

        results[col] = {
            'RS-Ratio': rs_ratio,
            'RS-Mom': rs_mom,
            'Quadrant': quadrant,
            'Diffusion': diffusion,
            'Score': score,
        }

    return results


def compute_nav(
    prices: pd.DataFrame,
    rrg_signals: dict,
    rebal_dates: pd.DatetimeIndex,
    cost_bp: int = COST_BP,
) -> pd.Series:
    """根据 RRG 信号计算 NAV."""
    nav = pd.Series(1.0, index=prices.index, dtype=float)
    prev_weights = pd.Series(0, index=prices.columns, dtype=float)

    assets = [c for c in prices.columns if c != BENCHMARK]

    for i in range(1, len(prices)):
        date = prices.index[i]

        if date in rebal_dates:
            # 收集当日信号
            scores = {}
            quadrants = {}
            for asset in assets:
                if asset in rrg_signals:
                    q = rrg_signals[asset]['Quadrant'].loc[date] if date in rrg_signals[asset]['Quadrant'].index else '未知'
                    s = rrg_signals[asset]['Score'].loc[date] if date in rrg_signals[asset]['Score'].index else 0
                    d = rrg_signals[asset]['Diffusion'].loc[date] if date in rrg_signals[asset]['Diffusion'].index else 0
                    quadrants[asset] = q
                    # 领先 → 高分, 改善 → 中分, 疲软/滞后 → 低分/负分
                    if q == '领先':
                        scores[asset] = s * (1 + d / 100)
                    elif q == '改善':
                        scores[asset] = s * 0.5
                    else:
                        scores[asset] = 0  # 疲软/滞后不持有

            # 选前 N 个
            valid_scores = {k: v for k, v in scores.items() if v > 0}
            if len(valid_scores) >= TOP_N:
                top_assets = sorted(valid_scores, key=valid_scores.get, reverse=True)[:TOP_N]
            elif len(valid_scores) > 0:
                top_assets = sorted(valid_scores, key=valid_scores.get, reverse=True)
            else:
                top_assets = []

            # 等权
            curr_weights = pd.Series(0, index=prices.columns)
            if top_assets:
                for a in top_assets:
                    curr_weights[a] = 1.0 / len(top_assets)
        else:
            curr_weights = prev_weights.copy()

        # 当日收益
        daily_ret = prices.iloc[i] / prices.iloc[i - 1] - 1

        # 组合收益
        port_ret = (curr_weights * daily_ret).sum()

        # 换手成本
        turnover = (curr_weights - prev_weights).abs().sum()
        cost = turnover * cost_bp / 10000

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)
        prev_weights = curr_weights

    return nav


def metrics(nav: pd.Series, ps: str = '2022-01-01', pe: str = '2026-05-29') -> dict:
    """计算标准指标."""
    seg = nav.loc[ps:pe].dropna()
    if len(seg) < 20:
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
    log('v10 Strategy 3: RRG Rotation')
    log('=' * 60)

    # 加载数据
    log('\n[1] 加载行业 ETF...')
    prices = load_all_sectors()

    # 计算 RRG 信号
    log('\n[2] 计算 RRG 信号...')
    rrg_signals = compute_rrg_signals(prices)

    # 四象限分布
    for asset in list(SECTOR_ETFS.keys())[:5]:
        code = SECTOR_ETFS[asset]
        if code in rrg_signals:
            q = rrg_signals[code]['Quadrant'].dropna()
            dist = q.value_counts()
            log(f'  {asset}: {dict(dist)}')

    # 月末调仓日
    rebal_dates = prices.resample('M').last().index
    log(f'\n  月末调仓日: {len(rebal_dates)} 个')

    # 计算 NAV
    log('\n[3] 计算 RRG NAV...')
    nav = compute_nav(prices, rrg_signals, rebal_dates, cost_bp=COST_BP)

    # 保存
    nav_path = OUT_DIR / 'rrg_rotation_nav.parquet'
    nav.to_frame('nav').to_parquet(nav_path)
    log(f'  保存: {nav_path}')

    # 指标
    log('\n[4] 指标...')
    for period, ps, pe in [
        ('Full', '2019-01-01', '2026-06-30'),
        ('OOS', '2022-01-01', '2026-05-29'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
    ]:
        m = metrics(nav, ps, pe)
        log(f'  {period:6s}: Sharpe={m["Sharpe"]:.3f} AnnRet={m["AnnRet"]:.2%} '
            f'MaxDD={m["MaxDD"]:.2%} MaxDDDays={m["MaxDDDays"]:.0f}')

    # vs 沪深300
    log('\n[5] vs 沪深300...')
    benchmark = prices[BENCHMARK]
    bench_nav = benchmark / benchmark.iloc[0]
    m_bench = metrics(bench_nav, '2022-01-01', '2026-05-29')
    m_strat = metrics(nav, '2022-01-01', '2026-05-29')
    log(f'  沪深300: Sharpe={m_bench["Sharpe"]:.3f} AnnRet={m_bench["AnnRet"]:.2%} MaxDD={m_bench["MaxDD"]:.2%}')
    log(f'  RRG:     Sharpe={m_strat["Sharpe"]:.3f} AnnRet={m_strat["AnnRet"]:.2%} MaxDD={m_strat["MaxDD"]:.2%}')

    log('\n[完成]')


if __name__ == '__main__':
    main()
