"""v10.3 Risk Warning — GARCH Volatility Forecasting Model.

GARCH(1,1) for each asset class. Predicts forward 20d conditional volatility.
Compared to rolling 80th percentile → binary risk signal.

Why GARCH instead of LightGBM:
  - GARCH is purpose-built for volatility modeling (vol clustering)
  - Generative model: directly models conditional variance
  - No feature engineering needed
  - Naturally handles regime changes through persistence parameter
  - Walk-forward refit every 20 days is computationally feasible
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10_3'
OUT_DIR.mkdir(exist_ok=True)
OOS_START = '2022-01-04'
OOS_END = '2026-05-29'
log = print


@dataclass
class AssetClassConfig:
    name: str
    etfs: list[str]
    weights: list[float] | None = None
    vol_window: int = 20
    vol_percentile: float = 0.80
    lookback: int = 252


EQUITY_CFG = AssetClassConfig(
    name='equity', etfs=['510300'],
    vol_window=20, vol_percentile=0.80, lookback=252,
)

COMMODITY_CFG = AssetClassConfig(
    name='commodity', etfs=['518880', '159985', '161226', '159981'],
    weights=[0.50, 0.20, 0.15, 0.15],
    vol_window=20, vol_percentile=0.80, lookback=252,
)


def load_etf_data() -> pd.DataFrame:
    df = pd.read_parquet(REPO / 'data/real/etf_nav_2018-01-01_2026-06-30.parquet')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    return df


def compute_asset_nav(etf_data: pd.DataFrame, cfg: AssetClassConfig) -> pd.Series:
    avail = [c for c in cfg.etfs if c in etf_data.columns]
    if not avail:
        raise ValueError(f'No ETFs for {cfg.name}: {cfg.etfs}')
    if cfg.weights:
        w = dict(zip(cfg.etfs, cfg.weights))
        total = sum(w[e] for e in avail)
        nav = sum(etf_data[e] * w[e] / total for e in avail)
    else:
        nav = etf_data[avail[0]]
    nav.name = cfg.name
    return nav


def compute_realized_metrics(rets: pd.Series, cfg: AssetClassConfig) -> tuple:
    """Compute realized forward vol and threshold.

    Returns:
        realized_fwd_vol: actual 20d forward vol (annualized)
        threshold: rolling 252d 80th percentile of realized vol
    """
    realized_fwd_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window) * np.sqrt(252)
    threshold = realized_fwd_vol.rolling(cfg.lookback, min_periods=60).quantile(cfg.vol_percentile)
    return realized_fwd_vol, threshold


def evaluate(y_true: pd.Series, y_prob: pd.Series) -> dict:
    """Evaluate binary classification performance."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    common = y_true.dropna().index.intersection(y_prob.dropna().index)
    yt, yp = y_true.loc[common], y_prob.loc[common]
    if yt.nunique() < 2:
        return {'AUC': 0.5, 'PR-AUC': yt.mean(), 'Brier': 0.25,
                'PosRate': yt.mean(), 'N': len(yt)}
    return {
        'AUC': roc_auc_score(yt, yp),
        'PR-AUC': average_precision_score(yt, yp),
        'Brier': brier_score_loss(yt, yp),
        'PosRate': yt.mean(), 'N': len(yt),
    }


def run_garch_model(cfg: AssetClassConfig):
    """Run GARCH(1,1) walk-forward volatility forecasting.

    Steps:
      1. Load NAV, compute returns
      2. For each walk-forward fold:
         a. Fit GARCH(1,1) on training returns
         b. Forecast 20d ahead conditional volatility
         c. Compare forecast to threshold → risk signal
      3. Compute AUC, Brier, PR-AUC
    """
    from arch import arch_model

    log('=' * 60)
    log(f'GARCH Volatility Model: {cfg.name}')
    log(f'  ETFs: {cfg.etfs}')
    if cfg.weights:
        log(f'  Weights: {dict(zip(cfg.etfs, cfg.weights))}')
    log('=' * 60)

    log('\n[1] Loading data...')
    etf_data = load_etf_data()
    nav = compute_asset_nav(etf_data, cfg)
    rets = nav.pct_change().dropna() * 100  # GARCH works better with %
    log(f'  Returns: {len(rets)}d, {rets.index[0].date()} ~ {rets.index[-1].date()}')

    log('\n[2] Computing realized metrics...')
    realized_fwd_vol, threshold = compute_realized_metrics(rets / 100, cfg)
    y_true = (realized_fwd_vol > threshold).astype(float).dropna()
    log(f'  y_true: {y_true.mean():.1%} positive ({int(y_true.sum())}/{len(y_true)})')

    log('\n[3] Walk-forward GARCH forecasting...')
    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.index)

    # Expanding window: refit GARCH every 20 days
    refit_dates = pd.date_range(OOS_START, rets.index[-1], freq='20B')
    refit_dates = [d for d in refit_dates if d in rets.index]

    # First training window: 504 days (2 years) before OOS
    first_train_end = rets.index.get_indexer([pd.Timestamp(OOS_START)], method='ffill')[0]
    train_start = 0
    train_end = max(train_start + 504, first_train_end - 20)

    forecasts = pd.Series(index=oos_idx, dtype=float)
    fit_times = []

    for refit_date in refit_dates:
        refit_idx = rets.index.get_indexer([refit_date], method='ffill')[0]
        if refit_idx < 504:
            continue

        train_rets = rets.iloc[train_start:refit_idx]
        log(f'  Fitting GARCH on {train_rets.index[0].date()} ~ {train_rets.index[-1].date()} ({len(train_rets)}d)')

        try:
            model = arch_model(train_rets, vol='Garch', p=1, q=1,
                               dist='normal', rescale=False)
            res = model.fit(disp='off', show_warning=False)

            # Forecast 20 steps ahead
            forecast = res.forecast(horizon=20)
            # GARCH forecasts are in variance units, convert to vol
            fwd_var = forecast.variance.iloc[-1]
            fwd_vol = np.sqrt(fwd_var.values)  # daily vol in %

            # Align forecast to dates
            forecast_start = refit_date + pd.Timedelta(days=1)
            forecast_dates = pd.bdate_range(forecast_start, periods=20)
            forecast_dates = [d for d in forecast_dates if d in forecasts.index]

            for j, fd in enumerate(forecast_dates):
                if j < len(fwd_vol):
                    # Annualize: daily vol * sqrt(252)
                    forecasts[fd] = fwd_vol[j] * np.sqrt(252) / 100

            fit_times.append(len(train_rets))

        except Exception as e:
            log(f'    GARCH fit failed: {e}')
            continue

    # Fill any missing forecasts with forward fill
    forecasts = forecasts.fillna(method='ffill')

    # Compute binary risk signal: GARCH forecast > threshold
    common = forecasts.dropna().index.intersection(y_true.index)
    y_garch = forecasts.loc[common]
    y_actual = y_true.loc[common]
    threshold_aligned = threshold.loc[common]

    y_prob = (y_garch > threshold_aligned).astype(float)

    log(f'\n  GARCH forecasts: {len(forecasts.dropna())} valid, '
        f'{len(common)} aligned with y_true')

    # Evaluate
    metrics = evaluate(y_actual, y_prob)
    log(f'\n  === GARCH Walk-forward Results ===')
    log(f'  AUC: {metrics["AUC"]:.4f}')
    log(f'  PR-AUC: {metrics["PR-AUC"]:.4f}')
    log(f'  Brier: {metrics["Brier"]:.4f}')
    log(f'  PosRate: {metrics["PosRate"]:.1%}')

    # Also evaluate GARCH forecast as continuous probability (not binary)
    # Normalize forecast to [0,1] using rank within rolling window
    y_prob_cont = pd.Series(index=common, dtype=float)
    for i in range(len(common)):
        d = common[i]
        hist = y_garch.loc[:d].rolling(252, min_periods=60).rank(pct=True)
        y_prob_cont.iloc[i] = hist.loc[d]

    metrics_cont = evaluate(y_actual, y_prob_cont)
    log(f'\n  === GARCH Continuous Probability ===')
    log(f'  AUC: {metrics_cont["AUC"]:.4f}')
    log(f'  PR-AUC: {metrics_cont["PR-AUC"]:.4f}')
    log(f'  Brier: {metrics_cont["Brier"]:.4f}')

    # Save results
    results = pd.DataFrame({
        'y_true': y_actual,
        'y_garch_vol': y_garch,
        'y_garch_binary': y_prob,
        'y_garch_prob': y_prob_cont,
        'threshold': threshold_aligned,
    })
    results.to_parquet(OUT_DIR / f'garch_{cfg.name}_results.parquet')
    log(f'\n  Saved: {OUT_DIR / f"garch_{cfg.name}_results.parquet"}')

    # Summary CSV
    summary = pd.DataFrame([{
        'asset': cfg.name,
        'model': 'GARCH(1,1)',
        'auc_binary': metrics['AUC'],
        'auc_cont': metrics_cont['AUC'],
        'prauc_binary': metrics['PR-AUC'],
        'prauc_cont': metrics_cont['PR-AUC'],
        'brier_binary': metrics['Brier'],
        'brier_cont': metrics_cont['Brier'],
        'pos_rate': metrics['PosRate'],
        'n_samples': metrics['N'],
        'n_fits': len(fit_times),
        'avg_train_size': np.mean(fit_times) if fit_times else 0,
    }])
    csv_path = OUT_DIR / f'garch_{cfg.name}_summary.csv'
    summary.to_csv(csv_path, index=False)
    log(f'  Saved: {csv_path}')

    return results


if __name__ == '__main__':
    log('=' * 60)
    log('GARCH Volatility Forecasting Model')
    log('=' * 60)
    for cfg in [EQUITY_CFG, COMMODITY_CFG]:
        run_garch_model(cfg)
        log('\n')
    log('[Done]')