"""v10.3 Risk Warning — GARCH Volatility Forecasting Model (v2).

GARCH family models for volatility forecasting per asset class.

Improvements over v1:
  - GJR-GARCH: captures leverage effect (negative returns → higher vol)
  - Student-t distribution: fatter tails, more realistic
  - Daily refit: maximum freshness
  - Side-by-side comparison with baseline GARCH(1,1) normal
"""
from __future__ import annotations
import sys, warnings, time
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


@dataclass
class ModelSpec:
    name: str
    vol: str = 'Garch'
    p: int = 1
    o: int = 0
    q: int = 1
    dist: str = 'normal'
    refit_freq: int = 1  # days between refits

    @property
    def label(self) -> str:
        if self.o > 0:
            base = f'GJR-GARCH({self.p},{self.o},{self.q})'
        else:
            base = f'GARCH({self.p},{self.q})'
        return f'{base} {self.dist}'


MODELS = [
    ModelSpec('garch_norm', vol='Garch', p=1, o=0, q=1, dist='normal', refit_freq=20),
    ModelSpec('garch_t', vol='Garch', p=1, o=0, q=1, dist='studentst', refit_freq=1),
    ModelSpec('gjrgarch_t', vol='Garch', p=1, o=1, q=1, dist='studentst', refit_freq=1),
]


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
    realized_fwd_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window) * np.sqrt(252)
    threshold = realized_fwd_vol.rolling(cfg.lookback, min_periods=60).quantile(cfg.vol_percentile)
    return realized_fwd_vol, threshold


def evaluate(y_true: pd.Series, y_prob: pd.Series) -> dict:
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


def run_model(rets: pd.Series, y_true: pd.Series, threshold: pd.Series,
              cfg: AssetClassConfig, model_spec: ModelSpec) -> dict:
    """Run a single GARCH model walk-forward."""
    from arch import arch_model

    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.index)

    # Refit dates
    refit_dates = rets.index[rets.index.isin(oos_idx)]
    refit_dates = refit_dates[::model_spec.refit_freq]
    if len(refit_dates) == 0:
        refit_dates = [oos_idx[0]]

    # Ensure first refit date has enough training data
    refit_dates = [d for d in refit_dates if rets.index.get_indexer([d], method='ffill')[0] >= 504]

    forecasts = pd.Series(index=oos_idx, dtype=float)
    fit_times = []

    for refit_date in refit_dates:
        refit_idx = rets.index.get_indexer([refit_date], method='ffill')[0]
        train_rets = rets.iloc[:refit_idx]

        try:
            m = arch_model(train_rets, vol=model_spec.vol,
                           p=model_spec.p, o=model_spec.o, q=model_spec.q,
                           dist=model_spec.dist, rescale=False)
            res = m.fit(disp='off', show_warning=False)

            forecast = res.forecast(horizon=20)
            fwd_var = forecast.variance.iloc[-1]
            fwd_vol = np.sqrt(fwd_var.values) * np.sqrt(252) / 100

            forecast_start = refit_date + pd.Timedelta(days=1)
            forecast_dates = pd.bdate_range(forecast_start, periods=20)
            forecast_dates = [d for d in forecast_dates if d in forecasts.index]

            for j, fd in enumerate(forecast_dates):
                if j < len(fwd_vol):
                    forecasts[fd] = fwd_vol[j]

            fit_times.append(len(train_rets))

        except Exception as e:
            log(f'    {model_spec.label}: fit failed at {refit_date.date()}: {e}')
            continue

    forecasts = forecasts.fillna(method='ffill')

    common = forecasts.dropna().index.intersection(y_true.index)
    y_garch = forecasts.loc[common]
    y_actual = y_true.loc[common]
    threshold_aligned = threshold.loc[common]

    # Binary signal
    y_prob_bin = (y_garch > threshold_aligned).astype(float)
    metrics_bin = evaluate(y_actual, y_prob_bin)

    # Continuous probability (rolling rank)
    y_prob_cont = pd.Series(index=common, dtype=float)
    for i in range(len(common)):
        d = common[i]
        hist = y_garch.loc[:d].rolling(252, min_periods=60).rank(pct=True)
        y_prob_cont.iloc[i] = hist.loc[d]
    metrics_cont = evaluate(y_actual, y_prob_cont)

    return {
        'name': model_spec.label,
        'key': model_spec.name,
        'auc_binary': metrics_bin['AUC'],
        'auc_cont': metrics_cont['AUC'],
        'prauc_binary': metrics_bin['PR-AUC'],
        'prauc_cont': metrics_cont['PR-AUC'],
        'brier_binary': metrics_bin['Brier'],
        'brier_cont': metrics_cont['Brier'],
        'pos_rate': metrics_bin['PosRate'],
        'n_samples': metrics_bin['N'],
        'n_fits': len(fit_times),
        'avg_train_size': np.mean(fit_times) if fit_times else 0,
        'forecasts': y_garch,
        'prob_cont': y_prob_cont,
        'y_actual': y_actual,
        'threshold': threshold_aligned,
    }


def run_asset(cfg: AssetClassConfig):
    log('=' * 60)
    log(f'GARCH Volatility Model: {cfg.name}')
    log(f'  ETFs: {cfg.etfs}')
    if cfg.weights:
        log(f'  Weights: {dict(zip(cfg.etfs, cfg.weights))}')
    log('=' * 60)

    log('\n[1] Loading data...')
    etf_data = load_etf_data()
    nav = compute_asset_nav(etf_data, cfg)
    rets = nav.pct_change().dropna() * 100
    log(f'  Returns: {len(rets)}d, {rets.index[0].date()} ~ {rets.index[-1].date()}')

    log('\n[2] Computing realized metrics...')
    realized_fwd_vol, threshold = compute_realized_metrics(rets / 100, cfg)
    y_true = (realized_fwd_vol > threshold).astype(float).dropna()
    log(f'  y_true: {y_true.mean():.1%} positive ({int(y_true.sum())}/{len(y_true)})')

    results = []
    for ms in MODELS:
        label = ms.label
        log(f'\n[3] Running {label}...')
        t0 = time.time()
        r = run_model(rets, y_true, threshold, cfg, ms)
        elapsed = time.time() - t0
        log(f'  AUC (binary): {r["auc_binary"]:.4f}')
        log(f'  AUC (cont):   {r["auc_cont"]:.4f}')
        log(f'  PR-AUC (cont): {r["prauc_cont"]:.4f}')
        log(f'  Brier (cont): {r["brier_cont"]:.4f}')
        log(f'  Fits: {r["n_fits"]}, Avg train: {r["avg_train_size"]:.0f}d, Time: {elapsed:.0f}s')
        results.append(r)

    # Comparison table
    log(f'\n{"=" * 70}')
    log(f'{"Model":<30} {"AUC(bin)":<10} {"AUC(cont)":<10} {"PR-AUC(cont)":<12} {"Brier(cont)":<10} {"Fits":<6}')
    log(f'{"-" * 70}')
    for r in results:
        log(f'{r["name"]:<30} {r["auc_binary"]:<10.4f} {r["auc_cont"]:<10.4f} '
            f'{r["prauc_cont"]:<12.4f} {r["brier_cont"]:<10.4f} {r["n_fits"]:<6}')
    log(f'{"=" * 70}')

    # Save results
    summary = pd.DataFrame([{
        'asset': cfg.name,
        'model': r['name'],
        'model_key': r['key'],
        'auc_binary': r['auc_binary'],
        'auc_cont': r['auc_cont'],
        'prauc_binary': r['prauc_binary'],
        'prauc_cont': r['prauc_cont'],
        'brier_binary': r['brier_binary'],
        'brier_cont': r['brier_cont'],
        'pos_rate': r['pos_rate'],
        'n_samples': r['n_samples'],
        'n_fits': r['n_fits'],
        'avg_train_size': r['avg_train_size'],
    } for r in results])
    summary.to_csv(OUT_DIR / f'garch_{cfg.name}_comparison.csv', index=False)
    log(f'\nSaved: {OUT_DIR / f"garch_{cfg.name}_comparison.csv"}')

    # Save detailed forecasts for best model
    best = max(results, key=lambda r: r['auc_cont'])
    best_df = pd.DataFrame({
        'y_true': best['y_actual'],
        'forecast_vol': best['forecasts'],
        'risk_prob': best['prob_cont'],
        'threshold': best['threshold'],
    })
    best_df.to_parquet(OUT_DIR / f'garch_{cfg.name}_best_{best["key"]}.parquet')
    best_key = best['key']
    log(f'Saved: {OUT_DIR / f"garch_{cfg.name}_best_{best_key}.parquet"}')

    # Save all forecasts for comparison
    all_df = {}
    for r in results:
        rk = r['key']
        all_df[f'forecast_{rk}'] = r['forecasts']
        all_df[f'prob_{rk}'] = r['prob_cont']
    all_df['y_true'] = results[0]['y_actual']
    all_df['threshold'] = results[0]['threshold']
    pd.DataFrame(all_df).to_parquet(OUT_DIR / f'garch_{cfg.name}_all_forecasts.parquet')
    log(f'Saved: {OUT_DIR / f"garch_{cfg.name}_all_forecasts.parquet"}')

    return results


if __name__ == '__main__':
    log('=' * 60)
    log('GARCH Volatility Forecasting Model v2')
    log('=' * 60)
    for cfg in [EQUITY_CFG, COMMODITY_CFG]:
        run_asset(cfg)
        log('\n')
    log('[Done]')