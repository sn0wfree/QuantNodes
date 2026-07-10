# v7.0 — 宏观状态驱动的 TAA 资产配置 / 风格轮动 (Stage 30+)
from .factor_macro import (
    fetch_macro_factor,
    fetch_all_macro,
    get_pit_value,
    get_pit_series,
    META,
    RELEASE_LAG_DAYS,
    CACHE_DIR,
)
from .regime_macro import (
    REGIME_NAMES,
    REGIME_VOL_TARGETS,
    build_regime_timeline,
    train_5state_hmm,
    predict_5state,
)
from .dynamic_allocation import (
    compute_state_conditional_means,
    compute_dynamic_topk_weights,
    run_topk_v7_backtest,
)
from .black_litterman import (
    compute_expanding_cov,
    compute_state_view_q,
    compute_view_uncertainty_omega,
    compute_bl_posterior,
    compute_bl_weights,
    run_bl_v7_backtest,
)
from .macro_beta import (
    compute_etf_macro_betas,
    predict_etf_returns,
    run_beta_v7_backtest,
)
from .state_momentum import (
    compute_etf_momentum,
    compute_state_conditional_momentum,
    run_momentum_v7_backtest,
)
from .state_inverse_vol import (
    compute_etf_vol,
    compute_inverse_vol_weights,
    run_iv_v7_backtest,
)
from .transaction_cost import (
    compute_turnover,
    apply_turnover_cost,
    portfolio_drag,
)
from .liquidity_cap import (
    apply_max_weight_cap,
    apply_turnover_cap,
)
from .state_allocation import STATE_ALLOCATIONS, ETF_LAUNCH_DATES
from .taa_backtest import (
    V7Config,
    run_v7_taa_backtest,
    state_history_to_df,
)

__all__ = [
    "fetch_macro_factor",
    "fetch_all_macro",
    "get_pit_value",
    "get_pit_series",
    "META",
    "RELEASE_LAG_DAYS",
    "CACHE_DIR",
    "REGIME_NAMES",
    "REGIME_VOL_TARGETS",
    "build_regime_timeline",
    "train_5state_hmm",
    "predict_5state",
    "compute_state_conditional_means",
    "compute_dynamic_topk_weights",
    "run_topk_v7_backtest",
    "compute_expanding_cov",
    "compute_state_view_q",
    "compute_view_uncertainty_omega",
    "compute_bl_posterior",
    "compute_bl_weights",
    "run_bl_v7_backtest",
    "compute_etf_macro_betas",
    "predict_etf_returns",
    "run_beta_v7_backtest",
    "compute_etf_momentum",
    "compute_state_conditional_momentum",
    "run_momentum_v7_backtest",
    "compute_etf_vol",
    "compute_inverse_vol_weights",
    "run_iv_v7_backtest",
    "compute_turnover",
    "apply_turnover_cost",
    "portfolio_drag",
    "apply_max_weight_cap",
    "apply_turnover_cap",
    "STATE_ALLOCATIONS",
    "ETF_LAUNCH_DATES",
    "V7Config",
    "run_v7_taa_backtest",
    "state_history_to_df",
]
