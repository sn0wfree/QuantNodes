"""Risk Warning System — Ensemble Volatility Forecasting + Model Monitor + Self-Healing.

Module structure:
  ensemble.py   — EnsembleRiskWarning (5-layer ensemble), layer models, configs
  monitor.py    — ModelMonitor, SelfEvaluator, ALERT_RULES
  auto.py       — AutoRiskWarning, AutoOptimizer, VersionManager
  calibrate.py  — PlattCalibrator
  scheduler.py  — RiskScheduler
  llm.py        — LLMInterpreter
  drawdown.py   — Grossman-Zhou drawdown controller (from common)
"""

from .ensemble import (
    AssetConfig, CONFIGS, EnsembleRiskWarning, LAYER_CONFIGS, MODEL_FNS,
    MACRO_MODELS, OOS_START, OOS_END, OUT_DIR,
    _model_harrv, _model_climatology, _model_ewma, _model_current_vol,
    _model_macro_equity, _model_macro_commodity, _model_macro_gold,
    _jump_overlay, _cont_prob, compute_nav, compute_y_true, evaluate,
    load_etf_data,
)
from .monitor import ModelMonitor, SelfEvaluator, ALERT_RULES
from .auto import AutoRiskWarning, AutoOptimizer, VersionManager
from .calibrate import PlattCalibrator
from .scheduler import RiskScheduler
from .llm import LLMInterpreter

__all__ = [
    "AssetConfig", "CONFIGS", "EnsembleRiskWarning", "LAYER_CONFIGS",
    "MODEL_FNS", "MACRO_MODELS", "OOS_START", "OOS_END", "OUT_DIR",
    "ModelMonitor", "SelfEvaluator", "ALERT_RULES",
    "AutoRiskWarning", "AutoOptimizer", "VersionManager",
    "PlattCalibrator", "RiskScheduler", "LLMInterpreter",
    "load_etf_data", "compute_nav", "compute_y_true", "evaluate",
    "_model_harrv", "_model_climatology", "_model_ewma", "_model_current_vol",
    "_model_macro_equity", "_model_macro_commodity", "_model_macro_gold",
    "_jump_overlay", "_cont_prob",
]