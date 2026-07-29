"""v10.2 — v10 + CA-GCP 独立风控层.

模块组成:
  - ca_gcp/         独立 CA-GCP 包（可单独 import）
  - integration/    Phase B 集成（ca_gcp_risk_filter hook）
  - experiments/    8 个实验脚本（独立可运行）
  - tests/          29 个单元测试

Phase A: 实验验证 CA-GCP 框架
  python -m v10.2.experiments.01_build_graph
  python -m v10.2.experiments.02_run_baselines
  python -m v10.2.experiments.03_coverage_compare
  python -m v10.2.experiments.04_width_compare
  python -m v10.2.experiments.05_crisis_day_compare
  python -m v10.2.experiments.06_early_warning
  python -m v10.2.experiments.07_calibrate  # 343 grid + 早停
  python -m v10.2.experiments.08_v10_2_backtest

详见 docs/82-ca-gcp-pool-size-test.md
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
)
from integration import RiskFilterRules, ca_gcp_risk_filter  # noqa: E402

__version__ = "0.2.0"


def load_calibrated_config() -> CAGCPConfig:
    """Load the calibrated CAGCPConfig from data/results/best_params.json.

    Falls back to paper defaults if the file is missing (e.g. before
    running experiments/07_calibrate.py).
    """
    best_path = _HERE / "data" / "results" / "best_params.json"
    if not best_path.exists():
        return CAGCPConfig()

    with best_path.open() as f:
        params = json.load(f)
    return CAGCPConfig(
        k=int(params.get("k", 8)),
        sensitivity_eta=float(params.get("eta", 0.5)),
        recency_tau=float(params.get("tau", 60.0)),
    )


DEFAULT_CONFIG = load_calibrated_config()


__all__ = [
    "CAGCPConfig",
    "CAGCPipeline",
    "RiskFilterRules",
    "ca_gcp_risk_filter",
    "load_calibrated_config",
    "DEFAULT_CONFIG",
]