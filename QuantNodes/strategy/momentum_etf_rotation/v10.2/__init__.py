"""v10.2 — v10 + CA-GCP 独立风控层.

模块组成:
  - ca_gcp/         独立 CA-GCP 包（可单独 import）
  - integration/    Phase B 集成（ca_gcp_risk_filter hook）
  - experiments/    8 个实验脚本（独立可运行）
  - tests/          12 个单元测试

Phase A: 实验验证 CA-GCP 框架
  python -m v10.2.experiments.01_build_graph
  python -m v10.2.experiments.02_run_baselines
  python -m v10.2.experiments.03_coverage_compare
  python -m v10.2.experiments.04_width_compare
  python -m v10.2.experiments.05_crisis_day_compare
  python -m v10.2.experiments.06_early_warning
  python -m v10.2.experiments.07_calibrate  # 慢
  python -m v10.2.experiments.08_v10_2_backtest

详见 docs/82-ca-gcp-pool-size-test.md
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
)
from integration import RiskFilterRules, ca_gcp_risk_filter  # noqa: E402

__version__ = "0.2.0"
__all__ = [
    "CAGCPConfig",
    "CAGCPipeline",
    "RiskFilterRules",
    "ca_gcp_risk_filter",
]