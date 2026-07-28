"""Helper to ensure ca_gcp package is importable in experiments."""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
_V102 = _PKG_ROOT.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))