"""5-gates integrated check (L-321).

Gates:
  1. data: OHLCV / dynamic pool / gap audit
  2. IC: |IC| > 0.05 + rolling stability + dedup
  3. factor: cross-section vs time-series IC + no symmetry
  4. OOS: 5-fold + start-dependency CV% + expanding + walk-forward
  5. hardened: dead code + docs + factory
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class FiveGatesResult:
    """Result of 5-gates check."""

    gates: dict = field(default_factory=dict)
    overall_pass: bool = False
    n_passed: int = 0
    n_failed: int = 0

    def to_dict(self) -> dict:
        return {
            "gates": self.gates,
            "overall_pass": self.overall_pass,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
        }


class FiveGates:
    """5-gates integrated check.

    The default run() does a structural check (which gates can be evaluated).
    For actual gate verification, pass custom check functions.
    """

    def __init__(self):
        self.gate_definitions = {
            "data": "OHLCV 前复权 / 动态资产池 / 缺数据审计",
            "ic": "|IC| > 0.05 + 滚动 IC 稳定 + 因子去重",
            "factor": "宏观时序 vs PV 截面 + 禁 Symmetry 正交化",
            "oos": "5-fold + 起点依赖 CV% + expanding + walk-forward",
            "hardened": "起点依赖测试 + 死代码清理 + 文档 + 工厂函数",
        }

    def run(
        self,
        backtest_fn: Callable | None = None,
        data_path: Path | None = None,
        ic_report: Path | None = None,
        check_functions: dict[str, Callable[[], bool]] | None = None,
    ) -> FiveGatesResult:
        """Run 5-gates check.

        Args:
            backtest_fn: Optional backtest function for OOS gate
            data_path: Optional data path for data gate
            ic_report: Optional IC report CSV for IC gate
            check_functions: Custom check functions per gate

        Returns:
            FiveGatesResult
        """
        gates: dict = {}
        check_fns = check_functions or {}

        for gate_name, description in self.gate_definitions.items():
            check_fn = check_fns.get(gate_name)
            if check_fn is None:
                gates[gate_name] = {
                    "description": description,
                    "status": "MANUAL",
                    "note": "no check function provided",
                }
            else:
                try:
                    passed = bool(check_fn())
                    gates[gate_name] = {
                        "description": description,
                        "status": "PASS" if passed else "FAIL",
                    }
                except Exception as e:
                    gates[gate_name] = {
                        "description": description,
                        "status": "ERROR",
                        "error": str(e),
                    }

        n_passed = sum(1 for g in gates.values() if g.get("status") == "PASS")
        n_failed = sum(1 for g in gates.values() if g.get("status") == "FAIL")
        all_evaluated = all(
            g.get("status") in ("PASS", "FAIL", "MANUAL") for g in gates.values()
        )
        no_failures = n_failed == 0
        overall_pass = all_evaluated and no_failures

        return FiveGatesResult(
            gates=gates,
            overall_pass=overall_pass,
            n_passed=n_passed,
            n_failed=n_failed,
        )
