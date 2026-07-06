"""Tests for strategy_library: read/write/list/index/registry bridge.

覆盖:
  - read_strategy_yaml: 正常 / 缺失 / 无效 YAML / 附加文件 (code.py, backtest, meta)
  - write_strategy_yaml: 创建 / 更新 / 原子性 / 附加文件写入
  - list_strategies: 空 / 缺失 / 正常
  - list_strategies_by_signal_type: 多类别 / 空目录
  - update_index: 全扫描 / stats 准确
  - get_strategy_node_from_yaml: built-in registry / missing file / unknown signal_type
  - save/read_backtest_duckdb: roundtrip
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from QuantNodes.research.persist import strategy_library as sl


# ── 公共 fixture ────────────────────────────────────────────


@pytest.fixture
def strategy_workspace(tmp_path: Path) -> Path:
    """创建空 workspace: {tmp_path}/quant/strategies/."""
    strategies_dir = tmp_path / "quant" / "strategies"
    strategies_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def strategy_data() -> dict[str, Any]:
    """标准 4-layer strategy 测试数据."""
    return {
        "strategy": {
            "name": "ma_cross_test_001",
            "name_cn": "测试双均线策略",
            "signal_type": "ma_cross",
            "asset_type": "stk",
            "status": "已注册",
            "l1": {
                "description": "Test MA cross strategy for unit testing",
                "signal_params": {"fast": 5, "slow": 20},
            },
            "l2": {"factors": ["alpha_001", "alpha_005"]},
            "l3": {},
            "l4": {"created_by": "test"},
        },
    }


# ── read_strategy_yaml ──────────────────────────────────────


class TestReadStrategyYaml:
    def test_basic(self, strategy_workspace: Path, strategy_data: dict) -> None:
        """基本: strategies/{name}/strategy.yaml 可读."""
        import yaml
        sdir = strategy_workspace / "quant" / "strategies" / "ma_cross_test_001"
        sdir.mkdir()
        (sdir / "strategy.yaml").write_text(yaml.dump(strategy_data), encoding="utf-8")

        result = sl.read_strategy_yaml("ma_cross_test_001", project_root=strategy_workspace)
        assert result is not None
        assert result["strategy"]["name"] == "ma_cross_test_001"
        assert result["strategy"]["signal_type"] == "ma_cross"

    def test_missing_returns_none(self, strategy_workspace: Path) -> None:
        result = sl.read_strategy_yaml("nonexistent", project_root=strategy_workspace)
        assert result is None

    def test_invalid_yaml_returns_none(self, strategy_workspace: Path) -> None:
        bad = strategy_workspace / "quant" / "strategies" / "bad_strategy"
        bad.mkdir()
        (bad / "strategy.yaml").write_text(":\n  - [unclosed", encoding="utf-8")
        result = sl.read_strategy_yaml("bad_strategy", project_root=strategy_workspace)
        assert result is None

    def test_loads_code_py(self, strategy_workspace: Path, strategy_data: dict) -> None:
        """附加文件: code.py 内容附加到 data['code']."""
        import yaml
        sdir = strategy_workspace / "quant" / "strategies" / "with_code"
        sdir.mkdir()
        (sdir / "strategy.yaml").write_text(yaml.dump(strategy_data), encoding="utf-8")
        (sdir / "code.py").write_text("class CustomStrategyNode:\n    pass\n", encoding="utf-8")

        result = sl.read_strategy_yaml("with_code", project_root=strategy_workspace)
        assert "code" in result
        assert "CustomStrategyNode" in result["code"]

    def test_loads_backtest_latest(self, strategy_workspace: Path, strategy_data: dict) -> None:
        """附加文件: backtest/latest.json 附加到 data['backtest']."""
        import yaml
        sdir = strategy_workspace / "quant" / "strategies" / "with_bt"
        sdir.mkdir()
        (sdir / "strategy.yaml").write_text(yaml.dump(strategy_data), encoding="utf-8")
        bt_dir = sdir / "backtest"
        bt_dir.mkdir()
        (bt_dir / "latest.json").write_text(
            json.dumps({"sharpe_ratio": 1.5, "status": "success"}), encoding="utf-8",
        )

        result = sl.read_strategy_yaml("with_bt", project_root=strategy_workspace)
        assert "backtest" in result
        assert result["backtest"]["sharpe_ratio"] == 1.5

    def test_loads_meta(self, strategy_workspace: Path, strategy_data: dict) -> None:
        """附加文件: meta.json 附加到 data['meta']."""
        import yaml
        sdir = strategy_workspace / "quant" / "strategies" / "with_meta"
        sdir.mkdir()
        (sdir / "strategy.yaml").write_text(yaml.dump(strategy_data), encoding="utf-8")
        (sdir / "meta.json").write_text(
            json.dumps({"verified": True, "tags": ["test"]}), encoding="utf-8",
        )

        result = sl.read_strategy_yaml("with_meta", project_root=strategy_workspace)
        assert result["meta"]["verified"] is True

    def test_handles_unicode(self, strategy_workspace: Path) -> None:
        import yaml
        data = {"strategy": {"name": "中文策略名", "signal_type": "ma_cross"}}
        sdir = strategy_workspace / "quant" / "strategies" / "unicode_strategy"
        sdir.mkdir()
        (sdir / "strategy.yaml").write_text(
            yaml.dump(data, allow_unicode=True), encoding="utf-8",
        )

        result = sl.read_strategy_yaml("unicode_strategy", project_root=strategy_workspace)
        assert result["strategy"]["name"] == "中文策略名"


# ── write_strategy_yaml ──────────────────────────────────────


class TestWriteStrategyYaml:
    def test_creates_new(self, strategy_workspace: Path, strategy_data: dict) -> None:
        result = sl.write_strategy_yaml(
            "new_strategy_001", strategy_data, project_root=strategy_workspace,
        )
        assert "Created" in result
        sdir = strategy_workspace / "quant" / "strategies" / "new_strategy_001"
        assert (sdir / "strategy.yaml").exists()

    def test_updates_existing(self, strategy_workspace: Path, strategy_data: dict) -> None:
        sl.write_strategy_yaml("update_test", strategy_data, project_root=strategy_workspace)
        result = sl.write_strategy_yaml("update_test", strategy_data, project_root=strategy_workspace)
        assert "Updated" in result

    def test_writes_code_py(self, strategy_workspace: Path, strategy_data: dict) -> None:
        strategy_data["code"] = "class MyStrategyNode:\n    pass\n"
        sl.write_strategy_yaml("with_code", strategy_data, project_root=strategy_workspace)
        code_path = (
            strategy_workspace / "quant" / "strategies" / "with_code" / "code.py"
        )
        assert code_path.exists()
        assert "MyStrategyNode" in code_path.read_text(encoding="utf-8")

    def test_writes_meta(self, strategy_workspace: Path, strategy_data: dict) -> None:
        strategy_data["meta"] = {"verified": True}
        sl.write_strategy_yaml("with_meta", strategy_data, project_root=strategy_workspace)
        meta_path = (
            strategy_workspace / "quant" / "strategies" / "with_meta" / "meta.json"
        )
        assert meta_path.exists()

    def test_writes_backtest(self, strategy_workspace: Path, strategy_data: dict) -> None:
        strategy_data["backtest"] = {"sharpe_ratio": 1.2, "status": "success"}
        sl.write_strategy_yaml("with_bt", strategy_data, project_root=strategy_workspace)
        bt_path = (
            strategy_workspace / "quant" / "strategies" / "with_bt"
            / "backtest" / "latest.json"
        )
        assert bt_path.exists()
        data = json.loads(bt_path.read_text(encoding="utf-8"))
        assert data["sharpe_ratio"] == 1.2


# ── list_strategies / list_strategies_by_signal_type ──────


class TestListStrategies:
    def test_empty_when_no_index(self, strategy_workspace: Path) -> None:
        result = sl.list_strategies(project_root=strategy_workspace)
        assert result == []

    def test_reads_index(self, strategy_workspace: Path) -> None:
        import yaml
        index_path = strategy_workspace / "quant" / "strategies" / "index.yaml"
        index_path.write_text(
            yaml.dump({
                "strategies": [
                    {"name": "s1", "signal_type": "ma_cross"},
                    {"name": "s2", "signal_type": "momentum"},
                ],
            }),
            encoding="utf-8",
        )
        result = sl.list_strategies(project_root=strategy_workspace)
        assert len(result) == 2
        assert result[0]["name"] == "s1"

    def test_by_signal_type_empty(self, strategy_workspace: Path) -> None:
        result = sl.list_strategies_by_signal_type(project_root=strategy_workspace)
        assert result == {}

    def test_by_signal_type_groups(self, strategy_workspace: Path, strategy_data: dict) -> None:
        import yaml
        root = strategy_workspace / "quant" / "strategies"
        for name, st in [("macd_1", "ma_cross"), ("macd_2", "ma_cross"), ("mom_1", "momentum")]:
            data = {
                "strategy": {
                    "name": name,
                    "signal_type": st,
                    "status": "已注册",
                    "asset_type": "stk",
                    "l1": {"description": "test"},
                },
            }
            (root / name).mkdir()
            (root / name / "strategy.yaml").write_text(yaml.dump(data), encoding="utf-8")

        result = sl.list_strategies_by_signal_type(project_root=strategy_workspace)
        assert "ma_cross" in result
        assert "momentum" in result
        assert len(result["ma_cross"]) == 2
        assert len(result["momentum"]) == 1


# ── update_index ────────────────────────────────────────────


class TestUpdateIndex:
    def test_writes_index_yaml(self, strategy_workspace: Path, strategy_data: dict) -> None:
        import yaml
        root = strategy_workspace / "quant" / "strategies"
        (root / "alpha_001").mkdir()
        (root / "alpha_001" / "strategy.yaml").write_text(
            yaml.dump(strategy_data), encoding="utf-8",
        )

        sl.update_index(project_root=strategy_workspace)
        index_path = root / "index.yaml"
        assert index_path.exists()
        data = yaml.safe_load(index_path.read_text(encoding="utf-8"))
        assert data["statistics"]["total"] == 1
        assert data["statistics"]["by_signal_type"]["ma_cross"] == 1

    def test_noop_when_missing(self, strategy_workspace: Path) -> None:
        # strategies/ exists but empty
        sl.update_index(project_root=strategy_workspace)
        # Should not raise; index should NOT be created if no strategy.yaml files
        root = strategy_workspace / "quant" / "strategies"
        index_path = root / "index.yaml"
        # Update_index writes index only if it finds strategy.yaml files
        # With none, statistics.total == 0, but index.yaml IS still written (empty list)
        assert index_path.exists()
        import yaml
        data = yaml.safe_load(index_path.read_text(encoding="utf-8"))
        assert data["statistics"]["total"] == 0


# ── get_strategy_node_from_yaml ────────────────────────────────


class TestGetStrategyNodeFromYaml:
    def test_builtin_registry(
        self, strategy_workspace: Path, strategy_data: dict,
    ) -> None:
        """Built-in signal_type → StrategyNode instance."""
        from QuantNodes.research.backtest.strategies import MACrossStrategyNode
        sl.write_strategy_yaml("macd_test", strategy_data, project_root=strategy_workspace)

        node = sl.get_strategy_node_from_yaml(
            "macd_test", project_root=strategy_workspace,
        )
        assert isinstance(node, MACrossStrategyNode)
        assert node._fast == 5
        assert node._slow == 20

    def test_missing_file(self, strategy_workspace: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sl.get_strategy_node_from_yaml(
                "nonexistent", project_root=strategy_workspace,
            )

    def test_missing_signal_type(
        self, strategy_workspace: Path, strategy_data: dict,
    ) -> None:
        del strategy_data["strategy"]["signal_type"]
        sl.write_strategy_yaml("no_st", strategy_data, project_root=strategy_workspace)
        with pytest.raises(ValueError, match="signal_type"):
            sl.get_strategy_node_from_yaml(
                "no_st", project_root=strategy_workspace,
            )

    def test_unknown_signal_type_no_code(
        self, strategy_workspace: Path, strategy_data: dict,
    ) -> None:
        strategy_data["strategy"]["signal_type"] = "nonexistent_signal"
        sl.write_strategy_yaml("bad_st", strategy_data, project_root=strategy_workspace)
        with pytest.raises(ValueError, match="nonexistent_signal"):
            sl.get_strategy_node_from_yaml(
                "bad_st", project_root=strategy_workspace,
            )

    def test_custom_code_loads(
        self, strategy_workspace: Path, strategy_data: dict,
    ) -> None:
        strategy_data["strategy"]["signal_type"] = "totally_custom"
        strategy_data["code"] = (
            "import pandas as pd\n"
            "from QuantNodes.backtest.strategy_node import StrategyNode, Signal\n"
            "from typing import List\n"
            "class TotallyCustomStrategyNode(StrategyNode):\n"
            "    def _generate_signals(self, input_data: pd.DataFrame, **kwargs) -> List[Signal]:\n"
            "        return []\n"
        )
        sl.write_strategy_yaml("custom", strategy_data, project_root=strategy_workspace)
        node = sl.get_strategy_node_from_yaml(
            "custom", project_root=strategy_workspace,
        )
        assert type(node).__name__ == "TotallyCustomStrategyNode"


# ── save/read_backtest_duckdb ────────────────────────────────


class TestBacktestDuckDB:
    def test_roundtrip(self, strategy_workspace: Path, strategy_data: dict) -> None:
        sl.write_strategy_yaml("macd_bt", strategy_data, project_root=strategy_workspace)

        bt = {
            "status": "success",
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.1,
            "win_rate": 0.55,
            "total_return": 0.25,
            "final_cash": 125000.0,
            "n_trades": 42,
            "equity_curve": [
                {"date": "20240101", "nav": 1.0},
                {"date": "20240102", "nav": 1.01},
            ],
            "trades": [
                {"trade_idx": 0, "code": "600000.SH", "side": "buy",
                 "price": 10.0, "size": 100.0, "trade_date": "20240101"},
            ],
        }

        db_path = sl.save_backtest_duckdb(
            "macd_bt", "run_001", bt, project_root=strategy_workspace,
        )
        assert db_path.exists()

        runs = sl.read_backtest_duckdb(
            "macd_bt", project_root=strategy_workspace,
        )
        assert len(runs) == 1
        run = runs[0]
        assert run["metrics"]["sharpe_ratio"] == pytest.approx(1.5)
        assert run["metrics"]["n_trades"] == 42
        assert run["signal_type"] == "ma_cross"
        assert len(run["equity_curve"]) == 2
        assert len(run["trades"]) == 1

    def test_nan_handling(self, strategy_workspace: Path, strategy_data: dict) -> None:
        sl.write_strategy_yaml("nan_test", strategy_data, project_root=strategy_workspace)

        bt = {
            "status": "failed",
            "sharpe_ratio": float("nan"),
            "max_drawdown": None,
            "win_rate": 0.0,
            "n_trades": 0,
        }
        sl.save_backtest_duckdb(
            "nan_test", "run_nan", bt, project_root=strategy_workspace,
        )
        runs = sl.read_backtest_duckdb(
            "nan_test", project_root=strategy_workspace,
        )
        assert runs[0]["metrics"]["sharpe_ratio"] is None
        assert runs[0]["metrics"]["max_drawdown"] is None

    def test_empty_db(self, strategy_workspace: Path) -> None:
        runs = sl.read_backtest_duckdb(
            "missing", project_root=strategy_workspace,
        )
        assert runs == []


# ── strategy_dir ────────────────────────────────────────────


class TestStrategyDir:
    def test_creates_if_missing(self, strategy_workspace: Path) -> None:
        d = sl.strategy_dir("new_strat", project_root=strategy_workspace)
        assert d.exists()
        assert d.name == "new_strat"

    def test_returns_existing(self, strategy_workspace: Path) -> None:
        existing = strategy_workspace / "quant" / "strategies" / "exists"
        existing.mkdir()
        d = sl.strategy_dir("exists", project_root=strategy_workspace)
        assert d == existing

    def test_explicit_dir_overrides(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_strategies"
        custom.mkdir()
        d = sl.strategy_dir("x", strategies_dir=custom)
        assert d == custom / "x"
        assert d.exists()