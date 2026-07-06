"""Strategy Library — high-level API for strategy YAML management.

Provides read/write access to strategy definitions stored as YAML files in
quant/strategies/. This is the canonical storage for strategy definitions
(not wiki markdown) and complements the runtime registry at
``QuantNodes.research.backtest.strategies.SIGNAL_NODE_REGISTRY``.

Storage layout (mirrors factor_library.py):

    quant/strategies/
    ├── index.yaml
    └── {strategy_name}/
        ├── strategy.yaml   # 4-layer strategy definition (root key 'strategy')
        ├── code.py         # optional custom StrategyNode subclass source
        ├── meta.json       # optional metadata blob
        └── backtest/
            └── latest.json # last backtest result

Used by:
- run_101_alphas_v2.py (M3.4 integration; auto-persist strategy when --strategy-mode)
- cli/commands/alpha.py (AlphaPipeline outputs feed into write_strategy_yaml)
- backtest/strategies.py (registry of built-in StrategyNode subclasses)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "list_strategies",
    "read_strategy_yaml",
    "write_strategy_yaml",
    "list_strategies_by_signal_type",
    "update_index",
    "get_strategy_node_from_yaml",
    "save_backtest_duckdb",
    "read_backtest_duckdb",
    "strategy_dir",
]


# ─── Path resolution ──────────────────────────────────────────


def _get_strategies_dir(project_root: Path | None = None, strategies_dir: Path | None = None) -> Path:
    """Get the quant/strategies/ directory path.

    Priority: strategies_dir > project_root/quant/strategies > Path.cwd()/quant/strategies
    """
    if strategies_dir is not None:
        return Path(strategies_dir)
    root = project_root or Path.cwd()
    return root / "quant" / "strategies"


def strategy_dir(name: str, project_root: Path | None = None, strategies_dir: Path | None = None) -> Path:
    """Resolve a strategy directory path from name.

    Supports:
    1. Exact match: strategies/{name}/strategy.yaml exists
    2. Fuzzy match: search for *{name}* in subdirectories
    3. Fallback: return strategies/{name}/ (created if missing)
    """
    strategies_root = _get_strategies_dir(project_root, strategies_dir)

    exact = strategies_root / name
    if exact.is_dir():
        return exact

    if strategies_root.exists():
        for subdir in strategies_root.iterdir():
            if not subdir.is_dir():
                continue
            if name in subdir.name:
                return subdir

    exact.mkdir(parents=True, exist_ok=True)
    return exact


# ─── List ──────────────────────────────────────────


def list_strategies(project_root: Path | None = None, strategies_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read quant/strategies/index.yaml and return strategy list.

    Returns:
        List of strategy summary dicts from the index.
    """
    root = _get_strategies_dir(project_root, strategies_dir)
    index_path = root / "index.yaml"

    if not index_path.exists():
        return []

    try:
        content = index_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return data.get("strategies", []) if data else []
    except Exception as exc:
        logger.warning("could not read strategy index.yaml: %s", exc)
        return []


def list_strategies_by_signal_type(
    project_root: Path | None = None, strategies_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """List all strategies grouped by signal_type.

    Returns:
        Dict like {'ma_cross': [...], 'momentum': [...]} keyed by signal_type.
    """
    root = _get_strategies_dir(project_root, strategies_dir)
    if not root.exists():
        return {}

    groups: dict[str, list[dict]] = {}

    for yaml_file in sorted(root.rglob("*/strategy.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if not data:
                continue
            strategy = data.get("strategy", data)
            st = strategy.get("signal_type", "unknown")
            rel = yaml_file.parent.relative_to(root)
            strategy["_name"] = str(rel)
            strategy["_path"] = str(rel / "strategy.yaml")
            groups.setdefault(st, []).append(strategy)
        except Exception as exc:
            logger.warning("could not read %s: %s", yaml_file, exc)

    return groups


# ─── Read ──────────────────────────────────────────


def read_strategy_yaml(
    name: str, project_root: Path | None = None, strategies_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Read a single strategy YAML file.

    Args:
        name: Strategy path relative to strategies/ (e.g. 'ma_cross_001')

    Returns:
        Full strategy dict, or None if not found.
    """
    dir_path = strategy_dir(name, project_root, strategies_dir)
    yaml_path = dir_path / "strategy.yaml"

    if not yaml_path.exists():
        return None

    try:
        content = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if data is None:
            return None

        code_path = dir_path / "code.py"
        if code_path.exists():
            data["code"] = code_path.read_text(encoding="utf-8")

        backtest_path = dir_path / "backtest" / "latest.json"
        if backtest_path.exists():
            data["backtest"] = json.loads(backtest_path.read_text(encoding="utf-8"))

        meta_path = dir_path / "meta.json"
        if meta_path.exists():
            data["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

        return data
    except Exception as exc:
        logger.warning("could not read strategy YAML %s: %s", name, exc)
        return None


# ─── Write ──────────────────────────────────────────


def write_strategy_yaml(
    name: str, data: dict, project_root: Path | None = None, strategies_dir: Path | None = None,
) -> str:
    """Write a strategy YAML file and update index.yaml.

    Args:
        name: Strategy path relative to strategies/
        data: Full strategy dict (with 'strategy' root key)

    Returns:
        "Created: strategies/{name}" or "Updated: strategies/{name}"
    """
    dir_path = strategy_dir(name, project_root, strategies_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    yaml_path = dir_path / "strategy.yaml"
    is_new = not yaml_path.exists()

    content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    yaml_path.write_text(content, encoding="utf-8")

    if "code" in data:
        (dir_path / "code.py").write_text(data["code"], encoding="utf-8")

    if "meta" in data:
        (dir_path / "meta.json").write_text(
            json.dumps(data["meta"], indent=2, ensure_ascii=False), encoding="utf-8",
        )

    if "backtest" in data:
        backtest_dir = dir_path / "backtest"
        backtest_dir.mkdir(exist_ok=True)
        (backtest_dir / "latest.json").write_text(
            json.dumps(data["backtest"], indent=2, ensure_ascii=False), encoding="utf-8",
        )

    try:
        update_index(project_root, strategies_dir)
    except Exception as exc:
        logger.warning("strategy index.yaml update failed after writing %s: %s", name, exc)

    action = "Created" if is_new else "Updated"
    return f"{action}: strategies/{name}"


# ─── Index ──────────────────────────────────────────


def update_index(project_root: Path | None = None, strategies_dir: Path | None = None) -> None:
    """Regenerate quant/strategies/index.yaml from actual YAML files.

    Scans all strategy YAML files and rebuilds the index, mirroring
    factor_library.update_index structure (statistics + entries).
    """
    root = _get_strategies_dir(project_root, strategies_dir)
    if not root.exists():
        return

    strategies: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "by_asset_type": {},
        "by_signal_type": {},
        "by_status": {},
    }

    for yaml_file in sorted(root.rglob("*/strategy.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            if not data:
                continue
            strategy = data.get("strategy", data)
            rel = yaml_file.parent.relative_to(root)

            entry = {
                "name": strategy.get("name", ""),
                "name_cn": strategy.get("name_cn", ""),
                "asset_type": strategy.get("asset_type", ""),
                "signal_type": strategy.get("signal_type", ""),
                "status": strategy.get("status", "已注册"),
                "definition": strategy.get("l1", {}).get("description", ""),
                "file": str(rel),
            }
            strategies.append(entry)

            stats["total"] += 1
            at = entry["asset_type"] or "unknown"
            stats["by_asset_type"][at] = stats["by_asset_type"].get(at, 0) + 1
            st = entry["signal_type"] or "unknown"
            stats["by_signal_type"][st] = stats["by_signal_type"].get(st, 0) + 1
            stt = entry["status"] or "unknown"
            stats["by_status"][stt] = stats["by_status"].get(stt, 0) + 1
        except Exception as exc:
            logger.warning("could not read %s: %s", yaml_file, exc)

    index = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": stats,
        "strategies": strategies,
    }

    index_path = root / "index.yaml"
    index_path.write_text(
        yaml.dump(index, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("strategy index.yaml updated with %d strategies", len(strategies))


# ─── Registry bridge ──────────────────────────────────────────


def get_strategy_node_from_yaml(
    name: str, project_root: Path | None = None, strategies_dir: Path | None = None,
) -> Any:
    """Load a YAML strategy and instantiate the corresponding StrategyNode.

    Falls back to SIGNAL_NODE_REGISTRY (built-in nodes). Custom subclasses
    in code.py are loaded via importlib if signal_type isn't in the
    built-in registry.

    Returns:
        A StrategyNode instance ready to plug into run_backtest().
    """
    from ..backtest.strategies import SIGNAL_NODE_REGISTRY
    from QuantNodes.backtest.strategy_node import StrategyNode

    data = read_strategy_yaml(name, project_root, strategies_dir)
    if data is None:
        raise FileNotFoundError(f"strategy YAML not found: {name}")

    strategy = data.get("strategy", data)
    signal_type = strategy.get("signal_type")
    if signal_type is None:
        raise ValueError(f"strategy {name} missing 'signal_type' field")

    config = dict(strategy.get("l1", {}).get("signal_params") or {})
    config.update(strategy.get("l1", {}).get("config") or {})

    if signal_type in SIGNAL_NODE_REGISTRY:
        cls = SIGNAL_NODE_REGISTRY[signal_type]
        return cls(name=signal_type, config=config)

    custom_code = data.get("code")
    if custom_code:
        import sys
        import types
        import uuid

        module_name = f"_custom_strategy_{uuid.uuid4().hex[:8]}"
        module = types.ModuleType(module_name)
        exec(compile(custom_code, f"<strategy:{name}>", "exec"), module.__dict__)
        sys.modules[module_name] = module

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not (isinstance(attr, type) and attr_name.endswith("StrategyNode")):
                continue
            if attr is StrategyNode:
                continue
            if not issubclass(attr, StrategyNode):
                continue
            return attr(name=attr_name, config=config)

        raise ValueError(
            f"custom code in strategy {name} defines no *StrategyNode subclass",
        )

    raise ValueError(
        f"unknown signal_type {signal_type!r}; not in SIGNAL_NODE_REGISTRY and no code.py",
    )


# ─── DuckDB backtest storage ──────────────────────────────────────────


def _nan(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None


def save_backtest_duckdb(
    strategy_name: str,
    run_id: str,
    backtest: dict,
    project_root: Path | None = None,
    strategies_dir: Path | None = None,
) -> Path:
    """Write backtest to strategy's strategy.duckdb.

    Schema is similar to factor_library.save_backtest_duckdb but without
    factor_values (strategies don't store per-date per-stock factor values).

    Args:
        strategy_name: slug or full path
        run_id: unique run identifier
        backtest: dict with sharpe_ratio, max_drawdown, win_rate, total_return,
                 final_cash, trades, ic_series, equity_curve, status, created_at.
    """
    import duckdb

    dir_path = strategy_dir(strategy_name, project_root, strategies_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    db_path = dir_path / "strategy.duckdb"

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                status VARCHAR,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                win_rate DOUBLE,
                total_return DOUBLE,
                final_cash DOUBLE,
                n_trades INTEGER,
                signal_type VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                run_id VARCHAR,
                date BIGINT,
                nav DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                run_id VARCHAR,
                trade_idx INTEGER,
                code VARCHAR,
                side VARCHAR,
                price DOUBLE,
                size DOUBLE,
                trade_date VARCHAR
            )
        """)

        strategy_meta = read_strategy_yaml(strategy_name, project_root, strategies_dir) or {}
        signal_type = strategy_meta.get("strategy", {}).get("signal_type", "")

        conn.execute("""
            INSERT OR REPLACE INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            backtest.get("created_at") or __import__("datetime").datetime.now().isoformat(),
            backtest.get("status", "success"),
            _nan(backtest.get("sharpe_ratio")),
            _nan(backtest.get("max_drawdown")),
            _nan(backtest.get("win_rate")),
            _nan(backtest.get("total_return")),
            _nan(backtest.get("final_cash")),
            int(backtest.get("n_trades") or 0),
            signal_type,
        ])

        equity = backtest.get("equity_curve") or []
        if equity:
            import pandas as pd
            eq_df = pd.DataFrame(equity, columns=["date", "nav"])
            eq_df.insert(0, "run_id", run_id)
            conn.register("_eq_df", eq_df)
            conn.execute("INSERT INTO equity_curve SELECT * FROM _eq_df")
            conn.unregister("_eq_df")

        trades = backtest.get("trades") or []
        if trades:
            import pandas as pd
            tdf = pd.DataFrame(trades)
            tdf.insert(0, "run_id", run_id)
            conn.register("_t_df", tdf)
            conn.execute("INSERT INTO trades SELECT * FROM _t_df")
            conn.unregister("_t_df")

        logger.info("saved strategy backtest to %s (run_id=%s)", db_path, run_id)
    finally:
        conn.close()

    return db_path


def read_backtest_duckdb(
    strategy_name: str,
    limit: int = 10,
    project_root: Path | None = None,
    strategies_dir: Path | None = None,
) -> list[dict]:
    """Read backtest runs from strategy's DuckDB.

    Returns:
        List of run dicts with metrics + equity_curve + trades.
        NULL DOUBLE columns are returned as None (not NaN).
    """
    import math

    import duckdb
    import pandas as pd

    def _clean(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    dir_path = strategy_dir(strategy_name, project_root, strategies_dir)
    db_path = dir_path / "strategy.duckdb"
    if not db_path.exists():
        return []

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        if "backtest_runs" not in tables:
            return []

        runs_df = conn.execute(
            "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", [limit],
        ).fetchdf()

        results = []
        for _, row in runs_df.iterrows():
            rid = row["run_id"]

            equity: list = []
            if "equity_curve" in tables:
                eq_df = conn.execute(
                    "SELECT date, nav FROM equity_curve WHERE run_id = ? ORDER BY date", [rid],
                ).fetchdf()
                equity = [
                    {"date": str(r["date"]), "nav": _clean(r["nav"])}
                    for _, r in eq_df.iterrows()
                ]

            trades: list = []
            if "trades" in tables:
                t_df = conn.execute(
                    "SELECT * FROM trades WHERE run_id = ? ORDER BY trade_idx", [rid],
                ).fetchdf()
                trades = [
                    {k: (_clean(v) if isinstance(v, float) else v) for k, v in r.items()}
                    for _, r in t_df.iterrows()
                ]

            results.append({
                "run_id": rid,
                "created_at": str(row.get("created_at", "")),
                "status": row.get("status", ""),
                "metrics": {
                    "sharpe_ratio": _clean(row.get("sharpe_ratio")),
                    "max_drawdown": _clean(row.get("max_drawdown")),
                    "win_rate": _clean(row.get("win_rate")),
                    "total_return": _clean(row.get("total_return")),
                    "final_cash": _clean(row.get("final_cash")),
                    "n_trades": int(row.get("n_trades") or 0),
                },
                "signal_type": row.get("signal_type", ""),
                "equity_curve": equity,
                "trades": trades,
            })

        return results
    finally:
        conn.close()