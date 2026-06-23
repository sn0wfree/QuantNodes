"""
Backtest Service - Bridge between FastAPI and ConfigBacktestTool.

v3.0.0 TODO: decouple from ``QuantNodes.agent.tools`` by writing a
``QuantNodes.backtest.api_adapter.BacktestApiAdapter`` that converts
(config_yaml str) → (StrategyConfig + polars.LazyFrame) and calls
``ConfigBacktestRunner.run()`` directly. For now we still use
``ConfigBacktestTool`` (which is kept under ``agent/tools/``).
"""

import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path


class BacktestService:
    """Backtest service for API layer"""

    def __init__(self, data_dir: str = ".quant_agent"):
        self.data_dir = data_dir
        self._backtest_tool = None
        self._results: Dict[str, Dict[str, Any]] = {}
        self._history_file = Path(data_dir) / "backtest_history.jsonl"

    def _get_backtest_tool(self):
        """Get or create ConfigBacktestTool instance"""
        if self._backtest_tool is None:
            try:
                from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
                self._backtest_tool = ConfigBacktestTool()
            except Exception as e:
                print(f"Failed to initialize ConfigBacktestTool: {e}")
                return None
        return self._backtest_tool

    async def run_backtest(
        self,
        config_yaml: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_cash: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run backtest with given config"""
        backtest_id = f"bt-{uuid.uuid4().hex[:8]}"
        
        # Store pending result
        self._results[backtest_id] = {
            "id": backtest_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_yaml": config_yaml,
        }

        tool = self._get_backtest_tool()
        if not tool:
            return self._create_error_result(backtest_id, "BacktestTool not available")

        try:
            # Execute backtest
            result = await tool.execute(
                config_yaml=config_yaml,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                data_path=data_path,
            )

            # Format result
            backtest_result = {
                "id": backtest_id,
                "status": result.get("status", "completed"),
                "summary": self._format_summary(result.get("summary", {})),
                "config_info": result.get("config_info", {}),
                "warnings": result.get("warnings", []),
                "errors": result.get("errors", []),
                "output_files": result.get("output_files", {}),
                "created_at": self._results[backtest_id]["created_at"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store result
            self._results[backtest_id] = backtest_result
            
            # Save to history
            await self._save_to_history(backtest_result)

            return backtest_result

        except Exception as e:
            return self._create_error_result(backtest_id, str(e))

    async def get_result(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest result by ID"""
        return self._results.get(backtest_id)

    async def get_history(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get backtest history"""
        # Try to load from file first
        if self._history_file.exists():
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    None, lambda: self._history_file.read_text(encoding="utf-8")
                )
                results = []
                for line in content.splitlines():
                    try:
                        data = json.loads(line)
                        results.append(data)
                    except json.JSONDecodeError:
                        continue
                # Sort by created_at descending
                results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return results[offset:offset + limit]
            except (IOError, OSError) as e:
                print(f"Failed to read backtest history file: {e}")
        
        # Fallback to in-memory results
        results = list(self._results.values())
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[offset:offset + limit]

    async def get_templates(self) -> List[Dict[str, Any]]:
        """Get backtest templates"""
        return [
            {
                "name": "Momentum Strategy",
                "description": "Buy stocks with strong recent momentum",
                "yaml": self._get_template("momentum"),
            },
            {
                "name": "Mean Reversion",
                "description": "Buy oversold, sell overbought",
                "yaml": self._get_template("mean_reversion"),
            },
            {
                "name": "Trend Following",
                "description": "Follow strong trends",
                "yaml": self._get_template("trend_following"),
            },
        ]

    def _format_summary(self, raw_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Format summary for API response"""
        return {
            "total_return": raw_summary.get("total_return", 0),
            "annual_return": raw_summary.get("annualized_return", 0),
            "sharpe_ratio": raw_summary.get("sharpe_ratio", 0),
            "max_drawdown": raw_summary.get("max_drawdown", 0),
            "win_rate": raw_summary.get("win_rate", 0),
            "total_trades": raw_summary.get("total_trades", 0),
            "final_cash": raw_summary.get("final_cash", 0),
            "total_commission": raw_summary.get("total_commission", 0),
            "sortino_ratio": raw_summary.get("sortino_ratio", 0),
            "calmar_ratio": raw_summary.get("calmar_ratio", 0),
            "profit_factor": raw_summary.get("profit_factor", 0),
            "avg_trade_pnl": raw_summary.get("avg_trade_pnl", 0),
            "trading_days": raw_summary.get("trading_days", 0),
        }

    def _create_error_result(self, backtest_id: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        result = {
            "id": backtest_id,
            "status": "failed",
            "summary": {},
            "config_info": {},
            "warnings": [],
            "errors": [error],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._results[backtest_id] = result
        return result

    async def _save_to_history(self, result: Dict[str, Any]) -> None:
        """Save result to history file"""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            history_entry = {
                "id": result["id"],
                "status": result["status"],
                "summary": result["summary"],
                "config_info": result["config_info"],
                "created_at": result.get("created_at"),
                "completed_at": result.get("completed_at"),
            }
            import asyncio

            def _append_write(path: Path, content: str) -> None:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)

            loop = asyncio.get_event_loop()
            content = json.dumps(history_entry, ensure_ascii=False) + "\n"
            await loop.run_in_executor(
                None, lambda: _append_write(self._history_file, content)
            )
        except Exception as e:
            print(f"Failed to save backtest history: {e}")

    def _get_template(self, template_name: str) -> str:
        """Get template YAML"""
        templates = {
            "momentum": """# Momentum Strategy
strategy:
  name: momentum_strategy
  
signals:
  - name: momentum_20d
    formula: "close / close.shift(20) - 1"
    weight: 0.5
  - name: momentum_60d
    formula: "close / close.shift(60) - 1"
    weight: 0.3
  - name: volume_momentum
    formula: "volume / volume.rolling(20).mean()"
    weight: 0.2
    
portfolio:
  rebalance_days: 5
  max_positions: 20
  position_sizing: equal_weight
  
risk:
  max_drawdown: 0.15
  stop_loss: 0.05
""",
            "mean_reversion": """# Mean Reversion Strategy
strategy:
  name: mean_reversion_strategy
  
signals:
  - name: rsi_14
    formula: "100 - 100 / (1 + rs(close, 14))"
    weight: 0.4
  - name: bollinger_signal
    formula: "(close - bb_lower(close, 20, 2)) / (bb_upper(close, 20, 2) - bb_lower(close, 20, 2))"
    weight: 0.3
  - name: volume_spike
    formula: "volume / volume.rolling(20).mean()"
    weight: 0.3
    
portfolio:
  rebalance_days: 1
  max_positions: 30
  position_sizing: inverse_volatility
  
risk:
  max_drawdown: 0.12
  stop_loss: 0.03
""",
            "trend_following": """# Trend Following Strategy
strategy:
  name: trend_following_strategy
  
signals:
  - name: trend_strength
    formula: "close / ts_max(close, 60) - 1"
    weight: 0.4
  - name: ma_cross
    formula: "ts_mean(close, 20) / ts_mean(close, 60) - 1"
    weight: 0.3
  - name: volatility_regime
    formula: "ts_std(close, 20) / ts_std(close, 60)"
    weight: 0.3
    
portfolio:
  rebalance_days: 10
  max_positions: 15
  position_sizing: volatility_target
  
risk:
  max_drawdown: 0.20
  trailing_stop: 0.08
""",
        }
        return templates.get(template_name, templates["momentum"])


# Singleton instance
backtest_service = BacktestService()
