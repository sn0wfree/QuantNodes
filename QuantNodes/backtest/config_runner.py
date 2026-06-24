# coding=utf-8
"""
配置驱动的回测运行器

直接从 StrategyConfig + Polars 数据执行完整回测，
不经过代码生成，直接调用 backtest/ 引擎。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from QuantNodes.agent.config.types import StrategyConfig
from QuantNodes.agent.config.executor import ConfigExecutor
from QuantNodes.backtest.config_strategy import ConfigStrategyNode
from QuantNodes.backtest.backtest_node import BacktestResult
from QuantNodes.backtest.strategy_node import OrdersResult
from QuantNodes.backtest.broker_node import ExecutionBrokerNode
from QuantNodes.backtest.risk_node import PositionLimitRiskNode, RiskNode
from QuantNodes.core.path_utils import ensure_dir


class ConfigBacktestRunner:
    """从 StrategyConfig + Polars 数据执行完整回测"""

    def run(
        self, config: StrategyConfig, data: pl.LazyFrame
    ) -> BacktestResult:
        """执行回测

        Args:
            config: 策略配置
            data: Polars LazyFrame 数据

        Returns:
            BacktestResult 包含交易、统计等信息
        """
        if config.backtest is None:
            return BacktestResult()

        # 1. 因子计算 + 信号生成
        executor = ConfigExecutor()
        result = executor.run_backtest(config, data)

        if result.status == "error":
            return BacktestResult()

        # 2. Polars → Pandas
        df = result.data.collect().to_pandas()

        # 3. 列名标准化
        df = self._normalize_columns(df)

        # 4. 确保 signal 列存在
        if "signal" not in df.columns:
            return BacktestResult()

        # 5. 策略 → 风控 → 经纪商
        strategy = ConfigStrategyNode(signal_col="signal")
        orders_result = strategy.execute(df)

        risk_nodes = self._build_risk_nodes(config)
        filtered = self._apply_risk(orders_result, risk_nodes)

        broker = self._build_broker(config)
        trade_result = broker.execute((filtered, df))

        # 6. 计算绩效统计
        return self._compute_statistics(trade_result, df, config)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列名大小写"""
        rename_map = {}

        # Code/code → Code
        if "Code" not in df.columns and "code" in df.columns:
            rename_map["code"] = "Code"

        # close → Close
        if "Close" not in df.columns and "close" in df.columns:
            rename_map["close"] = "Close"

        # open → Open
        if "Open" not in df.columns and "open" in df.columns:
            rename_map["open"] = "Open"

        if rename_map:
            df = df.rename(columns=rename_map)

        # 确保 Open 列存在（fallback 到 Close）
        if "Open" not in df.columns and "Close" in df.columns:
            df["Open"] = df["Close"]

        return df

    def _build_risk_nodes(self, config: StrategyConfig) -> List[RiskNode]:
        """从 config 构建风控节点"""
        nodes = []
        bt = config.backtest
        if bt and bt.positions:
            max_pos = bt.positions.get("max_positions")
            if max_pos is not None:
                nodes.append(PositionLimitRiskNode(
                    config={"max_position": max_pos}
                ))
        return nodes

    def _build_broker(self, config: StrategyConfig) -> ExecutionBrokerNode:
        """从 config 构建经纪商"""
        bt = config.backtest
        broker_config = {
            "cash": bt.initial_cash if bt else 1_000_000,
            "commission": bt.commission if bt else 0.001,
            "slippage": bt.slippage if bt else 0.001,
        }

        # 传递可选的Broker参数
        if bt and hasattr(bt, 'positions'):
            for key in ("trade_on_close", "hedging"):
                if key in bt.positions:
                    broker_config[key] = bt.positions[key]

        return ExecutionBrokerNode(config=broker_config)

    def _apply_risk(
        self, orders_result: OrdersResult, risk_nodes: List[RiskNode]
    ) -> OrdersResult:
        """应用风控过滤"""
        current_orders = orders_result
        for node in risk_nodes:
            risk_result = node.execute(current_orders)
            new_orders = OrdersResult()
            new_orders.orders = risk_result.passed_orders
            new_orders.signals = orders_result.signals
            current_orders = new_orders
        return current_orders

    def _compute_statistics(
        self, trade_result, df: pd.DataFrame, config: StrategyConfig
    ) -> BacktestResult:
        """计算绩效统计（含权益曲线和风险指标）"""
        bt = config.backtest
        initial_cash = bt.initial_cash if bt else 1_000_000

        trades_df = trade_result.to_dataframe()

        # ── 1. 权益曲线 ──────────────────────────────────────────────
        equity_curve = self._build_equity_curve(trades_df, df, initial_cash)

        # ── 2. 日收益率序列 ──────────────────────────────────────────
        # v2.9.0: short-circuit empty equity curve to avoid pandas.pct_change
        # raising on empty Series (np.argmax(empty) ValueError).
        if len(equity_curve) == 0:
            daily_returns = pd.Series([], dtype=float)
        else:
            daily_returns = equity_curve["equity"].pct_change().fillna(0.0)

        # ── 3. 基础指标 ──────────────────────────────────────────────
        final_equity = equity_curve["equity"].iloc[-1] if len(equity_curve) > 0 else initial_cash
        total_return = (final_equity - initial_cash) / initial_cash

        n_days = len(equity_curve)
        n_years = n_days / 252 if n_days > 0 else 0
        annualized_return = ((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # ── 4. 风险指标 ──────────────────────────────────────────────
        ann_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        risk_free = 0.03
        sharpe = (
            (daily_returns.mean() - risk_free / 252) / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 1e-12 else 0.0
        )
        downside = daily_returns[daily_returns < 0]
        sortino = (
            (daily_returns.mean() - risk_free / 252) / downside.std() * np.sqrt(252)
            if len(downside) > 0 and downside.std() > 1e-12 else 0.0
        )
        max_drawdown = self._max_drawdown(equity_curve["equity"])
        calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else 0.0

        # ── 5. 交易统计 ──────────────────────────────────────────────
        total_trades = len(trade_result.trades)
        trade_pnls = self._compute_trade_pnl(trades_df)
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
        profit_factor = (
            abs(sum(wins)) / abs(sum(losses))
            if losses else float("inf") if wins else 0.0
        )
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0

        statistics = {
            "total_trades": total_trades,
            "total_commission": trade_result.commission,
            "executed_value": trade_result.executed_value,
            "annualized_return": annualized_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "num_wins": len(wins),
            "num_losses": len(losses),
            "trading_days": n_days,
        }

        return BacktestResult(
            trades=trades_df,
            orders=pd.DataFrame(),
            equity_curve=equity_curve,
            statistics=statistics,
            final_cash=trade_result.cash,
            final_positions=trade_result.to_dataframe().groupby("code").apply(
                lambda g: (g["size"] * np.where(g["side"] == "buy", 1, -1)).sum()
            ).to_dict() if len(trades_df) > 0 else {},
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
        )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_equity_curve(
        trades_df: pd.DataFrame,
        quote_df: pd.DataFrame,
        initial_cash: float,
    ) -> pd.DataFrame:
        """从交易记录和行情数据重建逐日权益曲线。

        对每个交易日：
        1. 回放截至当日的交易 → 得到持仓 + 可用现金
        2. 用当日收盘价 mark-to-market → equity = cash + sum(pos * close)
        """
        if quote_df.empty:
            return pd.DataFrame(columns=["date", "equity", "cash", "position_value"])

        dates = sorted(quote_df["date"].unique())

        # 构建 close 价格查找表: {date_str: {code: close}}
        close_map: Dict[str, Dict[str, float]] = {}
        for d, grp in quote_df.groupby("date"):
            d_str = str(d)[:10]  # 统一为 YYYY-MM-DD 格式
            close_map[d_str] = dict(zip(grp["Code"], grp["Close"]))

        # 逐日回放交易
        positions: Dict[str, float] = {}
        cash = initial_cash
        records: List[Dict[str, Any]] = []

        trade_idx = 0
        trades_sorted = (
            trades_df.sort_values("dt").to_dict("records")
            if len(trades_df) > 0 else []
        )

        for d in dates:
            d_str = str(d)[:10]  # 统一为 YYYY-MM-DD 格式

            # 处理当日交易
            while (
                trade_idx < len(trades_sorted)
                and str(trades_sorted[trade_idx]["dt"])[:10] == d_str
            ):
                t = trades_sorted[trade_idx]
                sign = 1.0 if t["side"] == "buy" else -1.0
                qty = t["size"] * sign
                cash -= t["adjusted_price"] * qty + t["fee"] * sign
                positions[t["code"]] = positions.get(t["code"], 0.0) + qty
                trade_idx += 1

            # mark-to-market
            c_map = close_map.get(d_str, {})
            pos_value = sum(
                positions[code] * c_map.get(code, 0.0) for code in positions
            )
            equity = cash + pos_value
            records.append({
                "date": d,
                "equity": equity,
                "cash": cash,
                "position_value": pos_value,
            })

        return pd.DataFrame(records)

    @staticmethod
    def _compute_trade_pnl(trades_df: pd.DataFrame) -> List[float]:
        """按 code 分组配对买卖，计算每轮盈亏。"""
        if trades_df.empty:
            return []

        pnls: List[float] = []
        for code, grp in trades_df.groupby("code"):
            buys = grp[grp["side"] == "buy"]
            sells = grp[grp["side"] == "sell"]
            total_buy_cost = (buys["adjusted_price"] * buys["size"]).sum()
            total_sell_rev = (sells["adjusted_price"] * sells["size"]).sum()
            pnls.append(total_sell_rev - total_buy_cost)
        return pnls

    @staticmethod
    def _max_drawdown(equity_series: pd.Series) -> float:
        """计算最大回撤（返回负值，如 -0.05 表示 5% 回撤）。"""
        if equity_series.empty:
            return 0.0
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        return float(dd.min())

    # ── Output 保存 ────────────────────────────────────────────────

    def save_output(
        self,
        bt_result: BacktestResult,
        config: StrategyConfig,
        signals_df: Optional[pd.DataFrame] = None,
        positions_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """根据 OutputConfig 保存回测结果到文件。

        Returns:
            保存路径字典，如 {"equity_curve": "outputs/equity.parquet", ...}
        """
        output_cfg = config.output
        if output_cfg is None:
            return {}

        out_dir = Path(output_cfg.path).parent
        ensure_dir(out_dir)

        fmt = output_cfg.format.lower()
        saved: Dict[str, str] = {}

        stem = Path(output_cfg.path).stem

        if (
            output_cfg.save_equity_curve
            and bt_result.equity_curve is not None
            and not bt_result.equity_curve.empty
        ):
            p = str(out_dir / f"{stem}_equity.{fmt}")
            self._save_dataframe(bt_result.equity_curve, p, fmt)
            saved["equity_curve"] = p

        if output_cfg.save_signals and signals_df is not None and not signals_df.empty:
            p = str(out_dir / f"{stem}_signals.{fmt}")
            self._save_dataframe(signals_df, p, fmt)
            saved["signals"] = p

        if (
            output_cfg.save_positions
            and bt_result.trades is not None
            and not bt_result.trades.empty
        ):
            p = str(out_dir / f"{stem}_trades.{fmt}")
            self._save_dataframe(bt_result.trades, p, fmt)
            saved["trades"] = p

        if output_cfg.save_positions and positions_df is not None and not positions_df.empty:
            p = str(out_dir / f"{stem}_positions.{fmt}")
            self._save_dataframe(positions_df, p, fmt)
            saved["positions"] = p

        stats_path = str(out_dir / f"{stem}_statistics.json")
        import json
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(bt_result.statistics, f, indent=2, ensure_ascii=False, default=str)
        saved["statistics"] = stats_path

        return saved

    @staticmethod
    def _save_dataframe(df: pd.DataFrame, path: str, fmt: str) -> None:
        """保存 DataFrame 到文件。"""
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
