"""scripts/cmf/test_put_spread.py — 阶段2: Put Spread独立策略 (低成本减MaxDD).

策略: VIX>25时买入Put价差 (Buy OTM Put + Sell 更远OTM Put)
- Buy OTM Put (otm_buy) 保护下行
- Sell OTM Put (otm_sell) 收权利金抵消Buy Put成本
- 最大亏损 = Buy Premium - Sell Premium (远小于Buy Put单独)
- VIX<22时平仓
- 换月: 提前1-2周

回测范围: 2020-2025
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import (
    load_options, calc_metrics,
    next_month_date, select_otm_option,
)
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_base():
    """加载基础数据 (金价, VIX)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def get_base_position(close, vix):
    """计算基础策略仓位."""
    vix_pct = vix.rolling(252).rank(pct=True)
    trend = compute_trend_bnh_signal(
        trailing_stop=0.08, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    return final_base.clip(0, 1.2)


def _scalar(series_or_float):
    """安全地从pd.Series或float提取数值标量 (跳过Timestamp列)."""
    if isinstance(series_or_float, pd.Series):
        for v in series_or_float:
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                return float(v)
        raise ValueError(f"No numeric value in Series: {series_or_float}")
    if pd.isna(series_or_float):
        return 0.0
    return float(series_or_float)


def _get_premium(options_df, date, option_type, spot, otm, min_volume=50):
    """获取单只期权权利金 (返回float或None)."""
    daily = options_df[
        (options_df["trade_date"] == date) &
        (options_df["option_type"] == option_type) &
        (options_df["volume"] >= min_volume)
    ]
    if option_type == "P":
        target = spot * (1 - otm)
        valid = daily[daily["strike_price"] <= target]
        if valid.empty:
            return None
        return _scalar(valid.loc[valid["strike_price"].idxmax(), "close"])
    else:
        target = spot * (1 + otm)
        valid = daily[daily["strike_price"] >= target]
        if valid.empty:
            return None
        return _scalar(valid.loc[valid["strike_price"].idxmin(), "close"])


class PutSpreadStrategy:
    """Put Spread独立策略 (VIX触发,减MaxDD,降低成本).

    开仓: Buy otm_buy OTM Put + Sell otm_sell OTM Put
    平仓: VIX<exit 或 换月
    """

    def __init__(self, options_df, vix, close,
                 vix_entry=25, vix_exit=22,
                 otm_buy=0.03, otm_sell=0.07,
                 min_volume=50):
        self.options = options_df
        self.vix = vix
        self.close = close
        self.vix_entry = vix_entry
        self.vix_exit = vix_exit
        self.otm_buy = otm_buy
        self.otm_sell = otm_sell
        self.min_volume = min_volume
        self.current_spread = None
        self.net_entry_cost = 0.0
        self.entry_date = None

    def run(self, base_position, start="2020-01-01", end="2025-12-31"):
        result = base_position.copy()
        dates = result[(result.index >= start) & (result.index <= end)].index

        for raw_date in dates:
            date = pd.Timestamp(raw_date)
            if date not in self.vix.index or date not in self.close.index:
                continue
            vix_val = float(self.vix.at[date])
            spot = float(self.close.at[date])

            if self.current_spread is not None:
                expiry = pd.Timestamp(self.current_spread["expiry"])
                days_to_expiry = (expiry - date).days
                if days_to_expiry <= 14:
                    self._close_spread(date, spot, result)
                else:
                    self._mark_to_market(date, spot, result)

            if vix_val > self.vix_entry and self.current_spread is None:
                self._open_spread(date, spot, result)
            elif vix_val < self.vix_exit and self.current_spread is not None:
                self._close_spread(date, spot, result)

        return result

    def _open_spread(self, date, spot, result):
        """开仓 Put Spread."""
        long_premium = _get_premium(
            self.options, date, "P", spot, self.otm_buy, self.min_volume
        )
        short_premium = _get_premium(
            self.options, date, "P", spot, self.otm_sell, self.min_volume
        )
        if long_premium is None or short_premium is None:
            return
        net_cost = long_premium - short_premium
        if net_cost <= 0:
            return
        entry_expiry = next_month_date(date)
        self.current_spread = {
            "entry_date": date,
            "expiry": entry_expiry,
            "long_strike": spot * (1 - self.otm_buy),
            "short_strike": spot * (1 - self.otm_sell),
        }
        self.net_entry_cost = net_cost
        self.entry_date = date
        cost_pct = net_cost / spot
        result.at[date] = result.at[date] - cost_pct

    def _mark_to_market(self, date, spot, result):
        """盯市 Put Spread."""
        long_now = _get_premium(
            self.options, date, "P", spot, self.otm_buy, self.min_volume
        )
        short_now = _get_premium(
            self.options, date, "P", spot, self.otm_sell, self.min_volume
        )
        if long_now is not None and short_now is not None:
            net_now = long_now - short_now
            pnl = (self.net_entry_cost - net_now) / spot
            result.at[date] = result.at[date] + pnl
            self.net_entry_cost = net_now

    def _close_spread(self, date, spot, result):
        """平仓 Put Spread."""
        long_now = _get_premium(
            self.options, date, "P", spot, self.otm_buy, self.min_volume
        )
        short_now = _get_premium(
            self.options, date, "P", spot, self.otm_sell, self.min_volume
        )
        if long_now is not None and short_now is not None:
            net_now = long_now - short_now
            pnl = (self.net_entry_cost - net_now) / spot
            result.at[date] = result.at[date] + pnl
        self.current_spread = None
        self.net_entry_cost = 0.0
        self.entry_date = None


def run_put_spread_tests():
    close, vix = load_base()
    print("加载期权数据...")
    options = load_options()
    print(f"期权数据: {len(options)} 行")

    base_pos = get_base_position(close, vix)
    r_base = calc_metrics(base_pos, close, "2020-01-01", "2025-12-31")
    print(f"\n=== 基础策略 (2020-2025, 无Put Spread) ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}")

    print("\n=== Put Spread策略测试 (2020-2025) ===")
    print(f'{"配置":<45} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')

    for vix_entry, vix_exit, otm_buy, otm_sell in [
        (25, 22, 0.03, 0.07),
        (25, 22, 0.03, 0.10),
        (25, 22, 0.05, 0.10),
        (25, 22, 0.05, 0.12),
        (28, 23, 0.03, 0.07),
        (28, 23, 0.03, 0.10),
        (28, 23, 0.05, 0.10),
        (30, 24, 0.05, 0.10),
        (30, 24, 0.05, 0.12),
    ]:
        try:
            strategy = PutSpreadStrategy(
                options, vix, close,
                vix_entry=vix_entry, vix_exit=vix_exit,
                otm_buy=otm_buy, otm_sell=otm_sell, min_volume=50
            )
            combined = strategy.run(base_pos, "2020-01-01", "2025-12-31")
            r = calc_metrics(combined, close, "2020-01-01", "2025-12-31")
            print(f'  VIX>{vix_entry}/退<{vix_exit}/Buy={otm_buy}/Sell={otm_sell}: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')
        except Exception as e:
            import traceback
            print(f'  VIX>{vix_entry}/Buy={otm_buy}/Sell={otm_sell}: 错误 {e}')
            traceback.print_exc()


if __name__ == "__main__":
    run_put_spread_tests()
