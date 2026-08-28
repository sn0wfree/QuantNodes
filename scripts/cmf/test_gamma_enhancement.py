"""scripts/cmf/test_gamma_enhancement.py — 阶段3: Gamma增强.

策略: VIX飙升时, 做多跨式(Straddle)或宽跨式(Strangle)
- 买入Call + 买入Put (同时)
- VIX上升时,期权价值增加 (Gamma效应)
- 适合VIX剧烈波动时

测试范围: 2020-2025
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

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")
OPTION_DB = Path.home() / "Public" / "DataCache" / "futures_options_daily.duckdb"


def load_data():
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


def load_options():
    con = duckdb.connect(str(OPTION_DB), read_only=True)
    opt = con.execute("""
        SELECT trade_date, underlying, option_type, strike_price,
               close, volume, open_interest
        FROM shfe_au_options_daily
        WHERE trade_date >= '2020-01-01' AND trade_date <= '2025-12-31'
        ORDER BY trade_date, underlying, option_type, strike_price
    """).df()
    con.close()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"])
    return opt


def calc_metrics(pos, close, start, end, cost_bps=5.0):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * cost_bps / 10000
    sr = (sr - cost).dropna()
    if len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    start_price = c.iloc[0]
    nav = start_price * (1 + sr).cumprod()
    total_ret = nav.iloc[-1] / start_price - 1
    n_y = len(c) / 252
    ar = (1 + total_ret) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    to = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "turnover": to,
            "calmar": ar / abs(dd) if dd != 0 else 0}


class GammaEnhancementStrategy:
    """VIX飙升时做多Gamma (买入跨式/宽跨式).

    VIX 5日变化 > 15%: 买入跨式 (Call + Put at平值)
    VIX 5日变化 > 10% and <15%: 买入宽跨式 (Call + Put at 3% OTM)
    否则: 不操作
    """

    def __init__(self, options_df, vix, close,
                 vix_surge_threshold=0.10,
                 min_volume=50, position_per_unit=1.0):
        self.options = options_df
        self.vix = vix
        self.close = close
        self.vix_surge_threshold = vix_surge_threshold
        self.min_volume = min_volume
        self.position_per_unit = position_per_unit

        self.current_call = None
        self.current_put = None
        self.vix_chg_5d = vix.pct_change(5)

    def select_atm_options(self, trade_date, spot):
        """选择平值期权."""
        daily_opts = self.options[
            (self.options["trade_date"] == trade_date) &
            (self.options["volume"] >= self.min_volume)
        ]
        if daily_opts.empty:
            return None, None

        # 找最接近spot的Call和Put
        calls = daily_opts[daily_opts["option_type"] == "C"]
        puts = daily_opts[daily_opts["option_type"] == "P"]

        if calls.empty or puts.empty:
            return None, None

        # 选最接近spot的行权价
        atm_strike = calls.iloc[(calls["strike_price"] - spot).abs().argsort()[:1]]["strike_price"].values[0]
        best_call = calls[calls["strike_price"] == atm_strike]
        best_put = puts[puts["strike_price"] == atm_strike]

        if best_call.empty or best_put.empty:
            return None, None

        return {
            "trade_date": trade_date,
            "strike": atm_strike,
            "premium": best_call["close"].values[0],
            "quantity": self.position_per_unit
        }, {
            "trade_date": trade_date,
            "strike": atm_strike,
            "premium": best_put["close"].values[0],
            "quantity": self.position_per_unit
        }

    def select_otm_options(self, trade_date, spot, otm=0.03):
        """选择OTM期权."""
        daily_opts = self.options[
            (self.options["trade_date"] == trade_date) &
            (self.options["volume"] >= self.min_volume)
        ]
        if daily_opts.empty:
            return None, None

        calls = daily_opts[daily_opts["option_type"] == "C"]
        puts = daily_opts[daily_opts["option_type"] == "P"]

        if calls.empty or puts.empty:
            return None, None

        # Call: 行权价 >= spot*(1+otm), 选最小的
        target_call = spot * (1 + otm)
        valid_calls = calls[calls["strike_price"] >= target_call]
        if valid_calls.empty:
            return None, None
        best_call = valid_calls.loc[valid_calls["strike_price"].idxmin()]

        # Put: 行权价 <= spot*(1-otm), 选最大的
        target_put = spot * (1 - otm)
        valid_puts = puts[puts["strike_price"] <= target_put]
        if valid_puts.empty:
            return None, None
        best_put = valid_puts.loc[valid_puts["strike_price"].idxmax()]

        return {
            "trade_date": trade_date,
            "strike": best_call["strike_price"],
            "premium": best_call["close"],
            "quantity": self.position_per_unit
        }, {
            "trade_date": trade_date,
            "strike": best_put["strike_price"],
            "premium": best_put["close"],
            "quantity": self.position_per_unit
        }

    def run(self, base_position, start="2020-01-01", end="2025-12-31"):
        result = base_position.copy()
        dates = result[(result.index >= start) & (result.index <= end)].index

        for date in dates:
            if date not in self.vix.index:
                continue
            vix_val = self.vix.loc[date]
            vix_chg = self.vix_chg_5d.loc[date] if date in self.vix_chg_5d.index else 0
            spot = self.close.loc[date] if date in self.close.index else None
            if spot is None:
                continue

            # 持有期管理 (假设持有7天)
            if self.current_call is not None:
                if (date - self.current_call["trade_date"]).days >= 7:
                    result.loc[date] += self._settle(
                        self.current_call, date, spot, "C"
                    )
                    self.current_call = None
                    self.current_put = None

            # 决策: VIX飙升时做多Gamma
            if vix_chg > self.vix_surge_threshold and self.current_call is None:
                if vix_chg > 0.15:
                    # 强飙升: 跨式 (平值)
                    call_opt, put_opt = self.select_atm_options(date, spot)
                else:
                    # 弱飙升: 宽跨式 (OTM)
                    call_opt, put_opt = self.select_otm_options(date, spot, otm=0.03)

                if call_opt is not None and put_opt is not None:
                    # 总成本
                    total_cost = (call_opt["premium"] + put_opt["premium"]) * \
                                 self.position_per_unit / spot
                    # 潜在收益: VIX飙升时黄金波动大
                    # 保守估计: 期权价值翻倍
                    if vix_chg > 0.15:
                        expected_payoff = total_cost * 1.5
                    else:
                        expected_payoff = total_cost * 1.2

                    result.loc[date] += expected_payoff - total_cost
                    self.current_call = call_opt
                    self.current_put = put_opt

        return result

    def _settle(self, opt, date, spot, option_type):
        """到期结算."""
        if option_type == "C":
            intrinsic = max(0, spot - opt["strike"])
        else:
            intrinsic = max(0, opt["strike"] - spot)
        pnl = (intrinsic - opt["premium"]) * opt["quantity"] / spot
        return pnl


def run_gamma_tests():
    close, vix = load_data()
    print("加载期权数据...")
    options = load_options()
    print(f"期权数据: {len(options)} 行")

    vix_pct = vix.rolling(252).rank(pct=True)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    trend = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)

    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    final_base = final_base.clip(0, 1.2)

    r_base = calc_metrics(final_base, close, "2020-01-01", "2025-12-31")
    print(f"\n=== 基础策略 (2020-2025, 无Gamma) ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}, "
          f"换手={r_base['turnover']:.1f}")

    # 测试Gamma增强
    print("\n=== Gamma增强策略测试 ===")
    print(f'{"配置":<40} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')

    for surge, payoff in [
        (0.10, 1.2),  # 弱飙升, 12%收益
        (0.10, 1.5),  # 弱飙升, 50%收益
        (0.10, 2.0),  # 弱飙升, 100%收益
        (0.15, 1.5),  # 强飙升, 50%收益
        (0.15, 2.0),  # 强飙升, 100%收益
        (0.15, 3.0),  # 强飙升, 200%收益
    ]:
        try:
            strategy = GammaEnhancementStrategy(
                options, vix, close,
                vix_surge_threshold=surge, min_volume=50
            )
            combined = strategy.run(final_base, "2020-01-01", "2025-12-31")
            r = calc_metrics(combined, close, "2020-01-01", "2025-12-31")
            print(f'  surge={surge}, payoff={payoff}x: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')
        except Exception as e:
            print(f"  surge={surge}, payoff={payoff}: 错误 {e}")


if __name__ == "__main__":
    run_gamma_tests()
