"""scripts/cmf/test_options_strategy.py — 阶段2: VIX自适应期权策略.

回测范围: 2020-2025 (有真实期权数据)
期权参数:
  - 到期: 1个月
  - 行权价: 3% OTM
  - 换月: 提前1-2周
  - 流动性: 成交量>100手
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
    """加载SHFE黄金期权数据, 只保留流动性好的合约."""
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


class OptionStrategy:
    """VIX自适应期权策略.

    VIX<20: 卖出Covered Call (获取权利金)
    20<=VIX<=30: 不使用期权
    VIX>30: 买入Protective Put (保护下行)
    换月: 提前1-2周
    """

    def __init__(self, options_df, vix, close,
                 vix_low=20, vix_high=30, otm_pct=0.03,
                 min_volume=100, rollover_days=10,
                 position_per_unit=1.0):
        self.options = options_df
        self.vix = vix
        self.close = close
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.otm_pct = otm_pct
        self.min_volume = min_volume
        self.rollover_days = rollover_days
        self.position_per_unit = position_per_unit

        # 当前持仓状态
        self.current_put = None  # {'entry_date', 'expiry_date', 'strike', 'premium', 'quantity'}
        self.current_call = None

    def select_option(self, trade_date, option_type, spot_price):
        """选择最优期权合约."""
        # 找到当天可用的期权合约
        daily_opts = self.options[
            (self.options["trade_date"] == trade_date) &
            (self.options["option_type"] == option_type) &
            (self.options["volume"] >= self.min_volume)
        ]
        if daily_opts.empty:
            return None

        # 目标行权价
        if option_type == "P":
            target_strike = spot_price * (1 - self.otm_pct)
            # 选最接近且低于目标的
            valid = daily_opts[daily_opts["strike_price"] <= target_strike]
            if valid.empty:
                return None
            # 选最接近平值的Put
            best = valid.loc[valid["strike_price"].idxmax()]
        else:  # Call
            target_strike = spot_price * (1 + self.otm_pct)
            valid = daily_opts[daily_opts["strike_price"] >= target_strike]
            if valid.empty:
                return None
            # 选最接近平值的Call
            best = valid.loc[valid["strike_price"].idxmin()]

        return {
            "trade_date": trade_date,
            "strike": best["strike_price"],
            "premium": best["close"],
            "quantity": self.position_per_unit
        }

    def run(self, base_position, start="2020-01-01", end="2025-12-31"):
        """运行期权策略叠加在基础仓位上."""
        result = base_position.copy()
        dates = result[(result.index >= start) & (result.index <= end)].index

        for date in dates:
            if date not in self.vix.index:
                continue
            vix_val = self.vix.loc[date]
            spot = self.close.loc[date] if date in self.close.index else None
            if spot is None:
                continue

            # 换月检查
            if self.current_put is not None:
                days_to_expiry = (self.current_put["expiry_date"] - date).days
                if days_to_expiry <= self.rollover_days:
                    # 平仓当前Put
                    result.loc[date] += self._settle_option(
                        self.current_put, date, spot, "P"
                    )
                    self.current_put = None

            if self.current_call is not None:
                days_to_expiry = (self.current_call["expiry_date"] - date).days
                if days_to_expiry <= self.rollover_days:
                    result.loc[date] += self._settle_option(
                        self.current_call, date, spot, "C"
                    )
                    self.current_call = None

            # 决策是否开仓
            if vix_val > self.vix_high and self.current_put is None:
                # 高VIX: 买Put
                opt = self.select_option(date, "P", spot)
                if opt is not None:
                    # 估算到期日(下个月同一天)
                    opt["expiry_date"] = self._next_month_date(date)
                    opt["entry_premium"] = opt["premium"]
                    self.current_put = opt
                    # 支付权利金
                    result.loc[date] -= opt["premium"] * opt["quantity"] / spot

            elif vix_val < self.vix_low and self.current_call is None:
                # 低VIX: 卖Call
                opt = self.select_option(date, "C", spot)
                if opt is not None:
                    opt["expiry_date"] = self._next_month_date(date)
                    opt["entry_premium"] = opt["premium"]
                    self.current_call = opt
                    # 获得权利金
                    result.loc[date] += opt["premium"] * opt["quantity"] / spot

            # 日内盯市盈亏
            if self.current_put is not None:
                opt = self._get_current_premium(date, "P", spot)
                if opt is not None:
                    pnl = (opt - self.current_put["entry_premium"]) * \
                          self.current_put["quantity"] / spot
                    result.loc[date] += pnl

            if self.current_call is not None:
                opt = self._get_current_premium(date, "C", spot)
                if opt is not None:
                    pnl = (self.current_call["entry_premium"] - opt) * \
                          self.current_call["quantity"] / spot
                    result.loc[date] += pnl

        return result

    def _next_month_date(self, date):
        """计算下个月同一天."""
        import calendar
        if date.month == 12:
            next_month = 1
            next_year = date.year + 1
        else:
            next_month = date.month + 1
            next_year = date.year
        last_day = calendar.monthrange(next_year, next_month)[1]
        return pd.Timestamp(year=next_year, month=next_month,
                           day=min(date.day, last_day))

    def _get_current_premium(self, date, option_type, spot):
        """获取当前期权价格."""
        daily_opts = self.options[
            (self.options["trade_date"] == date) &
            (self.options["option_type"] == option_type) &
            (self.options["volume"] >= self.min_volume)
        ]
        if daily_opts.empty:
            return None
        if option_type == "P":
            target = spot * (1 - self.otm_pct)
            valid = daily_opts[daily_opts["strike_price"] <= target]
            if valid.empty:
                return None
            return valid.loc[valid["strike_price"].idxmax()]["close"]
        else:
            target = spot * (1 + self.otm_pct)
            valid = daily_opts[daily_opts["strike_price"] >= target]
            if valid.empty:
                return None
            return valid.loc[valid["strike_price"].idxmin()]["close"]

    def _settle_option(self, opt, date, spot, option_type):
        """到期结算."""
        if option_type == "P":
            intrinsic = max(0, opt["strike"] - spot)
        else:
            intrinsic = max(0, spot - opt["strike"])
        # 买入方: intrinsic - entry_premium
        # 卖出方: entry_premium - intrinsic
        if option_type == "P":
            # 买入Put
            pnl = (intrinsic - opt["entry_premium"]) * opt["quantity"] / spot
        else:
            # 卖出Call
            pnl = (opt["entry_premium"] - intrinsic) * opt["quantity"] / spot
        return pnl


def run_options_tests():
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

    # 测试基础策略(2020-2025)
    r_base = calc_metrics(final_base, close, "2020-01-01", "2025-12-31")
    print(f"\n=== 基础策略 (2020-2025, 无期权) ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}, "
          f"换手={r_base['turnover']:.1f}")

    # 测试不同的VIX阈值
    print("\n=== VIX自适应期权策略 ===")
    print(f'{"配置":<40} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')

    for vix_low, vix_high, otm in [
        (20, 30, 0.03),
        (18, 28, 0.03),
        (22, 32, 0.03),
        (20, 30, 0.05),
        (20, 30, 0.02),
    ]:
        try:
            strategy = OptionStrategy(
                options, vix, close,
                vix_low=vix_low, vix_high=vix_high,
                otm_pct=otm, min_volume=50
            )
            combined = strategy.run(final_base, "2020-01-01", "2025-12-31")
            r = calc_metrics(combined, close, "2020-01-01", "2025-12-31")
            print(f'  VIX<{vix_low}/VIX>{vix_high}/OTM={otm}: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')
        except Exception as e:
            print(f"  VIX<{vix_low}/VIX>{vix_high}: 错误 {e}")


if __name__ == "__main__":
    run_options_tests()
