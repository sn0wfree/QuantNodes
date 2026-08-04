"""IV Skew computation for CSI 300 Index Options (IO).

Computes:
1. Implied Volatility for each option using Black-Scholes
2. IV Skew = IV(25-delta put) - IV(25-delta call)
3. ATM IV = average IV of near-the-money options

Output: DataFrame with date index and iv_skew, iv_atm columns.
"""
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import norm
from scipy.optimize import brentq
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO.parent / "DataCache/futures_options_daily.duckdb")


def _bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if option_type == "call" else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def _implied_vol(price, S, K, T, r, option_type="call"):
    """Compute implied volatility using Brent's method."""
    if price <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if price <= intrinsic + 0.01:
        return np.nan
    try:
        iv = brentq(
            lambda sigma: _bs_price(S, K, T, r, sigma, option_type) - price,
            0.01, 3.0, xtol=1e-6, maxiter=100
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def _extract_expiry_from_symbol(symbol):
    """Extract expiry year/month from symbol like 'IO2609-C-4000'."""
    parts = symbol.split("-")
    underlying = parts[0]  # e.g., IO2609
    yymm = underlying[2:]  # e.g., 2609
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    return year, month


def compute_iv_skew(trade_date, options_df, if_futures_df, r=0.02):
    """Compute IV skew for a single date.
    
    Args:
        trade_date: datetime
        options_df: IO options for this date
        if_futures_df: IF futures for this date
        r: risk-free rate (default 2%)
    
    Returns:
        dict with iv_skew, iv_atm, or None if computation fails
    """
    # Get underlying price (IF futures close)
    if if_futures_df.empty:
        return None
    
    # Use the most traded IF contract as underlying proxy
    S = if_futures_df.sort_values("volume", ascending=False).iloc[0]["close"]
    
    # Parse options
    results = []
    for _, row in options_df.iterrows():
        try:
            K = row["strike_price"]
            price = row["close"]
            option_type = "call" if row["option_type"] == "C" else "put"
            
            # Extract expiry
            year, month = _extract_expiry_from_symbol(row["symbol"])
            expiry = pd.Timestamp(year=year, month=month, day=15)  # Approximate
            T = max((expiry - trade_date).days / 365.0, 1/365)
            
            # Compute IV
            iv = _implied_vol(price, S, K, T, r, option_type)
            if np.isnan(iv):
                continue
            
            # Compute delta
            delta = _bs_delta(S, K, T, r, iv, option_type)
            
            results.append({
                "strike": K,
                "option_type": option_type,
                "price": price,
                "iv": iv,
                "delta": delta,
                "volume": row["volume"],
            })
        except Exception:
            continue
    
    if len(results) < 10:
        return None
    
    df = pd.DataFrame(results)
    
    # Find 25-delta put (delta ≈ -0.25) and 25-delta call (delta ≈ +0.25)
    puts = df[df["option_type"] == "put"]
    calls = df[df["option_type"] == "call"]
    
    if len(puts) < 3 or len(calls) < 3:
        return None
    
    # Find closest to -0.25 delta for puts
    put_25 = puts.iloc[(puts["delta"] - (-0.25)).abs().argsort()[:3]]
    iv_put_25 = put_25["iv"].mean()
    
    # Find closest to +0.25 delta for calls
    call_25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:3]]
    iv_call_25 = call_25["iv"].mean()
    
    # Find ATM options (delta close to 0.5 for calls, -0.5 for puts)
    atm_calls = calls.iloc[(calls["delta"] - 0.5).abs().argsort()[:3]]
    atm_puts = puts.iloc[(puts["delta"] - (-0.5)).abs().argsort()[:3]]
    iv_atm = pd.concat([atm_calls["iv"], atm_puts["iv"]]).mean()
    
    # IV Skew = IV(25-delta put) - IV(25-delta call)
    iv_skew = iv_put_25 - iv_call_25
    
    return {
        "iv_skew": iv_skew,
        "iv_atm": iv_atm,
        "iv_put_25": iv_put_25,
        "iv_call_25": iv_call_25,
        "n_options": len(df),
    }


def build_iv_skew_series():
    """Build IV skew time series from IO options data."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    # Get all trading dates with IO options
    dates = conn.execute("""
        SELECT DISTINCT trade_date 
        FROM cffex_io_options_daily 
        WHERE volume > 0
        ORDER BY trade_date
    """).fetchdf()["trade_date"].tolist()
    
    results = []
    for date in dates:
        # Get IO options for this date
        options = conn.execute(f"""
            SELECT symbol, option_type, strike_price, close, volume
            FROM cffex_io_options_daily
            WHERE trade_date = '{date}' AND volume > 0 AND close > 0
        """).fetchdf()
        
        # Get IF futures for this date
        if_futures = conn.execute(f"""
            SELECT symbol, close, volume
            FROM cffex_if_futures_daily
            WHERE trade_date = '{date}' AND volume > 0
        """).fetchdf()
        
        result = compute_iv_skew(pd.Timestamp(date), options, if_futures)
        if result is not None:
            result["date"] = pd.Timestamp(date)
            results.append(result)
    
    conn.close()
    
    df = pd.DataFrame(results).set_index("date").sort_index()
    return df


if __name__ == "__main__":
    print("Computing IV Skew from IO options data...")
    df = build_iv_skew_series()
    print(f"Computed IV Skew for {len(df)} dates")
    print(f"Date range: {df.index[0]} ~ {df.index[-1]}")
    print(f"\nIV Skew stats:")
    print(f"  Mean: {df['iv_skew'].mean():.4f}")
    print(f"  Std:  {df['iv_skew'].std():.4f}")
    print(f"  Min:  {df['iv_skew'].min():.4f}")
    print(f"  Max:  {df['iv_skew'].max():.4f}")
    print(f"\nIV ATM stats:")
    print(f"  Mean: {df['iv_atm'].mean():.4f}")
    print(f"  Std:  {df['iv_atm'].std():.4f}")
    
    # Save to parquet
    output_path = REPO / "data" / "high_freq_macro" / "io_iv_skew_daily.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
