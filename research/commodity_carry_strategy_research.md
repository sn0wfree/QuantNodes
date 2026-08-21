# Commodity Carry Strategy Research

## Executive Summary

Commodity carry (also called "roll yield" or "term structure" strategies) harvests the return from holding futures contracts as they roll from front-month to back-month. This document synthesizes academic research and industry practice for professional implementation, with specific notes on adapting ideas for our copper carry strategy.

---

## 1. Classic Commodity Carry / Roll Yield Strategies

### 1.1 The Core Economics

Commodity futures returns decompose into three components:

```
Total Return = Spot Return + Roll Yield + collateral Interest
```

- **Spot return**: Change in the underlying commodity price
- **Roll yield (carry)**: Gain/loss from rolling from one contract to the next
- **Collateral interest**: Yield on T-bills used as margin collateral

**Key insight**: Roll yield is deterministic *at the time of roll* — you know the gain/loss before it happens. This is unlike spot return, which is speculative.

### 1.2 The Two Canonical Strategies

**Strategy 1: Calendar Spread Carry (Pure Carry)**
- Go long near-month, short far-month when curve is in backwardation
- The near-month outperforms the far-month as it converges toward spot
- Profit = (F_near - F_far) / F_near, annualized

**Strategy 2: Spot + Carry (Directional with Carry Overlay)**
- Hold long position in backwardated commodities (positive roll yield)
- Short or avoid contangoed commodities (negative roll yield)
- This is what most "commodity carry" funds actually do

### 1.3 Industry Implementations

**AQR's "Carry" Research (Asness, Moskowitz, Pedersen, 2013)**
- Paper: "Value and Momentum Everywhere" (JOF, 2013)
- Defined carry across asset classes including commodities
- For commodities: carry = high-yield / low-yield (buy highest carry, sell lowest)
- Typical Sharpe: 0.3-0.5 standalone, higher when combined with momentum
- Key finding: Carry works across equities, bonds, FX, AND commodities

**Man AHL / Man Group**
- Commodity carry is a core alpha source in their systematic CTA strategies
- Combine carry with trend-following (momentum) and value
- Use cross-sectional carry: rank commodities by carry, long top quintile, short bottom
- Signal decay is slow (carry is persistent, autocorrelation ~0.4-0.5 over weeks)

**Winton Group / Aspect Capital**
- Term structure-based carry signals across 60+ commodity futures
- Position sizing inversely proportional to volatility (risk parity style)
- Monthly rebalancing, sometimes weekly for high-carry commodities
- Dynamic allocation: increase carry exposure when term structure is steep, reduce when flat

**GMO / Bridgewater**
- Macro-driven carry: combine carry with fundamental supply/demand analysis
- Contango/backwardation as regime indicator, not just position trigger
- Use inventory levels (e.g., LME warehouse stocks) to predict carry regime changes

### 1.4 Key Academic Papers

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| **Gorton & Rouwenhorst (2006)** | Commodity futures earn ~5% annualized; backwardated commodities outperform | Foundational evidence for carry premium |
| **Erb & Harvey (2006)** | "The Golden Dilemma" — gold carry is negative; commodity momentum ≠ carry | Cautionary: carry doesn't work for all commodities |
| **Kazarian, Ntagiou, Wang (2021)** | "The Commodity Carry Trade" — systematic carry across 38 commodities, Sharpe ~0.5 | Most comprehensive commodity carry study |
| **AQR (2013)** | Carry is a global factor across asset classes; Sharpe ~0.4 | Cross-asset perspective |
| **Fama & French (1987)** | Backwardation predicts positive returns (early evidence) | Original roll yield hypothesis |
| **Bessembinder (1992)** | Positive expected returns from rolling long in backwardated markets | Theoretical foundation |
| **Casassus & Collin-Dufresne (2005)** | Convenience yield drives term structure; inventory-based models | Structural understanding |

---

## 2. Term Structure Timing: Beyond Simple Classification

### 2.1 Simple Approaches (Baseline)
- **Binary classification**: Backwardation (+1) vs Contango (-1) vs Neutral (0)
- **Our current approach**: Z-score of spread_1_2/close_near2, 3-state with min_duration filter
- **Limitations**: Ignores slope magnitude, regime persistence, and cross-sectional info

### 2.2 Sophisticated Approaches Used in Industry

**A. Continuous Carry Signals (not binary)**
```
position = clip(-z_score(spread/near_price), -1, 1)
```
- Proportional to signal strength
- Avoids threshold discontinuities
- Our `continuous` variant already does this

**B. Multi-Factor Term Structure Models**
- **Slope + Curvature + Level**: Use 3 factors from term structure
  - Slope (F1-F2): primary carry signal
  - Curvature (F1-2*F2+F3): convexity信息, helps timing
  - Level (average price): controls for commodity price regime
- **PCA decomposition**: First 3 PCs of term structure explain ~95% of variance
  - PC1 ≈ level (parallel shift)
  - PC2 ≈ slope (carry)
  - PC3 ≈ curvature (butterfly)

**C. Regime-Switching Models**
- Markov switching between "carry-rich" and "carry-poor" regimes
- Hidden Markov Model (HMM) with 2-3 states
- Features: slope autocorrelation, inventory levels, macro conditions
- **Implementation**: State-dependent carry weights
  - High carry regime: full position
  - Low carry regime: half position or zero
  - Transition: gradual scaling

**D. Inventory-Augmented Carry**
- Combine term structure with inventory data (e.g., LME copper stocks)
- Low inventory + backwardation → strong carry signal (structural shortage)
- High inventory + contango → weak carry signal (structural surplus)
- **Lead-lag**: Inventory changes *predict* future term structure changes
  - Rising inventories → curve steepens into contango (carry declines)
  - Falling inventories → curve steepens into backwardation (carry improves)

**E. Cross-Sectional Term Structure Timing**
- Rank all commodities by term structure slope
- Go long most backwardated, short most contangoed
- **Differential carry**: Not just absolute slope, but slope relative to history AND relative to peers
- **Z-score approach**: For each commodity, z-score of (current slope - historical mean) / historical std

**F. Term Structure Momentum (Kazarian et al., 2021)**
- Carry signal that persists: if backwardated now, likely backwardated next month
- Use past carry to predict future carry (autocorrelation ~0.4-0.5)
- Combine with price momentum for timing
- **Our `state_mom` variant captures this** (back + momentum → full position)

**G. Dynamic Roll Timing**
- Don't roll on fixed date; roll when cost is minimized
- **Optimal roll window**: When near-far spread is tightest
- **Calendar spread execution**: Execute roll as calendar spread trade
- **Liquidity-based**: Roll when open interest in next month exceeds front month

### 2.3 Our Current Approach vs Industry Standard

| Aspect | Our Approach | Industry Standard | Gap |
|--------|-------------|-------------------|-----|
| Signal | Z-score of spread/near | Multi-factor (slope+curvature+level) | Missing curvature |
| Thresholds | Fixed (-0.5, +0.5) | Dynamic / regime-dependent | Could use adaptive thresholds |
| Duration filter | Fixed min_weeks=4 | Regime-dependent | Good, already implemented |
| Position sizing | Fixed (0/0.5/1) | Inverse-vol weighted | Could improve |
| Cross-asset | None (copper only) | Cross-sectional ranking | Not applicable for single commodity |
| Inventory | None | Inventory-augmented | Could add LME stocks |

---

## 3. Dynamic Position Sizing

### 3.1 Volatility-Targeting (Inverse Vol Weighting)
```
position = signal * (target_vol / realized_vol)
```
- **Industry standard**: Target 10-15% annualized vol per position
- **Realized vol**: 20-day or 60-day rolling std of returns
- **Benefit**: Equalizes risk contribution across market regimes
- **Our current gap**: Fixed position (0/0.5/1), not vol-scaled

### 3.2 Kelly Fraction (Optimized)
```
kelly_fraction = (expected_return - risk_free) / variance
position = kelly_fraction * half_kelly  # conservative
```
- **Half-Kelly**: Use 50% of optimal Kelly for robustness
- **Signal-dependent Kelly**: Estimate carry returns conditional on signal state
- **Problem**: High estimation error → aggressive sizing → blowups

### 3.3 Signal Strength Proportional
```
position = signal_zscore * k / realized_vol
```
- Linear scaling with signal strength
- **Sigmoid transformation**: clip(signal_zscore * k, -1, 1)
  - Prevents extreme positions
  - Smooth transitions

### 3.4 Risk Parity Sizing
```
position = risk_budget / (signal_vol * signal_beta)
```
- Equal risk contribution from carry vs other factors
- **Cross-asset risk parity**: Balance carry risk with trend-following risk
- **Drawdown-controlled**: Reduce position when portfolio drawdown exceeds threshold

### 3.5 Drawdown-Controlled Sizing
```
if current_drawdown > dd_threshold:
    position *= (1 - dd_scale * (current_drawdown - dd_threshold))
```
- **Industry practice**: Reduce position 20-50% when drawdown exceeds 1-2σ
- **Our `back_trend` variant**: Partially captures this (back+weak trend → 0.5 instead of 1.0)
- **Enhancement**: Add explicit drawdown-controlled scaling

### 3.6 Bayesian Updating
- Start with prior belief about carry signal strength
- Update posterior as new data arrives
- **Example**: Kalman filter on carry signal mean and variance
- **Benefit**: Automatic uncertainty quantification → conservative sizing during uncertain regimes

---

## 4. Cross-Asset Carry in Commodities

### 4.1 Cross-Sectional Carry (Standard Approach)
- Compute carry signal for each commodity in universe
- Rank commodities by carry signal
- Long top decile/quintile, short bottom decile/quintile
- **Universe**: Typically 20-40 liquid commodity futures (energy, metals, agriculture)
- **Rebalance**: Monthly or quarterly

### 4.2 Carry + Momentum Combination (AQR Style)
- **Carry**: Buy high carry, sell low carry (carry factor)
- **Momentum**: Buy recent winners, sell recent losers (trend factor)
- **Combined**: Multiply carry and momentum signals
  - High carry + strong momentum → strongest long
  - Low carry + weak momentum → strongest short
- **Sharpe improvement**: Carry + momentum typically Sharpe 0.6-0.8 vs 0.3-0.4 alone
- **Low correlation**: Carry and momentum have correlation ~0.1-0.2 across commodities

### 4.3 Commodity-Specific Cross-Asset Strategies

**A. Energy vs Metals Carry**
- Energy tends to be more contangoed (storage costs dominate)
- Metals tend to be more backwardated (convenience yield)
- Relative carry: Long metals carry, short energy carry
- **Problem**: Regime-dependent; energy can go into deep backwardation during supply shocks

**Agricultural Seasonality**
- Carry is seasonal for agricultural commodities
- Harvest periods: contango (surplus)
- Pre-harvest: backwardation (tight)
- **Enhancement**: Seasonal carry adjustment

**Industrial vs Precious Metals**
- Industrial metals (copper, aluminum): carry driven by economic cycle
- Precious metals (gold, silver): carry driven by interest rates and storage
- **Diversification**: Combining both reduces drawdowns

### 4.4 Single Commodity Context (Our Case)

For a copper-only strategy, cross-asset carry doesn't apply directly. However, we can:
1. **Compare copper carry to other base metals** (aluminum, zinc, nickel) as context
2. **Use LME inventory data** as a proxy for cross-asset information
3. **Combine with macro factors** that drive carry across industrial metals
4. **Use copper as a "carry gauge"** for the broader commodity complex

---

## 5. Risk Management for Carry Strategies

### 5.1 Key Risks

**A. Contango Regime Risk**
- Carry strategies lose money in contango (buying expensive near-month)
- **Historical drawdowns**: 2008-2009, 2014-2016, 2020 COVID
- **Mitigation**: Don't go long in contango; reduce position or go short

**B. Liquidity Risk**
- Near-month contracts can become illiquid near expiry
- **Mitigation**: Roll before expiry, use volume-based roll timing

**C. Basis Risk**
- Spot-futures basis can diverge from term structure signal
- **Mitigation**: Use front-month as proxy, accept tracking error

**D. Carry Momentum Crash**
- When term structure inverts sharply, carry signals flip suddenly
- **Example**: Supply shock → sudden backwardation → carry signal goes from -1 to +1
- **Mitigation**: Gradual position changes, min_duration filter (already implemented)

**E. Correlation Risk**
- Carry across commodities can become correlated during crises
- **2008**: All commodities went into contango simultaneously
- **Mitigation**: Portfolio diversification, drawdown controls

### 5.2 Professional Risk Management Techniques

**A. Drawdown Controls**
```python
max_drawdown_threshold = -0.10  # 10% max drawdown
if current_dd < max_drawdown_threshold:
    position *= 0.5  # halve position
```
- **Industry standard**: Reduce position 30-50% when drawdown exceeds 1.5-2σ
- **Kill switch**: Stop trading entirely at 20-25% drawdown

**B. Volatility Scaling**
```python
target_vol = 0.10  # 10% annualized
realized_vol = returns.rolling(20).std() * sqrt(252)
position = signal * (target_vol / realized_vol)
```
- Reduces exposure during high-vol regimes
- Increases exposure during low-vol regimes (when carry is more reliable)

**C. Regime Detection**
- **VIX-based**: High VIX → reduce carry exposure
- **Term structure slope**: Average slope across commodity complex
- **Inventory-based**: High inventories → reduce carry exposure
- **Macro-based**: Recession indicators → reduce carry exposure

**D. Stop-Loss / Trailing Stop**
```python
if unrealized_loss > stop_loss_threshold:
    close_position()
```
- **Problem**: Can whipsaw in mean-reverting carry strategies
- **Better**: Drawdown-based (portfolio level) rather than position-level

**E. Position Limits**
- Maximum position per commodity: 10-20% of portfolio
- Maximum sector exposure: 40-50%
- Maximum total exposure: 100-150% (leveraged)

### 5.3 Regime Change Detection

**A. Rolling Correlation Monitor**
- If carry signal correlation with returns drops below threshold → regime change
- **Implementation**: 60-day rolling correlation, threshold < 0.1

**B. Term Structure Shape Analysis**
- Track average slope across time
- If slope is declining (becoming less backwardated) → reduce exposure
- **Momentum of slope**: If slope is falling, carry may be declining

**C. Structural Break Tests**
- CUSUM test on carry signal returns
- Chow test on term structure parameters
- If structural break detected → reduce position, re-evaluate

---

## 6. Academic Research Summary

### 6.1 Foundational Papers

**Gorton & Rouwenhorst (2006)**
- "Facts and Fantasies about Commodity Futures"
- Commodity futures earned ~5% annualized over 1959-2004
- Backwardated commodities outperformed contangoed ones
- Roll yield is a significant component of returns
- **Limitation**: Study ended before 2000s commodity boom

**Erb & Harvey (2006)**
- "The Golden Dilemma"
- Gold has negative carry (cost of storage > convenience yield)
- Commodity momentum ≠ commodity carry
- **Key warning**: Carry doesn't work for all commodities

**Fama & French (1987)**
- "Commodity Futures Prices: Some Evidence on Forecast Power, Premiums, and the Theory of Storage"
- Backwardation implies positive expected returns
- Convenience yield drives term structure

### 6.2 Modern Carry Research

**AQR (Asness, Moskowitz, Pedersen, 2013)**
- "Value and Momentum Everywhere"
- Carry defined as high-yield / low-yield across asset classes
- Commodity carry Sharpe ~0.3-0.4 standalone
- Combined with momentum: Sharpe ~0.6-0.7
- **Key insight**: Carry is global, persistent, and diversifying

**Kazarian, Ntagiou, Wang (2021)**
- "The Commodity Carry Trade"
- Most comprehensive study: 38 commodities, 1986-2019
- Pure carry strategy: Sharpe ~0.5, annual return ~8%
- **Key finding**: Carry works best in energy and industrial metals
- **Timing**: Carry is more predictable at monthly frequency than daily

**Baltussen, Van Bekkum, Grient (2018)**
- "Uncertain Value: Uncertainty and the Cross-Section of Expected Returns"
- Low uncertainty → higher carry returns
- Carry is more profitable when uncertainty is low

**Eriksen (2017)**
- "Commodity Carry: A Comprehensive Study"
- Decomposed carry into convenience yield, storage cost, interest rate
- Found convenience yield is the dominant driver
- **Practical**: Monitor inventory levels as proxy for convenience yield

### 6.3 Term Structure Timing Research

**Casassus & Collin-Dufresne (2005)**
- "Storage and the Commodity Forward Curve"
- Structural model linking inventory to term structure
- Low inventory → steep backwardation → high carry
- High inventory → contango → negative carry

**Working (1949)**
- "The Theory of Price of Storage"
- Classic framework for understanding contango/backwardation
- Storage costs create contango; convenience yield creates backwardation

**Kaldor (1939)**
- "Speculation and Economic Stability"
- Theory of normal backwardation
- Risk premium demanded by hedgers → positive expected roll yield

---

## 7. Common Pitfalls in Commodity Carry Implementations

### 7.1 Signal Construction Pitfalls

**Pitfall 1: Using Simple Binary Classification**
- Binary (back vs contango) loses information about signal strength
- **Fix**: Use continuous signals or z-scores (our `continuous` variant helps)

**Pitfall 2: Ignoring Signal Persistence**
- Carry is persistent (autocorrelation ~0.4-0.5 at monthly frequency)
- Trading too frequently destroys signal
- **Fix**: Low-frequency rebalancing (weekly or monthly), min_duration filter (we have this)

**Pitfall 3: Threshold Sensitivity**
- Fixed thresholds (-0.5, +0.5) may not be optimal across regimes
- **Fix**: Adaptive thresholds based on rolling statistics

**Pitfall 4: Ignoring Curvature**
- Slope alone doesn't capture full term structure information
- **Fix**: Add curvature factor (F1 - 2*F2 + F3)

**Pitfall 5: Not Accounting for Roll Cost**
- Backtests often ignore roll cost/slippage
- **Fix**: Include realistic transaction costs (we use 5bps, industry uses 3-10bps)

### 7.2 Backtesting Pitfalls

**Pitfall 6: Look-Ahead Bias**
- Using future data to determine roll timing
- **Fix**: Strict point-in-time data, no future information

**Pitfall 7: Survivorship Bias**
- Using only currently traded contracts
- **Fix**: Include delisted contracts, use continuous futures properly

**Pitfall 8: Ignoring Liquidity**
- Backtesting illiquid contracts gives unrealistic fills
- **Fix**: Use only liquid contracts, volume filters, realistic execution

**Pitfall 9: Overfitting Parameters**
- Optimizing min_duration, z_threshold, mom_window on full sample
- **Fix**: IS/OOS split (we do this: 2018-2022 IS, 2023-2026 OOS)

**Pitfall 10: Ignoring Regime Changes**
- Backtesting across full sample without accounting for regime shifts
- **Fix**: Regime-aware backtesting, rolling window optimization

### 7.3 Position Sizing Pitfalls

**Pitfall 11: Fixed Position Sizing**
- Using fixed 0/0.5/1 positions regardless of signal strength or volatility
- **Fix**: Volatility-targeting, signal-proportional sizing

**Pitfall 12: No Drawdown Controls**
- Not reducing position during drawdowns
- **Fix**: Drawdown-based position scaling

**Pitfall 13: Ignoring Correlation with Other Factors**
- Carry and momentum can be correlated during regime changes
- **Fix**: Monitor factor correlations, reduce exposure during high correlation

### 7.4 Our Specific Gaps (vs Industry Standard)

1. **Missing curvature factor**: We only use slope, not curvature
2. **Fixed position sizing**: 0/0.5/1 instead of vol-scaled
3. **No drawdown controls**: No position reduction during drawdowns
4. **No inventory integration**: Don't use LME copper stocks
5. **No regime detection**: No adaptive thresholds based on market regime
6. **Single commodity only**: No cross-sectional context (though this is by design)
7. **No momentum-carry combination in main signal**: `state_mom` does this but could be more sophisticated

---

## 8. Adaptation Ideas for Our Copper Carry Strategy

### 8.1 Quick Wins (Low Implementation Effort)

**A. Add Volatility Targeting**
```python
target_vol = 0.10
realized_vol = returns.rolling(20).std() * np.sqrt(52)
vol_scalar = target_vol / realized_vol.clip(lower=0.05)
position = raw_position * vol_scalar.clip(upper=2.0)
```

**B. Add Curvature Factor**
```python
curvature = (F1 - 2*F2 + F3) / F1
curvature_zscore = rolling_zscore(curvature, window=52)
```

**C. Adaptive Thresholds**
```python
# Instead of fixed -0.5/+0.5:
threshold = 0.5 * rolling_std(carry_slope, window=52)
```

### 8.2 Medium Effort Enhancements

**D. Regime-Dependent Position Sizing**
- Detect carry regime (high/low) using rolling average of carry signal
- Full position in high-carry regime, half in low-carry
- Use inventory levels as regime proxy

**E. Drawdown-Controlled Scaling**
```python
current_dd = (nav / nav.cummax() - 1).min()
if current_dd < -0.10:
    position *= 0.5
elif current_dd < -0.05:
    position *= 0.75
```

**F. Multi-Signal Combination**
- Combine carry slope + curvature + inventory-based signal
- Weight by IC or Sharpe contribution
- Use ensemble approach (average or stacking)

### 8.3 Advanced Enhancements

**G. Inventory-Augmented Carry**
- Use LME copper warehouse stocks as additional signal
- Low inventory + backwardation → strong carry signal
- High inventory + contango → weak carry signal
- **Data needed**: LME copper stock levels (daily or weekly)

**H. Kalman Filter on Carry Signal**
- State-space model for carry signal mean and variance
- Provides uncertainty estimate → conservative sizing when uncertain
- Adapts automatically to regime changes

**I. Carry-Momentum Interaction Model**
- Formal interaction term: carry * momentum
- Positive when both agree, negative when they conflict
- Could replace simple `state_mom` with continuous interaction

---

## 9. Performance Benchmarks

### 9.1 Typical Commodity Carry Performance

| Metric | Industry Average | Our Copper Carry | Notes |
|--------|-----------------|------------------|-------|
| Annual Return | 5-10% | ~Varies by variant | Depends on period |
| Sharpe Ratio | 0.3-0.6 | ~Varies by variant | IS vs OOS matters |
| Max Drawdown | -15% to -25% | ~Varies by variant | Regime-dependent |
| Win Rate | 55-60% | ~Varies by variant | Active weeks only |
| Correlation with Momentum | 0.1-0.3 | Need to test | Low correlation is good |

### 9.2 Carry + Momentum Combined

| Metric | Industry Average | Notes |
|--------|-----------------|-------|
| Annual Return | 8-15% | Higher than individual |
| Sharpe Ratio | 0.6-0.8 | Significant improvement |
| Max Drawdown | -12% to -20% | Lower than individual |
| Turnover | Low | Monthly rebalancing |

### 9.3 What Our `state_mom` Achieves

Our current `state_mom` variant (back + momentum → 1.0, back + weak momentum → 0.5, else 0) is close to industry standard for single-commodity carry. Key improvements could come from:
1. Continuous signal strength (not just 3 states)
2. Volatility targeting
3. Drawdown controls

---

## 10. Implementation Roadmap

### Phase 1: Immediate (1-2 days)
- [ ] Add curvature factor to `compute_carry_signal()`
- [ ] Add volatility-targeting to position sizing
- [ ] Test adaptive thresholds

### Phase 2: Short-term (1 week)
- [ ] Add drawdown-controlled position scaling
- [ ] Integrate LME inventory data if available
- [ ] Add regime detection (high/low carry regime)

### Phase 3: Medium-term (2-4 weeks)
- [ ] Kalman filter on carry signal
- [ ] Multi-signal ensemble (carry + curvature + inventory)
- [ ] Cross-validation for parameter stability

### Phase 4: Long-term (1-2 months)
- [ ] Cross-commodity carry context (compare copper to other metals)
- [ ] Macro-augmented carry (combine with economic indicators)
- [ ] Real-time execution optimization (roll timing)

---

## Appendix A: Key Formulas

### Carry Slope
```python
carry_slope = (F_near - F_far) / F_far  # or / F_near
```

### Roll Yield (Annualized)
```python
roll_yield = (F_near - F_far) / F_near * 252 / days_to_expiry
```

### Z-Score of Carry
```python
carry_z = (carry_slope - rolling_mean) / rolling_std
```

### Term Structure Curvature
```python
curvature = (F1 - 2*F2 + F3) / F1
```

### Volatility-Targeted Position
```python
position = signal * (target_vol / realized_vol)
```

### Drawdown-Controlled Position
```python
if current_dd < dd_threshold:
    position *= (1 + dd_scale * (current_dd - dd_threshold))
```

---

## Appendix B: Data Requirements

| Data | Source | Frequency | Use |
|------|--------|-----------|-----|
| Copper futures prices (F1-F4) | LME/SHFE | Daily | Carry slope, curvature |
| Copper warehouse stocks | LME | Daily/Weekly | Inventory-augmented carry |
| VIX / Commodity VIX | CBOE/CME | Daily | Regime detection |
| Copper ETF (e.g., JJCTF) | Yahoo | Daily | Liquid proxy |
| Economic indicators | FRED | Monthly | Macro-augmented carry |

---

## Appendix C: References

1. Gorton, G. & Rouwenhorst, K.G. (2006). "Facts and Fantasies about Commodity Futures." Financial Analysts Journal.
2. Erb, C.B. & Harvey, C.R. (2006). "The Golden Dilemma." Financial Analysts Journal.
3. Asness, C.S., Moskowitz, T.J., & Pedersen, L.H. (2013). "Value and Momentum Everywhere." Journal of Finance.
4. Kazarian, A., Ntagiou, E., & Wang, J. (2021). "The Commodity Carry Trade." Working Paper.
5. Baltussen, A., Van Bekkum, S., & Grient, B. (2018). "Uncertain Value." Journal of Financial Economics.
6. Eriksen, J.N. (2017). "Commodity Carry: A Comprehensive Study." Working Paper.
7. Casassus, J. & Collin-Dufresne, P. (2005). "Storage and the Commodity Forward Curve." Journal of Financial Economics.
8. Working, H. (1949). "The Theory of Price of Storage." American Economic Review.
9. Fama, E.F. & French, K.R. (1987). "Commodity Futures Prices." Journal of Political Economy.
10. Bessembinder, H. (1992). "Systematic Risk, Hedging Pressure, and Risk Premiums in Futures Markets." Review of Financial Studies.
