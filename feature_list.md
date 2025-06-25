# Feature Definitions by Strategy

This document categorizes all features into three distinct groups for building specialized models. Features in the "Common" category can be combined with features from either of the two primary strategies.

## Momentum Features
*(Signals that suggest the current price trend will continue)*

### Trend & Moving Averages
*   MA_5_above_MA_13_1min: 5-period MA above 13-period MA on 1-minute chart
*   MA_8_above_MA_21_5min: 8-period MA above 21-period MA on 5-minute chart
*   Price_above_MA_20_5min: Current price above 20-period MA on 5-minute chart
*   EMA_12_above_EMA_26_5min: 12-period EMA above 26-period EMA (MACD signal)
*   Price_above_VWAP_5min: Current price above VWAP on 5-minute chart
*   VWAP_above_MA_20_5min: VWAP above 20-period MA (institutional vs retail sentiment)
*   Price_velocity_positive_5min: Price velocity positive over 5-minute lookback
*   Price_acceleration_positive_5min: Price acceleration positive over 5-minute period
*   RSI_above_50_5min: RSI above 50 (bullish momentum)
*   Stoch_K_above_D_5min: %K line above %D line (bullish crossover)
*   MACD_bullish_crossover_5min: MACD line crosses above signal line
*   MACD_bearish_crossover_5min: MACD line crosses below signal line
*   MACD_histogram_positive: MACD histogram above zero
*   MACD_above_zero_line: MACD line above zero (trending up)
*   ROC_positive_5min: Rate of Change positive over 5-minute period
*   Plus_DI_above_Minus_DI: +DI above -DI (bullish trend)
*   ADX_rising_5bars: ADX increasing over last 5 bars (strengthening trend)
*   CCI_zero_line_cross_up: CCI crossing above zero line



## Mean Reversion Features
*(Signals that suggest the current price trend will reverse)*

### Overbought/Oversold Oscillators
*   RSI_overbought_70_5min: RSI above 70 on 5-minute chart
*   RSI_oversold_30_5min: RSI below 30 on 5-minute chart
*   Stoch_overbought_80_5min: Stochastic above 80 on 5-minute chart
*   Stoch_oversold_20_5min: Stochastic below 20 on 5-minute chart
*   Price_above_BB_upper_5min: Price above upper Bollinger Band (overbought)
*   Price_below_BB_lower_5min: Price below lower Bollinger Band (oversold)
*   WilliamsR_overbought_minus20: Williams %R above -20 (overbought)
*   CCI_overbought_above_100: CCI above +100 (overbought)
*   CCI_oversold_below_minus100: CCI below -100 (oversold)


## Common Features
*(Contextual features that apply to all strategies)*

### Volatility Context
*   ADX_strong_trend_above_25: ADX above 25 (strong trend present filter)
*   BB_squeeze_5min: Bollinger Bands width below 20-period average (low volatility)
*   BB_expansion_5min: Bollinger Bands width above 20-period average (high volatility)
*   High_ATR_5min: ATR above 20-period average (high volatility period)
*   Low_ATR_5min: ATR below 20-period average (consolidation period)

### Volume Context
*   Volume_above_20MA_5min: Current volume above 20-period volume MA
*   Volume_increasing_3bars: Volume increasing for 3 consecutive bars


### Time-Based Context
*   IS_MORNING_SESSION (e.g., 9:15 - 11:30)
*   IS_LUNCH_SESSION (e.g., 11:30 - 13:30)
*   IS_AFTERNOON_SESSION (e.g., 1:30 - 15:30)
*   DAY_OF_WEEK (One-hot encoded)


## To be integrated in future
*   Price within 1% of last 5 day high
*   Price within 1% of last 5 day low
*   Price within 1% of last 15 day high
*   Price within 1% of last 15 day low
*   Price within 1% of last 30 day high
*   Price within 1% of last 30 day low
*   Price within 1% of last 60 day high
*   Price within 1% of last 60 day low
*   Price within 1% of last 180 day high
*   Price within 1% of last 180 day low
*   Price above 101% of last 5 day high
*   Price above 101% of last 5 day low
*   Price above 101% of last 15 day high
*   Price above 101% of last 15 day low
*   Price above 101% of last 30 day high
*   Price above 101% of last 30 day low
*   Price above 101% of last 60 day high
*   Price above 101% of last 60 day low
*   Price above 101% of last 180 day high
*   Price above 101% of last 180 day low
*   Price below 99% of last 5 day high
*   Price below 99% of last 5 day low
*   Price below 99% of last 15 day high
*   Price below 99% of last 15 day low
*   Price below 99% of last 30 day high
*   Price below 99% of last 30 day low
*   Price below 99% of last 60 day high
*   Price below 99% of last 60 day low
*   Price below 99% of last 180 day high
*   Price below 99% of last 180 day low

*   High_above_prev_high_5bars: Current high above highest high in last 5 bars
*   High_above_prev_high_3bars: Current high above highest high in last 3 bars
*   Price_move_above_1p5_ATR: Price movement exceeds 1.5x ATR (significant move)
*   RSI_bullish_divergence_5min: RSI making higher lows while price makes lower lows
*   OBV_bullish_divergence: On-Balance Volume diverging bullishly from price
*   Volume_above_avg_at_VWAP_cross: Above-average volume during VWAP crossover
*   Relative_volume_above_1p5: Relative volume above 1.5x average
*   Volume_confirmation_breakout: Above-average volume on price breakout
*   Volume_spike_3x: Volume exceeds 3x average volume
