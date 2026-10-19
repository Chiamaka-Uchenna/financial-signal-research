"""Signal construction: cross-sectional short-term reversal.

Everything here at row t uses only prices dated <= t.
"""

from __future__ import annotations

import pandas as pd


def past_return(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Return over the last `lookback` trading days, close(t) / close(t-lookback) - 1.

    NaN if either endpoint is missing (no fill_method, so gaps are not bridged
    silently).
    """
    return prices / prices.shift(lookback) - 1


def decile_signal(past_ret: pd.DataFrame, quantile: float = 0.10,
                  min_stocks: int = 50) -> pd.DataFrame:
    """+1 for the bottom `quantile` (losers), -1 for the top (winners), else 0.

    Ranks are cross-sectional percentiles among stocks with a valid past return
    that day. Days with fewer than `min_stocks` valid names get no positions.
    """
    ranks = past_ret.rank(axis=1, pct=True)
    signal = pd.DataFrame(0, index=past_ret.index, columns=past_ret.columns, dtype="int8")
    signal = signal.mask(ranks <= quantile, 1).mask(ranks > 1 - quantile, -1)
    too_few = past_ret.notna().sum(axis=1) < min_stocks
    signal.loc[too_few] = 0
    return signal


def dollar_neutral_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Equal weight within each leg: long leg sums to +0.5, short leg to -0.5.

    Gross exposure is 1 (per unit of capital) and net exposure is 0 by
    construction, even when the two legs hold different numbers of names.
    Days where either leg is empty are flat.
    """
    longs = (signal == 1).astype(float)
    shorts = (signal == -1).astype(float)
    n_long = longs.sum(axis=1)
    n_short = shorts.sum(axis=1)
    w = 0.5 * longs.div(n_long, axis=0) - 0.5 * shorts.div(n_short, axis=0)
    both = (n_long > 0) & (n_short > 0)
    return w.where(both, 0.0).fillna(0.0)


def holding_period_weights(daily_weights: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Overlapping-portfolio implementation of an `hold`-day holding period.

    Capital is split into `hold` tranches. Each day one tranche is re-formed
    from that day's signal and kept for `hold` days (Jegadeesh-Titman style).
    The aggregate book is therefore the mean of the last `hold` daily target
    books. Uses only current and past signals.
    """
    return daily_weights.rolling(hold, min_periods=1).sum() / hold


def build_weights(prices: pd.DataFrame, lookback: int, hold: int,
                  quantile: float = 0.10) -> pd.DataFrame:
    sig = decile_signal(past_return(prices, lookback), quantile=quantile)
    return holding_period_weights(dollar_neutral_weights(sig), hold)
