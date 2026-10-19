"""Diagnostics that are not part of the tradable strategy.

The signal-decay event study deliberately looks at FUTURE prices: it measures
what happened after the signal, to check whether the hypothesised reversal
exists at all, independent of portfolio construction and costs. It is never
used to choose parameters.
"""

from __future__ import annotations

import pandas as pd

from .signals import decile_signal, past_return


def signal_decay(prices: pd.DataFrame, lookback: int, horizons: int = 10,
                 quantile: float = 0.10, start=None, end=None) -> pd.DataFrame:
    """Average cumulative forward return of the loser and winner deciles.

    For each formation date t in [start, end] and each k in 1..horizons:
    forward return of stock i = P_{t+k} / P_t - 1, averaged equally across the
    decile, minus the equal-weight average of all stocks with data (so the
    numbers are market-adjusted). Then averaged over formation dates.
    """
    sig = decile_signal(past_return(prices, lookback), quantile=quantile)
    rows = []
    for k in range(0, horizons + 1):
        fwd = prices.shift(-k) / prices - 1
        univ = fwd.mean(axis=1)
        losers = fwd.where(sig == 1).mean(axis=1) - univ
        winners = fwd.where(sig == -1).mean(axis=1) - univ
        frame = pd.DataFrame({"losers": losers, "winners": winners}).loc[start:end].dropna()
        rows.append({"horizon": k, "losers": frame["losers"].mean(),
                     "winners": frame["winners"].mean(), "n_dates": len(frame)})
    out = pd.DataFrame(rows).set_index("horizon")
    out["losers_minus_winners"] = out["losers"] - out["winners"]
    return out
