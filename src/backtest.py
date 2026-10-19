"""Vectorised daily backtest with an explicit one-day lag and transaction costs.

Timing convention (per trading day t):
  1. At the close of t we observe prices up to and including t.
  2. We compute target weights w_t and trade to them at the close of t.
  3. Those weights earn the close-to-close return of day t+1.

So the P&L recorded on day t is  sum_i w_{t-1,i} * r_{t,i}  minus the cost of
the trades made at the close of t. Using w_t * r_t instead would let the
signal earn the very return that produced it (look-ahead).
"""

from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns, r_t = P_t / P_{t-1} - 1 (no gap bridging)."""
    return prices / prices.shift(1) - 1


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Run the backtest.

    Parameters
    ----------
    weights : target weights decided at the close of each day (rows = dates).
    returns : simple daily returns aligned to the same index/columns.
    cost_bps : cost per unit of notional traded, in basis points, charged on
        each side of every trade (buying 1.0 of capital costs cost_bps; a full
        flip from +w to -w trades 2w and costs twice as much).

    Turnover is drift-aware: before rebalancing at the close of t, yesterday's
    book has drifted to w_{t-1} * (1 + r_t), so the notional actually traded is
    sum_i |w_t - w_{t-1}(1 + r_t)|. Returns of held names with missing data are
    treated as 0 (rare: only gaps longer than the cleaning forward-fill limit).
    """
    weights, returns = weights.align(returns, join="left", axis=None)
    r = returns.fillna(0.0)
    held = weights.shift(1).fillna(0.0)          # w_{t-1}: the book held during day t

    gross = (held * r).sum(axis=1)
    drifted = held * (1 + r)
    traded = (weights.fillna(0.0) - drifted).abs().sum(axis=1)
    costs = traded * cost_bps / 1e4
    net = gross - costs

    return pd.DataFrame({
        "gross_return": gross,
        "costs": costs,
        "net_return": net,
        "turnover": traded,                       # notional traded / capital
        "gross_exposure": held.abs().sum(axis=1),
        "net_exposure": held.sum(axis=1),
        "n_long": (held > 0).sum(axis=1),
        "n_short": (held < 0).sum(axis=1),
    })
