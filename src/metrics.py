"""Performance metrics and a simple significance test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-statistic of the mean of `x` with Newey-West (Bartlett) HAC standard error.

    Daily returns of an overlapping-portfolio strategy are autocorrelated by
    construction, so the naive t-stat (mean / (std / sqrt(n))) overstates
    significance. HAC with lags ~= holding period corrects for that.
    """
    x = np.asarray(x.dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def compute_metrics(bt: pd.DataFrame, which: str = "net_return", hac_lags: int = 5) -> dict:
    """Metrics for one return column of a backtest frame.

    - ann_return: arithmetic mean daily return x 252
    - cagr: geometric, from the compounded equity curve
    - sharpe: ann_return / ann_vol, risk-free rate assumed 0 (the book is
      dollar-neutral, so this is close to an excess-return Sharpe anyway)
    - hit_rate: share of days with a positive return
    - avg_daily_turnover: mean notional traded per day as a fraction of capital
    - ann_cost_drag: mean daily cost x 252 (gross ann_return - net ann_return)
    """
    r = bt[which].dropna()
    n = len(r)
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    years = n / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    t_hac = newey_west_tstat(r, hac_lags)
    return {
        "days": n,
        "ann_return": float(ann_ret),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        "t_stat_nw": t_hac,
        "p_value_nw": float(2 * stats.norm.sf(abs(t_hac))) if np.isfinite(t_hac) else float("nan"),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "total_return": total,
        "avg_daily_turnover": float(bt["turnover"].mean()),
        "ann_cost_drag": float(bt["costs"].mean() * TRADING_DAYS),
        "avg_gross_exposure": float(bt["gross_exposure"].mean()),
    }


def yearly_returns(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt[["gross_return", "net_return"]].groupby(bt.index.year)
    return g.apply(lambda d: (1 + d).prod() - 1)
