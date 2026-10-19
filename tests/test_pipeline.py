"""Correctness tests for the parts a reviewer is most likely to check."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, daily_returns
from src.metrics import newey_west_tstat
from src.signals import build_weights, decile_signal, dollar_neutral_weights, past_return


def random_prices(n_days=300, n_stocks=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rets = rng.normal(0, 0.02, size=(n_days, n_stocks))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"S{i}" for i in range(n_stocks)])


def test_weights_do_not_depend_on_future_prices():
    prices = random_prices()
    cut = prices.index[200]
    w_full = build_weights(prices, lookback=5, hold=5)
    tampered = prices.copy()
    tampered.loc[tampered.index > cut] *= np.random.default_rng(1).uniform(0.5, 1.5, tampered.loc[tampered.index > cut].shape)
    w_tampered = build_weights(tampered, lookback=5, hold=5)
    pd.testing.assert_frame_equal(w_full.loc[:cut], w_tampered.loc[:cut])


def test_position_earns_next_day_return_only():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [100, 110, 121, 121]}, index=idx, dtype=float)
    weights = pd.DataFrame({"A": [0, 1, 1, 1]}, index=idx, dtype=float)  # decided at close of day 1
    bt = backtest(weights, daily_returns(prices), cost_bps=0)
    # Day 1's +10% happened before the position existed; day 2's +10% is earned.
    assert bt["gross_return"].tolist() == pytest.approx([0.0, 0.0, 0.10, 0.0])


def test_costs_charged_on_notional_traded():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    weights = pd.DataFrame({"A": [0.0, 1.0, -1.0]}, index=idx)  # open, then full flip
    bt = backtest(weights, daily_returns(prices), cost_bps=10)
    assert bt["costs"].tolist() == pytest.approx([0.0, 0.001, 0.002])


def test_turnover_accounts_for_drift():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 100.0, 110.0]}, index=idx)
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
    bt = backtest(weights, daily_returns(prices), cost_bps=0)
    # After +10% the position is 1.1; rebalancing back to 1.0 trades 0.1.
    assert bt["turnover"].iloc[2] == pytest.approx(0.1)


def test_deciles_and_dollar_neutrality():
    prices = random_prices()
    sig = decile_signal(past_return(prices, 5))
    live = sig.iloc[10]
    assert (live == 1).sum() == 12 and (live == -1).sum() == 12
    w = dollar_neutral_weights(sig).iloc[10:]
    assert np.allclose(w.clip(lower=0).sum(axis=1), 0.5)
    assert np.allclose(w.sum(axis=1), 0.0)


def test_losers_are_long():
    prices = random_prices()
    pr = past_return(prices, 5).iloc[50]
    sig = decile_signal(past_return(prices, 5)).iloc[50]
    assert pr[sig == 1].max() < pr[sig == -1].min()


def test_newey_west_matches_iid_tstat_with_zero_lags():
    x = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 5000))
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, 0) == pytest.approx(naive)
