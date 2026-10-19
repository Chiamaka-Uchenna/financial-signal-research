# Methodology and design decisions

This document records every modelling choice, the formula behind it, and why it was made. Section 9 lists where the implementation deliberately departs from the original build guide ([`BUILD_GUIDE.md`](BUILD_GUIDE.md)) and why. Most departures fix code in the guide that would have produced wrong numbers.

## 1. Universe and data

- **Constituents.** The table with id `constituents` on Wikipedia's *List of S&P 500 companies* page, scraped on 2026-10-19: 503 symbols. Index symbols are mapped to Yahoo symbols by replacing `.` with `-` (BRK.B → BRK-B). The CSV is committed and reused on every run, so the universe never changes silently.
- **Prices.** `yfinance.download(..., auto_adjust=True)`. The close is adjusted for splits and dividends, so close-to-close returns are total returns. Volume is downloaded and stored but not used.
- **Download robustness.** Requests go out in chunks of 50 with up to 3 retries per chunk. Any ticker missing from a batch is then retried on its own. In this build the first pass silently dropped 9 tickers under Yahoo rate limiting, including ZTS, TGT, AKAM and SRE. The single-ticker retry recovered everything except 5 symbols with no data in the window (FDXF, HONA, Q, SNDK, VYLR).
- **Date range.** Data is downloaded from 2009-11-01 to 2025-01-01. The yfinance `end` is *exclusive*, so this includes 2024-12-31. The two months of 2009 exist only so that a 20-day lookback plus a 5-day ramp-up are live on 2010-01-04.

### Cleaning (`src/data.py: clean_prices`)

| Rule | Reason |
|---|---|
| Non-positive prices → missing | They are not valid prices |
| Forward-fill at most 5 consecutive missing days | Bridges short data gaps without inventing long runs of zero returns |
| Never fill before a stock's first or after its last observation | Filling those would create fake flat prices before listing or after the data ends |
| **No column is dropped for incomplete history** | Dropping stocks with <90% coverage over 2010–2024 uses future knowledge (whether the stock still has data in 2024) to define the 2010 universe |
| Extreme returns kept (13 days with \|r\| > 50%) | They look like real corporate events; removing them would be an ex-post judgement |

## 2. Signal

For lookback *L*, at each close *t*:

    past_ret[t, i] = P[t, i] / P[t − L, i] − 1

This is NaN if either price is missing. Stocks are ranked across the cross-section with `rank(axis=1, pct=True)`, ignoring NaN.

- `signal = +1` if `rank ≤ 0.10` (losers, long)
- `signal = −1` if `rank > 0.90` (winners, short)
- otherwise 0

The strict `>` on the top side makes the two legs equal in size (for 470 names, 47 each). Days with fewer than 50 valid stocks would be flat. The minimum observed was 415, so this never triggered.

## 3. Portfolio construction

**Dollar-neutral, equal-weighted legs** (`dollar_neutral_weights`):

    w[t, i] = +0.5 / N_long(t)   if long
              −0.5 / N_short(t)  if short

Gross exposure is 1 and net exposure is exactly 0, even when the leg counts differ.

**5-day holding period via overlapping tranches** (`holding_period_weights`):

    W[t] = (1/5) · Σ_{k=0..4} w[t − k]

Equivalently, capital is split into 5 tranches. Each day the oldest tranche is closed and a new one is formed from today's signal, and each tranche lives 5 days. This is the standard Jegadeesh-Titman (1993) construction. It gives a smooth daily return series and actually implements the "hold 5 days" in the hypothesis. During the first 4 days of the sample the book is partially invested (ramp-up). These days fall in the 2009 warm-up period.

## 4. Backtest (`src/backtest.py`)

**Timing.** Weights `W[t]` are computed from information available at the close of *t* and traded at that close. They earn the next day's return.

    r[t, i]  = P[t, i] / P[t−1, i] − 1
    gross[t] = Σ_i W[t−1, i] · r[t, i]

This is implemented as `weights.shift(1) * returns`, and the test `test_position_earns_next_day_return_only` checks it.

**Turnover and costs.** Between closes, yesterday's book drifts with prices. Just before rebalancing, the holding in stock *i* is `W[t−1, i]·(1 + r[t, i])`. The notional actually traded at the close of *t* is

    traded[t] = Σ_i | W[t, i] − W[t−1, i]·(1 + r[t, i]) |
    cost[t]   = traded[t] · c / 10,000
    net[t]    = gross[t] − cost[t]

where *c* is the cost in bps per unit of notional traded, charged on every buy and every sell. Opening a position of size *w* costs *w·c*. Flipping from +*w* to −*w* trades 2*w* and costs twice as much. Turnover is reported as `traded[t]`, the total notional traded as a fraction of capital. It is not halved. Missing returns on held names are treated as 0. They are rare because of the 5-day forward fill.

**Why 10 bps is the base case.** It roughly covers half-spread plus commission plus modest impact for liquid US large caps for a non-HFT trader. Sensitivity at 0, 5, 20 and 50 bps is reported. Borrow fees for the short leg are not included.

**Break-even cost** is the cost at which net annualised return is zero:

    c* = gross_ann_return / (mean(traded) · 252) · 10,000  bps

## 5. Train/test protocol

| | Dates | Purpose |
|---|---|---|
| Train | 2010-01-04 to 2019-12-23 | Choose *L* ∈ {1, 3, 5, 10, 20} |
| Embargo | 2019-12-24 to 2019-12-31 (last 5 trading days of 2019) | Left out of the train evaluation |
| Test | 2020-01-02 to 2024-12-31 | One evaluation of the frozen choice |

- **Physical separation.** `--stage train` slices the price panel to dates before 2020-01-01 *before* computing returns, signals or anything else. Nothing from the test period exists in memory during selection. The chosen *L*, the criterion, the visible date range, the SHA-256 of the data file and a UTC timestamp are written to `results/selection.json`. `--stage test` refuses to run without that file and never modifies it.
- **Selection criterion, fixed before running:** highest annualised Sharpe net of 10 bps on the train window. Holding period (5), decile cut-off (10%) and costs were not searched over. The total search is 5 configurations.
- **About the embargo.** The signals use only past prices and nothing is fitted, and each day's P&L uses only prices up to that day. So there are no overlapping labels that could leak across the boundary the way they do in ML label construction (López de Prado 2018, ch. 7). The 5-day embargo is kept as a conservative safeguard and to match the stated protocol. In practice its effect on this strategy is negligible.
- **Computing test signals on the full panel.** The test stage builds weights from the full 2009–2024 panel and then keeps only test dates. Because every signal is causal, this is identical to computing day by day. It also avoids throwing away the first *L* days of 2020 as warm-up. The test `test_weights_do_not_depend_on_future_prices` scrambles all prices after a cut-off date and checks that weights up to that date do not change.
- **Test runs.** The test stage was run twice. The second run happened only after a plot-margin change (a clipped label), with the same frozen selection and the same data file. The numbers were identical. No parameter, rule or data choice was changed after test results were seen.

## 6. Metrics (`src/metrics.py`)

| Metric | Definition |
|---|---|
| Annualised return | mean(daily return) × 252 (arithmetic) |
| CAGR | (Π(1 + r))^(252/n) − 1 |
| Annualised volatility | std(daily return, ddof=1) × √252 |
| Sharpe | ann. return / ann. volatility; risk-free rate = 0 (the book is dollar-neutral, and cash on the margin would add roughly T-bill to both legs' collateral) |
| Newey-West t-stat | mean / HAC standard error, Bartlett kernel, 5 lags. Overlapping 5-day tranches make daily returns autocorrelated, so the naive t-stat would overstate significance |
| Max drawdown | min over t of (equity / running max − 1), on compounded returns |
| Hit rate | share of days with return > 0 |
| Avg daily turnover | mean(traded[t]) |
| Annual cost drag | mean(cost[t]) × 252 |

## 7. Signal-decay diagnostic (`src/analysis.py`)

For each formation date *t* and horizon *k* = 0…10:

    fwd[t, i, k] = P[t + k, i] / P[t, i] − 1
    losers(k)    = mean_i∈long fwd − mean_i∈all fwd
    winners(k)   = mean_i∈short fwd − mean_i∈all fwd

Both are averaged over formation dates in the period. This uses future prices **on purpose**. It answers "does the reversal exist?" without any portfolio or cost assumptions. It is run only after selection, at the selected lookback, and never feeds into a decision. The train-period decay is computed from a panel truncated at 2019-12-31, so its forward windows near the end simply shrink rather than reaching into 2020.

## 8. Tests (`tests/test_pipeline.py`)

| Test | What it guarantees |
|---|---|
| `test_weights_do_not_depend_on_future_prices` | No look-ahead anywhere in signal → weights |
| `test_position_earns_next_day_return_only` | One-day execution lag in the P&L |
| `test_costs_charged_on_notional_traded` | 10 bps on opening 1.0 = 0.001; full flip = 0.002 |
| `test_turnover_accounts_for_drift` | A +10% move then rebalance trades 0.1 |
| `test_deciles_and_dollar_neutrality` | 12/12 names on 120 stocks; legs sum to ±0.5; net 0 |
| `test_losers_are_long` | Sign convention: long = lowest past return |
| `test_newey_west_matches_iid_tstat_with_zero_lags` | HAC estimator reduces to the naive t-stat at 0 lags |

## 9. Departures from the original build guide

| # | Guide | Problem | Implemented instead |
|---|---|---|---|
| 1 | `returns = prices.pct_change().shift(-1)` **and** `held = signal.shift(1)` | Lags twice: the signal from *t−1* is matched with the return from *t* to *t+1*, a 2-day gap. That isn't look-ahead, but it misstates the strategy, and the code comments contradict each other | One lag: `W[t−1] · r[t]`, unit-tested |
| 2 | Signal recomputed and fully rebalanced every day | That is a 1-day hold, not the 5-day hold the hypothesis states | Overlapping 5-tranche portfolios |
| 3 | Weights = signal / total number of positions | Not dollar-neutral when the legs differ in size | Each leg normalised to ±0.5 |
| 4 | `dropna(axis=1, thresh=0.9·len)` | Uses future data availability to pick the universe | No completeness filter |
| 5 | Unlimited `ffill()` | Creates long runs of fake zero returns | Forward-fill ≤ 5 days, interior only |
| 6 | `end='2024-12-31'` | yfinance `end` is exclusive, so 2024-12-31 is dropped | `end='2025-01-01'` |
| 7 | Signal recomputed on `test_prices` alone; purge done with index arithmetic | Wastes the first *L* test days, and the purge code mixes up indices | Causal signals on the full panel, sliced to the test window; train stage physically truncated |
| 8 | `cost_bps` docstring says "round-trip" but it is applied per unit of |Δw| | Ambiguous | Defined as per side, per unit of notional traded |
| 9 | Turnover = Σ|Δw| / 2, ignoring drift | Understates trading | Drift-aware total notional traded |
| 10 | Naive Sharpe only | No significance test, autocorrelated returns | Newey-West t-stat with 5 lags |
| 11 | `ranks >= 0.9` on the top side | Slightly asymmetric legs | `> 0.9` |
| 12 | Template says "Best lookback selected on train: 5 days" | It was a placeholder | The rule chose 20; reported as such |
| 13 | "~1.25M observations" / CV "~2.5M" | Neither matches this build | 1,779,302 stock-day price observations |
