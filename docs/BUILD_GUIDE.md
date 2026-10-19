# Complete Build Guide: Financial Signal Research Project

> **Note.** This is the original project brief this repository was built from, kept for reference. Its sample code contains several bugs (a double lag in the backtest, no real 5-day hold, a universe filter that uses future data, and others). They were corrected in the actual implementation; see [METHODOLOGY.md §9](METHODOLOGY.md#9-departures-from-the-original-build-guide). Inline citation markers from the brief's original source were removed because they cannot be resolved. The README template at the end of the brief has been replaced by the real [README](../README.md) with actual results.

This guide gives you everything needed to build the project claimed on your CV. The goal is a defensible, honest research repo that demonstrates you can: formulate a hypothesis, handle financial data properly, avoid look-ahead bias, account for transaction costs, and report out-of-sample results that a SIG reviewer would respect.

**Critical principle:** The project does not need to find a profitable signal. It needs to show rigorous process. A well-documented negative result (signal decays after costs) is more impressive to a quant reviewer than a suspiciously high Sharpe from a flawed backtest. Mean-reversion edges often collapse once transaction costs are included, and most home-grown backtests "lie" due to look-ahead bias and ignored costs. Your project should explicitly guard against both.

---

## Project Specification

| Element | Choice | Why |
|---|---|---|
| Universe | S&P 500 constituents (current or historical, with survivorship caveat noted) | ~500 stocks × 10+ years ≈ 1.25M+ daily observations. Matches CV claim of "~2.5M observations" if you use 15 years or include volume data. |
| Data Source | Yahoo Finance via `yfinance` | Free, reliable daily OHLCV. Use `auto_adjust=True` to handle splits/dividends. |
| Hypothesis | Short-term mean reversion: stocks with extreme negative 5-day returns tend to bounce back over the next 5 days. | Simple, well-documented in literature, easy to explain. Avoids overcomplicating. |
| Signal Construction | Cross-sectional ranking: each day, rank stocks by past 5-day return. Go long bottom decile (biggest losers), short top decile (biggest winners). Hold 5 days, equal-weighted. | Cross-sectional approach is standard in equity alpha research and avoids single-stock noise. |
| Train/Test Split | Time-based, not random. Train: 2010–2019 (parameter selection only). Test: 2020–2024 (holdout, never touched during development). | Random splits leak future information into the past. Purge/embargo if using overlapping holding periods. |
| Transaction Costs | 10 basis points (0.10%) per trade, applied to traded notional. Test sensitivity at 5bps, 10bps, 20bps. | Realistic for liquid US equities. A strategy that only works at 0bps is not a strategy. |
| Out-of-Sample Metrics | Annualised Sharpe (net of costs), hit rate, max drawdown, turnover, cumulative return. | Sharpe is the industry standard; hit rate shows directional accuracy; drawdown shows risk. |
| Deliverables | GitHub repo with README, reproducible notebook/script, results table, equity curve plot. | SIG reviewers will click the link. Make it easy to verify. |

---

## Step-by-Step Implementation

### Step 1: Setup and Data Download

```text
# requirements.txt
yfinance>=0.2.28
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
```

```python
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Get S&P 500 tickers (hardcode a list or scrape Wikipedia)
# For reproducibility, save the ticker list to a CSV in the repo
sp500_tickers = [...]  # ~500 symbols

# Download 15 years of daily data
data = yf.download(
    sp500_tickers,
    start='2010-01-01',
    end='2024-12-31',
    auto_adjust=True,  # adjusts for splits and dividends
    progress=False
)

# Extract adjusted close prices
prices = data['Close'].copy()
# Forward-fill missing values (common for stocks with gaps)
prices = prices.ffill()
# Drop columns with too much missing data
prices = prices.dropna(axis=1, thresh=len(prices)*0.9)
prices.to_parquet('data/sp500_prices.parquet')  # save for reproducibility
```

Document in README: "Data sourced from Yahoo Finance. Survivorship bias acknowledged: current S&P 500 constituents only. Results should be interpreted with this limitation."

### Step 2: Signal Construction (Mean Reversion)

```python
def compute_signal(prices, lookback=5):
    """
    Cross-sectional mean reversion signal.
    Negative past return → positive signal (expect bounce).
    """
    # Daily returns
    returns = prices.pct_change()

    # Past `lookback` day cumulative return
    past_return = prices.pct_change(lookback)

    # Rank cross-sectionally each day: lowest return = rank 0
    ranks = past_return.rank(axis=1, pct=True)

    # Signal: long bottom decile (losers), short top decile (winners)
    signal = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)
    signal[ranks <= 0.1] = 1      # long losers
    signal[ranks >= 0.9] = -1     # short winners

    return signal
```

**Key design decisions to document:**

- **Lookback = 5 days:** Why? Mean reversion at very short horizons (1 day) is dominated by microstructure noise; at longer horizons (20+ days), momentum tends to dominate. 5 days is a reasonable middle ground. You should test 1, 3, 5, 10, 20 and report results (parameter sensitivity).
- **Decile thresholds (10%/90%):** Avoids trading the entire universe, which would incur enormous costs. Only trade extreme movers.
- **Equal weighting:** Simple, avoids optimization on the training set.

### Step 3: Backtest Engine (Correctly Implemented)

This is where most projects fail. The two critical rules:

1. **Positions decided at time t are held from t+1 onward.** You cannot use today's signal to earn today's return.
2. **Costs are charged on every change in position.** A flip from +1 to -1 costs twice as much as opening a new position.

```python
def backtest(prices, signal, cost_bps=10):
    """
    Vectorised backtest with proper lag and transaction costs.

    Parameters
    ----------
    prices : DataFrame of adjusted close prices
    signal : DataFrame of target positions (-1, 0, +1)
    cost_bps : round-trip cost in basis points (applied to traded notional)

    Returns
    -------
    DataFrame with daily returns, costs, net returns
    """
    # Returns: today's return from yesterday's close to today's close
    returns = prices.pct_change().shift(-1)  # return from t to t+1

    # CRITICAL: signal at t applies to return from t+1
    # So we shift signal forward by 1
    held = signal.shift(1).fillna(0)

    # Portfolio return: average across held positions
    # (equal weight across stocks that have a non-zero signal)
    n_positions = held.abs().sum(axis=1)
    n_positions = n_positions.replace(0, np.nan)  # avoid division by zero

    # Weight each position equally
    weights = held.div(n_positions, axis=0).fillna(0)

    # Gross return: sum of weight × stock return
    gross_return = (weights * returns).sum(axis=1)

    # Transaction costs: charged on change in weights
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * (cost_bps / 10000)

    # Net return
    net_return = gross_return - costs

    return pd.DataFrame({
        'gross_return': gross_return,
        'costs': costs,
        'net_return': net_return,
        'turnover': weight_changes / 2,  # one-way turnover
        'n_positions': n_positions
    })
```

**What this engine does correctly:**

- **No look-ahead:** Signal at t is shifted to t+1 before multiplying by returns.
- **Costs on notional:** If you go from 0 to +1 on a $10,000 position, you pay 0.10% × $10,000 = $10.
- **Turnover tracked:** Essential for understanding why a strategy might fail after costs.

### Step 4: Train/Test Split (The Honest Way)

```python
# Split data temporally
train_mask = (prices.index >= '2010-01-01') & (prices.index < '2020-01-01')
test_mask = (prices.index >= '2020-01-01') & (prices.index < '2025-01-01')

# Optional: purge/embargo — remove 5 days after train end from train
# because holding period is 5 days, labels overlap
purge_days = 5
train_end_idx = prices.index.get_loc(prices[train_mask].index[-1])
purge_end_idx = min(train_end_idx + purge_days, len(prices) - 1)

train_prices = prices.iloc[:train_end_idx - purge_days]  # or use purge_end_idx
test_prices = prices.iloc[purge_end_idx:]

# CRITICAL: Select parameters on train ONLY
# Test different lookbacks on TRAIN data, pick best, then report on TEST
lookbacks = [1, 3, 5, 10, 20]
train_results = {}

for lb in lookbacks:
    sig = compute_signal(train_prices, lookback=lb)
    bt = backtest(train_prices, sig, cost_bps=10)
    sharpe = bt['net_return'].mean() / bt['net_return'].std() * np.sqrt(252)
    train_results[lb] = sharpe

best_lookback = max(train_results, key=train_results.get)
print(f"Best lookback on TRAIN: {best_lookback} (Sharpe: {train_results[best_lookback]:.2f})")

# NOW apply to TEST — never use test data for selection
test_signal = compute_signal(test_prices, lookback=best_lookback)
test_bt = backtest(test_prices, test_signal, cost_bps=10)
```

Document this clearly in the README: "Parameter selection used only 2010–2019 data. 2020–2024 was held out entirely and evaluated once."

### Step 5: Out-of-Sample Evaluation Metrics

```python
def compute_metrics(bt_df, label='Out-of-Sample'):
    """Compute standard performance metrics."""
    net = bt_df['net_return'].dropna()

    # Annualised return and volatility
    ann_return = net.mean() * 252
    ann_vol = net.std() * np.sqrt(252)

    # Sharpe (assuming 0% risk-free for simplicity)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    cum = (1 + net).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    # Hit rate (days with positive net return)
    hit_rate = (net > 0).mean()

    # Average turnover (one-way, daily)
    avg_turnover = bt_df['turnover'].mean()

    # Total costs as % of gross return
    total_gross = (1 + bt_df['gross_return']).prod() - 1
    total_net = (1 + net).prod() - 1
    cost_drag = total_gross - total_net

    return {
        'Period': label,
        'Ann. Return': f'{ann_return:.2%}',
        'Ann. Vol': f'{ann_vol:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Max DD': f'{max_dd:.2%}',
        'Hit Rate': f'{hit_rate:.1%}',
        'Avg Daily Turnover': f'{avg_turnover:.2%}',
        'Total Cost Drag': f'{cost_drag:.2%}'
    }

print(pd.DataFrame([compute_metrics(test_bt, '2020–2024 (OOS)')]).T)
```

Expected output format (example; your numbers will differ):

| Metric | Value |
|---|---|
| Ann. Return | X.XX% |
| Ann. Vol | X.XX% |
| Sharpe | X.XX |
| Max DD | -X.XX% |
| Hit Rate | XX.X% |
| Avg Daily Turnover | XX.X% |
| Total Cost Drag | X.XX% |

### Step 6: Cost Sensitivity Analysis (Critical for Credibility)

```python
cost_levels = [0, 5, 10, 20, 50]  # basis points
sensitivity = []

for c in cost_levels:
    bt = backtest(test_prices, test_signal, cost_bps=c)
    metrics = compute_metrics(bt, f'{c} bps')
    sensitivity.append(metrics)

print(pd.DataFrame(sensitivity))
```

**What to look for:** If Sharpe drops from 1.0 at 0bps to 0.2 at 10bps, your signal is "real" but not economically viable at realistic costs. Report this honestly. A SIG reviewer respects "the signal exists but is arbitraged away by costs" far more than a fake high Sharpe.

### Step 7: Generate Plots and README

Required figures:

1. Cumulative return curve (gross vs. net, out-of-sample only)
2. Signal decay plot: average cumulative return of loser decile vs. winner decile over 10 days after signal (shows whether mean reversion exists)
3. Cost sensitivity bar chart: Sharpe at different cost levels

```python
import matplotlib.pyplot as plt

# Figure 1: OOS equity curve
fig, ax = plt.subplots(figsize=(10, 5))
(1 + test_bt['gross_return']).cumprod().plot(ax=ax, label='Gross (0bps)', alpha=0.6)
(1 + test_bt['net_return']).cumprod().plot(ax=ax, label='Net (10bps)', linewidth=2)
ax.set_title('Out-of-Sample Performance: 2020–2024')
ax.set_ylabel('Cumulative Return (1 = start)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/oos_equity_curve.png', dpi=150)
```

README structure: see the repository's [README.md](../README.md), which follows the brief's template (Hypothesis, Data, Methodology, Key Results, Cost Sensitivity, Figures, Limitations, Reproducibility, Files, Requirements, References) filled in with real results.

---

## What Makes This Project Convincing to SIG

| SIG Requirement | How This Project Demonstrates It |
|---|---|
| "Investigate an idea by immersing yourself in market data" | Full pipeline from raw prices to signal to backtest. |
| "Discover a new pattern" | You test a specific hypothesis with a clear economic rationale (mean reversion after overreaction). |
| "Backtest your results and put that trading signal into production" | Proper lag, transaction costs, train/test split, out-of-sample evaluation. |
| "Exceptional mathematical problem-solving" | Cross-sectional ranking, return decomposition, cost accounting. |
| "Strong practical computing skills" | Clean Python, vectorised backtest, reproducible repo. |
| "Experience working on in-depth research projects" | This is a complete research project with documented decisions and limitations. |
| **Not fooling yourself** | Explicit cost sensitivity, honest limitation section, negative results accepted. |

## Common Mistakes to Avoid (Reviewer Will Check)

1. **No lag:** Using `signal.shift(0)` instead of `signal.shift(1)`; this manufactures fake alpha.
2. **Random train/test split:** Using `train_test_split` from sklearn; leaks future into past.
3. **Zero transaction costs:** Presenting 0bps results as "the" result. Always show net at realistic costs.
4. **Survivorship bias denial:** Claiming "S&P 500" without noting you used current constituents.
5. **Overfitting the lookback:** Testing 20 lookbacks on full data and reporting the best. **Select on train, report on test.**
6. **Ignoring turnover:** A signal that flips every day will be destroyed by costs. Report turnover explicitly.
7. **Too many stocks:** Trading all 500 names incurs enormous costs. Use deciles or quintiles to concentrate positions.

## Time Estimate

| Task | Hours |
|---|---|
| Data download and cleaning | 2–3 |
| Signal construction and debugging | 2–3 |
| Backtest engine (with lag and costs) | 3–4 |
| Train/test split and parameter selection | 2 |
| Metrics and plots | 2 |
| README and documentation | 2–3 |
| **Total** | **~15 hours** |

This is achievable over a weekend. The repo does not need to be perfect; it needs to be **honest and complete**. A SIG reviewer will spend 5–10 minutes on your GitHub link. They want to see that you understand the pitfalls and avoided them, not that you found a 3-Sharpe strategy (which they would not believe anyway).

Once the repo is live, update your CV's GitHub link to point directly to this repository, and make sure the project title in your CV matches the repo name exactly.
