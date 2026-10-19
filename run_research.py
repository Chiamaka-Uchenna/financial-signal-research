"""End-to-end research pipeline: short-term reversal in S&P 500 equities.

Usage
-----
    python run_research.py                # data -> train -> test (default)
    python run_research.py --stage data   # download / cache data only
    python run_research.py --stage train  # parameter selection on 2010-2019 only
    python run_research.py --stage test   # one-shot evaluation on 2020-2024

The train stage truncates the price panel before 2020-01-01 *before* computing
anything, and writes the chosen lookback to results/selection.json. The test
stage refuses to run without that file and never changes it, so the holdout is
physically separated from parameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis import signal_decay
from src.backtest import backtest, daily_returns
from src.data import clean_prices, data_summary, download_prices, fetch_sp500_tickers, to_wide
from src.metrics import compute_metrics, yearly_returns
from src.signals import build_weights

# ---- Research configuration (fixed before looking at any results) ----------
DOWNLOAD_START = "2009-11-01"   # warm-up so the 20-day lookback + 5-day hold are live by Jan 2010
DOWNLOAD_END = "2025-01-01"     # yfinance end is exclusive -> includes 2024-12-31
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
LOOKBACKS = [1, 3, 5, 10, 20]   # candidate formation periods (trading days)
HOLD = 5                        # holding period from the hypothesis (trading days)
QUANTILE = 0.10                 # decile sorts
EMBARGO = 5                     # trading days dropped from the end of the train evaluation window
BASE_COST_BPS = 10
COST_GRID = [0, 5, 10, 20, 50]

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
TICKERS_CSV = DATA / "sp500_tickers.csv"
PRICES_PARQUET = DATA / "sp500_prices.parquet"
SELECTION_JSON = RESULTS / "selection.json"

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


# ---- helpers ---------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prices() -> pd.DataFrame:
    tickers = fetch_sp500_tickers(TICKERS_CSV)
    long = download_prices(tickers["yahoo_symbol"].tolist(), DOWNLOAD_START, DOWNLOAD_END, PRICES_PARQUET)
    return clean_prices(to_wide(long, "close"))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)


# ---- stages ----------------------------------------------------------------
def stage_data() -> None:
    prices = load_prices()
    summary = data_summary(prices, daily_returns(prices))
    summary["universe_csv_sha256"] = sha256(TICKERS_CSV)
    summary["prices_parquet_sha256"] = sha256(PRICES_PARQUET)
    tickers = pd.read_csv(TICKERS_CSV)
    summary["universe_size_requested"] = len(tickers)
    summary["universe_retrieved_utc"] = str(tickers["retrieved_utc"].iloc[0])
    failed = PRICES_PARQUET.with_suffix(".failed.json")
    summary["tickers_failed_download"] = json.loads(failed.read_text()) if failed.exists() else []
    save_json(summary, RESULTS / "data_summary.json")
    print(json.dumps(summary, indent=2))


def stage_train() -> dict:
    prices = load_prices()
    # Hard wall: nothing dated in the test period exists from here on.
    prices = prices.loc[: pd.Timestamp(TEST_START) - pd.Timedelta(days=1)]
    returns = daily_returns(prices)
    train_dates = prices.loc[TRAIN_START:TRAIN_END].index
    eval_dates = train_dates[:-EMBARGO]

    rows = []
    for lb in LOOKBACKS:
        w = build_weights(prices, lookback=lb, hold=HOLD, quantile=QUANTILE)
        bt = backtest(w, returns, cost_bps=BASE_COST_BPS).loc[eval_dates]
        net = compute_metrics(bt, "net_return", hac_lags=HOLD)
        gross = compute_metrics(bt, "gross_return", hac_lags=HOLD)
        rows.append({"lookback": lb, "sharpe_gross": gross["sharpe"], "sharpe_net": net["sharpe"],
                     "ann_return_gross": gross["ann_return"], "ann_return_net": net["ann_return"],
                     "t_stat_nw_gross": gross["t_stat_nw"], "max_drawdown_net": net["max_drawdown"],
                     "avg_daily_turnover": net["avg_daily_turnover"]})
    table = pd.DataFrame(rows).set_index("lookback")
    table.to_csv(RESULTS / "train_lookback_selection.csv")
    best = int(table["sharpe_net"].idxmax())

    selection = {
        "selected_lookback": best,
        "criterion": f"max annualised Sharpe net of {BASE_COST_BPS} bps, train window only",
        "train_eval_window": [str(eval_dates[0].date()), str(eval_dates[-1].date())],
        "embargo_trading_days": EMBARGO,
        "candidates": LOOKBACKS,
        "hold": HOLD,
        "quantile": QUANTILE,
        "last_price_date_visible_to_train": str(prices.index.max().date()),
        "prices_parquet_sha256": sha256(PRICES_PARQUET),
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(selection, SELECTION_JSON)
    print(table.round(4).to_string())
    print(f"\nSelected lookback on TRAIN: {best}")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(LOOKBACKS))
    ax.bar([i - 0.2 for i in x], table["sharpe_gross"], width=0.38, color=GRAY, label="Gross (0 bps)")
    ax.bar([i + 0.2 for i in x], table["sharpe_net"], width=0.38, color=BLUE,
           label=f"Net ({BASE_COST_BPS} bps)")
    ax.axhline(0, color="#9a9993", linewidth=0.8)
    ax.set_xticks(list(x), [f"{lb}d" for lb in LOOKBACKS])
    ax.set_xlabel("Lookback (formation period)", color=MUTED)
    style_axes(ax, f"Train 2010–2019: Sharpe by lookback (hold {HOLD}d)", "Annualised Sharpe")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "train_lookback_sharpe.png", dpi=150)
    plt.close(fig)
    return selection


def stage_test() -> None:
    if not SELECTION_JSON.exists():
        sys.exit("results/selection.json not found - run `--stage train` first.")
    selection = json.loads(SELECTION_JSON.read_text())
    lb = int(selection["selected_lookback"])

    prices = load_prices()
    returns = daily_returns(prices)
    # Signals at t use only prices <= t, so computing them on the full panel is
    # equivalent to computing them day by day (verified in tests/).
    weights = build_weights(prices, lookback=lb, hold=HOLD, quantile=QUANTILE)
    test_dates = prices.loc[TEST_START:TEST_END].index

    def run(cost):
        return backtest(weights, returns, cost_bps=cost).loc[test_dates]

    bt = run(BASE_COST_BPS)
    gross_m = compute_metrics(bt, "gross_return", hac_lags=HOLD)
    net_m = compute_metrics(bt, "net_return", hac_lags=HOLD)

    sens = []
    for c in COST_GRID:
        m = compute_metrics(run(c), "net_return", hac_lags=HOLD)
        sens.append({"cost_bps": c, **m})
    sens = pd.DataFrame(sens).set_index("cost_bps")
    breakeven_bps = gross_m["ann_return"] / (bt["turnover"].mean() * 252) * 1e4 \
        if bt["turnover"].mean() > 0 else float("nan")

    yearly = yearly_returns(bt)
    decay_test = signal_decay(prices, lb, 10, QUANTILE, TEST_START, TEST_END)
    decay_train = signal_decay(prices.loc[:TRAIN_END], lb, 10, QUANTILE, TRAIN_START, TRAIN_END)

    RESULTS.mkdir(exist_ok=True)
    bt.to_csv(RESULTS / "oos_daily_backtest.csv")
    sens.to_csv(RESULTS / "oos_cost_sensitivity.csv")
    yearly.to_csv(RESULTS / "oos_yearly_returns.csv")
    decay_test.to_csv(RESULTS / "signal_decay_test.csv")
    decay_train.to_csv(RESULTS / "signal_decay_train.csv")
    save_json({"selection": selection, "test_window": [str(test_dates[0].date()), str(test_dates[-1].date())],
               "gross": gross_m, f"net_{BASE_COST_BPS}bps": net_m,
               "breakeven_cost_bps": breakeven_bps,
               "environment": {"python": platform.python_version(), "pandas": pd.__version__},
               "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              RESULTS / "oos_results.json")

    print(f"Selected lookback (frozen from train): {lb}")
    print(pd.DataFrame({"gross (0 bps)": gross_m, f"net ({BASE_COST_BPS} bps)": net_m}).round(4).to_string())
    print("\nCost sensitivity (net):")
    print(sens[["sharpe", "ann_return", "max_drawdown", "t_stat_nw"]].round(4).to_string())
    print(f"\nBreak-even cost: {breakeven_bps:.2f} bps per unit traded")

    # Figure 1: OOS equity curve, gross vs net
    fig, ax = plt.subplots(figsize=(10, 5))
    eq_g = (1 + bt["gross_return"]).cumprod()
    eq_n = (1 + bt["net_return"]).cumprod()
    ax.plot(eq_g.index, eq_g, color=GRAY, linewidth=1.6, label="Gross (0 bps)")
    ax.plot(eq_n.index, eq_n, color=BLUE, linewidth=2, label=f"Net ({BASE_COST_BPS} bps)")
    ax.axhline(1, color="#9a9993", linewidth=0.8)
    for s, color in ((eq_g, GRAY), (eq_n, BLUE)):
        ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    style_axes(ax, f"Out-of-sample 2020–2024: lookback {lb}d, hold {HOLD}d, decile long-short",
               "Growth of 1")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "oos_equity_curve.png", dpi=150)
    plt.close(fig)

    # Figure 2: signal decay (market-adjusted cumulative forward returns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (name, d) in zip(axes, (("Train 2010–2019", decay_train), ("Test 2020–2024", decay_test))):
        ax.plot(d.index, d["losers"] * 1e4, color=BLUE, linewidth=2, marker="o", markersize=4,
                label="Loser decile (long)")
        ax.plot(d.index, d["winners"] * 1e4, color=ORANGE, linewidth=2, marker="o", markersize=4,
                label="Winner decile (short)")
        ax.axhline(0, color="#9a9993", linewidth=0.8)
        ax.axvline(HOLD, color="#c9c8c2", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Trading days after formation", color=MUTED)
        style_axes(ax, name, "Cum. return vs universe (bps)" if ax is axes[0] else "")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Signal decay after formation (lookback {lb}d)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "signal_decay.png", dpi=150)
    plt.close(fig)

    # Figure 3: cost sensitivity
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [BLUE if v >= 0 else ORANGE for v in sens["sharpe"]]
    bars = ax.bar([str(c) for c in sens.index], sens["sharpe"], color=colors, width=0.6)
    for b, v in zip(bars, sens["sharpe"]):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4 if v >= 0 else -12),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.axhline(0, color="#9a9993", linewidth=0.8)
    ax.margins(y=0.12)
    ax.set_xlabel("Transaction cost (bps per unit notional traded)", color=MUTED)
    style_axes(ax, "Out-of-sample Sharpe vs transaction cost", "Annualised Sharpe (net)")
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["data", "train", "test", "all"], default="all")
    stage = p.parse_args().stage
    for d in (DATA, RESULTS, FIGURES):
        d.mkdir(exist_ok=True)
    if stage in ("data", "all"):
        stage_data()
    if stage in ("train", "all"):
        stage_train()
    if stage in ("test", "all"):
        stage_test()


if __name__ == "__main__":
    main()
