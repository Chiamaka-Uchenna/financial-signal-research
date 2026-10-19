"""Data acquisition and cleaning.

Universe: current S&P 500 constituents scraped from Wikipedia (survivorship
bias acknowledged). Prices: Yahoo Finance daily bars via yfinance with
auto_adjust=True (splits and dividends folded into the close).
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_tickers(path: Path) -> pd.DataFrame:
    """Scrape the current S&P 500 constituent table and cache it as CSV.

    The cached CSV is the source of truth for reproducibility: once it exists
    it is reused, so a rerun months later uses the same universe.
    """
    if path.exists():
        return pd.read_csv(path)

    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    table = table.rename(columns={"Symbol": "symbol", "Security": "security", "GICS Sector": "sector",
                                  "Date added": "date_added"})
    table = table[["symbol", "security", "sector", "date_added"]].copy()
    # Yahoo uses '-' where the index uses '.', e.g. BRK.B -> BRK-B.
    table["yahoo_symbol"] = table["symbol"].str.replace(".", "-", regex=False)
    table["retrieved_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def download_prices(tickers: list[str], start: str, end: str, path: Path,
                    chunk_size: int = 50, max_retries: int = 3) -> pd.DataFrame:
    """Download adjusted daily closes and volumes; cache to parquet.

    `end` is exclusive in yfinance, so end='2025-01-01' includes 2024-12-31.
    Downloads in chunks with retries because Yahoo rate-limits large requests.
    Returns a long-format frame (date, ticker, close, volume).
    """
    if path.exists():
        return pd.read_parquet(path)

    import yfinance as yf

    frames, failed = [], []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False,
                                  group_by="column", threads=True)
                break
            except Exception as exc:  # network / rate limit
                if attempt == max_retries:
                    raise
                wait = 10 * attempt
                print(f"  chunk {i // chunk_size}: {exc!r}; retrying in {wait}s")
                time.sleep(wait)
        close = raw["Close"].dropna(axis=1, how="all")
        volume = raw["Volume"].reindex(columns=close.columns)
        failed += [t for t in chunk if t not in close.columns]
        long = (close.stack().rename("close").to_frame()
                .join(volume.stack().rename("volume")))
        frames.append(long)
        print(f"  downloaded {i + len(chunk)}/{len(tickers)} tickers")
        time.sleep(1)

    # Second pass: batch requests sometimes drop names under rate limiting even
    # when Yahoo has the data, so retry each missing ticker on its own.
    still_failed = []
    for t in failed:
        time.sleep(3)
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False,
                          group_by="column", multi_level_index=False)
        if raw.empty or raw["Close"].dropna().empty:
            still_failed.append(t)
            continue
        long = pd.DataFrame({"close": raw["Close"], "volume": raw["Volume"]})
        long.index = pd.MultiIndex.from_product([long.index, [t]])
        frames.append(long)
        print(f"  recovered {t} on retry")
    failed = still_failed

    df = pd.concat(frames).reset_index()
    df.columns = ["date", "ticker", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix(".failed.json").write_text(json.dumps(sorted(failed), indent=2))
    return df


def to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Fill short interior gaps only.

    - Forward-fill at most `max_ffill` consecutive missing days, and only
      between a stock's first and last valid observation (no filling before
      listing or after the data ends, which would invent flat prices).
    - No columns are dropped based on how complete they are over the full
      sample: that filter would use future information (whether a stock
      survives with data until 2024) to decide the 2010 universe.
    - Non-positive prices are treated as missing.
    """
    prices = prices.where(prices > 0)
    inside = prices.ffill().notna() & prices.bfill().notna()
    return prices.ffill(limit=max_ffill).where(inside)


def data_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    valid = prices.notna()
    big_moves = (returns.abs() > 0.5).sum().sum()
    return {
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
        "trading_days": int(len(prices)),
        "tickers_with_data": int(valid.any().sum()),
        "stock_day_price_observations": int(valid.sum().sum()),
        "stock_day_return_observations": int(returns.notna().sum().sum()),
        "median_stocks_per_day": float(valid.sum(axis=1).median()),
        "min_stocks_per_day": int(valid.sum(axis=1).min()),
        "daily_abs_returns_over_50pct": int(big_moves),
        "max_daily_return": float(np.nanmax(returns.values)),
        "min_daily_return": float(np.nanmin(returns.values)),
    }
