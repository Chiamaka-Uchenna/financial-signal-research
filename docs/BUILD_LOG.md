# Build log

A chronological record of how this repository was built and run, including the problems hit along the way. All times are 2026-10-19 (UTC date).

## 1. Environment

- Windows 11 Pro, Python 3.14.0, fresh virtualenv in `.venv/`.
- Installed: yfinance 1.7.0, pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2, scipy 1.18.1, pyarrow 26.0.0, lxml 6.1.3 (needed by `pandas.read_html`), pytest 9.1.1. These are pinned in `requirements.txt`.
- The project lives in its own folder (`financial-signal-research/`) and its own git repository. It is separate from any other project.

## 2. Research configuration fixed before any data was examined

These were set in `run_research.py` before the first download:

- Train 2010–2019, test 2020–2024, 5-day embargo.
- Lookback candidates {1, 3, 5, 10, 20}, hold = 5, decile cut-off = 10%.
- Base cost 10 bps; cost grid {0, 5, 10, 20, 50}.
- Selection rule: highest net-of-10-bps Sharpe on train.

The guide's sample code was reviewed before writing the engine, and the bugs listed in [METHODOLOGY.md §9](METHODOLOGY.md#9-departures-from-the-original-build-guide) were fixed in the design. Results played no part in those fixes.

## 3. Code and tests

- Wrote `src/data.py`, `src/signals.py`, `src/backtest.py`, `src/metrics.py`, `src/analysis.py` and `run_research.py`.
- Wrote 7 unit tests (`tests/test_pipeline.py`). **Result: 7 passed** before any real data was loaded.

## 4. Data download

- `python run_research.py --stage data`
- Universe: 503 symbols scraped from Wikipedia and saved to `data/sp500_tickers.csv`.
- **Problem:** the first download reported 9 failures, including ZTS, TGT, AKAM and SRE. These are long-listed stocks, so the cause was Yahoo rate limiting dropping names from batch requests, not missing data.
- **Fix:** added a second pass in `download_prices` that retries each missing ticker individually with a 3-second pause. The cached parquet was deleted and everything was downloaded again from scratch, so the final data comes from one consistent run.
- **Final:** 498 tickers with data. 5 symbols have no 2009–2024 history: FDXF, HONA, Q, SNDK, VYLR. The data covers 3,816 trading days from 2009-11-02 to 2024-12-31, with 1,779,302 stock-day price observations. Parquet SHA-256: `434ecfcb…5239b` (full hash in `results/data_summary.json`).

## 5. Train stage (run once)

- `python run_research.py --stage train`
- Gross Sharpe ≈ 1.0 for lookbacks 1–10, but net Sharpe is negative for every lookback at 10 bps.
- The rule selected **lookback = 20** (net Sharpe −0.13, the least negative). It was frozen to `results/selection.json` at 2026-10-19T22:21:33Z.
- Recorded observation: the hypothesis named 5 days, but the pre-set rule picked 20. The rule was kept as stated. Switching to 5 after seeing the table would have been a researcher degree of freedom.

## 6. Test stage

- `python run_research.py --stage test`
- Out-of-sample: gross Sharpe 0.05 (NW t = 0.12), net Sharpe at 10 bps −0.31, break-even cost 1.48 bps.
- Checked the figures visually. The −1.77 label in `cost_sensitivity.png` was clipped at the bottom edge.
- **Re-run:** added `ax.margins(y=0.12)` to that chart only and re-ran `--stage test` with the same frozen selection and the same data. Every number came out identical (only the PNG changed). No parameters or rules were changed after seeing test results.

## 7. Documentation

- `README.md`: every number was copied from the files in `results/` produced by the run above.
- `docs/METHODOLOGY.md`: formulas, decisions, and departures from the guide.
- `docs/BUILD_GUIDE.md`: the original brief, kept for reference.

## How to verify this log

- `results/selection.json` holds the frozen choice, its timestamp, and the hash of the data it saw.
- `results/data_summary.json` holds the data hashes and counts.
- `results/oos_results.json` holds the out-of-sample metrics and the run timestamp.
- `git log` holds the history once the repository is committed.
