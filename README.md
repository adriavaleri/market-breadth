# Market Breadth: NYSE and Nasdaq

This standalone project requires Python 3.10 or later. Run these commands from the project folder:

```sh
python -m pip install -r requirements.txt
python tools/market_breadth.py validate
python tools/market_breadth.py bootstrap --workers 32
python tools/market_breadth.py update --workers 32
python tools/market_breadth.py update --start 2022-08-22 --end 2022-08-23
```

To download data, set MASSIVE_API_KEY, MASSIVE_S3_ACCESS_KEY, and MASSIVE_S3_SECRET_KEY in your environment or in a copy of .env.example named .env. Environment variables take precedence. Set the optional MARKET_BREADTH_CACHE environment variable to use a different cache directory outside the project. Relative cache paths are resolved from the project folder. You can invoke the CLI from any working directory by passing the path to its script. The default cache is LOCALAPPDATA/market-breadth on Windows, or ~/.cache/market-breadth if LOCALAPPDATA is unset.

- **bootstrap** rebuilds all series from the first trading session of 2017. It downloads missing Massive inputs, including the warmup period beginning October 3, 2016. Use an empty external cache to download all inputs again.
- **update** downloads and adds pending sessions. The default cutoff accounts for the Daily file publication time in New York. If a download fails, the published series remain available and successful downloads remain cached.
- **update with dates** refreshes the specified inclusive date range. The CLI preserves the validated history before the first affected session, retrieves the required warmup data, and recalculates from that session onward. It recomputes the EMAs and RASI percentile using the full series. The requested range controls which inputs are refreshed.
- **validate** runs locally. It checks the trading calendar, counts, formulas, split coverage, CSV values, and the Parquet hash embedded in each PNG.

For each downloaded session, the CLI combines historical reference data from the Massive REST API with one S3 Daily file shared by both universes. The project folder contains the published results and canonical reference files. The external cache stores the listings, closing prices, and volumes required to rebuild the series. The CLI derives the CSV and PNG from the Parquet and publishes the files after validation succeeds.

All published series are under the project’s indicators/ folder. The indicators/combined/ folder provides one Parquet and CSV with a column for each indicator. The CLI joins the six canonical Parquet files by date and preserves nulls.

## Indicator series

### Full NYSE universe (XNYS)

- **AD Line** is the cumulative difference between advancing and declining instruments. Only instruments with two comparable closing prices contribute.
- **ADn** applies a 21-session stochastic calculation to the AD Line, then smooths the result with a 7-session EMA.
- **RASI** accumulates EMA19 minus EMA39 of RANA, where RANA = 1,000 × (advancing − declining) / (advancing + declining). The initial RASI value is 483 on January 3, 2017, and both EMAs start at zero.
- **RASI Percentile 252D** is the percentage of the previous 252 sessions whose RASI is less than or equal to the current value.

### Combined NYSE and Nasdaq common stocks (XNYS + XNAS, type CS)

- **Above SMA40** is 100 times the combined count of stocks closing above their SMA40 divided by the combined count of eligible stocks. Eligibility requires 40 consecutive valid closing prices.
- **Ratio10D** divides the sum of Up4 by the sum of Down4 over the last 10 sessions, including the current session. A zero denominator produces a null value. Up4 counts returns of at least +4%, and Down4 counts returns of −4% or lower. Both require current volume of at least 100,000 shares and higher than the previous comparable volume.

Each CSV preserves its Parquet columns and uses ISO dates and empty fields for nulls. Each PNG shows the full historical period.

## References and limitations

The project uses the instrument type and primary exchange recorded for each session. Raw closing prices are unadjusted. For split comparisons, the calculation divides the previous close by the split factor. The Stockbee calculation also multiplies previous volume by the same factor.

The split Parquet in the reference folder records coverage and retains 12 verified resolutions linked to the original event IDs. The Stockbee calculation excludes unresolved transitions and records the exclusions in its coverage counts. Six historical transitions remain ambiguous: TWO, ALEX, SLG, SAFE, ABTC, and RJET.

The two 2019 JSON reference files preserve 422 verified Stockbee exclusion intervals and their EDGAR evidence, including the cases that remain unresolved. The August 22, 2022 classification jump remains in the source history. Missing closes and classifications affect coverage.

Sources: [Massive All Tickers](https://massive.com/docs/rest/stocks/tickers/all-tickers), [Day Aggregates](https://massive.com/docs/flat-files/stocks/day-aggregates), and [Splits](https://massive.com/docs/rest/stocks/corporate-actions/splits).

To run the local tests, install pytest and run python -m pytest -q.
