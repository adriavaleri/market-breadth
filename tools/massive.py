"""Massive historical reference and unadjusted S3 bars."""
from __future__ import annotations

import datetime as dt
import email.utils
import gzip
import json
import math
import os
import random
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import numpy as np
import requests
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import boto3
from botocore.config import Config


API_ROOT = "https://api.massive.com"

NY = ZoneInfo("America/New_York")

TICKERS_URL = f"{API_ROOT}/v3/reference/tickers"

TICKER_TYPES_URL = f"{API_ROOT}/v3/reference/tickers/types"


SPLITS_URL = f"{API_ROOT}/stocks/v1/splits"

S3_ENDPOINT = "https://files.massive.com"

S3_BUCKET = "flatfiles"

S3_DAY_AGGS_PREFIX = "us_stocks_sip/day_aggs_v1"

RETRYABLE_HTTP = {429, 500, 502, 503, 504}

MAX_ATTEMPTS = 8

LISTING_SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ms")),
        pa.field("ticker", pa.string()),
        pa.field("type", pa.string()),
        pa.field("exchange", pa.string()),
        pa.field("instrument_id", pa.string()),
        pa.field("instrument_id_type", pa.string()),
        pa.field("share_class_figi", pa.string()),
        pa.field("composite_figi", pa.string()),
        pa.field("cik", pa.string()),
        pa.field("name", pa.string()),
        pa.field("active", pa.bool_()),
        pa.field("delisted_utc", pa.string()),
        pa.field("last_updated_utc", pa.string()),
    ]
)

PRICE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ms")),
        pa.field("ticker", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("vwap", pa.float64()),
        pa.field("transactions", pa.int64()),
        pa.field("massive_timestamp_ms", pa.int64()),
    ]
)

SPLIT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("execution_date", pa.timestamp("ms")),
        pa.field("split_from", pa.float64()),
        pa.field("split_to", pa.float64()),
        pa.field("adjustment_type", pa.string()),
        pa.field("historical_adjustment_factor", pa.float64()),
    ]
)

class DownloadError(RuntimeError):
    """An API, data-integrity, or local persistence error."""

def session_dates(start_date: str, end_date: str) -> list[str]:
    """Return XNYS sessions in the inclusive range using the exchange calendar."""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    # The package's default calendar window starts in 2006. Expand its bounds
    # around the requested dates, then select the exact inclusive interval.
    calendar = xcals.get_calendar(
        "XNYS",
        start=(start - dt.timedelta(days=7)).isoformat(),
        end=(end + dt.timedelta(days=7)).isoformat(),
    )
    sessions = calendar.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return [pd.Timestamp(session).strftime("%Y-%m-%d") for session in sessions]

class MassiveClient:
    """Small REST client with shared pacing, retry handling, and safe pagination."""

    def __init__(self, api_key: str, requests_per_minute: float) -> None:
        if requests_per_minute < 0:
            raise DownloadError("--requests-per-minute no pot ser negatiu.")
        self.api_key = api_key
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_request = 0.0
        self._pace_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._thread_local = threading.local()
        self.request_count = 0
        self.pagination_pages = 0
        self.requests_by_path: dict[str, int] = {}

    def _pace(self) -> None:
        if self.interval == 0:
            return
        now = time.monotonic()
        with self._pace_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self.interval
        if delay:
            time.sleep(delay)

    @staticmethod
    def _retry_after(value: str | None, attempt: int) -> float:
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
                    return max(0.0, (retry_at - dt.datetime.now(dt.timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(30.0, 0.5 * (2 ** (attempt - 1)))

    def request_json(self, url: str) -> dict[str, Any]:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "api.massive.com":
            raise DownloadError("Massive ha retornat un URL no permès; no s’hi enviaran credencials.")
        last_error = "error desconegut"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            with self._metrics_lock:
                self.request_count += 1
                path = urllib.parse.urlsplit(url).path
                self.requests_by_path[path] = self.requests_by_path.get(path, 0) + 1
            try:
                session = getattr(self._thread_local, "session", None)
                if session is None:
                    session = requests.Session()
                    session.headers.update(
                        {
                            "Accept": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                            "User-Agent": "market-breadth/1.0",
                        }
                    )
                    self._thread_local.session = session
                response = session.get(url, timeout=(10, 90), allow_redirects=False)
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}"
                    try:
                        body = response.json()
                        detail = str(body.get("message") or body.get("error") or "") if isinstance(body, dict) else ""
                    except ValueError:
                        detail = ""
                    if self.api_key:
                        detail = detail.replace(self.api_key, "<redacted>")
                    if response.status_code not in RETRYABLE_HTTP or attempt == MAX_ATTEMPTS:
                        suffix = f": {detail[:250]}" if detail else ""
                        raise DownloadError(f"La petició a Massive ha fallat ({last_error}){suffix}.")
                    delay = self._retry_after(response.headers.get("Retry-After"), attempt)
                    response.close()
                    time.sleep(delay + random.uniform(0.0, min(0.5, delay * 0.1)))
                    continue
                payload = response.json()
                if not isinstance(payload, dict):
                    raise DownloadError("Massive ha retornat una resposta JSON inesperada.")
                status = str(payload.get("status", "")).upper()
                if status not in {"OK", "DELAYED"}:
                    message = str(payload.get("error") or payload.get("message") or status or "API error")
                    raise DownloadError(f"Massive ha retornat {message[:300]}.")
                return payload
            except (requests.exceptions.RequestException, TimeoutError, OSError, ValueError) as exc:
                last_error = type(exc).__name__
                if attempt == MAX_ATTEMPTS:
                    raise DownloadError(f"La petició a Massive ha fallat ({last_error}).") from None
                delay = self._retry_after(None, attempt)
                time.sleep(delay + random.uniform(0.0, min(0.5, delay * 0.1)))
            except DownloadError:
                raise
        raise DownloadError(f"La petició a Massive ha fallat ({last_error}).")

    def fetch_all(self, url: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        next_url: str | None = url
        seen_urls: set[str] = set()
        while next_url:
            parts = urllib.parse.urlsplit(next_url)
            if parts.scheme != "https" or parts.netloc != "api.massive.com":
                raise DownloadError("Massive ha retornat una pàgina fora de api.massive.com.")
            if next_url in seen_urls:
                raise DownloadError("Massive ha retornat un bucle de paginació.")
            seen_urls.add(next_url)
            payload = self.request_json(next_url)
            with self._metrics_lock:
                self.pagination_pages += 1
            results = payload.get("results") or []
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                raise DownloadError("Massive ha retornat una pàgina sense una llista de resultats.")
            if any(not isinstance(row, dict) for row in results):
                raise DownloadError("Massive ha retornat registres amb un format inesperat.")
            rows.extend(results)
            next_url = payload.get("next_url")
            if next_url is not None:
                next_url = str(next_url)
        return rows

    def fetch_ticker_types(self) -> set[str]:
        query = urllib.parse.urlencode({"asset_class": "stocks", "locale": "us"})
        rows = self.fetch_all(f"{TICKER_TYPES_URL}?{query}")
        if not rows:
            raise DownloadError("Massive no ha retornat la taxonomia de tipus d’acció.")
        codes = {str(row.get("code", "")).strip() for row in rows if row.get("code")}
        if not codes:
            raise DownloadError("La taxonomia de Massive no conté cap codi de tipus.")
        return codes

    def fetch_listings(self, date_string: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "market": "stocks",
                "date": date_string,
                "active": "true",
                "order": "asc",
                "limit": 1000,
                "sort": "ticker",
            }
        )
        return self.fetch_all(f"{TICKERS_URL}?{query}")


    def fetch_splits(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "execution_date.gte": start_date,
                "execution_date.lte": end_date,
                "sort": "execution_date.asc",
                "limit": 5000,
            }
        )
        return self.fetch_all(f"{SPLITS_URL}?{query}")

class MassiveFlatFilesClient:
    """Concurrent-friendly reader for Massive's unadjusted daily aggregate S3 objects."""

    def __init__(self, access_key: str, secret_key: str, max_connections: int) -> None:
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )
        self.client = session.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            config=Config(
                signature_version="s3v4",
                max_pool_connections=max_connections,
                retries={"max_attempts": 8, "mode": "adaptive"},
            ),
        )
        self.request_count = 0
        self._metrics_lock = threading.Lock()

    @staticmethod
    def object_key(date_string: str) -> str:
        year, month, _day = date_string.split("-")
        return f"{S3_DAY_AGGS_PREFIX}/{year}/{month}/{date_string}.csv.gz"

    def fetch_daily(self, date_string: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        key = self.object_key(date_string)
        try:
            with self._metrics_lock:
                self.request_count += 1
            response = self.client.get_object(Bucket=S3_BUCKET, Key=key)
            compressed = response["Body"].read()
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            code = str(error.get("Code") or "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise DownloadError(f"Massive Flat File absent per a la sessió {date_string}: {key}") from None
            raise DownloadError(
                f"No s’ha pogut descarregar el Flat File de {date_string} ({code or type(exc).__name__})."
            ) from None
        expected_length = response.get("ContentLength")
        if expected_length is not None and len(compressed) != int(expected_length):
            raise DownloadError(
                f"{date_string}: Flat File truncat ({len(compressed)} de {expected_length} bytes)."
            )
        try:
            csv_bytes = gzip.decompress(compressed)
            table = pacsv.read_csv(pa.BufferReader(csv_bytes))
        except Exception as exc:
            raise DownloadError(f"{date_string}: Flat File gzip/CSV invàlid ({type(exc).__name__}).") from None
        return table.to_pandas(), {
            "source": "flat-files",
            "source_key": key,
            "source_bytes": len(compressed),
            "source_etag": str(response.get("ETag", "")).strip('"') or None,
            "source_last_modified_utc": (
                response["LastModified"].astimezone(dt.timezone.utc).isoformat()
                if response.get("LastModified")
                else None
            ),
        }

def _text(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None

def classify_listings(
    date_string: str, raw_rows: Iterable[dict[str, Any]], known_types: set[str], *, all_xnys: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Select the requested XNYS scope and report classification issues separately."""
    listings: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    raw_count = 0
    duplicate_tickers: set[str] = set()
    seen: set[str] = set()
    for row in raw_rows:
        raw_count += 1
        ticker = _text(row.get("ticker"))
        if not ticker:
            issues.append(
                {
                    "date": date_string,
                    "ticker": None,
                    "name": _text(row.get("name")),
                    "type": _text(row.get("type")),
                    "exchange": _text(row.get("primary_exchange")),
                    "issue": "missing_ticker",
                    "share_class_figi": _text(row.get("share_class_figi")),
                    "composite_figi": _text(row.get("composite_figi")),
                    "cik": _text(row.get("cik")),
                }
            )
            continue
        if ticker in seen:
            duplicate_tickers.add(ticker)
        seen.add(ticker)

        ticker_type = _text(row.get("type"))
        exchange = _text(row.get("primary_exchange"))
        reasons: list[str] = []
        if not ticker_type:
            reasons.append("missing_type")
        elif ticker_type not in known_types:
            reasons.append("unknown_type_code")
        if not exchange:
            reasons.append("missing_primary_exchange")
        if reasons:
            issues.append(
                {
                    "date": date_string,
                    "ticker": ticker,
                    "name": _text(row.get("name")),
                    "type": ticker_type,
                    "exchange": exchange,
                    "issue": ";".join(reasons),
                    "share_class_figi": _text(row.get("share_class_figi")),
                    "composite_figi": _text(row.get("composite_figi")),
                    "cik": _text(row.get("cik")),
                }
            )
            if not all_xnys:
                continue
        if exchange != "XNYS" or (not all_xnys and ticker_type != "CS"):
            continue
        share_class_figi = _text(row.get("share_class_figi"))
        composite_figi = _text(row.get("composite_figi"))
        cik = _text(row.get("cik"))
        if share_class_figi:
            instrument_id, instrument_id_type = share_class_figi, "share_class_figi"
        elif composite_figi:
            instrument_id, instrument_id_type = composite_figi, "composite_figi"
        else:
            instrument_id, instrument_id_type = None, None
        listings.append(
            {
                "date": date_string,
                "ticker": ticker,
                "type": ticker_type,
                "exchange": exchange,
                "instrument_id": instrument_id,
                "instrument_id_type": instrument_id_type,
                "share_class_figi": share_class_figi,
                "composite_figi": composite_figi,
                "cik": cik,
                "name": _text(row.get("name")),
                "active": row.get("active") if isinstance(row.get("active"), bool) else None,
                "delisted_utc": _text(row.get("delisted_utc")),
                "last_updated_utc": _text(row.get("last_updated_utc")),
            }
        )
    duplicate_listing_rows = [row for row in listings if row["ticker"] in duplicate_tickers]
    for ticker in sorted({row["ticker"] for row in duplicate_listing_rows}):
        row = next(item for item in duplicate_listing_rows if item["ticker"] == ticker)
        issues.append(
            {
                "date": date_string,
                "ticker": ticker,
                "name": row["name"],
                "type": row["type"],
                "exchange": row["exchange"],
                "issue": "duplicate_ticker_records",
                "share_class_figi": row["share_class_figi"],
                "composite_figi": row["composite_figi"],
                "cik": row["cik"],
            }
        )
    listings = [row for row in listings if row["ticker"] not in duplicate_tickers]
    stats = {
        "reference_rows": raw_count,
        "listing_rows": len(listings),
        "classification_issue_rows": len(issues),
        "duplicate_ticker_count": len(duplicate_tickers),
    }
    return pd.DataFrame(listings), pd.DataFrame(issues), stats

def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None

def normalize_flatfile_daily(
    date_string: str, raw_frame: pd.DataFrame, expected_tickers: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Filter one Massive all-market Daily CSV to the point-in-time listings."""
    required = {"ticker", "open", "high", "low", "close", "volume", "window_start"}
    missing_columns = required - set(raw_frame.columns)
    if missing_columns:
        raise DownloadError(
            f"{date_string}: manquen columnes al Flat File: {', '.join(sorted(missing_columns))}."
        )

    source = raw_frame.copy()
    source["ticker"] = source["ticker"].astype("string").str.strip()
    source = source[source["ticker"].notna() & source["ticker"].ne("")].copy()
    selected = source[source["ticker"].isin(expected_tickers)].copy()
    returned = set(selected["ticker"].astype(str))
    gaps = pd.DataFrame(
        [{"date": date_string, "ticker": ticker} for ticker in sorted(expected_tickers - returned)]
    )

    issues: list[dict[str, Any]] = []
    duplicate_mask = selected["ticker"].duplicated(keep=False)
    duplicate_tickers = set(selected.loc[duplicate_mask, "ticker"].astype(str))
    duplicate_rows = int(selected["ticker"].duplicated().sum())
    for ticker in sorted(duplicate_tickers):
        issues.append({"date": date_string, "ticker": ticker, "issue": "duplicate_ticker_in_flat_file"})
    selected = selected[~selected["ticker"].isin(duplicate_tickers)].copy()

    numeric_columns = ["open", "high", "low", "close", "volume", "window_start"]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    values = selected[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float64", na_value=float("nan"))
    finite = pd.DataFrame(
        np.isfinite(values),
        columns=["open", "high", "low", "close", "volume"],
        index=selected.index,
    )
    invalid_values = ~finite.all(axis=1)
    for idx in selected.index[invalid_values]:
        bad_names = [column for column in ("open", "high", "low", "close", "volume") if not finite.at[idx, column]]
        issues.append(
            {
                "date": date_string,
                "ticker": str(selected.at[idx, "ticker"]),
                "issue": "missing_or_invalid_" + ",".join(bad_names),
            }
        )
    complete = finite.all(axis=1)
    inconsistent = complete & (
        (selected["low"] > selected[["open", "close"]].min(axis=1))
        | (selected["high"] < selected[["open", "close"]].max(axis=1))
        | (selected["low"] > selected["high"])
    )
    non_positive = complete & (selected[["open", "high", "low", "close"]].min(axis=1) <= 0)
    negative_volume = selected["volume"].notna() & (selected["volume"] < 0)
    for mask, issue in (
        (inconsistent, "inconsistent_ohlc_range"),
        (non_positive, "non_positive_ohlc"),
        (negative_volume, "negative_volume"),
    ):
        for idx in selected.index[mask]:
            issues.append({"date": date_string, "ticker": str(selected.at[idx, "ticker"]), "issue": issue})

    window_start = selected["window_start"]
    valid_timestamp = window_start.notna() & (window_start >= 0)
    session_matches = pd.Series(False, index=selected.index)
    if valid_timestamp.any():
        try:
            local_dates = (
                pd.to_datetime(window_start.loc[valid_timestamp], unit="ns", utc=True)
                .dt.tz_convert(NY)
                .dt.strftime("%Y-%m-%d")
            )
            session_matches.loc[valid_timestamp] = local_dates.eq(date_string)
        except (ValueError, OverflowError, TypeError):
            valid_timestamp[:] = False
    for idx in selected.index[~valid_timestamp]:
        issues.append({"date": date_string, "ticker": str(selected.at[idx, "ticker"]), "issue": "missing_or_invalid_window_start"})
    for idx in selected.index[valid_timestamp & ~session_matches]:
        issues.append({"date": date_string, "ticker": str(selected.at[idx, "ticker"]), "issue": "window_start_date_mismatch"})

    transactions = (
        pd.to_numeric(selected["transactions"], errors="coerce")
        if "transactions" in selected.columns
        else pd.Series(pd.NA, index=selected.index, dtype="Float64")
    )
    non_integer_transactions = transactions.notna() & ((transactions - transactions.round()).abs() > 1e-9)
    negative_transactions = transactions.notna() & (transactions < 0)
    for mask, issue in (
        (non_integer_transactions, "non_integer_transactions"),
        (negative_transactions, "negative_transactions"),
    ):
        for idx in selected.index[mask]:
            issues.append({"date": date_string, "ticker": str(selected.at[idx, "ticker"]), "issue": issue})

    timestamp_ms = (window_start.fillna(0).astype("int64") // 1_000_000).where(valid_timestamp, pd.NA)
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series([date_string] * len(selected), index=selected.index)),
            "ticker": selected["ticker"].astype(str),
            "open": selected["open"],
            "high": selected["high"],
            "low": selected["low"],
            "close": selected["close"],
            "volume": selected["volume"],
            "vwap": pd.Series(float("nan"), index=selected.index, dtype="float64"),
            "transactions": transactions.where(~non_integer_transactions & ~negative_transactions).round().astype("Int64"),
            "massive_timestamp_ms": timestamp_ms.astype("Int64"),
        }
    )
    stats = {
        "grouped_results_count": len(source),
        "price_rows": len(output),
        "price_gap_rows": len(gaps),
        "bar_issue_rows": len(issues),
        "duplicate_grouped_tickers": duplicate_rows,
    }
    return output, gaps, pd.DataFrame(issues), stats

def normalize_splits(raw_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for row in raw_rows:
        execution_date = _text(row.get("execution_date"))
        ticker = _text(row.get("ticker"))
        split_from = _number(row.get("split_from"))
        split_to = _number(row.get("split_to"))
        adjustment_type = _text(row.get("adjustment_type"))
        historical_factor = _number(row.get("historical_adjustment_factor"))
        if not execution_date or not ticker or split_from is None or split_to is None or split_from <= 0 or split_to <= 0:
            issues.append(
                {
                    "date": execution_date,
                    "ticker": ticker,
                    "issue": "missing_or_invalid_split_date_ticker_or_factor",
                }
            )
        if adjustment_type not in {"forward_split", "reverse_split", "stock_dividend"}:
            issues.append(
                {
                    "date": execution_date,
                    "ticker": ticker,
                    "issue": "missing_or_unknown_adjustment_type",
                }
            )
        if historical_factor is not None and historical_factor <= 0:
            issues.append(
                {"date": execution_date, "ticker": ticker, "issue": "non_positive_historical_adjustment_factor"}
            )
        rows.append(
            {
                "id": _text(row.get("id")),
                "ticker": ticker,
                "execution_date": execution_date,
                "split_from": split_from,
                "split_to": split_to,
                "adjustment_type": adjustment_type,
                "historical_adjustment_factor": historical_factor,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(issues)

def _frame_to_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    work = frame.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if "execution_date" in work.columns:
        work["execution_date"] = pd.to_datetime(work["execution_date"], errors="coerce")
    for field in schema:
        if field.name not in work:
            work[field.name] = None
    work = work[[field.name for field in schema]]
    return pa.Table.from_pandas(work, schema=schema, preserve_index=False, safe=False)


CREDENTIALS = ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY", "MASSIVE_S3_SECRET_KEY")


def clients(root, workers):
    local = {}
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in CREDENTIALS:
                local[key.strip()] = value.strip().strip('"').strip("'")
    values = [os.environ.get(key) or local.get(key) for key in CREDENTIALS]
    if not all(values):
        raise DownloadError("Configure the three Massive credentials in the environment or .env")
    return MassiveClient(values[0], 0), MassiveFlatFilesClient(values[1], values[2], workers)


def listings_on(client, day, known_types):
    raw = client.fetch_listings(day)
    nasdaq = [row for row in raw if row.get("primary_exchange") == "XNAS"]
    nyse, _, stats = classify_listings(day, raw, known_types, all_xnys=True)
    nasdaq, _, nasdaq_stats = classify_listings(
        day, (dict(row, primary_exchange="XNYS") for row in nasdaq), known_types
    )
    if nyse.empty or nasdaq.empty or stats["duplicate_ticker_count"] or nasdaq_stats["duplicate_ticker_count"]:
        raise DownloadError(f"{day}: missing market or ambiguous ticker listings")
    nasdaq["exchange"] = "XNAS"
    result = pd.concat([nyse, nasdaq], ignore_index=True)
    if result["ticker"].duplicated().any():
        raise DownloadError(f"{day}: a ticker appears in both markets")
    return result
