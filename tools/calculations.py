"""Validated point-in-time breadth calculations."""
from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


STOCHASTIC_PERIOD = 21

EMA_PERIOD = 7

class ADNError(ValueError):
    """The AD Line input is invalid or cannot be updated without data loss."""


def _float_values(values: Iterable[object]) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        if value is None:
            parsed.append(np.nan)
            continue
        try:
            parsed.append(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ADNError(f"Valor d’AD Line no numèric: {value!r}.") from exc
    result = np.asarray(parsed, dtype=np.float64)
    result[~np.isfinite(result)] = np.nan
    return result

def stochastic_21(ad_line: Iterable[object], period: int = STOCHASTIC_PERIOD) -> np.ndarray:
    """Trailing stochastic of the AD Line; undefined for warmup, gaps or flat ranges."""
    if period < 1:
        raise ADNError("El període estocàstic ha de ser positiu.")
    values = _float_values(ad_line)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1:index + 1]
        if not np.isfinite(window).all():
            continue
        low = float(window.min())
        high = float(window.max())
        if high == low:
            continue
        value = 100.0 * (float(values[index]) - low) / (high - low)
        result[index] = min(100.0, max(0.0, value))
    return result

def ema_sma_seed_with_gaps(values: Iterable[object], period: int = EMA_PERIOD) -> np.ndarray:
    """EMA seeded with SMA(period), independently for each contiguous valid run."""
    if period < 1:
        raise ADNError("El període EMA ha de ser positiu.")
    source = _float_values(values)
    result = np.full(source.shape, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    index = 0
    while index < len(source):
        while index < len(source) and not math.isfinite(float(source[index])):
            index += 1
        start = index
        while index < len(source) and math.isfinite(float(source[index])):
            index += 1
        end = index
        if end - start < period:
            continue
        seed_index = start + period - 1
        previous = float(source[start:seed_index + 1].mean())
        result[seed_index] = previous
        for current_index in range(seed_index + 1, end):
            current = float(source[current_index])
            previous = _ema_step(previous, current, alpha)
            result[current_index] = previous
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], 0.0, 100.0)
    return result

def _ema_step(previous: float, current: float, alpha: float) -> float:
    """One deterministic EMA recurrence shared by breadth EMA calculations."""
    return previous + alpha * (current - previous)

def ema_zero_state_with_gaps(values: Iterable[object], alpha: float) -> np.ndarray:
    """EMA initialized at zero; missing inputs stay null and preserve state."""
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ADNError("El factor de suavització EMA ha d’estar entre 0 i 1.")
    source = _float_values(values)
    result = np.full(source.shape, np.nan, dtype=np.float64)
    state = 0.0
    for index, current in enumerate(source):
        if not math.isfinite(float(current)):
            continue
        state = _ema_step(state, float(current), alpha)
        result[index] = state
    return result

def calculate_adn(
    ad_line: Iterable[object],
    stochastic_period: int = STOCHASTIC_PERIOD,
    ema_period: int = EMA_PERIOD,
) -> np.ndarray:
    """Calculate ADn using only current and past AD Line observations."""
    stochastic = stochastic_21(ad_line, period=stochastic_period)
    return ema_sma_seed_with_gaps(stochastic, period=ema_period)

ADVANCE_EMA_ALPHA = 0.10

DECLINE_EMA_ALPHA = 0.05


RASI_REFERENCE_START_VALUE = 483.0

class RASIError(ValueError):
    """Invalid breadth counts or an unsafe RASI Parquet update."""


def _validated_counts(values: Iterable[object], label: str) -> np.ndarray:
    try:
        result = _float_values(values)
    except ADNError as exc:
        raise RASIError(f"El camp {label} conté un valor no numèric: {exc}.") from exc
    finite = np.isfinite(result)
    if (result[finite] < 0).any() or (result[finite] != np.floor(result[finite])).any():
        raise RASIError(f"El camp {label} ha de contenir recomptes enters no negatius.")
    return result

def calculate_rasi(
    advancing: Iterable[object],
    declining: Iterable[object],
    *,
    first_rasi_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (RANA, McClellan Oscillator, RASI) with zero-state EMA recurrences.

    Missing counts and a zero advancing+declining denominator produce nulls for
    that session. EMA and cumulative states are retained and resume on the next
    valid session, without treating an unavailable value as a synthetic zero.
    When first_rasi_value is provided, the first valid RASI output is anchored
    to that level; otherwise the cumulative state starts at zero as before.
    """
    advances = _validated_counts(advancing, "advancing")
    declines = _validated_counts(declining, "declining")
    if len(advances) != len(declines):
        raise RASIError("advancing i declining han de tenir la mateixa longitud.")

    rana = np.full(advances.shape, np.nan, dtype=np.float64)
    valid_counts = np.isfinite(advances) & np.isfinite(declines)
    denominator = advances + declines
    valid = valid_counts & (denominator > 0)
    rana[valid] = 1000.0 * (advances[valid] - declines[valid]) / denominator[valid]

    fast = ema_zero_state_with_gaps(rana, ADVANCE_EMA_ALPHA)
    slow = ema_zero_state_with_gaps(rana, DECLINE_EMA_ALPHA)
    mcclellan = np.full(rana.shape, np.nan, dtype=np.float64)
    ema_valid = np.isfinite(fast) & np.isfinite(slow)
    mcclellan[ema_valid] = fast[ema_valid] - slow[ema_valid]

    rasi = np.full(rana.shape, np.nan, dtype=np.float64)
    cumulative = 0.0
    if first_rasi_value is not None:
        if not math.isfinite(float(first_rasi_value)):
            raise RASIError("first_rasi_value ha de ser finit.")
        first_valid = np.flatnonzero(ema_valid)
        if len(first_valid):
            cumulative = float(first_rasi_value) - float(mcclellan[first_valid[0]])
    for index, oscillator in enumerate(mcclellan):
        if not math.isfinite(float(oscillator)):
            continue
        cumulative += float(oscillator)
        rasi[index] = cumulative
    return rana, mcclellan, rasi

def rasi_percentile_252(values):
    series = pd.Series(values, dtype=float)
    return series.rolling(253, min_periods=253).apply(
        lambda window: 100.0 * (window[:-1] <= window[-1]).mean(), raw=True
    )

ADN_FIELD = pa.field("adn", pa.float64())
RANA_FIELD = pa.field("rana", pa.float64())
MCCLELLAN_FIELD = pa.field("mcclellan", pa.float64())
RASI_FIELD = pa.field("rasi", pa.float64())

AD_LINE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ms")),
        pa.field("universe_count", pa.int32()),
        pa.field("current_close_count", pa.int32()),
        pa.field("previous_close_count", pa.int32()),
        pa.field("comparable_count", pa.int32()),
        pa.field("not_comparable_count", pa.int32()),
        pa.field("advancing", pa.int32()),
        pa.field("declining", pa.int32()),
        pa.field("unchanged", pa.int32()),
        pa.field("net_advances", pa.int32()),
        pa.field("ad_line", pa.int64()),
        pa.field("missing_listing_ticker_count", pa.int32()),
        pa.field("duplicate_listing_ticker_count", pa.int32()),
        pa.field("missing_current_price_count", pa.int32()),
        pa.field("invalid_current_close_count", pa.int32()),
        pa.field("duplicate_current_price_ticker_count", pa.int32()),
        pa.field("missing_previous_listing_count", pa.int32()),
        pa.field("ambiguous_identity_count", pa.int32()),
        pa.field("identity_conflict_count", pa.int32()),
        pa.field("missing_previous_price_count", pa.int32()),
        pa.field("invalid_previous_close_count", pa.int32()),
        pa.field("duplicate_previous_price_ticker_count", pa.int32()),
        pa.field("ticker_change_match_count", pa.int32()),
        pa.field("ticker_fallback_match_count", pa.int32()),
        pa.field("split_adjusted_count", pa.int32()),
        pa.field("split_event_count", pa.int32()),
        pa.field("invalid_split_count", pa.int32()),
        pa.field("current_prices_available", pa.bool_()),
        pa.field("previous_prices_available", pa.bool_()),
        pa.field("previous_listings_available", pa.bool_()),
        ADN_FIELD,
        RANA_FIELD,
        MCCLELLAN_FIELD,
        RASI_FIELD,
    ]
)

class ADLineError(RuntimeError):
    """Input data or continuity does not support a trustworthy AD Line."""

def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None

def _day_string(value: Any) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ADLineError(f"Data no vàlida al camp date: {value!r}.") from exc

def _identity(row: dict[str, Any]) -> tuple[str, str] | None:
    value = _text(row.get("instrument_id"))
    kind = _text(row.get("instrument_id_type"))
    if value and kind in {"share_class_figi", "composite_figi"}:
        return kind, value
    return None

def _listing_index(frame: pd.DataFrame, all_xnys: bool = False) -> dict[str, Any]:
    required = {"ticker", "type", "exchange", "instrument_id", "instrument_id_type"}
    if not required.issubset(frame.columns):
        raise ADLineError(f"Listings: falten camps {', '.join(sorted(required - set(frame.columns)))}.")
    rows = [
        row
        for row in frame.to_dict(orient="records")
        if _text(row.get("exchange")) == "XNYS" and (all_xnys or _text(row.get("type")) == "CS")
    ]
    missing_tickers = sum(_text(row.get("ticker")) is None for row in rows)
    rows = [row for row in rows if _text(row.get("ticker")) is not None]
    ticker_counts = Counter(_text(row.get("ticker")) for row in rows)
    duplicate_tickers = {ticker for ticker, count in ticker_counts.items() if count > 1}
    identity_counts = Counter(identity for row in rows if (identity := _identity(row)) is not None)
    ambiguous_identities = {identity for identity, count in identity_counts.items() if count > 1}
    unique_ticker_rows = [row for row in rows if _text(row.get("ticker")) not in duplicate_tickers]
    by_ticker = {_text(row.get("ticker")): row for row in unique_ticker_rows}
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = _identity(row)
        if identity:
            by_identity[identity].append(row)
    return {
        "universe_count": len(ticker_counts),
        "missing_ticker_count": missing_tickers,
        "duplicate_tickers": duplicate_tickers,
        "ambiguous_identities": ambiguous_identities,
        "by_ticker": by_ticker,
        "by_identity": dict(by_identity),
        "rows": unique_ticker_rows,
    }

def _price_index(frame: pd.DataFrame) -> dict[str, Any]:
    if "ticker" not in frame.columns or "close" not in frame.columns:
        raise ADLineError("Daily prices: falten ticker o close.")
    rows = [row for row in frame.to_dict(orient="records") if _text(row.get("ticker")) is not None]
    ticker_counts = Counter(_text(row.get("ticker")) for row in rows)
    duplicates = {ticker for ticker, count in ticker_counts.items() if count > 1}
    by_ticker = {
        _text(row.get("ticker")): row
        for row in rows
        if _text(row.get("ticker")) not in duplicates
    }
    return {"by_ticker": by_ticker, "duplicate_tickers": duplicates}

def _valid_close(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    try:
        value = float(row.get("close"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None

def _split_index(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    required = {"ticker", "execution_date", "split_from", "split_to", "adjustment_type"}
    if not required.issubset(frame.columns):
        raise ADLineError(f"Splits: falten camps {', '.join(sorted(required - set(frame.columns)))}.")
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    allowed_types = {"forward_split", "reverse_split", "stock_dividend"}
    for row in frame.to_dict(orient="records"):
        ticker = _text(row.get("ticker"))
        if not ticker:
            raise ADLineError("Splits: hi ha un esdeveniment sense ticker; no es pot aplicar amb seguretat.")
        try:
            execution_date = pd.Timestamp(row.get("execution_date")).date()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ADLineError(f"Splits: data efectiva invàlida per a {ticker}.") from exc
        try:
            split_from = float(row.get("split_from"))
            split_to = float(row.get("split_to"))
        except (TypeError, ValueError, OverflowError):
            split_from = split_to = float("nan")
        adjustment_type = _text(row.get("adjustment_type"))
        valid = (
            math.isfinite(split_from)
            and math.isfinite(split_to)
            and split_from > 0
            and split_to > 0
            and adjustment_type in allowed_types
        )
        event_id = _text(row.get("id"))
        key = ("id", event_id) if event_id else (
            "event", ticker, execution_date, split_from, split_to, adjustment_type
        )
        signature = (ticker, execution_date, split_from, split_to, adjustment_type, valid)
        if key in seen:
            if seen[key] != signature:
                raise ADLineError(f"Splits: identificador repetit amb valors incompatibles ({event_id or ticker}).")
            continue
        seen[key] = signature
        indexed[ticker].append(
            {
                "date": execution_date,
                "split_from": split_from,
                "split_to": split_to,
                "adjustment_type": adjustment_type,
                "valid": valid,
                "key": key,
            }
        )
    for events in indexed.values():
        events.sort(key=lambda event: event["date"])
    return dict(indexed)

def _split_ratio(
    split_events: dict[str, list[dict[str, Any]]],
    tickers: set[str],
    previous_date: str,
    current_date: str,
) -> tuple[float | None, int, int]:
    event_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    previous_day = dt.date.fromisoformat(previous_date)
    current_day = dt.date.fromisoformat(current_date)
    for ticker in tickers:
        for event in split_events.get(ticker, []):
            if previous_day < event["date"] <= current_day:
                event_by_key[event["key"]] = event
    ratio = 1.0
    invalid_count = 0
    for event in event_by_key.values():
        if not event["valid"]:
            invalid_count += 1
            continue
        ratio *= event["split_to"] / event["split_from"]
    if invalid_count or not math.isfinite(ratio) or ratio <= 0:
        return None, len(event_by_key), invalid_count or 1
    return ratio, len(event_by_key), 0

def _match_previous(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    ticker = _text(current.get("ticker"))
    identity = _identity(current)
    if identity in previous["ambiguous_identities"]:
        return None, "ambiguous"
    if identity:
        matches = previous["by_identity"].get(identity, [])
        if len(matches) > 1:
            return None, "ambiguous"
        if len(matches) == 1:
            return matches[0], "identity"
    candidate = previous["by_ticker"].get(ticker)
    if candidate is None:
        if ticker in previous["duplicate_tickers"]:
            return None, "ambiguous"
        return None, "missing"
    previous_identity = _identity(candidate)
    if identity and previous_identity and identity != previous_identity:
        return None, "conflict"
    return candidate, "ticker"

def calculate_session_row(
    date_string: str,
    previous_date: str,
    current_listings: pd.DataFrame,
    current_prices: pd.DataFrame,
    previous_listings: pd.DataFrame,
    previous_prices: pd.DataFrame,
    split_events: dict[str, list[dict[str, Any]]],
    previous_ad_line: int = 0,
    *,
    all_xnys: bool = False,
    current_prices_available: bool = True,
    previous_prices_available: bool = True,
    previous_listings_available: bool = True,
) -> dict[str, Any]:
    """Calculate one session; only matched, valid close pairs enter the counts."""
    current = _listing_index(current_listings, all_xnys)
    previous = _listing_index(previous_listings, all_xnys)
    today_prices = _price_index(current_prices)
    yesterday_prices = _price_index(previous_prices)
    universe_tickers = set(current["by_ticker"]) | current["duplicate_tickers"]

    counts: Counter[str] = Counter()
    counts["universe_count"] = current["universe_count"]
    counts["missing_listing_ticker_count"] = current["missing_ticker_count"]
    counts["duplicate_listing_ticker_count"] = len(current["duplicate_tickers"])
    counts["duplicate_current_price_ticker_count"] = len(
        today_prices["duplicate_tickers"] & universe_tickers
    )

    for ticker in universe_tickers:
        if ticker in today_prices["duplicate_tickers"]:
            continue
        close = _valid_close(today_prices["by_ticker"].get(ticker))
        if close is not None:
            counts["current_close_count"] += 1
        elif ticker in today_prices["by_ticker"]:
            counts["invalid_current_close_count"] += 1
        else:
            counts["missing_current_price_count"] += 1

    for row in current["rows"]:
        ticker = _text(row.get("ticker"))
        identity = _identity(row)
        if ticker in today_prices["duplicate_tickers"] or ticker in current["duplicate_tickers"]:
            continue
        if identity and identity in current["ambiguous_identities"]:
            counts["ambiguous_identity_count"] += 1
            continue
        current_close_row = today_prices["by_ticker"].get(ticker)
        current_close = _valid_close(current_close_row)
        if current_close_row is None or current_close is None:
            continue
        if not previous_listings_available:
            counts["missing_previous_listing_count"] += 1
            continue

        previous_row, match_kind = _match_previous(row, previous)
        if match_kind == "ambiguous":
            counts["ambiguous_identity_count"] += 1
            continue
        if match_kind == "conflict":
            counts["identity_conflict_count"] += 1
            continue
        if previous_row is None:
            counts["missing_previous_listing_count"] += 1
            continue
        previous_ticker = _text(previous_row.get("ticker"))
        if match_kind == "ticker":
            counts["ticker_fallback_match_count"] += 1
        elif previous_ticker != ticker:
            counts["ticker_change_match_count"] += 1

        if previous_ticker in yesterday_prices["duplicate_tickers"]:
            counts["duplicate_previous_price_ticker_count"] += 1
            continue
        previous_close_row = yesterday_prices["by_ticker"].get(previous_ticker)
        if previous_close_row is None:
            counts["missing_previous_price_count"] += 1
            continue
        previous_close = _valid_close(previous_close_row)
        if previous_close is None:
            counts["invalid_previous_close_count"] += 1
            continue
        counts["previous_close_count"] += 1

        split_ratio, event_count, invalid_split_count = _split_ratio(
            split_events,
            {ticker, previous_ticker},
            previous_date,
            date_string,
        )
        counts["split_event_count"] += event_count
        if invalid_split_count:
            counts["invalid_split_count"] += invalid_split_count
            continue
        assert split_ratio is not None
        adjusted_previous_close = previous_close / split_ratio
        if not math.isclose(split_ratio, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            counts["split_adjusted_count"] += 1
        counts["comparable_count"] += 1

        tolerance = max(1e-10, max(abs(current_close), abs(adjusted_previous_close)) * 1e-12)
        if current_close > adjusted_previous_close + tolerance:
            counts["advancing"] += 1
        elif current_close < adjusted_previous_close - tolerance:
            counts["declining"] += 1
        else:
            counts["unchanged"] += 1

    net_advances = counts["advancing"] - counts["declining"]
    row: dict[str, Any] = {key: int(counts[key]) for key in AD_LINE_SCHEMA.names if key not in {
        "date", "ad_line", "adn", "rana", "mcclellan", "rasi",
        "current_prices_available", "previous_prices_available", "previous_listings_available"
    }}
    row.update(
        {
            "date": pd.Timestamp(date_string),
            "net_advances": net_advances,
            "ad_line": int(previous_ad_line + net_advances),
            "not_comparable_count": int(counts["universe_count"] - counts["comparable_count"]),
            "current_prices_available": bool(current_prices_available),
            "previous_prices_available": bool(previous_prices_available),
            "previous_listings_available": bool(previous_listings_available),
        }
    )
    _validate_row(row, previous_ad_line)
    return row

def _validate_row(row: dict[str, Any], previous_ad_line: int) -> None:
    if row["advancing"] + row["declining"] + row["unchanged"] != row["comparable_count"]:
        raise ADLineError(f"{row['date']}: advancing/declining/unchanged no sumen comparable_count.")
    if row["net_advances"] != row["advancing"] - row["declining"]:
        raise ADLineError(f"{row['date']}: net_advances inconsistent.")
    if row["ad_line"] != previous_ad_line + row["net_advances"]:
        raise ADLineError(f"{row['date']}: AD Line no és contínua.")
    if row["comparable_count"] + row["not_comparable_count"] != row["universe_count"]:
        raise ADLineError(f"{row['date']}: els recomptes de cobertura no quadren amb l’univers.")

def validate_series(rows: list[dict[str, Any]], expected_dates: list[str]) -> None:
    actual_dates = [_day_string(row["date"]) for row in rows]
    if actual_dates != expected_dates:
        raise ADLineError("La sèrie no conté exactament totes les sessions XNYS esperades, en ordre.")
    previous_line = 0
    for row in rows:
        _validate_row(row, previous_line)
        previous_line = int(row["ad_line"])

EXCHANGES = ("XNYS", "XNAS")

COUNTS = ("listed", "sma40_eligible", "above_sma40", "comparable", "volume_eligible", "up4", "down4")

def classify(history, close, volume, split):
    """Express earlier observations in today's shares before comparing."""
    if close is None or not math.isfinite(close) or close <= 0 or split is None:
        return deque(maxlen=40), False, False, False, False, False, False
    if split != 1:
        history = deque(((p / split, None if v is None else v * split)
                         for p, v in history), maxlen=40)
    comparable = bool(history) and volume is not None and history[-1][1] is not None
    eligible = comparable and volume >= 100_000 and volume > history[-1][1]
    up = eligible and close >= history[-1][0] * 1.04
    down = eligible and close <= history[-1][0] * 0.96
    history.append((close, volume))
    ready = len(history) == 40
    above = ready and close > math.fsum(p for p, _ in history) / 40
    return history, ready, above, comparable, eligible, up, down

def comparison_split(events, ticker, previous_ticker, prior_day, day, resolutions):
    tickers = {ticker, previous_ticker}
    ratio, event_count, _ = _split_ratio(events, tickers, prior_day, day)
    if event_count <= 1:
        return ratio, event_count
    first, last = map(dt.date.fromisoformat, (prior_day, day))
    by_date = {}
    for symbol in tickers:
        for event in events.get(symbol, ()):
            if first < event["date"] <= last:
                by_date.setdefault(event["date"], {})[event["key"]] = event
    ratio = 1.0
    for records in by_date.values():
        if len(records) == 1:
            event = next(iter(records.values()))
            ratio *= event["split_to"] / event["split_from"]
        else:
            ids = frozenset(key[1] for key in records if key[0] == "id")
            if len(ids) != len(records) or ids not in resolutions:
                return None, event_count
            ratio *= resolutions[ids]
    return ratio, event_count

def add_ratios(frame):
    result = frame.copy()
    for field in ("up4", "down4"):
        result[field + "_10d"] = result.groupby("exchange", sort=False)[field].transform(
            lambda values: values.rolling(10, min_periods=10).sum()
        ).astype("Int64")
    result["ratio10d"] = (result["up4_10d"] / result["down4_10d"].replace(0, pd.NA)).astype("Float64")
    result["above_sma40_pct"] = 100 * result["above_sma40"] / result["sma40_eligible"].replace(0, float("nan"))
    result["sma40_missing"] = result["listed"] - result["sma40_eligible"]
    result["comparison_missing"] = result["listed"] - result["comparable"]
    for field in ("sma40_eligible", "comparable"):
        result[field + "_coverage_pct"] = 100 * result[field] / result["listed"].replace(0, float("nan"))
    return result

def build_stockbee(listings, prices, reference, split_path, days):
    if not split_path.is_file():
        raise ValueError("Missing canonical splits")
    split_table = pq.read_table(split_path)
    coverage = split_table.schema.metadata or {}
    if (
        coverage.get(b"massive_split_scope") != b"all stocks"
        or coverage.get(b"massive_split_start", b"9999").decode() > days[0]
        or coverage.get(b"massive_split_end", b"").decode() < days[-1]
    ):
        raise ValueError("La validació de splits Stockbee ha de cobrir tot el període OHLCV.")
    decisions = json.loads(coverage.get(b"stockbee_split_resolutions", b"[]"))
    resolutions = {frozenset(item["ids"]): item["factor"] for item in decisions}
    canonical_ids = set(split_table["id"].to_pylist())
    if len(resolutions) != len(decisions) or any(
        len(ids) < 2 or not ids <= canonical_ids or not math.isfinite(factor) or factor <= 0
        for ids, factor in resolutions.items()
    ):
        raise ValueError("Resolucions canòniques de splits inconsistents.")
    events = _split_index(split_table.to_pandas())
    if any(not event["valid"] for ticker_events in events.values() for event in ticker_events):
        raise ValueError("El Parquet canònic conté factors de split invàlids.")
    exclusions_path = reference / "classification_2019_exclusions.json"
    exclusions = {}
    if exclusions_path.is_file():
        for rule in json.loads(exclusions_path.read_text(encoding="utf-8")):
            match = (tuple(rule["key"]), rule["ticker"])
            if match in exclusions or not ("2019-04-02" <= rule["start"] <= rule["end"] <= "2019-12-31"):
                raise ValueError("Interval de classificació Stockbee inconsistent.")
            exclusions[match] = (rule["start"], rule["end"])
    rows = []
    ambiguous = []
    previous = {}
    prior_day = None
    for index, day in enumerate(days):
        listed_table = listings.read_row_group(index, columns=[
            "date", "exchange", "ticker", "type", "instrument_id", "instrument_id_type"
        ])
        price_table = prices.read_row_group(index, columns=["date", "ticker", "close", "volume"])
        for table in (listed_table, price_table):
            bounds = pc.min_max(table["date"]).as_py()
            expected = dt.datetime.fromisoformat(day)
            if table["date"].null_count or bounds != {"min": expected, "max": expected}:
                raise ValueError(f"Data de sessió inconsistent: {day}")
        members = listed_table.filter(pc.equal(listed_table["type"], "CS")).drop(["date"]).to_pylist()
        symbols = pa.array([row["ticker"] for row in members], type=pa.string())
        bars = price_table.filter(pc.is_in(price_table["ticker"], value_set=symbols)).drop(["date"]).to_pylist()
        by_ticker = {bar["ticker"]: bar for bar in bars}
        if (
            not bars or len(by_ticker) != len(bars) or len({row["ticker"] for row in members}) != len(members)
            or any(row["type"] != "CS" for row in members)
            or {row["exchange"] for row in members} != set(EXCHANGES)
            or not by_ticker.keys() <= {row["ticker"] for row in members}
        ):
            raise ValueError(f"Sessió incompleta o inconsistent: {day}")
        counts = {exchange: Counter() for exchange in EXCHANGES}
        current = {}
        identities = [
            (row["instrument_id_type"], row["instrument_id"])
            if row["instrument_id_type"] and row["instrument_id"] else None
            for row in members
        ]
        # An exclusion changes membership, not the ambiguity of a source FIGI.
        # In 2019 ZF and excluded ZTR share a FIGI in the raw snapshot.
        frequencies = Counter(identities)
        current_identity = {row["ticker"]: identity for row, identity in zip(members, identities, strict=True)}
        if exclusions and "2019-04-02" <= day <= "2019-12-31":
            retained = []
            for row, identity in zip(members, identities, strict=True):
                match = (identity or ("ticker", row["ticker"]), row["ticker"])
                interval = exclusions.get(match) if row["exchange"] == "XNYS" else None
                if interval is None or not interval[0] <= day <= interval[1]:
                    retained.append((row, identity))
            members = [row for row, _ in retained]
            identities = [identity for _, identity in retained]
        prior_identity = {symbol: key for key, (symbol, _) in previous.items()}
        for row, identity in zip(members, identities, strict=True):
            ticker = row["ticker"]
            count = counts[row["exchange"]]
            count["listed"] += 1
            bar = by_ticker.get(ticker)
            if bar is None or (identity is not None and frequencies[identity] > 1):
                continue
            # A ticker-only match requires both sessions to lack a FIGI.
            key = identity or ("ticker", ticker)
            old_ticker, history = previous.pop(key, (ticker, deque(maxlen=40)))
            split = 1.0
            if history and (ticker in events or old_ticker in events):
                split, event_count = comparison_split(events, ticker, old_ticker, prior_day, day, resolutions)
                if event_count and ticker != old_ticker and (
                    (old_ticker in current_identity and current_identity[old_ticker] != identity)
                    or (ticker in prior_identity and prior_identity[ticker] != key)
                ):
                    split = None
                if split is None:
                    count["split_ambiguities"] += 1
                    ambiguous.append((day, ticker, row["exchange"], identity))
                else:
                    count["split_events"] += event_count
            volume = bar["volume"]
            if volume is not None and (not math.isfinite(volume) or volume < 0):
                volume = None
            history, *flags = classify(history, bar["close"], volume, split)
            if history:
                current[key] = ticker, history
            elif split is None and bar["close"] is not None:
                history, *_ = classify(history, bar["close"], volume, 1.0)
                current[key] = ticker, history
            for field, flag in zip(COUNTS[1:], flags, strict=True):
                count[field] += int(flag)
        previous, prior_day = current, day
        for exchange, count in (*counts.items(), ("COMBINED", sum(counts.values(), Counter()))):
            rows.append(dict(date=pd.Timestamp(day), exchange=exchange, split_events=count["split_events"], split_ambiguities=count["split_ambiguities"], **{field: count[field] for field in COUNTS}))
    frame = add_ratios(pd.DataFrame(rows))
    frame = frame[frame["date"] >= "2017-01-01"].reset_index(drop=True)
    frame.attrs["split_ambiguities"] = ambiguous
    return frame

def build_nyse(listings, prices, split_path, days, prefix=None):
    events = _split_index(pq.read_table(split_path).to_pandas())
    fields = ["ticker", "type", "exchange", "instrument_id", "instrument_id_type"]
    rows = []
    previous_listings = previous_prices = None
    ad_line = int(prefix.ad_line.iloc[-1]) if prefix is not None and not prefix.empty else 0
    prior = None
    for index, day in enumerate(days):
        if day < "2016-12-30":
            continue
        members = listings.read_row_group(index, columns=fields)
        members = members.filter(pc.equal(members["exchange"], "XNYS"))
        bars = prices.read_row_group(index, columns=["ticker", "close"])
        bars = bars.filter(pc.is_in(bars["ticker"], value_set=members["ticker"].combine_chunks()))
        current_listings = members.to_pandas()
        current_prices = bars.to_pandas()
        if current_listings.empty or current_prices.empty:
            raise ValueError(f"{day}: incomplete NYSE input")
        if day >= "2017-01-03" and (prefix is None or index > 0):
            if prior is None:
                raise ValueError("NYSE requires the previous session")
            row = calculate_session_row(day, prior, current_listings, current_prices,
                                        previous_listings, previous_prices, events, ad_line, all_xnys=True)
            rows.append(row)
            ad_line = row["ad_line"]
        previous_listings, previous_prices, prior = current_listings, current_prices, day
    frame = pd.DataFrame(rows)
    if prefix is not None:
        columns = list(prefix.columns)
        frame = pd.concat([prefix, frame[columns]], ignore_index=True)
        # The runner validates the complete series once the suffix is joined.
    else:
        validate_series(rows, [day for day in days if day >= "2017-01-03"])
    frame["adn"] = calculate_adn(frame["ad_line"])
    frame["rana"], frame["mcclellan"], frame["rasi"] = calculate_rasi(
        frame["advancing"], frame["declining"], first_rasi_value=RASI_REFERENCE_START_VALUE
    )
    frame["rasi_percentile_252"] = rasi_percentile_252(frame["rasi"])
    return frame
