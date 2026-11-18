"""Build, update and validate the standalone market breadth series."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import calculations as calc
import massive

ROOT = Path(__file__).resolve().parents[1]
START = "2016-10-03"
FIRST_OUTPUT = "2017-01-03"
SPLITS = ROOT / "reference/splits.parquet"
OUTPUT_DIR = Path("indicators")
NY_FIELDS = ("advancing", "declining", "unchanged", "net_advances", "universe_count",
             "comparable_count", "not_comparable_count")
SMA_FIELDS = ("above_sma40_pct", "above_sma40", "sma40_eligible", "listed",
              "sma40_missing", "sma40_eligible_coverage_pct")
RATIO_FIELDS = ("ratio10d", "up4", "down4", "up4_10d", "down4_10d", "listed",
                "comparable", "volume_eligible", "comparison_missing",
                "comparable_coverage_pct", "split_events", "split_ambiguities")
FIELDS = {
    "ad_line": ("ad_line", *NY_FIELDS), "adn": ("adn",), "rasi": ("rasi", "rana", "mcclellan"),
    "rasi_percentile_252d": ("rasi_percentile_252",), "above_sma40": SMA_FIELDS,
    "ratio10d": RATIO_FIELDS,
}


def cache_root():
    default = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "market-breadth"
    configured = Path(os.environ.get("MARKET_BREADTH_CACHE", default)).expanduser()
    path = (ROOT / configured).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("The cache must be outside the shared project")
    return path


def available_end():
    now = dt.datetime.now(massive.NY)
    return (now.date() - dt.timedelta(days=1 if now.hour >= 11 else 2)).isoformat()


DECIMAL_PLACES = 2


def export_series(dest, name, frame):
    folder = dest / OUTPUT_DIR / name
    folder.mkdir(exist_ok=True, parents=True)
    file = folder / f"{name}.parquet"
    frame.round(DECIMAL_PLACES).to_parquet(file, index=False, compression="zstd")
    write_derivatives(file)


def write_derivatives(file):
    frame = pd.read_parquet(file)
    frame.to_csv(file.with_suffix(".csv"), index=False, date_format="%Y-%m-%d", float_format="%.2f")
    label = frame.columns[1]
    figure, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(frame.date, frame[label].to_numpy(dtype=float, na_value=np.nan),
            color="#126782", linewidth=0.8)
    ax.set_title(file.stem.replace("_", " ").upper(), loc="left")
    ax.set_ylabel("%" if label == "above_sma40_pct" else label)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip("."))
    ax.grid(alpha=0.22)
    figure.savefig(file.with_suffix(".png"), dpi=150, metadata={"Software": "market_breadth", "SourceSHA256": hashlib.sha256(file.read_bytes()).hexdigest()})
    plt.close(figure)


def combined_series(root):
    combined = None
    for name, columns in FIELDS.items():
        source = root / OUTPUT_DIR / name / f"{name}.parquet"
        value = columns[0]
        series = pd.read_parquet(source, columns=["date", value]).rename(columns={value: name})
        series["date"] = series["date"].astype("datetime64[ns]")
        if series["date"].isna().any() or series["date"].duplicated().any():
            raise ValueError(f"{name}: invalid date key for combined output")
        combined = series if combined is None else combined.merge(
            series, on="date", how="outer", validate="one_to_one"
        )
    return combined[["date", *FIELDS]].sort_values("date").reset_index(drop=True)


def export_combined(root):
    frame = combined_series(root)
    folder = root / OUTPUT_DIR / "combined"
    folder.mkdir(exist_ok=True)
    file = folder / "combined.parquet"
    frame.to_parquet(file, index=False, compression="zstd")
    pd.read_parquet(file).to_csv(file.with_suffix(".csv"), index=False, date_format="%Y-%m-%d", float_format="%.2f")


def equal_csv(file, frame):
    types = {key: str(value) for key, value in frame.dtypes.items()
             if key != "date" and not pd.api.types.is_float_dtype(value)}
    csv = pd.read_csv(file, parse_dates=["date"], dtype=types, float_precision="round_trip")
    for name in csv:
        if name in frame and pd.api.types.is_float_dtype(frame[name]):
            csv[name] = csv[name].astype(frame[name].dtype)
    csv["date"] = csv["date"].astype(frame["date"].dtype)
    pd.testing.assert_frame_equal(csv, frame, check_exact=True)


def validate(root=ROOT):
    frames = {}
    from PIL import Image
    for name, columns in FIELDS.items():
        stem = root / OUTPUT_DIR / name / name
        parquet = stem.with_suffix(".parquet")
        frame = pd.read_parquet(parquet)
        frame["date"] = frame["date"].astype("datetime64[ns]")
        if frame.empty or list(frame.columns) != ["date", *columns]:
            raise ValueError(f"{name}: invalid schema")
        expected = massive.session_dates(FIRST_OUTPUT, str(frame.date.iloc[-1].date()))
        if frame.date.dt.strftime("%Y-%m-%d").tolist() != expected:
            raise ValueError(f"{name}: invalid session calendar")
        pd.testing.assert_frame_equal(frame, frame.round(DECIMAL_PLACES), check_exact=True)
        equal_csv(stem.with_suffix(".csv"), frame)
        with Image.open(stem.with_suffix(".png")) as image:
            if image.info.get("SourceSHA256") != hashlib.sha256(parquet.read_bytes()).hexdigest():
                raise ValueError(f"{name}: PNG does not match the Parquet")
            image.verify()
        frames[name] = frame
    combined = pd.read_parquet(root / OUTPUT_DIR / "combined/combined.parquet")
    pd.testing.assert_frame_equal(combined, combined_series(root), check_exact=True)
    equal_csv(root / OUTPUT_DIR / "combined/combined.csv", combined)
    date = frames["ad_line"].date
    if not all(date.equals(f.date) for f in frames.values()):
        raise ValueError("Inconsistent indicator dates")
    ad = frames["ad_line"]
    net = ad.advancing - ad.declining
    if not (ad.net_advances.eq(net).all() and ad.ad_line.eq(net.cumsum()).all()
            and (ad.advancing + ad.declining + ad.unchanged).eq(ad.comparable_count).all()
            and (ad.comparable_count + ad.not_comparable_count).eq(ad.universe_count).all()):
        raise ValueError("NYSE breadth count or AD Line mismatch")
    def check_values(actual, expected):
        if not np.array_equal(np.asarray(actual, dtype=float), np.round(np.asarray(expected, dtype=float), DECIMAL_PLACES), equal_nan=True):
            raise ValueError(f"Formula mismatch: {actual.name}")
    check_values(frames["adn"].adn, calc.calculate_adn(ad.ad_line))
    rana, oscillator, rasi = calc.calculate_rasi(ad.advancing, ad.declining,
                                                 first_rasi_value=calc.RASI_REFERENCE_START_VALUE)
    for name, values in (("rana", rana), ("mcclellan", oscillator), ("rasi", rasi)):
        check_values(frames["rasi"][name], values)
    check_values(frames["rasi_percentile_252d"].rasi_percentile_252, calc.rasi_percentile_252(rasi))
    sma, ratio = frames["above_sma40"], frames["ratio10d"]
    check_values(sma.above_sma40_pct, 100*sma.above_sma40/sma.sma40_eligible.replace(0, np.nan))
    check_values(ratio.ratio10d, ratio.up4_10d/ratio.down4_10d.replace(0, pd.NA))
    for name in ("up4", "down4"):
        check_values(ratio[name+"_10d"].iloc[9:], ratio[name].rolling(10).sum().iloc[9:])
    if not (sma.listed.eq(ratio.listed).all()
            and (sma.sma40_eligible+sma.sma40_missing).eq(sma.listed).all()
            and (ratio.comparable+ratio.comparison_missing).eq(ratio.listed).all()
            and (ratio.up4+ratio.down4).le(ratio.volume_eligible).all()
            and ratio.volume_eligible.le(ratio.comparable).all()
            and sma.above_sma40.le(sma.sma40_eligible).all()):
        raise ValueError("Stockbee coverage mismatch")
    splits = pq.read_table(root / "reference/splits.parquet")
    meta = splits.schema.metadata or {}
    ids = set(splits["id"].to_pylist())
    if (meta.get(b"massive_split_scope") != b"all stocks"
        or meta.get(b"massive_split_start", b"9999").decode() > START
        or meta.get(b"massive_split_end", b"").decode() < str(date.iloc[-1].date())
        or None in ids or len(ids) != len(splits)
        or any(not set(item["ids"]) <= ids for item in json.loads(meta.get(b"stockbee_split_resolutions", b"[]")))
        or any(not event["valid"] for items in calc._split_index(splits.to_pandas()).values() for event in items)):
        raise ValueError("Split reference coverage or integrity mismatch")
    print(f"Validated {len(date):,} sessions, {date.iloc[0]:%Y-%m-%d} to {date.iloc[-1]:%Y-%m-%d}")
    return str(date.iloc[-1].date())


class Sessions:
    """Expose the per-session cache in the calculation readers' row-group interface."""

    def __init__(self, folder, dates):
        self.paths = [folder / f"{day}.parquet" for day in dates]
        self.index = None
        self.data = None

    def read_row_group(self, index, columns):
        if index != self.index:
            self.data = pq.read_table(self.paths[index])
            self.index = index
        data = self.data
        if "close" in columns:
            data = data.filter(pa.compute.is_valid(data["close"]))
        return data.select(columns)


def cache_session(folder, day, rest, s3, types):
    listed = massive.listings_on(rest, day, types)
    raw, _ = s3.fetch_daily(day)
    bars, _, issues, _ = massive.normalize_flatfile_daily(day, raw, set(listed.ticker))
    if bars.empty or not issues.empty:
        raise ValueError(f"{day}: invalid Daily aggregate")
    fields = ["date", "ticker", "type", "exchange", "instrument_id", "instrument_id_type"]
    merged = listed[fields].merge(bars[["ticker", "close", "volume"]], on="ticker", how="left", validate="one_to_one")
    path = folder / f"{day}.parquet"
    temp = path.with_suffix(".tmp")
    merged.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def refresh_splits(rest, start, end, target):
    old = pq.read_table(SPLITS)
    meta = old.schema.metadata or {}
    covered = meta[b"massive_split_end"].decode()
    if start is None and end <= covered:
        target.write_bytes(SPLITS.read_bytes())
        return
    start = start or (dt.date.fromisoformat(covered) + dt.timedelta(days=1)).isoformat()
    if start > (dt.date.fromisoformat(covered) + dt.timedelta(days=1)).isoformat():
        start = (dt.date.fromisoformat(covered) + dt.timedelta(days=1)).isoformat()
    rows, issues = massive.normalize_splits(rest.fetch_splits(start, end))
    if not issues.empty and issues.issue.ne("non_positive_historical_adjustment_factor").any():
        raise ValueError("Massive returned invalid effective split factors")
    baseline = old.to_pandas()
    refreshed = pd.concat([baseline[~baseline.execution_date.between(start, end)], rows], ignore_index=True)
    if not refreshed.id.is_unique or refreshed.id.isna().any():
        raise ValueError("Missing or duplicate split IDs")
    keys = {key for item in json.loads(meta.get(b"stockbee_split_resolutions", b"[]")) for key in item["ids"]}
    columns = ["id", "ticker", "execution_date", "split_from", "split_to", "adjustment_type"]
    pd.testing.assert_frame_equal(baseline.loc[baseline.id.isin(keys), columns].sort_values("id").reset_index(drop=True),
                                  refreshed.loc[refreshed.id.isin(keys), columns].sort_values("id").reset_index(drop=True), check_dtype=False)
    result = massive._frame_to_table(refreshed, massive.SPLIT_SCHEMA)
    pq.write_table(result.replace_schema_metadata({**meta, b"massive_split_end": max(covered,end).encode()}),
                   target, compression="zstd")


def compute(folder, dates, split_file, affected=None):
    def readers(days):
        session = Sessions(folder, days)
        return session, session

    at = dates.index(affected) if affected else 0
    ny_days = dates[max(0, at-1):] if affected else dates
    sb_days = dates[max(0, at-49):] if affected else dates
    prefix = None
    if affected:
        prefix = pd.read_parquet(ROOT / OUTPUT_DIR / "ad_line/ad_line.parquet")
        prefix = prefix[prefix.date.lt(affected)]
    nyse = calc.build_nyse(*readers(ny_days), split_file, ny_days, prefix=prefix)
    stockbee = calc.build_stockbee(*readers(sb_days), ROOT / "reference", split_file, sb_days)
    combined = stockbee.loc[stockbee.exchange.eq("COMBINED")]
    result = {}
    for key, fields in FIELDS.items():
        is_stockbee = key in {"above_sma40", "ratio10d"}
        frame = (combined if is_stockbee else nyse)[["date", *fields]].reset_index(drop=True)
        if is_stockbee and affected:
            previous = pd.read_parquet(ROOT / OUTPUT_DIR / key / f"{key}.parquet")
            keep = previous[previous.date.lt(affected)]
            tail = frame[frame.date.ge(affected)]
            frame = pd.concat([keep, tail], ignore_index=True)
            frame.attrs["split_ambiguities"] = [
                entry for entry in previous.attrs.get("split_ambiguities", []) if entry[0] < affected
            ] + [entry for entry in stockbee.attrs.get("split_ambiguities", []) if entry[0] >= affected]
        result[key] = frame
    return result


def run(args):
    if args.command == "validate":
        validate()
        return
    latest = available_end()
    if bool(args.start) != bool(args.end):
        raise ValueError("Provide both --start and --end for a dated update")
    if args.start and not START <= args.start <= args.end <= latest:
        raise ValueError("The refresh interval is outside available history")
    existing = ROOT / OUTPUT_DIR / "ad_line/ad_line.parquet"
    published = str(pd.read_parquet(existing, columns=["date"]).date.iloc[-1].date()) if existing.exists() else None
    end = max(published or FIRST_OUTPUT, args.end or latest)
    dates = massive.session_dates(START, end)
    affected = next((day for day in dates if (args.start and day >= args.start)
                     or (published and day > published)), None)
    if args.start and not any(args.start <= day <= args.end for day in dates):
        raise ValueError("The refresh interval contains no sessions")
    if args.command == "update" and published and affected is None:
        validate()
        return
    partial = args.command == "update" and published and affected and affected > FIRST_OUTPUT
    required = dates[max(0, dates.index(affected)-49):] if partial else dates
    folder = cache_root() / "sessions"
    folder.mkdir(exist_ok=True, parents=True)
    pending = [day for day in required if not (folder / f"{day}.parquet").exists()
               or args.start and args.start <= day <= args.end]
    split_end = pq.read_schema(SPLITS).metadata[b"massive_split_end"].decode()
    rest = s3 = None
    if pending or args.start or split_end < dates[-1]:
        rest, s3 = massive.clients(ROOT, args.workers)
    if pending:
        known_types = rest.fetch_ticker_types()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for number, _ in enumerate(pool.map(lambda day: cache_session(folder, day, rest, s3, known_types), pending), 1):
                if number % 100 == 0 or number == len(pending):
                    print(f"Cached {number}/{len(pending)} sessions", flush=True)
    with TemporaryDirectory(dir=ROOT) as name:
        stage = Path(name)
        (stage / "reference").mkdir()
        split_file = stage / "reference/splits.parquet"
        refresh_splits(rest, args.start, max(args.end or dates[-1], dates[-1] if split_end < dates[-1] else ""), split_file)
        frames = compute(folder, dates, split_file, affected if partial else None)
        for key, frame in frames.items():
            export_series(stage, key, frame)
        export_combined(stage)
        validate(stage)
        publish(stage)


def publish(stage):
    for key in FIELDS:
        (ROOT / OUTPUT_DIR / key).mkdir(parents=True, exist_ok=True)
    (ROOT / OUTPUT_DIR / "combined").mkdir(exist_ok=True)
    # Prepare all files before publication. Keep backups for rollback on write failure.
    targets = [OUTPUT_DIR / key / f"{key}.{ext}" for key in FIELDS for ext in ("parquet","csv","png")]
    targets.extend(OUTPUT_DIR / "combined" / f"combined.{ext}" for ext in ("parquet", "csv"))
    if (stage / "reference/splits.parquet").exists():
        targets.append(Path("reference/splits.parquet"))
    backups = stage / "backups"
    published_files = []
    try:
        for rel in targets:
            backup = backups / rel
            if (ROOT / rel).exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes((ROOT / rel).read_bytes())
            (stage / rel).replace(ROOT / rel)
            published_files.append(rel)
    except OSError:
        for rel in reversed(published_files):
            backup = backups / rel
            if backup.exists():
                backup.replace(ROOT / rel)
            else:
                (ROOT / rel).unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    for command in ("bootstrap", "update"):
        action = commands.add_parser(command)
        action.add_argument("--workers", type=int, default=32)
        if command == "update":
            action.add_argument("--start", type=lambda date: dt.date.fromisoformat(date).isoformat())
            action.add_argument("--end", type=lambda date: dt.date.fromisoformat(date).isoformat())
        else:
            action.set_defaults(start=None, end=None)
    args = parser.parse_args()
    if hasattr(args, "workers") and not 1 <= args.workers <= 64:
        parser.error("--workers must be in 1..64")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
