import datetime as dt
import sys
from collections import deque
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import calculations as calc
import market_breadth as cli
import massive


def test_split_uses_consistent_close_and_volume_bases():
    previous = deque([(100.0, 60_000.0)], maxlen=40)
    history, _, _, comparable, eligible, up, down = calc.classify(previous, 52, 130_000, 2)
    assert comparable and eligible and up and not down
    assert list(history) == [(50, 120_000), (52, 130_000)]
    assert list(previous) == [(100, 60_000)]


def test_ambiguity_resets_and_does_not_signal():
    history, *flags = calc.classify(deque([(50, 100_000)] * 40, maxlen=40), 80, 150_000, None)
    assert not history and not any(flags)


def test_canonical_resolution_applies_to_exact_id_set_across_rename():
    events = {
        "OLD": [dict(key=("id","reverse"), date=dt.date(2026,1,5), split_from=10, split_to=1, valid=True)],
        "NEW": [dict(key=("id","other"), date=dt.date(2026,1,5), split_from=1, split_to=2, valid=True)],
    }
    args = (events, "NEW", "OLD", "2026-01-02", "2026-01-05")
    assert calc.comparison_split(*args, {}) == (None, 2)
    assert calc.comparison_split(*args, {frozenset(("reverse","other")): 0.1}) == (0.1, 2)


def test_percentile_has_complete_prior_window_and_no_lookahead():
    p = calc.rasi_percentile_252([1]*126+[3]*126+[1,4])
    assert p.iloc[:252].isna().all()
    assert p.iloc[252] == 50 and p.iloc[253] == 100


def test_ratio_warmup_and_denominator_zero():
    data = pd.DataFrame([dict(exchange="COMBINED", up4=2, down4=int(i==10), listed=3,
                             sma40_eligible=0, above_sma40=0, comparable=1) for i in range(11)])
    result = calc.add_ratios(data)
    assert result.ratio10d.iloc[:10].isna().all()
    assert result.ratio10d.iloc[10] == 20


def test_reference_filter_keeps_every_nyse_type_and_only_nasdaq_cs():
    class Client:
        def fetch_listings(self, day):
            return [{"ticker":t, "type":kind, "primary_exchange":mic}
                    for t,kind,mic in [("A",None,"XNYS"),("B","ETF","XNYS"),
                                       ("C","CS","XNAS"),("D","ETF","XNAS")]]
    result = massive.listings_on(Client(), "2022-08-22", {"CS","ETF"})
    assert dict(zip(result.ticker, result.exchange, strict=True)) == {"A":"XNYS","B":"XNYS","C":"XNAS"}
    assert result.loc[result.ticker=="A","type"].isna().all()


def test_derivatives_round_trip(tmp_path):
    frame = pd.DataFrame({"date":pd.to_datetime(["2026-01-02","2026-01-05"]),
                          "ratio10d":pd.Series([pd.NA, 1.2345678901234567], dtype="Float64")})
    cli.export_series(tmp_path, "ratio10d", frame)
    cli.equal_csv(tmp_path/"indicators/ratio10d/ratio10d.csv", frame)
    assert (tmp_path/"indicators/ratio10d/ratio10d.png").is_file()


def test_cache_is_not_inside_project(monkeypatch):
    monkeypatch.setenv("MARKET_BREADTH_CACHE", str(cli.ROOT/"scratch"))
    with pytest.raises(ValueError, match="outside"):
        cli.cache_root()

def test_split_refresh_keeps_canonical_resolution_metadata(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq
    source = tmp_path / "canonical.parquet"
    date = pd.Timestamp("2026-01-05")
    rows = pd.DataFrame([
        dict(id="a", ticker="A", execution_date=date, split_from=10.0, split_to=1.0,
             adjustment_type="reverse_split", historical_adjustment_factor=10.0),
        dict(id="b", ticker="A", execution_date=date, split_from=1.0, split_to=2.0,
             adjustment_type="forward_split", historical_adjustment_factor=0.5),
    ])
    metadata = {b"massive_split_scope": b"all stocks", b"massive_split_start": b"2016-10-03",
                b"massive_split_end": b"2026-01-05",
                b"stockbee_split_resolutions": json.dumps([dict(ids=["a","b"], factor=0.1)]).encode()}
    pq.write_table(massive._frame_to_table(rows, massive.SPLIT_SCHEMA).replace_schema_metadata(metadata), source)
    monkeypatch.setattr(cli, "SPLITS", source)

    class Client:
        def fetch_splits(self, start, end):
            assert (start,end) == ("2026-01-06","2026-01-07")
            return [dict(id="c", ticker="B", execution_date="2026-01-06", split_from=1,
                         split_to=2, adjustment_type="forward_split", historical_adjustment_factor=0.5)]

    result = tmp_path / "result.parquet"
    cli.refresh_splits(Client(), None, "2026-01-07", result)
    actual = pq.read_table(result)
    assert set(actual["id"].to_pylist()) == {"a","b","c"}
    assert actual.schema.metadata == {**metadata, b"massive_split_end":b"2026-01-07"}
    assert pq.read_table(source)["id"].to_pylist() == ["a","b"]
