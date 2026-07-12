"""Tests for data loading and the rate -> index conversion."""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_research_lab import data
from portfolio_research_lab.data import load_rate_series, parse_price_csv, rate_to_index

_VALID_CSV = "date,STOCKS,BONDS\n2020-01-01,100.0,100.0\n2020-01-02,101.0,100.5\n"


def test_rate_to_index_first_point_is_base():
    rate = pd.Series(
        [3.6, 3.6, 3.6],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    index = rate_to_index(rate, base=100.0)
    assert index.iloc[0] == pytest.approx(100.0)


def test_rate_to_index_compounds_by_calendar_days():
    # 3.6% annual on act/360 => 0.01% per calendar day => factor 1.0001/day.
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    rate = pd.Series([3.6, 3.6, 3.6, 3.6], index=dates)
    index = rate_to_index(rate, base=100.0, days_per_year=360)

    assert index.iloc[1] == pytest.approx(100.0 * 1.0001)
    assert index.iloc[3] == pytest.approx(100.0 * 1.0001**3)


def test_rate_to_index_respects_day_gaps():
    # A 10-day gap should accrue ten days of interest, not one row's worth.
    dates = pd.to_datetime(["2020-01-01", "2020-01-11"])
    rate = pd.Series([3.6, 3.6], index=dates)
    index = rate_to_index(rate, base=100.0, days_per_year=360)
    assert index.iloc[1] == pytest.approx(100.0 * (1 + 0.036 * 10 / 360))


def test_rate_to_index_monotonic_for_positive_rates():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rate = pd.Series([2.5] * 30, index=dates)
    index = rate_to_index(rate)
    assert (index.diff().dropna() > 0).all()


def test_rate_to_index_empty_raises():
    with pytest.raises(ValueError, match="empty rate series"):
        rate_to_index(pd.Series(dtype=float))


def test_load_rate_series_parses_quoted_us_dates(tmp_path):
    csv = tmp_path / "rate.csv"
    csv.write_text('"Date","Value"\n"07/01/1954",1.13\n"07/02/1954",1.25\n')
    series = load_rate_series(csv)

    assert list(series.index) == list(pd.to_datetime(["1954-07-01", "1954-07-02"]))
    assert series.iloc[0] == pytest.approx(1.13)
    assert series.iloc[1] == pytest.approx(1.25)
    assert series.index.name == "date"


def test_load_rate_series_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="rate data file not found"):
        load_rate_series(tmp_path / "nope.csv")


# --- Upload parsing: happy path and hostile / malformed inputs --------------


def test_parse_price_csv_accepts_valid_input():
    frame = parse_price_csv(_VALID_CSV)
    assert list(frame.columns) == ["STOCKS", "BONDS"]
    assert len(frame) == 2


def test_parse_price_csv_accepts_bytes():
    frame = parse_price_csv(_VALID_CSV.encode("utf-8"))
    assert len(frame) == 2


def test_parse_price_csv_rejects_non_positive_prices():
    csv = "date,STOCKS\n2020-01-01,100.0\n2020-01-02,0.0\n"
    with pytest.raises(ValueError, match="strictly positive"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_non_finite_prices():
    csv = "date,STOCKS\n2020-01-01,100.0\n2020-01-02,inf\n"
    with pytest.raises(ValueError, match="non-finite"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_duplicate_dates():
    csv = "date,STOCKS\n2020-01-01,100.0\n2020-01-01,101.0\n"
    with pytest.raises(ValueError, match="duplicate dates"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_duplicate_columns():
    csv = "date,STOCKS ,STOCKS\n2020-01-01,100.0,100.0\n2020-01-02,101.0,101.0\n"
    with pytest.raises(ValueError, match="duplicate asset columns"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_single_observation():
    csv = "date,STOCKS\n2020-01-01,100.0\n"
    with pytest.raises(ValueError, match="at least two"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_missing_date_column():
    csv = "when,STOCKS\n2020-01-01,100.0\n2020-01-02,101.0\n"
    with pytest.raises(ValueError, match="date column"):
        parse_price_csv(csv)


def test_parse_price_csv_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(data, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        parse_price_csv(_VALID_CSV)


def test_parse_price_csv_rejects_too_many_rows(monkeypatch):
    monkeypatch.setattr(data, "MAX_UPLOAD_ROWS", 1)
    with pytest.raises(ValueError, match="row limit"):
        parse_price_csv(_VALID_CSV)


def test_parse_price_csv_rejects_too_many_columns(monkeypatch):
    monkeypatch.setattr(data, "MAX_UPLOAD_COLUMNS", 1)
    with pytest.raises(ValueError, match="column limit"):
        parse_price_csv(_VALID_CSV)


def test_parse_price_csv_rejects_overlong_symbol():
    long_name = "A" * (data.MAX_SYMBOL_LENGTH + 1)
    csv = f"date,{long_name}\n2020-01-01,100.0\n2020-01-02,101.0\n"
    with pytest.raises(ValueError, match="asset name exceeds"):
        parse_price_csv(csv)
