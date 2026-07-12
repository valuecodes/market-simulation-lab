"""Tests for data loading and the rate -> index conversion."""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_research_lab.data import load_rate_series, rate_to_index


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
