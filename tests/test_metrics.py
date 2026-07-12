"""Tests for return, volatility and drawdown metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_research_lab import metrics


def _equity(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(values), name="date")
    return pd.Series(values, index=idx, dtype=float)


def test_periodic_returns_drops_first_period():
    equity = _equity([100.0, 110.0, 121.0])
    rets = metrics.periodic_returns(equity)
    assert len(rets) == 2
    assert rets.iloc[0] == pytest.approx(0.10)
    assert rets.iloc[1] == pytest.approx(0.10)


def test_total_return():
    equity = _equity([100.0, 150.0, 200.0])
    assert metrics.total_return(equity) == pytest.approx(1.0)


def test_total_return_requires_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        metrics.total_return(_equity([100.0]))


def test_cagr_over_two_years():
    # 100 -> 121 over exactly 2 years (2 periods at 1 period/year) => 10% CAGR.
    equity = _equity([100.0, 110.0, 121.0])
    assert metrics.cagr(equity, periods_per_year=1) == pytest.approx(0.10)


def test_annualized_volatility_matches_manual():
    equity = _equity([100.0, 110.0, 99.0, 108.9])
    rets = metrics.periodic_returns(equity)
    expected = rets.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(rets, 252) == pytest.approx(expected)


def test_volatility_of_constant_series_is_zero():
    equity = _equity([100.0, 100.0, 100.0])
    assert metrics.annualized_volatility(metrics.periodic_returns(equity)) == 0.0


def test_drawdown_series_is_non_positive_and_correct():
    equity = _equity([100.0, 120.0, 90.0, 150.0])
    dd = metrics.drawdown_series(equity)
    assert (dd <= 1e-12).all()
    # Trough at 90 against a peak of 120 => -25%.
    assert dd.iloc[2] == pytest.approx(-0.25)
    # New high resets drawdown to zero.
    assert dd.iloc[3] == pytest.approx(0.0)


def test_max_drawdown():
    equity = _equity([100.0, 120.0, 90.0, 150.0])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_increase_is_zero():
    equity = _equity([100.0, 101.0, 102.0, 103.0])
    assert metrics.max_drawdown(equity) == pytest.approx(0.0)


def test_summarize_keys():
    equity = _equity([100.0, 110.0, 105.0, 130.0])
    summary = metrics.summarize(equity)
    assert set(summary) == {
        "total_return",
        "cagr",
        "annualized_volatility",
        "max_drawdown",
    }
