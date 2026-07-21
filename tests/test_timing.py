"""Tests for the moving-average trend-timing engine."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest
from pydantic import ValidationError

from portfolio_research_lab.models import TimingConfig
from portfolio_research_lab.timing import run_ma_timing

STOCK = "S&P 500"
CASH = "Cash (Fed Funds)"


def _frame(stock: list[float], cash: list[float]) -> pd.DataFrame:
    """Build a two-column [stock, cash] price frame on business days."""
    dates = pd.bdate_range("2020-01-01", periods=len(stock), name="date")
    return pd.DataFrame(
        {STOCK: [float(x) for x in stock], CASH: [float(x) for x in cash]}, index=dates
    )


def _config(
    *,
    initial_capital: float = 1_000.0,
    ma_window: int = 2,
    ma_kind: str = "simple",
    band_pct: float = 0.0,
    cost_bps: float = 0.0,
) -> TimingConfig:
    """A timing config with a 2-day window (MA defined from the second row)."""
    return TimingConfig(
        initial_capital=initial_capital,
        ma_window=ma_window,
        ma_kind=ma_kind,
        band_pct=band_pct,
        cost_bps=cost_bps,
        stock_symbol=STOCK,
        cash_symbol=CASH,
    )


def test_starts_fully_invested():
    result = run_ma_timing(_frame([100, 110, 120, 130], [100] * 4), _config())
    assert result.equity.iloc[0] == pytest.approx(1_000.0)
    assert result.stock_value.iloc[0] == pytest.approx(1_000.0)
    assert result.cash.iloc[0] == pytest.approx(0.0)
    assert result.position.iloc[0] == 1.0


def test_uptrend_never_exits():
    # Price always above its rising MA => never a reason to step aside.
    result = run_ma_timing(_frame([100, 110, 120, 130], [100] * 4), _config())
    assert result.n_switches == 0
    assert (result.position == 1.0).all()
    # Behaves exactly like buy-and-hold: 10 units * 130.
    assert result.equity.iloc[-1] == pytest.approx(1_300.0)


def test_exit_is_lagged_one_bar_then_cash_accrues():
    # Price first prints below its MA at i=2 (90 < 95), but the exit is decided
    # from the *previous* close, so it executes at i=3 — proving no look-ahead.
    prices = _frame([100, 100, 90, 90, 90], [100, 100, 100, 110, 121])
    result = run_ma_timing(prices, _config())

    events = result.events
    assert list(events["type"]) == ["to_cash"]
    day3 = prices.index[3]
    assert events.loc[day3, "type"] == "to_cash"
    assert events.loc[day3, "price"] == pytest.approx(90.0)
    # 10 units sold at 90 => 900 cash, no cost.
    assert result.cash.iloc[3] == pytest.approx(900.0)
    assert result.stock_value.iloc[3] == pytest.approx(0.0)
    # Position entering day 3 was still invested (switch recorded before the day's
    # return); the earlier below-MA bar (i=2) did not yet trade.
    assert list(result.position) == [1.0, 1.0, 1.0, 1.0, 0.0]
    # Out of the market on day 4, the reserve accrues the +10% cash step.
    assert result.cash.iloc[4] == pytest.approx(990.0)
    assert result.equity.iloc[4] == pytest.approx(990.0)


def test_exit_then_reentry():
    # Drop below MA (exit), then climb back above it (re-enter): two switches.
    prices = _frame([100, 90, 90, 100, 110], [100] * 5)
    result = run_ma_timing(prices, _config())

    assert list(result.events["type"]) == ["to_cash", "to_stocks"]
    # Exit at i=2 (price[1]=90 < ma[1]=95), selling 10 units at 90 => 900 cash.
    assert result.cash.iloc[2] == pytest.approx(900.0)
    # Re-enter at i=4 (price[3]=100 > ma[3]=95), buying at 110 with 900 cash.
    assert result.stock_value.iloc[4] == pytest.approx(900.0)
    assert result.cash.iloc[4] == pytest.approx(0.0)


def test_band_and_equality_retain_state():
    # A 10% band: a shallow dip to 95 never breaches ma*(1-0.10), so no exit.
    banded = run_ma_timing(_frame([100, 100, 95, 95], [100] * 4), _config(band_pct=0.10))
    assert banded.n_switches == 0
    # At band 0, price exactly equal to its MA must also retain the position.
    flat = run_ma_timing(_frame([100, 100, 100], [100] * 3), _config(band_pct=0.0))
    assert flat.n_switches == 0


def test_transaction_cost_charged_on_both_directions():
    prices = _frame([100, 90, 90, 100, 110], [100] * 5)
    result = run_ma_timing(prices, _config(cost_bps=100.0))  # 1% per switch

    # Exit: 10 units * 90 = 900 notional, 1% cost = 9 => 891 cash.
    assert list(result.events["cost"]) == pytest.approx([9.0, 8.91])
    assert result.equity.iloc[2] == pytest.approx(891.0)
    # Re-enter: 891 cash notional, 1% cost = 8.91 => 882.09 invested.
    assert result.equity.iloc[4] == pytest.approx(882.09)


def test_time_in_market_and_switch_count():
    prices = _frame([100, 90, 90, 100, 110], [100] * 5)
    result = run_ma_timing(prices, _config())
    # Positions entering each day: invested, invested, invested, cash, cash.
    assert list(result.position) == [1.0, 1.0, 1.0, 0.0, 0.0]
    # Averaged over the 4 return intervals (drops the first observation): 2/4.
    assert result.time_in_market == pytest.approx(0.5)
    assert result.n_switches == 2


def test_warmup_masks_both_kinds_identically():
    prices = _frame([100, 101, 102, 103, 104], [100] * 5)
    for kind in ("simple", "exponential"):
        result = run_ma_timing(prices, _config(ma_window=3, ma_kind=kind))
        ma = result.moving_average
        assert ma.iloc[:2].isna().all()  # first ma_window - 1 rows are NaN
        assert ma.iloc[2:].notna().all()


def test_events_schema():
    result = run_ma_timing(_frame([100, 90, 90, 100, 110], [100] * 5), _config())
    assert list(result.events.columns) == ["type", "price", "ma", "cost", "value"]


@pytest.mark.parametrize(
    "make",
    [
        lambda: TimingConfig(cost_bps=10_000.0),
        lambda: TimingConfig(band_pct=1.0),
        lambda: TimingConfig(cost_bps=float("inf")),
        lambda: TimingConfig(ma_window=1),
        lambda: TimingConfig(ma_kind="triangular"),
    ],
)
def test_invalid_config_rejected(make: Callable[[], TimingConfig]):
    with pytest.raises(ValidationError):
        make()


def test_empty_prices_raise():
    with pytest.raises(ValueError, match="empty price data"):
        run_ma_timing(pd.DataFrame(), _config())


def test_single_row_raises():
    with pytest.raises(ValueError, match="at least two"):
        run_ma_timing(_frame([100], [100]), _config())


def test_non_monotonic_index_raises():
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-03", "2020-01-02"]), name="date")
    prices = pd.DataFrame({STOCK: [100.0, 101.0], CASH: [100.0, 100.0]}, index=dates)
    with pytest.raises(ValueError, match="monoton"):
        run_ma_timing(prices, _config())


def test_missing_columns_raise():
    dates = pd.bdate_range("2020-01-01", periods=3, name="date")
    prices = pd.DataFrame({STOCK: [100.0, 100.0, 100.0]}, index=dates)
    with pytest.raises(ValueError, match="missing required columns"):
        run_ma_timing(prices, _config())
