"""Tests for portfolio accounting in the simulation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_research_lab.models import StrategyConfig
from portfolio_research_lab.simulator import run_simulation
from portfolio_research_lab.strategies import BuyAndHold


def test_buy_and_hold_units_and_equity(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(
        name="Test",
        initial_capital=10_000.0,
        allocations={"UP": 0.5, "FLAT": 0.5},
    )
    result = run_simulation(two_asset_prices, config)

    # 5,000 into UP @100 => 50 units; 5,000 into FLAT @50 => 100 units.
    first_holdings = result.holdings.iloc[0]
    assert first_holdings["UP"] == pytest.approx(50.0)
    assert first_holdings["FLAT"] == pytest.approx(100.0)

    # Holdings never change for buy-and-hold.
    assert (result.holdings.nunique() == 1).all()


def test_initial_equity_equals_capital(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 0.5, "FLAT": 0.5}, initial_capital=10_000.0)
    result = run_simulation(two_asset_prices, config)
    assert result.equity.iloc[0] == pytest.approx(10_000.0)


def test_final_equity_tracks_prices(two_asset_prices: pd.DataFrame):
    # UP goes 100 -> 200 (x2), FLAT unchanged.
    config = StrategyConfig(allocations={"UP": 0.5, "FLAT": 0.5}, initial_capital=10_000.0)
    result = run_simulation(two_asset_prices, config)
    # 50 units UP @200 = 10,000 + 100 units FLAT @50 = 5,000 => 15,000.
    assert result.equity.iloc[-1] == pytest.approx(15_000.0)


def test_equity_equals_sum_of_asset_values(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 0.7, "FLAT": 0.3}, initial_capital=5_000.0)
    result = run_simulation(two_asset_prices, config)
    reconstructed = result.asset_values.sum(axis=1)
    pd.testing.assert_series_equal(result.equity, reconstructed.rename("equity"), check_names=True)


def test_single_asset_equity_scales_with_price(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 1.0}, initial_capital=1_000.0)
    result = run_simulation(two_asset_prices, config)
    # Fully invested in an asset that doubles => equity doubles.
    assert result.equity.iloc[-1] == pytest.approx(2_000.0)


def test_benchmark_scaled_to_initial_capital(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 1.0}, initial_capital=1_000.0, benchmark="BENCH")
    result = run_simulation(two_asset_prices, config)
    assert result.benchmark_equity is not None
    assert result.benchmark_equity.iloc[0] == pytest.approx(1_000.0)
    # BENCH grows 100 -> 150 => +50%.
    assert result.benchmark_equity.iloc[-1] == pytest.approx(1_500.0)


def test_no_benchmark_when_unset(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 1.0})
    result = run_simulation(two_asset_prices, config)
    assert result.benchmark_equity is None
    assert result.benchmark_metrics() is None


def test_missing_asset_raises(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"MISSING": 1.0})
    with pytest.raises(ValueError, match="missing allocated assets"):
        run_simulation(two_asset_prices, config)


def test_empty_prices_raise():
    config = StrategyConfig(allocations={"UP": 1.0})
    with pytest.raises(ValueError, match="empty price data"):
        run_simulation(pd.DataFrame(), config)


def test_default_strategy_is_buy_and_hold(two_asset_prices: pd.DataFrame):
    config = StrategyConfig(allocations={"UP": 1.0})
    result = run_simulation(two_asset_prices, config)
    assert result.strategy_name == BuyAndHold.name
