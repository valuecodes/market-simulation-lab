"""The simulation engine.

The simulator turns a strategy plus price data into an equity curve and the
underlying holdings. It performs plain portfolio accounting: buy fractional
units at the first period's prices, then mark the position to market every
period. If ``config.rebalance_frequency`` is set, holdings are reset back to the
strategy's target weights at each period boundary; otherwise the weights drift
(buy-and-hold). There is no leverage, no transaction cost and no cash yield —
this is a deliberately small, transparent foundation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_research_lab import metrics
from portfolio_research_lab.models import StrategyConfig
from portfolio_research_lab.strategies import BuyAndHold, Strategy

# Rebalancing cadence -> pandas period alias used to find period boundaries.
_FREQUENCY_CODES: dict[str, str] = {"monthly": "M", "quarterly": "Q", "annually": "Y"}


@dataclass(slots=True)
class SimulationResult:
    """Output of a simulation run.

    Attributes
    ----------
    config:
        The configuration that produced this run.
    strategy_name:
        Name of the strategy that was executed.
    holdings:
        Units held per asset over time (constant for buy-and-hold).
    asset_values:
        Market value of each asset position over time.
    equity:
        Total portfolio value over time (the equity curve).
    benchmark_equity:
        Equity curve of the benchmark scaled to the same starting capital, or
        ``None`` if no benchmark was configured or available.
    """

    config: StrategyConfig
    strategy_name: str
    holdings: pd.DataFrame
    asset_values: pd.DataFrame
    equity: pd.Series
    benchmark_equity: pd.Series | None = None

    def metrics(self) -> dict[str, float]:
        """Headline metrics for the strategy equity curve."""
        return metrics.summarize(self.equity, self.config.trading_days_per_year)

    def benchmark_metrics(self) -> dict[str, float] | None:
        """Headline metrics for the benchmark, if one exists."""
        if self.benchmark_equity is None:
            return None
        return metrics.summarize(self.benchmark_equity, self.config.trading_days_per_year)

    def drawdown(self) -> pd.Series:
        """Drawdown series of the strategy equity curve."""
        return metrics.drawdown_series(self.equity)


def run_simulation(
    prices: pd.DataFrame,
    config: StrategyConfig,
    strategy: Strategy | None = None,
) -> SimulationResult:
    """Run a portfolio simulation.

    Parameters
    ----------
    prices:
        Wide-format price data (see :mod:`portfolio_research_lab.data`).
    config:
        The strategy configuration to simulate.
    strategy:
        Strategy implementation. Defaults to :class:`BuyAndHold`.
    """
    if strategy is None:
        strategy = BuyAndHold()

    if prices.empty:
        raise ValueError("cannot simulate on empty price data")

    weights = strategy.initial_weights(prices, config)
    symbols = list(weights)
    asset_prices = prices[symbols]

    rebalance_dates = _rebalance_dates(asset_prices.index, config.rebalance_frequency)
    holdings = _holdings_schedule(
        asset_prices,
        weights,
        config.initial_capital,
        rebalance_dates,
    )
    asset_values = asset_prices.mul(holdings)
    equity = asset_values.sum(axis=1).rename("equity")

    benchmark_equity = _benchmark_equity(prices, config)

    return SimulationResult(
        config=config,
        strategy_name=strategy.name,
        holdings=holdings,
        asset_values=asset_values,
        equity=equity,
        benchmark_equity=benchmark_equity,
    )


def _benchmark_equity(prices: pd.DataFrame, config: StrategyConfig) -> pd.Series | None:
    """Scale the benchmark price series to the strategy's starting capital."""
    symbol = config.benchmark
    if symbol is None or symbol not in prices.columns:
        return None
    series = prices[symbol]
    scaled = series / series.iloc[0] * config.initial_capital
    return scaled.rename("benchmark")


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str | None) -> pd.DatetimeIndex:
    """Dates on which to reset holdings: the first row of each new period.

    Returns an empty index for ``None`` (buy-and-hold). The opening date is never
    a rebalance date — holdings are established there by the initial purchase.
    """
    if frequency is None:
        return index[:0]
    periods = index.to_period(_FREQUENCY_CODES[frequency])
    is_new_period = np.concatenate([[False], periods[1:] != periods[:-1]])
    return index[is_new_period]


def _holdings_schedule(
    asset_prices: pd.DataFrame,
    weights: dict[str, float],
    capital: float,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Units held per asset over time.

    Buys the target allocation at the opening prices, then carries units forward
    unchanged except on ``rebalance_dates``, where the current portfolio value is
    re-split according to the target weights. With no rebalance dates this yields
    constant units (buy-and-hold).
    """
    symbols = list(asset_prices.columns)
    weight_vec = pd.Series(weights, dtype=float).reindex(symbols)
    rebalance_set = set(rebalance_dates)

    units = weight_vec * capital / asset_prices.iloc[0]
    rows: list[pd.Series] = []
    for date, prices_today in asset_prices.iterrows():
        if date in rebalance_set:
            current_equity = float((prices_today * units).sum())
            units = weight_vec * current_equity / prices_today
        rows.append(units)

    return pd.DataFrame(rows, index=asset_prices.index, columns=symbols)
