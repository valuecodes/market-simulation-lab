"""The simulation engine.

The simulator turns a strategy plus price data into an equity curve and the
underlying holdings. It performs plain portfolio accounting: buy fractional
units at the first period's prices, then mark the position to market every
period. There is no leverage, no transaction cost and no cash yield — this is a
deliberately small, transparent foundation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from portfolio_research_lab import metrics
from portfolio_research_lab.models import StrategyConfig
from portfolio_research_lab.strategies import BuyAndHold, Strategy


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
    first_prices = asset_prices.iloc[0]

    # Buy fractional units once at the opening prices, then hold.
    capital_per_asset = pd.Series(weights, dtype=float) * config.initial_capital
    units = capital_per_asset / first_prices

    holdings = pd.DataFrame(
        [units.to_dict()] * len(asset_prices),
        index=asset_prices.index,
        columns=symbols,
    )
    asset_values = asset_prices.mul(units, axis=1)
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
