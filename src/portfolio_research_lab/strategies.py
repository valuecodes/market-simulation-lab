"""Strategy definitions.

A strategy decides how capital is allocated across assets. The engine keeps the
interface small on purpose: a strategy only needs to declare its *initial*
target weights. This is enough for a buy-and-hold backtest and leaves room to
add rebalancing or signal-driven strategies later without changing the
simulator's public API.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from portfolio_research_lab.models import StrategyConfig


@runtime_checkable
class Strategy(Protocol):
    """Protocol implemented by every strategy."""

    name: str

    def initial_weights(self, prices: pd.DataFrame, config: StrategyConfig) -> dict[str, float]:
        """Return the target weight per asset at the first period.

        Weights should sum to 1.0. ``prices`` is provided so future strategies
        can look at history; buy-and-hold ignores it.
        """
        ...


class BuyAndHold:
    """Buy the target allocation once and hold it for the whole period.

    No rebalancing occurs, so the realised weights drift with relative asset
    performance — exactly what a passive investor experiences.
    """

    name = "Buy & Hold"

    def initial_weights(
        self,
        prices: pd.DataFrame,
        config: StrategyConfig,
    ) -> dict[str, float]:
        missing = [symbol for symbol in config.symbols if symbol not in prices.columns]
        if missing:
            raise ValueError(f"price data is missing allocated assets: {missing}")
        return dict(config.allocations)
