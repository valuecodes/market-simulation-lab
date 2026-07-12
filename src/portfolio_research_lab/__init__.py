"""Portfolio Research Lab.

A local-first toolkit for backtesting and simulating investment strategies.

The package is split into a reusable simulation engine (this package) and a
Streamlit interface (see the ``app/`` directory). Nothing in this package
imports Streamlit, so the engine can be used from notebooks, scripts or tests.
"""

from __future__ import annotations

from portfolio_research_lab.data import generate_synthetic_prices, load_price_data
from portfolio_research_lab.models import StrategyConfig
from portfolio_research_lab.simulator import SimulationResult, run_simulation
from portfolio_research_lab.strategies import BuyAndHold, Strategy

__all__ = [
    "BuyAndHold",
    "SimulationResult",
    "Strategy",
    "StrategyConfig",
    "generate_synthetic_prices",
    "load_price_data",
    "run_simulation",
]

__version__ = "0.1.0"
